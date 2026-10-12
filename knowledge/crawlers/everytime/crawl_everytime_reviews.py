#!/usr/bin/env python3
"""Collect Everytime lecture reviews that the signed-in user can view.

Login is intentionally manual. The crawler stores only a Playwright browser
profile on the local machine and never asks for an Everytime password.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Keep the browser runtime beside this crawler instead of requiring a
# machine-wide installation or write access to the user's AppData folder.
os.environ.setdefault(
    "PLAYWRIGHT_BROWSERS_PATH",
    str(Path(__file__).resolve().parent / ".playwright-browsers"),
)

try:
    from playwright.sync_api import Locator, Page, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # Lets --help fail with a useful message instead of a traceback.
    Locator = Page = object  # type: ignore[assignment,misc]
    PlaywrightTimeoutError = TimeoutError
    PlaywrightError = RuntimeError
    sync_playwright = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA_DIR = REPO_ROOT / "knowledge/raw/everytime_lectures"

BASE_URL = "https://everytime.kr"
LECTURE_URL = BASE_URL + "/lecture/view/{lecture_id}?tab=article"

# Everytime can change its markup. Keep all DOM assumptions here so they are
# easy to update after inspecting one signed-in page.
REVIEW_CONTAINER_SELECTORS = (
    ".article_tab .articles > .article",
)
CONTENT_SELECTORS = (
    ":scope > .text",
)
MORE_BUTTON_NAMES = ("더 보기", "더보기", "다음", "계속 보기")
LOGIN_MARKERS = ("로그인", "아이디", "비밀번호")


class ReviewUnavailable(RuntimeError):
    """Lecture-local missing review DOM; safe to record and continue."""


@dataclass(frozen=True)
class Review:
    lecture_id: str
    review_id: str
    semester: str | None
    rating: float | None
    content: str
    source_url: str
    source_hash: str
    crawled_at: str
    course_name: str | None = None
    professor: str | None = None
    recommendation_count: int | None = None


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def content_hash(lecture_id: str, semester: str | None, content: str) -> str:
    canonical = "\n".join((lecture_id, semester or "", normalize_text(content)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_semester(text: str) -> str | None:
    patterns = (
        r"(?:20)?\d{2}\s*년\s*[12]\s*학기",
        r"(?:20)?\d{2}\s*[-.]\s*[12]\s*학기?",
        r"(?:20)?\d{2}\s*년\s*여름(?:학기)?",
        r"(?:20)?\d{2}\s*년\s*겨울(?:학기)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_text(match.group(0))
    return None


def extract_rating(text: str) -> float | None:
    patterns = (
        r"(?:별점|평점)\s*[:：]?\s*([0-5](?:\.\d+)?)",
        r"([0-5](?:\.\d+)?)\s*(?:점|/\s*5)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if 0 <= value <= 5:
                return value
    return None


def load_seen_hashes(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("잘못된 JSONL 행을 건너뜁니다: %s:%d", path, line_number)
                continue
            fingerprint = record.get("source_hash") or record.get("content_hash")
            if fingerprint:
                seen.add(str(fingerprint))
    return seen


def append_jsonl(path: Path, reviews: Iterable[Review]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for review in reviews:
            handle.write(json.dumps(asdict(review), ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
    return count


def update_recommendations(path: Path, reviews: Iterable[Review]) -> int:
    updates = {r.source_hash: r.recommendation_count for r in reviews if r.recommendation_count is not None}
    if not updates or not path.exists():
        return 0
    lines = path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    changed = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # Preserve malformed lines rather than deleting user data.
        if not isinstance(row, dict):
            continue
        key = row.get('source_hash') or row.get('content_hash')
        if key in updates and row.get('recommendation_count') != updates[key]:
            row['recommendation_count'] = updates[key]
            row['recommendation_updated_at'] = datetime.now(timezone.utc).isoformat()
            lines[index] = json.dumps(row, ensure_ascii=False) + '\n'
            changed += 1
    if changed:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(''.join(lines), encoding='utf-8')
        temporary.replace(path)
    return changed


def save_grouped_reviews(source: Path, by_course_path: Path, by_professor_path: Path) -> None:
    paths = [path.resolve() for path in (source, by_course_path, by_professor_path)]
    if len(set(paths)) != 3:
        raise RuntimeError('원본 JSONL과 두 JSON 출력 경로는 모두 달라야 합니다.')
    by_course: dict[str, dict] = {}
    by_professor: dict[str, dict] = {}
    if source.exists():
        with source.open('r', encoding='utf-8-sig') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    review = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning('잘못된 리뷰 JSONL 행을 그룹 파일에서 건너뜁니다: %s:%d', source, line_number)
                    continue
                course = review.get('course_name') or '과목명 미지정'
                professor = review.get('professor') or '교수 미표기'
                lecture_id = str(review.get('lecture_id') or '')
                course_prof = by_course.setdefault(course, {}).setdefault(professor, {}).setdefault(lecture_id, [])
                course_prof.append(review)
                prof_course = by_professor.setdefault(professor, {}).setdefault(course, {}).setdefault(lecture_id, [])
                prof_course.append(review)

    def stats(rows: list[dict]) -> dict:
        ratings = [row.get('rating') for row in rows
                   if isinstance(row.get('rating'), (int, float))
                   and not isinstance(row.get('rating'), bool)
                   and 0 <= row['rating'] <= 5]
        return {
            'review_count': len(rows),
            'rated_review_count': len(ratings),
            'average_rating': round(sum(ratings) / len(ratings), 2) if ratings else None,
        }

    def document(groups: dict, inner_label: str) -> dict:
        result = {}
        for outer in sorted(groups, key=str.casefold):
            inner_obj = {}
            inner_summaries = {}
            outer_rows = []
            for inner in sorted(groups[outer], key=str.casefold):
                lectures = []
                inner_rows = []
                for lecture_id in sorted(groups[outer][inner]):
                    rows = groups[outer][inner][lecture_id]
                    inner_rows.extend(rows)
                    semesters = sorted({row['semester'] for row in rows if row.get('semester')})
                    lectures.append({
                        'lecture_id': lecture_id,
                        'semesters': semesters,
                        **stats(rows),
                        'reviews': rows,
                    })
                inner_obj[inner] = lectures
                inner_summaries[inner] = stats(inner_rows)
                outer_rows.extend(inner_rows)
            result[outer] = {**stats(outer_rows), inner_label: inner_obj, 'summaries': inner_summaries}
        return {
            'grouped_by': 'course_name' if inner_label == 'professors' else 'professor',
            'rating_basis': 'collected_reviews_only; valid ratings 0-5; null excluded; review-weighted mean',
            'groups': result,
        }

    def write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    write(by_course_path, document(by_course, 'professors'))
    write(by_professor_path, document(by_professor, 'courses'))


def _first_text(container: Locator, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        candidate = container.locator(selector)
        try:
            if candidate.count() and candidate.first.is_visible():
                text = normalize_text(candidate.first.inner_text(timeout=1_000))
                if text:
                    return text
        except PlaywrightTimeoutError:
            continue
    return ""


def _rating_from_dom(container: Locator, fallback_text: str) -> float | None:
    stars = container.locator(':scope > .article_header .rate .star > .on')
    if stars.count() == 1:
        width = stars.first.evaluate('(el) => el.style.width')
        match = re.fullmatch(r'(\d+(?:\.\d+)?)%', width.strip())
        if match and 0 <= float(match.group(1)) <= 100:
            return round(float(match.group(1)) / 20, 4)
    candidates = container.locator(
        "[aria-label*='별점'], [title*='별점'], [data-rating], [class*='rating']"
    )
    for index in range(min(candidates.count(), 10)):
        candidate = candidates.nth(index)
        numeric = candidate.get_attribute('data-rating')
        if numeric and re.fullmatch(r'[0-5](?:\.\d+)?', numeric) and float(numeric) <= 5:
            return float(numeric)
        for attribute in ("aria-label", "title"):
            value = candidate.get_attribute(attribute)
            rating = extract_rating(value or "")
            if rating is not None:
                return rating
    # Review prose can mention someone else's score ("5점 주는 사람").
    # An unknown visual star encoding must remain null, not become that score.
    return None


def extract_recommendations(container: Locator) -> int | None:
    info = container.locator(':scope > .article_header .title .info')
    if info.count() != 1:
        return None
    count = info.locator('span.posvote')
    if count.count() == 0:
        return 0  # In the supplied markup, zero votes omit the number span.
    if count.count() != 1:
        return None
    text = normalize_text(count.inner_text(timeout=1_000))
    if re.fullmatch(r'(?:\d+|\d{1,3}(?:,\d{3})+)', text):
        return int(text.replace(',', ''))
    return None


def parse_review(
    container: Locator, lecture_id: str, source_url: str,
    course_name: str | None = None, professor: str | None = None,
) -> Review | None:
    full_text = normalize_text(container.inner_text(timeout=2_000))
    if not full_text:
        return None

    content = _first_text(container, CONTENT_SELECTORS)
    if not content:
        # Never save a whole page/card as a review when its body is unknown.
        return None

    semester = extract_semester(_first_text(container, (':scope > .article_header .semester',)))
    rating = _rating_from_dom(container, full_text)
    fingerprint = content_hash(lecture_id, semester, content)
    dom_id = container.get_attribute("data-id") or container.get_attribute("id")
    return Review(
        lecture_id=lecture_id,
        review_id=normalize_text(dom_id) or fingerprint[:16],
        semester=semester,
        rating=rating,
        content=content,
        source_url=source_url,
        source_hash=fingerprint,
        crawled_at=datetime.now(timezone.utc).isoformat(),
        course_name=course_name,
        professor=professor,
        recommendation_count=extract_recommendations(container),
    )


def read_review_batch(containers: Locator, lecture: dict, source_url: str) -> list[Review | None]:
    # One browser round-trip for all cards, rather than many per field per card.
    rows = containers.evaluate_all('''cards => cards.map(card => {
        const text = selector => card.querySelector(selector)?.innerText || '';
        const info = card.querySelector(':scope > .article_header .title .info');
        const votes = info ? [...info.querySelectorAll('span.posvote')] : [];
        const stars = card.querySelectorAll(':scope > .article_header .rate .star > .on');
        return {
            content: text(':scope > .text'),
            semester: text(':scope > .article_header .semester'),
            width: stars.length === 1 ? stars[0].style.width : '',
            votes: !info || votes.length > 1 ? null : votes.length ? votes[0].innerText : '0',
            id: card.getAttribute('data-id') || card.id || ''
        };
    })''')
    result = []
    for row in rows:
        content = normalize_text(row['content'])
        if not content:
            result.append(None)
            continue
        semester = extract_semester(row['semester'])
        width = re.fullmatch(r'(\d+(?:\.\d+)?)%', row['width'].strip())
        rating = round(float(width.group(1)) / 20, 4) if width and 0 <= float(width.group(1)) <= 100 else None
        votes = normalize_text(row['votes']) if row['votes'] is not None else ''
        count = int(votes.replace(',', '')) if re.fullmatch(r'(?:\d+|\d{1,3}(?:,\d{3})+)', votes) else None
        fingerprint = content_hash(lecture['lecture_id'], semester, content)
        result.append(Review(
            lecture_id=lecture['lecture_id'], review_id=normalize_text(row['id']) or fingerprint[:16],
            semester=semester, rating=rating, content=content, source_url=source_url,
            source_hash=fingerprint, crawled_at=datetime.now(timezone.utc).isoformat(),
            course_name=lecture.get('course_name'), professor=lecture.get('professor'),
            recommendation_count=count,
        ))
    return result


def find_review_containers(page: Page) -> Locator:
    best: Locator | None = None
    best_count = 0
    for selector in REVIEW_CONTAINER_SELECTORS:
        locator = page.locator(selector)
        count = locator.count()
        if count > best_count:
            best, best_count = locator, count
    if best is None or best_count == 0:
        raise RuntimeError(
            "강의평 요소를 찾지 못했습니다. 로그인 상태와 페이지를 확인한 뒤 "
            "REVIEW_CONTAINER_SELECTORS를 현재 DOM에 맞게 수정하세요."
        )
    return best


def looks_like_login_page(page: Page) -> bool:
    if "/login" in page.url:
        return True
    body = normalize_text(page.locator("body").inner_text(timeout=3_000))
    return sum(marker in body for marker in LOGIN_MARKERS) >= 2


def wait_for_manual_login(page: Page, target_url: str) -> None:
    print("\n브라우저에서 직접 로그인하세요. 비밀번호는 이 프로그램이 읽지 않습니다.")
    input("로그인 후 화면이 뜰 때까지 기다린 뒤, 브라우저를 닫지 말고 이 터미널에서 Enter를 누르세요: ")
    finish_login_navigation(page, target_url)
    if looks_like_login_page(page):
        raise RuntimeError("로그인 상태를 확인하지 못했습니다. 다시 실행해 로그인해 주세요.")


def finish_login_navigation(page: Page, target_url: str) -> None:
    # Login may already be redirecting to this lecture. Waiting must come
    # before goto, otherwise two identical navigations can cancel each other.
    try:
        page.wait_for_url(target_url, wait_until='domcontentloaded', timeout=5_000)
        return
    except PlaywrightTimeoutError:
        if page.url == target_url:
            page.wait_for_load_state('domcontentloaded', timeout=15_000)
            return
    try:
        check_response(page.goto(target_url, wait_until='domcontentloaded'))
    except PlaywrightError as exc:
        if 'is interrupted by another navigation' not in str(exc):
            raise
        # A late login redirect can race with the single explicit navigation.
        # Follow that navigation; never retry goto or swallow unrelated errors.
        page.wait_for_url(target_url, wait_until='domcontentloaded', timeout=15_000)
    if page.url != target_url:
        raise RuntimeError('로그인 후 목표 강의 화면에 도착하지 못했습니다. 브라우저의 로그인 상태를 확인하세요.')


def check_response(response) -> None:
    if response is None:
        raise RuntimeError('페이지 응답을 확인하지 못했습니다. 수집을 중단합니다.')
    if response.status in (401, 403, 429):
        raise RuntimeError(
            f'에타 접근 제한 또는 인증 오류(HTTP {response.status}). '
            '수집을 중단합니다. 반복 실행하지 말고 일반 브라우저에서 접근 상태를 확인하세요.'
        )
    if response.status >= 400:
        raise RuntimeError(f'강의 페이지 로딩 실패(HTTP {response.status}).')


def click_more_or_scroll(page: Page, previous_count: int) -> bool:
    containers = find_review_containers(page)
    before = containers.all_text_contents()
    for name in MORE_BUTTON_NAMES:
        button = page.get_by_role("button", name=name, exact=True)
        if button.count() and button.first.is_visible():
            button.first.click(timeout=3_000)
            page.wait_for_timeout(1_500)
            return True

    # The lecture review list has its own scrollbar. Scroll only ancestors
    # of review cards, never unrelated sidebars or the overview panel.
    moved = containers.evaluate_all('''(cards) => {
        const targets = new Set();
        for (const card of cards) {
            for (let el = card.parentElement; el; el = el.parentElement) {
                const style = getComputedStyle(el);
                if (/(auto|scroll)/.test(style.overflowY) &&
                    el.clientHeight > 0 && el.scrollHeight > el.clientHeight + 1) {
                    targets.add(el);
                    break;
                }
            }
        }
        if (!targets.size && document.scrollingElement) {
            targets.add(document.scrollingElement);
        }
        let moved = false;
        for (const el of targets) {
            const before = el.scrollTop;
            // Jump to the last loaded card in the observed accumulating list.
            // For layouts with large virtual spacers keep viewport-sized steps.
            const localCards = cards.filter(card => el.contains(card));
            const first = localCards[0], last = localCards[localCards.length - 1];
            const span = first && last ? last.getBoundingClientRect().bottom - first.getBoundingClientRect().top : 0;
            const accumulating = span >= el.scrollHeight * .8;
            el.scrollTop = accumulating ? el.scrollHeight : Math.min(el.scrollHeight, before + Math.max(1, el.clientHeight * .9));
            moved = moved || el.scrollTop !== before;
        }
        return moved;
    }''')
    page.wait_for_timeout(1_500)
    try:
        # Compare contents too: virtualized lists may replace cards without
        # changing their count. Moving through preloaded cards is progress.
        return moved or find_review_containers(page).all_text_contents() != before
    except RuntimeError:
        return False


def crawl_one(page: Page, lecture: dict, args: argparse.Namespace, seen: set[str], output: Path) -> int:
    lecture_id = lecture['lecture_id']
    target_url = LECTURE_URL.format(lecture_id=lecture_id)
    check_response(page.goto(target_url, wait_until="domcontentloaded"))
    if looks_like_login_page(page):
        if args.headless:
            raise RuntimeError("저장된 로그인 세션이 없습니다. 창을 띄워 직접 로그인한 뒤 다시 실행하세요.")
        wait_for_manual_login(page, target_url)

    # domcontentloaded precedes asynchronous review rendering.
    try:
        page.locator(', '.join(REVIEW_CONTAINER_SELECTORS)).first.wait_for(
            state='visible', timeout=args.review_wait_seconds * 1_000,
        )
    except PlaywrightTimeoutError as exc:
        if page.is_closed() or page.url != target_url or looks_like_login_page(page):
            raise RuntimeError('리뷰 대기 중 로그인 상태 또는 강의 페이지가 변경되어 중단합니다.') from exc
        body = page.locator('body').inner_text(timeout=3_000).lower()
        if any(marker in body for marker in ('abnormal access', 'requests have been blocked', '접근이 제한', '접근이 차단', 'captcha', '자동입력 방지')):
            raise RuntimeError('접근 제한 안내가 감지되어 수집을 중단합니다.') from exc
        raise ReviewUnavailable(f'{args.review_wait_seconds}초 동안 리뷰 요소 없음: 리뷰 0개·로딩 지연·화면 변경 여부 확인 필요') from exc

    new_count = 0
    stagnant_rounds = 0
    parse_failed = False
    lecture['_scan_status'] = 'partial'
    for page_number in range(1, args.max_pages + 1):
        containers = find_review_containers(page)
        reviews = read_review_batch(containers, lecture, page.url)
        before = len(reviews)
        batch: list[Review] = []
        existing_reviews: list[Review] = []
        for review in reviews:
            if review is None:
                parse_failed = True
            if review and review.source_hash not in seen:
                seen.add(review.source_hash)
                batch.append(review)
            elif review:
                existing_reviews.append(review)

        updated = update_recommendations(output, existing_reviews) if args.refresh else 0
        written = append_jsonl(output, batch)
        if updated:
            logging.info('강의 %s: 기존 리뷰 추천 수 %d개 갱신', lecture_id, updated)
        new_count += written
        logging.info(
            "%s / %s / 강의 %s / %d회차: 화면 %d개, 새 리뷰 %d개",
            lecture.get('course_name') or '과목명 미지정',
            lecture.get('professor') or '교수 미표기',
            lecture_id, page_number, before, written,
        )
        if page_number >= args.max_pages:
            logging.warning('강의 %s: 최대 확인 횟수에 도달했습니다. 전체 수집 여부는 미확인입니다.', lecture_id)
            break
        progressed = click_more_or_scroll(page, before)
        stagnant_rounds = 0 if progressed else stagnant_rounds + 1
        if stagnant_rounds >= 2:
            lecture['_scan_status'] = 'partial' if parse_failed else 'scanned_completeness_unverified'
            logging.info('강의 %s: 추가 로딩 변화 없음. 전체 수집을 보장하는 판정은 아닙니다.', lecture_id)
            break
        time.sleep(random.uniform(args.delay_min, args.delay_max))
    return new_count


def crawl_and_record(page, lecture, args, seen, output):
    lecture.pop('_scan_status', None)
    status_path = output.with_name(output.stem + '_attempts.jsonl')
    record = {
        'lecture_id': lecture['lecture_id'],
        'course_name': lecture.get('course_name'),
        'professor': lecture.get('professor'),
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }
    try:
        count = crawl_one(page, lecture, args, seen, output)
        record.update(status=lecture.pop('_scan_status', 'scanned_completeness_unverified'), new_reviews=count)
    except ReviewUnavailable as exc:
        count = 0
        record.update(status='needs_check', reason=str(exc))
        logging.warning('%s / %s / 강의 %s: %s. 다음 강의로 넘어갑니다.',
                        lecture.get('course_name'), lecture.get('professor'), lecture['lecture_id'], exc)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return count


def select_pending_lectures(lectures, output, *, retry_failed=False, refresh=False):
    # Absence of raw data must never result in skipping a previous scan.
    if refresh or not output.exists():
        return lectures, 0
    latest = {}
    status_path = output.with_name(output.stem + '_attempts.jsonl')
    if status_path.exists():
        for line in status_path.read_text(encoding='utf-8-sig').splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get('lecture_id'):
                    latest[str(row['lecture_id'])] = row
            except json.JSONDecodeError:
                continue
    saved_ids = set()
    with output.open(encoding='utf-8-sig') as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get('lecture_id'):
                    saved_ids.add(str(row['lecture_id']))
            except json.JSONDecodeError:
                continue
    pending = []
    for lecture in lectures:
        key = str(lecture['lecture_id'])
        status = latest.get(key, {}).get('status')
        skip = (status == 'scanned_completeness_unverified' and key in saved_ids)
        skip = skip or (status == 'needs_check' and not retry_failed)
        if not skip:
            pending.append(lecture)
    return pending, len(lectures) - len(pending)


def load_candidate_lectures(state, course_name=None, all_courses=False, professor=None):
    groups = state.get('course_groups', {})
    if not all_courses and course_name not in groups:
        raise RuntimeError(f"'{course_name}' 과목 그룹을 찾지 못했습니다.")
    selected = groups.items() if all_courses else [(course_name, groups[course_name])]
    unique = {}
    for name, group in selected:
        for professor_group in group.get('professors', {}).values():
            for original in professor_group.get('lectures', []):
                lecture = dict(original)
                lecture['lecture_id'] = str(lecture['lecture_id'])
                if not lecture['lecture_id'].isdigit():
                    raise RuntimeError('검색 결과의 강의 ID 형식이 올바르지 않습니다.')
                lecture['course_name'] = lecture.get('course_name') or group.get('course_name') or name
                if professor and lecture.get('professor') != professor:
                    continue
                existing = unique.get(lecture['lecture_id'])
                if existing and (existing.get('course_name'), existing.get('professor')) != (lecture.get('course_name'), lecture.get('professor')):
                    logging.warning('강의 %s의 과목/교수 정보가 중복 후보 간 다릅니다. 첫 후보를 유지합니다.', lecture['lecture_id'])
                unique.setdefault(lecture['lecture_id'], lecture)
    return list(unique.values())


def crawl(args: argparse.Namespace) -> int:
    if args.all_courses and (not args.candidate_json or args.lecture_id):
        raise RuntimeError('--all-courses는 --candidate-json과 함께 사용하고 개별 강의 ID는 지정하지 마세요.')
    if args.export_only:
        source = Path(args.output).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f'변환할 리뷰 원본이 없습니다: {source}')
        save_grouped_reviews(source, Path(args.by_course_output), Path(args.by_professor_output))
        print(f'과목별 JSON: {Path(args.by_course_output).resolve()}')
        print(f'교수별 JSON: {Path(args.by_professor_output).resolve()}')
        return 0
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright가 설치되지 않았습니다. `pip install -r requirements.txt` 후 "
            "`playwright install chromium`을 실행하세요."
        )

    output = Path(args.output).expanduser().resolve()
    by_course_output = Path(args.by_course_output).expanduser().resolve()
    by_professor_output = Path(args.by_professor_output).expanduser().resolve()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    seen = load_seen_hashes(output)
    if args.candidate_json:
        state = json.loads(args.candidate_json.expanduser().read_text(encoding='utf-8-sig'))
        lectures = load_candidate_lectures(state, args.course_name, args.all_courses, args.professor)
        if not lectures:
            raise RuntimeError('선택한 과목명/교수의 강의 ID 후보가 없습니다.')
    else:
        if not args.lecture_id:
            raise RuntimeError('lecture_id 또는 --candidate-json과 --course-name을 지정하세요.')
        lectures = [{'lecture_id':args.lecture_id, 'course_name':args.course_name, 'professor':args.professor}]
    new_count = 0

    total = len(lectures)
    lectures, skipped = select_pending_lectures(lectures, output, retry_failed=args.retry_failed, refresh=args.refresh)
    print(f'전체 {total}개 / 이전 처리 건너뜀 {skipped}개 / 이번 수집 {len(lectures)}개')
    if not lectures:
        save_grouped_reviews(output, by_course_output, by_professor_output)
        print('새 수집 대상이 없습니다. 확인 필요 재시도: --retry-failed / 전체 재확인·추천 수 갱신: --refresh')
        return 0

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=args.headless,
            locale="ko-KR",
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            for index, lecture in enumerate(lectures, 1):
                print(f"[{index}/{len(lectures)}] {lecture.get('course_name')} / {lecture.get('professor') or '교수 미표기'} / 강의 {lecture['lecture_id']}", flush=True)
                new_count += crawl_and_record(page, lecture, args, seen, output)
                save_grouped_reviews(output, by_course_output, by_professor_output)
                time.sleep(random.uniform(args.delay_min, args.delay_max))
        finally:
            context.close()
            if output.exists():
                save_grouped_reviews(output, by_course_output, by_professor_output)

    save_grouped_reviews(output, by_course_output, by_professor_output)
    print(f"실행 종료: 새 강의평 {new_count}개 → {output}")
    print(f"강의별 확인 기록(확인 필요 항목 포함): {output.with_name(output.stem + '_attempts.jsonl')}")
    print(f"과목명 → 교수명: {by_course_output}")
    print(f"교수명 → 과목명: {by_professor_output}")
    return new_count


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수여야 합니다.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="에브리타임 강의평을 JSONL로 저장합니다.")
    parser.add_argument("lecture_id", nargs="?", help="강의 URL의 숫자 ID (예: 2216165)")
    parser.add_argument("--candidate-json", type=Path, help="find_everytime_lectures.py의 검색 결과 파일")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--course-name", help="정확한 에타 과목명 그룹 (예: 자료구조)")
    selection.add_argument("--all-courses", action="store_true", help="검색 결과에 포함된 모든 과목의 강의를 순서대로 수집")
    parser.add_argument("--professor", help="지정하면 해당 교수의 강의만 수집")
    parser.add_argument("--export-only", action="store_true", help="접속 없이 기존 JSONL에서 두 그룹 JSON만 생성")
    parser.add_argument("--retry-failed", action="store_true", help="기본적으로 보류하는 확인 필요 강의도 다시 시도")
    parser.add_argument("--refresh", action="store_true", help="처리 기록과 관계없이 전체 재방문 및 기존 리뷰 추천 수 갱신")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "reviews.jsonl"),
        help="출력 JSONL 경로",
    )
    parser.add_argument(
        "--by-course-output",
        default=str(DATA_DIR / "reviews_by_course.json"),
        help="과목명 → 교수명 기준 리뷰 파일",
    )
    parser.add_argument(
        "--by-professor-output",
        default=str(DATA_DIR / "reviews_by_professor.json"),
        help="교수명 → 과목명 기준 리뷰 파일",
    )
    parser.add_argument(
        "--profile-dir",
        default=str(REPO_ROOT / ".cache/everytime-browser-profile"),
        help="로그인 세션을 보관할 로컬 브라우저 프로필 경로",
    )
    parser.add_argument("--max-pages", type=positive_int, default=20)
    parser.add_argument("--review-wait-seconds", type=positive_int, default=3,
                        help="첫 리뷰가 나타나길 기다릴 시간(초, 기본 3). 초과 시 확인 필요로 기록하고 다음 강의 진행")
    parser.add_argument("--delay-min", type=float, default=1.5)
    parser.add_argument("--delay-max", type=float, default=3.0)
    parser.add_argument("--headless", action="store_true", help="저장된 세션으로 창 없이 실행")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.delay_min < 0 or args.delay_max < args.delay_min:
        print("오류: 지연 시간은 0 이상이고 max가 min 이상이어야 합니다.", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        crawl(args)
    except (RuntimeError, PlaywrightError, PlaywrightTimeoutError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("사용자가 중단했습니다. 이미 기록한 JSONL은 보존됩니다.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
