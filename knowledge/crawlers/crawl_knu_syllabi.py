"""경북대학교 컴퓨터학부 강의계획서 수집기.

`crawl_knu_notices.py`와 동일하게 requests + BeautifulSoup으로 HTTP/HTML을 처리합니다.
강의계획서 포털이 제공하는 공개 조회 결과 및 강의계획서 링크를 따라가며 HTML 내용을 JSONL로 저장합니다.
포털의 검색 요청 파라미터가 변경되면 환경변수 KNU_SYLLABUS_LIST_URL로 검색 결과 URL을 지정할 수 있습니다.

예시:
    python knowledge/crawlers/crawl_knu_syllabi.py --help
    python knowledge/crawlers/crawl_knu_syllabi.py --list-url 'https://...'

저장 위치:
    knowledge/raw/syllabi/knu_cs_syllabi.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "knowledge" / "raw" / "syllabi" / "knu_cs_syllabi.jsonl"
DEFAULT_LIST_URL = os.getenv("KNU_SYLLABUS_LIST_URL", "https://sy.knu.ac.kr/")
TIMEOUT = 25
DELAY = 0.4
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_department_text(soup: BeautifulSoup) -> str:
    """페이지 안의 컴퓨터학부/컴퓨터학과 표시를 포함한 행 텍스트를 찾습니다."""
    for row in soup.select("tr"):
        text = clean_text(row.get_text(" ", strip=True))
        if any(name in text for name in ("컴퓨터학부", "컴퓨터학과")):
            return text
    return ""


def collect_course_links(soup: BeautifulSoup, page_url: str) -> list[dict[str, str]]:
    """강의계획서/강의상세 링크를 모읍니다. 검색 결과 행의 학과명이 있는 경우만 포함합니다."""
    courses: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in soup.select("tr"):
        row_text = clean_text(row.get_text(" ", strip=True))
        department = (
            row_text
            if any(name in row_text for name in ("컴퓨터학부", "컴퓨터학과"))
            else ""
        )
        if not department:
            continue
        for anchor in row.select("a[href]"):
            href = anchor.get("href", "").strip()
            label = clean_text(anchor.get_text(" ", strip=True))
            if not href or href.startswith(("javascript:", "#")):
                continue
            # 포털에서 강의계획서 상세 링크를 구분하는 링크 텍스트 또는 URL만 대상으로 합니다.
            if not any(token in (label + " " + href).lower() for token in ("강의계획", "교과목상세", "lectpln", "syllabus")):
                continue
            url = urljoin(page_url, href)
            if url in seen:
                continue
            seen.add(url)
            courses.append({"url": url, "title": label or row_text, "department_row": department})
    return courses


def parse_course_detail(html: str, url: str, fallback_title: str, department_row: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("script, style, noscript, nav, header, footer"):
        element.decompose()
    main = soup.select_one("#content, #contents, main, .content, .sub_content") or soup.body or soup
    text = "\n".join(
        clean_text(line)
        for line in main.get_text("\n", strip=True).splitlines()
        if clean_text(line)
    )
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else fallback_title
    title = title or fallback_title
    course_code = ""
    match = re.search(r"(?<!\d)(\d{5,10})(?!\d)", department_row)
    if match:
        course_code = match.group(1)
    return {
        "course_code": course_code,
        "title": title,
        "department": "컴퓨터학부",
        "department_row": department_row,
        "url": url,
        "content": text,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="경북대학교 컴퓨터학부 강의계획서 수집기")
    parser.add_argument(
        "--list-url",
        default=DEFAULT_LIST_URL,
        help="강의계획서 조회 결과 URL (기본값: KNU_SYLLABUS_LIST_URL 또는 sy.knu.ac.kr)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"JSONL 출력 경로 (기본값: {OUTPUT})")
    parser.add_argument("--max-pages", type=int, default=0, help="다음/페이지 링크를 따라갈 최대 목록 페이지 수 (0: 제한 없음)")
    parser.add_argument("--delay", type=float, default=DELAY, help="요청 사이 대기 시간(초)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = make_session()
    queue = [args.list_url]
    visited_pages: set[str] = set()
    visited_courses: set[str] = set()
    page_count = 0
    # 목록 화면은 사용자가 포털에서 조회한 결과 URL로 제공할 수 있습니다.
    # 시스템 내부에서 조회 조건을 임의 추측해 전체 과목을 누락시키지 않습니다.
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        while queue and (args.max_pages <= 0 or page_count < args.max_pages):
            page_url = queue.pop(0)
            if page_url in visited_pages:
                continue
            visited_pages.add(page_url)
            page_count += 1
            print(f"[목록 {page_count}] {page_url}")
            try:
                response = session.get(page_url, timeout=TIMEOUT)
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"[목록 요청 실패] {exc}", file=sys.stderr)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            course_links = collect_course_links(soup, response.url)
            if not course_links and page_count == 1:
                print(
                    "[안내] 강의계획서 조회 결과에서 강의 링크를 찾지 못했습니다. "
                    "sy.knu.ac.kr에서 컴퓨터학부 검색을 실행한 결과 URL을 --list-url로 전달해 주세요.",
                    file=sys.stderr,
                )
            for course in course_links:
                url = course["url"]
                if url in visited_courses:
                    continue
                visited_courses.add(url)
                try:
                    detail_response = session.get(url, timeout=TIMEOUT)
                    detail_response.raise_for_status()
                except requests.RequestException as exc:
                    print(f"[강의계획서 요청 실패] {url}: {exc}", file=sys.stderr)
                    continue
                item = parse_course_detail(
                    detail_response.text,
                    detail_response.url,
                    course["title"],
                    course["department_row"],
                )
                output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                output.flush()
                print(f"[저장] {item['title']}")
                time.sleep(max(args.delay, 0))

            # 동일 호스트의 다음/페이지 이동 링크만 수집합니다.
            host = urlparse(response.url).netloc
            for anchor in soup.select("a[href]"):
                label = clean_text(anchor.get_text(" ", strip=True)).lower()
                href = anchor.get("href", "").strip()
                if not href or href.startswith(("javascript:", "#")):
                    continue
                if not any(token in label for token in ("다음", "next", "›", ">")):
                    continue
                next_url = urljoin(response.url, href)
                if urlparse(next_url).netloc == host and next_url not in visited_pages:
                    queue.append(next_url)
            time.sleep(max(args.delay, 0))

    print(f"완료: 목록 {page_count}페이지, 상세 {len(visited_courses)}건 확인, 저장 위치: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
