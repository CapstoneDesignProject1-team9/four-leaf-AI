"""Search official course names using a manually authenticated browser.

Selectors must be taken from the real search screen; no private API is used.
Results are candidates, not verified course/professor matches.
"""
import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlencode, parse_qs

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA_DIR = REPO_ROOT / 'knowledge/raw/everytime_lectures'
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(HERE / '.playwright-browsers'))
DEFAULT_COURSES = HERE / 'knu_computer_courses.json'
if not DEFAULT_COURSES.exists():
    DEFAULT_COURSES = HERE.parent / 'knu_courses' / 'knu_computer_courses.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def normalized(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def load_courses(path, include_other=False, selected=None):
    courses = read_json(path)["courses"]
    result = []
    seen = set()
    for course in courses:
        code = course["course_code"]
        if code in seen:
            raise ValueError(f"중복 과목코드: {code}")
        seen.add(code)
        if selected and code not in selected:
            continue
        if not include_other and not course.get("is_major_course", False):
            continue
        queries = list(dict.fromkeys(q.strip() for q in course.get("search_queries", [course["course_name"]]) if q.strip()))
        if not queries:
            raise ValueError(f"검색어 없음: {code}")
        result.append({"course_code": code, "course_name": course["course_name"], "queries": queries})
    if selected and set(selected) - {c['course_code'] for c in result}:
        raise ValueError("선택한 과목코드가 없거나 전공 필터에서 제외되었습니다.")
    return result


def candidate(link, search_url):
    url = urljoin(search_url, link['href'])
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ('everytime.kr', 'www.everytime.kr'):
        return None
    match = re.fullmatch(r'/lecture/view/(\d+)/?', parsed.path)
    if not match:
        return None
    lecture_id = match.group(1)
    return {'lecture_id': lecture_id, 'url': f'https://everytime.kr/lecture/view/{lecture_id}?tab=article',
            'result_text': link.get('text', '').strip(), 'verification': 'unverified'}


def fingerprint(course, config):
    # Change the resume key whenever candidate-matching rules change.
    return hashlib.sha256(json.dumps([course, config, 'exact-title-v4-scroll'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def search_url(query):
    return 'https://everytime.kr/lecture/search?' + urlencode({'keyword':query, 'condition':'name'})


class SearchSessionError(RuntimeError):
    """The session needs user attention; do not keep sending searches."""


def search_direct(page, query, config, timeout, max_scrolls):
    # A full document navigation gives each query a fresh results page.
    response = page.goto(search_url(query), wait_until='domcontentloaded')
    if response is not None and response.status in (401, 403, 429):
        raise SearchSessionError(f'검색 접근 제한 응답({response.status}). 실행을 중단합니다.')
    parsed = urlparse(page.url)
    params = parse_qs(parsed.query)
    if parsed.path != '/lecture/search' or params.get('keyword') != [query] or params.get('condition') != ['name']:
        raise SearchSessionError('검색 페이지 대신 다른 화면으로 이동했습니다. 로그인 상태를 확인하세요.')
    selector = 'a[href*="/lecture/view/"]'
    try:
        page.locator(selector).first.wait_for(state='visible',timeout=timeout)
    except Exception:
        if page.is_closed() or '/login' in page.url:
            raise SearchSessionError('브라우저가 닫혔거나 로그인 화면으로 이동했습니다.')
        empty = config.get('empty_selector')
        if empty and page.locator(empty).is_visible():
            return {'candidates': [], 'coverage': 'explicit_empty_state', 'scroll_rounds': 0, 'loaded_lecture_count': 0}
        raise RuntimeError('강의 링크를 확인하지 못했습니다. 결과 없음·로딩 실패·DOM 변경 중 원인을 확인해야 합니다.')
    if config.get('loading_selector'):
        page.locator(config['loading_selector']).wait_for(state='hidden',timeout=timeout)
    found = {}
    loaded_ids = set()
    stable_rounds = 0
    coverage = 'no_new_ids_observed_completeness_unverified'
    scroll_rounds = 0
    for scroll_rounds in range(1, max_scrolls + 1):
        links = page.locator(selector).evaluate_all(
            "els => els.filter(e => e.getClientRects().length).map(e => ({href:e.getAttribute('href'),text:e.innerText}))")
        before_ids = len(loaded_ids)
        for link in links:
            item = candidate(link,page.url)
            if not item:
                continue
            loaded_ids.add(item['lecture_id'])
            lines = [line.strip() for line in link.get('text','').splitlines() if line.strip()]
            if not lines or normalized(lines[0]) != normalized(query):
                continue
            if item:
                item['course_name']=lines[0]
                item['professor']=lines[1] if len(lines)>1 else None
                item['name_match']='exact_normalized_title'
                found[item['lecture_id']]=item

        print(f'[{query}] 로딩된 강의 {len(loaded_ids)}개 / 과목명 일치 {len(found)}개 / 확인 {scroll_rounds}회', flush=True)
        if len(loaded_ids) == before_ids:
            stable_rounds += 1
        else:
            stable_rounds = 0
        if stable_rounds >= 3:
            break
        if scroll_rounds == max_scrolls:
            coverage = 'max_scroll_rounds_reached'
            break

        # Click explicit result expansion controls when present, then scroll all
        # page panes so infinite-scroll result lists can load more entries.
        expanded = False
        for label in ('더보기', '더 보기', '다음 페이지', '다음'):
            for role in ('button', 'link'):
                control = page.get_by_role(role, name=re.compile(rf'^\s*{re.escape(label)}\s*$'))
                if control.count() and control.first.is_visible() and control.first.is_enabled():
                    control.first.click(timeout=2_000)
                    expanded = True
                    break
            if expanded:
                break
        metrics = page.evaluate("""() => {
          const scrollables=[...document.querySelectorAll('*')].filter(e => {
            const s=getComputedStyle(e); return /(auto|scroll)/.test(s.overflowY) && e.scrollHeight>e.clientHeight+20;
          });
          const panes=scrollables.map(e => { e.scrollTop=e.scrollHeight; return `${e.scrollTop}/${e.scrollHeight}`; });
          const root=document.scrollingElement; root.scrollTop=root.scrollHeight;
          return {height:root.scrollHeight, panes:panes.join('|')};
        }""")
        page.wait_for_timeout(2000)
    else:
        coverage = 'max_scroll_rounds_reached'

    return {'candidates':list(found.values()), 'coverage':coverage, 'scroll_rounds':scroll_rounds, 'loaded_lecture_count':len(loaded_ids)}


def build_course_groups(course_states):
    groups = {}
    for state in course_states.values():
        name = state.get('course_name','').strip()
        key = normalized(name)
        if not key:
            continue
        group = groups.setdefault(key, {'course_name':name,'course_codes':[],'professors':{}})
        if state.get('course_code') not in group['course_codes']:
            group['course_codes'].append(state['course_code'])
        for candidate in state.get('candidates',[]):
            professor = candidate.get('professor') or '교수 미표기'
            prof_group = group['professors'].setdefault(professor,{'lectures':[]})
            if not any(row.get('lecture_id') == candidate.get('lecture_id') for row in prof_group['lectures']):
                prof_group['lectures'].append(candidate)
    return {group['course_name']:group for group in groups.values()}


def read_config(path):
    config = read_json(path)
    keys = ('search_url',) if config.get('mode') == 'direct_url' else ('search_url', 'input_selector', 'results_selector', 'empty_selector')
    for key in keys:
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'설정 파일의 {key}를 실제 검색 화면에 맞춰 입력하세요.')
    url = urlparse(config['search_url'])
    if url.scheme != 'https' or url.hostname not in ('everytime.kr', 'www.everytime.kr'):
        raise ValueError('search_url은 https://everytime.kr의 검색 화면이어야 합니다.')
    return config


def search(page, query, config, timeout, max_scrolls):
    if config.get('mode') == 'direct_url':
        return search_direct(page,query,config,timeout,max_scrolls)
    page.goto(config['search_url'], wait_until='domcontentloaded')
    if '/login' in page.url:
        raise RuntimeError('로그인 세션이 만료되었습니다. 다시 직접 로그인하세요.')
    field = page.locator(config['input_selector'])
    if field.count() != 1:
        raise RuntimeError('검색 입력란 선택자는 요소 하나만 가리켜야 합니다.')
    # Snapshot before submit prevents treating the previously displayed list as
    # a response to the new query. Unchanged results time out conservatively.
    script = """c => {
      const visible = e => !!e && !!(e.getClientRects().length);
      const r = document.querySelector(c.results_selector);
      const e = document.querySelector(c.empty_selector);
      return JSON.stringify([visible(r) ? r.innerHTML : null, visible(e) ? e.innerText : null]);
    }"""
    before = page.evaluate(script, config)
    field.fill(query)
    if config.get('submit_selector'):
        page.locator(config['submit_selector']).click()
    else:
        field.press('Enter')
    page.wait_for_function("""({c, before}) => {
      const visible = e => !!e && !!e.getClientRects().length;
      const r = document.querySelector(c.results_selector), e = document.querySelector(c.empty_selector);
      const current = JSON.stringify([visible(r) ? r.innerHTML : null, visible(e) ? e.innerText : null]);
      return current !== before && (visible(e) || (visible(r) && !!r.querySelector('a[href*="/lecture/view/"]')));
    }""", arg={'c': config, 'before': before}, timeout=timeout)
    # Explicit UI configuration may include a loading indicator.
    if config.get('loading_selector'):
        page.locator(config['loading_selector']).wait_for(state='hidden', timeout=timeout)
    links = page.locator(config['results_selector']).locator('a[href*="/lecture/view/"]').evaluate_all(
        "els => els.filter(e => e.getClientRects().length).map(e => ({href:e.getAttribute('href'),text:e.innerText}))")
    found = {}
    for link in links:
        item = candidate(link, page.url)
        if item:
            found[item['lecture_id']] = item
    if not found and not page.locator(config['empty_selector']).is_visible():
        raise RuntimeError('검색 결과도 명시적인 결과 없음 표시도 확인되지 않았습니다.')
    return {'candidates':list(found.values()), 'coverage':'single_search_results_container', 'scroll_rounds':1}


def main(argv=None):
    parser = argparse.ArgumentParser(description='경북대 과목 목록으로 에타 강의 링크 후보 검색')
    parser.add_argument('--courses', type=Path, default=DEFAULT_COURSES)
    parser.add_argument('--config', type=Path, default=HERE/'search_config.json')
    parser.add_argument('--output', type=Path, default=DATA_DIR/'lecture_candidates.json')
    parser.add_argument('--profile-dir', type=Path, default=REPO_ROOT/'.cache/everytime-browser-profile')
    parser.add_argument('--course-code', action='append')
    parser.add_argument('--include-other', action='store_true')
    parser.add_argument('--limit', type=int, default=3, help='이번 실행 과목 수. 0이면 전체. 기본 3')
    parser.add_argument('--delay', type=float, default=3)
    parser.add_argument('--timeout', type=int, default=20000, help='검색 응답 대기 밀리초')
    parser.add_argument('--max-result-scrolls', type=int, default=80, help='검색 결과 추가 로딩 최대 스크롤 회수')
    parser.add_argument('--plan', action='store_true', help='브라우저 없이 검색 대상 확인')
    parser.add_argument('--retry', action='store_true', help='저장된 검색 완료 과목도 재검색')
    args = parser.parse_args(argv)
    if args.limit < 0 or not math.isfinite(args.delay) or args.delay < 1 or args.timeout < 1 or args.max_result_scrolls < 2:
        parser.error('limit >= 0, delay >= 1, timeout >= 1, max-result-scrolls >= 2이어야 합니다.')
    courses = load_courses(args.courses, args.include_other, args.course_code)
    if args.plan:
        print(json.dumps({'count':len(courses), 'courses':courses}, ensure_ascii=False, indent=2))
        return 0
    config = read_config(args.config)
    from playwright.sync_api import sync_playwright
    state = read_json(args.output) if args.output.exists() else {'schema_version':1, 'courses':{}}
    state.setdefault('courses',{})
    state.setdefault('course_groups',{})
    pending = [c for c in courses if args.retry or state['courses'].get(c['course_code'], {}).get('fingerprint') != fingerprint(c,config)
               or state['courses'].get(c['course_code'],{}).get('search_status') not in ('candidates_found','no_results')]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print('검색할 새 과목이 없습니다. 재검색은 --retry를 사용하세요.'); return 0
    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(str(args.profile_dir.resolve()),headless=False,locale='ko-KR')
        try:
            page = context.new_page()
            page.goto(config['search_url'], wait_until='domcontentloaded')
            input('브라우저에서 직접 로그인하고 경북대 강의평 검색 화면을 확인한 뒤 Enter: ')
            failure_count = 0
            for course_number, course in enumerate(pending, 1):
                print(f"[{course_number}/{len(pending)}] {course['course_name']} 검색 시작", flush=True)
                record = {**course, 'fingerprint':fingerprint(course,config), 'search_status':'in_progress',
                          'coverage':'scanning_results', 'queries_completed':[], 'query_scans':[], 'candidates':[],
                          'checked_at':datetime.now(timezone.utc).isoformat()}
                previous = state['courses'].get(course['course_code'])
                try:
                    items = {}
                    query_errors = []
                    for query in course['queries']:
                        page.wait_for_timeout(args.delay*1000)
                        print(f'  검색어: {query}', flush=True)
                        try:
                            scan = search(page,query,config,args.timeout,args.max_result_scrolls)
                        except SearchSessionError:
                            raise
                        except Exception as exc:
                            if page.is_closed():
                                raise SearchSessionError('브라우저가 닫혀 검색을 중단합니다.') from exc
                            query_errors.append({'query':query, 'url':page.url, 'error':str(exc)})
                            record['query_errors'] = query_errors
                            record['search_status'] = 'needs_check'
                            state['courses'][course['course_code']] = record
                            save_json(args.output,state)
                            print(f'  확인 필요: {exc}. 다음 검색어/과목으로 진행합니다.', flush=True)
                            continue
                        found = scan['candidates']
                        record['query_scans'].append({'query':query,'candidate_count':len(found),
                                                      'coverage':scan['coverage'],'scroll_rounds':scan['scroll_rounds'],
                                                      'loaded_lecture_count':scan.get('loaded_lecture_count')})
                        record['coverage'] = scan['coverage']
                        for item in found:
                            saved = items.setdefault(item['lecture_id'],{**item,'found_by_queries':[]})
                            saved['found_by_queries'].append(query)
                        record['queries_completed'].append(query)
                        record['candidates'] = list(items.values())
                        state['courses'][course['course_code']] = record
                        state['course_groups'] = build_course_groups(state['courses'])
                        save_json(args.output,state)
                    record['search_status'] = ('needs_check' if query_errors else
                                               'candidates_found' if items else 'no_results')
                    if query_errors:
                        failure_count += 1
                        record['coverage'] = 'incomplete_queries_need_check'
                except Exception as exc:
                    record['search_status']='error'
                    record['error']=str(exc)
                    if previous:
                        record['previous_candidates']=previous.get('candidates',[])
                    state['courses'][course['course_code']]=record
                    state['course_groups'] = build_course_groups(state['courses'])
                    save_json(args.output,state)
                    raise
                state['courses'][course['course_code']]=record
                state['course_groups'] = build_course_groups(state['courses'])
                save_json(args.output,state)
                print(course['course_code'],course['course_name'],record['search_status'],len(record['candidates']))
        finally:
            context.close()
    print(f'저장: {args.output.resolve()}')
    print(f'이번 실행 확인 필요 과목: {failure_count}개 (query_errors에 원인 기록)')
    return 2 if failure_count else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('중단했습니다. 저장된 결과는 유지됩니다.'); raise SystemExit(130)
    except Exception as exc:
        print(f'실행 실패: {exc}'); raise SystemExit(1)
