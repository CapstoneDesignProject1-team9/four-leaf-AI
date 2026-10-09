"""Selenium crawler for public KNU Computer Science syllabi."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import sys
import threading
import time
from datetime import date
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "knowledge" / "raw" / "syllabi" / "knu_cs_syllabi.jsonl"
START_URL = "https://knuin.knu.ac.kr/public/stddm/lectPlnInqr.knu"
DEFAULT_TIMEOUT = 30
IT_COLLEGE_DEPARTMENTS = (
    "글로벌소프트웨어융합전공",
    "심화컴퓨팅전공",
    "인공지능컴퓨팅전공",
    "컴퓨터학부",
    "플랫폼소프트웨어전공",
)
GRID_ID = "grid01"
GRID_COLUMNS = {
    "year": 0,
    "semester": 1,
    "grade": 2,
    "category": 3,
    "college": 4,
    "department": 5,
    "course_code": 6,
    "course_name": 7,
    "credits": 8,
    "lecture_hours": 9,
    "practice_hours": 10,
    "instructor": 11,
    "class_time": 12,
    "actual_time": 13,
    "classroom": 14,
    "room": 15,
    "capacity": 16,
    "enrolled": 17,
}
SYLLABUS_TABS = {
    "general": "tabs1",
    "core_competencies": "tabs2",
    "evaluation_methods": "tabs3",
    "disability_support": "tabs4",
    "weekly_schedule": "tabs5",
    "course_evaluation": "tabs6",
}
COURSE_FIELDS = tuple(GRID_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="경북대학교 IT대학 강의계획서 수집기")
    parser.add_argument("--year", type=int, default=date.today().year, help="개설연도")
    parser.add_argument("--semester", default="2학기", help="개설학기 (예: 1학기, 2학기, 계절학기(하계))")
    parser.add_argument(
        "--department",
        action="append",
        choices=IT_COLLEGE_DEPARTMENTS,
        help="조회할 IT대학 학과/전공. 생략하면 지정된 5개 모두 조회 (여러 번 지정 가능)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="JSONL 결과 파일")
    parser.add_argument("--max-courses", type=int, default=0, help="최대 강좌 수 (0은 제한 없음)")
    parser.add_argument("--delay", type=float, default=0.25, help="과목 간 대기 시간(초)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="요소 대기 시간(초)")
    parser.add_argument("--headless", action="store_true", help="브라우저 창 없이 실행")
    parser.add_argument(
        "--manual-details",
        action="store_true",
        help="브라우저에서 강의를 더블클릭하면 상세 내용을 자동 수집하고 창을 닫음",
    )
    parser.add_argument("--append", action="store_true", help="기존 JSONL에 이어 쓰기")
    return parser.parse_args()


def make_driver(headless: bool = False) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--lang=ko-KR")
    options.add_argument("--window-size=1600,1100")
    options.add_argument("--disable-notifications")
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def wait_for_id(driver: webdriver.Chrome, element_id: str, timeout: int):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, element_id))
    )


def choose_websquare_option(
    driver: webdriver.Chrome, element_id: str, value: str, timeout: int
) -> None:
    wait_for_id(driver, element_id, timeout)
    tag_name = driver.execute_script(
        "const element = document.getElementById(arguments[0]); return element && element.tagName.toLowerCase();",
        element_id,
    )
    if tag_name == "select":
        script = """
          const select = document.getElementById(arguments[0]);
          if (!select) return false;
          const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
          const wanted = normalize(arguments[1]);
          const option = Array.from(select.options).find(item =>
            normalize(item.textContent) === wanted || normalize(item.value) === wanted
          );
          if (!option) return false;
          select.value = option.value;
          option.selected = true;
          select.dispatchEvent(new Event('input', { bubbles: true }));
          select.dispatchEvent(new Event('change', { bubbles: true }));
          return true;
        """
        for attempt in range(3):
            try:
                WebDriverWait(
                    driver,
                    timeout,
                    ignored_exceptions=(StaleElementReferenceException,),
                ).until(
                    lambda current: current.execute_script(
                        """
                        const select = document.getElementById(arguments[0]);
                        if (!select) return false;
                        const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
                        const wanted = normalize(arguments[1]);
                        return Array.from(select.options).some(item =>
                          normalize(item.textContent) === wanted || normalize(item.value) === wanted
                        );
                        """,
                        element_id,
                        value,
                    )
                )
                if driver.execute_script(script, element_id, value):
                    return
            except StaleElementReferenceException:
                if attempt == 2:
                    raise
                time.sleep(0.2)
        raise TimeoutException(f"{element_id}에서 '{value}' 선택지를 찾지 못했습니다.")

    for _ in range(3):
        try:
            element = wait_for_id(driver, element_id, timeout)
            element.click()
            options = driver.find_elements(
                By.XPATH, f"//*[normalize-space(.)={json.dumps(value, ensure_ascii=False)}]"
            )
            for option in options:
                if option.is_displayed() and option.is_enabled():
                    option.click()
                    return
            element.send_keys(value)
            element.send_keys(Keys.ENTER)
            return
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise StaleElementReferenceException(f"{element_id}이 계속 갱신되어 '{value}'를 선택하지 못했습니다.")


def set_search_filters(driver: webdriver.Chrome, year: int, semester: str, timeout: int) -> None:
    for attempt in range(3):
        try:
            year_input = wait_for_id(driver, "schEstblYear___input", timeout)
            year_input.click()
            year_input.clear()
            year_input.send_keys(str(year))
            actual_year = driver.execute_script(
                "const input = document.getElementById('schEstblYear___input'); return input && input.value"
            )
            if actual_year != str(year):
                driver.execute_script(
                    """
                    const input = document.getElementById('schEstblYear___input');
                    input.value = arguments[0];
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                    """,
                    str(year),
                )
            wait_for_id(driver, "schEstblYear___input", timeout).send_keys(Keys.TAB)
            actual_year = driver.execute_script(
                "const input = document.getElementById('schEstblYear___input'); return input && input.value"
            )
            if actual_year != str(year):
                raise RuntimeError(f"개설연도 입력 실패: 기대값={year}, 입력값={actual_year}")
            break
        except StaleElementReferenceException:
            if attempt == 2:
                raise
            time.sleep(0.2)
    choose_websquare_option(driver, "schEstblSmstrSctcd", semester, timeout)
    choose_websquare_option(driver, "schSbjetCd1", "대학", timeout)
    choose_websquare_option(driver, "schSbjetCd2", "IT대학", timeout)


def search_department(
    driver: webdriver.Chrome, department: str, timeout: int
) -> list[dict[str, str]]:
    choose_websquare_option(driver, "schSbjetCd3", department, timeout)
    click_search(driver, timeout)
    rows = collect_all_rows(driver, timeout)
    for row in rows:
        row["_department_filter"] = department
    return rows


def click_search(driver: webdriver.Chrome, timeout: int) -> None:
    previous_codes = tuple(row["course_code"] for row in read_rendered_rows(driver))
    wait_for_id(driver, "btnSearch", timeout)
    driver.execute_script(
        "const button = document.getElementById('btnSearch'); if (!button) throw new Error('조회 버튼을 찾지 못했습니다'); button.click();"
    )
    if previous_codes:
        WebDriverWait(driver, timeout).until(
            lambda current: tuple(
                row["course_code"] for row in read_rendered_rows(current)
            ) != previous_codes
        )
    else:
        WebDriverWait(driver, timeout).until(
            lambda current: current.find_elements(
                By.CSS_SELECTOR, f"[id^='{GRID_ID}_cell_'][id$='_6']"
            )
        )


def read_rendered_rows(driver: webdriver.Chrome) -> list[dict[str, str]]:
    cells = driver.find_elements(By.CSS_SELECTOR, f"[id^='{GRID_ID}_cell_'][id$='_6']")
    indexes = set()
    for cell in cells:
        try:
            cell_id = cell.get_attribute("id") or ""
        except StaleElementReferenceException:
            continue
        match = re.match(rf"{GRID_ID}_cell_(\d+)_6$", cell_id)
        if match:
            indexes.add(int(match.group(1)))

    rows = []
    for index in sorted(indexes):
        row = {}
        for field, column in GRID_COLUMNS.items():
            try:
                element = driver.find_element(By.ID, f"{GRID_ID}_cell_{index}_{column}")
                row[field] = " ".join(element.text.split())
            except WebDriverException:
                row[field] = ""
        if row.get("course_code") and "컴퓨터학부" in row.get("department", ""):
            row["_grid_index"] = str(index)
            rows.append(row)
    return rows


def collect_all_rows(driver: webdriver.Chrome, timeout: int) -> list[dict[str, str]]:
    """그리드가 가상 렌더링을 사용해도 스크롤해 전체 강좌를 읽습니다."""
    rows_by_code: dict[str, dict[str, str]] = {}
    body = wait_for_id(driver, f"{GRID_ID}_body_tbody", timeout)
    container = driver.execute_script(
        """
        let node = arguments[0];
        while (node && node !== document.body) {
          if (node.scrollHeight > node.clientHeight + 5) return node;
          node = node.parentElement;
        }
        return arguments[0];
        """,
        body,
    )
    last_top = -1
    while True:
        for row in read_rendered_rows(driver):
            rows_by_code[row["course_code"]] = row
        top, height, view = driver.execute_script(
            "return [arguments[0].scrollTop, arguments[0].scrollHeight, arguments[0].clientHeight]",
            container,
        )
        if top + view >= height - 2 or top == last_top:
            break
        last_top = top
        driver.execute_script(
            "arguments[0].scrollTop = Math.min(arguments[0].scrollTop + arguments[0].clientHeight * 0.8, arguments[0].scrollHeight)",
            container,
        )
        time.sleep(0.2)
    driver.execute_script("arguments[0].scrollTop = 0", container)
    return list(rows_by_code.values())


def open_course_detail(driver: webdriver.Chrome, row: dict[str, str], timeout: int):
    main_handle = driver.current_window_handle
    handles_before = set(driver.window_handles)
    cell = driver.find_element(By.ID, f"{GRID_ID}_cell_{row['_grid_index']}_0")
    ActionChains(driver).double_click(cell).perform()
    try:
        WebDriverWait(driver, timeout).until(
            lambda current: len(set(current.window_handles) - handles_before) > 0
        )
    except TimeoutException:
        return None
    detail_handle = (set(driver.window_handles) - handles_before).pop()
    driver.switch_to.window(detail_handle)
    wait_for_id(driver, "popupContent", timeout)
    return main_handle, detail_handle


def read_syllabus_tabs(driver: webdriver.Chrome, timeout: int) -> dict[str, str]:
    details = {}
    for name, suffix in SYLLABUS_TABS.items():
        tab_id = f"popupContent_tabCon_tab_{suffix}_tabHTML"
        try:
            tab = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.ID, tab_id)))
            tab.click()
            time.sleep(0.15)
            details[name] = driver.find_element(By.ID, "popupContent").text.strip()
        except TimeoutException:
            details[name] = ""
    return details


def close_course_detail(driver: webdriver.Chrome, handles: tuple[str, str]) -> None:
    main_handle, detail_handle = handles
    if detail_handle in driver.window_handles:
        driver.switch_to.window(detail_handle)
        driver.close()
    if main_handle in driver.window_handles:
        driver.switch_to.window(main_handle)


def save_item(output, row: dict[str, str], details: dict[str, str]) -> None:
    content = "\n\n".join(f"[{name}]\n{text}" for name, text in details.items() if text)
    item = {field: row.get(field, "") for field in COURSE_FIELDS}
    item.update(
        {
            "department_filter": row.get("_department_filter", ""),
            "syllabus": {name: details.get(name, "") for name in SYLLABUS_TABS},
            "content": content,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "url": START_URL,
        }
    )
    output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    output.flush()


def course_metadata_from_syllabus(details: dict[str, str]) -> dict[str, str]:
    text = " ".join(details.get("general", "").split())
    metadata = {field: "" for field in COURSE_FIELDS}

    identity_patterns = (
        r"\bCode\s+([A-Z0-9-]+)\s+Course\s+Title\s+(.+?)\s+Credits\b",
        r"(?:교과목\s*코드|강좌번호)\s+([A-Z0-9-]+)\s+교과목명\s+(.+?)\s+(?:학점|이수구분)",
    )
    for pattern in identity_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["course_code"] = match.group(1).strip()
            metadata["course_name"] = match.group(2).strip()
            break
    if not metadata["course_code"]:
        return metadata

    field_patterns = {
        "year": (r"\bYear\s+(\d{4})\b", r"개설연도\s+(\d{4})\b"),
        "semester": (
            r"\bTerm\s+(.+?)\s+Course\s+Code\b",
            r"개설학기\s+(.+?)\s+(?:강좌번호|교과목\s*코드)\b",
        ),
        "category": (
            r"Course Categories\s+(.+?)\s+Classroom\b",
            r"교과목구분\s+(.+?)\s+강의언어\b",
        ),
        "instructor": (
            r"\bInstructor\s+(.+?)\s+Class\s+Time\b",
            r"담당교수\s+(.+?)\s+강의시간\b",
        ),
        "class_time": (
            r"\bClass\s+Time\s+(.+?)\s+Classroom\b",
            r"강의시간\s+(.+?)\s+강의실\b",
        ),
        "classroom": (
            r"\bClass\s+Time\s+.+?\s+Classroom\s+(.+?)\s+Subtitle\s+Support\b",
            r"강의실\s+(.+?)\s+장애학생\b",
        ),
    }
    for field, patterns in field_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[field] = match.group(1).strip()
                break

    if re.search(r"College of IT Engineering|개설대학\s+IT대학", text, re.IGNORECASE):
        metadata["college"] = "IT대학"
    if re.search(r"School of Computer Science and Engineering|컴퓨터학부", text, re.IGNORECASE):
        metadata["department"] = "컴퓨터학부"

    credits = re.search(r"(?:\bCredits\b|학점)\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", text, re.IGNORECASE)
    if credits:
        metadata["credits"], metadata["lecture_hours"], metadata["practice_hours"] = credits.groups()

    room = re.search(r"(?:^|\s)([A-Z]?\d{3}[A-Z]?)\s*$", metadata["classroom"])
    if room:
        metadata["room"] = room.group(1)
        metadata["classroom"] = metadata["classroom"][: room.start(1)].strip()
    return metadata


def course_identity_from_syllabus(details: dict[str, str]) -> dict[str, str]:
    metadata = course_metadata_from_syllabus(details)
    return {key: metadata[key] for key in ("course_code", "course_name") if metadata[key]}


def normalize_semester(value: str) -> str:
    normalized = " ".join((value or "").split()).lower()
    aliases = {
        "1st semester": "1학기",
        "first semester": "1학기",
        "2nd semester": "2학기",
        "second semester": "2학기",
    }
    return aliases.get(normalized, normalized)


def course_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    code = (row.get("course_code") or "").strip()
    if not code:
        return None
    return (
        (row.get("year") or "").strip(),
        normalize_semester(row.get("semester", "")),
        code,
    )


def load_existing_course_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = course_metadata_from_syllabus(item.get("syllabus", {}))
            row = dict(metadata)
            row.update({key: value for key, value in item.items() if value})
            key = course_key(row)
            if key:
                keys.add(key)
    return keys


def listen_for_manual_commands(
    commands: queue.Queue[str], ready_for_command: threading.Event
) -> None:
    while True:
        ready_for_command.wait()
        ready_for_command.clear()
        try:
            command = input("현재 학과 완료: Enter=다음 학과, q=종료: ").strip().lower()
        except EOFError:
            command = "q"
        commands.put(command)
        if command == "q":
            return


def collect_manual_details(
    driver: webdriver.Chrome,
    output,
    department: str,
    rows: list[dict[str, str]],
    timeout: int,
    max_courses: int,
    total_saved: int,
    commands: queue.Queue[str],
    ready_for_command: threading.Event,
    saved_course_keys: set[tuple[str, str, str]],
) -> tuple[int, bool]:
    main_handle = driver.current_window_handle
    rows_by_code = {row.get("course_code"): row for row in rows}
    processed_handles: set[str] = set()
    print(
        f"{department}: 브라우저에서 강의를 더블클릭하면 자동 저장 후 상세 창을 닫습니다.",
        flush=True,
    )
    ready_for_command.set()

    while max_courses <= 0 or total_saved < max_courses:
        try:
            open_handles = set(driver.window_handles)
        except WebDriverException:
            return total_saved, True
        for detail_handle in open_handles - {main_handle} - processed_handles:
            processed_handles.add(detail_handle)
            try:
                if detail_handle not in driver.window_handles:
                    continue
                driver.switch_to.window(detail_handle)
                wait_for_id(driver, "popupContent", timeout)
                details = read_syllabus_tabs(driver, timeout)
                driver.switch_to.window(main_handle)
                metadata = course_metadata_from_syllabus(details)
                row = {key: value for key, value in metadata.items() if value}
                catalog_row = rows_by_code.get(metadata.get("course_code"), {})
                row.update({key: value for key, value in catalog_row.items() if value})
                row.update(
                    {
                        key: metadata[key]
                        for key in ("course_code", "course_name")
                        if metadata.get(key)
                    }
                )
                row["_department_filter"] = department
                key = course_key(row)
                if key is None:
                    print("  [건너뜀: 강좌 코드를 읽지 못함]", flush=True)
                    continue
                if key in saved_course_keys:
                    print(f"  [중복 건너뜀] {row['course_code']}", flush=True)
                    continue
                save_item(output, row, details)
                saved_course_keys.add(key)
                total_saved += 1
                print(
                    f"  [자동 저장 후 창 닫음] {row.get('course_code', '과목 정보')} "
                    f"({total_saved}개 저장)",
                    flush=True,
                )
            except (TimeoutException, WebDriverException) as exc:
                if "no such window" not in str(exc).lower():
                    print(f"  [상세 읽기 실패] {exc.msg if isinstance(exc, WebDriverException) else exc}", flush=True)
            finally:
                try:
                    current_handles = set(driver.window_handles)
                    if detail_handle in current_handles:
                        driver.switch_to.window(detail_handle)
                        driver.close()
                    if main_handle in set(driver.window_handles):
                        driver.switch_to.window(main_handle)
                except WebDriverException:
                    pass

        if max_courses > 0 and total_saved >= max_courses:
            return total_saved, False
        try:
            command = commands.get(timeout=0.25)
        except queue.Empty:
            continue
        if command == "q":
            return total_saved, True
        if command in ("", "next", "n"):
            return total_saved, False
        print("Enter는 다음 학과, q는 종료입니다.", flush=True)
        ready_for_command.set()

    return total_saved, False


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    driver = make_driver(args.headless and not args.manual_details)
    stage = "페이지 열기"
    try:
        driver.set_page_load_timeout(args.timeout)
        driver.get(START_URL)
        stage = "연도·학기·대학 선택"
        set_search_filters(driver, args.year, args.semester, args.timeout)
        departments = args.department or list(IT_COLLEGE_DEPARTMENTS)
        mode = "a" if args.append or args.manual_details else "w"
        total_saved = 0
        total_attempted = 0
        seen_course_codes: set[str] = set()
        saved_course_keys = load_existing_course_keys(args.output)
        commands: queue.Queue[str] = queue.Queue()
        ready_for_command = threading.Event()
        if args.manual_details:
            threading.Thread(
                target=listen_for_manual_commands,
                args=(commands, ready_for_command),
                daemon=True,
            ).start()

        with args.output.open(mode, encoding="utf-8", newline="\n") as output:
            for department in departments:
                stage = f"{department} 선택 및 조회"
                rows = search_department(driver, department, args.timeout)
                print(
                    f"검색 완료: {args.year} {args.semester}, {department} {len(rows)}개 강좌",
                    flush=True,
                )
                if args.manual_details:
                    total_saved, should_stop = collect_manual_details(
                        driver,
                        output,
                        department,
                        rows,
                        args.timeout,
                        args.max_courses,
                        total_saved,
                        commands,
                        ready_for_command,
                        saved_course_keys,
                    )
                    if should_stop or (args.max_courses > 0 and total_saved >= args.max_courses):
                        break
                    continue

                for row in rows:
                    if row["course_code"] in seen_course_codes:
                        continue
                    if args.max_courses > 0 and total_attempted >= args.max_courses:
                        break
                    seen_course_codes.add(row["course_code"])
                    total_attempted += 1
                    number = total_attempted
                    print(
                        f"[{number}] {row['course_code']} {row['course_name']} "
                        f"({department})",
                        flush=True,
                    )
                    handles = open_course_detail(driver, row, args.timeout)
                    if handles is None:
                        print("  [강의계획서 없음 또는 열기 실패]")
                        continue
                    try:
                        details = read_syllabus_tabs(driver, args.timeout)
                        save_item(output, row, details)
                        total_saved += 1
                        print(f"  [저장] {args.output}")
                    except (TimeoutException, WebDriverException) as exc:
                        print(f"  [상세 읽기 실패] {exc}")
                    finally:
                        close_course_detail(driver, handles)
                    time.sleep(max(args.delay, 0))
                if args.max_courses > 0 and total_attempted >= args.max_courses:
                    break

        print(f"완료: {total_saved}개 강좌 저장, 저장 위치: {args.output}")
        return 0
    except TimeoutException as exc:
        print(f"[시간 초과] 조회 결과가 나타나지 않았습니다: {exc}", file=sys.stderr)
        return 1
    except WebDriverException as exc:
        print(
            f"[Selenium 오류: {stage}] "
            f"{exc.msg if isinstance(exc, WebDriverException) else exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())
