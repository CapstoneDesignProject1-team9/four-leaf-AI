"""Selenium crawler for public KNU Computer Science syllabi."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "knowledge" / "raw" / "syllabi" / "knu_cs_syllabi.jsonl"
START_URL = "https://knuin.knu.ac.kr/public/stddm/lectPlnInqr.knu"
DEFAULT_TIMEOUT = 30
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="경북대학교 컴퓨터학부 강의계획서 수집기")
    parser.add_argument("--year", type=int, default=date.today().year, help="개설연도")
    parser.add_argument("--semester", default="2학기", help="개설학기 (예: 1학기, 2학기, 계절학기(하계))")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="JSONL 결과 파일")
    parser.add_argument("--max-courses", type=int, default=0, help="최대 강좌 수 (0은 제한 없음)")
    parser.add_argument("--delay", type=float, default=0.25, help="과목 간 대기 시간(초)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="요소 대기 시간(초)")
    parser.add_argument("--headless", action="store_true", help="브라우저 창 없이 실행")
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
    element = wait_for_id(driver, element_id, timeout)
    if element.tag_name.lower() == "select":
        selector = Select(element)
        normalized = lambda text: " ".join((text or "").split())
        target = next(
            (
                option
                for option in selector.options
                if normalized(option.text) == normalized(value)
                or normalized(option.get_attribute("value")) == normalized(value)
            ),
            None,
        )
        if target is None:
            choices = [option.text or option.get_attribute("value") for option in selector.options]
            raise RuntimeError(f"{element_id}: '{value}' 선택지를 찾지 못했습니다. 현재 옵션: {choices}")
        selector.select_by_value(target.get_attribute("value"))
        driver.execute_script(
            "arguments[0].dispatchEvent(new Event('change', { bubbles: true }));",
            element,
        )
        return

    element.click()
    options = driver.find_elements(
        By.XPATH, f"//*[normalize-space(.)={json.dumps(value, ensure_ascii=False)}]"
    )
    for option in options:
        try:
            if option.is_displayed() and option.is_enabled():
                option.click()
                return
        except StaleElementReferenceException:
            continue
    element.send_keys(value)
    element.send_keys(Keys.ENTER)


def set_search_filters(driver: webdriver.Chrome, year: int, semester: str, timeout: int) -> None:
    year_input = wait_for_id(driver, "schEstblYear___input", timeout)
    year_input.click()
    year_input.send_keys(Keys.COMMAND, "a")
    year_input.send_keys(str(year))
    year_input.send_keys(Keys.TAB)
    choose_websquare_option(driver, "schEstblSmstrSctcd", semester, timeout)
    choose_websquare_option(driver, "schSbjetCd1", "대학", timeout)
    choose_websquare_option(driver, "schSbjetCd2", "IT대학", timeout)
    choose_websquare_option(driver, "schSbjetCd3", "컴퓨터학부", timeout)


def click_search(driver: webdriver.Chrome, timeout: int) -> None:
    wait_for_id(driver, "btnSearch", timeout).click()
    WebDriverWait(driver, timeout).until(
        lambda current: current.find_elements(
            By.CSS_SELECTOR, f"[id^='{GRID_ID}_cell_'][id$='_6']"
        )
    )


def read_rendered_rows(driver: webdriver.Chrome) -> list[dict[str, str]]:
    cells = driver.find_elements(By.CSS_SELECTOR, f"[id^='{GRID_ID}_cell_'][id$='_6']")
    indexes = set()
    for cell in cells:
        match = re.match(rf"{GRID_ID}_cell_(\d+)_6$", cell.get_attribute("id") or "")
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
    item = {
        **{key: value for key, value in row.items() if not key.startswith("_")},
        "department": "컴퓨터학부",
        "syllabus": details,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "url": START_URL,
    }
    output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    output.flush()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    driver = make_driver(args.headless)
    try:
        driver.set_page_load_timeout(args.timeout)
        driver.get(START_URL)
        set_search_filters(driver, args.year, args.semester, args.timeout)
        click_search(driver, args.timeout)
        rows = collect_all_rows(driver, args.timeout)
        if args.max_courses > 0:
            rows = rows[: args.max_courses]
        print(f"검색 완료: {args.year} {args.semester}, 컴퓨터학부 {len(rows)}개 강좌")

        with args.output.open("a" if args.append else "w", encoding="utf-8", newline="\n") as output:
            for number, row in enumerate(rows, start=1):
                print(f"[{number}/{len(rows)}] {row['course_code']} {row['course_name']}")
                handles = open_course_detail(driver, row, args.timeout)
                if handles is None:
                    print("  [강의계획서 없음 또는 열기 실패]")
                    continue
                try:
                    details = read_syllabus_tabs(driver, args.timeout)
                    save_item(output, row, details)
                    print(f"  [저장] {args.output}")
                except (TimeoutException, WebDriverException) as exc:
                    print(f"  [상세 읽기 실패] {exc}")
                finally:
                    close_course_detail(driver, handles)
                time.sleep(max(args.delay, 0))

        print(f"완료: {len(rows)}개 강좌 확인, 저장 위치: {args.output}")
        return 0
    except TimeoutException as exc:
        print(f"[시간 초과] 조회 결과가 나타나지 않았습니다: {exc}", file=sys.stderr)
        return 1
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())
