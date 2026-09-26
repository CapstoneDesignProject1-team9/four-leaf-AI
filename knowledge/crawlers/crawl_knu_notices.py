# 컴퓨터학부 공지사항 크롤링.

import json
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pytesseract
import requests
import urllib3
from bs4 import BeautifulSoup
from PIL import Image

# 운영체제(OS)에 따른 Tesseract 경로 설정
# 2026/09/22  탁 : 해보니까 window에서 tesseract 경로 하드코딩되어 있어서 수정함.
# mac의 경우 homebrew 로 설치하면 자동으로 위치 찾아줌.

if sys.platform == "win32":
    # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # Mac/Linux
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:  # path 존재하면 바로 사용.
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        # PATH에 안 걸릴 경우 M1/M2 Mac Homebrew 기본 경로로 폴백
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# SSL 경고 메시지 숨김
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 현재 파일 위치:
# four-leaf-AI/knowledge/crawlers/crawl_knu_notices.py
#
# parents[0] = crawlers
# parents[1] = knowledge
# parents[2] = four-leaf-AI
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 크롤링 원본 데이터 저장 위치
OUTPUT_DIR = PROJECT_ROOT / "knowledge" / "raw" / "notices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "knu_notices_all.jsonl"


# ============================================================
# 공지사항 전체 크롤링
# ============================================================


def crawl_knu_notices_all_pages() -> None:
    """경북대학교 컴퓨터학부 공지사항을 전체 페이지 순회하며 JSONL로 저장한다."""

    base_url = "https://computer.knu.ac.kr/bbs/board.php"
    bo_table = "sub6_1_a"
    lang = "kor"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    session = requests.Session()
    session.headers.update(headers)

    seen_links: set[str] = set()

    page = 1
    total_count = 0

    print("=" * 60)
    print("경북대학교 컴퓨터학부 공지사항 크롤링 시작")
    print(f"저장 위치: {OUTPUT_FILE}")
    print("=" * 60)

    # "w" 모드이므로 실행할 때마다 기존 결과 파일을 새로 생성한다.
    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        while True:
            target_url = f"{base_url}?bo_table={bo_table}&lang={lang}&page={page}"

            print(f"\n[목록 페이지 {page}] {target_url}")

            try:
                response = session.get(
                    target_url,
                    timeout=15,
                    verify=False,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"[오류] 목록 페이지 요청 실패: {exc}")
                break

            soup = BeautifulSoup(response.text, "html.parser")

            # 사이트 HTML 구조가 조금 바뀌어도 대응할 수 있도록
            # 여러 selector를 함께 사용한다.
            post_elements = soup.select(".subject a, td.subject a, .bo_tit a")

            # 같은 링크가 여러 selector에 중복으로 잡힐 수 있으므로
            # 페이지 안에서도 URL 기준으로 한 번 더 중복 제거한다.
            page_posts: list[tuple[str, str]] = []
            page_seen_links: set[str] = set()

            for element in post_elements:
                href = element.get("href")

                if not href:
                    continue

                link = urljoin(target_url, href)

                if link in page_seen_links:
                    continue

                title = element.get_text(" ", strip=True)

                if not title:
                    continue

                page_seen_links.add(link)
                page_posts.append((title, link))

            # 현재 페이지에서 아직 처리하지 않은 게시글만 남긴다.
            new_posts = [(title, link) for title, link in page_posts if link not in seen_links]

            # 새로운 게시글이 하나도 없으면 마지막 페이지로 판단한다.
            if not new_posts:
                print(f"\n[종료] {page}페이지에서 새로운 게시글을 찾지 못했습니다.")
                break

            print(f"[확인] 새로운 게시글 {len(new_posts)}개 발견")

            for index, (title, link) in enumerate(new_posts, start=1):
                print(f"  [{index}/{len(new_posts)}] 수집 중: {title}")

                seen_links.add(link)

                try:
                    detail_response = session.get(
                        link,
                        timeout=15,
                        verify=False,
                    )
                    detail_response.raise_for_status()
                except requests.RequestException as exc:
                    print(f"    [오류] 상세 페이지 요청 실패: {exc}")
                    continue

                detail_soup = BeautifulSoup(
                    detail_response.text,
                    "html.parser",
                )

                # 게시글 본문
                content_element = detail_soup.select_one("#bo_v_con") or detail_soup.select_one(
                    ".bo_v_atc"
                )

                if content_element is not None:
                    content = content_element.get_text(
                        "\n",
                        strip=True,
                    )
                else:
                    content = ""

                # ========================================================
                # 게시글 본문에 포함된 이미지 OCR
                # ========================================================

                ocr_texts: list[str] = []

                if content_element is not None:
                    image_elements = content_element.select("img")

                    for image_index, image_element in enumerate(
                        image_elements,
                        start=1,
                    ):
                        src = image_element.get("src")

                        if not src:
                            continue

                        image_url = urljoin(link, src)

                        try:
                            image_response = session.get(
                                image_url,
                                timeout=15,
                                verify=False,
                            )
                            image_response.raise_for_status()

                            image = Image.open(BytesIO(image_response.content)).convert("RGB")

                            ocr_text = pytesseract.image_to_string(
                                image,
                                lang="kor+eng",
                            ).strip()

                            if ocr_text:
                                ocr_texts.append(ocr_text)

                        except Exception as exc:
                            # 이미지 다운로드 실패, 깨진 이미지,
                            # OCR 오류 등이 있어도 전체 크롤링은 계속 진행한다.
                            print(f"    [OCR 오류] 이미지 {image_index}: {exc}")

                # ========================================================
                # JSONL 한 줄에 게시글 하나 저장
                # ========================================================

                post_data = {
                    "title": title,
                    "url": link,
                    "page": page,
                    "content": content,
                    "ocr_text": "\n\n".join(ocr_texts),
                }

                output_file.write(
                    json.dumps(
                        post_data,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # 중간에 프로그램이 종료되어도 이미 수집한 결과가
                # 최대한 파일에 남도록 매 게시글마다 flush 한다.
                output_file.flush()

                total_count += 1

                # 서버에 과도한 요청을 보내지 않도록 잠시 대기
                time.sleep(0.5)

            page += 1

            # 페이지 이동 사이에도 잠시 대기
            time.sleep(1)

    print("\n" + "=" * 60)
    print("크롤링 완료")
    print(f"수집 게시글 수: {total_count}")
    print(f"저장 위치: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    crawl_knu_notices_all_pages()
