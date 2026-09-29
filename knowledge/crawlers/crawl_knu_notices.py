"""경북대학교 컴퓨터학부 공지사항 크롤러.

권장 실행
---------
최초 전체 수집 (기존 결과 삭제):
    python crawl_knu_notices.py --full

전체 수집 중 중단 후 이어받기:
    python crawl_knu_notices.py --resume-full

평소 신규/수정 공지 확인:
    python crawl_knu_notices.py

테스트:
    python crawl_knu_notices.py --max-pages 2

기존 JSONL의 빈 PDF 텍스트만 복구:
    python crawl_knu_notices.py --repair-pdfs

파일 위치
---------
<repo>/knowledge/crawlers/crawl_knu_notices.py

저장 위치
---------
<repo>/knowledge/raw/notices/knu_notices_all.jsonl
<repo>/knowledge/raw/notices/attachments/

저장 정책
---------
- 새 게시글은 게시글 1개 처리가 끝날 때마다 즉시 JSONL에 append + flush + fsync 합니다.
- --full은 기존 JSONL/attachments를 먼저 삭제한 뒤 처음부터 수집합니다.
- --resume-full은 기존 진행분을 삭제하지 않고 이미 저장된 post_id는 건너뛰며 계속 수집합니다.
- 일반 실행은 신규 글을 즉시 추가하고 최근 페이지의 수정 공지만 검사합니다.
- 기존 글의 수정 여부는 먼저 상세 HTML의 경량 source_hash로 검사합니다.
  변경이 없으면 이미지 OCR/PDF 다운로드를 다시 하지 않습니다.
- 수정된 게시글의 첨부파일은 다운로드한 SHA-256이 같으면 성공한 추출 결과를 재사용합니다.
  텍스트가 없거나 추출에 실패한 첨부파일은 다시 시도합니다.

필수 패키지
-----------
requests, beautifulsoup4, pillow, pytesseract
PDF 텍스트 추출: pypdf
스캔 PDF OCR fallback: pymupdf
HWPX: 추가 의존성 없음 (표준 라이브러리 ZIP/XML)
HWP 5.x(binary): olefile (선택 의존성)
    python -m pip install olefile

olefile이 없으면 HWP 추출만 건너뜁니다. 암호화/배포용 HWP와 HWP 3.x는
지원하지 않습니다. 기존 저장 공지는 기존 증분 정책에 따라 재처리됩니다.

권장 설치:
    python -m pip install requests beautifulsoup4 pillow pytesseract pypdf pymupdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import olefile
except ImportError:
    olefile = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError:
        fitz = None


# ============================================================
# 설정
# ============================================================

# 이 파일은 <repo>/knowledge/crawlers/crawl_knu_notices.py 위치에 둡니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "knowledge" / "raw" / "notices"
OUTPUT = OUTPUT_DIR / "knu_notices_all.jsonl"
ATTACHMENT_DIR = OUTPUT_DIR / "attachments"
ROOT = PROJECT_ROOT

BASE_URL = "https://computer.knu.ac.kr/bbs/board.php"
BOARD = "sub6_1_a"
LANG = "kor"
DEFAULT_MIN_DATE = datetime(2021, 1, 1)
CURRENT_MIN_DATE = DEFAULT_MIN_DATE

OCR_LANG = "kor+eng"
OCR_CONFIG = "--psm 6"
OCR_UPSCALE_WIDTH = 1200
OCR_SCALE = 2
PDF_OCR_RENDER_SCALE = 2.0
PDF_TEXT_MIN_CHARS = 40

REQUEST_TIMEOUT = 20
PAGE_DELAY = 0.35
ITEM_DELAY = 0.15
RECENT_PAGES_TO_RECHECK = 2
CONSECUTIVE_KNOWN_PAGES_TO_STOP = 2

VERIFY_SSL = os.getenv("KNU_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
DOWNLOAD_ATTACHMENTS = os.getenv("KNU_DOWNLOAD_ATTACHMENTS", "true").lower() not in {
    "0",
    "false",
    "no",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

DATE_RE = re.compile(r"(20\d{2})[-./년\s]+(\d{1,2})[-./월\s]+(\d{1,2})")
DATE_COMPACT_RE = re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b")

ATTACHMENT_SELECTORS = (
    "#bo_v_file a[href]",
    ".bo_v_file a[href]",
    "#bo_v_file li a[href]",
    "a[href*='download.php']",
    "a[href*='file_download.php']",
)

IMAGE_SELECTORS = (
    ("#bo_v_con img, .bo_v_con img, .bo_v_atc img", "src"),
    ("#bo_v_img img, .bo_v_img img", "src"),
    ('a[href*="view_image.php"]', "href"),
)

HWP_LONG_CONTROLS = frozenset(range(1, 10)) | {11, 12} | frozenset(range(14, 24))
HWP_CONTROL_TEXT = {
    9: b"\t\x00",
    10: b"\n\x00",
    13: b"\n\x00",
    24: b"-\x00",
    30: b" \x00",
    31: b" \x00",
}


# ============================================================
# 자료형
# ============================================================


@dataclass(slots=True)
class ListPost:
    title: str
    url: str
    post_id: str
    date: datetime | None


@dataclass(slots=True)
class PostSnapshot:
    """상세 HTML만 읽어서 만든 가벼운 게시글 상태.

    이 단계에서는 이미지 OCR/PDF 다운로드를 하지 않습니다.
    기존 글의 source_hash가 같으면 여기서 처리를 끝내므로 반복 실행 비용이 크게 줄어듭니다.
    """

    post_id: str
    title: str
    url: str
    normalized_url: str
    page: int
    date: str
    content: str
    image_urls: list[str]
    attachment_links: list[dict]
    source_hash: str


Mode = Literal["full", "resume", "update"]


# ============================================================
# CLI / 환경
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="경북대학교 컴퓨터학부 공지사항 수집기")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help="기존 JSONL/첨부파일을 삭제하고 기준일 이후 전체 재수집",
    )
    mode.add_argument(
        "--resume-full",
        action="store_true",
        help="중단된 전체 수집 이어받기. 기존 데이터는 보존하고 없는 post_id만 수집",
    )
    mode.add_argument(
        "--repair-pdfs",
        action="store_true",
        help="기존 JSONL에서 text가 비어 있는 로컬 PDF만 다시 텍스트 추출",
    )
    parser.add_argument(
        "--min-date",
        default=DEFAULT_MIN_DATE.strftime("%Y-%m-%d"),
        help="수집 기준일 (YYYY-MM-DD), 기본값: 2021-01-01",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="테스트용 최대 페이지 수. 0이면 제한 없음",
    )
    return parser.parse_args()


def configure_tesseract() -> bool:
    env_path = os.getenv("TESSERACT_CMD")
    auto_path = shutil.which("tesseract")
    windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    for candidate in (
        env_path,
        auto_path,
        str(windows_default) if windows_default.exists() else None,
    ):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True

    print("[주의] Tesseract를 찾지 못했습니다. 이미지/PDF OCR은 건너뜁니다.")
    return False


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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    response = session.get(
        url,
        timeout=kwargs.pop("timeout", REQUEST_TIMEOUT),
        verify=VERIFY_SSL,
        **kwargs,
    )
    response.raise_for_status()
    return response


# ============================================================
# URL / 날짜 / 해시
# ============================================================


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query.pop("page", None)
    normalized_query = urlencode(
        sorted((key, value) for key, values in query.items() for value in values)
    )
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, normalized_query, "")
    )


def normalize_attachment_url(url: str) -> str:
    """nonce/page처럼 실행 때마다 달라질 수 있는 값은 source_hash에서 제외한다."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    for key in ("nonce", "page"):
        query.pop(key, None)
    normalized_query = urlencode(
        sorted((key, value) for key, values in query.items() for value in values)
    )
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, normalized_query, "")
    )


def extract_post_id(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    for key in ("wr_id", "no", "idx"):
        value = query.get(key)
        if value and value[0]:
            return value[0]
    return hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:16]


def parse_date(text: str | None) -> datetime | None:
    text = text or ""
    for regex in (DATE_RE, DATE_COMPACT_RE):
        match = regex.search(text)
        if not match:
            continue
        try:
            return datetime(*(int(value) for value in match.groups()))
        except ValueError:
            continue
    return None


def _stable_attachment_ref(attachment: dict) -> dict:
    return {
        "name": str(attachment.get("source_name") or attachment.get("name") or ""),
        "url": normalize_attachment_url(str(attachment.get("url") or "")),
    }


def calculate_source_hash(
    *,
    title: str,
    date: str,
    content: str,
    image_urls: list[str],
    attachments: list[dict],
) -> str:
    """무거운 OCR/PDF 처리 전에 HTML 수준 변경 여부를 판단하는 해시."""
    payload = {
        "title": title,
        "date": date,
        "content": content,
        "image_urls": sorted(normalize_url(url) for url in image_urls),
        "attachments": sorted(
            (_stable_attachment_ref(a) for a in attachments),
            key=lambda x: (x["url"], x["name"]),
        ),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_hash_from_item(item: dict) -> str:
    return calculate_source_hash(
        title=str(item.get("title", "")),
        date=str(item.get("date", "")),
        content=str(item.get("content", "")),
        image_urls=list(item.get("image_urls", [])),
        attachments=list(item.get("attachments", [])),
    )


def calculate_content_hash(item: dict) -> str:
    """실제 RAG에 영향을 주는 의미 있는 내용만 사용해 안정적인 해시를 만든다."""
    attachments = []
    for attachment in item.get("attachments", []):
        attachments.append(
            {
                "name": attachment.get("name", ""),
                "url": normalize_attachment_url(str(attachment.get("url", ""))),
                "text": attachment.get("text", ""),
                "size_bytes": attachment.get("size_bytes", 0),
                "sha256": attachment.get("sha256", ""),
            }
        )

    payload = {
        "title": item.get("title", ""),
        "date": item.get("date", ""),
        "content": item.get("content", ""),
        "ocr_text": item.get("ocr_text", ""),
        "image_urls": sorted(normalize_url(url) for url in item.get("image_urls", [])),
        "attachments": sorted(attachments, key=lambda x: (x["url"], x["name"])),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ============================================================
# JSONL 저장
# ============================================================


def load_existing() -> tuple[list[dict], dict[str, dict]]:
    if not OUTPUT.exists():
        return [], {}

    items: list[dict] = []
    by_id: dict[str, dict] = {}

    with OUTPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[주의] JSONL {line_number}번째 줄이 깨져 있어 건너뜁니다.")
                continue

            url = str(item.get("url", ""))
            post_id = str(item.get("post_id") or extract_post_id(url))
            item["post_id"] = post_id
            if "normalized_url" not in item:
                item["normalized_url"] = normalize_url(url) if url else ""
            item.setdefault("attachments", [])
            item.setdefault("image_urls", [])
            # setdefault의 기본값은 키가 있어도 평가되므로 해시를 명시적으로 지연 계산한다.
            if "source_hash" not in item:
                item["source_hash"] = source_hash_from_item(item)
            if "content_hash" not in item:
                item["content_hash"] = calculate_content_hash(item)
            items.append(item)
            by_id[post_id] = item

    return items, by_id


def reset_full_output() -> None:
    print(f"[FULL 초기화] JSONL: {OUTPUT}")
    print(f"[FULL 초기화] 첨부파일: {ATTACHMENT_DIR}")

    for path in (
        OUTPUT,
        OUTPUT.with_suffix(OUTPUT.suffix + ".tmp"),
        OUTPUT.with_suffix(OUTPUT.suffix + ".bak"),
    ):
        if path.exists():
            path.unlink()
            print(f"[삭제 완료] {path}")

    if ATTACHMENT_DIR.exists():
        shutil.rmtree(ATTACHMENT_DIR)
        print(f"[삭제 완료] {ATTACHMENT_DIR}")

    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.touch()
    print(f"[생성 완료] {OUTPUT}")


def append_jsonl_item(item: dict, output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())


def atomic_write_jsonl(items: Iterable[dict], output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, output)


def replace_item_and_save(items: list[dict], new_item: dict) -> None:
    for index, saved in enumerate(items):
        if saved.get("post_id") == new_item.get("post_id"):
            items[index] = new_item
            break
    else:
        items.append(new_item)
    atomic_write_jsonl(items)


# ============================================================
# HTML 파싱
# ============================================================


def list_date(anchor) -> datetime | None:
    row = anchor.find_parent("tr")
    if row is None:
        return None

    for selector in (
        ".td_datetime",
        ".date",
        ".datetime",
        "td:nth-last-child(2)",
        "td:last-child",
    ):
        element = row.select_one(selector)
        if element:
            date = parse_date(element.get_text(" ", strip=True))
            if date:
                return date
    return parse_date(row.get_text(" ", strip=True))


def detail_date(soup: BeautifulSoup) -> datetime | None:
    for selector in (
        "#bo_v_info",
        ".bo_v_info",
        ".if_date",
        ".date",
        ".datetime",
        ".td_datetime",
    ):
        for element in soup.select(selector):
            date = parse_date(element.get_text(" ", strip=True))
            if date:
                return date
    return None


def extract_content(soup: BeautifulSoup) -> str:
    element = (
        soup.select_one("#bo_v_con") or soup.select_one(".bo_v_con") or soup.select_one(".bo_v_atc")
    )
    if element is None:
        return ""
    for trash in element.select("script, style, noscript"):
        trash.decompose()
    return element.get_text("\n", strip=True)


def collect_image_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for selector, attr in IMAGE_SELECTORS:
        for element in soup.select(selector):
            value = element.get(attr)
            if not value:
                continue
            url = urljoin(page_url, value)
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def collect_attachment_links(soup: BeautifulSoup, page_url: str) -> list[dict]:
    attachments: list[dict] = []
    seen: set[str] = set()

    for selector in ATTACHMENT_SELECTORS:
        for anchor in soup.select(selector):
            href = anchor.get("href")
            if not href:
                continue
            url = urljoin(page_url, href)
            canonical = normalize_attachment_url(url)
            if canonical in seen:
                continue
            seen.add(canonical)
            name = anchor.get_text(" ", strip=True) or Path(urlparse(url).path).name or "attachment"
            attachments.append({"name": name, "url": url})

    return attachments


def parse_list_page(soup: BeautifulSoup, page_url: str) -> list[ListPost]:
    posts: list[ListPost] = []
    seen_ids: set[str] = set()

    for anchor in soup.select(".subject a, td.subject a, .bo_tit a"):
        href = anchor.get("href")
        title = anchor.get_text(" ", strip=True)
        if not href or not title:
            continue

        url = urljoin(page_url, href)
        post_id = extract_post_id(url)
        if post_id in seen_ids:
            continue
        seen_ids.add(post_id)
        posts.append(ListPost(title, url, post_id, list_date(anchor)))

    return posts


def fetch_snapshot(
    session: requests.Session,
    list_post: ListPost,
    page: int,
    known_date: datetime | None = None,
) -> PostSnapshot | None:
    """상세 HTML만 수집한다. OCR/PDF 처리는 하지 않는다."""
    try:
        response = get(session, list_post.url)
    except requests.RequestException as exc:
        print(f"  [상세 요청 실패] {list_post.title}: {exc}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    post_date = list_post.date or known_date or detail_date(soup)
    if post_date and post_date < CURRENT_MIN_DATE:
        return None

    date_text = post_date.strftime("%Y-%m-%d") if post_date else ""
    content = extract_content(soup)
    image_urls = collect_image_urls(soup, list_post.url)
    attachment_links = collect_attachment_links(soup, list_post.url)
    source_hash = calculate_source_hash(
        title=list_post.title,
        date=date_text,
        content=content,
        image_urls=image_urls,
        attachments=attachment_links,
    )

    return PostSnapshot(
        post_id=list_post.post_id,
        title=list_post.title,
        url=list_post.url,
        normalized_url=normalize_url(list_post.url),
        page=page,
        date=date_text,
        content=content,
        image_urls=image_urls,
        attachment_links=attachment_links,
        source_hash=source_hash,
    )


# ============================================================
# 이미지 OCR
# ============================================================


def tesseract_available() -> bool:
    command = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    return Path(command).is_file() or shutil.which(command) is not None


def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image.convert("L"))
    width, height = image.size
    if 0 < width < OCR_UPSCALE_WIDTH and height > 0:
        image = image.resize((width * OCR_SCALE, height * OCR_SCALE))
    return image


def fetch_image_bytes(session: requests.Session, url: str) -> bytes | None:
    response = get(session, url)
    content_type = response.headers.get("Content-Type", "").lower()
    if "image" in content_type:
        return response.content
    if "html" not in content_type and content_type:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    image = soup.find("img")
    if image is None or not image.get("src"):
        return None

    real_response = get(session, urljoin(url, image["src"]))
    if "image" not in real_response.headers.get("Content-Type", "").lower():
        return None
    return real_response.content


def run_ocr(session: requests.Session, url: str, ocr_enabled: bool) -> str:
    if not ocr_enabled:
        return ""
    try:
        image_bytes = fetch_image_bytes(session, url)
        if not image_bytes:
            return ""
        with Image.open(BytesIO(image_bytes)) as image:
            processed = preprocess_image(image)
            return pytesseract.image_to_string(
                processed,
                lang=OCR_LANG,
                config=OCR_CONFIG,
            ).strip()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"    [OCR 오류] {exc} | {url}")
        return ""


# ============================================================
# PDF / 첨부파일
# ============================================================


def safe_filename(name: str, fallback: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip(" .")
    return name or fallback


def guess_extension(name: str, content_type: str, url: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix:
        return suffix
    url_suffix = Path(urlparse(url).path).suffix.lower()
    if url_suffix:
        return url_suffix

    mapping = {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/zip": ".zip",
        "application/x-hwpx": ".hwpx",
        "application/vnd.hancom.hwpx": ".hwpx",
        "application/x-hwp": ".hwp",
        "application/haansofthwp": ".hwp",
    }
    for key, value in mapping.items():
        if key in content_type:
            return value
    return ""


def has_pdf_signature(data: bytes) -> bool:
    return b"%PDF" in data[:1024]


def describe_binary_signature(data: bytes) -> str:
    head = data[:64].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return "html"
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    if data.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return "ole"
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if has_pdf_signature(data):
        return "pdf"
    return "unknown"


def extract_pdf_text_with_pypdf(data: bytes) -> str:
    if PdfReader is None:
        print("    [PDF] pypdf 미설치 - 텍스트 레이어 추출 건너뜀")
        return ""

    try:
        reader = PdfReader(BytesIO(data))
        texts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:
                print(f"    [PDF {page_number}페이지 추출 오류] {exc}")
                continue
            if text:
                texts.append(text)
        return "\n\n".join(texts).strip()
    except Exception as exc:
        print(f"    [PDF 열기 오류] {exc}")
        return ""


def extract_pdf_text_with_ocr(data: bytes) -> str:
    if fitz is None:
        print("    [PDF OCR 불가] PyMuPDF 미설치: python -m pip install pymupdf")
        return ""
    if not tesseract_available():
        print("    [PDF OCR 불가] Tesseract 실행 파일을 찾지 못했습니다.")
        return ""

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        print(f"    [PDF OCR 열기 오류] {exc}")
        return ""

    texts: list[str] = []
    try:
        total_pages = len(document)
        for page_index in range(total_pages):
            try:
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(PDF_OCR_RENDER_SCALE, PDF_OCR_RENDER_SCALE),
                    alpha=False,
                )
                with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
                    processed = preprocess_image(image)
                    text = pytesseract.image_to_string(
                        processed,
                        lang=OCR_LANG,
                        config=OCR_CONFIG,
                    ).strip()
                if text:
                    texts.append(text)
                print(f"    [PDF OCR] {page_index + 1}/{total_pages}페이지 ({len(text)}자)")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"    [PDF OCR {page_index + 1}페이지 오류] {exc}")
    finally:
        document.close()

    return "\n\n".join(texts).strip()


def extract_pdf_text(data: bytes) -> tuple[str, str]:
    if not has_pdf_signature(data):
        detected = describe_binary_signature(data)
        print(f"    [PDF 경고] 실제 PDF 시그니처 없음 (감지={detected})")
        return "", f"invalid_pdf:{detected}"

    text = extract_pdf_text_with_pypdf(data)
    if len(text.strip()) >= PDF_TEXT_MIN_CHARS:
        print(f"    [PDF 텍스트] pypdf / {len(text)}자")
        return text, "pypdf"

    if text:
        print(f"    [PDF] 추출 텍스트가 짧아 OCR fallback 시도 ({len(text)}자)")
    else:
        print("    [PDF] 텍스트 레이어 없음 - OCR fallback 시도")

    ocr_text = extract_pdf_text_with_ocr(data)
    if len(ocr_text) > len(text):
        print(f"    [PDF OCR 성공] {len(ocr_text)}자")
        return ocr_text, "ocr"
    if text:
        return text, "pypdf"
    return "", "none"


def extract_hwpx_text(data: bytes) -> tuple[str, str]:
    """본문 XML을 숫자 순서로 읽고 표 안의 문단도 한 번씩 추출한다."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            sections = []
            for name in archive.namelist():
                match = re.fullmatch(r"Contents/section(\d+)\.xml", name, re.IGNORECASE)
                if match:
                    sections.append((int(match.group(1)), name))
            if not sections:
                return "", "invalid_hwpx:no_sections"
            parts: list[str] = []

            def visit(element: ET.Element, in_text: bool = False) -> None:
                tag = element.tag.rsplit("}", 1)[-1]
                in_text = in_text or tag == "t"
                if tag == "tab":
                    parts.append("\t")
                elif tag in {"lineBreak", "br"}:
                    parts.append("\n")
                if in_text and element.text:
                    parts.append(element.text)
                for child in element:
                    visit(child, in_text)
                    if in_text and child.tail:
                        parts.append(child.tail)
                if tag == "p":
                    parts.append("\n")

            for _, name in sorted(sections):
                visit(ET.fromstring(archive.read(name)))
            text = "\n".join(line.rstrip() for line in "".join(parts).splitlines()).strip()
        return text, "hwpx_xml" if text else "none"
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"    [HWPX 추출 오류] {exc}")
        return "", "hwpx_error"


def extract_hwp_section_text(data: bytes) -> str:
    """PARA_TEXT(67)를 읽고 8-WCHAR 제어 정보를 건너뛴다."""
    paragraphs: list[str] = []
    offset = 0
    view = memoryview(data)
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("잘린 HWP 레코드 헤더")
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        size = header >> 20
        if size == 0xFFF:
            if offset + 4 > len(data):
                raise ValueError("잘린 HWP 확장 길이")
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        end = offset + size
        if end > len(data):
            raise ValueError("잘린 HWP 레코드 본문")
        if header & 0x3FF == 67:
            payload = view[offset:end]
            if len(payload) % 2:
                raise ValueError("잘못된 HWP UTF-16 길이")
            clean = bytearray()
            index = 0
            run_start = 0
            while index < len(payload):
                code = payload[index] | (payload[index + 1] << 8)
                if code < 32:
                    width = 16 if code in HWP_LONG_CONTROLS else 2
                    if index + width > len(payload):
                        raise ValueError("잘린 HWP 제어 문자")
                    # 일반 문자는 연속 구간 단위로 복사해 문자별 임시 bytes 생성을 줄인다.
                    clean.extend(payload[run_start:index])
                    clean.extend(HWP_CONTROL_TEXT.get(code, b""))
                    index += width
                    run_start = index
                else:
                    index += 2
            clean.extend(payload[run_start:])
            text = clean.decode("utf-16le", errors="replace").strip()
            if text:
                paragraphs.append(text)
        offset = end
    return "\n".join(paragraphs)


def extract_hwp_text(data: bytes) -> tuple[str, str]:
    """FileHeader의 압축 플래그에 따라 HWP 5.x BodyText를 읽는다."""
    if olefile is None:
        print("    [HWP] olefile 미설치: python -m pip install olefile")
        return "", "olefile_missing"
    try:
        with olefile.OleFileIO(BytesIO(data)) as document:
            header = document.openstream("FileHeader").read()
            if len(header) < 40 or not header[:32].startswith(b"HWP Document File\x00"):
                return "", "invalid_hwp:header"
            if header[35] != 5:
                return "", "unsupported_hwp:version"
            flags = int.from_bytes(header[36:40], "little")
            if flags & 2:
                return "", "unsupported_hwp:encrypted"
            if flags & 4:
                return "", "unsupported_hwp:distribution"
            sections = []
            for path in document.listdir(streams=True, storages=False):
                if len(path) == 2 and path[0] == "BodyText":
                    match = re.fullmatch(r"Section(\d+)", path[1])
                    if match:
                        sections.append((int(match.group(1)), path))
            if not sections:
                return "", "invalid_hwp:no_sections"
            texts: list[str] = []
            for _, path in sorted(sections):
                section = document.openstream(path).read()
                if flags & 1:
                    section = zlib.decompress(section, -15)
                text = extract_hwp_section_text(section)
                if text:
                    texts.append(text)
        text = "\n\n".join(texts).strip()
        return text, "hwp_bodytext" if text else "none"
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"    [HWP 추출 오류] {exc}")
        return "", "hwp_error"


def get_attachment_response(
    session: requests.Session,
    url: str,
    referer_url: str,
) -> requests.Response:
    return get(
        session,
        url,
        headers={
            "Referer": referer_url,
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
        },
        allow_redirects=True,
    )


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def content_disposition_filename(disposition: str) -> str:
    match = re.search(
        r"filename\*?=(?:UTF-8''|\")?([^\";]+)",
        disposition,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return requests.utils.unquote(match.group(1)).strip('"')


def file_sha256(path: Path) -> str:
    """기존 첨부파일을 통째로 메모리에 올리지 않고 비교한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attachment_text_kind(suffix: str, content_type: str, is_pdf: bool) -> str:
    if is_pdf:
        return "pdf"
    if suffix in {".hwp", ".hwpx"}:
        return suffix[1:]
    if suffix in {".txt", ".csv", ".md"} or content_type.startswith("text/"):
        return "text"
    return ""


def process_attachments(
    session: requests.Session,
    attachments: list[dict],
    post_id: str,
    referer_url: str,
    previous_attachments: Iterable[dict] = (),
) -> list[dict]:
    processed: list[dict] = []
    method_kinds = {
        "pypdf": "pdf",
        "ocr": "pdf",
        "hwpx_xml": "hwpx",
        "hwp_bodytext": "hwp",
        "text_decode": "text",
    }
    cached_text: dict[tuple[str, str], tuple[str, str]] = {}
    for previous in previous_attachments:
        method = previous.get("text_extraction", "")
        text = previous.get("text", "")
        digest = previous.get("sha256", "")
        if digest and isinstance(text, str) and text.strip() and method in method_kinds:
            cached_text[(digest, method_kinds[method])] = (text, method)

    for index, attachment in enumerate(attachments, start=1):
        source_name = str(attachment.get("name", ""))
        result = {
            "name": source_name,
            "source_name": source_name,
            "url": attachment["url"],
            "content_type": "",
            "local_path": "",
            "text": "",
            "text_extraction": "",
            "size_bytes": 0,
            "sha256": "",
        }

        try:
            response = get_attachment_response(session, attachment["url"], referer_url)
            data = response.content
            content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
            disposition = response.headers.get("Content-Disposition", "")

            result["content_type"] = content_type
            result["size_bytes"] = len(data)
            result["sha256"] = hashlib.sha256(data).hexdigest()

            server_name = content_disposition_filename(disposition)
            filename = server_name or source_name
            extension = guess_extension(filename, content_type, attachment["url"])
            filename = safe_filename(filename, f"attachment_{index}{extension}")
            if not Path(filename).suffix and extension:
                filename += extension
            result["name"] = filename

            suffix = Path(filename).suffix.lower()
            is_pdf = has_pdf_signature(data)
            kind = attachment_text_kind(suffix, content_type, is_pdf)
            looks_like_pdf = suffix == ".pdf" or "pdf" in content_type or is_pdf
            invalid_pdf = looks_like_pdf and not is_pdf

            if invalid_pdf:
                detected = describe_binary_signature(data)
                result["text_extraction"] = f"invalid_pdf:{detected}"
                print(
                    f"    [첨부 PDF 실패] {filename} - 실제 응답은 {detected} "
                    f"(Content-Type={content_type or '없음'}, HTTP={response.status_code})"
                )
                # HTML 오류 페이지를 .pdf 파일로 저장해 정상 파일처럼 보이게 하지 않는다.
                if detected == "html":
                    processed.append(result)
                    continue

            if DOWNLOAD_ATTACHMENTS:
                post_dir = ATTACHMENT_DIR / post_id
                post_dir.mkdir(parents=True, exist_ok=True)
                local_path = post_dir / filename

                # 같은 바이트가 이미 있으면 불필요한 디스크 write를 피한다.
                should_write = True
                if local_path.exists() and local_path.stat().st_size == len(data):
                    try:
                        should_write = file_sha256(local_path) != result["sha256"]
                    except OSError:
                        should_write = True
                if should_write:
                    local_path.write_bytes(data)

                try:
                    result["local_path"] = str(local_path.relative_to(ROOT))
                except ValueError:
                    result["local_path"] = str(local_path)

            cached = cached_text.get((result["sha256"], kind))
            if cached is not None and not invalid_pdf:
                result["text"], result["text_extraction"] = cached
            elif is_pdf:
                result["text"], result["text_extraction"] = extract_pdf_text(data)
            elif suffix == ".hwpx":
                result["text"], result["text_extraction"] = extract_hwpx_text(data)
            elif suffix == ".hwp":
                result["text"], result["text_extraction"] = extract_hwp_text(data)
            elif suffix in {".txt", ".csv", ".md"} or content_type.startswith("text/"):
                result["text"] = decode_text_bytes(data)
                result["text_extraction"] = "text_decode" if result["text"] else "none"

            if (
                result["text"].strip()
                and result["text_extraction"] in method_kinds
                and not invalid_pdf
            ):
                cached_text[(result["sha256"], kind)] = (result["text"], result["text_extraction"])

        except KeyboardInterrupt:
            raise
        except requests.RequestException as exc:
            print(f"    [첨부파일 요청 실패] {source_name}: {exc}")
            result["text_extraction"] = "request_failed"
        except OSError as exc:
            print(f"    [첨부파일 저장 실패] {source_name}: {exc}")
            result["text_extraction"] = "save_failed"

        processed.append(result)
        time.sleep(ITEM_DELAY)

    return processed


# ============================================================
# 게시글 완성
# ============================================================


def materialize_item(
    session: requests.Session,
    snapshot: PostSnapshot,
    ocr_enabled: bool,
    previous_item: dict | None = None,
) -> dict:
    """변경/신규 게시글에 대해서만 OCR과 첨부파일 다운로드를 수행한다."""
    ocr_texts: list[str] = []
    for image_url in snapshot.image_urls if ocr_enabled else ():
        text = run_ocr(session, image_url, ocr_enabled)
        if text:
            ocr_texts.append(text)
        time.sleep(ITEM_DELAY)

    attachments = process_attachments(
        session,
        snapshot.attachment_links,
        snapshot.post_id,
        snapshot.url,
        previous_attachments=(previous_item or {}).get("attachments", []),
    )

    item = {
        "post_id": snapshot.post_id,
        "title": snapshot.title,
        "url": snapshot.url,
        "normalized_url": snapshot.normalized_url,
        "page": snapshot.page,
        "date": snapshot.date,
        "content": snapshot.content,
        "ocr_text": "\n\n".join(ocr_texts).strip(),
        "image_urls": snapshot.image_urls,
        "attachments": attachments,
        "source_hash": snapshot.source_hash,
        "crawled_at": datetime.now().isoformat(timespec="seconds"),
    }
    item["content_hash"] = calculate_content_hash(item)
    return item


# ============================================================
# PDF 복구
# ============================================================


def is_pdf_attachment(attachment: dict) -> bool:
    name = str(attachment.get("name", ""))
    local_path = str(attachment.get("local_path", ""))
    content_type = str(attachment.get("content_type", "")).lower()
    return (
        Path(name).suffix.lower() == ".pdf"
        or Path(local_path).suffix.lower() == ".pdf"
        or "pdf" in content_type
    )


def repair_existing_pdf_text() -> None:
    if not OUTPUT.exists():
        print(f"[PDF 복구] JSONL이 없습니다: {OUTPUT}")
        return

    items, _ = load_existing()
    targets = [
        (item, attachment)
        for item in items
        for attachment in item.get("attachments", [])
        if is_pdf_attachment(attachment) and not str(attachment.get("text", "")).strip()
    ]

    print("=" * 72)
    print("PDF 텍스트 복구 모드")
    print(f"대상 PDF: {len(targets)}개")
    print("=" * 72)

    repaired = 0
    failed = 0

    try:
        for item in items:
            changed = False
            for attachment in item.get("attachments", []):
                if not is_pdf_attachment(attachment) or str(attachment.get("text", "")).strip():
                    continue

                local_value = str(attachment.get("local_path", "")).strip()
                if not local_value:
                    failed += 1
                    continue

                local_path = Path(local_value)
                if not local_path.is_absolute():
                    local_path = ROOT / local_path
                if not local_path.exists():
                    print(f"  [PDF 복구 실패] 파일 없음: {local_path}")
                    failed += 1
                    continue

                try:
                    data = local_path.read_bytes()
                except OSError as exc:
                    print(f"  [PDF 읽기 실패] {exc}")
                    failed += 1
                    continue

                print(f"  [PDF 복구] {attachment.get('name', local_path.name)}")
                text, method = extract_pdf_text(data)
                attachment["text"] = text
                attachment["text_extraction"] = method
                if text:
                    repaired += 1
                    changed = True
                    print(f"    [성공] {method} / {len(text)}자")
                else:
                    failed += 1

            if changed:
                item["content_hash"] = calculate_content_hash(item)
                item["crawled_at"] = datetime.now().isoformat(timespec="seconds")
                atomic_write_jsonl(items)
                print(f"  [JSONL 즉시 반영] {item.get('title', '')}")

    except KeyboardInterrupt:
        print("\n[중단] 이미 복구 완료된 게시글은 JSONL에 반영되어 있습니다.")

    print(f"[PDF 복구 종료] 성공 {repaired} / 실패·텍스트없음 {failed}")


# ============================================================
# 크롤링
# ============================================================


def mode_name(mode: Mode) -> str:
    return {
        "full": "전체 재수집",
        "resume": "전체 수집 이어받기",
        "update": "신규/수정 공지 확인",
    }[mode]


def parse_saved_date(item: dict | None) -> datetime | None:
    if not item or not item.get("date"):
        return None
    try:
        return datetime.strptime(str(item["date"]), "%Y-%m-%d")
    except ValueError:
        return None


def crawl(mode: Mode = "update", max_pages: int = 0) -> None:
    session = make_session()
    ocr_enabled = configure_tesseract()

    if mode == "full":
        reset_full_output()
        existing_items: list[dict] = []
        existing_by_id: dict[str, dict] = {}
    else:
        existing_items, existing_by_id = load_existing()

    stats = {
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "old": 0,
        "empty": 0,
        "failed": 0,
        "lightweight_skip": 0,
    }

    print("=" * 72)
    print(f"모드: {mode_name(mode)}")
    print(f"기준일: {CURRENT_MIN_DATE:%Y-%m-%d}")
    print(f"저장 위치: {OUTPUT}")
    print("저장: 게시글 단위 즉시 저장 (flush + fsync)")
    print(f"기존 데이터: {len(existing_by_id)}개")
    print("=" * 72)

    page = 1
    seen_post_ids: set[str] = set()
    consecutive_known_pages = 0

    try:
        while True:
            if max_pages and page > max_pages:
                print(f"[종료] --max-pages={max_pages} 도달")
                break

            page_url = f"{BASE_URL}?bo_table={BOARD}&lang={LANG}&page={page}"
            try:
                response = get(session, page_url)
            except requests.RequestException as exc:
                print(f"[목록 요청 실패] {page_url}: {exc}")
                stats["failed"] += 1
                break

            soup = BeautifulSoup(response.text, "html.parser")
            posts = [
                post
                for post in parse_list_page(soup, page_url)
                if post.post_id not in seen_post_ids
            ]
            if not posts:
                print(f"[종료] {page}페이지에서 게시글을 찾지 못했습니다.")
                break

            print(f"\n[페이지 {page}] {len(posts)}개")
            page_dates: list[datetime] = []
            unknown_dates = 0
            page_new_count = 0

            for post in posts:
                seen_post_ids.add(post.post_id)
                old_item = existing_by_id.get(post.post_id)
                old_date = parse_saved_date(old_item)
                effective_date = post.date or old_date

                if effective_date:
                    page_dates.append(effective_date)
                    if effective_date < CURRENT_MIN_DATE:
                        stats["old"] += 1
                        continue
                else:
                    unknown_dates += 1

                # 이어받기 모드에서는 이미 저장된 글은 상세/OCR/PDF를 다시 받지 않는다.
                if mode == "resume" and old_item is not None:
                    stats["unchanged"] += 1
                    continue

                # 일반 업데이트는 기존 글을 최신 N페이지에서만 재검사한다.
                if mode == "update" and old_item is not None and page > RECENT_PAGES_TO_RECHECK:
                    stats["unchanged"] += 1
                    continue

                snapshot = fetch_snapshot(session, post, page, known_date=old_date)
                if snapshot is None:
                    if effective_date and effective_date < CURRENT_MIN_DATE:
                        stats["old"] += 1
                    else:
                        stats["failed"] += 1
                    continue

                if snapshot.date:
                    try:
                        actual_date = datetime.strptime(snapshot.date, "%Y-%m-%d")
                        page_dates.append(actual_date)
                        if effective_date is None:
                            unknown_dates = max(0, unknown_dates - 1)
                    except ValueError:
                        pass

                # 핵심 최적화: 기존 글의 HTML 수준 source_hash가 같으면 OCR/PDF 처리 생략.
                if mode == "update" and old_item is not None:
                    old_source_hash = str(
                        old_item.get("source_hash") or source_hash_from_item(old_item)
                    )
                    if snapshot.source_hash == old_source_hash:
                        stats["unchanged"] += 1
                        stats["lightweight_skip"] += 1
                        continue

                item = materialize_item(session, snapshot, ocr_enabled, previous_item=old_item)

                if not item["content"].strip() and not item["ocr_text"].strip():
                    if not any(str(a.get("text", "")).strip() for a in item["attachments"]):
                        stats["empty"] += 1
                        print(f"  [텍스트 없음] {item['title']}")

                if old_item is None:
                    append_jsonl_item(item)
                    existing_items.append(item)
                    existing_by_id[item["post_id"]] = item
                    stats["new"] += 1
                    page_new_count += 1
                    print(
                        f"  [즉시 저장] {item['title']} "
                        f"(본문 {len(item['content'])} / OCR {len(item['ocr_text'])} / "
                        f"첨부 {len(item['attachments'])})"
                    )
                elif item["content_hash"] != old_item.get("content_hash"):
                    replace_item_and_save(existing_items, item)
                    existing_by_id[item["post_id"]] = item
                    stats["updated"] += 1
                    print(f"  [수정 즉시 반영] {item['title']}")
                else:
                    # source_hash만 달라졌지만 실제 추출 결과가 같은 경우에도
                    # 새 source_hash를 저장해 다음 실행에서 무거운 처리를 반복하지 않는다.
                    replace_item_and_save(existing_items, item)
                    existing_by_id[item["post_id"]] = item
                    stats["unchanged"] += 1

                time.sleep(ITEM_DELAY)

            # 전체/이어받기 모드는 기준일까지 계속 내려간다.
            if mode in {"full", "resume"}:
                if page_dates and unknown_dates == 0 and max(page_dates) < CURRENT_MIN_DATE:
                    print("[종료] 기준일 이전 게시글 구간에 도달했습니다.")
                    break
            else:
                if page_new_count == 0:
                    consecutive_known_pages += 1
                else:
                    consecutive_known_pages = 0

                if (
                    page >= RECENT_PAGES_TO_RECHECK
                    and consecutive_known_pages >= CONSECUTIVE_KNOWN_PAGES_TO_STOP
                ):
                    print(
                        f"[종료] 신규 공지가 없는 페이지가 "
                        f"{CONSECUTIVE_KNOWN_PAGES_TO_STOP}개 연속 확인되었습니다."
                    )
                    break

                if page_dates and unknown_dates == 0 and max(page_dates) < CURRENT_MIN_DATE:
                    print("[종료] 기준일 이전 게시글 구간에 도달했습니다.")
                    break

            page += 1
            time.sleep(PAGE_DELAY)

    except KeyboardInterrupt:
        print("\n[중단] 사용자가 크롤링을 중단했습니다.")
        print("[중단] 이미 [즉시 저장]된 게시글은 JSONL에 그대로 남아 있습니다.")
        if mode == "full":
            print("[안내] 다음에는 --resume-full로 실행하면 기존 진행분을 유지하고 이어받습니다.")
    finally:
        session.close()

    print("\n" + "=" * 72)
    print("크롤링 종료")
    print(f"신규 저장: {stats['new']}개")
    print(f"수정 반영: {stats['updated']}개")
    print(f"변경 없음/건너뜀: {stats['unchanged']}개")
    print(f"  └ 경량 비교로 OCR/PDF 생략: {stats['lightweight_skip']}개")
    print(f"기준일 이전: {stats['old']}개")
    print(f"텍스트 없음: {stats['empty']}개")
    print(f"요청/처리 실패: {stats['failed']}개")
    print(f"JSONL: {OUTPUT}")
    if OUTPUT.exists():
        print(f"JSONL 크기: {OUTPUT.stat().st_size:,} bytes")
    print("=" * 72)


if __name__ == "__main__":
    args = parse_args()
    try:
        CURRENT_MIN_DATE = datetime.strptime(args.min_date, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit("--min-date는 YYYY-MM-DD 형식이어야 합니다.") from exc

    if args.repair_pdfs:
        configure_tesseract()
        repair_existing_pdf_text()
    elif args.full:
        crawl(mode="full", max_pages=max(0, args.max_pages))
    elif args.resume_full:
        crawl(mode="resume", max_pages=max(0, args.max_pages))
    else:
        crawl(mode="update", max_pages=max(0, args.max_pages))
