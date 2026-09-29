"""공지 JSONL을 출처별 RAG 문서로 전처리한다. Python 3.10+, 추가 패키지 없음.

권장 위치: knowledge/processors/prepare_knu_notices.py
실행 (프로젝트 루트):
    python knowledge/processors/prepare_knu_notices.py
입출력 직접 지정:
    python prepare_knu_notices.py --input raw.jsonl --output documents.jsonl

원본은 변경하지 않는다. 출력 파일이 존재하면 --overwrite가 있어야 교체한다.
본문/이미지 OCR/첨부를 별도 문서로 보존하며 청킹과 임베딩은 수행하지 않는다.
OCR 오류나 조각난 표를 추측해서 복원하지 않는다. 품질 보고서는 출력 옆에 저장한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def stable_url(value: object) -> str:
    parts = urlsplit(str(value or ""))
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k not in {"nonce", "page"}
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(path: Path) -> tuple[list[dict], dict]:
    posts: dict[str, dict] = {}
    duplicates = 0
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("공지 객체가 아닙니다")
                key = str(item.get("post_id") or stable_url(item.get("url")))
                if not key:
                    raise ValueError("post_id와 url이 모두 없습니다")
                if not isinstance(item.get("attachments", []), list):
                    raise ValueError("attachments는 배열이어야 합니다")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{line_number}번째 줄: {exc}") from exc
            duplicates += int(key in posts)
            posts[key] = item  # 같은 공지의 마지막 레코드를 사용한다.

    documents = []
    issues = []
    methods = Counter()
    dates = []
    for post_id, item in posts.items():
        title = normalize_text(item.get("title"))
        notice_url = stable_url(item.get("normalized_url") or item.get("url"))
        date = str(item.get("date") or "")
        if date:
            dates.append(date)
        sources = [
            ("body", "body", item.get("content"), "html", None),
            ("image_ocr", "image_ocr", item.get("ocr_text"), "ocr", None),
        ]
        attachment_keys = Counter()
        for index, attachment in enumerate(item.get("attachments", []), 1):
            if not isinstance(attachment, dict):
                raise ValueError(f"공지 {post_id}: {index}번째 첨부가 객체가 아닙니다")
            method = str(attachment.get("text_extraction") or "unknown")
            methods[method] += 1
            ref = stable_url(attachment.get("url")) or str(attachment.get("name") or index)
            attachment_keys[ref] += 1
            source_key = f"attachment:{ref}:{attachment_keys[ref]}"
            sources.append(("attachment", source_key, attachment.get("text"), method, attachment))
        before = len(documents)
        for source_type, source_key, raw_text, method, attachment in sources:
            text = normalize_text(raw_text)
            if not text:
                if attachment is not None:
                    issues.append(
                        {
                            "post_id": post_id,
                            "issue": "attachment_without_text",
                            "name": attachment.get("name", ""),
                            "text_extraction": method,
                        }
                    )
                continue
            flags = []
            if method == "ocr":
                flags.append("ocr_review_recommended")
            if "\ufffd" in text:
                flags.append("replacement_character")
            lines = [line for line in text.splitlines() if line]
            if len(lines) >= 10 and sum(len(line) <= 3 for line in lines) / len(lines) > 0.3:
                flags.append("fragmented_lines")
            metadata = {
                "post_id": post_id,
                "title": title,
                "date": date,
                "url": notice_url,
                "source_type": source_type,
                "text_extraction": method,
                "attachment_name": str((attachment or {}).get("name") or ""),
                "attachment_url": stable_url((attachment or {}).get("url")),
                "attachment_sha256": str((attachment or {}).get("sha256") or ""),
                "local_path": str((attachment or {}).get("local_path") or ""),
                "quality_flags": ",".join(flags),
                "crawled_at": str(item.get("crawled_at") or ""),
            }
            prefix = f"제목: {title}\n게시일: {date}" if date else f"제목: {title}"
            if attachment is not None:
                prefix += f"\n첨부파일: {metadata['attachment_name']}"
            document_text = f"{prefix}\n\n{text}"
            documents.append(
                {
                    "document_id": digest(f"knu_notice:{post_id}:{source_key}"),
                    "text": text,
                    "document_text": document_text,
                    "content_hash": digest(document_text),
                    "metadata": metadata,
                }
            )
        if before == len(documents):
            issues.append({"post_id": post_id, "issue": "notice_without_text"})
    report = {
        "input": str(path.resolve()),
        "input_posts": len(posts),
        "duplicate_post_records": duplicates,
        "documents": len(documents),
        "date_min": min(dates, default=""),
        "date_max": max(dates, default=""),
        "source_types": dict(Counter(d["metadata"]["source_type"] for d in documents)),
        "attachment_methods": dict(methods),
        "flagged_documents": sum(bool(d["metadata"]["quality_flags"]) for d in documents),
        "issues": issues,
    }
    return documents, report


def write_output(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
        return
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as file:
            name = file.name
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("knowledge/raw/notices/knu_notices_all.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("knowledge/processed/notices/knu_documents.jsonl")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report_path = args.output.with_suffix(".report.json")
    paths = [p.resolve() for p in (args.input, args.output, report_path)]
    if len(set(paths)) != 3:
        parser.error("입력·출력·보고서 경로는 서로 달라야 합니다")
    if not args.overwrite and any(p.exists() for p in (args.output, report_path)):
        parser.error("출력이 이미 있습니다. 다른 경로나 --overwrite를 사용하세요")
    documents, report = prepare(args.input)
    write_output(
        args.output,
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents),
        args.overwrite,
    )
    write_output(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n", args.overwrite
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
