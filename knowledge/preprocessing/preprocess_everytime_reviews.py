"""Everytime review preprocessing. Standard library only; never edits raw input.

One output row per review; no chunking, embedding, summarization or rating weighting.
Existing derived output and report are replaced after input has been processed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "knowledge/raw/everytime_lectures/reviews.jsonl"
DEFAULT_OUTPUT = ROOT / "knowledge/processed/everytime_lectures/reviews_clean.jsonl"
VERSION = "everytime-preprocess-v1"

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
PHONE = re.compile(
    r"(?<!\d)(?:(?:\+82[ .-]?(?:0)?|0)(?:1[016789]|2|[3-6][1-5]|70))"
    r"[ .-]?\d{3,4}[ .-]?\d{4}(?!\d)"
)
RESIDENT_ID = re.compile(r"(?<!\d)\d{6}[ -]?[1-8]\d{6}(?!\d)")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(value))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:p|div|span|b|strong|i|em)\b[^>]*>", "", text, flags=re.I)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def mask_personal_data(text: str) -> tuple[str, dict]:
    counts = {}
    for name, pattern, token in (
        ("email", EMAIL, "[EMAIL]"),
        ("resident_id", RESIDENT_ID, "[RESIDENT_ID]"),
        ("phone", PHONE, "[PHONE]"),
    ):
        text, counts[name] = pattern.subn(token, text)
    return text, {name: count for name, count in counts.items() if count}


def optional_text(value) -> str | None:
    return normalize(value) if isinstance(value, str) and normalize(value) else None


def numeric(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if integer:
        return int(value) if float(value).is_integer() else None
    return float(value) if value <= 5 else None


def clean_record(row: dict) -> dict:
    lecture_id = str(row.get("lecture_id") or "")
    if not re.fullmatch(r"[0-9]+", lecture_id):
        raise ValueError("invalid_lecture_id")
    if not isinstance(row.get("content"), str):
        raise ValueError("invalid_content_type")
    content_before_mask = normalize(row["content"])
    if not content_before_mask:
        raise ValueError("empty_content")
    semester = optional_text(row.get("semester"))
    review_id = optional_text(row.get("review_id"))
    source_hash = optional_text(row.get("source_hash"))
    # Identity is determined BEFORE masking; distinct private data must not
    # collapse two unrelated reviews into a single record.
    identity = review_id or source_hash or digest(
        json.dumps([semester, content_before_mask], ensure_ascii=False)
    )
    content, masks = mask_personal_data(content_before_mask)
    result = {
        "schema_version": 1,
        "preprocessing_version": VERSION,
        "document_id": "everytime_" + digest(lecture_id + "\n" + identity),
        "lecture_id": lecture_id,
        "review_id": review_id,
        "course_name": optional_text(row.get("course_name")),
        "professor": optional_text(row.get("professor")),
        "semester": semester,
        "rating": numeric(row.get("rating")),
        "recommendation_count": numeric(row.get("recommendation_count"), integer=True),
        "content": content,
        "content_hash": digest(content),
        "source_type": "everytime_review",
        # Canonical known source: never copy session/query tokens into output.
        "source_url": f"https://everytime.kr/lecture/view/{lecture_id}?tab=article",
        "source_hash": source_hash,
        "crawled_at": optional_text(row.get("crawled_at")),
        "recommendation_updated_at": optional_text(row.get("recommendation_updated_at")),
        "pii_masks": masks,
    }
    flags = []
    for field in ("course_name", "professor", "semester", "rating", "recommendation_count"):
        if result[field] is None:
            flags.append("missing_or_invalid_" + field)
    if not review_id and not source_hash:
        flags.append("content_based_identity")
    result["quality_flags"] = flags
    return result


def prepare(source: Path) -> tuple[list[dict], dict]:
    records = {}
    rejected = Counter()
    blank = duplicates = conflicts = input_count = 0
    examples = []
    # Iterate lines so a malformed input row cannot invalidate good rows.
    with source.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                blank += 1
                continue
            input_count += 1
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("not_object")
                clean = clean_record(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                reason = "invalid_json" if isinstance(exc, json.JSONDecodeError) else str(exc)
                rejected[reason] += 1
                if len(examples) < 100:
                    examples.append({"line": number, "reason": reason})
                continue
            key = clean["document_id"]
            if key in records:
                duplicates += 1
                if records[key]["content_hash"] != clean["content_hash"]:
                    conflicts += 1
            # Append-ordered snapshots: last input occurrence wins, including
            # updated ratings/votes. Preserve distinct IDs with identical text.
            records[key] = clean
    rows = [records[key] for key in sorted(records)]
    masks = Counter()
    flags = Counter()
    for row in rows:
        masks.update(row["pii_masks"])
        flags.update(row["quality_flags"])
    report = {
        "schema_version": 1,
        "preprocessing_version": VERSION,
        "input_records": input_count,
        "blank_lines": blank,
        "output_reviews": len(rows),
        "duplicates_removed": duplicates,
        "duplicate_content_conflicts": conflicts,
        "rejected_records": sum(rejected.values()),
        "rejection_reasons": dict(rejected),
        "rejected_line_examples": examples,
        "quality_flags": dict(flags),
        "masked_reviews": sum(bool(row["pii_masks"]) for row in rows),
        "pii_mask_counts": dict(masks),
        "unique_lectures": len({row["lecture_id"] for row in rows}),
        "short_reviews_retained_under_20_chars": sum(len(row["content"]) < 20 for row in rows),
        "max_content_characters": max((len(row["content"]) for row in rows), default=0),
        "limitations": [
            "Pattern-based PII masking is not a guarantee of full anonymization.",
            "Names, student IDs, addresses and indirect identifiers require separate review.",
            "Same lecture/review ID: last input row wins; distinct review IDs remain separate.",
            "Missing professor is preserved; no faculty/course identity is inferred.",
        ],
    }
    assert input_count == len(rows) + duplicates + sum(rejected.values())
    return rows, report


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(source: Path, output: Path, report_path: Path) -> dict:
    source, output, report_path = [path.expanduser().resolve() for path in (source, output, report_path)]
    if len({source, output, report_path}) != 3:
        raise ValueError("입력·출력·보고서 경로는 모두 달라야 합니다.")
    for destination in (output, report_path):
        if destination.exists() and source.samefile(destination):
            raise ValueError("원본과 같은 파일을 덮어쓸 수 없습니다.")
    before_hash = file_hash(source)
    rows, report = prepare(source)
    if file_hash(source) != before_hash:
        raise RuntimeError("처리 중 원본이 변경됐습니다. 수집을 중단하고 다시 실행하세요.")
    if not rows:
        raise ValueError("유효한 리뷰가 없어 기존 결과를 보존하고 중단합니다.")
    report.update(
        input_sha256=before_hash,
        input_path=str(source),
        output_path=str(output),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    output_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    report["output_sha256"] = digest(output_text)
    atomic_write(output, output_text)
    atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="에타 리뷰 전처리: 원본 보존, 청킹·임베딩 없음")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report_path = args.report or args.output.with_suffix(".report.json")
    try:
        report = run(args.input, args.output, report_path)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"전처리 실패: {exc}")
        return 1
    print(f"입력 {report['input_records']} / 출력 {report['output_reviews']} / "
          f"중복 제거 {report['duplicates_removed']} / 제외 {report['rejected_records']}")
    print(f"개인정보 패턴 마스킹 리뷰: {report['masked_reviews']}개")
    print(f"결과: {args.output.resolve()}\n보고서: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
