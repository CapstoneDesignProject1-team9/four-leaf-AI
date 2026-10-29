"""Split long Everytime reviews without mixing reviews. No embedding or network."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .preprocess_everytime_reviews import atomic_write, digest, file_hash
else:
    from preprocess_everytime_reviews import atomic_write, digest, file_hash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "knowledge/processed/everytime_lectures"
VERSION = "everytime-chunk-v1"


def validate_settings(size: int, overlap: int) -> None:
    if size < 32 or overlap < 0 or overlap * 2 >= size:
        raise ValueError("chunk-size는 32 이상, overlap은 0 이상이며 크기의 절반 미만이어야 합니다.")


def split_spans(text: str, size=700, overlap=100) -> list[tuple[int, int]]:
    """Offsets cover the entire input. Overlap never exceeds the requested size."""
    validate_settings(size, overlap)
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            minimum = start + size // 2
            window = text[minimum:end]
            for pattern in (r"\n\n+", r'[.!?。！？][\"\u201d\u2019]?\s+|\n', r"\s+"):
                boundaries = list(re.finditer(pattern, window))
                if boundaries:
                    end = minimum + boundaries[-1].end()
                    break
        spans.append((start, end))
        if end == len(text):
            break
        next_start = end - overlap
        # Prefer starting on a word boundary within the overlap, not mid-word.
        if overlap:
            boundary = re.search(r"\s+", text[next_start:end])
            if boundary:
                next_start += boundary.end()
        start = max(start + 1, next_start)
    return spans


def chunk_review(row: dict, size=700, overlap=100) -> list[dict]:
    for field in ("document_id", "lecture_id", "content"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"missing_or_invalid_{field}")
    text = row["content"]
    if row.get("content_hash") != digest(text):
        raise ValueError("content_hash_mismatch")
    spans = split_spans(text, size, overlap)
    metadata = {
        field: row[field] for field in (
            "lecture_id", "review_id", "course_name", "professor", "semester", "rating",
            "recommendation_count", "source_hash", "crawled_at", "recommendation_updated_at",
        ) if row.get(field) is not None
    }
    metadata.update(
        source_type="everytime_review", url=row.get("source_url", ""),
        post_id=row["document_id"],
        quality_flags_json=json.dumps(row.get("quality_flags", []), ensure_ascii=False),
    )
    chunks = []
    for index, (start, end) in enumerate(spans):
        body = text[start:end]
        chunks.append({
            "schema_version": 1,
            "chunking_version": VERSION,
            "chunk_id": "everytime_chunk_" + digest(
                f"{VERSION}:{size}:{overlap}:{row['document_id']}:{index}"
            ),
            "document_id": row["document_id"],
            "chunk_index": index,
            "chunk_count": len(spans),
            "start_char": start,
            "end_char": end,
            "overlap_chars": 0 if index == 0 else spans[index - 1][1] - start,
            "chunk_text": body,
            "content_hash": digest(body),
            "metadata": {**metadata, "chunk_index": index, "chunk_count": len(spans)},
        })
    return chunks


def run(source: Path, output: Path, report_path: Path, size=700, overlap=100) -> dict:
    validate_settings(size, overlap)
    source, output, report_path = [p.expanduser().resolve() for p in (source, output, report_path)]
    if len({source, output, report_path}) != 3:
        raise ValueError("입력·출력·보고서 경로는 모두 달라야 합니다.")
    for destination in (output, report_path):
        if destination.exists() and source.samefile(destination):
            raise ValueError("원본을 덮어쓸 수 없습니다.")
    input_hash = file_hash(source)
    chunks = []
    document_ids = set()
    distribution = Counter()
    blank = 0
    with source.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                blank += 1
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not_object")
                if row.get("document_id") in document_ids:
                    raise ValueError("duplicate_document_id")
                batch = chunk_review(row, size, overlap)
                document_ids.add(row["document_id"])
                chunks.extend(batch)
                distribution[len(batch)] += 1
            except (ValueError, TypeError) as exc:
                reason = "invalid_json" if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(f"입력 {number}행 오류: {reason}. 기존 결과를 보존합니다.") from exc
    if not chunks:
        raise ValueError("유효한 리뷰가 없어 기존 결과를 보존합니다.")
    if file_hash(source) != input_hash:
        raise RuntimeError("실행 중 입력이 변경됐습니다. 전처리 완료 후 재실행하세요.")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks)
    report = {
        "schema_version": 1, "chunking_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source), "output_path": str(output),
        "input_sha256": input_hash, "output_sha256": digest(text),
        "chunk_size_chars": size, "max_overlap_chars": overlap,
        "input_reviews": len(document_ids), "output_chunks": len(chunks),
        "single_chunk_reviews": distribution[1],
        "split_reviews": len(document_ids) - distribution[1],
        "chunks_per_review": dict(sorted(distribution.items())), "blank_lines": blank,
        "max_chunk_chars": max(len(c["chunk_text"]) for c in chunks),
        "overlap_characters": sum(c["overlap_chars"] for c in chunks),
        "notes": ["Character bounds are not token bounds. Check model token limits before embedding.",
                  "Ratings and recommendation counts belong to reviews; do not aggregate per chunk."],
    }
    atomic_write(output, text)
    atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="에타 리뷰 청킹: 긴 리뷰만 분할, 서로 다른 리뷰 혼합 없음")
    parser.add_argument("--input", type=Path, default=DATA / "reviews_clean.jsonl")
    parser.add_argument("--output", type=Path, default=DATA / "review_chunks.jsonl")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args(argv)
    report_path = args.report or args.output.with_suffix(".report.json")
    try:
        report = run(args.input, args.output, report_path, args.chunk_size, args.chunk_overlap)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"청킹 실패: {exc}")
        return 1
    print(f"리뷰 {report['input_reviews']}개 → 청크 {report['output_chunks']}개 / 긴 리뷰 분할 {report['split_reviews']}개")
    print(f"결과: {args.output.resolve()}\n보고서: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
