"""전처리된 경북대 공지 문서를 검색용 청크로 분할한다.

권장 위치: knowledge/preprocessing/chunk_notices.py

기본 실행 (프로젝트 루트 또는 어느 경로에서든 가능):
    python knowledge/preprocessing/chunk_notices.py

기존 결과를 교체해 다시 생성:
    python knowledge/preprocessing/chunk_notices.py --overwrite

직접 지정:
    python knowledge/preprocessing/chunk_notices.py \
        --input knowledge/processed/notices/knu_documents.jsonl \
        --output knowledge/processed/notices/knu_notice_chunks.jsonl \
        --chunk-size 700 --chunk-overlap 100 --overwrite

추가 패키지는 필요 없다. 입력 파일은 수정하지 않으며 출력은 원자적으로 저장한다.
문단·문장·단어 경계를 우선해 나누고, 각 청크에 제목·게시일·첨부파일명을
반복해서 넣어 청크 하나만 검색되어도 출처 문맥을 이해할 수 있게 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "knowledge" / "processed" / "notices" / "knu_documents.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "knowledge" / "processed" / "notices" / "knu_notice_chunks.jsonl"

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100
MIN_BODY_SIZE = 100


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def build_header(metadata: dict) -> str:
    lines = [f"제목: {normalize_text(metadata.get('title')) or '(제목 없음)'}"]
    date = normalize_text(metadata.get("date"))
    attachment_name = normalize_text(metadata.get("attachment_name"))
    if date:
        lines.append(f"게시일: {date}")
    if attachment_name:
        lines.append(f"첨부파일: {attachment_name}")
    return "\n".join(lines)


def split_long_piece(piece: str, limit: int) -> list[str]:
    """긴 문장을 단어 경계로 나누며, 긴 단일 문자열은 문자 단위로 나눈다."""
    if len(piece) <= limit:
        return [piece]

    words = re.findall(r"\S+\s*", piece)
    if len(words) <= 1:
        return [piece[index : index + limit] for index in range(0, len(piece), limit)]

    parts: list[str] = []
    current = ""
    for word in words:
        word = word.rstrip()
        if len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.extend(word[index : index + limit] for index in range(0, len(word), limit))
            continue
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def semantic_pieces(text: str, limit: int) -> list[str]:
    """빈 줄, 줄, 문장 순서로 경계를 찾은 뒤 limit 이하 조각을 만든다."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        for line in paragraph.splitlines():
            line = line.strip()
            if not line:
                continue
            sentences = re.split(r"(?<=[.!?。！？])\s+", line)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    pieces.extend(split_long_piece(sentence, limit))
    return pieces


def overlap_tail(text: str, target: int) -> str:
    if target <= 0 or not text:
        return ""
    if len(text) <= target:
        return text
    tail = text[-target:]
    boundary = re.search(r"\s+", tail)
    return tail[boundary.end() :] if boundary and boundary.end() < len(tail) else tail


def split_text(text: str, limit: int, overlap: int) -> list[str]:
    """내용 조각을 누락하지 않으면서 최대 길이와 가능한 overlap을 적용한다."""
    pieces = semantic_pieces(normalize_text(text), limit)
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
        tail = overlap_tail(current, overlap)
        candidate = f"{tail}\n{piece}" if tail else piece
        current = candidate if len(candidate) <= limit else piece

    if current:
        chunks.append(current)
    return chunks


def load_documents(path: Path) -> list[dict]:
    documents: list[dict] = []
    document_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_number}번째 줄 JSON 오류: {exc}") from exc
            if not isinstance(document, dict):
                raise ValueError(f"{line_number}번째 줄은 JSON 객체가 아닙니다")
            document_id = str(document.get("document_id") or "").strip()
            if not document_id:
                raise ValueError(f"{line_number}번째 줄에 document_id가 없습니다")
            if document_id in document_ids:
                raise ValueError(f"중복 document_id: {document_id}")
            if not isinstance(document.get("metadata"), dict):
                raise ValueError(f"{line_number}번째 줄의 metadata가 객체가 아닙니다")
            document_ids.add(document_id)
            documents.append(document)
    return documents


def chunk_documents(
    documents: Iterable[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict], dict]:
    if chunk_size <= MIN_BODY_SIZE:
        raise ValueError(f"chunk_size는 {MIN_BODY_SIZE}보다 커야 합니다")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap은 0 이상 chunk_size 미만이어야 합니다")

    chunks: list[dict] = []
    skipped: list[dict] = []
    source_types: Counter[str] = Counter()
    document_count = 0

    for document in documents:
        document_count += 1
        document_id = str(document["document_id"])
        metadata = dict(document["metadata"])
        header = build_header(metadata)
        body_limit = chunk_size - len(header) - 2
        if body_limit < MIN_BODY_SIZE:
            raise ValueError(
                f"문서 {document_id}: 제목/메타데이터가 너무 길어 본문 공간이 "
                f"{body_limit}자뿐입니다. chunk_size를 늘리세요."
            )

        text = normalize_text(document.get("text"))
        if not text:
            skipped.append({"document_id": document_id, "reason": "empty_text"})
            continue

        bodies = split_text(text, body_limit, min(chunk_overlap, body_limit - 1))
        total = len(bodies)
        source_types[str(metadata.get("source_type") or "unknown")] += total
        for index, body in enumerate(bodies):
            chunk_text = f"{header}\n\n{body}"
            chunk_id = sha256_text(f"{document_id}:{index}:{chunk_text}")
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": index,
                    "chunk_count": total,
                    "chunk_text": chunk_text,
                    "content_hash": sha256_text(chunk_text),
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "chunk_index": index,
                        "chunk_count": total,
                    },
                }
            )

    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    report = {
        "documents": document_count,
        "chunked_documents": document_count - len(skipped),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_length_min": min(lengths, default=0),
        "chunk_length_max": max(lengths, default=0),
        "chunk_length_average": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "chunks_by_source_type": dict(source_types),
        "skipped": skipped,
    }
    return chunks, report


def atomic_write(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"출력이 이미 있습니다: {path} (--overwrite로 교체)")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.output.with_suffix(".report.json")
    resolved = {args.input.resolve(), args.output.resolve(), report_path.resolve()}
    if len(resolved) != 3:
        raise SystemExit("입력·출력·보고서 경로는 서로 달라야 합니다")
    if not args.input.is_file():
        raise SystemExit(f"입력 파일을 찾을 수 없습니다: {args.input}")
    if not args.overwrite and (args.output.exists() or report_path.exists()):
        raise SystemExit("출력 또는 보고서가 이미 있습니다. --overwrite를 사용하세요.")

    try:
        documents = load_documents(args.input)
        chunks, report = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"청킹 실패: {exc}") from exc

    report["input"] = str(args.input.resolve())
    report["output"] = str(args.output.resolve())
    jsonl = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    atomic_write(args.output, jsonl, args.overwrite)
    atomic_write(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n", args.overwrite
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
