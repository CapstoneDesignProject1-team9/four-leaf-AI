"""Everytime BGE-M3 indexing using the shared notice embedding engine.

Separate fixed DB/collection. check never loads a model; index/search may do so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from . import embed_notices as engine
else:
    import embed_notices as engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "knowledge/processed/everytime_lectures/review_chunks.jsonl"
DB = ROOT / "knowledge/vectorstores/everytime_bge_m3"
COLLECTION = "everytime_reviews_bge_m3_v1"


def validate_input(path):
    _, report = engine.inspect_input(path)
    for row in engine.read_rows(path):
        meta = row["metadata"]
        if meta.get("source_type") != "everytime_review":
            raise ValueError("에타 리뷰가 아닌 청크가 포함되어 있습니다.")
        if not str(meta.get("lecture_id", "")).isdigit():
            raise ValueError("강의 ID가 없거나 잘못되었습니다.")
        if not row["chunk_id"].startswith("everytime_chunk_"):
            raise ValueError("에타 전용 chunk_id가 아닙니다.")
    return report


def search(args):
    _, collection = engine.open_collection(args.db, collection_name=args.collection)
    meta = collection.metadata or {}
    if meta.get("index_state") != "ready" or not meta.get("model_revision"):
        raise ValueError("색인이 완료되지 않았습니다. 먼저 index를 완료하세요.")
    count = int(meta.get("active_chunks", 0))
    if count < 1:
        raise ValueError("활성 청크가 없습니다.")
    filters = [{"active": True}]
    for field, value in (("course_name", args.course_name), ("professor", args.professor)):
        if value:
            filters.append({field: value})
    where = filters[0] if len(filters) == 1 else {"$and": filters}
    embedder = engine.Embedder(args.device, args.cache_dir, meta["model_revision"], args.offline)
    results = collection.query(
        query_embeddings=embedder.encode([args.question]),
        n_results=min(args.top_k, count), where=where,
        include=["documents", "metadatas", "distances"],
    )
    for index, (text, metadata, distance) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), 1):
        print(f"\n[{index}] {metadata.get('course_name', '')} / {metadata.get('professor') or '교수 미표기'}")
        print(f"학기: {metadata.get('semester', '')} / 별점: {metadata.get('rating')} / 추천: {metadata.get('recommendation_count')}")
        print(f"거리: {distance:.4f} (낮을수록 유사, 정답 확률 아님)")
        print(f"리뷰 문서: {metadata.get('document_id')} / 출처: {metadata.get('url')}\n{text}")
    if not results["ids"][0]:
        print("조건에 맞는 검색 결과가 없습니다.")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "index", "search"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/embedding_models")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="기존 모델 캐시만 사용, 다운로드 금지")
    parser.add_argument("--course-name", help="검색 시 과목명 정확 일치 필터")
    parser.add_argument("--professor", help="검색 시 교수명 정확 일치 필터")
    args = parser.parse_args(argv)
    args.db, args.collection, args.corpus = DB, COLLECTION, "everytime_reviews"
    if not 1 <= args.batch_size <= 128 or args.top_k < 1:
        parser.error("batch-size는 1~128, top-k는 1 이상이어야 합니다.")
    if args.command == "search" and not (args.question or "").strip():
        parser.error("검색 질문이 필요합니다.")
    if args.command != "search" and (args.course_name or args.professor):
        parser.error("과목명·교수 필터는 search에서만 사용하세요.")
    try:
        if args.command == "check":
            print(json.dumps(validate_input(args.input), ensure_ascii=False, indent=2))
        elif args.command == "index":
            validate_input(args.input)
            # Prevent concurrent index processes through this entry point.
            args.db.parent.mkdir(parents=True, exist_ok=True)
            lock = args.db.parent / "everytime_bge_m3.index.lock"
            try:
                handle = lock.open("x", encoding="utf-8")
            except FileExistsError as exc:
                raise ValueError("에타 index 잠금 파일이 있습니다. 다른 작업 종료 여부를 먼저 확인하세요.") from exc
            try:
                with handle:
                    handle.write("Everytime index in progress. Do not run concurrent writers.\n")
                engine.index(args)
            finally:
                lock.unlink(missing_ok=True)
        else:
            search(args)
    except ImportError as exc:
        parser.exit(1, f"필수 패키지 없음: {exc}. requirements-embedding.txt를 설치하세요.\n")
    except KeyboardInterrupt:
        parser.exit(130, "\n중단했습니다. 같은 index 명령으로 저장된 배치부터 재개할 수 있습니다.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"실행 실패: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
