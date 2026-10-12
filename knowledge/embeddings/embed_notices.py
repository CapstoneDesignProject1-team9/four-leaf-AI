"""BGE-M3 공지 임베딩 및 ChromaDB 검색. Python 3.10~3.12 권장.

배치: knowledge/embeddings/embed_notices.py
설치: python -m pip install -r knowledge/embeddings/requirements-embedding.txt
검사: python knowledge/embeddings/embed_notices.py check
저장: python knowledge/embeddings/embed_notices.py index
검색: python knowledge/embeddings/embed_notices.py search "수강정정 신청 방법"
강의계획서 확인: python knowledge/embeddings/embed_notices.py check --corpus syllabi
강의계획서 저장: python knowledge/embeddings/embed_notices.py index --corpus syllabi
강의계획서 검색: python knowledge/embeddings/embed_notices.py search "자연어처리개론 강의 내용" --corpus syllabi

첫 index 실행 시 BGE-M3 가중치(약 2.3GB)를 다운로드한다.
CPU/FP32, 배치 8, 정규화한 1024차원 dense 벡터와 cosine 거리 사용.
index 재실행으로 재개 가능. 저장 완료된 동일 텍스트는 재임베딩하지 않는다.
입력에서 사라진 청크는 전체 index 성공 후 inactive로 보존하며 검색에서 제외한다.
색인 중단/실패 중에는 검색을 차단한다. 한 DB에 index를 동시에 실행하지 말 것.
기존 KR-ELECTRA DB와 다른 경로/컬렉션을 사용한다. LLM API는 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "knowledge/processed/notices/knu_notice_chunks.jsonl"
DEFAULT_DB = PROJECT_ROOT / "knowledge/vectorstores/knu_notice_bge_m3"
MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION_NAME = "knu_notices_bge_m3_v1"
DEFAULT_SYLLABI_INPUT = PROJECT_ROOT / "knowledge/processed/syllabi/knu_syllabus_chunks.jsonl"
DEFAULT_SYLLABI_DB = PROJECT_ROOT / "knowledge/vectorstores/knu_syllabi_bge_m3"
SYLLABI_COLLECTION_NAME = "knu_syllabi_bge_m3_v1"
CORPUS_DEFAULTS = {
    "notices": {
        "input": DEFAULT_INPUT,
        "db": DEFAULT_DB,
        "collection": COLLECTION_NAME,
    },
    "syllabi": {
        "input": DEFAULT_SYLLABI_INPUT,
        "db": DEFAULT_SYLLABI_DB,
        "collection": SYLLABI_COLLECTION_NAME,
    },
}
SPEC = {
    "schema_version": 1,
    "embedding_model": MODEL_NAME,
    "dimension": 1024,
    "normalized": True,
    "distance_metric": "cosine",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSON 객체가 필요합니다")
                for key in ("chunk_id", "document_id", "chunk_text"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f"{key}가 비어 있거나 문자열이 아닙니다")
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError("metadata 객체가 필요합니다")
                for key, value in metadata.items():
                    if not isinstance(key, str) or not key:
                        raise ValueError("metadata 키는 비어 있지 않은 문자열이어야 합니다")
                    if not isinstance(value, (str, int, float, bool)):
                        raise ValueError(f"metadata.{key}: 문자열/숫자/불리언만 지원합니다")
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(f"metadata.{key}: 유한한 숫자가 필요합니다")
                if not metadata.get("post_id") or not metadata.get("url"):
                    raise ValueError("metadata에 post_id와 url이 필요합니다")
                calculated = digest(row["chunk_text"])
                if row.get("content_hash") and row["content_hash"] != calculated:
                    raise ValueError("content_hash가 실제 텍스트와 다릅니다")
                row["content_hash"] = calculated
            except (ValueError, TypeError) as exc:
                raise ValueError(f"입력 {number}번째 줄: {exc}") from exc
            yield row


def inspect_input(path: Path):
    ids, document_ids, post_ids = set(), set(), set()
    sources = Counter()
    for row in read_rows(path):
        if row["chunk_id"] in ids:
            raise ValueError(f"중복 chunk_id: {row['chunk_id']}")
        ids.add(row["chunk_id"])
        document_ids.add(row["document_id"])
        post_ids.add(row["metadata"]["post_id"])
        sources[str(row["metadata"].get("source_type", "unknown"))] += 1
    if not ids:
        raise ValueError("입력이 비었습니다. 기존 색인을 비활성화하지 않습니다")
    return ids, {
        "chunks": len(ids),
        "documents": len(document_ids),
        "posts": len(post_ids),
        "sources": dict(sources),
    }


def batches(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def stored_metadata(row):
    return {
        **row["metadata"],
        "document_id": row["document_id"],
        "content_hash": row["content_hash"],
        "active": True,
    }


class Embedder:
    def __init__(self, device, cache_dir, revision=MODEL_REVISION, local_only=False):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.revision = revision
        self.local_only = local_only
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        if self.device == "cpu":
            import torch

            torch.set_num_threads(4)

        print("BGE-M3 로딩 중 (최초 실행은 다운로드 포함)...", flush=True)
        model_source = MODEL_NAME
        if self.local_only:
            if not self.revision:
                snapshots = list((self.cache_dir / "models--BAAI--bge-m3" / "snapshots").glob("*"))
                if len(snapshots) == 1 and snapshots[0].is_dir():
                    self.revision = snapshots[0].name
            if not self.revision:
                raise ValueError("캐시된 BGE-M3 revision을 찾지 못했습니다")
            model_source = self.cache_dir / "models--BAAI--bge-m3" / "snapshots" / self.revision
            if not model_source.is_dir():
                raise ValueError(
                    "캐시된 모델 revision을 찾지 못했습니다. 먼저 인터넷 연결 상태에서 index를 실행하세요"
                )
        self.model = SentenceTransformer(
            str(model_source),
            device=self.device,
            cache_folder=str(self.cache_dir),
            revision=self.revision,
            local_files_only=self.local_only,
            trust_remote_code=False,
        )
        # 최초로 다운로드한 모델 revision을 컬렉션에 고정한다.
        resolved = getattr(self.model[0].auto_model.config, "_commit_hash", None) or self.revision
        if not resolved:
            raise ValueError("모델 revision을 확인하지 못해 색인을 중단합니다")
        if self.revision and resolved != self.revision:
            raise ValueError("저장된 모델 revision과 로딩된 revision이 다릅니다")
        self.revision = resolved
        if self.model.get_sentence_embedding_dimension() != SPEC["dimension"]:
            raise ValueError("BGE-M3 출력 차원이 예상과 다릅니다")

    def encode(self, texts):
        self.load()
        # max_seq_length 초과를 묵시적으로 잘라내지 않는다.
        tokens = self.model.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        lengths = [len(ids) for ids in tokens]
        if max(lengths) > self.model.max_seq_length:
            raise ValueError(f"모델 입력 한도 초과: {max(lengths)} 토큰. 해당 청크를 확인하세요")
        vectors = self.model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        import numpy as np

        if vectors.shape != (len(texts), SPEC["dimension"]) or not np.isfinite(vectors).all():
            raise ValueError("임베딩 출력이 잘못되었습니다")
        return vectors.tolist()


def open_collection(db_path, create=False, collection_name=COLLECTION_NAME):
    import chromadb
    from chromadb.config import Settings

    if not create and not db_path.exists():
        raise ValueError("색인이 없습니다. 먼저 index를 실행하세요")
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )
    if create:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={**SPEC, "index_state": "empty"},
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
    else:
        collection = client.get_collection(name=collection_name, embedding_function=None)
    for key, expected in SPEC.items():
        if (collection.metadata or {}).get(key) != expected:
            raise ValueError(f"컬렉션 설정 불일치: {key}. 다른 DB 경로를 사용하세요")
    return client, collection


def upsert_batch(collection, embedder, rows):
    existing = collection.get(ids=[r["chunk_id"] for r in rows], include=["metadatas"])
    saved = dict(zip(existing["ids"], existing["metadatas"]))
    changed, metadata_updates = [], []
    reused = 0
    for row in rows:
        previous = saved.get(row["chunk_id"]) or {}
        if previous.get("content_hash") != row["content_hash"]:
            changed.append(row)
        else:
            reused += 1
            if previous != stored_metadata(row):
                metadata_updates.append(row)
    if changed:
        vectors = embedder.encode([r["chunk_text"] for r in changed])
        collection.upsert(
            ids=[r["chunk_id"] for r in changed],
            embeddings=vectors,
            documents=[r["chunk_text"] for r in changed],
            metadatas=[stored_metadata(r) for r in changed],
        )
    if metadata_updates:
        collection.update(
            ids=[r["chunk_id"] for r in metadata_updates],
            metadatas=[stored_metadata(r) for r in metadata_updates],
        )
    return len(changed), reused


def deactivate_stale(collection, active_ids):
    count = 0
    # delete 대신 update이므로 페이지 순서/전체 행 수는 유지된다.
    for offset in range(0, collection.count(), 1000):
        page = collection.get(limit=1000, offset=offset, include=["metadatas"])
        ids, metadata = [], []
        for chunk_id, meta in zip(page["ids"], page["metadatas"]):
            if chunk_id not in active_ids and (meta or {}).get("active", True):
                ids.append(chunk_id)
                metadata.append({**(meta or {}), "active": False})
        if ids:
            collection.update(ids=ids, metadatas=metadata)
            count += len(ids)
    return count


def index(args):
    # 전체 파일 검사 후 DB를 열어 잘못된 입력의 부분 저장을 막는다.
    fingerprint = input_digest(args.input)
    active_ids, report = inspect_input(args.input)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    client, collection = open_collection(
        args.db, create=True, collection_name=args.collection
    )
    meta = dict(collection.metadata or {})
    revision = meta.get("model_revision")
    if collection.count() and not revision:
        raise ValueError("기존 벡터의 모델 revision이 없습니다. 새 DB 경로를 사용하세요")
    embedder = Embedder(args.device, args.cache_dir, revision or MODEL_REVISION, args.offline)
    if not revision:
        embedder.load()
        meta["model_revision"] = embedder.revision
    meta.update(index_state="indexing", input_sha256=fingerprint)
    collection.modify(metadata=meta)
    started = time.perf_counter()
    encoded = reused = 0
    for batch in batches(read_rows(args.input), args.batch_size):
        new, old = upsert_batch(collection, embedder, batch)
        encoded += new
        reused += old
        print(
            f"[{encoded + reused}/{report['chunks']}] 새 임베딩 {encoded}, 재사용 {reused}",
            flush=True,
        )
    if input_digest(args.input) != fingerprint:
        raise ValueError("처리 중 입력 파일이 변경됐습니다. index를 다시 실행하세요")
    inactive = deactivate_stale(collection, active_ids)
    report.update(
        encoded=encoded,
        reused=reused,
        deactivated=inactive,
        seconds=round(time.perf_counter() - started, 2),
        model_revision=meta["model_revision"],
        db=str(args.db.resolve()),
        collection=args.collection,
        corpus=args.corpus,
    )
    meta.update(index_state="ready", active_chunks=len(active_ids))
    collection.modify(metadata=meta)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def search(args):
    client, collection = open_collection(args.db, collection_name=args.collection)
    meta = collection.metadata or {}
    if meta.get("index_state") != "ready":
        raise ValueError("색인이 완료되지 않았습니다. index를 다시 실행하세요")
    if not meta.get("model_revision"):
        raise ValueError("모델 revision이 없는 색인입니다")
    embedder = Embedder(args.device, args.cache_dir, meta["model_revision"], args.offline)
    vector = embedder.encode([args.question])
    results = collection.query(
        query_embeddings=vector,
        n_results=min(args.top_k, int(meta["active_chunks"])),
        where={"active": True},
        include=["documents", "metadatas", "distances"],
    )
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        1,
    ):
        print(f"\n[{rank}] {metadata.get('title', '')}")
        detail = metadata.get("attachment_name") or metadata.get("course_code", "")
        print(f"출처: {metadata.get('source_type', '')} / {detail}")
        print(f"URL: {metadata.get('url', '')}")
        print(f"cosine 거리: {distance:.4f} (낮을수록 유사, 정답 확률 아님)")
        print(f"chunk_id: {chunk_id}\n{document}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "index", "search"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--corpus", choices=tuple(CORPUS_DEFAULTS), default="notices")
    parser.add_argument("--input", type=Path, help="입력 청크 JSONL (선택한 코퍼스 기본값 사용)")
    parser.add_argument("--db", type=Path, help="ChromaDB 경로 (선택한 코퍼스 기본값 사용)")
    parser.add_argument("--collection", help="ChromaDB 컬렉션 이름 (선택한 코퍼스 기본값 사용)")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache/embedding_models")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="캐시된 모델만 사용")
    args = parser.parse_args()
    defaults = CORPUS_DEFAULTS[args.corpus]
    args.input = args.input or defaults["input"]
    args.db = args.db or defaults["db"]
    args.collection = args.collection or defaults["collection"]
    if not 1 <= args.batch_size <= 128 or args.top_k < 1:
        parser.error("batch-size: 1~128, top-k: 1 이상")
    if args.command == "search" and not (args.question or "").strip():
        parser.error("검색 질문을 입력하세요")
    try:
        if args.command == "check":
            _, report = inspect_input(args.input)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "index":
            index(args)
        else:
            search(args)
    except ImportError as exc:
        parser.exit(1, f"필수 패키지 없음: {exc}\nrequirements-embedding.txt를 설치하세요.\n")
    except KeyboardInterrupt:
        parser.exit(130, "\n중단되었습니다. index를 다시 실행하면 저장된 배치를 재사용합니다.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"실행 실패: {exc}\n")


if __name__ == "__main__":
    main()
