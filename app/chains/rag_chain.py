"""BGE-M3 공지 ChromaDB 검색과 Gemini 답변 생성."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)
_collection: Any | None = None
_embedder: "BgeM3Embedder | None" = None


class BgeM3Embedder:
    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None

    def load(self) -> None:
        if self.model:
            return
        snapshot = (Path(settings.EMBEDDING_MODEL_CACHE_DIR) / "models--BAAI--bge-m3"
                    / "snapshots" / settings.EMBEDDING_MODEL_REVISION)
        if not snapshot.is_dir():
            raise RuntimeError("BGE-M3 모델 캐시가 없습니다. embed_notices.py index를 먼저 실행하세요.")
        if settings.EMBEDDING_DEVICE == "cpu":
            import torch
            torch.set_num_threads(settings.EMBEDDING_CPU_THREADS)
        self.model = SentenceTransformer(str(snapshot), device=settings.EMBEDDING_DEVICE,
                                         local_files_only=True, trust_remote_code=False)
        if self.model.get_sentence_embedding_dimension() != 1024:
            raise RuntimeError("BGE-M3 임베딩 차원이 1024가 아닙니다.")

    def embed_query(self, query: str) -> list[float]:
        self.load()
        assert self.model is not None
        if len(self.model.tokenizer(query, truncation=False)["input_ids"]) > self.model.max_seq_length:
            raise ValueError("질문이 BGE-M3 입력 길이를 초과했습니다.")
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0].tolist()


def _validate(collection: Any) -> None:
    metadata = collection.metadata or {}
    expected = {"schema_version": 1, "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "model_revision": settings.EMBEDDING_MODEL_REVISION, "dimension": 1024,
                "normalized": True, "distance_metric": "cosine", "index_state": "ready"}
    bad = [key for key, value in expected.items() if metadata.get(key) != value]
    if bad or int(metadata.get("active_chunks", 0)) < 1:
        raise RuntimeError("공지 벡터 DB 설정이 일치하지 않거나 색인이 완료되지 않았습니다: " + ", ".join(bad))


async def init_vectorstore() -> None:
    global _collection, _embedder
    db_path = Path(settings.CHROMA_PERSIST_DIR)
    if not db_path.is_dir():
        raise RuntimeError(f"공지 벡터 DB가 없습니다: {db_path}. embed_notices.py index를 실행하세요.")
    client = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME, embedding_function=None)
    except Exception as exc:
        raise RuntimeError("공지 ChromaDB 컬렉션을 열지 못했습니다.") from exc
    _validate(collection)
    embedder = BgeM3Embedder()
    embedder.load()
    _collection, _embedder = collection, embedder
    logger.info("공지 RAG 준비 완료: %s개 청크", collection.metadata["active_chunks"])


def _ready() -> tuple[Any, BgeM3Embedder]:
    if _collection is None or _embedder is None:
        raise RuntimeError("RAG가 초기화되지 않았습니다.")
    return _collection, _embedder


def search_notices(question: str) -> list[Document]:
    collection, embedder = _ready()
    result = collection.query(query_embeddings=[embedder.embed_query(question)],
                              n_results=min(settings.RAG_TOP_K, int(collection.metadata["active_chunks"])),
                              where={"active": True}, include=["documents", "metadatas", "distances"])
    return [Document(page_content=text, metadata={**metadata, "distance": float(distance)})
            for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def _context(documents: list[Document]) -> str:
    return "\n\n".join(f"[공지 {i}]\n제목: {d.metadata.get('title', '')}\n게시일: {d.metadata.get('date', '')}\nURL: {d.metadata.get('url', '')}\n내용:\n{d.page_content}"
                       for i, d in enumerate(documents, 1)) or "검색된 공지가 없습니다."


class NoticeRagChain:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        question = str(inputs.get("query", "")).strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        documents = search_notices(question)
        prompt = f"""당신은 경북대학교 학생을 돕는 AI 튜터입니다. 아래 공지 자료만 근거로 한국어로 답변하세요. 자료에 없는 내용은 추측하지 말고 알 수 없다고 답하세요. 사용한 공지의 제목과 URL을 밝혀 출처를 확인할 수 있게 하세요.

[공지 자료]
{_context(documents)}

[질문]
{question}"""
        response = self.llm.invoke(prompt)
        return {"result": str(response.content), "source_documents": documents}


def build_rag_chain() -> NoticeRagChain:
    _ready()
    return NoticeRagChain(ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL,
                                                  google_api_key=settings.GOOGLE_API_KEY or None,
                                                  temperature=0))
