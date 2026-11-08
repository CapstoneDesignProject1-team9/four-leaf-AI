"""BGE-M3 공지·강의계획서 통합 코사인 검색과 vLLM 답변 생성."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

_notice_collection: Any | None = None
_syllabi_collection: Any | None = None
_embedder: Any | None = None


class BgeM3Embedder:
    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None

    def load(self) -> None:
        if self.model:
            return
        snapshot = (
            Path(settings.EMBEDDING_MODEL_CACHE_DIR)
            / "models--BAAI--bge-m3"
            / "snapshots"
            / settings.EMBEDDING_MODEL_REVISION
        )
        if not snapshot.is_dir():
            # Download the pinned public model once into the persistent cache volume.
            # Subsequent container starts reuse this snapshot without another download.
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=settings.EMBEDDING_MODEL_NAME,
                    revision=settings.EMBEDDING_MODEL_REVISION,
                    cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
                )
            )
        if settings.EMBEDDING_DEVICE == "cpu":
            import torch

            torch.set_num_threads(settings.EMBEDDING_CPU_THREADS)
        self.model = SentenceTransformer(
            str(snapshot),
            device=settings.EMBEDDING_DEVICE,
            local_files_only=True,
            trust_remote_code=False,
        )
        if self.model.get_sentence_embedding_dimension() != 1024:
            raise RuntimeError("BGE-M3 임베딩 차원이 1024가 아닙니다.")

    def embed_query(self, query: str) -> list[float]:
        self.load()
        assert self.model is not None
        if (
            len(self.model.tokenizer(query, truncation=False)["input_ids"])
            > self.model.max_seq_length
        ):
            raise ValueError("질문이 BGE-M3 입력 길이를 초과했습니다.")
        return self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].tolist()


def _validate(collection: Any, corpus_name: str) -> None:
    metadata = collection.metadata or {}
    expected = {
        "schema_version": 1,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "model_revision": settings.EMBEDDING_MODEL_REVISION,
        "dimension": 1024,
        "normalized": True,
        "distance_metric": "cosine",
        "index_state": "ready",
    }
    bad = [key for key, value in expected.items() if metadata.get(key) != value]
    if bad or int(metadata.get("active_chunks", 0)) < 1:
        details = ", ".join(bad) or "활성 청크 없음"
        raise RuntimeError(
            f"{corpus_name} 벡터 DB 설정이 일치하지 않거나 색인이 완료되지 않았습니다: " + details
        )


async def init_vectorstore() -> None:
    global _notice_collection, _syllabi_collection, _embedder
    notice_db_path = Path(settings.CHROMA_PERSIST_DIR)
    if not notice_db_path.is_dir():
        raise RuntimeError(
            f"공지 벡터 DB가 없습니다: {notice_db_path}. embed_notices.py index를 실행하세요."
        )
    notice_client = chromadb.PersistentClient(
        path=str(notice_db_path), settings=Settings(anonymized_telemetry=False)
    )
    try:
        notice_collection = notice_client.get_collection(
            settings.CHROMA_COLLECTION_NAME, embedding_function=None
        )
    except Exception as exc:
        raise RuntimeError("공지 ChromaDB 컬렉션을 열지 못했습니다.") from exc
    _validate(notice_collection, "공지")

    syllabi_db_path = Path(settings.CHROMA_SYLLABI_PERSIST_DIR)
    syllabi_collection = None
    if syllabi_db_path.is_dir():
        syllabi_client = chromadb.PersistentClient(
            path=str(syllabi_db_path), settings=Settings(anonymized_telemetry=False)
        )
        try:
            syllabi_collection = syllabi_client.get_collection(
                settings.CHROMA_SYLLABI_COLLECTION_NAME, embedding_function=None
            )
        except Exception as exc:
            raise RuntimeError("강의계획서 ChromaDB 컬렉션을 열지 못했습니다.") from exc
        _validate(syllabi_collection, "강의계획서")
        logger.info(
            "강의계획서 RAG 준비 완료: %s개 청크",
            syllabi_collection.metadata["active_chunks"],
        )
    else:
        logger.warning(
            "강의계획서 벡터 DB가 아직 없습니다: %s. 공지 자료만 검색합니다.",
            syllabi_db_path,
        )

    embedder = BgeM3Embedder()
    embedder.load()
    _notice_collection, _syllabi_collection, _embedder = (
        notice_collection,
        syllabi_collection,
        embedder,
    )
    logger.info("공지 RAG 준비 완료: %s개 청크", notice_collection.metadata["active_chunks"])


def _ready() -> tuple[Any, Any | None, BgeM3Embedder]:
    if _notice_collection is None or _embedder is None:
        raise RuntimeError("RAG가 초기화되지 않았습니다.")
    return _notice_collection, _syllabi_collection, _embedder


def _search_collection(
    collection: Any,
    vector: list[float],
    top_k: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Document]:
    where_filter: dict[str, Any] = {"active": True}
    if metadata_filter:
        where_filter = {"$and": [{"active": True}, metadata_filter]}
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, int(collection.metadata["active_chunks"])),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    return [
        Document(page_content=text, metadata={**metadata, "distance": float(distance)})
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def search_documents(question: str) -> list[Document]:
    """공지·강의계획서 두 컬렉션을 질문 키워드와 무관하게 함께 검색한다.

    두 컬렉션은 같은 BGE-M3(정규화) 임베딩과 cosine 공간을 쓰므로 거리를 그대로 비교할 수 있다.
    컬렉션마다 상위 k개를 가져와 cosine 거리 오름차순으로 합친 뒤 전체 상위 k개를 반환한다.
    """
    notice_collection, syllabi_collection, embedder = _ready()
    vector = embedder.embed_query(question)
    docs = _search_collection(notice_collection, vector, settings.RAG_TOP_K)
    if syllabi_collection is not None:
        docs.extend(_search_collection(syllabi_collection, vector, settings.RAG_TOP_K))
    docs.sort(key=lambda d: d.metadata["distance"])
    return docs[: settings.RAG_TOP_K]


def _context(documents: list[Document]) -> str:
    contexts: list[str] = []
    for idx, doc in enumerate(documents, 1):
        md = doc.metadata
        if md.get("source_type") == "syllabus":
            term = " ".join(v for v in (md.get("year", ""), md.get("semester", "")) if v)
            department = md.get("department_filter") or md.get("department", "")
            contexts.append(
                f"[강의계획서 {idx}]\n과목: {md.get('title', '')}\n"
                f"개설: {term}\n학과: {department}\n수강학년: {md.get('grade', '')}\n"
                f"교과목구분: {md.get('category', '')}\n학점: {md.get('credits', '')}\n"
                f"담당교수: {md.get('instructor', '')}\n강좌번호: {md.get('course_code', '')}\n"
                f"URL: {md.get('url', '')}\n근거 요약: {doc.page_content[:1200]}"
            )
        else:
            contexts.append(
                f"[공지 {idx}]\n제목: {md.get('title', '')}\n"
                f"게시일: {md.get('date', '')}\nURL: {md.get('url', '')}\n"
                f"내용:\n{doc.page_content}"
            )
    return "\n\n".join(contexts) or "질문에 해당하는 검색 자료가 없습니다."


ANSWER_INSTRUCTIONS = (
    "공지와 강의계획서 중 질문에 가장 직접 관련된 내용으로 먼저 답하세요. 자료에 없는 내용은 추측하지 마세요. "
    "자료 내용을 통째로 나열하지 말고, 질문에 맞춰 핵심만 설명하세요. 사용한 자료의 제목과 URL을 확인할 수 있게 하세요."
)


class NoticeRagChain:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        question = str(inputs.get("query", "")).strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        documents = search_documents(question)
        prompt = (
            f"당신은 경북대학교 학생을 돕는 AI 튜터입니다. 한국어로 자연스럽고 직접적으로 답변하세요.\n\n"
            f"[답변 지침]\n{ANSWER_INSTRUCTIONS}\n\n"
            f"[검색 자료]\n{_context(documents)}\n\n"
            f"[질문]\n{question}"
        )
        response = self.llm.invoke(prompt)
        return {"result": str(response.content), "source_documents": documents}


def _served_model_names() -> set[str]:
    adapters = json.loads(settings.VLLM_LORA_ADAPTERS or "{}")
    if not isinstance(adapters, dict):
        raise ValueError("VLLM_LORA_ADAPTERS는 JSON 객체여야 합니다.")
    return {settings.VLLM_BASE_MODEL, *adapters}


def build_rag_chain(model_name: str | None = None) -> NoticeRagChain:
    """vLLM OpenAI 호환 API로 결정적(temperature 0) 답변을 생성하는 RAG 체인."""
    _ready()
    if not settings.VLLM_BASE_URL or not settings.VLLM_BASE_MODEL:
        raise RuntimeError("VLLM_BASE_URL과 VLLM_BASE_MODEL을 설정해야 합니다.")
    served_model = model_name or settings.VLLM_DEFAULT_MODEL or settings.VLLM_BASE_MODEL
    if served_model not in _served_model_names():
        raise ValueError(f"vLLM에 등록되지 않은 모델입니다: {served_model}")
    headers = None
    if settings.VLLM_PROXY_KEY and settings.VLLM_PROXY_SECRET:
        headers = {"Modal-Key": settings.VLLM_PROXY_KEY, "Modal-Secret": settings.VLLM_PROXY_SECRET}
    return NoticeRagChain(
        ChatOpenAI(
            model=served_model,
            base_url=f"{settings.VLLM_BASE_URL.rstrip('/')}/v1",
            api_key=settings.VLLM_API_KEY,
            default_headers=headers,
            temperature=0.0,
            max_tokens=settings.VLLM_MAX_TOKENS,
            timeout=settings.VLLM_TIMEOUT_S,
        )
    )
