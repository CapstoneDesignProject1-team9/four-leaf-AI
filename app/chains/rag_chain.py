"""BGE-M3 공지·강의계획서 ChromaDB 검색과 Ollama 답변 생성."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)
_notice_collection: Any | None = None
_syllabi_collection: Any | None = None
_embedder: BgeM3Embedder | None = None


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
            raise RuntimeError(
                "BGE-M3 모델 캐시가 없습니다. embed_notices.py index로 모델 캐시를 준비하세요."
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
            f"{corpus_name} 벡터 DB 설정이 일치하지 않거나 색인이 완료되지 않았습니다: "
            + details
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
            "강의계획서 RAG 준비 완료: %s개 청크", syllabi_collection.metadata["active_chunks"]
        )
    else:
        logger.warning(
            "강의계획서 벡터 DB가 아직 없습니다: %s. 공지 자료만 검색합니다.", syllabi_db_path
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


def search_notices(question: str) -> list[Document]:
    notice_collection, _, embedder = _ready()
    return _search_collection(
        notice_collection, embedder.embed_query(question), settings.RAG_TOP_K
    )


def _requested_grade(question: str) -> str | None:
    match = re.search(r"([1-6])\s*학년", question)
    return match.group(1) if match else None


def _is_syllabus_question(question: str) -> bool:
    normalized = question.replace(" ", "")
    if any(
        term in normalized
        for term in ("강의계획서", "강의목록", "강의내용", "강의추천", "강좌", "교과목", "과목")
    ):
        return True
    return _requested_grade(question) is not None and any(
        term in normalized for term in ("강의", "수강", "들을수")
    )


def search_syllabi(question: str, grade: str | None = None) -> list[Document]:
    _, syllabi_collection, embedder = _ready()
    if syllabi_collection is None:
        return []
    # Several chunks belong to each syllabus. Retrieve a wider pool, then keep
    # the best matching chunk per course so one course cannot fill the results.
    candidates = _search_collection(
        syllabi_collection,
        embedder.embed_query(question),
        max(settings.RAG_TOP_K * 20, 100),
        {"grade": grade} if grade else None,
    )
    best_by_course: dict[str, Document] = {}
    for document in candidates:
        course_id = str(
            document.metadata.get("course_code")
            or document.metadata.get("post_id")
            or document.metadata.get("title")
        )
        best_by_course.setdefault(course_id, document)
    return list(best_by_course.values())[: settings.RAG_TOP_K]


def search_documents(question: str) -> list[Document]:
    """Use syllabi for course questions; otherwise search both corpora."""
    if _is_syllabus_question(question):
        return search_syllabi(question, _requested_grade(question))

    notice_collection, syllabi_collection, embedder = _ready()
    vector = embedder.embed_query(question)
    documents = _search_collection(notice_collection, vector, settings.RAG_TOP_K)
    if syllabi_collection is not None:
        documents.extend(_search_collection(syllabi_collection, vector, settings.RAG_TOP_K))
    documents.sort(key=lambda document: document.metadata["distance"])
    return documents[: settings.RAG_TOP_K]


def _context(documents: list[Document]) -> str:
    contexts = []
    for index, document in enumerate(documents, 1):
        metadata = document.metadata
        if metadata.get("source_type") == "syllabus":
            term = " ".join(
                value
                for value in (metadata.get("year", ""), metadata.get("semester", ""))
                if value
            )
            department = metadata.get("department_filter") or metadata.get("department", "")
            contexts.append(
                f"[강의계획서 {index}]\n과목: {metadata.get('title', '')}\n"
                f"개설: {term}\n학과: {department}\n수강학년: {metadata.get('grade', '')}\n"
                f"학점: {metadata.get('credits', '')}\n담당교수: {metadata.get('instructor', '')}\n"
                f"강좌번호: {metadata.get('course_code', '')}\n"
                f"URL: {metadata.get('url', '')}\n내용:\n{document.page_content}"
            )
        else:
            contexts.append(
                f"[공지 {index}]\n제목: {metadata.get('title', '')}\n"
                f"게시일: {metadata.get('date', '')}\nURL: {metadata.get('url', '')}\n"
                f"내용:\n{document.page_content}"
            )
    return "\n\n".join(contexts) or "검색된 공지와 강의계획서가 없습니다."


class NoticeRagChain:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        question = str(inputs.get("query", "")).strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        documents = search_documents(question)
        syllabus_question = _is_syllabus_question(question)
        grade = _requested_grade(question)
        if syllabus_question:
            instructions = """강의계획서 질의입니다. 공지사항은 참고하지 말고, 검색된 강의계획서만 답변 근거로 사용하세요.
질문이 학년별 수강 과목을 묻는다면 강의계획서의 수강학년 값이 해당 학년인 과목만 최대 5개 제안하세요. 과목명, 강좌번호, 학점, 개설학기와 강의계획서 내용에 근거한 짧은 설명을 간결하게 제시하세요. 검색된 자료를 그대로 나열하거나 자신의 역할을 소개하지 마세요.
강의계획서의 수강학년은 표시된 대상 학년이지 실제 수강 자격의 보증이 아닙니다. 선수과목이나 수강 제한이 자료에 없으면 임의로 단정하지 말고, 최종 수강 가능 여부는 수강편람에서 확인해야 한다고 짧게 덧붙이세요.
근거 자료가 없으면 그 사실과 확인할 수 없는 점을 간단히 설명하세요."""
        else:
            instructions = """공지와 강의계획서 중 질문에 가장 직접 관련된 내용으로 먼저 답하세요. 자료에 없는 내용은 추측하지 마세요. 자료 내용을 통째로 나열하지 말고, 질문에 맞춰 핵심만 설명하세요. 사용한 자료의 제목과 URL을 확인할 수 있게 하세요."""
        prompt = f"""당신은 경북대학교 학생을 돕는 AI 튜터입니다. 한국어로 자연스럽고 직접적으로 답변하세요.

[답변 지침]
{instructions}

[요청 학년]
{grade or '지정되지 않음'}

[검색 자료]
{_context(documents)}

[질문]
{question}"""
        response = self.llm.invoke(prompt)
        return {"result": str(response.content), "source_documents": documents}


def build_rag_chain() -> NoticeRagChain:
    _ready()
    return NoticeRagChain(
        ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2,
            num_ctx=settings.OLLAMA_NUM_CTX,
            num_predict=settings.OLLAMA_NUM_PREDICT,
        )
    )
