"""BGE-M3 공지·강의계획서 통합 코사인 검색과 vLLM 답변 생성."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from app.chains.lexical import Bm25Index
from app.chains.semester import Term, current_term, parse_question, parse_title
from app.core.config import settings

logger = logging.getLogger(__name__)

_notice_collection: Any | None = None
_syllabi_collection: Any | None = None
_embedder: Any | None = None
_lexical: tuple[Bm25Index, list[tuple[Any, str, str, dict]]] | None = (
    None  # 색인, (컬렉션, id, 본문, 메타)
)


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
        # 폴더만 있고 가중치가 없으면(다운로드가 중간에 끊긴 경우) 이어서 받는다.
        if not (snapshot / "pytorch_model.bin").is_file():
            # Download the pinned public model once into the persistent cache volume.
            # Subsequent container starts reuse this snapshot without another download.
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=settings.EMBEDDING_MODEL_NAME,
                    revision=settings.EMBEDDING_MODEL_REVISION,
                    cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
                    # 쓰지 않는 ONNX 사본(약 2GB)과 이미지는 받지 않는다.
                    ignore_patterns=["onnx/*", "imgs/*", "*.jpg"],
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


def _open_collection(db_path: Path, collection_name: str, corpus_name: str) -> Any:
    client = chromadb.PersistentClient(
        path=str(db_path), settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection(collection_name, embedding_function=None)
    except Exception as exc:
        raise RuntimeError(f"{corpus_name} ChromaDB 컬렉션을 열지 못했습니다.") from exc
    _validate(collection, corpus_name)
    logger.info("%s RAG 준비 완료: %s개 청크", corpus_name, collection.metadata["active_chunks"])
    return collection


async def init_vectorstore() -> None:
    global _notice_collection, _syllabi_collection, _embedder, _lexical
    notice_db_path = Path(settings.CHROMA_PERSIST_DIR)
    if not notice_db_path.is_dir():
        raise RuntimeError(
            f"공지 벡터 DB가 없습니다: {notice_db_path}. embed_notices.py index를 실행하세요."
        )
    notice_collection = _open_collection(notice_db_path, settings.CHROMA_COLLECTION_NAME, "공지")

    syllabi_db_path = Path(settings.CHROMA_SYLLABI_PERSIST_DIR)
    syllabi_collection = None
    if syllabi_db_path.is_dir():
        syllabi_collection = _open_collection(
            syllabi_db_path, settings.CHROMA_SYLLABI_COLLECTION_NAME, "강의계획서"
        )
    else:
        logger.warning(
            "강의계획서 벡터 DB가 아직 없습니다: %s. 공지 자료만 검색합니다.",
            syllabi_db_path,
        )

    embedder = BgeM3Embedder()
    embedder.load()
    lexical = _build_lexical([c for c in (notice_collection, syllabi_collection) if c is not None])
    _notice_collection, _syllabi_collection, _embedder, _lexical = (
        notice_collection,
        syllabi_collection,
        embedder,
        lexical,
    )


def _build_lexical(collections: list[Any]) -> tuple[Bm25Index, list[tuple[Any, str, str, dict]]]:
    docs: list[tuple[Any, str, str, dict]] = []
    for collection in collections:
        got = collection.get(where={"active": True}, include=["documents", "metadatas"])
        docs.extend(
            (collection, i, text, metadata)
            for i, text, metadata in zip(got["ids"], got["documents"], got["metadatas"])
        )
    logger.info("키워드(BM25) 색인 준비 완료: %s개 청크", len(docs))
    return Bm25Index([text for _, _, text, _ in docs]), docs


def _ready() -> tuple[Any, Any | None, BgeM3Embedder]:
    if _notice_collection is None or _embedder is None:
        raise RuntimeError("RAG가 초기화되지 않았습니다.")
    return _notice_collection, _syllabi_collection, _embedder


def _search_collection(collection: Any, vector: list[float], top_k: int) -> list[Document]:
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, int(collection.metadata["active_chunks"])),
        where={"active": True},
        include=["documents", "metadatas", "distances"],
    )
    return [
        Document(page_content=text, metadata={**metadata, "distance": float(distance)})
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


# 학기가 다른 자료를 걸러내도 top_k가 남도록 컬렉션마다 top_k의 몇 배를 본다.
CANDIDATE_FACTOR = 4
# 기준 학기 자료가 빠진 다른 학기 자료보다 이만큼(cosine 거리)까지 멀어도 같은 주제로 본다.
RELEVANCE_MARGIN = 0.05
MAX_TERM_OPTIONS = 3
MAX_CHUNKS_PER_SOURCE = (
    2  # top_k 안에서 한 공지·강의계획서가 차지할 수 있는 청크 수  # 되물을 때 보여줄 학기 수
)


@dataclass
class Retrieval:
    documents: list[Document]
    target: Term  # 답변 기준 학기
    explicit: bool  # 질문에 학기가 있었는지
    other_terms: list[Term] = field(default_factory=list)  # 걸러낸 다른 학기(검색 순위 순)
    displaced: bool = False  # 거르지 않았다면 top_k에 들었을 다른 학기 자료가 있었는지
    has_target: bool = False  # 같은 주제의 기준 학기 자료가 남았는지
    top_excluded: bool = False  # 가장 관련 높은 자료가 다른 학기라 빠졌는지
    section_options: list[str] = field(default_factory=list)  # 분반을 되물을 때의 선택지

    @property
    def note(self) -> str | None:
        """학기를 말하지 않았고 다른 학기 자료를 실제로 밀어냈다면 기준 학기를 밝힌다."""
        if self.explicit or not (self.displaced and self.has_target):
            return None
        return f"{self.target.label()} 기준으로 안내드려요. 다른 학기가 궁금하면 말씀해 주세요."


def doc_term(doc: Document) -> Term | None:
    md = doc.metadata
    if md.get("source_type") == "syllabus":
        return parse_title(f"{md.get('year', '')} {md.get('semester', '')}")
    return parse_title(str(md.get("title", "")))


def _excluded(term: Term | None, target: Term, explicit: bool) -> bool:
    """학기가 명시된 자료는 기준 학기와 다르면 뺀다. 연도만 있는 자료(교육과정 등)는
    질문이 연도를 콕 집었을 때만 연도로 거른다."""
    if term is None:
        return False
    if term.season is None:
        return explicit and term.year != target.year
    return not target.matches(term)


def search_candidates(vector: list[float]) -> list[Document]:
    """두 컬렉션에서 학기 필터 전 후보를 cosine 거리순으로 가져온다."""
    notice_collection, syllabi_collection, _ = _ready()
    n = settings.RAG_TOP_K * CANDIDATE_FACTOR
    candidates = _search_collection(notice_collection, vector, n)
    if syllabi_collection is not None:
        candidates.extend(_search_collection(syllabi_collection, vector, n))
    candidates.sort(key=lambda d: d.metadata["distance"])
    return candidates


def merge_candidates(*lists: list[Document]) -> list[Document]:
    """같은 청크는 더 가까운 거리 하나만 남기고 거리순으로 합친다."""
    best: dict[str, Document] = {}
    for doc in (d for docs in lists for d in docs):
        key = doc.page_content
        if key not in best or doc.metadata["distance"] < best[key].metadata["distance"]:
            best[key] = doc
    return sorted(best.values(), key=lambda d: d.metadata["distance"])


def gather_candidates(question: str, target: Term, explicit: bool) -> list[Document]:
    """벡터 검색과 키워드(BM25) 검색 결과를 순위 융합(RRF)으로 합친다.

    - 학기를 말하지 않은 질문은 기준 학기를 붙여 한 번 더 벡터 검색한다. 수강신청처럼 학기마다
      비슷한 공지가 수십 개면 최신 학기 공지가 후보에 들지 못하기 때문이다.
    - 키워드 검색은 공지 제목·과목명이 그대로 겹치는 자료를 보완한다.
    """
    _, _, embedder = _ready()
    vector = embedder.embed_query(question)
    dense = search_candidates(vector)
    if not explicit:
        with_term = search_candidates(embedder.embed_query(f"{question} {target.label()}"))
        dense = merge_candidates(dense, with_term)
    return fuse_rankings(dense, _lexical_candidates(question, vector, target, explicit))


def _lexical_candidates(
    question: str, vector: list[float], target: Term, explicit: bool
) -> list[Document]:
    """BM25 상위 자료 중 기준 학기에 맞는 것. 학기 판정에 쓰도록 실제 cosine 거리를 붙인다."""
    if _lexical is None:
        return []
    index, docs = _lexical
    n = settings.RAG_TOP_K * CANDIDATE_FACTOR
    picked = [
        docs[i]
        for i in index.search(question, n * 3)
        if not _excluded(doc_term(Document(page_content="", metadata=docs[i][3])), target, explicit)
    ][:n]
    query = np.asarray(vector)
    distance: dict[tuple[int, str], float] = {}
    for collection in {id(c): c for c, _, _, _ in picked}.values():
        got = collection.get(
            ids=[i for c, i, _, _ in picked if c is collection], include=["embeddings"]
        )
        distance.update(
            ((id(collection), i), float(1 - np.dot(query, e)))
            for i, e in zip(got["ids"], got["embeddings"])
        )
    return [
        Document(page_content=text, metadata={**metadata, "distance": distance[(id(c), i)]})
        for c, i, text, metadata in picked
    ]


def fuse_rankings(*rankings: list[Document], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion. 같은 청크는 거리가 가까운 쪽 메타데이터를 쓴다."""
    score: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            score[key] = score.get(key, 0.0) + 1 / (k + rank)
            if key not in docs or doc.metadata["distance"] < docs[key].metadata["distance"]:
                docs[key] = doc
    return sorted(docs.values(), key=lambda d: -score[d.page_content])


def retrieve(question: str, today: date | None = None) -> Retrieval:
    """공지·강의계획서 두 컬렉션을 함께 검색하고 기준 학기와 다른 학기 자료를 거른다.

    두 컬렉션은 같은 BGE-M3(정규화) 임베딩과 cosine 공간을 쓰므로 거리를 그대로 비교할 수 있다.
    기준 학기는 질문에 학기가 있으면 그 학기, 없으면 현재 학기다.
    """
    asked = parse_question(question, today)
    target = asked or current_term(today)
    candidates = gather_candidates(question, target, asked is not None)
    candidates, section_options = pick_section(candidates, question)
    retrieval = filter_by_term(candidates, target, asked is not None)
    retrieval.section_options = section_options
    return retrieval


def pick_section(candidates: list[Document], question: str) -> tuple[list[Document], list[str]]:
    """가장 관련 높은 자료가 여러 분반 과목의 강의계획서일 때 분반을 고른다.

    질문에 강좌번호·'N분반'·담당교수 이름이 있으면 그 분반만 남기고,
    없으면 후보를 그대로 두고 되물을 분반 선택지를 돌려준다.
    """
    if not candidates or candidates[0].metadata.get("source_type") != "syllabus":
        return candidates, []
    course = str(candidates[0].metadata.get("course_code", "")).split("-")[0]
    sections: dict[str, str] = {}  # 강좌번호 → 담당교수
    for doc in candidates:
        code = str(doc.metadata.get("course_code", ""))
        if doc.metadata.get("source_type") == "syllabus" and code.startswith(f"{course}-"):
            sections.setdefault(code, str(doc.metadata.get("instructor", "")))
    if len(sections) < 2:
        return candidates, []

    def asked_for(code: str, instructor: str) -> bool:
        number = int(code.rsplit("-", 1)[1])
        return (
            code in question
            or re.search(rf"(?<!\d)0*{number}\s*분반", question) is not None
            or (bool(instructor) and instructor in question)
        )

    chosen = {code for code, instructor in sections.items() if asked_for(code, instructor)}
    if not chosen:
        return candidates, [
            f"{code} ({instructor} 교수)" if instructor else code
            for code, instructor in sorted(sections.items())
        ]
    kept = [
        d
        for d in candidates
        if not (
            d.metadata.get("source_type") == "syllabus"
            and str(d.metadata.get("course_code", "")).startswith(f"{course}-")
            and d.metadata.get("course_code") not in chosen
        )
    ]
    return kept, []


def clarify(retrieval: Retrieval) -> tuple[str, list[str]] | None:
    """답하기 전에 되물어야 하면 (되묻는 말, 선택지)를 돌려준다. LLM을 거치지 않는다."""
    if retrieval.section_options:
        top = retrieval.documents[0].metadata if retrieval.documents else {}
        title = (
            str(top.get("title", "")).split(" (")[0] if top.get("source_type") == "syllabus" else ""
        )
        return (
            f"{title or '이 과목'}은(는) 분반마다 담당 교수님과 수업 운영이 달라요. 몇 분반 기준으로 알려드릴까요?",
            retrieval.section_options,
        )
    if retrieval.top_excluded and not retrieval.has_target and retrieval.other_terms:
        terms = retrieval.other_terms
        # '겨울계절학기'처럼 콕 집어 물었으면 다른 해의 같은 학기를 먼저 보여준다.
        same = [t for t in terms if retrieval.explicit and t.season == retrieval.target.season]
        terms = sorted(same or terms, key=Term.order, reverse=True)[:MAX_TERM_OPTIONS]
        ask = (
            "아래 학기 자료로 알려드릴까요?"
            if retrieval.explicit
            else "어느 학기 기준으로 알려드릴까요?"
        )
        return (
            f"{retrieval.target.label()} 자료는 찾지 못했어요. {ask}",
            [t.label() for t in terms],
        )
    return None


def source_key(doc: Document) -> str:
    """자료 하나를 가리키는 키. 강의계획서는 URL이 모두 같아 강좌번호로 구분한다."""
    md = doc.metadata
    if md.get("source_type") == "syllabus" and md.get("course_code"):
        return str(md["course_code"])
    return str(md.get("url") or doc.page_content)


def _limit_per_source(docs: list[Document], top_k: int) -> list[Document]:
    """한 공지의 청크가 top_k를 다 차지하지 않도록 자료당 MAX_CHUNKS_PER_SOURCE개까지만 고른다."""
    picked: list[Document] = []
    counts: dict[str, int] = {}
    for doc in docs:
        source = source_key(doc)
        if counts.get(source, 0) >= MAX_CHUNKS_PER_SOURCE:
            continue
        counts[source] = counts.get(source, 0) + 1
        picked.append(doc)
        if len(picked) == top_k:
            break
    return picked


def filter_by_term(candidates: list[Document], target: Term, explicit: bool) -> Retrieval:
    """거리순 후보에서 기준 학기와 다른 학기 자료를 빼고 top_k를 고른다."""
    top_k = settings.RAG_TOP_K
    kept: list[Document] = []
    others: list[Term] = []
    displaced = False
    best_excluded = float("inf")
    for rank, doc in enumerate(candidates):
        term = doc_term(doc)
        if not _excluded(term, target, explicit):
            kept.append(doc)
            continue
        displaced = displaced or rank < top_k
        best_excluded = min(best_excluded, doc.metadata["distance"])
        if term is not None and term not in others:
            others.append(term)
    kept = _limit_per_source(kept, top_k)
    # 강의계획서처럼 주제가 다른 기준 학기 자료만 있으면 기준 학기 자료가 없는 것으로 본다.
    has_target = any(
        (t := doc_term(d))
        and t.season
        and target.matches(t)
        and d.metadata["distance"] <= best_excluded + RELEVANCE_MARGIN
        for d in kept
    )
    return Retrieval(
        documents=kept,
        target=target,
        explicit=explicit,
        other_terms=others,
        displaced=displaced,
        has_target=has_target,
        top_excluded=bool(candidates) and _excluded(doc_term(candidates[0]), target, explicit),
    )


def search_documents(question: str) -> list[Document]:
    return retrieve(question).documents


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


def build_prompt(question: str, documents: list[Document]) -> str:
    """서빙과 LoRA 학습(train/build_rag_sft.py)이 함께 쓰는 프롬프트. 바꾸면 재학습이 필요하다."""
    return (
        f"당신은 경북대학교 학생을 돕는 AI 튜터입니다. 한국어로 자연스럽고 직접적으로 답변하세요.\n\n"
        f"[답변 지침]\n{ANSWER_INSTRUCTIONS}\n\n"
        f"[검색 자료]\n{_context(documents)}\n\n"
        f"[질문]\n{question}"
    )


class NoticeRagChain:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def invoke(self, inputs: dict[str, str], retrieval: Retrieval | None = None) -> dict[str, Any]:
        question = str(inputs.get("query", "")).strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        retrieval = retrieval or retrieve(question)
        response = self.llm.invoke(build_prompt(question, retrieval.documents))
        answer = str(response.content)
        if retrieval.note:
            answer = f"{retrieval.note}\n\n{answer}"
        return {"result": answer, "source_documents": retrieval.documents}


def _served_model_names() -> set[str]:
    adapters = json.loads(settings.VLLM_LORA_ADAPTERS or "{}")
    if not isinstance(adapters, dict):
        raise ValueError("VLLM_LORA_ADAPTERS는 JSON 객체여야 합니다.")
    return {settings.VLLM_BASE_MODEL, *adapters}


def resolve_model_name(model_name: str | None = None) -> str:
    """요청 모델 → 기본 모델(VLLM_DEFAULT_MODEL) → 기반 모델 순으로 실제 사용할 모델명을 정한다."""
    return model_name or settings.VLLM_DEFAULT_MODEL or settings.VLLM_BASE_MODEL


def build_rag_chain(model_name: str | None = None) -> NoticeRagChain:
    """vLLM OpenAI 호환 API로 결정적(temperature 0) 답변을 생성하는 RAG 체인."""
    _ready()
    if not settings.VLLM_BASE_URL or not settings.VLLM_BASE_MODEL:
        raise RuntimeError("VLLM_BASE_URL과 VLLM_BASE_MODEL을 설정해야 합니다.")
    served_model = resolve_model_name(model_name)
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
