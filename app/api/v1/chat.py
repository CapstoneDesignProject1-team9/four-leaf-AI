import logging
import re
import time

from fastapi import APIRouter, HTTPException

from app.chains.rag_chain import build_rag_chain, clarify, resolve_model_name, retrieve
from app.chains.semester import parse_question
from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from app.storage.chat_logs import (
    complete_chat_log,
    create_chat_log,
    fail_chat_log,
    pending_clarification,
)

logger = logging.getLogger(__name__)
router = APIRouter()

COLD_START_DETAIL = "AI 모델을 준비하고 있어요. 1~2분 후 다시 질문해 주세요."
# 되묻는 답변을 chat_logs에 표시하는 model_name. 다음 턴에서 직전 질문과 합칠 때 쓴다.
CLARIFY_MODEL = "clarify"
# 되물음에 대한 답으로 볼 표현: 학기(parse_question)·강좌번호·N분반·교수
_SECTION_REPLY = re.compile(r"[A-Z]{4}\d{4}-\d{3}|\d+\s*분반|교수")


def _with_previous_question(request: ChatRequest) -> str:
    """직전 턴에 학기·분반을 되물었고 이번 메시지가 그 답이면 직전 질문과 합친다."""
    message = request.message.strip()
    if not request.session_id:
        return message
    if parse_question(message) is None and not _SECTION_REPLY.search(message):
        return message
    try:
        previous = pending_clarification(request.session_id)
    except Exception:
        logger.exception("직전 대화 조회 실패")
        return message
    return f"{previous} {message}" if previous else message


def _is_cold_start(exc: Exception) -> bool:
    """Modal 서버가 꺼져 있다가 켜지는 중이라 요청을 받을 컨테이너가 없는 경우."""
    text = str(exc).lower()
    return "no upstreams available" in text or "no healthy upstream" in text


# 임베딩·LLM 호출·DB 저장이 모두 블로킹이라 일부러 sync def로 둔다.
# FastAPI가 스레드풀에서 실행하므로 답변 생성 중에도 /health 등 다른 요청이 막히지 않는다.
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    AI 튜터 채팅 엔드포인트

    RAG 파이프라인:
    1. BGE-M3로 사용자 질문 임베딩
    2. 키워드 라우팅 없이 공지·강의계획서 전체에서 cosine 거리 상위 5개 청크 검색
    3. vLLM으로 서빙하는 Llama-3.1-8B-Instruct(temperature 0)가 검색된 공지와 강의계획서를 근거로 답변 생성

    질문은 답변 생성 전에 chat_logs에 저장하고, 답변이 나오면 같은 행에 답변·모델·출처를 기록한다.
    기록 저장이 실패해도 학생에게는 답변을 돌려준다.
    """
    question = _with_previous_question(request)
    log_id: int | None = None
    try:
        log_id = create_chat_log(question=question, session_id=request.session_id)
    except Exception:
        logger.exception("질문 기록 저장 실패")

    deadline = time.monotonic() + settings.VLLM_COLD_START_WAIT_S
    try:
        chain = build_rag_chain(request.model_name)
        retrieval = retrieve(question)
        if asked_back := clarify(retrieval):
            return _clarification(request, log_id, *asked_back)
        while True:
            try:
                result = chain.invoke(
                    {"query": question, "model_name": request.model_name}, retrieval
                )
                break
            except Exception as e:
                # 첫 요청이 Modal 컨테이너를 깨우므로, 켜질 때까지 잠시 다시 시도한다.
                if not _is_cold_start(e) or time.monotonic() >= deadline:
                    raise
                logger.info("LLM 서버가 켜지는 중이라 다시 시도합니다.")
                time.sleep(settings.VLLM_COLD_START_RETRY_INTERVAL_S)
    except Exception as e:
        cold = _is_cold_start(e)
        if cold:
            logger.warning("LLM 서버가 아직 켜지지 않았습니다: %s", e)
        else:
            logger.exception("채팅 처리 중 오류: %s", e)
        if log_id is not None:
            try:
                fail_chat_log(log_id, error=f"{type(e).__name__}: {e}")
            except Exception:
                logger.exception("실패 상태 기록 실패")
        if cold:
            raise HTTPException(status_code=503, detail=COLD_START_DETAIL)
        raise HTTPException(status_code=500, detail="AI 튜터 응답 생성 중 오류가 발생했습니다.")

    sources = [
        SourceDocument(
            content=doc.page_content[:200],  # 요약
            source=doc.metadata.get("url", "unknown"),
            category=doc.metadata.get("source_type"),
        )
        for doc in result.get("source_documents", [])
    ]
    answer = result["result"]

    if log_id is not None:
        try:
            complete_chat_log(
                log_id,
                answer=answer,
                model_name=resolve_model_name(request.model_name),
                sources=[s.source for s in sources],
            )
        except Exception:
            logger.exception("답변 기록 저장 실패")

    return ChatResponse(answer=answer, sources=sources, session_id=request.session_id)


def _clarification(
    request: ChatRequest, log_id: int | None, answer: str, options: list[str]
) -> ChatResponse:
    """학기·분반이 애매하면 LLM을 부르지 않고 선택지와 함께 되묻는다."""
    if log_id is not None:
        try:
            complete_chat_log(log_id, answer=answer, model_name=CLARIFY_MODEL, sources=[])
        except Exception:
            logger.exception("답변 기록 저장 실패")
    return ChatResponse(answer=answer, options=options, session_id=request.session_id)
