import logging

from fastapi import APIRouter, HTTPException

from app.chains.rag_chain import build_rag_chain, resolve_model_name
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from app.storage.chat_logs import complete_chat_log, create_chat_log, fail_chat_log

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    AI 튜터 채팅 엔드포인트

    RAG 파이프라인:
    1. BGE-M3로 사용자 질문 임베딩
    2. 키워드 라우팅 없이 공지·강의계획서 전체에서 cosine 거리 상위 5개 청크 검색
    3. vLLM으로 서빙하는 Llama-3.1-8B-Instruct(temperature 0)가 검색된 공지와 강의계획서를 근거로 답변 생성

    질문은 답변 생성 전에 chat_logs에 저장하고, 답변이 나오면 같은 행에 답변·모델·출처를 기록한다.
    기록 저장이 실패해도 학생에게는 답변을 돌려준다.
    """
    log_id: int | None = None
    try:
        log_id = create_chat_log(question=request.message, session_id=request.session_id)
    except Exception:
        logger.exception("질문 기록 저장 실패")

    try:
        chain = build_rag_chain(request.model_name)
        result = chain.invoke({"query": request.message, "model_name": request.model_name})
    except Exception as e:
        logger.exception("채팅 처리 중 오류: %s", e)
        if log_id is not None:
            try:
                fail_chat_log(log_id, error=f"{type(e).__name__}: {e}")
            except Exception:
                logger.exception("실패 상태 기록 실패")
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
