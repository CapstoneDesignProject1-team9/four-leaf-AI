import logging

from fastapi import APIRouter, HTTPException

from app.chains.rag_chain import build_rag_chain
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    AI 튜터 채팅 엔드포인트

    RAG 파이프라인:
    1. 사용자 질문 임베딩
    2. ChromaDB에서 유사 문서 검색 (top-3)
    3. HyperCLOVA X로 답변 생성
    """
    try:
        chain = build_rag_chain()
        result = chain.invoke({"query": request.message})

        # 참고 문서 파싱
        sources = []
        for doc in result.get("source_documents", []):
            sources.append(SourceDocument(
                content=doc.page_content[:200],  # 요약
                source=doc.metadata.get("source", "unknown"),
                category=doc.metadata.get("category"),
            ))

        return ChatResponse(
            answer=result["result"],
            sources=sources,
            session_id=request.session_id,
        )

    except Exception as e:
        logger.exception("채팅 처리 중 오류: %s", e)
        raise HTTPException(status_code=500, detail="AI 튜터 응답 생성 중 오류가 발생했습니다.")
