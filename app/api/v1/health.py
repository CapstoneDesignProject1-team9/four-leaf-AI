from fastapi import APIRouter

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """헬스체크 엔드포인트 (CD 파이프라인용)"""
    return HealthResponse(
        status="ok",
        service="four-leaf-ai",
        version="1.0.0",
    )
