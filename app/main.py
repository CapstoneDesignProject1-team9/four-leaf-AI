from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import advisor, chat, health
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 lifecycle 훅"""
    # 시작: 벡터스토어 초기화
    from app.chains.rag_chain import init_vectorstore

    await init_vectorstore()
    yield
    # 종료: 리소스 정리 (필요시)


app = FastAPI(
    title="Four-Leaf AI Tutor",
    description="대학생활 및 진로 상담 AI 튜터 서비스 (LangChain + HyperCLOVA X)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (Frontend → AI 직접 호출 또는 Nginx 프록시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api/v1/tutor", tags=["tutor"])
app.include_router(advisor.router, prefix="/api/v1/advisor", tags=["advisor"])
