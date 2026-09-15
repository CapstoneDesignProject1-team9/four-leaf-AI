from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────
    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Clova Studio API 엔드포인트

    # 사용 모델 (HyperCLOVA X)
    # HCX-DASH-001: 빠른 응답 (추천)
    # HCX-003: 고성능

    # ─────────────────────────────────────────
    # 앱 설정
    # ─────────────────────────────────────────
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS (Frontend URL)
    ALLOWED_ORIGINS: list[str] = ["http://localhost", "http://localhost:3000"]

    # ─────────────────────────────────────────
    # Vector Store (ChromaDB)
    # ─────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"  # 로컬 영속 저장 경로
    KNOWLEDGE_BASE_DIR: str = "./knowledge"  # RAG용 문서 디렉토리

    # ─────────────────────────────────────────
    # Backend API (AI → Spring Boot 호출 시)
    # ─────────────────────────────────────────
    BACKEND_BASE_URL: str = "http://backend:8080"

    # LangChain Tracing (선택)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "four-leaf-ai"


settings = Settings()
