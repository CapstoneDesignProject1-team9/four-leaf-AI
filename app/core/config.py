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

    # Local chat generation through Ollama (M1 Air friendly Q4_K_M quantization)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_NUM_CTX: int = 4096
    OLLAMA_NUM_PREDICT: int = 768

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
    CHROMA_PERSIST_DIR: str = "./knowledge/vectorstores/knu_notice_bge_m3"
    CHROMA_COLLECTION_NAME: str = "knu_notices_bge_m3_v1"
    CHROMA_SYLLABI_PERSIST_DIR: str = "./knowledge/vectorstores/knu_syllabi_bge_m3"
    CHROMA_SYLLABI_COLLECTION_NAME: str = "knu_syllabi_bge_m3_v1"
    CHROMA_EVERYTIME_PERSIST_DIR: str = "./knowledge/vectorstores/everytime_bge_m3"
    CHROMA_EVERYTIME_COLLECTION_NAME: str = "everytime_reviews_bge_m3_v1"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_REVISION: str = "5617a9f61b028005a4858fdac845db406aefb181"
    EMBEDDING_MODEL_CACHE_DIR: str = "./.cache/embedding_models"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_CPU_THREADS: int = 4
    RAG_TOP_K: int = 5

    # PostgreSQL connection used for persisted student questions
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "fourleaf_dev"
    DB_USER: str = "fourleaf"
    DB_PASSWORD: str = "devpassword"

    # ─────────────────────────────────────────
    # Backend API (AI → Spring Boot 호출 시)
    # ─────────────────────────────────────────
    BACKEND_BASE_URL: str = "http://backend:8080"

    # LangChain Tracing (선택)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "four-leaf-ai"


settings = Settings()
