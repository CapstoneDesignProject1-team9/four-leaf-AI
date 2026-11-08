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
    # 교수자 어드바이저 답변에 사용. 서비스 중인 모델명으로 .env에서 지정하세요.
    GEMINI_MODEL: str = "gemini-2.5-flash"
    # 교수자 질문 하나에 Gemini로 보낼 최근 학생 질문의 최대 개수
    ADVISOR_MAX_QUESTIONS: int = 800

    # RAG 답변 생성: vLLM OpenAI 호환 서버 (bf16, temperature 0)
    # VLLM_BASE_URL에는 /v1을 붙이지 않습니다. (예: http://localhost:8001, http://vllm:8000)
    VLLM_BASE_URL: str = "http://localhost:8001"
    VLLM_BASE_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    VLLM_LORA_ADAPTERS: str = '{"student-lora":"student-lora","professor-lora":"professor-lora"}'
    # 요청에 model_name이 없을 때 쓸 모델. 비우면 VLLM_BASE_MODEL(RAG만), "student-lora"면 LoRA+RAG.
    VLLM_DEFAULT_MODEL: str = ""
    VLLM_API_KEY: str = "EMPTY"
    # Modal 프록시 인증 토큰(Modal 서버 사용 시). 둘 다 설정하면 Modal-Key/Modal-Secret 헤더로 보냅니다.
    VLLM_PROXY_KEY: str = ""
    VLLM_PROXY_SECRET: str = ""
    VLLM_MAX_TOKENS: int = 768
    VLLM_TIMEOUT_S: float = 180.0
    # Modal이 꺼져 있다가 켜지는 동안('no upstreams available') 다시 시도할 최대 시간(초).
    # nginx /api/ 대기 시간(60초)보다 짧아야 합니다.
    VLLM_COLD_START_WAIT_S: float = 40.0
    VLLM_COLD_START_RETRY_INTERVAL_S: float = 5.0

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
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_REVISION: str = "5617a9f61b028005a4858fdac845db406aefb181"
    EMBEDDING_MODEL_CACHE_DIR: str = "./.cache/embedding_models"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_CPU_THREADS: int = 4
    RAG_TOP_K: int = 5

    # PostgreSQL: 학생 질문·응답 기록(chat_logs)
    # 기록 보관 기간(개월). 지난 기록은 자동 삭제합니다.
    CHAT_LOG_RETENTION_MONTHS: int = 4
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
