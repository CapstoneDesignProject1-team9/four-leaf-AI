"""
Modal 서버리스 배포 엔트리포인트

FastAPI 앱을 Modal에서 서버리스로 실행합니다.
- Scale-to-Zero: 요청 없으면 비용 0원
- 요청 시 자동으로 컨테이너 시작 (콜드 스타트 ~수 초)
- ChromaDB + 임베딩 모델은 Modal Volume에 캐시

배포:
  modal deploy modal_app.py

로컬 테스트:
  modal serve modal_app.py
"""

import modal

# ─────────────────────────────────────────
# Modal 앱 정의
# ─────────────────────────────────────────
app = modal.App("four-leaf-ai")

# ─────────────────────────────────────────
# 컨테이너 이미지 정의 (Dockerfile 대체)
# ─────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libgomp1")
    .pip_install(
        # FastAPI Core
        "fastapi==0.115.0",
        "uvicorn[standard]==0.30.6",
        "python-dotenv==1.0.1",
        "pydantic-settings==2.5.2",
        "httpx==0.27.2",
        # LangChain
        "langchain==0.3.0",
        "langchain-core==0.3.0",
        "langchain-community==0.3.0",
        "langchain-text-splitters==0.3.0",
        # Vector Store & Embeddings (RAG)
        "langchain-chroma==0.1.4",
        "chromadb==0.5.11",
        "sentence-transformers==3.1.1",
        # Document Loaders
        "pypdf==4.3.1",
        "python-docx==1.1.2",
        # Google Gemini
        "langchain-google-genai>=1.0.8",
    )
    # 소스 코드를 컨테이너에 마운트 (기존 modal.Mount 대체)
    .add_local_dir("app", remote_path="/root/app")
    .add_local_dir("knowledge", remote_path="/root/knowledge")
    .add_local_dir("data", remote_path="/root/data")
)

# ─────────────────────────────────────────
# 영속 볼륨: ChromaDB 인덱스 + 임베딩 모델 캐시
# ─────────────────────────────────────────
# Modal Volume은 컨테이너가 종료되어도 데이터가 유지됩니다.
# → ChromaDB 인덱스를 매번 다시 빌드하지 않아도 됩니다.
chroma_volume = modal.Volume.from_name("four-leaf-chroma-db", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("four-leaf-model-cache", create_if_missing=True)

# ─────────────────────────────────────────
# FastAPI 앱을 Modal ASGI 앱으로 서빙
# ─────────────────────────────────────────
@app.function(
    image=image,
    # 볼륨 마운트: ChromaDB와 HuggingFace 모델 캐시
    volumes={
        "/root/chroma_db": chroma_volume,
        "/root/.cache/huggingface": model_cache_volume,
    },
    # 시크릿: Modal Dashboard에서 설정 (Settings → Secrets)
    secrets=[modal.Secret.from_name("four-leaf-ai-secrets")],
    # 리소스 설정
    cpu=2.0,           # CPU 코어 (필요 시 조정)
    memory=2048,       # 메모리 2GB (임베딩 모델 로딩용)
    # 타임아웃 & 동시성
    timeout=300,       # 요청 최대 5분 (RAG 응답 생성 시간 고려)
    # 최소 0개 → Scale-to-Zero (비용 0원)
    # 트래픽이 있을 때만 컨테이너가 올라옵니다.
    min_containers=0,
    max_containers=5,  # 동시 최대 5개 (필요 시 조정)
    # 콜드 스타트 최적화: 10분간 요청 없으면 컨테이너 종료
    scaledown_window=600,
)
@modal.asgi_app()
def fastapi_app():
    """기존 FastAPI 앱을 그대로 Modal에서 서빙"""
    import os
    import sys

    # 작업 디렉토리 설정 (기존 코드의 상대 경로 호환)
    os.chdir("/root")
    sys.path.insert(0, "/root")

    # 환경 변수 설정 (Modal Volume 경로에 맞게)
    os.environ.setdefault("CHROMA_PERSIST_DIR", "/root/chroma_db")
    os.environ.setdefault("KNOWLEDGE_BASE_DIR", "/root/knowledge")
    os.environ.setdefault("APP_ENV", "production")

    # HuggingFace 모델 캐시 디렉토리 (Volume에 저장 → 재시작 시 다운로드 불필요)
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")

    from app.main import app as _app

    return _app
