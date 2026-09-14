# ─────────────────────────────────────────
# Stage 1: 의존성 설치
# ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# 시스템 패키지 (chromadb, sentence-transformers 빌드 의존)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (레이어 캐시)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────
# Stage 2: 실행 이미지
# ─────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# 런타임 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 보안: non-root 사용자
RUN useradd -m -u 1000 appuser

# 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 & 지식베이스 디렉토리
COPY --chown=appuser:appuser . .
RUN mkdir -p chroma_db knowledge && chown -R appuser:appuser chroma_db knowledge

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
