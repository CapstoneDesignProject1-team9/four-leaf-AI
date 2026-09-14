# 🍀 four-leaf-AI

> **Four-Leaf** 프로젝트의 AI 튜터 서비스입니다.  
> NAVER HyperCLOVA X와 LangChain RAG 파이프라인을 활용하여  
> 대학생들의 **대학생활 상담**과 **진로 상담**을 제공하는 대화형 AI 튜터입니다.

---

## 🏗 기술 스택

| 분류 | 기술 |
|------|------|
| 프레임워크 | FastAPI 0.115 |
| 언어 | Python 3.12 |
| LLM | NAVER HyperCLOVA X (HCX-DASH-001) |
| AI 오케스트레이션 | LangChain 0.3 |
| 벡터 DB | ChromaDB 0.5 |
| 임베딩 | KR-ELECTRA (한국어 특화, sentence-transformers) |
| 린터 | ruff |
| 컨테이너 | Docker (python:3.12-slim) |
| CI | GitHub Actions |

---

## 🏛 아키텍처

### 전체 흐름

```
사용자 질문 (POST /api/v1/chat)
        │
        ▼
  ┌─────────────────────────────────┐
  │         RAG 파이프라인           │
  │                                  │
  │  1. 질문 임베딩 (KR-ELECTRA)    │
  │         │                        │
  │         ▼                        │
  │  2. ChromaDB 유사 문서 검색      │
  │     (knowledge/ 폴더 인덱싱)    │
  │         │                        │
  │         ▼                        │
  │  3. 시스템 프롬프트 조합         │
  │     + 검색 문서 (context)        │
  │     + 사용자 질문                │
  └─────────────────────────────────┘
        │
        ▼
  HyperCLOVA X (Clova Studio API)
        │
        ▼
  답변 + 참고 문서 반환
```

### RAG (Retrieval Augmented Generation) 이란?

```
일반 LLM:   질문 → 모델 내 학습 데이터만으로 답변 (환각 발생 가능)

RAG:        질문 → [관련 문서 검색] → 검색 결과 + 질문 → 답변
                         ↑
                   우리 학교 자료
                   (PDF, TXT 등)
```

> RAG를 사용하면 **우리 학교/서비스에 특화된 정보**로 답변할 수 있습니다.

### Docker 이미지 구조 (멀티스테이지)

```
Stage 1: python:3.12-slim + build-essential
  → pip install -r requirements.txt

Stage 2: python:3.12-slim (경량화)
  → non-root 사용자 실행
  → uvicorn 서버 구동 (포트 8000)
```

---

## 📁 프로젝트 구조

```
four-leaf-AI/
├── .github/
│   └── workflows/
│       ├── ci.yml            # ruff lint + pytest (mock 기반)
│       └── trigger-cd.yml    # main push 시 ECR push → infra dispatch
├── app/
│   ├── main.py               # FastAPI 앱 진입점
│   ├── core/
│   │   └── config.py         # 환경변수 설정 (Clova API 키 등)
│   ├── models/
│   │   ├── clova_llm.py      # HyperCLOVA X LangChain 커스텀 래퍼
│   │   └── schemas.py        # 요청/응답 Pydantic 스키마
│   ├── chains/
│   │   └── rag_chain.py      # RAG 파이프라인 (ChromaDB + LangChain)
│   └── api/v1/
│       ├── chat.py           # POST /api/v1/chat
│       └── health.py         # GET /health
├── knowledge/                # 📚 RAG 지식베이스 문서 폴더
│   └── README.md             # 문서 추가 가이드
├── tests/
│   └── test_health.py        # pytest (Clova API mock)
├── Dockerfile                # 프로덕션
├── requirements.txt
└── ruff.toml                 # lint/format 설정
```

---

## 🌐 API 엔드포인트

### `GET /health`

헬스체크 (CD 파이프라인 자동 호출)

```json
{
  "status": "ok",
  "service": "four-leaf-ai",
  "version": "1.0.0"
}
```

### `POST /api/v1/chat`

AI 튜터 대화

**Request:**
```json
{
  "message": "복수전공 신청은 어떻게 해야 하나요?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "복수전공 신청은 2학년 1학기부터 가능하며...",
  "sources": [
    {
      "content": "복수전공 신청 관련 학사 안내...",
      "source": "학사안내_2024.pdf",
      "category": "학사정보"
    }
  ],
  "session_id": "optional-session-id"
}
```

---

## 📚 지식베이스 구성 (RAG)

`knowledge/` 폴더에 문서를 추가하면 앱 시작 시 자동으로 ChromaDB에 인덱싱됩니다.

```
knowledge/
├── 학사안내_2024.pdf         ← 학사 정보, 수강신청 등
├── 취업지원센터_FAQ.txt      ← 자주 묻는 취업 질문
├── 전공별_진로안내.pdf        ← 전공별 진로 정보
└── 장학금_안내.txt           ← 장학금 신청 안내
```

> 문서 추가/변경 후 `chroma_db/` 폴더를 삭제하고 앱을 재시작하면 재인덱싱됩니다.

---

## 🔄 CI/CD 파이프라인

### CI (`ci.yml`) — PR + main push

```
pip install -r requirements.txt
  → ruff check . (lint)
  → ruff format --check . (포매팅)
  → pytest tests/ (Clova API는 mock 처리 → 실제 키 없이 실행 가능)
```

### CD (`trigger-cd.yml`) — main push만

```
CI 통과
  → Docker image build
  → AWS ECR push (SHA tag + latest)
  → repository_dispatch → four-leaf-infra
    → EC2에 자동 배포
```

---

## 💻 로컬 개발

### 가상환경 설정

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 환경변수 설정

```bash
cp .env.example .env   # 없는 경우 직접 생성
```

`.env` 파일:
```env
CLOVA_API_KEY=발급받은_NCP_API_KEY
CLOVA_API_KEY_PRIMARY_VAL=발급받은_Clova_Studio_API_KEY
CLOVA_MODEL=HCX-DASH-001
```

### 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/health
# http://localhost:8000/docs  (Swagger UI)
```

### Docker로 실행 (전체 스택)

```bash
cd ../four-leaf-infra
docker compose -f docker-compose.dev.yml up -d --build
```

---

## 🔑 NAVER Clova Studio API 키 발급

1. [NAVER Cloud Platform](https://www.ncloud.com) 가입
2. **AI·NAVER API** → **CLOVA Studio** 신청
3. 테스트 앱 생성 → API 키 발급
4. 사용 모델:
   - `HCX-DASH-001` : 빠른 응답 속도 (권장)
   - `HCX-003` : 높은 성능

---

## 🔐 GitHub Secrets (CD 사용 시)

| Secret | 설명 |
|--------|------|
| `AWS_ACCESS_KEY_ID` | ECR push 권한 IAM 키 |
| `AWS_SECRET_ACCESS_KEY` | IAM 시크릿 |
| `INFRA_DISPATCH_TOKEN` | infra 레포 트리거용 PAT |