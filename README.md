# 🍀 four-leaf-AI

> **Four-Leaf** 프로젝트의 AI 튜터 서비스 파이프라인입니다.  
> 학생 채팅은 로컬 Ollama의 Llama 3.1 8B 모델로 답변하며, Cluade 합성 학습 데이터 생성과 교수자 리포트에 사용합니다. 프로젝트에는 A100 GPU 기반 오픈소스 LLM 파인튜닝 및 모델 서빙 파이프라인도 포함되어 있습니다.

---

## 🏗 기술 스택

| 분류 | 기술 |
|------|------|
| **Chat Model** | Ollama `llama3.1:8b` (로컬 양자화 모델) |
| **Fine-Tuning Base Model** | 오픈소스 Llama-3.1 8b |
| **Data Generation** | Claude Opus 5 (Synthetic Data Generation) |
| **Fine-Tuning** | Unsloth, Hugging Face `trl`, `peft` (QLoRA) |
| **프레임워크** | FastAPI 0.115 (모델 데모 서빙용) |
| **언어** | Python 3.12 |
| **오케스트레이션** | LangChain 0.3 |
| **린터** | ruff |
| **CI** | GitHub Actions |

---

## 🏛 전체 파이프라인 아키텍처

```
1. Data Crawling (knowledge/crawler/)
         |
         ▼ 
1. Data Generation (scripts/)
   원천 텍스트 (규정집 등)
         │
         ▼ (Claude Opus API)
   고품질 Q&A 학습 데이터셋 생성 (JSONL)

2. Model Fine-Tuning (train/)
   베이스 모델 (Llama-3 8B) + 학습 데이터셋
         │
         ▼ (A100 GPU / Unsloth 4bit QLoRA)
   대학 특화 AI 튜터 파인튜닝 모델 (LoRA Weights)

3. Model Serving (app/)
   파인튜닝된 모델
         │
         ▼ (FastAPI / RAG)
   프론트엔드/백엔드와 연동되어 실제 사용자에게 데모 제공
```

---

## 📁 프로젝트 구조

```text
four-leaf-AI/
├── app/
│   ├── main.py               # FastAPI 앱 진입점 (데모 서빙)
│   ├── core/
│   │   └── config.py         # 환경변수 설정
│   ├── models/
│   │   ├── local_llm.py      # 파인튜닝된 로컬 모델 서빙 래퍼
│   │   └── schemas.py        # 요청/응답 스키마
│   ├── chains/
│   │   └── rag_chain.py      # 향후 확장용 RAG 파이프라인
│   └── api/v1/
│       ├── chat.py           # POST /api/v1/chat
│       └── advisor.py        # AI 튜터 라우터
├── data/
│   ├── knu_schedule.txt      # 원본 텍스트 데이터 (Seed Data)
│   └── synthetic_dataset.jsonl # Gemini가 생성한 최종 파인튜닝용 데이터셋
├── scripts/
│   ├── generate_dataset.py   # Gemini API를 이용한 합성 데이터 생성 스크립트
│   └── test_api.py           # Gemini API 연결 및 모델 권한 테스트용 스크립트
├── train/
│   └── finetune.py           # Unsloth 기반 A100 파인튜닝 스크립트 (QLoRA)
├── eval/                     # 모델 평가 스크립트 폴더
├── tests/                    # pytest 폴더
├── requirements.txt
└── ruff.toml                 # lint/format 설정 (ruff check .)
```

---

## 👥 팀원 역할 및 협업 가이드

본 프로젝트는 깃(Git) 충돌을 최소화하고 효율적으로 작업하기 위해 4개의 역할로 나누어 진행합니다.
각 역할별로 주로 수정하는 **'담당 폴더'**가 분리되어 있습니다.

| 역할 | 주 담당 업무 | 주 작업 폴더 및 파일 |
|------|------------|----------------|
| **1. Data Engineer** | 학사 규정, 공지사항 등 원본 데이터 크롤링 및 RAG 지식베이스 구축 | `knowledge/`, `app/chains/` |
| **2. Data Synthesizer** | Gemini를 활용한 고품질 Instruction 데이터셋(Q&A) 합성 및 프롬프트 엔지니어링 | `scripts/`, `data/` |
| **3. LLM Trainer** | A100 서버에서 Unsloth & QLoRA를 활용한 오픈소스 LLM(Llama-3) 파인튜닝 | `train/`, `logs/` |
| **4. MLOps Engineer** | 파인튜닝된 모델의 FastAPI 서빙, 자동화(CI/CD) 파이프라인 구축 및 평가 | `app/`, `eval/`, `.github/`, `Dockerfile` |

> 💡 **협업 규칙 (GitHub Flow)**
> 1. `main` 브랜치에 직접 푸시하지 않고, 각자의 기능 브랜치(`feat/데이터생성` 등)를 생성하여 작업합니다.
> 2. 기능 구현이 완료되면 Pull Request(PR)를 올리고, 다른 팀원의 리뷰를 거친 후 `main`에 병합(Merge)합니다.
> 3. `data/*.jsonl` 이나 파인튜닝된 모델 가중치(`logs/`) 같은 **대용량 파일은 절대 GitHub에 올리지 않고(.gitignore)** 구글 드라이브 등을 통해 공유합니다.

---

## 🚀 시작하기

### 1. 가상환경 및 의존성 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 로컬 채팅 모델 준비 (Mac)

학생 채팅은 Gemini API를 호출하지 않고 Ollama를 통해 로컬 Llama 3.1 8B 양자화 모델을 실행합니다. Ollama를 설치한 뒤 모델을 한 번 받아두세요.

```bash
ollama pull llama3.1:8b
```

기본 연결값은 `.env.example`에 적힌 `OLLAMA_BASE_URL=http://localhost:11434`이며, 기본 컨텍스트 길이는 메모리 사용을 줄이도록 4096 토큰입니다. 모델 파일은 약 4.9GB이고 실행 중에는 추가 메모리가 필요합니다. Ollama가 실행 중인 상태에서 API를 시작하세요. Docker 안에서 API를 실행한다면 `.env`의 `OLLAMA_BASE_URL`을 `http://host.docker.internal:11434`로 설정해야 합니다.

### 2. 학습 데이터 생성 (Gemini API 활용)

프로젝트 루트에 `.env` 파일을 생성하고 Google AI Studio에서 발급받은 API 키를 입력합니다.

```env
# .env 파일
GOOGLE_API_KEY="AIzaSy..."
```

그 후 데이터 생성 스크립트를 실행합니다.

```bash
python scripts/generate_dataset.py
```
> 실행이 완료되면 `data/synthetic_dataset.jsonl` 파일에 파인튜닝용 Q&A 데이터가 생성됩니다.

### 3. 모델 파인튜닝 (A100 서버 환경)

파인튜닝은 로컬 Mac이 아닌 **NVIDIA Ampere 아키텍처 이상(A100 등)의 Linux 서버**에서 진행하는 것을 권장합니다.
서버에서 아래 명령어로 초고속 파인튜닝 라이브러리인 `unsloth`를 추가 설치합니다.

```bash
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
```

그 후 학습 스크립트를 실행합니다.

```bash
python train/finetune.py
```
> 학습이 완료되면 `logs/lora_model` 에 LoRA 가중치가 저장됩니다.

### 4. 모델 서빙 (데모 확인)

학습된 모델을 백엔드에서 통신해 볼 수 있도록 FastAPI 서버를 띄웁니다.

```bash
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs (Swagger UI)
```

---

## 🔄 코드 컨벤션 (Lint)

본 프로젝트는 `ruff`를 통해 엄격한 코드 스타일을 유지합니다.

```bash
# 린트 에러 검사
ruff check .

# 린트 에러 자동 수정
ruff check --fix .
```
# 학생 질문 저장 및 교수자 리포트

AI 서버는 채팅 요청의 학생 질문을 기존 PostgreSQL 데이터베이스의
`student_questions` 테이블에 저장합니다. 연결 설정은 `DB_HOST`, `DB_PORT`, `DB_NAME`,
`DB_USER`, `DB_PASSWORD` 환경 변수로 지정하며, Docker Compose에서는 기존 PostgreSQL
서비스에 연결합니다.

로컬에서 AI 서버를 직접 실행할 때는 `.env.example`을 `.env`로 복사한 뒤
Ollama 및 PostgreSQL 연결 정보를 환경에 맞게 설정합니다. `GOOGLE_API_KEY`는 학생 채팅에는 필요하지 않으며 Gemini 기반 교수자 리포트 등 별도 기능에 사용됩니다. Docker Compose 전체 스택은
`four-leaf-infra/.env.dev`의 설정을 사용합니다.

채팅 요청에 `course_name`을 포함하면 강의별로 분류됩니다. 생략하면 `미지정`으로 저장됩니다.
교수자 리포트는 질문 목록을 직접 전달해도 되고, `student_questions`를 생략하거나 빈 배열로
보내면 해당 강의명으로 저장된 질문을 불러와 요약합니다.

```http
POST /api/v1/tutor/chat
Content-Type: application/json

{
  "message": "자료구조 과제 범위가 어디까지인가요?",
  "course_name": "자료구조"
}
```

```http
POST /api/v1/advisor/report
Content-Type: application/json

{
  "course_name": "자료구조"
}
```

현재 채팅 화면에는 강의 선택 기능이 없으므로, 해당 화면에서 들어온 질문은 `미지정`에 저장됩니다.
