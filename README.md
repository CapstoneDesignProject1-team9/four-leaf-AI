# 🍀 four-leaf-AI

> **Four-Leaf** 프로젝트의 AI 튜터 서비스 파이프라인입니다.  
> 대학생들의 **대학생활 상담**과 **진로 상담**을 제공하는 대화형 AI 튜터를 만들기 위해, **Gemini를 활용한 합성 데이터(Synthetic Data) 생성**부터 **A100 GPU 기반의 오픈소스 LLM(Llama-3) 파인튜닝**, 그리고 결과 모델 서빙까지의 전체 파이프라인을 포함하고 있습니다.

---

## 🏗 기술 스택

| 분류 | 기술 |
|------|------|
| **Base Model** | 오픈소스 Llama-3 (Bllossom-8B) |
| **Data Generation** | Google Gemini 1.5 Flash (Synthetic Data Generation) |
| **Fine-Tuning** | Unsloth, Hugging Face `trl`, `peft` (QLoRA) |
| **프레임워크** | FastAPI 0.115 (모델 데모 서빙용) |
| **언어** | Python 3.12 |
| **오케스트레이션** | LangChain 0.3 |
| **린터** | ruff |
| **CI** | GitHub Actions |

---

## 🏛 전체 파이프라인 아키텍처

```
1. Data Generation (scripts/)
   원천 텍스트 (규정집 등)
         │
         ▼ (Google Gemini API)
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

## 🚀 시작하기

### 1. 가상환경 및 의존성 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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