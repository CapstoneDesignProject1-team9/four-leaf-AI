# 🍀 four-leaf-AI

Four-Leaf 프로젝트의 AI 서비스 저장소입니다. 경북대학교 컴퓨터학부 공지사항을 수집·가공하고, **BGE-M3와 ChromaDB로 관련 문서를 검색한 뒤 Gemini로 근거 기반 답변을 생성**합니다.

대학생활·진로 상담으로의 확장을 목표로 하며, 합성 학습 데이터 생성과 오픈소스 모델 파인튜닝 스크립트도 포함합니다. **현재 공지 RAG의 답변 모델은 Gemini**이며, 파인튜닝 모델로 자동 연결되는 구조는 아닙니다.

## 1. 구현 범위 및 기술 스택

| 영역 | 구현 및 사용 기술 |
| --- | --- |
| 공지 수집 | requests, BeautifulSoup / 본문·첨부 수집, 증분 업데이트, 중단 후 재개 |
| 텍스트 추출 | pypdf, PyMuPDF, pytesseract, Pillow, olefile, ZIP/XML |
| 전처리·청킹 | 출처별 문서 생성, 품질 보고서, 청크와 메타데이터 생성 |
| 임베딩·검색 | BAAI/bge-m3, sentence-transformers, ChromaDB 1.0.20 |
| 답변·API | Gemini API, LangChain, FastAPI, Uvicorn |
| 학습 개발 | Bllossom 계열 모델, Unsloth, TRL, PEFT / QLoRA |
| 실행·개발 환경 | Python 3.12 권장, pytest, Ruff 설정, Docker, GitHub Actions 워크플로 |

로컬 웹 연동에서 답변 성공을 확인했습니다. 전체 데이터 검색 품질·동시 요청·배포 성능은 별도로 검증해야 합니다. 교수자 분석, 합성 데이터 품질 평가, 파인튜닝 모델 서빙 연동도 별도 검증 대상입니다.

`session_id` 필드의 존재만으로 이전 대화를 기억하는 멀티턴 기능이 구현된 것은 아닙니다.

## 2. 전체 흐름

```text
공지 수집 → 원본 JSONL → 전처리 → 문서 JSONL → 청킹 JSONL
                                               ↓
                                         BGE-M3 임베딩
                                               ↓
                                            ChromaDB

학생 질문 → 질문 임베딩 → 관련 청크 검색 → Gemini → 답변 + 출처
```

합성 데이터와 학습은 별도 흐름입니다.

```text
원천 텍스트 → Gemini Q&A 생성 → 데이터 검토 → QLoRA 학습
                                              ↓
                                         LoRA 가중치
                                              ↓
                                    평가 및 서빙 연동 검증
```

## 3. 프로젝트 구조

```text
four-leaf-AI/
├── app/
│   ├── main.py                       # FastAPI 진입점
│   ├── core/config.py                # 환경변수 및 RAG 설정
│   ├── chains/rag_chain.py           # BGE-M3 검색 + Gemini 답변
│   ├── models/                       # 요청·응답 스키마, 모델 관련 코드
│   └── api/v1/
│       ├── chat.py                   # 학생용 채팅
│       ├── advisor.py                # 교수자 보고서
│       └── health.py                 # 상태 확인
├── knowledge/
│   ├── crawlers/crawl_knu_notices.py
│   ├── preprocessing/
│   │   ├── preprocess_notices.py
│   │   └── chunk_notices.py
│   ├── embeddings/
│   │   ├── embed_notices.py
│   │   └── requirements-embedding.txt
│   ├── raw/notices/                  # 원본 JSONL·첨부파일
│   ├── processed/notices/            # 전처리·청킹 결과와 보고서
│   └── vectorstores/knu_bge_m3/       # 생성되는 로컬 ChromaDB
├── .cache/embedding_models/          # 다운로드되는 모델 캐시
├── data/                             # 합성 데이터 원천 및 생성 결과
├── scripts/generate_dataset.py       # 합성 학습 데이터 생성
├── train/finetune.py                  # 파인튜닝 스크립트
├── eval/                             # 평가 관련 영역
├── tests/
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── ruff.toml
```

생성 데이터·모델·가상환경은 저장소를 내려받는 것만으로 제공되지 않습니다.

## 4. 로컬 실행 준비

아래 명령어는 **Windows PowerShell, four-leaf-AI 프로젝트 루트 기준**입니다.

### 가상환경 및 패키지

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt -r knowledge/embeddings/requirements-embedding.txt
.\.venv312\Scripts\python.exe -m pip install requests beautifulsoup4 pillow pytesseract pypdf pymupdf olefile
.\.venv312\Scripts\python.exe -m pip check
```

이미 구성한 `.venv312`는 다시 만들 필요가 없습니다. 가상환경 활성화 없이 해당 Python을 직접 실행하는 방식입니다. 전처리·청킹은 추가 패키지가 필요 없지만, 크롤링·임베딩·서버에는 패키지가 필요합니다.

OCR에는 Python 패키지 외에 **Tesseract 실행 프로그램과 한국어·영어 언어 데이터**가 필요합니다. HWPX는 표준 라이브러리로 처리하고 binary HWP는 `olefile`을 사용합니다. 암호화·배포용 HWP와 HWP 3.x는 지원하지 않습니다.

학습용 GPU 환경은 로컬 RAG 환경과 분리해 관리합니다. 패키지 버전은 실제 의존성 파일을 확인합니다.

### 환경변수

프로젝트 루트에 `.env`를 생성합니다.

```dotenv
GOOGLE_API_KEY=발급받은_API_키
GEMINI_MODEL=계정에서_호출_가능한_모델_ID
CHROMA_PERSIST_DIR=./knowledge/vectorstores/knu_bge_m3
CHROMA_COLLECTION_NAME=knu_notices_bge_m3_v1
EMBEDDING_MODEL_CACHE_DIR=./.cache/embedding_models
EMBEDDING_DEVICE=cpu
RAG_TOP_K=5
LANGCHAIN_TRACING_V2=false
```

API 키와 모델 ID의 안내 문구는 실제 값으로 교체해야 합니다. `GEMINI_MODEL`은 프로젝트 이름이 아닙니다. 코드의 기본 모델 ID가 현재 계정에서 사용 가능한지는 별도 확인이 필요합니다. 추적 기능을 사용하지 않으면 `LANGCHAIN_API_KEY`는 필요하지 않습니다.

`.env`와 API 키는 GitHub에 올리지 않습니다. API 비용·할당량은 사용하는 계정과 모델을 확인합니다.

## 5. 공지 데이터 구축

### ① 크롤링

최신 공지부터 2022년 1월 1일까지, 기존 결과를 유지하면서 없는 공지를 수집합니다.

```powershell
.\.venv312\Scripts\python.exe knowledge/crawlers/crawl_knu_notices.py --resume-full --min-date 2022-01-01
```

일상적인 신규·최근 수정 공지 확인:

```powershell
.\.venv312\Scripts\python.exe knowledge/crawlers/crawl_knu_notices.py --min-date 2022-01-01
```

- `--resume-full`은 저장된 공지를 건너뛰므로 기존 공지 전체의 재검사 옵션이 아닙니다.
- 일반 업데이트는 최근 페이지의 기존 공지만 재검사하므로 모든 과거 수정 사항을 탐지하지는 않습니다.
- **`--full`은 기존 원본 JSONL과 첨부파일을 삭제하고 재수집합니다.** 초기화가 필요할 때만 백업 후 사용합니다.

### ② 전처리

```powershell
.\.venv312\Scripts\python.exe knowledge/preprocessing/preprocess_notices.py
```

본문·이미지 OCR·첨부 텍스트를 출처별 문서로 정리합니다. `notice_without_text`는 세 종류의 텍스트가 모두 비어 문서를 만들지 못한 공지입니다. 해당 원본과 추출 상태를 확인합니다.

### ③ 청킹

```powershell
.\.venv312\Scripts\python.exe knowledge/preprocessing/chunk_notices.py
```

기본 청크 크기는 **700자**, 겹침 목표는 **100자**입니다. 토큰 수 기준이 아니며 제목·메타데이터와 문장 경계에 따라 실제 크기·겹침이 달라집니다.

전처리·청킹은 기존 결과와 보고서를 **자동으로 교체**하고 입력은 변경하지 않습니다. `--overwrite` 옵션은 제거되었습니다. 기본 경로는 프로젝트 위치 기준이므로 `종프` 폴더를 열고 각 파일에서 F5로 실행해도 됩니다. 직접 지정한 `--input`·`--output`의 상대 경로는 실행 폴더 기준입니다.

### ④ 입력 검사 및 임베딩

```powershell
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_notices.py check
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_notices.py index
```

- `check`는 입력 청크 검사이며 실제 임베딩·검색 성공 검증은 아닙니다.
- 기본값은 CPU, 배치 8, 정규화된 1024차원 dense 벡터와 cosine 거리입니다.
- 최초 실행에는 모델 다운로드·로딩 시간이 추가됩니다.
- 재실행 시 기존 임베딩을 재사용합니다. 중단 후 같은 명령으로 실행하면 완료된 배치를 활용합니다.
- 입력은 전체 스냅샷으로 취급합니다. 성공 후 입력에 없는 기존 청크는 비활성화하며 물리적으로 삭제하지 않습니다.
- 색인 중 입력 파일을 교체하거나 같은 DB에 다른 색인을 실행하지 않습니다. AI 서버도 중지한 상태에서 갱신하고 완료 후 재시작합니다.

메모리 부담을 줄이려면 배치를 낮출 수 있습니다.

```powershell
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_notices.py index --batch-size 4
```

### ⑤ 검색 확인

```powershell
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_notices.py search "수강정정 신청 방법은?"
```

관련 청크·출처를 확인하는 검색 테스트이며 Gemini 답변을 생성하지 않습니다. 수집 문서에 답이 있는 질문으로 원문과 비교합니다. 검색 거리는 정답 확률이 아닙니다.

### 저장 결과

| 단계 | 프로젝트 기준 경로 |
| --- | --- |
| 원본 | `knowledge/raw/notices/knu_notices_all.jsonl` |
| 첨부파일 | `knowledge/raw/notices/attachments/` |
| 전처리 | `knowledge/processed/notices/knu_documents.jsonl` |
| 청킹 | `knowledge/processed/notices/knu_notice_chunks.jsonl` |
| 보고서 | 각 전처리·청킹 출력 옆의 `*.report.json` |
| 벡터 DB | `knowledge/vectorstores/knu_bge_m3/` |
| 모델 캐시 | `.cache/embedding_models/` |

## 6. 서버 및 답변 테스트

완료된 ChromaDB, 같은 revision의 BGE-M3 모델 캐시, Gemini 설정을 준비한 뒤 **프로젝트 루트에서** 실행합니다.

```powershell
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- API 문서: <http://127.0.0.1:8000/docs>
- 상태 확인: <http://127.0.0.1:8000/health>
- AI 직접 호출: `POST /api/v1/tutor/chat`
- 교수자 보고서: `POST /api/v1/advisor/report` (별도 기능 검증 대상)

API 문서에서 채팅 요청을 실행합니다.

```json
{
  "message": "수강정정 신청 방법을 알려줘"
}
```

응답의 `answer`와 `sources`를 확인합니다. `sources`는 문자열 배열이 아니라 `content`, `source`, `category`를 가진 객체 배열입니다. 상태 확인 성공만으로 Gemini 응답 성공을 보장하지는 않습니다.

전체 웹 테스트는 별도 저장소 `four-leaf-frontend`, `four-leaf-backend`, `four-leaf-infra`가 필요합니다. Docker Compose 실행·환경변수·포트·볼륨은 infra 저장소 설정을 따릅니다. 웹 프록시 경로와 AI 직접 호출 경로는 다를 수 있습니다.

## 7. 팀원에게 DB 공유하기

1. 임베딩 완료를 확인하고 해당 DB를 사용하는 프로그램을 종료합니다.
2. `knowledge/vectorstores/knu_bge_m3` 폴더 전체를 공유합니다. ZIP 압축을 권장하지만 필수는 아닙니다.
3. 팀원은 같은 프로젝트 위치에 복원합니다. `chroma.sqlite3`뿐 아니라 하위 폴더도 모두 필요합니다.
4. ChromaDB 버전·컬렉션·모델 revision을 맞추고 검색 테스트를 수행합니다.

**DB와 모델 캐시는 별개입니다.** 현재 서버는 로컬 BGE-M3 캐시를 요구합니다. 캐시를 함께 전달받거나, 인터넷 연결 상태에서 위 `search` 명령으로 같은 revision의 모델을 내려받은 뒤 서버를 시작합니다. 문서 임베딩을 다시 계산할 필요는 없습니다.

데이터 기준일·청크 수·모델 revision·패키지 버전·검증 결과를 함께 기록합니다. Google Drive는 전달용으로 사용하며, 동기화 폴더의 DB를 여러 사람이 동시에 열지 않습니다.

## 8. 합성 데이터 및 파인튜닝

현재 Gemini 기반 공지 RAG 실행에 필수인 단계는 아닙니다.

### 합성 데이터 생성

```powershell
.\.venv312\Scripts\python.exe scripts/generate_dataset.py
```

- 입력: `data/knu_schedule.txt`
- 출력: `data/synthetic_dataset.jsonl` (추가 저장)
- 현재 생성 스크립트는 모델 ID를 코드에 직접 지정합니다. `.env`의 `GEMINI_MODEL`만 바꿔서는 이 스크립트의 모델이 바뀌지 않습니다. 실행 전 모델 ID와 API 접근 가능 여부를 확인합니다.
- 생성 결과는 정답으로 간주하지 않고 원문 근거·중복·형식 오류를 검토합니다.

### 파인튜닝

별도 Linux/NVIDIA GPU 환경에서 CUDA·PyTorch·Unsloth·TRL 호환성을 맞춘 뒤 실행합니다. 이 README의 로컬 설치만으로 GPU 학습 환경이 완성되는 것은 아닙니다.

```bash
python train/finetune.py
```

입력은 `data/synthetic_dataset.jsonl`, LoRA 가중치 저장 위치는 `logs/lora_model`입니다. 모델 평가와 Modal 등 추론 서비스 연동은 별도 검증합니다.

## 9. 팀원 역할 및 협업

| 역할 | 담당 업무 | 주요 영역 |
| --- | --- | --- |
| Data Engineer | 수집·전처리·청킹·RAG 지식베이스 | `knowledge/`, `app/chains/` |
| Data Synthesizer | 합성 데이터·프롬프트·품질 관리 | `scripts/`, `data/` |
| LLM Trainer | 파인튜닝·학습 결과 관리 | `train/`, `logs/` |
| MLOps Engineer | 서빙·배포·평가·자동화 | `app/`, `eval/`, `.github/`, `Dockerfile` |

- 기능 브랜치에서 작업하고 PR 리뷰 후 `main`에 병합합니다.
- 공통 영역인 `app/chains/`, 설정·의존성 변경은 담당자끼리 먼저 조율합니다.
- 가상환경, `.env`, 원본·가공 데이터, ChromaDB, 모델 캐시·가중치는 Git에 올리지 않습니다. 커밋 전 `.gitignore`와 스테이징 목록을 확인합니다.
- 전체 수집 → 검색 → 웹 답변을 먼저 검증하고 반복 작업 자동화와 운영 문서를 보강합니다.

## 10. 점검 및 문제 해결

| 증상 | 확인 사항 |
| --- | --- |
| 입력 파일을 찾지 못함 | 이전 단계 결과 존재 여부, 직접 지정한 입력 경로 |
| `notice_without_text` | 원본의 `content`, `ocr_text`, 첨부 `text` 및 추출 상태 |
| 모듈을 찾지 못함 | 패키지를 설치한 환경과 실행 Python 일치 여부 |
| 서버 시작 시 DB·캐시 오류 | 색인 완료 여부, DB 위치, 캐시·모델 revision |
| 질문 후 지연·오류 | 서버 로그의 모델 로딩·Gemini 권한·할당량·네트워크 오류 |
| 팀원 검색 결과가 다름 | DB 버전·활성 청크·모델 revision·검색 설정 |

테스트 실행:

```powershell
.\.venv312\Scripts\python.exe -m pytest
```

Ruff는 별도 설치한 개발 환경에서 사용합니다.

```powershell
.\.venv312\Scripts\python.exe -m pip install ruff
.\.venv312\Scripts\python.exe -m ruff check .
```

테스트 통과 여부와 전체 RAG 품질은 별개입니다. 수집 기준일·문서 수·청크 수·검색 근거·답변 정확성·소요 시간·최대 메모리를 기록해 검증합니다.

> 하위 문서 일부에는 과거 ChromaDB 0.5.x 분리 환경이나 KR-ELECTRA 연결 안내가 남아 있습니다. 현재 실행은 실제 코드·의존성 파일과 이 루트 README를 기준으로 확인하세요.
