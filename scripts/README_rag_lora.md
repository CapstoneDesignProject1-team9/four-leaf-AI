# RAG + 학생용 LoRA 질문 실행

`ask_rag_lora.py`는 새 질문을 Chroma에서 검색하고 그 문서를 학생용 LoRA에 전달하는 단일 질문 CLI다. 기본값은 기존 사용법과 같은 로컬 Transformers/PEFT 실행이며, `--backend vllm`으로 vLLM 서버를 호출한다. Ollama는 사용하지 않는다. FastAPI 서비스나 기존 평가 스크립트를 변경하지 않는다.

## 공통 준비

저장소 루트에서 실행한다. 검색용 Conda 환경 `embed`에는 프로젝트의 공지 임베딩 의존성(ChromaDB, sentence-transformers 등)이 필요하다. `--search-env`로 환경 이름을 바꿀 수 있다. 검색은 별도 프로세스의 CPU에서 실행한다.

- 기본 DB: `knowledge/vectorstores/knu_notice_bge_m3`
- 기본 검색: 공지 청크 5개, 최대 출력 768토큰
- BGE-M3 캐시: `.cache/embedding_models` (`--cache-dir`로 변경)
- 색인은 `ready` 상태이고 활성 청크와 모델 revision 정보가 있어야 한다. 해당 revision의 모델이 로컬 캐시에 있어야 한다.
- 프로젝트 루트의 `.env`가 있으면 python-dotenv로 읽는다. 이미 설정된 환경변수를 덮어쓰지 않는다. 키를 Git에 올리지 않는다.

이 CLI는 선택한 한 코퍼스만 검색한다. 강의계획서 색인이 준비된 경우 `--corpus syllabi`로 선택한다. 공지와 강의계획서 자동 통합 검색이나 에브리타임 검색은 하지 않는다.

## vLLM으로 실행

vLLM 서버가 학생용 LoRA를 등록해 서빙하고 있어야 한다. 팀원에게 **서버 주소와 학생용 LoRA의 등록 이름**을 확인한다. 어댑터의 디스크 경로가 아니라 `/v1/models`에 나타나는 이름이다. 목록에 존재하는지 검사하지만, 그 이름이 어떤 학습 가중치를 가리키는지는 서버 설정으로 확인해야 한다.

```bash
export VLLM_BASE_URL="http://실제-vllm-호스트:8000"
export VLLM_STUDENT_LORA_MODEL_NAME="실제-학생용-LoRA-등록이름"
python -u scripts/ask_rag_lora.py --backend vllm --question "2026학년도 2학기 등록금 납부 일정과 주의사항을 알려줘"
```

주소는 `/v1` 유무 모두 지원한다. 명령줄의 `--vllm-base-url`, `--served-model`이 환경변수보다 우선한다. 인증 서버는 `VLLM_API_KEY`를 설정한다. Modal 프록시는 `VLLM_PROXY_KEY`, `VLLM_PROXY_SECRET`을 함께 설정한다. 인증 값은 실행 기록에 저장하지 않는다. 호출 제한 시간은 `--timeout 180`으로 조정한다.

등록되지 않은 모델은 오류로 종료하며 기본 모델로 대체하지 않는다. 요청에는 검색 문맥과 질문, `temperature=0`, `max_tokens=768`을 보낸다. vLLM 실행에는 이 클라이언트의 로컬 생성 모델·어댑터·torch·peft가 필요하지 않다. GPU는 vLLM 서버 측에서 사용한다. BGE 검색 환경과 DB는 클라이언트 측에 필요하다.

`--context-limit`는 로컬 Transformers용이다. vLLM의 입력 한도는 서버 설정을 따른다. 입력이 너무 길면 `--top-k`를 줄인다. 응답의 `finish_reason`이 `length`이면 출력이 잘렸다는 안내를 표시한다.

## 로컬 GPU에서 실행

학교 Slurm에서 GPU를 할당받은 셸에서 실행한다. 이미 할당된 셸이라면 추가 할당하지 않는다.

```bash
srun -p p02 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=rag-lora --pty bash
conda activate ai
cd /abr/coss36/four-leaf-AI
python -u scripts/ask_rag_lora.py --question "2026학년도 2학기 등록금 납부 일정과 주의사항을 알려줘"
```

현재 Python 환경에는 torch, transformers, peft, accelerate가 필요하다. 학생용 어댑터의 기본 위치는 `adapter_student_llama`이며 `--adapter`로 변경한다. 교수용 어댑터는 필요하지 않다. 모델 로딩은 이 파일 안에서 수행하며 `eval/generate_answers.py`에 의존하지 않는다.

기본 모델은 `--model-dir`, `MODEL_DIR`, 어댑터 설정의 `base_model_name_or_path` 순서로 선택한다. 자동 다운로드하지 않으므로 가중치와 토크나이저가 지정 경로 또는 Hugging Face 캐시에 있어야 한다. 비교 실험에서는 실제 스냅샷 경로를 고정한다.

```bash
python -u scripts/ask_rag_lora.py --backend transformers --model-dir /실제/모델/스냅샷 --adapter adapter_student_llama --question "질문"
```

기본 입력·출력 합계 한도는 8192토큰이며 모델 최대 길이와 비교해 작은 값을 적용한다. BF16으로 모델을 로드하고 `do_sample=False`로 생성한다. 매 실행마다 모델을 다시 로드한다. GPU 셸 작업이 끝나면 `exit`로 할당 자원을 반환한다.

## 실행 기록 및 검증

`output/rag_student_lora/<UTC시각>_<실행ID>/`에 다음을 저장하며 Git에서는 제외한다.

- `request.json`: 질문과 검색 설정
- `retrieval.json`: 청크 ID, 본문, 제목·출처, 거리, 색인 메타데이터
- `prompt.txt`: 모델에 전달한 사용자 프롬프트
- `answer.json`: 답변과 참고 문서, 실행 방식 및 생성 설정. 로컬 실행은 어댑터 해시·패키지 버전, vLLM은 요청/응답 모델 이름·사용 토큰·종료 사유 포함

문서가 없으면 생성 없이 근거 부족을 기록한다. 실패 시 앞 단계의 기록은 남는다. 모델 서버의 실제 가중치 해시·양자화·서빙 설정은 클라이언트가 확인할 수 없으므로 논문 실험에서는 서버 설정과 로그도 별도 보관한다. 같은 문서의 여러 청크가 검색될 수 있으며, 출처 목록은 사용한 청크 순서를 유지한다.

기존 학교 서버에서 공지 검색과 로컬 학생용 LoRA 답변 생성은 확인됐다. 이번 vLLM 옵션은 모의 HTTP 응답으로 요청 구성·모델 이름 검사·결과 저장을 검증했으며, 실제 vLLM 서버 주소·등록 이름으로 실행하는 연동 확인은 별도로 필요하다. 검색 결과가 있다는 사실만으로 답변의 정확성을 보장하지 않으므로 원문과 대조한다.
