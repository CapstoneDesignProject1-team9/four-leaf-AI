# 평가 파이프라인

평가기는 저장된 모델 응답을 채점해 문항별 결과, 조건별 요약, 계획된 가설 검정 결과를 CSV로 만듭니다. 같은 질문 ID를 조건 간 대응시켜 비교합니다.

## 비교 조건

학생용 데이터셋은 아래 5조건입니다.

- Base Model
- Base Model + RAG
- Base Model + Student LoRA
- Base Model + Professor LoRA
- Base Model + Student LoRA + RAG

교수용 데이터셋은 Base Model, Student LoRA, Professor LoRA를 비교합니다. 교수자용 RAG 문서는 없으므로 RAG 지표는 학생용 RAG 응답에만 계산됩니다.

## 응답 생성과 평가

교수자용 합성 평가 데이터 초안은 다음 명령으로 재생성합니다. 기본 출력은 기존 파일을 덮어쓰지 않는 `professor_eval_dataset_v3.json`입니다. 시나리오와 학생 질문 묶음은 실제 수강생 자료가 아니므로 교수자 검토 및 학습·평가 데이터 중복 검토 후 분석에 사용하세요.

```bash
python eval/generate_professor_eval_dataset.py
```

각 요청은 서로 다른 수업 준비·과제 운영·평가·접근성·소통 지원 작업을 요청합니다. 모델 응답 생성기는 `student_questions`를 프롬프트에 포함해야 합니다.

먼저 두 어댑터를 다음 위치에 둡니다. 각 디렉터리에는 PEFT 형식의 `adapter_config.json`과 `adapter_model.safetensors`(또는 `adapter_model.bin`)가 있어야 합니다. `adapter_config.json`의 `base_model_name_or_path`는 두 어댑터 모두 `meta-llama/Llama-3.1-8B-Instruct`이며 `VLLM_BASE_MODEL`도 이 값으로 맞춥니다. 현재 rank는 16입니다.

```text
four-leaf-AI/models/student-lora/adapter_student_llama/
four-leaf-AI/models/professor-lora/adapter_professor_llama/
```

vLLM은 두 폴더를 `student-lora`, `professor-lora`라는 모델명으로 등록합니다. NVIDIA GPU가 있는 Linux Docker host에서 저장소 루트 기준으로 서비스를 시작합니다.

```bash
cd four-leaf-infra
# 처음 한 번만: .env.dev.example을 복사하고 필요한 API 키를 채움
cp .env.dev.example .env.dev
docker compose -f docker-compose.dev.yml --profile gpu up -d --build
```

RAG 검색은 키워드 라우팅 없이 공지·강의계획서 두 컬렉션에서 cosine 거리 상위 5개 청크를 가져옵니다(에브리타임 강의평 제외). vLLM과 AI/RAG 서비스가 준비되면 `four-leaf-AI/`에서 생성·평가를 실행합니다. 생성기는 vLLM `/v1/models`에서 base와 두 어댑터가 등록됐는지 확인하고, 준비 전이면 최대 3분 대기합니다.

```bash
python eval/generate_answers.py --dataset student_eval_dataset.json
python eval/evaluate.py
```

평가할 JSON은 `EVAL_DATASET_PATH`로 선택할 수 있습니다.

```bash
EVAL_DATASET_PATH=eval/professor_eval_dataset.json python eval/evaluate.py
```

RAGAS는 `ANTHROPIC_API_KEY`로 Claude를 평가 LLM으로 사용합니다. RAGAS 점수는 해당 응답의 검색 문맥이 저장되어 있을 때만 계산됩니다. 모델 응답이 비어 있거나 `[ERROR]`로 시작하면 평가에서 제외됩니다.

## 자동 생성되는 CSV

| 파일 | 내용 |
|---|---|
| `student_eval_results.csv` / `professor_eval_results.csv` | 문항별 모델 응답, Claude 평가 항목별 점수, ROUGE-L, RAGAS, 환각 여부, 생성 latency |
| `student_condition_summary.csv` / `professor_condition_summary.csv` | 조건별(학생은 주제별 포함) 평균·표준편차·중앙값·사분위수·95% 신뢰구간 |
| `student_hypothesis_tests.csv` / `professor_hypothesis_tests.csv` | 사전 지정 대비의 paired t-test, Wilcoxon, Holm 보정 p값, 평균차, Cohen's dz, 신뢰구간. human 점수가 있으면 사람 평가 결과도 포함 |
| `student_human_scores_by_item.csv` / `professor_human_scores_by_item.csv` | 문항·모델·평가 차원별 평가자 평균 |
| `student_human_rater_agreement.csv` / `professor_human_rater_agreement.csv` | 평가자 점수가 충분히 입력된 경우 Fleiss’ kappa |
| `student_human_judge_correlation.csv` / `professor_human_judge_correlation.csv` | 사람 점수와 Claude 점수의 Pearson·Spearman 상관, p값 및 문항 단위 bootstrap 신뢰구간 |

학생용 가설 검정은 Base의 RAG 효과, RAG 없는 Student LoRA 효과, 결합 조건 대 Base, RAG×Student LoRA 상호작용, Professor LoRA의 학생 질문 전이, 두 역할 LoRA 차이를 계산합니다. 교수용은 Professor LoRA 개선, Student LoRA 전이, 두 어댑터 차이를 계산합니다. 출력되는 모든 쌍별 통계가 연구 가설은 아니므로, 논문에는 사전에 정한 주요 대비를 중심으로 보고합니다.

## 사람 평가 (학생용)

학생용 인간평가는 **정확성(1~5)과 응답 유형(1 정답·부분정답 / 2 답변 유보 / 3 다른 학기 정보 사용 / 4 근거 없는 사실 날조)** 두 가지만 기록합니다. 기준과 절차는 [HUMAN_EVAL_GUIDE.md](HUMAN_EVAL_GUIDE.md)를 따릅니다.

```bash
python eval/human_eval.py sheet     # 맹검 시트·템플릿·키 생성 (응답 생성 완료 후)
python eval/human_eval.py analyze   # eval/human_eval/ratings/*.csv 분석
```

분석 결과(`student_human_by_item.csv`, `student_human_summary.csv`, `student_human_tests.csv`, `student_human_agreement.csv`)는 `eval/human_eval/`에 저장됩니다. 이전 형식의 `student_human_evaluation.csv`(정확성·유용성·간결성·유창성)는 v2 30문항 평가 기록으로 남겨 둡니다.

## 해석 범위

Claude 채점은 자동 보조 평가이며 논문 주장의 유일한 근거로 사용하지 않습니다. 사람 평가를 주요 결과로 보고, 평가자 간 일치도와 사전 지정 통계 대비를 함께 확인하세요. latency는 생성 API 왕복시간을 저장하며 평가자 호출 시간과 구분합니다. 생성 latency가 없는 기존 응답은 요약에서 제외됩니다.

현재 `generate_answers.py`는 학생 질문 생성 경로입니다. 교수용 평가를 실행하려면 교수용 세 조건의 응답을 같은 문항 ID로 `professor_eval_dataset.json`의 `models`에 채운 뒤 `EVAL_DATASET_PATH`로 지정합니다.
