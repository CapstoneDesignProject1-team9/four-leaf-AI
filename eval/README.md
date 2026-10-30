# Four‑Leaf AI 튜터 평가 스위트

## 📖 개요
`four-leaf-AI/eval/` 패키지는 **자체 포함형 파이프라인**을 제공합니다.
1️⃣ 네 가지 어댑터 구성(Base, Base + RAG, Base + LoRA, Base + RAG + LoRA)의 답변을 생성합니다 – `generate_answers.py`(이전 `generate_answers_final.py`)를 참고하십시오.
2️⃣ 각 답변을 Claude(Anthropic API)에게 보내 **1~5점**의 품질 점수와 이유 텍스트를 얻습니다.
3️⃣ `rouge_score` 라이브러리를 사용해 **ROUGE‑L** 유사도를 계산합니다.
4️⃣ 구성별 점수를 평균내고 CSV(`eval_results.csv`)로 저장합니다.

비동기 처리, 동시 실행 제한, 진행 로그 출력 등 실용적인 기능을 갖추었습니다.

---
## 🛠️ 핵심 파일
| 파일 | 목적 |
|------|------|
| [`generate_answers.py`](file:///Users/tak/Desktop/CDP1-team9-fourleaf/four-leaf-AI/eval/generate_answers.py) | 네 가지 답변 변형을 생성하고 `eval_dataset.json`에 저장합니다. |
| [`evaluate.py`](file:///Users/tak/Desktop/CDP1-team9-fourleaf/four-leaf-AI/eval/evaluate.py) | Claude 기반 평가, ROUGE‑L 계산, CSV 요약 생성 (`eval_results.csv`). |
| `eval_dataset.json` | 입력 데이터셋: `question`, `ground_truth`, 그리고 `models` dict(답변이 채워질 자리) 포함. |
| `eval_results.csv` | 논문에 활용할 최종 결과 테이블. |

---
## 📊 구현된 평가 지표
| 지표 | 얻는 방법 | 의미 |
|------|----------|------|
| **Opus Score** (Claude) | `evaluate.py`가 LangChain 프롬프트(164‑191줄)를 구성하고 JSON 파서를 통해 `EvalResult` 객체로 파싱합니다. 점수는 1‑5 정수. | **전체 답변 품질**을 LLM(Claude) 관점에서 평가(정확성, 완전성, 어조, 환각 포함). |
| **ROUGE‑L** | `calculate_rouge_l`(62‑74줄)에서 `rouge_score` 라이브러리 사용. | **표면적인 어휘 겹침**을 측정하여 정답과의 내용 일치 정도를 반영. |

두 지표는 각각 질문‑답변마다(266‑291줄) 로그로 출력되고, 구성별 평균으로 요약됩니다(296‑302줄).

---
## ▶️ 파이프라인 실행 방법
```bash
# 1️⃣ 가상환경 활성화 & API 키 로드
source .venv/bin/activate
export $(cat four-leaf-AI/.env | xargs)   # ANTHROPIC_API_KEY 등

# 2️⃣ 네 가지 구성 모두에 대한 답변 생성 (기본)
python -m four_leaf_AI.eval.generate_answers   # 옵션: --mode base|rag|lora|rag_lora

# 3️⃣ 평가 실행
python -m four_leaf_AI.eval.evaluate
```
명령을 수행하면 `eval_results.csv` 파일이 동일 폴더에 생성됩니다.

---
## 📑 논문에 포함할 내용
| 섹션 | 제안 내용 |
|------|----------|
| **데이터셋 설명** | 질문 수(현재 30개), 정답 출처, 각 질문에 네 가지 어댑터 답변이 존재함을 기술. |
| **자동 평가 지표** | **평균 Opus Score**와 **평균 ROUGE‑L**을 표(Table 1) 형태로 보고. 사용한 Claude 모델 버전(`claude-3-opus-20240229`) 명시. |
| **통계적 비교** | Opus Score에 대한 **ANOVA** 혹은 **pairwise t‑test** 결과를 제시해 구성 간 유의미한 차이를 입증. (현재 코드에 미구현 – 아래 *권고사항* 참고) |
| **인간‑AI 상관관계** | 인간 평가(`human_evaluation.csv`)와 Opus Score 간 **Pearson / Spearman** 상관계수와 95 % CI를 보고. (미구현) |
| **판정자 간 일관성** | 인간 평점에 대한 **Fleiss’ Kappa**와 p‑값 제시. (미구현) |
| **시각화** | 구성별 평균 Opus/ROUGE‑L 바 차트, Opus vs. 인간 점수 산점도, 구성별 ROUGE‑L 박스플롯 등. 현재 자동 생성 코드는 없으므로 추가 필요. |

---
## ✅ 현재 지표 적합성 평가
| 지표 | 논문에 적합? | 비고 |
|------|-------------|------|
| Opus Score | ✅ | AI‑튜터 품질을 종합적으로 평가하는 핵심 지표. |
| ROUGE‑L | ✅ (주의 필요) | 표면적 겹침만 반영하므로 보조 지표로 활용. |
| 인간‑AI 상관관계 | ❌ *미구현* | “Claude Opus 점수가 인간 평가와 0.82 이상 상관한다”는 주장을 뒷받침하려면 추가 구현 필요. |
| Fleiss’ Kappa | ❌ *미구현* | 인간 평점의 신뢰성을 보여주려면 필요. |
| 통계적 유의성 검사 (ANOVA, t‑test, 효과크기) | ❌ *미구현* | 구성 간 차이를 실증하려면 반드시 포함해야 함. |
| 시각화 (그래프) | ❌ *미구현* | 논문에 그림이 요구되므로 자동 생성 로직을 추가하는 것이 바람직. |

핵심 자동 지표는 적절하지만, **인간 평가와의 상관관계, 판정자 간 일관성, 통계적 검정, 시각화**가 현재 파이프라인에 없으므로 보강이 필요합니다.

---
## 📈 권고사항 – 추가·제거 항목
### 추가
1. **인간‑AI 상관관계**
   - `human_evaluation.csv`(칼럼: `question_id`, `rater_id`, `score`)를 읽어 `scipy.stats.bootstrap`으로 Pearson·Spearman 계수를 95 % CI와 함께 계산.
2. **판정자 간 일관성**
   - `statsmodels.stats.inter_rater.fleiss_kappa`를 이용해 Fleiss’ Kappa와 p‑값을 산출.
3. **통계 검정**
   - Opus Score에 대한 일원 ANOVA를 수행하고, 사후 검정으로 Bonferroni 보정된 pairwise t‑test와 Cohen’s d 효과크기를 보고.
4. **시각화 자동 생성** (PNG 저장)
   - 구성별 평균 Opus ± 95 % CI 바 차트.
   - Opus vs. 인간 평균 점수 산점도(회귀선 포함).
   - 구성별 ROUGE‑L 박스플롯.
5. **추가 지표 (선택)** – Faithfulness 혹은 Hallucination 여부를 Claude 프롬프팅에 추가하고, Boolean 필드로 파싱.

### 제거 / 단순화
- 인간 평가를 수집하지 않을 경우, 논문에서 상관관계·Kappa 섹션을 완전히 삭제하고 자동 지표만 기술.
- 현재 `evaluate.py`가 출력하는 Claude 사유(`reason`)는 필요에 따라 정성적 분석에만 활용하고, 표에는 포함하지 않아도 무방.

---
## 📂 디렉터리 구조 (README 포함 후)
```
four-leaf-AI/
└─ eval/
   ├─ generate_answers.py      # 답변 생성 (네 구성)
   ├─ evaluate.py             # Claude 평가 + ROUGE‑L, CSV 출력
   ├─ eval_dataset.json       # 입력 데이터셋
   ├─ eval_results.csv        # 출력 테이블 (논문에 바로 활용)
   ├─ README.md               # **이 파일**
   └─ (향후) *.png            # 권고사항에 따라 자동 생성되는 그림 파일
```

---
## 🛠️ 빠른 시작 체크리스트
- [ ] `requirements.txt`에 명시된 파이썬 패키지 설치 (`pip install -r requirements.txt`).
- [ ] Ollama가 실행 중인지 확인 (`ollama serve`).
- [ ] RAG FastAPI 서비스가 `http://localhost:8000/qa`에서 동작하는지 확인.
- [ ] `four-leaf-AI/.env`에 `ANTHROPIC_API_KEY`(및 필요시 `ANTHROPIC_MODEL`) 설정.
- [ ] 위 **파이프라인 실행 방법** 순서대로 실행.
- [ ] 논문에 넣을 통계·시각화 코드를 *권고사항*에 따라 `evaluate.py`에 추가.

---
*작성자: Antigravity 코딩 어시스턴트*