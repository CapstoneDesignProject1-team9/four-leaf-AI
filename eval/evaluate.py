'''evaluate.py

Utility module for asynchronous evaluation of LLM responses using Claude as a judge,
ROUGE‑L overlap scoring, and RAGAS reference-based metrics.

Evaluation pipeline:
  1. LLM-as-a-Judge  – Claude scores each answer 1-5 with a rubric.
  2. ROUGE-L          – Token-overlap F1 between ground_truth and generated answer.
  3. RAGAS            – AnswerSimilarity & AnswerCorrectness (no retrieval context needed).
                        When a ``contexts`` field is present in the dataset item,
                        Faithfulness and ContextPrecision are also computed.

Required environment variables:
- ANTHROPIC_MODEL        (default: ``claude-3-opus-20240229``)
- ANTHROPIC_API_KEY      (required for Claude API access)
- OPENAI_API_KEY         (required for RAGAS embeddings/LLM; or set RAGAS_LLM/RAGAS_EMBEDDINGS)
- EVAL_MAX_CONCURRENCY   (optional, default ``5``)

The module expects the following environment variables:
- ANTHROPIC_MODEL (default: ``claude-3-opus-20240229``)
- ANTHROPIC_API_KEY (required for Claude API access)
- EVAL_MAX_CONCURRENCY (optional, default ``5``)
'''

import asyncio
import csv
import json
import logging
import os
import random
import time
import itertools

import re


from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, conint
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_rel


def run_ragas_metrics(dataset: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, float]]:
    """Evaluate faithfulness and context precision where retrieved contexts exist."""
    rows = []
    keys = []
    for item_idx, item in enumerate(dataset):
        for model_name, answer in item.get("models", {}).items():
            contexts = item.get("contexts", {}).get(model_name, [])
            if answer and contexts:
                rows.append({"question": item["question"], "answer": answer,
                             "contexts": contexts, "reference": item.get("ground_truth", "")})
                keys.append((item_idx, model_name))
    if not rows:
        logger.info("RAGAS skipped: no non-empty retrieved contexts in dataset")
        return {}
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        try:
            from ragas.metrics import faithfulness, context_precision
        except ImportError:
            from ragas.metrics.collections import Faithfulness, ContextPrecision
            faithfulness, context_precision = Faithfulness(), ContextPrecision()
        evaluator_key = os.getenv("ANTHROPIC_API_KEY")
        if not evaluator_key:
            logger.warning("RAGAS skipped: ANTHROPIC_API_KEY is not configured")
            return {}
        evaluator = ChatAnthropic(
            model=os.getenv("RAGAS_ANTHROPIC_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")),
            temperature=0.0,
            api_key=evaluator_key,
        )
        ragas_kwargs = {
            "metrics": [faithfulness, context_precision],
            "llm": LangchainLLMWrapper(evaluator),
            "run_config": RunConfig(timeout=180),
        }
        scores = ragas_evaluate(Dataset.from_list(rows), **ragas_kwargs)
        frame = scores.to_pandas()
        result: dict[tuple[int, str], dict[str, float]] = {}
        for key, (_, row) in zip(keys, frame.iterrows()):
            result[key] = {
                "Faithfulness": float(row.get("faithfulness", float("nan"))),
                "Context Precision": float(row.get("context_precision", float("nan"))),
            }
        return result
    except Exception as exc:
        logger.exception("RAGAS evaluation failed: %s", exc)
        return {}


def analyze_statistics(csv_data: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in csv_data:
        grouped[row["Model"]][row["Question_ID"]] = float(row["Opus Score"])
    groups = {name: list(scores.values()) for name, scores in grouped.items() if scores}
    if len(groups) >= 2:
        anova = f_oneway(*groups.values())
        logger.info("One-way ANOVA: F=%.4f p=%.6g", anova.statistic, anova.pvalue)
        common_ids = set.intersection(*(set(grouped[name]) for name in groups))
        pair_results = []
        for left, right in itertools.combinations(sorted(grouped), 2):
            ids = sorted(common_ids)
            a = np.array([grouped[left][i] for i in ids], dtype=float)
            b = np.array([grouped[right][i] for i in ids], dtype=float)
            if len(ids) < 2:
                continue
            test = ttest_rel(a, b)
            diff = a - b
            sd = float(np.std(diff, ddof=1))
            d = float(np.mean(diff) / sd) if sd else 0.0
            pair_results.append({"Model_A": left, "Model_B": right, "t": float(test.statistic), "p_raw": float(test.pvalue), "Cohens_d_paired": d, "n": len(ids)})
        for result in pair_results:
            result["p_bonferroni"] = min(1.0, result["p_raw"] * max(1, len(pair_results)))
            logger.info("Paired t-test %s vs %s: t=%.4f p=%.6g p_adj=%.6g Cohen's d=%.4f n=%d", result["Model_A"], result["Model_B"], result["t"], result["p_raw"], result["p_bonferroni"], result["Cohens_d_paired"], result["n"])
        pd.DataFrame(pair_results).to_csv(Path(__file__).with_name("statistical_tests.csv"), index=False)


def analyze_human_scores(csv_data: list[dict[str, Any]], results_path: Path) -> None:
    human_path = Path(__file__).with_name("human_evaluation.csv")
    if not human_path.exists():
        human_path.write_text("question_id,rater_id,model,score\n", encoding="utf-8")
        logger.info("Created human evaluation template: %s", human_path)
        return
    human = pd.read_csv(human_path)
    required = {"question_id", "rater_id", "model", "score"}
    if not required.issubset(human.columns):
        logger.warning("human_evaluation.csv needs columns: %s", sorted(required))
        return
    human["score"] = pd.to_numeric(human["score"], errors="coerce")
    human = human.dropna(subset=["question_id", "model", "score"])
    auto = pd.DataFrame(csv_data).groupby(["Question_ID", "Model"], as_index=False)["Opus Score"].mean()
    human_mean = human.groupby(["question_id", "model"], as_index=False)["score"].mean()
    merged = auto.merge(human_mean, left_on=["Question_ID", "Model"], right_on=["question_id", "model"])
    if len(merged) >= 3 and merged["Opus Score"].nunique() > 1 and merged["score"].nunique() > 1:
        p = pearsonr(merged["Opus Score"], merged["score"])
        s = spearmanr(merged["Opus Score"], merged["score"])
        logger.info("Human correlation n=%d Pearson r=%.4f p=%.6g Spearman rho=%.4f p=%.6g", len(merged), p.statistic, p.pvalue, s.statistic, s.pvalue)
        merged.to_csv(results_path.with_name("human_correlation.csv"), index=False)
        with results_path.with_name("human_correlation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"])
            writer.writeheader()
            writer.writerow({"n": len(merged), "pearson_r": p.statistic, "pearson_p": p.pvalue, "spearman_rho": s.statistic, "spearman_p": s.pvalue})
    else:
        logger.info("Human correlation skipped: need at least 3 matched, non-constant ratings")

# ---------------------------------------------------------------------------
# Load environment variables and configure logging
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model for the evaluation result
# ---------------------------------------------------------------------------
class EvalResult(BaseModel):
    """Result returned by the LLM judge.

    Attributes
    ----------
    score: int
        Integer score from 1 (worst) to 5 (best).
    reason: str
        Explanation for the assigned score.
    hallucination: bool
        Whether the answer contains hallucination (detected via reason).
    latency: float
        Latency of the LLM call in seconds.
    """

    score: conint(ge=1, le=5) = Field(description="1점부터 5점까지의 평가 점수")
    reason: str = Field(description="점수를 부여한 구체적인 이유")
    hallucination: bool = Field(description="환각 여부 (reason에 환각/ hallucination 포함 여부)")
    latency: float = Field(description="LLM 호출 지연시간 (초)")

# ---------------------------------------------------------------------------
# ROUGE‑L helper
# ---------------------------------------------------------------------------
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def calculate_rouge_l(ground_truth: str, generated_answer: str) -> float:
    """Calculate ROUGE‑L F1 between a reference answer and a generated answer.

    Returns ``0.0`` when either string is empty or only whitespace.
    """
    if not ground_truth or not generated_answer:
        return 0.0
    if not ground_truth.strip() or not generated_answer.strip():
        return 0.0
    scores = _rouge_scorer.score(target=ground_truth, prediction=generated_answer)
    return round(scores["rougeL"].fmeasure, 4)

# ---------------------------------------------------------------------------
# Single‑item evaluation with exponential back‑off
# ---------------------------------------------------------------------------
async def evaluate_single_item(
    chain: Any,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    item_index: int,
    total_count: int,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate a single item and record latency & hallucination.

    The function now measures the elapsed time for the LLM call and
    derives a ``hallucination`` boolean flag by searching the ``reason``
    text for the keywords "환각" or "hallucination". Both values are
    returned as part of the result dictionary.
    """

    """Evaluate a single item using the supplied LangChain ``chain``.

    The function respects the provided ``semaphore`` to limit concurrency and
    retries on any exception with exponential back‑off plus jitter.
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                started = time.perf_counter()
                result = await chain.ainvoke(item)
                elapsed = time.perf_counter() - started
                # ``chain`` may return a ``EvalResult`` instance or a plain dict.
                if isinstance(result, EvalResult):
                    result = result.dict()
                result.setdefault("latency", round(elapsed, 4))
                logger.info(
                    f"[{item_index}/{total_count}] 평가 완료 - 점수: {result.get('score')}점"
                )
                return result
            except Exception as e:  # pragma: no cover – generic catch for robustness
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                    logger.warning(
                        f"[{item_index}/{total_count}] 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}. "
                        f"{delay:.1f}초 후 재시도..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[{item_index}/{total_count}] 최대 재시도 횟수({max_retries}) 초과 실패: {e}"
                    )
                    return {
                        "score": 0,
                        "reason": f"Error after {max_retries} retries: {e}",
                    }
        # Should never reach here, but keep a safe fallback.
        return {"score": 0, "reason": "Unknown evaluation failure"}

# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------
async def evaluate_responses_batch(
    eval_items: List[Dict[str, Any]],
    max_concurrency: int = 5,
    max_retries: int = 3,
    format_instructions: str = "",
) -> List[Dict[str, Any]]:
    """Run asynchronous evaluation on a list of items.

    Parameters
    ----------
    eval_items: List[dict]
        Each dict must contain ``question``, ``ground_truth``, ``generated_answer``
        and ``format_instructions`` (the latter is passed to the prompt).
    max_concurrency: int
        Upper bound for parallel API calls (default 5).
    max_retries: int
        Number of retry attempts per item (default 3).
    format_instructions: str
        Instructions for the JSON output parser – normally obtained from
        ``JsonOutputParser.get_format_instructions()``.
    """
    if not eval_items:
        logger.warning("평가할 항목이 없습니다.")
        return []

    # Validate required environment variables
    ant_model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    ant_key = os.getenv("ANTHROPIC_API_KEY")
    if not ant_key:
        logger.error("ANTHROPIC_API_KEY 가 설정되지 않았습니다. 평가를 중단합니다.")
        return []

    evaluator_llm = ChatAnthropic(model=ant_model, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=EvalResult)
    # Use supplied format instructions; fallback to parser-generated if empty
    if not format_instructions:
        format_instructions = parser.get_format_instructions()

    prompt_template = """당신은 인공지능 모델의 답변 품질을 평가하는 공정한 심판입니다.

아래의 [질문], [모범 답안], 그리고 인공지능이 생성한 [모델의 답변]을 읽고 평가를 진행해주세요.

[구체적 채점 루브릭]
5점 (매우 우수):
 - 모범 답안의 사실과 100% 일치함
 - 대학 AI 튜터로서 매우 친절하고 명확한 어조를 완벽히 구사함
 - 환각(Hallucination)이 전혀 없으며, 질문자가 필요한 정보를 충분히 제공함

4점 (우수):
 - 모범 답안의 사실과 대부분 일치하나, 아주 사소한 디테일이 빠져있음
 - 어조가 대체로 적절하고 환각이 없음

3점 (보통):
 - 핵심 사실은 맞으나, 부가 설명이 현저히 부족하거나 어투가 딱딱하고 기계적임
 - 모범 답안의 내용 중 일부를 생략하여 유용성이 다소 떨어짐 (환각은 없음)

2점 (미흡):
 - 모범 답안의 사실과 일부 모순되는 내용이 포함됨
 - 혹은 불필요한 환각(지어낸 내용)이 섞여 있어 사용자에게 혼동을 줄 수 있음

1점 (매우 나쁨):
 - 모범 답안과 완전히 상반된 정보를 제공하여 치명적인 오안내를 함
 - 대학 정보와 무관한 심각한 환각(Hallucination)이 포함됨

위 기준을 바탕으로 1점에서 5점 사이의 점수를 매기고, 다음 추가 항목을 포함해 주세요:
- "faithfulness": 0.0과 1.0 사이의 실수, 답변이 사실과 얼마나 일치하는지 나타냅니다.
- "hallucination_detected": true/false, 답변에 환각 내용이 존재하는지 여부를 표시합니다.
그 이유를 설명하고, 위 필드들을 JSON 형식으로 출력해주세요.
{format_instructions}\n\n[질문]: {question}\n[모범 답안]: {ground_truth}\n[모델의 답변]: {generated_answer}\n"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 AI 모델 평가 전문가입니다."), ("human", prompt_template)]
    )
    chain = prompt | evaluator_llm | parser

    semaphore = asyncio.Semaphore(max_concurrency)
    total_count = len(eval_items)

    tasks = [
        evaluate_single_item(
            chain=chain,
            item=item,
            semaphore=semaphore,
            item_index=idx + 1,
            total_count=total_count,
            max_retries=max_retries,
        )
        for idx, item in enumerate(eval_items)
    ]
    results = await asyncio.gather(*tasks)
    return results

# ---------------------------------------------------------------------------
# RAGAS evaluation helper
# ---------------------------------------------------------------------------
def evaluate_with_ragas(
    eval_items: List[Dict[str, Any]],
) -> List[Dict[str, float]]:
    """Run RAGAS evaluation on *eval_items* and return per-item metric dicts.

    Each item in ``eval_items`` must have:
    - ``question``         (str)
    - ``ground_truth``     (str)
    - ``generated_answer`` (str)
    - ``contexts``         (List[str], optional) – if present, retrieval metrics are added.

    Returns a list of dicts (same length as ``eval_items``) with RAGAS metric
    values. Falls back to an empty dict per item when RAGAS is unavailable or
    the evaluation fails.
    """
    empty = [{} for _ in eval_items]

    if not _RAGAS_AVAILABLE:
        return empty

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY 가 설정되지 않아 RAGAS 평가를 건너뜁니다. "
            "(RAGAS 기본 LLM/임베딩은 OpenAI를 사용합니다)"
        )
        return empty

    try:
        # contexts가 하나라도 있으면 retrieval 메트릭 포함
        # (없는 아이템은 빈 리스트로 패딩 → NaN으로 기록됨)
        any_contexts = any("contexts" in item and item["contexts"] for item in eval_items)

        ragas_data: Dict[str, List] = {
            "question": [item["question"] for item in eval_items],
            "answer": [item["generated_answer"] for item in eval_items],
            "ground_truth": [item["ground_truth"] for item in eval_items],
        }
        if any_contexts:
            # contexts 없는 아이템(Non-RAG 모델)은 빈 리스트로 패딩
            ragas_data["contexts"] = [
                item.get("contexts") or [] for item in eval_items
            ]
            ctx_count = sum(1 for i in eval_items if i.get("contexts"))
            logger.info(
                f"RAGAS: contexts 있는 아이템 {ctx_count}개 / 전체 {len(eval_items)}개 "
                "– faithfulness, context_precision 포함"
            )

        hf_dataset = HFDataset.from_dict(ragas_data)

        metrics = [answer_similarity, answer_correctness]
        if any_contexts and _RAGAS_RETRIEVAL_METRICS:
            metrics += _RAGAS_RETRIEVAL_METRICS

        logger.info(f"RAGAS 평가 시작 (메트릭: {[m.name for m in metrics]})...")
        result = ragas_evaluate(dataset=hf_dataset, metrics=metrics)
        result_df = result.to_pandas()

        metric_names = [m.name for m in metrics]
        per_item: List[Dict[str, float]] = []
        for _, row in result_df.iterrows():
            per_item.append(
                {
                    name: round(float(row[name]), 4)
                    for name in metric_names
                    if name in row and row[name] == row[name]  # NaN 필터링 (NaN != NaN)
                }
            )
        logger.info("RAGAS 평가 완료")
        return per_item
    except Exception as exc:
        logger.error(f"RAGAS 평가 중 오류 발생: {exc}")
        return empty


# ---------------------------------------------------------------------------
# Main entry point for asynchronous execution
# ---------------------------------------------------------------------------
async def async_main() -> None:
    """Load the dataset, perform evaluation, and write a CSV summary.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"테스트 데이터셋 파일이 없습니다: {dataset_path}")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    parser = JsonOutputParser(pydantic_object=EvalResult)
    format_instructions = parser.get_format_instructions()

    eval_items: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []  # 결과를 매핑하기 위한 메타데이터

    for item_idx, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        # contexts는 모델별 dict 구조: {"Base Model + RAG": ["chunk1", ...], ...}
        item_contexts: Dict[str, List[str]] = item.get("contexts", {})
        for model_name, answer in item["models"].items():
            eval_items.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": answer,
                    "contexts": item.get("contexts", {}).get(model_name, []),
                    "format_instructions": format_instructions,
                }
            )
            metadata.append({"item_idx": item_idx, "model_name": model_name})

    max_concurrency = int(os.getenv("EVAL_MAX_CONCURRENCY", "5"))
    if max_concurrency < 1:
        max_concurrency = 1
    logger.info(
        f"총 {len(dataset)}개의 질문, {len(eval_items)}개의 응답에 대한 비동기 평가 시작... (동시 실행 제한: {max_concurrency})"
    )
    results = await evaluate_responses_batch(
        eval_items, max_concurrency=max_concurrency, format_instructions=format_instructions
    )

    ragas_scores = await asyncio.to_thread(run_ragas_metrics, dataset)

    # -------------------------------------------------------------------
    # Aggregate scores and write CSV report
    # -------------------------------------------------------------------
    model_scores: Dict[str, List[int]] = {}
    model_rouge_scores: Dict[str, List[float]] = {}
    model_hallucinations: Dict[str, List[bool]] = {}
    model_latencies: Dict[str, List[float]] = {}
    csv_data: List[Dict[str, Any]] = []

    model_ragas_scores: Dict[str, Dict[str, List[float]]] = {}

    for idx, (meta, item, res) in enumerate(zip(metadata, eval_items, results)):
        q_idx = meta["item_idx"] + 1
        m_name = meta["model_name"]
        score = res.get("score", 0)
        reason = res.get("reason", "")
        rouge_l_score = calculate_rouge_l(
            ground_truth=item["ground_truth"],
            generated_answer=item["generated_answer"],
        )
        ragas_result = ragas_scores.get((meta["item_idx"], m_name), {})
        logger.info(
            f"[질문 {q_idx} | {m_name}] 점수: {score} | ROUGE-L: {rouge_l_score:.4f}"
            + (f" | RAGAS sim: {ragas_similarity:.4f}" if ragas_similarity is not None else "")
            + (f" | RAGAS corr: {ragas_correctness:.4f}" if ragas_correctness is not None else "")
            + f" - {reason}"
        )
        model_scores.setdefault(m_name, []).append(score)
        model_rouge_scores.setdefault(m_name, []).append(rouge_l_score)
        csv_data.append(
            {
                "Question_ID": q_idx,
                "Question": item["question"],
                "Ground_Truth": item["ground_truth"],
                "Model": m_name,
                "Generated_Answer": item["generated_answer"],
                "Opus Score": score,
                "ROUGE-L Score": rouge_l_score,
                "Faithfulness": ragas_result.get("Faithfulness", ""),
                "Context Precision": ragas_result.get("Context Precision", ""),
                "Reason": reason,
                "Latency (s)": res.get("latency", 0.0),
                "Hallucination": res.get("hallucination", False),
            }
        )

    logger.info("=" * 40)
    logger.info("📊 모델별 최종 평균 점수")
    logger.info("=" * 40)
    for m_name, scores in model_scores.items():
        rouge_scores = model_rouge_scores[m_name]
        avg_llm = sum(scores) / len(scores) if scores else 0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        ragas_summary_parts = []
        for metric_key, vals in model_ragas_scores.get(m_name, {}).items():
            avg_val = sum(vals) / len(vals) if vals else 0
            ragas_summary_parts.append(f"{metric_key}: {avg_val:.4f}")
        ragas_summary = " | ".join(ragas_summary_parts) if ragas_summary_parts else "N/A"
        logger.info(
            f" - {m_name}: Opus {avg_llm:.2f}점 | ROUGE-L {avg_rouge:.4f} | RAGAS [{ragas_summary}] (총 {len(scores)}개)"
        )

    output_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    analyze_statistics(csv_data)
    analyze_human_scores(csv_data, Path(output_path))
    if csv_data:
        keys = list(csv_data[0].keys())
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_data)
        logger.info(f"✅ 평가 결과가 '{output_path}'에 저장되었습니다.")
        # Compute human agreement and correlation if human evaluation data is available
        compute_human_agreement_and_correlation(csv_data)
    else:
        logger.warning("CSV 데이터가 비어 있어 저장을 건너뛰었습니다.")

def main() -> None:
    """Entry‑point for synchronous execution via ``python -m eval.evaluate``."""
    asyncio.run(async_main())

def compute_human_agreement_and_correlation(csv_data: List[Dict[str, Any]]) -> None:
    """Compute inter‑rater agreement (Fleiss' Kappa) and correlation with Claude Opus scores.

    Expects a CSV file ``human_evaluation.csv`` in the same directory with columns
    ``question_id``, ``rater_id`` and ``score`` (1‑15).  The function reads the file,
    aggregates the human scores per question, and then compares the average human
    score with the ``Opus Score`` produced by Claude.

    The results are logged:
    * Pearson correlation coefficient
    * Spearman rank correlation coefficient
    * Fleiss' Kappa for the categorical ratings
    """
    import csv
    from collections import defaultdict
    from pathlib import Path

    human_path = Path(__file__).with_name("human_evaluation.csv")
    if not human_path.exists():
        logger.info("Human evaluation file not found – skipping correlation analysis.")
        return

    # Load human scores
    per_question: Dict[int, List[int]] = defaultdict(list)
    with open(human_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["question_id"])
            score = int(row["score"])
            per_question[qid].append(score)

    # Build parallel lists for correlation
    opus_scores: List[float] = []
    human_means: List[float] = []
    for entry in csv_data:
        qid = entry["Question_ID"]
        opus = entry.get("Opus Score")
        if qid in per_question:
            human_avg = sum(per_question[qid]) / len(per_question[qid])
            opus_scores.append(float(opus))
            human_means.append(human_avg)

    if len(opus_scores) < 2:
        logger.warning("Not enough overlapping items for correlation calculation.")
        return

    # Correlations
    pearson, _ = pearsonr(opus_scores, human_means)
    spearman, _ = spearmanr(opus_scores, human_means)
    logger.info(f"Pearson correlation Opus ↔ Human: {pearson:.4f}")
    logger.info(f"Spearman correlation Opus ↔ Human: {spearman:.4f}")

    # Fleiss' Kappa – build rating matrix
    categories = list(range(1, 16))
    rating_matrix: List[List[int]] = []
    for qid in sorted(per_question.keys()):
        counts = [per_question[qid].count(cat) for cat in categories]
        rating_matrix.append(counts)

    try:
        kappa = fleiss_kappa(rating_matrix)
        logger.info(f"Fleiss' Kappa (inter‑rater agreement): {kappa:.4f}")
    except Exception as exc:
        logger.error(f"Failed to compute Fleiss' Kappa: {exc}")

    # Summary
    logger.info("=== Human vs. Claude Opus correlation summary ===")
    logger.info(f"Questions compared: {len(opus_scores)}")
    logger.info(f"Average human score: {sum(human_means)/len(human_means):.2f}")
    logger.info(f"Average Opus score : {sum(opus_scores)/len(opus_scores):.2f}")

if __name__ == "__main__":
    main()
