"""Evaluate saved model outputs and write paper-oriented CSV tables.

Set EVAL_DATASET_PATH to select student_eval_dataset.json or
professor_eval_dataset.json. Claude scores every non-empty answer; human
scores are read from human_evaluation.csv when present. Generation latency
must be recorded by generate_answers.py and is never inferred by the judge.
"""
from __future__ import annotations

import asyncio
import csv
import itertools
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, conint
from rouge_score import rouge_scorer
from scipy.stats import pearsonr, spearmanr, t, ttest_1samp, wilcoxon

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
EVAL_DIR = Path(__file__).parent
DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", str(EVAL_DIR / "student_eval_dataset.json")))
if not DATASET_PATH.is_absolute():
    DATASET_PATH = (Path.cwd() / DATASET_PATH).resolve()

SCORE_FIELDS = {
    "Overall Score": "overall_score",
    "Accuracy": "accuracy",
    "Completeness": "completeness",
    "Groundedness": "groundedness",
    "Clarity": "clarity",
    "Usefulness": "usefulness",
    "Role Fit": "role_fit",
}
HUMAN_FIELDS = {
    "Overall Score": "overall_score",
    "Accuracy": "accuracy",
    "Completeness": "completeness",
    "Groundedness": "groundedness",
    "Clarity": "clarity",
    "Usefulness": "usefulness",
    "Role Fit": "role_fit",
}
CONDITION_ALIASES = {
    "Base": "Base Model", "Base Model": "Base Model",
    "Base Model + RAG": "Base Model + RAG",
    "Student LoRA": "Base Model + Student LoRA",
    "Base Model + Student LoRA": "Base Model + Student LoRA",
    "Professor LoRA": "Base Model + Professor LoRA",
    "Base Model + Professor LoRA": "Base Model + Professor LoRA",
    "Base Model + Student LoRA + RAG": "Base Model + Student LoRA + RAG",
    "Fine-tuned Model (LoRA)": "Base Model + Student LoRA",
    "Fine-tuned Model (LoRA) + RAG": "Base Model + Student LoRA + RAG",
}


class JudgeResult(BaseModel):
    overall_score: conint(ge=1, le=5) = Field(description="전체 품질 1~5")
    accuracy: conint(ge=1, le=5) = Field(description="정확성 1~5")
    completeness: conint(ge=1, le=5) = Field(description="완결성 1~5")
    groundedness: conint(ge=1, le=5) = Field(description="제공된 근거와의 일치도 1~5")
    clarity: conint(ge=1, le=5) = Field(description="명료성 1~5")
    usefulness: conint(ge=1, le=5) = Field(description="유용성 1~5")
    role_fit: conint(ge=1, le=5) = Field(description="역할·과제 적합성 1~5")
    hallucination: bool = Field(description="근거 없는 사실 주장이 있는가")
    reason: str = Field(description="점수 근거를 간단히 기술")


rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def model_condition(name: str) -> str:
    return CONDITION_ALIASES.get(name, name)


def question_id(item: dict[str, Any], index: int) -> str:
    return str(item.get("question_id") or f"Q{index + 1:03d}")


def build_judge_input(item: dict[str, Any], model_name: str, answer: str) -> dict[str, Any]:
    professor_questions = item.get("student_questions")
    question = item["question"]
    if professor_questions:
        question += "\n\n학생 질문 묶음:\n" + "\n".join(f"- {q}" for q in professor_questions)
    contexts = item.get("contexts", {}).get(model_name, [])
    return {
        "question": question,
        "ground_truth": item.get("ground_truth", ""),
        "generated_answer": answer,
        "contexts": "\n\n".join(contexts) if contexts else "(제공된 검색 문맥 없음)",
        "role": item.get("role", "student"),
        "criteria": json.dumps(item.get("evaluation_criteria", []), ensure_ascii=False),
    }


async def judge_one(chain: Any, values: dict[str, Any], sem: asyncio.Semaphore, idx: int, total: int) -> dict[str, Any]:
    async with sem:
        for attempt in range(3):
            try:
                result = await chain.ainvoke(values)
                if isinstance(result, JudgeResult):
                    return result.model_dump()
                return JudgeResult.model_validate(result).model_dump()
            except Exception as exc:
                if attempt == 2:
                    logger.exception("Judge failed for response %s/%s", idx, total)
                    return {"overall_score": np.nan, "accuracy": np.nan, "completeness": np.nan,
                            "groundedness": np.nan, "clarity": np.nan, "usefulness": np.nan,
                            "role_fit": np.nan, "hallucination": None, "reason": f"JUDGE_ERROR: {exc}"}
                await asyncio.sleep(2 ** attempt + random.uniform(.1, .7))


def create_judge_chain() -> Any:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is required to score model answers")
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    llm = ChatAnthropic(model=model, temperature=0.0, api_key=key)
    parser = JsonOutputParser(pydantic_object=JudgeResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 대학 AI 튜터 응답 평가자입니다. 제공된 질문·기준·검색 문맥에 근거해 독립적으로 채점하세요. 기준 답안 문구가 다르더라도 의미가 맞으면 인정하고, 근거에 없는 사실은 환각으로 판단하세요."),
        ("human", """평가 역할: {role}
[질문 및 과제]
{question}

[기준 답안/기대 분석]
{ground_truth}

[검토 기준]
{criteria}

[검색 문맥]
{contexts}

[모델 답변]
{generated_answer}

각 항목을 1~5점으로 채점하세요: 전체 품질, 정확성, 완결성, 근거성, 명료성, 유용성, 역할 적합성. 환각 여부와 간단한 근거도 포함하세요. 점수 정의: 1 매우 부족, 2 부족, 3 부분 충족, 4 충족, 5 매우 우수.
{format_instructions}""")
    ])
    return prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser


def ragas_scores(dataset: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, float]]:
    rows, keys = [], []
    for i, item in enumerate(dataset):
        for raw_name, answer in item.get("models", {}).items():
            name = model_condition(raw_name)
            contexts = item.get("contexts", {}).get(raw_name, [])
            if answer and contexts and not str(answer).startswith("[ERROR]"):
                rows.append({"question": item["question"], "answer": answer, "contexts": contexts,
                             "reference": item.get("ground_truth", "")})
                keys.append((i, name))
    if not rows:
        logger.info("RAGAS skipped: no answer/context pairs")
        return {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        try:
            from ragas.metrics import faithfulness, context_precision
        except ImportError:
            from ragas.metrics.collections import Faithfulness, ContextPrecision
            faithfulness, context_precision = Faithfulness(), ContextPrecision()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("RAGAS skipped: ANTHROPIC_API_KEY is not set")
            return {}
        evaluator = ChatAnthropic(
            model=os.getenv("RAGAS_ANTHROPIC_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")),
            temperature=0.0, api_key=api_key)
        result = evaluate(Dataset.from_list(rows), metrics=[faithfulness, context_precision],
                          llm=LangchainLLMWrapper(evaluator), run_config=RunConfig(timeout=180)).to_pandas()
        output = {}
        for key, (_, row) in zip(keys, result.iterrows()):
            output[key] = {"Faithfulness": float(row.get("faithfulness", np.nan)),
                           "Context Precision": float(row.get("context_precision", np.nan))}
        return output
    except Exception:
        logger.exception("RAGAS evaluation failed")
        return {}


def mean_ci(values: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(v) < 2:
        return (np.nan, np.nan)
    half = t.ppf(.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
    return float(np.mean(v) - half), float(np.mean(v) + half)


def condition_summaries(rows: list[dict[str, Any]], dataset_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    metrics = [*SCORE_FIELDS, "ROUGE-L", "Faithfulness", "Context Precision", "Generation Latency (s)"]
    outputs = []
    groupings = [["Model"]]
    if "Topic" in frame and frame["Topic"].notna().any():
        groupings.append(["Topic", "Model"])
    for grouping in groupings:
        for keys, group in frame.groupby(grouping, dropna=False):
            if not isinstance(keys, tuple): keys = (keys,)
            base = dict(zip(grouping, keys))
            if "Topic" not in base: base["Topic"] = "ALL"
            for metric in metrics:
                if metric not in group: continue
                vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                if vals.empty: continue
                lo, hi = mean_ci(vals)
                outputs.append({"Dataset": dataset_name, "Score Source": "Claude", **base, "Metric": metric, "N": len(vals),
                                "Mean": vals.mean(), "SD": vals.std(ddof=1) if len(vals)>1 else np.nan,
                                "Median": vals.median(), "Q1": vals.quantile(.25), "Q3": vals.quantile(.75),
                                "P95": vals.quantile(.95), "CI95_Lower": lo, "CI95_Upper": hi})
            if "Hallucination" in group:
                h = group["Hallucination"].dropna()
                if len(h):
                    outputs.append({"Dataset": dataset_name, "Score Source": "Claude", **base, "Metric": "Hallucination Rate",
                                    "N": len(h), "Mean": h.astype(bool).mean(), "SD": np.nan,
                                    "Median": np.nan, "Q1": np.nan, "Q3": np.nan, "P95": np.nan,
                                    "CI95_Lower": np.nan, "CI95_Upper": np.nan})
    return pd.DataFrame(outputs)


def holm_adjust(p_values: list[float]) -> list[float]:
    valid = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [np.nan] * len(p_values)
    ordered = sorted(valid, key=lambda pair: pair[1])
    running = 0.0
    m = len(ordered)
    for rank, (idx, p) in enumerate(ordered):
        running = max(running, min(1.0, (m-rank) * p))
        adjusted[idx] = running
    return adjusted


def test_differences(diff: np.ndarray) -> dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return {"N": n, "Mean Difference": np.nan, "CI95_Lower": np.nan, "CI95_Upper": np.nan,
                "Paired_t": np.nan, "p_raw": np.nan, "Wilcoxon": np.nan, "p_wilcoxon": np.nan, "Cohen_dz": np.nan}
    avg = float(np.mean(diff)); sd = float(np.std(diff, ddof=1))
    half = float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    tt = ttest_1samp(diff,0.0)
    try:
        wx = wilcoxon(diff, zero_method="wilcox", alternative="two-sided") if np.any(diff != 0) else None
    except ValueError:
        wx = None
    return {"N":n,"Mean Difference":avg,"CI95_Lower":avg-half,"CI95_Upper":avg+half,
            "Paired_t":float(tt.statistic),"p_raw":float(tt.pvalue),
            "Wilcoxon":float(wx.statistic) if wx else np.nan,"p_wilcoxon":float(wx.pvalue) if wx else np.nan,
            "Cohen_dz":avg/sd if sd else np.nan}


def cluster_bootstrap_corr(frame: pd.DataFrame, x: str, y: str, reps: int = 1000) -> tuple[float,float,float,float]:
    """Percentile CIs resampling questions as clusters, preserving repeated model rows."""
    clusters=frame["Question_ID"].drop_duplicates().tolist()
    if len(clusters)<3: return (np.nan,np.nan,np.nan,np.nan)
    rng=np.random.default_rng(20261031)
    pearsons=[]; spearmans=[]
    by_id={qid:frame[frame.Question_ID==qid] for qid in clusters}
    for _ in range(reps):
        sampled=rng.choice(clusters,size=len(clusters),replace=True)
        sample=pd.concat([by_id[qid] for qid in sampled],ignore_index=True)
        a=pd.to_numeric(sample[x],errors="coerce"); b=pd.to_numeric(sample[y],errors="coerce")
        valid=a.notna() & b.notna(); a=a[valid]; b=b[valid]
        if len(a)<3 or a.nunique()<2 or b.nunique()<2: continue
        pearsons.append(pearsonr(a,b).statistic); spearmans.append(spearmanr(a,b).statistic)
    if not pearsons: return (np.nan,np.nan,np.nan,np.nan)
    return tuple(float(np.percentile(v,q)) for v in (pearsons,spearmans) for q in (2.5,97.5))


def planned_contrasts(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Paired, pre-specified comparisons by question ID; excludes missing pairs."""
    if frame.empty: return pd.DataFrame()
    if "Metric" not in frame:
        metric_columns=[m for m in [*SCORE_FIELDS,"ROUGE-L","Faithfulness","Context Precision"] if m in frame.columns]
        frame=frame.melt(id_vars=["Question_ID","Model"],value_vars=metric_columns,var_name="Metric",value_name="Value")
    elif "Value" not in frame:
        metric_columns=[m for m in [*SCORE_FIELDS,"ROUGE-L","Faithfulness","Context Precision"] if m in frame.columns]
        frame=frame.melt(id_vars=["Question_ID","Model","Metric"],value_vars=metric_columns,var_name="Score Metric",value_name="Value")
    outputs=[]
    score_metrics=[*SCORE_FIELDS,"ROUGE-L","Faithfulness","Context Precision"]
    available=set(frame["Model"])
    def paired_diff(metric: str, formula: dict[str,float]):
        maps={m:dict(zip(g["Question_ID"],pd.to_numeric(g["Value"],errors="coerce")))
              for m,g in frame[frame["Metric"]==metric].groupby("Model")}
        names=list(formula)
        if any(n not in maps for n in names): return None
        ids=set.intersection(*(set(maps[n]) for n in names))
        ids=[i for i in ids if all(np.isfinite(maps[n].get(i,np.nan)) for n in names)]
        if len(ids)<2:return None
        return np.array([sum(formula[n]*maps[n][i] for n in names) for i in ids],dtype=float)
    if dataset_name.startswith("student"):
        hypotheses={
            "RAG effect on Base": {"Base Model + RAG":1,"Base Model":-1},
            "Student LoRA effect without RAG": {"Base Model + Student LoRA":1,"Base Model":-1},
            "Student LoRA + RAG vs Base": {"Base Model + Student LoRA + RAG":1,"Base Model":-1},
            "RAG x Student LoRA interaction": {"Base Model + Student LoRA + RAG":1,"Base Model + Student LoRA":-1,"Base Model + RAG":-1,"Base Model":1},
            "Professor LoRA transfer to student questions": {"Base Model + Professor LoRA":1,"Base Model":-1},
            "Professor LoRA vs Student LoRA on student questions": {"Base Model + Professor LoRA":1,"Base Model + Student LoRA":-1},
        }
    else:
        hypotheses={
            "Professor LoRA effect on professor task": {"Base Model + Professor LoRA":1,"Base Model":-1},
            "Student LoRA transfer to professor task": {"Base Model + Student LoRA":1,"Base Model":-1},
            "Professor LoRA vs Student LoRA on professor task": {"Base Model + Professor LoRA":1,"Base Model + Student LoRA":-1},
        }
    for source, metric in itertools.product(["Claude"],score_metrics):
        if metric not in frame["Metric"].values: continue
        for hypothesis,formula in hypotheses.items():
            if not set(formula).issubset(available): continue
            diff=paired_diff(metric,formula)
            if diff is None: continue
            result=test_differences(diff)
            outputs.append({"Dataset":dataset_name,"Score Source":source,"Metric":metric,"Hypothesis":hypothesis,**result})
    # Human ratings are added by the caller using the same contrasts and mapping.
    result_frame=pd.DataFrame(outputs)
    if not result_frame.empty:
        result_frame["p_Holm"] = np.nan
        result_frame["p_wilcoxon_Holm"] = np.nan
        for _, ix in result_frame.groupby(["Score Source","Metric"]).groups.items():
            result_frame.loc[ix,"p_Holm"] = holm_adjust(result_frame.loc[ix,"p_raw"].tolist())
            result_frame.loc[ix,"p_wilcoxon_Holm"] = holm_adjust(result_frame.loc[ix,"p_wilcoxon"].tolist())
    return result_frame


def human_outputs(raw_rows: list[dict[str,Any]], human_path: Path, dataset_name: str) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if not human_path.exists(): return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    human=pd.read_csv(human_path)
    required={"question_id","rater_id","model",*HUMAN_FIELDS.values()}
    if not required.issubset(human.columns):
        logger.warning("Human CSV missing columns: %s", sorted(required-set(human.columns)))
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    human=human.dropna(subset=["question_id","rater_id","model"])
    for col in HUMAN_FIELDS.values(): human[col]=pd.to_numeric(human[col],errors="coerce")
    human["Model"]=human["model"].map(model_condition)
    # Keep this run's item IDs and conditions only.
    valid={(r["Question_ID"],r["Model"]) for r in raw_rows}
    human=human[human.apply(lambda r:(str(r["question_id"]),r["Model"]) in valid,axis=1)]
    hlong=human.melt(id_vars=["question_id","rater_id","Model"],value_vars=list(HUMAN_FIELDS.values()),var_name="Metric",value_name="Score")
    reverse={v:k for k,v in HUMAN_FIELDS.items()}; hlong["Metric"]=hlong["Metric"].map(reverse)
    hlong=hlong.dropna(subset=["Score"])
    human_summary=hlong.groupby(["question_id","Model","Metric"],as_index=False)["Score"].mean().rename(columns={"question_id":"Question_ID","Score":"Value"})
    # Rater agreement per question-condition-metric; Fleiss requires equal ratings per unit.
    agreements=[]
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa
        for (condition,metric),grp in hlong.groupby(["Model","Metric"]):
            counts=[]
            for _, unit in grp.groupby(["question_id"]):
                vals=unit["Score"].dropna().round().astype(int).tolist()
                if len(vals)<2 or any(v<1 or v>5 for v in vals): continue
                counts.append([vals.count(k) for k in range(1,6)])
            if len(counts)>=2 and len({sum(row) for row in counts})==1:
                agreements.append({"Dataset":dataset_name,"Model":condition,"Metric":metric,
                                   "N_Items":len(counts),"N_Raters_Per_Item":sum(counts[0]),
                                   "Fleiss_Kappa":float(fleiss_kappa(np.asarray(counts),method="fleiss"))})
    except Exception:
        logger.exception("Could not compute rater agreement")
    # Pair human mean with Claude metric; correlations are descriptive calibration checks.
    auto=pd.DataFrame(raw_rows)
    correlations=[]
    for metric in SCORE_FIELDS:
        h=human_summary[human_summary.Metric==metric]
        a=auto[["Question_ID","Model",metric]].rename(columns={metric:"Claude"})
        joined=h.merge(a,on=["Question_ID","Model"])
        joined=joined.dropna(subset=["Value","Claude"])
        if len(joined)>=3 and joined.Value.nunique()>1 and joined.Claude.nunique()>1:
            pr=pearsonr(joined.Claude,joined.Value); sr=spearmanr(joined.Claude,joined.Value)
            p_lo,p_hi,s_lo,s_hi=cluster_bootstrap_corr(joined,"Claude","Value")
            correlations.append({"Dataset":dataset_name,"Metric":metric,"N_Response_Pairs":len(joined),
                                 "Pearson_r":pr.statistic,"Pearson_p":pr.pvalue,"Pearson_CI95_Lower":p_lo,"Pearson_CI95_Upper":p_hi,
                                 "Spearman_rho":sr.statistic,"Spearman_p":sr.pvalue,"Spearman_CI95_Lower":s_lo,"Spearman_CI95_Upper":s_hi})
    return human_summary,pd.DataFrame(agreements),pd.DataFrame(correlations)


async def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Evaluation dataset not found: {DATASET_PATH}")
    dataset=json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset_name="student" if "student" in DATASET_PATH.name.lower() else "professor" if "professor" in DATASET_PATH.name.lower() else DATASET_PATH.stem
    prefix=f"{dataset_name}_"
    chain=create_judge_chain()
    jobs=[]; metas=[]
    for idx,item in enumerate(dataset):
        qid=question_id(item,idx)
        for raw_name,answer in item.get("models",{}).items():
            if not answer or str(answer).startswith("[ERROR]"):
                logger.warning("Skipping empty/failed answer: %s %s",qid,raw_name)
                continue
            condition=model_condition(raw_name)
            jobs.append(build_judge_input(item,raw_name,answer))
            metas.append((idx,qid,condition,raw_name,item,answer))
    if not jobs: raise SystemExit("No generated answers found in selected dataset")
    sem=asyncio.Semaphore(max(1,int(os.getenv("EVAL_MAX_CONCURRENCY","5"))))
    judged=await asyncio.gather(*(judge_one(chain,j,sem,i+1,len(jobs)) for i,j in enumerate(jobs)))
    ragas=await asyncio.to_thread(ragas_scores,dataset)
    rows=[]
    for meta,result in zip(metas,judged):
        idx,qid,condition,raw_name,item,answer=meta
        gt=item.get("ground_truth","")
        try: rouge_l=rouge.score(str(gt),str(answer))["rougeL"].fmeasure if gt and answer else np.nan
        except Exception: rouge_l=np.nan
        rscore=ragas.get((idx,condition),ragas.get((idx,raw_name),{}))
        latency=item.get("generation_latency_s",{}).get(raw_name,np.nan)
        row={"Dataset":dataset_name,"Question_ID":qid,"Question":item.get("question",""),
             "Ground_Truth":gt,"Topic":item.get("topic",item.get("scenario_category","")),
             "Base Model Identifier":os.getenv("VLLM_BASE_MODEL","not_recorded"),
             "Served Model Name":item.get("served_model_names",{}).get(raw_name,"not_recorded"),
             "Judge Model":os.getenv("ANTHROPIC_MODEL","claude-3-5-sonnet-20241022"),
             "Model":condition,"Generated_Answer":answer,"ROUGE-L":rouge_l,
             "Faithfulness":rscore.get("Faithfulness",np.nan),"Context Precision":rscore.get("Context Precision",np.nan),
             "Retrieved_Contexts":json.dumps(item.get("contexts",{}).get(raw_name,[]),ensure_ascii=False),
             "Generation Latency (s)":latency,"Hallucination":result.get("hallucination"),"Judge Reason":result.get("reason","")}
        for label,key in SCORE_FIELDS.items(): row[label]=result.get(key,np.nan)
        rows.append(row)
    raw=pd.DataFrame(rows)
    raw.to_csv(EVAL_DIR/f"{prefix}eval_results.csv",index=False,encoding="utf-8-sig")
    summary=condition_summaries(rows,dataset_name)
    tests=planned_contrasts(raw,dataset_name)
    hsummary,agreement,correlations=human_outputs(rows,EVAL_DIR/"human_evaluation.csv",dataset_name)
    if not hsummary.empty:
        # Long-format human scores are kept as a separate file for transparent aggregation.
        hsummary.to_csv(EVAL_DIR/f"{prefix}human_scores_by_item.csv",index=False,encoding="utf-8-sig")
        hs=[]
        for (condition,metric),group in hsummary.groupby(["Model","Metric"]):
            vals=pd.to_numeric(group["Value"],errors="coerce").dropna()
            if vals.empty: continue
            lo,hi=mean_ci(vals)
            hs.append({"Dataset":dataset_name,"Score Source":"Human","Model":condition,"Metric":metric,"N":len(vals),
                       "Mean":vals.mean(),"SD":vals.std(ddof=1) if len(vals)>1 else np.nan,"Median":vals.median(),
                       "Q1":vals.quantile(.25),"Q3":vals.quantile(.75),"P95":vals.quantile(.95),"CI95_Lower":lo,"CI95_Upper":hi})
        if hs: summary=pd.concat([summary,pd.DataFrame(hs)],ignore_index=True)
        # Add human paired contrasts.
        human_tests=planned_contrasts(hsummary,dataset_name)
        if not human_tests.empty:
            human_tests["Score Source"]="Human"
            if not tests.empty: tests=pd.concat([tests,human_tests],ignore_index=True)
            else: tests=human_tests
    if not summary.empty: summary.to_csv(EVAL_DIR/f"{prefix}condition_summary.csv",index=False,encoding="utf-8-sig")
    if not tests.empty: tests.to_csv(EVAL_DIR/f"{prefix}hypothesis_tests.csv",index=False,encoding="utf-8-sig")
    else: pd.DataFrame(columns=["Dataset","Score Source","Metric","Hypothesis","N","Mean Difference","CI95_Lower","CI95_Upper","Paired_t","p_raw","p_Holm","Wilcoxon","p_wilcoxon","p_wilcoxon_Holm","Cohen_dz"]).to_csv(EVAL_DIR/f"{prefix}hypothesis_tests.csv",index=False)
    if not agreement.empty: agreement.to_csv(EVAL_DIR/f"{prefix}human_rater_agreement.csv",index=False,encoding="utf-8-sig")
    if not correlations.empty: correlations.to_csv(EVAL_DIR/f"{prefix}human_judge_correlation.csv",index=False,encoding="utf-8-sig")
    logger.info("Wrote %seval_results.csv, %scondition_summary.csv, %shypothesis_tests.csv",prefix,prefix,prefix)
    if hsummary.empty: logger.info("No completed human ratings found; human-based tables will be generated after human_evaluation.csv is filled")


def cli() -> None:
    asyncio.run(main())


if __name__=="__main__":
    cli()
