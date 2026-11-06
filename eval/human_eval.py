"""학생용 인간평가: 맹검 평가 시트 생성(sheet)과 정확성·응답 유형 분석(analyze).

평가 항목은 두 가지입니다(평가 기준은 eval/HUMAN_EVAL_GUIDE.md).
- accuracy: 정확성 1~5
- response_type: 1 정답·부분정답, 2 답변 유보, 3 다른 학기 정보 사용, 4 근거 없는 사실 날조

사용법 (four-leaf-AI/에서):
    python eval/human_eval.py sheet
    # 평가자는 템플릿을 eval/human_eval/ratings/R1.csv 처럼 복사해 채웁니다.
    python eval/human_eval.py analyze
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel, wilcoxon

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "student_eval_dataset.json"
DEFAULT_OUT = EVAL_DIR / "human_eval"
CONDITIONS = [
    "Base Model",
    "Base Model + RAG",
    "Base Model + Student LoRA",
    "Base Model + Professor LoRA",
]
SHORT = {
    "Base Model": "Base",
    "Base Model + RAG": "RAG",
    "Base Model + Student LoRA": "StudentLoRA",
    "Base Model + Professor LoRA": "ProfessorLoRA",
}
TYPES = {1: "correct", 2: "abstain", 3: "other_semester", 4: "fabricated"}
TYPE_KO = {1: "정답·부분정답", 2: "답변 유보", 3: "다른 학기 정보 사용", 4: "근거 없는 사실 날조"}
CONTRASTS = [
    ("Base Model + RAG", "Base Model"),
    ("Base Model + Student LoRA", "Base Model"),
    ("Base Model + Professor LoRA", "Base Model"),
    ("Base Model + RAG", "Base Model + Student LoRA"),
]
SEED = 20261106


# ───────────────────────────── sheet ─────────────────────────────
def build_sheet(dataset_path: Path, out_dir: Path, seed: int, allow_missing: bool) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    problems = []
    for item in dataset:
        for cond in CONDITIONS:
            answer = str((item.get("models") or {}).get(cond, "")).strip()
            if not answer or answer.startswith("[ERROR]"):
                problems.append(f"{item['question_id']} / {cond}")
    if problems and not allow_missing:
        sys.exit(
            f"응답이 비었거나 오류인 항목 {len(problems)}개: {problems[:5]} ... "
            "생성을 마친 뒤 다시 실행하세요(--allow-missing은 점검용)."
        )

    rng = random.Random(seed)
    labels = "ABCD"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ratings").mkdir(exist_ok=True)
    key_rows, template_rows = [], []
    lines = [
        "# 학생 챗봇 응답 인간평가 시트 (2026학년도 2학기)",
        "",
        "- 각 문항의 응답 A~D를 읽고 **정확성(1~5)** 과 **응답 유형(1~4)** 을 기록합니다.",
        "- 기준은 `eval/HUMAN_EVAL_GUIDE.md`를 따릅니다. 모델 이름은 가려져 있고 문항마다 순서가 다릅니다.",
        "- 응답 유형: 1 정답·부분정답 / 2 답변 유보 / 3 다른 학기 정보 사용 / 4 근거 없는 사실 날조",
        "- 기록은 `ratings/본인ID.csv`(템플릿 복사)에 합니다. `student_human_eval_key.csv`는 평가가 끝날 때까지 열지 마세요.",
        "",
    ]
    for no, item in enumerate(dataset, 1):
        qid = item["question_id"]
        order = CONDITIONS[:]
        rng.shuffle(order)
        lines += [
            "---",
            "",
            f"## 문항 {no:02d} · {qid} · {item.get('topic', '')}",
            "",
            f"**질문** {item['question']}",
            "",
            f"**정답(Ground Truth)** {item['ground_truth']}",
            "",
            "**평가 유의사항**",
            *[f"- {c}" for c in item.get("evaluation_criteria", [])],
            "",
        ]
        for label, cond in zip(labels, order):
            key_rows.append({"question_id": qid, "blind_label": label, "model": cond})
            template_rows.append(
                {
                    "question_id": qid,
                    "blind_label": label,
                    "rater_id": "",
                    "accuracy": "",
                    "response_type": "",
                    "notes": "",
                }
            )
            answer = str((item.get("models") or {}).get(cond, "")).strip() or "(응답 없음)"
            lines += [f"### 응답 {label}", "", answer, ""]

    (out_dir / "student_human_eval_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(key_rows).to_csv(
        out_dir / "student_human_eval_key.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(template_rows).to_csv(
        out_dir / "student_human_ratings_template.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"시트 {len(dataset)}문항 × {len(CONDITIONS)}조건 = {len(template_rows)}건 → {out_dir}"
        + (f" (빈 응답 {len(problems)}건 포함)" if problems else "")
    )


# ──────────────────────────── analyze ────────────────────────────
def load_ratings(ratings_dir: Path, key_path: Path) -> pd.DataFrame:
    files = sorted(ratings_dir.glob("*.csv"))
    if not files:
        sys.exit(f"평가 파일이 없습니다: {ratings_dir}/*.csv")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"question_id": str, "blind_label": str, "rater_id": str})
        df["rater_id"] = df["rater_id"].fillna("").astype(str).str.strip()
        df.loc[df["rater_id"] == "", "rater_id"] = f.stem
        frames.append(df)
    r = pd.concat(frames, ignore_index=True)
    r["accuracy"] = pd.to_numeric(r["accuracy"], errors="coerce")
    r["response_type"] = pd.to_numeric(r["response_type"], errors="coerce")
    bad = r[~r["accuracy"].isin([1, 2, 3, 4, 5]) | ~r["response_type"].isin(list(TYPES))]
    if len(bad):
        print(f"[경고] 값이 비었거나 범위 밖인 행 {len(bad)}개는 제외합니다.", file=sys.stderr)
        print(bad[["rater_id", "question_id", "blind_label"]].head(10).to_string(), file=sys.stderr)
    r = r.drop(bad.index)
    r[["accuracy", "response_type"]] = r[["accuracy", "response_type"]].astype(int)
    dup = r.duplicated(["rater_id", "question_id", "blind_label"], keep=False)
    if dup.any():
        sys.exit(f"같은 평가자·문항·응답이 중복된 행이 있습니다:\n{r[dup].head(10).to_string()}")
    key = pd.read_csv(key_path, dtype=str)
    r = r.merge(key, on=["question_id", "blind_label"], how="left", validate="many_to_one")
    if r["model"].isna().any():
        sys.exit(
            "키 파일과 맞지 않는 문항/응답 라벨이 있습니다. 시트를 다시 만들었는지 확인하세요."
        )
    expected = len(key)
    for rater, grp in r.groupby("rater_id"):
        if len(grp) != expected:
            print(f"[경고] {rater}: {len(grp)}/{expected}건만 평가됨", file=sys.stderr)
    return r


def majority_type(votes: list[int]) -> str:
    counts = pd.Series(votes).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "no_majority"
    return TYPES[int(counts.index[0])]


def by_item(r: pd.DataFrame, scope_of: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (qid, model), g in r.groupby(["question_id", "model"]):
        votes = g["response_type"].tolist()
        row = {
            "question_id": qid,
            "model": model,
            "scope": scope_of.get(qid, "in_scope"),
            "n_raters": len(g),
            "accuracy_mean": g["accuracy"].mean(),
            "accurate": int(g["accuracy"].mean() >= 4),
            "type_majority": majority_type(votes),
        }
        for code, name in TYPES.items():
            row[f"votes_{name}"] = votes.count(code)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(items: pd.DataFrame, r: pd.DataFrame, scope_of: dict[str, str]) -> pd.DataFrame:
    r = r.assign(scope=r["question_id"].map(scope_of).fillna("in_scope"))
    rows = []
    for scope in ("all", "in_scope", "out_of_scope"):
        it = items if scope == "all" else items[items.scope == scope]
        rr = r if scope == "all" else r[r.scope == scope]
        for model in CONDITIONS:
            m, mr = it[it.model == model], rr[rr.model == model]
            if m.empty:
                continue
            row = {
                "scope": scope,
                "model": model,
                "n_items": len(m),
                "accuracy_mean": m.accuracy_mean.mean(),
                "accuracy_sd": m.accuracy_mean.std(ddof=1),
                "accuracy_rate_pct": 100 * m.accurate.mean(),
            }
            for name in [*TYPES.values(), "no_majority"]:
                row[f"{name}_pct"] = 100 * (m.type_majority == name).mean()
            for code, name in TYPES.items():
                row[f"rater_{name}_pct"] = 100 * (mr.response_type == code).mean()
            rows.append(row)
    out = pd.DataFrame(rows).rename(
        columns={
            "fabricated_pct": "hallucination_pct",
            "rater_fabricated_pct": "rater_hallucination_pct",
        }
    )
    # 답변 유보 성능: 범위 밖 문항에서의 유보는 올바른 행동, 답할 수 있는 문항에서의 유보는 오류.
    abst = []
    for model in CONDITIONS:
        m = items[items.model == model]
        if m.empty:
            continue
        pred = m.type_majority == "abstain"
        oos = m.scope == "out_of_scope"
        tp, fp, fn = int((pred & oos).sum()), int((pred & ~oos).sum()), int((~pred & oos).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        abst.append(
            {
                "scope": "abstention",
                "model": model,
                "n_items": len(m),
                "oos_abstain_rate_pct": 100 * rec if rec == rec else np.nan,
                "in_scope_false_abstain_pct": 100 * fp / max(int((~oos).sum()), 1),
                "abstain_precision": prec,
                "abstain_recall": rec,
                "abstain_f1": f1,
            }
        )
    return pd.concat([out, pd.DataFrame(abst)], ignore_index=True)


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def tests(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "in_scope"):
        it = items if scope == "all" else items[items.scope == "in_scope"]
        acc = it.pivot(index="question_id", columns="model", values="accuracy_mean")
        hal = it.assign(h=(it.type_majority == "fabricated").astype(int)).pivot(
            index="question_id", columns="model", values="h"
        )
        block = []
        for a, b in CONTRASTS:
            if a not in acc or b not in acc:
                continue
            pair = acc[[a, b]].dropna()
            diff = (pair[a] - pair[b]).to_numpy()
            sd = diff.std(ddof=1)
            p_w = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            block.append(
                {
                    "scope": scope,
                    "metric": "accuracy",
                    "contrast": f"{SHORT[a]} - {SHORT[b]}",
                    "n_items": len(diff),
                    "mean_diff": diff.mean(),
                    "cohen_dz": diff.mean() / sd if sd > 0 else np.nan,
                    "wilcoxon_p": p_w,
                    "paired_t_p": ttest_rel(pair[a], pair[b]).pvalue if sd > 0 else np.nan,
                }
            )
            hp = hal[[a, b]].dropna()
            only_a, only_b = (
                int(((hp[a] == 1) & (hp[b] == 0)).sum()),
                int(((hp[a] == 0) & (hp[b] == 1)).sum()),
            )
            block.append(
                {
                    "scope": scope,
                    "metric": "hallucination",
                    "contrast": f"{SHORT[a]} - {SHORT[b]}",
                    "n_items": len(hp),
                    "mean_diff": 100 * (hp[a].mean() - hp[b].mean()),
                    "discordant_a_only": only_a,
                    "discordant_b_only": only_b,
                    "mcnemar_exact_p": binomtest(only_a, only_a + only_b).pvalue
                    if only_a + only_b
                    else 1.0,
                }
            )
        for metric, pcol in (("accuracy", "wilcoxon_p"), ("hallucination", "mcnemar_exact_p")):
            sel = [x for x in block if x["metric"] == metric]
            for x, adj in zip(sel, holm([x[pcol] for x in sel])):
                x["holm_p"] = adj
        rows += block
    return pd.DataFrame(rows)


def kappa(a: np.ndarray, b: np.ndarray, cats: list[int], quadratic: bool) -> float:
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = ((i - j) / (k - 1)) ** 2 if quadratic else (i != j).astype(float)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else np.nan


def fleiss(r: pd.DataFrame, col: str, cats: list[int]) -> tuple[float, int]:
    units = r.groupby(["question_id", "model"])[col].apply(list)
    n = units.map(len).max()
    units = units[units.map(len) == n]
    if n < 2 or len(units) < 2:
        return np.nan, len(units)
    counts = np.array([[v.count(c) for c in cats] for v in units])
    p_i = ((counts**2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = (p_j**2).sum()
    return (float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else np.nan), len(units)


def agreement(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = (("accuracy", [1, 2, 3, 4, 5], True), ("response_type", list(TYPES), False))
    for group, sub in [("all", r), *((SHORT[m], r[r.model == m]) for m in CONDITIONS)]:
        if sub.empty:
            continue
        for col, cats, quad in specs:
            wide = sub.pivot_table(
                index=["question_id", "model"], columns="rater_id", values=col, aggfunc="first"
            )
            for r1, r2 in itertools.combinations(wide.columns, 2):
                pair = wide[[r1, r2]].dropna().astype(int)
                rows.append(
                    {
                        "model": group,
                        "metric": col,
                        "statistic": "QWK" if quad else "Cohen_kappa",
                        "raters": f"{r1}-{r2}",
                        "n_units": len(pair),
                        "value": kappa(pair[r1].to_numpy(), pair[r2].to_numpy(), cats, quad)
                        if len(pair)
                        else np.nan,
                    }
                )
            value, n_units = fleiss(sub, col, cats)
            rows.append(
                {
                    "model": group,
                    "metric": col,
                    "statistic": "Fleiss_kappa",
                    "raters": "all",
                    "n_units": n_units,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def analyze(dataset_path: Path, out_dir: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    scope_of = {
        it["question_id"]: "out_of_scope"
        if it.get("reference_type") == "abstention"
        else "in_scope"
        for it in dataset
    }
    r = load_ratings(out_dir / "ratings", out_dir / "student_human_eval_key.csv")
    items = by_item(r, scope_of)
    outputs = {
        "student_human_by_item.csv": items,
        "student_human_summary.csv": summarize(items, r, scope_of),
        "student_human_tests.csv": tests(items),
        "student_human_agreement.csv": agreement(r),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(
        f"평가자 {r.rater_id.nunique()}명, 평가 {len(r)}건 → "
        + ", ".join(str(out_dir / n) for n in outputs)
    )
    s = outputs["student_human_summary.csv"]
    view = s[s.scope == "all"][
        [
            "model",
            "n_items",
            "accuracy_mean",
            "accuracy_rate_pct",
            "hallucination_pct",
            "abstain_pct",
            "other_semester_pct",
        ]
    ]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="맹검 평가 시트·평가 템플릿·키 파일 생성")
    s.add_argument("--allow-missing", action="store_true", help="빈 응답이 있어도 생성(점검용)")
    s.add_argument("--seed", type=int, default=SEED)
    a = sub.add_parser("analyze", help="ratings/*.csv를 분석해 out-dir에 student_human_*.csv 생성")
    for p in (s, a):
        p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.cmd == "sheet":
        build_sheet(args.dataset, args.out_dir, args.seed, args.allow_missing)
    else:
        analyze(args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
