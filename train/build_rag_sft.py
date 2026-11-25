"""QA 쌍에 실제 검색 자료를 붙여 RAG용 LoRA 학습 데이터(RAFT 방식)를 만든다.

기존 학습 데이터는 [질문]→[답변]만 있어서, 모델이 "처음 보는 질문이면 거절"을 배웠다.
여기서는 서빙과 같은 프롬프트(build_prompt)에 검색 자료 top-k를 넣어
- 정답 문서가 자료에 있으면(기본 80%) → 그 자료로 답하고 출처를 밝히고,
- 정답 문서를 빼고 관련 없는 자료만 남기면(20%) → 거절하도록
가르친다. 검색 자료는 서빙(rag_chain.retrieve)과 같은 후보 검색·분반·학기 필터를 거친다.

입력
- data/notice_qa_dataset.jsonl (+ .sources.jsonl): 공지 QA. 정답 공지는 원문 URL로 찾는다.
- --generated output/synthetic_qa_dataset.jsonl: scripts/generate_qa_student.py 결과(있으면).
  공지는 post_id(wr_id), 강의계획서는 course_code로 정답 문서를 찾는다.
  '자료없음' 범주는 이미 거절·부분 답변이므로 그대로 쓴다(문서부재: 정답 문서 포함, 범위밖: 검색 자료만).
- data/rag_sft_rewrites.jsonl: 챗봇 말투로 다시 쓴 답변(질문 → 답변). 있으면 원래 답변 대신 쓴다.

실행 (프로젝트 루트, 벡터 DB와 임베딩 모델이 있는 곳):
    python train/build_rag_sft.py [--generated output/synthetic_qa_dataset.jsonl]
출력: data/rag_sft_train.jsonl, data/rag_sft_val.jsonl  ({"prompt": [...], "completion": [...]})
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from app.chains import rag_chain  # noqa: E402
from app.chains.semester import Term, current_term, parse_question  # noqa: E402
from app.core.config import settings  # noqa: E402

REFUSAL = (
    "죄송해요. 제가 가진 학과 공지사항과 강의계획서 자료에서는 해당 내용을 찾지 못했어요. "
    "컴퓨터학부 홈페이지(https://computer.knu.ac.kr) 공지사항이나 학과 사무실에 확인해 주세요."
)
MAX_GOLD = 2  # 정답 문서에서 넣을 최대 청크 수
ABSTAIN = "자료없음"  # generate_qa_student.py의 거절 범주


@dataclass
class Item:
    question: str
    answer: str
    mode: str  # qa: 80/20 정답 포함·거절, abstain_doc: 정답 문서 포함·답 그대로, out_of_scope: 검색 자료만
    collection: str | None = None  # notice | syllabi
    where: dict | None = None  # 정답 문서 조건


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def load_pairs(qa_path: str, sources_path: str) -> list[Item]:
    """공지 QA. sources 파일은 공지별 n_pairs 순서로 QA 파일과 줄이 맞는다."""
    with open(qa_path, encoding="utf-8") as f:
        qa = [json.loads(line) for line in f if line.strip()]
    with open(sources_path, encoding="utf-8") as f:
        sources = [json.loads(line) for line in f if line.strip()]
    urls = [s["url"] for s in sources for _ in range(s["n_pairs"])]
    if len(urls) != len(qa):
        raise ValueError(f"QA {len(qa)}쌍과 sources n_pairs 합 {len(urls)}이 다릅니다.")
    return [
        Item(r["instruction"].strip(), r["output"].strip(), "qa", "notice", {"url": url})
        for r, url in zip(qa, urls)
    ]


def load_generated(path: str, url_by_wr_id: dict[str, str]) -> tuple[list[Item], int]:
    """generate_qa_student.py 결과. 정답 문서를 찾을 수 없는 줄 수도 함께 돌려준다."""
    items, unmatched = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question, answer = row["instruction"].strip(), row["output"].strip()
            kind = row.get("source") if row.get("category") == ABSTAIN else "qa"
            if kind == "범위 밖":
                items.append(Item(question, answer, "out_of_scope"))
                continue
            mode = "abstain_doc" if row.get("category") == ABSTAIN else "qa"
            doc_kind = kind if mode == "abstain_doc" else row.get("source")
            if doc_kind == "강의계획서" and row.get("course_code"):
                items.append(
                    Item(question, answer, mode, "syllabi", {"course_code": row["course_code"]})
                )
            elif doc_kind == "공지사항" and str(row.get("post_id")) in url_by_wr_id:
                url = url_by_wr_id[str(row["post_id"])]
                items.append(Item(question, answer, mode, "notice", {"url": url}))
            else:
                unmatched += 1
    return items, unmatched


def notice_urls_by_wr_id() -> dict[str, str]:
    """공지 원본 post_id(게시판 wr_id) → 벡터 DB의 URL. 게시판끼리 번호가 겹치면 뺀다."""
    notice, _, _ = rag_chain._ready()
    found: dict[str, set[str]] = {}
    for meta in notice.get(include=["metadatas"])["metadatas"]:
        if m := re.search(r"wr_id=(\d+)", str(meta.get("url", ""))):
            found.setdefault(m.group(1), set()).add(meta["url"])
    return {wr_id: urls.pop() for wr_id, urls in found.items() if len(urls) == 1}


def make_example(
    answer: str,
    retrieved: list[Document],
    gold: list[Document],
    gold_url: str,
    with_oracle: bool,
    rng: random.Random,
    top_k: int,
) -> tuple[list[Document], str]:
    """검색 자료와 정답 문서로 학습용 (자료 목록, 정답 출력)을 만든다."""
    distractors = [d for d in retrieved if d.metadata.get("url") != gold_url]
    if not with_oracle:
        return distractors[:top_k], REFUSAL
    oracle = gold[:MAX_GOLD]
    docs = oracle + distractors[: top_k - len(oracle)]
    rng.shuffle(docs)  # 정답이 늘 1번에 오지 않게 해 위치에 기대지 않도록 한다.
    return docs, answer


def with_source(answer: str, gold: Document) -> str:
    """서빙 지침('사용한 자료의 제목과 URL')대로 끝에 출처 줄을 붙인다. 굵은 글씨는 지운다."""
    answer = answer.replace("**", "").strip()
    if "참고:" in answer:
        return answer
    return f"{answer}\n\n참고: {gold.metadata.get('title', '')} ({gold.metadata.get('url', '')})"


def serving_question(
    question: str, gold_term: Term | None, today: date | None = None
) -> tuple[str, Term, bool] | None:
    """서빙에서 이 질문이 정답 문서에 닿는 모습으로 (질문, 기준 학기, 학기 명시 여부)를 만든다.

    - 질문에 학기가 있으면 그대로. 정답 문서와 학기가 어긋나면 None(학습에서 뺀다).
    - 학기가 없는데 정답 문서가 특정 학기 문서면, 서빙에서는 학기를 되물은 뒤
      학생 답을 합친 질문('… 2025학년도 1학기')으로 답하므로 그 모습으로 만든다.
    - 둘 다 아니면(학기 무관 문서) 현재 학기 기준 그대로.
    """
    asked = parse_question(question, today)
    if asked is not None:
        if gold_term is not None and rag_chain._excluded(gold_term, asked, True):
            return None
        return question, asked, True
    if gold_term is not None and gold_term.season is not None:
        return f"{question} {gold_term.label()}", gold_term, True
    return question, current_term(today), False


def _gold(vector: list[float], collection: str, where: dict) -> list[Document]:
    notice, syllabi, _ = rag_chain._ready()
    target = notice if collection == "notice" else syllabi
    if target is None:
        return []
    key, value = next(iter(where.items()))
    result = target.query(
        query_embeddings=[vector],
        n_results=MAX_GOLD,
        where={"$and": [{key: value}, {"active": True}]},
        include=["documents", "metadatas", "distances"],
    )
    return [
        Document(page_content=t, metadata={**m, "distance": float(d)})
        for t, m, d in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def _distractors(
    question: str, target: Term, explicit: bool, gold_url: str
) -> tuple[list[Document], bool]:
    """서빙과 같은 후보 검색·분반·학기 필터를 거친 top_k에서 정답 문서만 뺀 자료와,
    서빙에서 정답 문서가 top_k에 들었는지."""
    candidates = rag_chain.gather_candidates(question, target, explicit)
    candidates, _ = rag_chain.pick_section(candidates, question)
    served = rag_chain.filter_by_term(candidates, target, explicit).documents
    others = [d for d in candidates if d.metadata.get("url") != gold_url]
    distractors = rag_chain.filter_by_term(others, target, explicit).documents
    return distractors, any(d.metadata.get("url") == gold_url for d in served)


def is_val(key: str, ratio: float) -> bool:
    """같은 문서에서 나온 질문은 같은 쪽에 두고, 입력이 늘어도 기존 분할이 바뀌지 않게 해시로 나눈다."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < ratio * 1000


def load_rewrites(path: str) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["question"]: r["answer"] for r in map(json.loads, f) if r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/notice_qa_dataset.jsonl")
    ap.add_argument("--sources", default="data/notice_qa_dataset.sources.jsonl")
    ap.add_argument(
        "--generated", help="generate_qa_student.py 결과 (output/synthetic_qa_dataset.jsonl)"
    )
    ap.add_argument("--rewrites", default="data/rag_sft_rewrites.jsonl")
    ap.add_argument("--exclude", default="eval/student_eval_dataset.json", help="평가셋 질문 제외")
    ap.add_argument("--out-prefix", default="data/rag_sft")
    ap.add_argument("--p-oracle", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    top_k = settings.RAG_TOP_K

    with open(a.exclude, encoding="utf-8") as f:
        excluded = {normalize(r["question"]) for r in json.load(f)}
    rewrites = load_rewrites(a.rewrites)
    asyncio.run(rag_chain.init_vectorstore())
    items = load_pairs(a.qa, a.sources)
    stats = dict.fromkeys(
        ["eval_overlap", "duplicate", "no_gold", "term_mismatch", "term_appended", "rewritten"], 0
    )
    stats |= {"answer": 0, "refusal": 0, "abstain_doc": 0, "out_of_scope": 0, "gold_in_top_k": 0}
    if a.generated:
        generated, stats["generated_unmatched"] = load_generated(
            a.generated, notice_urls_by_wr_id()
        )
        items += generated

    rows = {"train": [], "val": []}
    seen: set[str] = set()
    _, _, embedder = rag_chain._ready()
    for i, item in enumerate(items, 1):
        key = normalize(item.question)
        if key in excluded:
            stats["eval_overlap"] += 1
            continue
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        if item.mode == "out_of_scope":
            # 범위 밖 질문: 서빙에서 실제로 들어올 검색 자료를 보고 거절하도록 가르친다.
            retrieval = rag_chain.retrieve(item.question)
            question, docs, completion, group = (
                item.question,
                retrieval.documents,
                item.answer,
                item.question,
            )
            source = None
            stats["out_of_scope"] += 1
        else:
            gold = _gold(embedder.embed_query(item.question), item.collection, item.where)
            if not gold:  # 원문 문서가 현재 색인에 없음
                stats["no_gold"] += 1
                continue
            served = serving_question(item.question, rag_chain.doc_term(gold[0]))
            if served is None:
                stats["term_mismatch"] += 1
                continue
            stats["term_appended"] += served[0] != item.question
            question, target, explicit = served
            answer = rewrites.get(question, item.answer)
            stats["rewritten"] += question in rewrites
            gold_url = gold[0].metadata.get("url", "")
            distractors, gold_hit = _distractors(question, target, explicit, gold_url)
            stats["gold_in_top_k"] += gold_hit
            with_oracle = item.mode == "abstain_doc" or rng.random() < a.p_oracle
            docs, completion = make_example(
                answer, distractors, gold, gold_url, with_oracle, rng, top_k
            )
            if completion != REFUSAL:
                completion = with_source(completion, gold[0])
            stats[
                "abstain_doc"
                if item.mode == "abstain_doc"
                else "answer"
                if completion != REFUSAL
                else "refusal"
            ] += 1
            group = gold_url
            source = {"title": gold[0].metadata.get("title", ""), "url": gold_url}

        rows["val" if is_val(group, a.val_ratio) else "train"].append(
            {
                "prompt": [{"role": "user", "content": rag_chain.build_prompt(question, docs)}],
                "completion": [{"role": "assistant", "content": completion}],
                # 학습에는 쓰지 않는다. 답변을 다시 쓸 때 출처 표시에 쓴다.
                "source": source,
            }
        )
        if i % 100 == 0:
            print(f"  {i}/{len(items)}")

    for split, split_rows in rows.items():
        path = f"{a.out_prefix}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(split_rows)}개")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
