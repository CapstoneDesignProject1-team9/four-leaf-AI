"""공지 QA 쌍에 실제 검색 자료를 붙여 RAG용 LoRA 학습 데이터(RAFT 방식)를 만든다.

기존 학습 데이터는 [질문]→[답변]만 있어서, 모델이 "처음 보는 질문이면 거절"을 배웠다.
여기서는 서빙과 같은 프롬프트(build_prompt)에 검색 자료 top-k를 넣어
- 정답 공지가 자료에 있으면(기본 80%) → 그 자료로 답하고,
- 정답 공지를 빼고 관련 없는 자료만 남기면(20%) → 거절하도록
가르친다. 정답 공지는 QA 쌍을 만든 원문 URL로 찾는다.

실행 (프로젝트 루트, 벡터 DB와 임베딩 모델이 있는 곳):
    python train/build_rag_sft.py
출력: data/rag_sft_train.jsonl, data/rag_sft_val.jsonl  ({"prompt": [...], "completion": [...]})
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from app.chains import rag_chain  # noqa: E402
from app.core.config import settings  # noqa: E402

REFUSAL = (
    "죄송해요. 제가 가진 학과 공지사항과 강의계획서 자료에서는 해당 내용을 찾지 못했어요. "
    "컴퓨터학부 홈페이지(https://computer.knu.ac.kr) 공지사항이나 학과 사무실에 확인해 주세요."
)
MAX_GOLD = 2  # 정답 공지에서 넣을 최대 청크 수


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def load_pairs(qa_path: str, sources_path: str) -> list[tuple[str, str, str]]:
    """(질문, 답변, 원문 URL). sources 파일은 공지별 n_pairs 순서로 QA 파일과 줄이 맞는다."""
    with open(qa_path, encoding="utf-8") as f:
        qa = [json.loads(line) for line in f if line.strip()]
    with open(sources_path, encoding="utf-8") as f:
        sources = [json.loads(line) for line in f if line.strip()]
    urls = [s["url"] for s in sources for _ in range(s["n_pairs"])]
    if len(urls) != len(qa):
        raise ValueError(f"QA {len(qa)}쌍과 sources n_pairs 합 {len(urls)}이 다릅니다.")
    return [(r["instruction"].strip(), r["output"].strip(), url) for r, url in zip(qa, urls)]


def make_example(
    answer: str,
    retrieved: list[Document],
    gold: list[Document],
    gold_url: str,
    with_oracle: bool,
    rng: random.Random,
    top_k: int,
) -> tuple[list[Document], str]:
    """검색 자료와 정답 공지로 학습용 (자료 목록, 정답 출력)을 만든다."""
    distractors = [d for d in retrieved if d.metadata.get("url") != gold_url]
    if not with_oracle:
        return distractors[:top_k], REFUSAL
    oracle = gold[:MAX_GOLD]
    docs = oracle + distractors[: top_k - len(oracle)]
    rng.shuffle(docs)  # 정답이 늘 1번에 오지 않게 해 위치에 기대지 않도록 한다.
    return docs, answer


def _retrieve(question: str, top_k: int, gold_url: str) -> tuple[list[Document], list[Document]]:
    notice, syllabi, embedder = rag_chain._ready()
    vector = embedder.embed_query(question)
    # 정답 공지를 빼도 top_k가 남도록 넉넉히 가져온다.
    docs = rag_chain._search_collection(notice, vector, top_k * 3)
    if syllabi is not None:
        docs += rag_chain._search_collection(syllabi, vector, top_k * 3)
    docs.sort(key=lambda d: d.metadata["distance"])
    result = notice.query(
        query_embeddings=[vector],
        n_results=MAX_GOLD,
        where={"$and": [{"url": gold_url}, {"active": True}]},
        include=["documents", "metadatas", "distances"],
    )
    gold = [
        Document(page_content=t, metadata={**m, "distance": float(d)})
        for t, m, d in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
    return docs, gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/notice_qa_dataset.jsonl")
    ap.add_argument("--sources", default="data/notice_qa_dataset.sources.jsonl")
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
    pairs = load_pairs(a.qa, a.sources)
    asyncio.run(rag_chain.init_vectorstore())

    # 같은 공지에서 나온 질문이 train/val에 갈라지지 않도록 URL 단위로 나눈다.
    urls = sorted({url for _, _, url in pairs})
    rng.shuffle(urls)
    val_urls = set(urls[: max(1, int(len(urls) * a.val_ratio))])

    rows = {"train": [], "val": []}
    stats = {"eval_overlap": 0, "no_gold": 0, "answer": 0, "refusal": 0, "gold_in_top_k": 0}
    for i, (question, answer, url) in enumerate(pairs, 1):
        if normalize(question) in excluded:
            stats["eval_overlap"] += 1
            continue
        retrieved, gold = _retrieve(question, top_k, url)
        if not gold:  # 원문 공지가 현재 색인에 없음
            stats["no_gold"] += 1
            continue
        stats["gold_in_top_k"] += any(d.metadata.get("url") == url for d in retrieved[:top_k])
        docs, completion = make_example(
            answer, retrieved, gold, url, rng.random() < a.p_oracle, rng, top_k
        )
        stats["answer" if completion != REFUSAL else "refusal"] += 1
        rows["val" if url in val_urls else "train"].append(
            {
                "prompt": [{"role": "user", "content": rag_chain.build_prompt(question, docs)}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}")

    for split, items in rows.items():
        path = f"{a.out_prefix}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{path}: {len(items)}개")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
