"""Build a mixed SFT set: notices/syllabi/out-of-scope (new) + cleaned reviews (subsampled)."""
import argparse
import json
import random
import re

JUNK = re.compile(r"scanned_|completeness|unverified|리뷰 요소 없음|수집을 시도|로딩 지연|화면 변경 여부")
OPENING = "후기에 따르면"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="notice/syllabus/out-of-scope jsonl")
    ap.add_argument("--reviews", required=True, help="everytime review jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reviews-n", type=int, default=3000)
    ap.add_argument("--max-opening-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    docs = [{"instruction": r["instruction"], "output": r["output"]} for r in load(a.docs)]

    seen, rev = set(), []
    for r in load(a.reviews):
        q, o = r["instruction"].strip(), r["output"].strip()
        key = re.sub(r"\s+", "", q)
        if key in seen or JUNK.search(q + o) or len(q) < 5 or len(o) < 30:
            continue
        seen.add(key)
        rev.append({"instruction": q, "output": o})
    rnd.shuffle(rev)
    cap = int(a.reviews_n * a.max_opening_ratio)
    picked, n_open = [], 0
    for r in rev:
        if OPENING in r["output"]:
            if n_open >= cap:
                continue
            n_open += 1
        picked.append(r)
        if len(picked) >= a.reviews_n:
            break

    out = docs + picked
    rnd.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"docs={len(docs)} reviews={len(picked)} (opening={n_open}) total={len(out)}")


if __name__ == "__main__":
    main()
