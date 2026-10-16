"""공지사항 청크에서 Q&A 학습 데이터를 생성합니다.

scripts/generate_qa.py 의 generate_synthetic_data()를 재사용하므로
API 키/모델 설정은 기존 스크립트와 동일합니다.

실행 (프로젝트 루트, 인터넷이 되는 로그인 노드에서):
    python scripts/generate_qa_from_notices.py --dry-run          # API 호출 없이 선별 결과만 확인
    python scripts/generate_qa_from_notices.py --num-docs 20      # 소량 테스트
    python scripts/generate_qa_from_notices.py --num-docs 300     # 본 생성

결과:
    data/notice_qa_dataset.jsonl           {"instruction", "output"}  (학습용)
    data/notice_qa_dataset.sources.jsonl   출처(공지 제목/URL)와 진행 기록 (이어하기용)
"""

import argparse
import json
import os
import random
import re
import sys
import time

DEFAULT_INPUT = "knowledge/processed/notices/knu_notice_chunks.jsonl"
DEFAULT_OUTPUT = "data/notice_qa_dataset.jsonl"

MIN_BODY_CHARS = 150
MIN_HANGUL_RATIO = 0.4
MAX_DIGIT_RATIO = 0.3


def body_of(chunk_text):
    """'제목/게시일/첨부파일' 헤더를 제외한 본문."""
    return chunk_text.split("\n\n", 1)[-1]


def hangul_ratio(text):
    if not text:
        return 0.0
    return len(re.findall(r"[가-힣]", text)) / len(text)


def is_good_chunk(row, since):
    """OCR 깨짐/표 조각/숫자 위주 청크를 제외합니다."""
    meta = row.get("metadata", {})
    flags = meta.get("quality_flags") or ""
    if "fragmented_lines" in flags or "ocr_review_recommended" in flags:
        return False
    if meta.get("source_type") == "image_ocr":
        return False
    if (meta.get("date") or "") < since:
        return False

    body = body_of(row["chunk_text"])
    if len(body) < MIN_BODY_CHARS:
        return False
    if hangul_ratio(body) < MIN_HANGUL_RATIO:
        return False
    digit_ratio = sum(c.isdigit() for c in body) / len(body)
    return digit_ratio <= MAX_DIGIT_RATIO


def select_chunks(path, since, num_docs, seed, done_docs):
    """문서당 가장 앞쪽의 좋은 청크 1개를 고른 뒤 무작위로 num_docs개 추립니다."""
    best = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row["document_id"]
            if doc_id in done_docs or not is_good_chunk(row, since):
                continue
            if doc_id not in best or row["chunk_index"] < best[doc_id]["chunk_index"]:
                best[doc_id] = row
    rows = list(best.values())
    random.Random(seed).shuffle(rows)
    return rows[:num_docs], len(best)


def normalize(text):
    return re.sub(r"[\s\W_]+", "", text.lower())


def clean_pair(pair):
    """형식이 이상한 Q&A를 걸러냅니다. (instruction, output) 또는 None."""
    question = (pair.get("instruction") or "").strip()
    answer = (pair.get("output") or "").strip()
    if not (8 <= len(question) <= 200):
        return None
    if not (20 <= len(answer) <= 1000):
        return None
    if hangul_ratio(answer) < 0.3:
        return None
    return question, answer


def load_progress(output_path, sources_path):
    seen_questions = set()
    done_docs = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                seen_questions.add(normalize(json.loads(line).get("instruction", "")))
    if os.path.exists(sources_path):
        with open(sources_path, encoding="utf-8") as f:
            for line in f:
                done_docs.add(json.loads(line)["document_id"])
    return seen_questions, done_docs


def load_generator():
    # scripts/ 폴더가 sys.path[0]이므로 같은 폴더의 generate_qa를 바로 import 할 수 있습니다.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_qa import generate_synthetic_data

    return generate_synthetic_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-docs", type=int, default=300, help="사용할 공지(문서) 수")
    parser.add_argument("--pairs-per-doc", type=int, default=2, help="문서당 생성할 Q&A 수")
    parser.add_argument("--since", default="2025-01-01", help="이 날짜 이후 공지만 사용")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=1.0, help="API 호출 간 대기(초)")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 선별만 확인")
    args = parser.parse_args()

    sources_path = args.output.replace(".jsonl", ".sources.jsonl")
    seen_questions, done_docs = load_progress(args.output, sources_path)

    rows, total_candidates = select_chunks(
        args.input, args.since, args.num_docs, args.seed, done_docs
    )
    print(f"조건을 통과한 공지: {total_candidates}건 (이미 처리: {len(done_docs)}건)")
    print(f"이번에 처리할 공지: {len(rows)}건")

    if args.dry_run:
        for row in rows[:5]:
            meta = row["metadata"]
            print("-" * 60)
            print(meta["date"], meta["title"])
            print(body_of(row["chunk_text"])[:200].replace("\n", " "))
        return

    generate = load_generator()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    saved = 0
    for i, row in enumerate(rows, 1):
        meta = row["metadata"]
        try:
            pairs = generate(row["chunk_text"], num_pairs=args.pairs_per_doc)
        except Exception as e:  # API 오류가 나도 다음 문서로 계속
            print(f"[{i}/{len(rows)}] 생성 실패: {e}")
            time.sleep(args.sleep * 3)
            continue

        kept = []
        for pair in pairs or []:
            cleaned = clean_pair(pair)
            if cleaned is None:
                continue
            key = normalize(cleaned[0])
            if key in seen_questions:
                continue
            seen_questions.add(key)
            kept.append(cleaned)

        with open(args.output, "a", encoding="utf-8") as f:
            for question, answer in kept:
                line = {"instruction": question, "output": answer}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        with open(sources_path, "a", encoding="utf-8") as f:
            source = {
                "document_id": row["document_id"],
                "title": meta["title"],
                "url": meta["url"],
                "n_pairs": len(kept),
            }
            f.write(json.dumps(source, ensure_ascii=False) + "\n")

        saved += len(kept)
        print(f"[{i}/{len(rows)}] {meta['title'][:40]} -> {len(kept)}쌍 (누적 {saved})")
        time.sleep(args.sleep)

    print(f"완료: {saved}쌍을 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
