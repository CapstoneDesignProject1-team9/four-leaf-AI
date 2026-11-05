"""QA 학습 데이터(JSONL)에서 쓰레기 줄과 중복 질문을 제거한다.

입력의 각 줄은 {"instruction": ..., "output": ...} 형식이다 (그 외 필드는 버린다).
제거 대상:
  - 크롤링 상태 메시지(예: scanned_completeness_unverified, 리뷰 요소 없음)를 다룬 줄
  - 같은 질문(공백 정규화 후 동일)의 두 번째 이후 줄
  - 너무 짧은 질문 또는 답변

사용법:
    python scripts/clean_qa_dataset.py --input data/in.jsonl --output data/out.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

JUNK_PATTERNS = {
    "crawl_status": re.compile(
        r"scanned_|completeness|unverified|리뷰 요소 없음|수집을 시도|로딩 지연|"
        r"화면 변경 여부"
    ),
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def junk_reason(instruction: str, output: str) -> str | None:
    combined = f"{instruction}\n{output}"
    for name, pattern in JUNK_PATTERNS.items():
        if pattern.search(combined):
            return name
    return None


def clean(input_path: Path, output_path: Path, min_instruction: int, min_output: int) -> Counter:
    stats: Counter = Counter()
    seen: set[str] = set()
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            stats["read"] += 1
            try:
                record = json.loads(line)
                instruction = str(record["instruction"]).strip()
                output = str(record["output"]).strip()
            except (json.JSONDecodeError, KeyError):
                stats["drop_malformed"] += 1
                continue
            if len(instruction) < min_instruction or len(output) < min_output:
                stats["drop_too_short"] += 1
                continue
            reason = junk_reason(instruction, output)
            if reason:
                stats[f"drop_{reason}"] += 1
                continue
            key = normalize(instruction)
            if key in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(key)
            row = {"instruction": instruction, "output": output}
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instruction", type=int, default=5, help="질문 최소 글자 수")
    parser.add_argument("--min-output", type=int, default=30, help="답변 최소 글자 수")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--input 과 --output 은 서로 다른 파일이어야 합니다")
    stats = clean(args.input, args.output, args.min_instruction, args.min_output)
    for name, count in sorted(stats.items()):
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
