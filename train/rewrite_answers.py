"""RAFT 학습 답변을 실제 챗봇 말투로 다시 쓴다 (Claude Message Batches, 50% 가격).

build_rag_sft.py가 만든 답변은 '~입니다/~요'가 섞이고 프롬프트 지침(사용한 자료 제목·URL 표시)과
달리 출처가 없다. 같은 사실을 유지한 채 말투·구성만 바꾼다. 거절 답변은 고정 문구라 그대로 둔다.

실행 (프로젝트 루트, ANTHROPIC_API_KEY 필요):
    python train/rewrite_answers.py sample --n 3      # 동기 호출로 몇 개 미리 보기
    python train/rewrite_answers.py submit            # 배치 제출 → data/rewrite_batch_id.txt
    python train/rewrite_answers.py collect           # 결과를 받아 rag_sft_*.jsonl 덮어쓰기
API 없이 직접 다시 쓸 때:
    python train/rewrite_answers.py export work.jsonl # 질문·원래 답변·출처만 뽑기
    python train/rewrite_answers.py apply out.jsonl   # {"id", "text"} 줄을 검사 후 반영
build_rag_sft.py를 다시 돌리면 답변이 원래대로 돌아가므로 이 스크립트도 다시 돌려야 한다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.build_rag_sft import REFUSAL  # noqa: E402

MODEL = "claude-opus-5-5"
SPLITS = ("train", "val")
BATCH_ID_FILE = Path("data/rewrite_batch_id.txt")

SYSTEM = """당신은 경북대학교 학생 질문에 답하는 학과 AI 튜터 챗봇의 답변을 다듬는 편집자입니다.
[원래 답변]의 사실은 그대로 두고 말투와 구성만 실제 챗봇처럼 자연스럽게 다시 쓰세요.

규칙:
- 첫 문장에서 질문에 바로 답하세요(결론 먼저).
- 날짜·장소·방법·대상처럼 항목이 여럿이면 '-' 목록으로 2~6줄 정리하세요. 짧은 답은 문단으로 충분합니다.
- 해요체로 통일하고, 친근하되 과장·감탄·이모지는 쓰지 마세요.
- 마크다운 굵은 글씨(**), 제목(#), 표는 쓰지 마세요. 채팅 화면이 일반 텍스트로 보여 줍니다.
- [원래 답변]과 [검색 자료]에 없는 사실(날짜, 숫자, 장소, 연락처, 절차)을 새로 만들지 마세요.
  원래 답변이 '공지에 없다'고 한 내용은 그대로 '없다'고 하세요.
- 마지막 줄에 '참고: 공지 제목 (URL)' 형식으로 답에 실제로 쓴 자료를 [검색 자료]에서 골라 적으세요.
  둘 이상이면 '참고:' 줄을 자료마다 하나씩 쓰세요. 제목과 URL은 [검색 자료]에 적힌 그대로 옮기세요.
- 길이는 원래 답변과 비슷하거나 더 짧게 하세요.
- 다시 쓴 답변만 출력하세요. 설명이나 머리말은 쓰지 마세요."""


def _path(split: str) -> Path:
    return Path(f"data/rag_sft_{split}.jsonl")


def load(split: str) -> list[dict]:
    with open(_path(split), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def targets() -> list[tuple[str, int, dict]]:
    """다시 쓸 (split, 줄 번호, 행). 거절 답변은 뺀다."""
    return [
        (split, i, row)
        for split in SPLITS
        for i, row in enumerate(load(split))
        if row["completion"][0]["content"] != REFUSAL
    ]


def params(row: dict) -> MessageCreateParamsNonStreaming:
    prompt = row["prompt"][0]["content"]
    answer = row["completion"][0]["content"]
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        # 말투 편집이라 깊은 추론이 필요 없다.
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n[원래 답변]\n{answer}\n\n위 [원래 답변]을 규칙에 맞게 다시 쓰세요.",
            }
        ],
    )


_NUMBER = re.compile(r"\d+")
_URL = re.compile(r"https?://[^\s)]+")


def problems(rewritten: str, row: dict) -> list[str]:
    """새로 만든 숫자·URL이 있거나 형식이 깨졌으면 이유를 돌려준다(빈 목록이면 통과)."""
    source = row["prompt"][0]["content"] + row["completion"][0]["content"]
    found = []
    new_numbers = set(_NUMBER.findall(rewritten)) - set(_NUMBER.findall(source))
    if new_numbers:
        found.append(f"새 숫자 {sorted(new_numbers)[:5]}")
    new_urls = set(_URL.findall(rewritten)) - set(_URL.findall(source))
    if new_urls:
        found.append(f"새 URL {sorted(new_urls)[:2]}")
    if "참고:" not in rewritten:
        found.append("출처 줄 없음")
    if "**" in rewritten or rewritten.lstrip().startswith("#"):
        found.append("마크다운")
    return found


def _text(message: anthropic.types.Message) -> str | None:
    if message.stop_reason != "end_turn":
        return None
    return "".join(b.text for b in message.content if b.type == "text").strip() or None


def sample(n: int) -> None:
    client = anthropic.Anthropic()
    for split, i, row in targets()[:n]:
        text = _text(client.messages.create(**params(row)))
        print(f"## {split}:{i} 질문: {row['prompt'][0]['content'].split('[질문]')[-1].strip()}")
        print(f"[원래]\n{row['completion'][0]['content']}\n\n[다시 씀]\n{text}")
        print(f"[검사] {problems(text, row) if text else '응답 없음'}\n")


def submit() -> None:
    items = targets()
    batch = anthropic.Anthropic().messages.batches.create(
        requests=[Request(custom_id=f"{s}-{i}", params=params(row)) for s, i, row in items]
    )
    BATCH_ID_FILE.write_text(batch.id)
    print(f"배치 제출: {batch.id} ({len(items)}개). 끝나면 collect를 실행하세요.")


def collect() -> None:
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(BATCH_ID_FILE.read_text().strip())
    if batch.processing_status != "ended":
        raise SystemExit(f"아직 처리 중: {batch.request_counts}")
    _apply(
        (r.custom_id, _text(r.result.message) if r.result.type == "succeeded" else None)
        for r in client.messages.batches.results(batch.id)
    )


def export(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for split, i, row in targets():
            item = {
                "id": f"{split}-{i}",
                "question": row["prompt"][0]["content"].split("[질문]\n")[-1],
                "answer": row["completion"][0]["content"],
                **row.get("source", {}),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def apply(paths: list[str]) -> None:
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            items += [json.loads(line) for line in f if line.strip()]
    _apply((item["id"], item["text"]) for item in items)


def _finish(text: str, row: dict) -> str:
    """굵은 글씨를 지우고, 출처 줄이 없으면 정답 공지 제목·URL로 붙인다."""
    text = text.replace("**", "").strip()
    source = row.get("source")
    if source and "참고:" not in text:
        text += f"\n\n참고: {source['title']} ({source['url']})"
    return text


def _apply(results) -> None:
    """다시 쓴 답(없으면 원래 답)을 다듬어 반영한다. 검사에 걸리면 원래 답을 다듬어 쓴다."""
    rewritten = {custom_id: text for custom_id, text in results if text}
    rows = {split: load(split) for split in SPLITS}
    counts = {"다시 씀": 0, "원래 답 유지(검사 실패)": 0, "원래 답 다듬기만": 0}
    for split, items in rows.items():
        for i, row in enumerate(items):
            original = row["completion"][0]["content"]
            if original == REFUSAL:
                continue
            custom_id = f"{split}-{i}"
            final = _finish(rewritten.get(custom_id, original), row)
            if custom_id not in rewritten:
                counts["원래 답 다듬기만"] += 1
            elif issues := problems(final, row):
                print(f"  {custom_id}: 원래 답 유지 ({', '.join(issues)})")
                final = _finish(original, row)
                counts["원래 답 유지(검사 실패)"] += 1
            else:
                counts["다시 씀"] += 1
            row["completion"][0]["content"] = final
    for split, items in rows.items():
        with open(_path(split), "w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sample", "submit", "collect", "export", "apply"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    commands = {
        "sample": lambda: sample(a.n),
        "submit": submit,
        "collect": collect,
        "export": lambda: export(a.files[0]),
        "apply": lambda: apply(a.files),
    }
    commands[a.command]()


if __name__ == "__main__":
    main()
