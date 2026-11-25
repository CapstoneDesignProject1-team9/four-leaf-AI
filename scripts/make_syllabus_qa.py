"""강의계획서 메타데이터(교수·시간·강의실·학점·학년·구분)로 학습용 QA를 만든다. API 없이 결정적.

답이 구조화된 필드에서 바로 나오므로 틀릴 일이 없다. 결과는 generate_qa_student.py와 같은 형식이라
build_rag_sft.py --generated 에 그대로 넣는다.

실행: python scripts/make_syllabus_qa.py  → data/syllabus_qa.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json

TIME_PLACE = [
    "{name}({code}) 수업은 언제, 어디서 해요?",
    "{name} {code} 강의실이랑 수업 시간 알려줘",
    "{name}({code}) 시간표가 어떻게 돼요?",
]
INSTRUCTOR = [
    "{name}({code}) 담당 교수님이 누구예요?",
    "{name} {code} 분반은 어느 교수님 수업이야?",
    "{name}({code}) 누가 가르쳐요?",
]
CREDITS = [
    "{name}({code})은 몇 학점이고 몇 학년 과목이에요?",
    "{name} {code} 학점이랑 교과구분 알려줘",
    "{name}({code}) 수업은 몇 학년 대상이고 학점은 어떻게 돼요?",
]


def _pick(options: list[str], key: str) -> str:
    return options[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(options)]


def _room(row: dict) -> str:
    room = str(row.get("room") or "").strip()
    return f"{row['classroom']} {room}호" if room else row["classroom"]


def make_pairs(row: dict) -> list[tuple[str, str]]:
    name, code = row["course_name"], row["course_code"]
    f = {"name": name, "code": code}
    subject = f"{name}({code}) 수업"
    pairs = []
    if row.get("class_time") and row.get("classroom"):
        pairs.append(
            (
                _pick(TIME_PLACE, code + "t").format(**f),
                f"{subject}은 {row['class_time']} 시간에 {_room(row)}에서 진행돼요. "
                "시간은 강의계획서의 요일·교시 표기 그대로예요.",
            )
        )
    if row.get("instructor"):
        pairs.append(
            (
                _pick(INSTRUCTOR, code + "i").format(**f),
                f"{subject}은 {row['instructor']} 교수님이 담당해요.",
            )
        )
    if row.get("credits"):
        hours = (
            f"강의 {row.get('lecture_hours') or 0}시간, 실습 {row.get('practice_hours') or 0}시간"
        )
        target = " ".join(
            part
            for part in (
                f"{row['grade']}학년 대상" if row.get("grade") else "",
                f"{row['category']} 과목" if row.get("category") else "",
            )
            if part
        )
        answer = f"{subject}은 {row['credits']}학점({hours})"
        answer += f"이고 {target}이에요." if target else "이에요."
        pairs.append((_pick(CREDITS, code + "c").format(**f), answer))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="knowledge/raw/syllabi/knu_cs_syllabi.jsonl")
    ap.add_argument("--output", default="data/syllabus_qa.jsonl")
    a = ap.parse_args()
    n = 0
    with open(a.input, encoding="utf-8") as src, open(a.output, "w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            for question, answer in make_pairs(row):
                item = {
                    "category": "수업·학생 운영",
                    "source": "강의계획서",
                    "course_code": row["course_code"],
                    "instruction": question,
                    "output": answer,
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
    print(f"{a.output}: {n}개")


if __name__ == "__main__":
    main()
