"""KNU 강의계획서 JSONL을 기존 청커 입력 형식의 문서 JSONL로 변환한다.

프로젝트 루트에서 실행:
    python knowledge/preprocessing/preprocess_syllabi.py

경로를 직접 지정하거나 기존 결과를 교체:
    python knowledge/preprocessing/preprocess_syllabi.py \
        --input knowledge/raw/syllabi/knu_cs_syllabi.jsonl \
        --output knowledge/processed/syllabi/knu_syllabus_documents.jsonl \
        --overwrite

표준 라이브러리만 사용한다. 원본은 수정하지 않으며, 강좌마다 안정적인
document_id와 임베더 호환 metadata(post_id, url)를 만든다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "knowledge/raw/syllabi/knu_cs_syllabi.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "knowledge/processed/syllabi/knu_syllabus_documents.jsonl"

COURSE_METADATA_FIELDS = (
    "year",
    "semester",
    "grade",
    "category",
    "college",
    "department",
    "course_code",
    "course_name",
    "credits",
    "lecture_hours",
    "practice_hours",
    "instructor",
    "class_time",
    "actual_time",
    "classroom",
    "room",
    "capacity",
    "enrolled",
    "department_filter",
)

SYLLABUS_SECTION_NAMES = {
    "general": "일반사항",
    "core_competencies": "핵심역량",
    "evaluation_methods": "평가방법",
    "disability_support": "장애학생 학습지원사항",
    "weekly_schedule": "주별강의",
    "course_evaluation": "강의평가 문항",
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def text_value(value: object) -> str:
    """Return a scalar string suitable for JSON and Chroma metadata."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    return normalize_text(value)


def normalize_semester(value: str) -> str:
    semester = normalize_text(value)
    english_terms = {
        "1st semester": "1학기",
        "first semester": "1학기",
        "2nd semester": "2학기",
        "second semester": "2학기",
        "summer semester": "계절학기(하계)",
        "winter semester": "계절학기(동계)",
    }
    return english_terms.get(semester.casefold(), semester)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def course_key(item: dict[str, Any]) -> tuple[str, str, str] | None:
    code = text_value(item.get("course_code")).upper()
    if not code:
        return None
    year = text_value(item.get("year")) or "unknown"
    semester = normalize_semester(text_value(item.get("semester"))) or "unknown"
    return year, semester, code


def syllabus_text(item: dict[str, Any]) -> str:
    """Return captured tab text, preserving the tab boundaries."""
    syllabus = item.get("syllabus")
    sections: list[str] = []
    if isinstance(syllabus, dict):
        for key, label in SYLLABUS_SECTION_NAMES.items():
            body = normalize_text(syllabus.get(key))
            if body:
                sections.append(f"[강의계획서: {label}]\n{body}")
    return "\n\n".join(sections) or normalize_text(item.get("content"))


def course_text(item: dict[str, Any]) -> str:
    """Compose searchable course facts followed by the captured syllabus tabs."""
    labels = {
        "year": "개설연도",
        "semester": "개설학기",
        "course_code": "강좌번호",
        "course_name": "교과목명",
        "college": "개설대학",
        "department": "학과",
        "department_filter": "조회학과",
        "grade": "수강학년",
        "category": "교과목구분",
        "credits": "학점",
        "lecture_hours": "강의시간(시수)",
        "practice_hours": "실습시간(시수)",
        "instructor": "담당교수",
        "class_time": "강의시간",
        "actual_time": "실제수업시간",
        "classroom": "강의실",
        "room": "호실",
        "capacity": "수강정원",
        "enrolled": "수강인원",
    }
    facts = []
    for field in COURSE_METADATA_FIELDS:
        value = text_value(item.get(field))
        if value:
            facts.append(f"{labels[field]}: {value}")

    # Older or partial crawler records may have only the combined content field.
    captured_text = syllabus_text(item)
    return "\n".join(facts) + (f"\n\n{captured_text}" if captured_text else "")


def load_documents(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    courses: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    duplicate_records = 0
    skipped: list[dict[str, Any]] = []
    input_records = 0

    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            input_records += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_number}번째 줄 JSON 오류: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{line_number}번째 줄은 JSON 객체가 아닙니다")

            key = course_key(item)
            if key is None:
                skipped.append({"line": line_number, "reason": "missing_course_code"})
                continue
            if not text_value(item.get("course_name")):
                skipped.append(
                    {"line": line_number, "course_code": key[2], "reason": "missing_course_name"}
                )
                continue
            if not syllabus_text(item):
                skipped.append(
                    {"line": line_number, "course_code": key[2], "reason": "empty_syllabus"}
                )
                continue

            if key in courses:
                duplicate_records += 1
                # Keep the more complete capture when an appended crawler run
                # produced a duplicate; ties use the later record.
                previous = courses[key][1]
                previous_size = len(course_text(previous))
                current_size = len(course_text(item))
                if current_size < previous_size:
                    continue
            courses[key] = (line_number, item)

    documents: list[dict[str, Any]] = []
    for (year, semester, code), (_, item) in courses.items():
        name = text_value(item.get("course_name"))
        title = f"{name} ({code})"
        course_id = f"knu_syllabus:{year}:{semester}:{code}"
        url = text_value(item.get("url"))
        text = course_text(item)
        metadata = {
            "post_id": course_id,
            "title": title,
            "date": "",
            "url": url,
            "source_type": "syllabus",
        }
        for field in COURSE_METADATA_FIELDS:
            value = text_value(item.get(field))
            if field == "semester" and value:
                value = normalize_semester(value)
            metadata[field] = value

        documents.append(
            {
                "document_id": digest(course_id),
                "text": text,
                "document_text": f"{title}\n\n{text}",
                "content_hash": digest(text),
                "metadata": metadata,
            }
        )

    report = {
        "input": str(path.resolve()),
        "input_records": input_records,
        "documents": len(documents),
        "duplicate_course_records": duplicate_records,
        "skipped_records": len(skipped),
        "departments": dict(
            Counter(
                document["metadata"]["department_filter"]
                or document["metadata"]["department"]
                or "(학과 미상)"
                for document in documents
            )
        ),
        "issues": skipped,
    }
    return documents, report


def write_output(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"출력 파일이 이미 있습니다: {path} (--overwrite로 교체)")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.output.with_suffix(".report.json")
    resolved = {args.input.resolve(), args.output.resolve(), report_path.resolve()}
    if len(resolved) != 3:
        raise SystemExit("입력·출력·보고서 경로는 서로 달라야 합니다")
    if not args.input.is_file():
        raise SystemExit(f"입력 파일을 찾을 수 없습니다: {args.input}")
    if not args.overwrite and (args.output.exists() or report_path.exists()):
        raise SystemExit("출력 또는 보고서가 이미 있습니다. --overwrite를 사용하세요.")

    try:
        documents, report = load_documents(args.input)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"전처리 실패: {exc}") from exc
    if not documents:
        raise SystemExit("유효한 강의계획서가 없습니다. 출력 파일을 만들지 않았습니다.")

    report["output"] = str(args.output.resolve())
    jsonl = "".join(json.dumps(document, ensure_ascii=False) + "\n" for document in documents)
    write_output(args.output, jsonl, args.overwrite)
    write_output(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
