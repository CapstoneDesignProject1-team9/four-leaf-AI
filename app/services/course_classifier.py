"""사용자의 자연어 질문으로부터 관련 과목명을 자동 추출/분류하는 서비스."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 기본 주요 과목 목록 (fallback 및 기본 등록)
DEFAULT_COURSES = [
    "컴퓨터구조",
    "운영체제",
    "자료구조",
    "알고리즘",
    "데이터베이스",
    "컴퓨터네트워크",
    "소프트웨어공학",
    "인공지능",
    "기계학습개론",
    "딥러닝개론",
    "시스템프로그래밍",
    "웹프로그래밍",
    "프로그래밍언어론",
    "오토마타및형식언어",
    "컴파일러",
    "정보보호개론",
    "클라우드컴퓨팅",
    "빅데이터컴퓨팅",
    "객체지향프로그래밍기초",
    "임베디드시스템",
    "컴퓨터그래픽스",
    "이산수학",
    "선형대수학",
    "종합설계",
]

# 학생들이 자주 사용하는 약어/별칭 -> 정규 과목명 매핑
COURSE_ALIASES: dict[str, str] = {
    "컴구": "컴퓨터구조",
    "컴구조": "컴퓨터구조",
    "computer architecture": "컴퓨터구조",
    "운체": "운영체제",
    "operating system": "운영체제",
    "자구": "자료구조",
    "data structure": "자료구조",
    "알고": "알고리즘",
    "algorithm": "알고리즘",
    "데베": "데이터베이스",
    "database": "데이터베이스",
    "컴넷": "컴퓨터네트워크",
    "네트워크": "컴퓨터네트워크",
    "소공": "소프트웨어공학",
    "sw공학": "소프트웨어공학",
    "웹프": "웹프로그래밍",
    "시프": "시스템프로그래밍",
    "프언": "프로그래밍언어론",
    "pl론": "프로그래밍언어론",
    "선대": "선형대수학",
    "오토마타": "오토마타및형식언어",
    "정보보호": "정보보호개론",
    "정보보안": "정보보호개론",
    "보안": "정보보호개론",
    "머신러닝": "기계학습개론",
    "기계학습": "기계학습개론",
    "딥러닝": "딥러닝개론",
    "객프": "객체지향프로그래밍기초",
    "객체지향": "객체지향프로그래밍기초",
    "산프": "산학프로젝트",
    "산학프로젝트": "산학프로젝트",
    "캡디": "종합설계",
    "캡스톤": "종합설계",
    "종설": "종합설계",
}

# 단어 경계가 필요한 영문 약어 매핑 (예: 'os', 'db', 'ai', 'ml', 'dl')
ENGLISH_WORD_ALIASES: dict[str, str] = {
    "os": "운영체제",
    "db": "데이터베이스",
    "ai": "인공지능",
    "ml": "기계학습개론",
    "dl": "딥러닝개론",
    "pl": "프로그래밍언어론",
    "se": "소프트웨어공학",
    "ca": "컴퓨터구조",
    "oop": "객체지향프로그래밍기초",
}

_KNOWN_COURSES: list[str] = []


def _load_known_courses() -> list[str]:
    """공식 교육과정 JSON 파일 및 기본 목록에서 과목명을 로드합니다."""
    global _KNOWN_COURSES
    if _KNOWN_COURSES:
        return _KNOWN_COURSES

    courses_set = set(DEFAULT_COURSES)

    # knu_computer_courses.json 경로 탐색
    possible_paths = [
        Path(__file__).resolve().parent.parent.parent
        / "knowledge"
        / "crawlers"
        / "everytime"
        / "knu_computer_courses.json",
        Path("./knowledge/crawlers/everytime/knu_computer_courses.json"),
    ]

    for p in possible_paths:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("courses", []):
                        name = item.get("course_name", "").strip()
                        if name:
                            courses_set.add(name)
                logger.info("공식 과목 목록 로드 완료: 총 %d개 과목", len(courses_set))
                break
            except Exception as e:
                logger.warning("과목 목록 파일 로드 실패 (%s): %s", p, e)

    # 긴 과목명 우선 매칭을 위해 글자 수 내림차순 정렬
    _KNOWN_COURSES = sorted(courses_set, key=len, reverse=True)
    return _KNOWN_COURSES


def classify_course(question: str) -> str:
    """
    사용자의 질문 문자열을 분석하여 가장 적합한 과목명을 반환합니다.
    특정 과목과 관련이 없거나 일반 학사/행정 질문인 경우 '일반'을 반환합니다.
    """
    if not question or not question.strip():
        return "일반"

    q_lower = question.strip().lower()
    q_compact = re.sub(r"\s+", "", q_lower)

    # 1. 영문 단어 경계 약어 매칭 (os, db, ai 등)
    # 주변에 다른 알파벳이 붙어있지 않은 독립 단어/토큰 형태인지 확인
    for abbr, canonical_name in ENGLISH_WORD_ALIASES.items():
        pattern = rf"(?:\b|[^a-zA-Z]){re.escape(abbr)}(?:\b|[^a-zA-Z])"
        if re.search(pattern, q_lower):
            return canonical_name

    # 2. 한글/복합 축약어 및 별칭 매칭 (긴 키워드 우선)
    sorted_aliases = sorted(COURSE_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        alias_compact = re.sub(r"\s+", "", alias.lower())
        if alias_compact in q_compact:
            return COURSE_ALIASES[alias]

    # 3. 공식 과목명 목록 직접 매칭 (긴 과목명 우선)
    known_courses = _load_known_courses()
    for course in known_courses:
        course_compact = re.sub(r"\s+", "", course.lower())
        if course_compact in q_compact:
            return course

    # 4. 어떤 과목에도 해당하지 않으면 "일반" (학사/행정/진로/생활 등)
    return "일반"

