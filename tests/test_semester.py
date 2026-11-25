from datetime import date

import pytest

from app.chains.semester import Term, current_term, parse_question, parse_title

NOV = date(2026, 11, 25)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("[수업] 2025학년도 2학기 수강변경 안내(첨부파일 필독)", Term(2025, "2")),
        ("2023-2학기 산학협력프로젝트 경진대회 결과 안내", Term(2023, "2")),
        ("2021.2학기] 미입력 강의계획서 공지 안내", Term(2021, "2")),
        ("[TUTOR] 2025학년도 여름계절학기 학부생 TUTOR 모집", Term(2025, "여름")),
        ("2024학년도 겨울계절수업 수강신청 안내", Term(2024, "겨울")),
        ("2026 2학기", Term(2026, "2")),  # 강의계획서 year + semester
        ("2026 AI동행프로젝트 청년 멘토 모집 안내", Term(2026)),
        ("졸업논문 제출 안내", None),
    ],
)
def test_parse_title(title, expected):
    assert parse_title(title) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("수강신청 언제야?", None),
        ("종합설계프로젝트1 설명해줘", None),
        ("2025학년도 1학기 수강신청 언제야?", Term(2025, "1")),
        ("이번 학기 수강정정 기간은?", Term(2026, "2")),
        ("다음 학기 수강신청은?", Term(2027, "1")),
        ("지난학기 성적 정정 기간", Term(2026, "1")),
        ("1학기 수강신청 언제야?", Term(2026, "1")),
        ("겨울계절학기 신청 어떻게 해?", Term(2026, "겨울")),
        ("2025년 장학금 공지", Term(2025)),
    ],
)
def test_parse_question(question, expected):
    assert parse_question(question, NOV) == expected


def test_current_term_and_matching():
    assert current_term(date(2026, 2, 1)) == Term(2026, "1")
    assert current_term(NOV) == Term(2026, "2")
    assert Term(2026, "2").matches(Term(2026))
    assert not Term(2026, "2").matches(Term(2026, "1"))
    assert not Term(2025, "2").matches(Term(2026, "2"))


def test_title_variants():
    assert parse_title("2025년도 1학기 종합설계프로젝트 설명회 안내") == Term(2025, "1")
    assert parse_title("2025학년 1학기 강의개선 설문") == Term(2025, "1")
    assert parse_title("[TUTOR] 2025 동계 계절학기 튜터 추가모집") == Term(2025, "겨울")
