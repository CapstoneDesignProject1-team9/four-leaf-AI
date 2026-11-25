import pytest
from langchain_core.documents import Document

from app.chains.rag_chain import clarify, filter_by_term, pick_section
from app.chains.semester import Term

NOW = Term(2026, "2")


def _notice(title: str) -> Document:
    return Document(page_content=title, metadata={"title": title, "distance": 0.0})


def test_drops_other_semesters_and_notes_target():
    docs = [
        _notice("2025학년도 2학기 지도교수 변경 안내"),
        _notice("지도교수 배정 및 변경 관련"),
        _notice("2026학년도 2학기 지도교수 변경 안내"),
    ]
    r = filter_by_term(docs, NOW, explicit=False)
    assert [d.metadata["title"] for d in r.documents] == [
        "지도교수 배정 및 변경 관련",
        "2026학년도 2학기 지도교수 변경 안내",
    ]
    assert r.other_terms == [Term(2025, "2")]
    assert r.top_excluded and r.has_target
    assert r.note and r.note.startswith("2026학년도 2학기 기준")


def test_no_note_when_nothing_displaced_or_explicit():
    docs = [_notice("인컴 졸업기준"), _notice("2024학년도 교육과정 안내")]
    assert filter_by_term(docs, NOW, explicit=False).note is None
    explicit = filter_by_term([_notice("2025학년도 2학기 안내")], Term(2025, "2"), True)
    assert explicit.note is None and explicit.documents


def test_year_only_docs_filtered_only_when_year_explicit():
    docs = [_notice("2024학년도 교육과정 안내")]
    assert filter_by_term(docs, NOW, explicit=False).documents
    assert not filter_by_term(docs, Term(2025), explicit=True).documents


def test_unrelated_target_semester_docs_do_not_count():
    close = Document(
        page_content="", metadata={"title": "2025학년도 1학기 수강신청", "distance": 0.2}
    )
    far = Document(
        page_content="",
        metadata={"source_type": "syllabus", "year": "2026", "semester": "2학기", "distance": 0.42},
    )
    r = filter_by_term([close, far], NOW, explicit=False)
    assert r.top_excluded and not r.has_target


def _syllabus(code: str, instructor: str, distance: float) -> Document:
    return Document(
        page_content="",
        metadata={
            "source_type": "syllabus",
            "title": f"종합설계프로젝트1 ({code})",
            "course_code": code,
            "instructor": instructor,
            "year": "2026",
            "semester": "2학기",
            "distance": distance,
        },
    )


SECTIONS = [_syllabus("ITEC0401-002", "이교수", 0.1), _syllabus("ITEC0401-001", "김교수", 0.2)]


def test_pick_section_asks_when_unspecified():
    docs, options = pick_section(SECTIONS, "종합설계프로젝트1 설명해줘")
    assert docs == SECTIONS
    assert options == ["ITEC0401-001 (김교수 교수)", "ITEC0401-002 (이교수 교수)"]


@pytest.mark.parametrize(
    "question", ["종설 1분반 알려줘", "김교수 종설", "ITEC0401-001 (김교수 교수)"]
)
def test_pick_section_keeps_chosen(question):
    docs, options = pick_section(SECTIONS, question)
    assert options == []
    assert [d.metadata["course_code"] for d in docs] == ["ITEC0401-001"]


def test_clarify_semester_when_target_missing():
    close = Document(
        page_content="", metadata={"title": "2025학년도 1학기 수강신청", "distance": 0.2}
    )
    older = Document(
        page_content="", metadata={"title": "2021학년도 2학기 수강신청", "distance": 0.3}
    )
    text, options = clarify(filter_by_term([close, older], NOW, explicit=False))
    assert "2026학년도 2학기 자료는 찾지 못했어요" in text
    assert options == ["2025학년도 1학기", "2021학년도 2학기"]
