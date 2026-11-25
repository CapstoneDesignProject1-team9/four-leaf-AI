from langchain_core.documents import Document

from app.chains.rag_chain import filter_by_term
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
