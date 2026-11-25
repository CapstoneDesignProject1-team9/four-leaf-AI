from langchain_core.documents import Document

from app.chains.lexical import Bm25Index
from app.chains.rag_chain import _limit_per_source, fuse_rankings


def test_bm25_prefers_title_match():
    index = Bm25Index(["2026학년도 2학기 수강신청 안내", "졸업논문 제출 안내", "장학금 신청"])
    assert index.search("수강신청 언제야", 3)[0] == 0
    assert index.search("xyz", 3) == []


def _doc(text: str, distance: float, url: str = "") -> Document:
    return Document(page_content=text, metadata={"distance": distance, "url": url or text})


def test_fuse_rankings_rewards_both_lists():
    a, b, c = _doc("a", 0.1), _doc("b", 0.2), _doc("c", 0.3)
    assert [d.page_content for d in fuse_rankings([a, b, c], [b])] == ["b", "a", "c"]


def test_limit_per_source_caps_chunks_of_one_notice():
    docs = [_doc(str(i), 0.1, "same") for i in range(4)] + [_doc("x", 0.2), _doc("y", 0.3)]
    assert [d.page_content for d in _limit_per_source(docs, 4)] == ["0", "1", "x", "y"]


def test_syllabi_sharing_one_url_are_separate_sources():
    def syl(code: str) -> Document:
        meta = {
            "distance": 0.1,
            "url": "https://knuin/lect",
            "source_type": "syllabus",
            "course_code": code,
        }
        return Document(page_content=code, metadata=meta)

    docs = [syl("A-001"), syl("A-001"), syl("B-001"), syl("C-001"), syl("A-001")]
    assert [d.page_content for d in _limit_per_source(docs, 4)] == [
        "A-001",
        "A-001",
        "B-001",
        "C-001",
    ]
