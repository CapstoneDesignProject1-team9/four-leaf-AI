import random
from datetime import date

from langchain_core.documents import Document

from app.chains.semester import Term
from train.build_rag_sft import REFUSAL, make_example, serving_question


def _doc(url: str) -> Document:
    return Document(page_content=url, metadata={"url": url})


RETRIEVED = [_doc("gold"), _doc("a"), _doc("b"), _doc("c"), _doc("d"), _doc("e"), _doc("f")]
GOLD = [_doc("gold"), _doc("gold")]


def test_oracle_keeps_answer_and_gold_docs():
    docs, out = make_example("답", RETRIEVED, GOLD, "gold", True, random.Random(0), 5)
    assert out == "답"
    assert len(docs) == 5
    assert sum(d.metadata["url"] == "gold" for d in docs) == 2


def test_without_oracle_removes_gold_and_refuses():
    docs, out = make_example("답", RETRIEVED, GOLD, "gold", False, random.Random(0), 5)
    assert out == REFUSAL
    assert [d.metadata["url"] for d in docs] == ["a", "b", "c", "d", "e"]


NOV = date(2026, 11, 25)


def test_serving_question_appends_gold_semester_like_clarification():
    q, target, explicit = serving_question("수강신청 언제야?", Term(2025, "1"), NOV)
    assert q == "수강신청 언제야? 2025학년도 1학기"
    assert (target, explicit) == (Term(2025, "1"), True)


def test_serving_question_keeps_explicit_and_drops_mismatch():
    assert serving_question("2025학년도 1학기 수강신청", Term(2025, "1"), NOV)[0].startswith("2025")
    assert serving_question("2024학년도 2학기 수강신청", Term(2025, "1"), NOV) is None


def test_serving_question_timeless_uses_current_term():
    assert serving_question("휴학 어떻게 해?", None, NOV) == (
        "휴학 어떻게 해?",
        Term(2026, "2"),
        False,
    )
