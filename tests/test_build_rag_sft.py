import random

from langchain_core.documents import Document

from train.build_rag_sft import REFUSAL, make_example


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
