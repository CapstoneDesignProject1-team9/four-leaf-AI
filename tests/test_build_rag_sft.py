import json
import random
from datetime import date

from langchain_core.documents import Document

from app.chains.semester import Term
from train.build_rag_sft import REFUSAL, load_generated, make_example, serving_question, with_source


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


def test_load_generated_maps_notice_syllabus_and_abstain(tmp_path):
    rows = [
        {
            "category": "수강·학사일정",
            "source": "공지사항",
            "post_id": 29454,
            "instruction": "q1",
            "output": "a1",
        },
        {
            "category": "교육과정·전공",
            "source": "강의계획서",
            "post_id": "",
            "course_code": "ITEC0401-001",
            "instruction": "q2",
            "output": "a2",
        },
        {
            "category": "자료없음",
            "source": "공지사항",
            "post_id": 29454,
            "instruction": "q3",
            "output": "a3",
        },
        {
            "category": "자료없음",
            "source": "범위 밖",
            "post_id": "",
            "instruction": "q4",
            "output": "a4",
        },
        {
            "category": "장학·지원",
            "source": "공지사항",
            "post_id": 1,
            "instruction": "q5",
            "output": "a5",
        },
    ]
    path = tmp_path / "gen.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    items, unmatched = load_generated(str(path), {"29454": "https://x/?wr_id=29454"})
    assert [(i.mode, i.collection, i.where) for i in items] == [
        ("qa", "notice", {"url": "https://x/?wr_id=29454"}),
        ("qa", "syllabi", {"course_code": "ITEC0401-001"}),
        ("abstain_doc", "notice", {"url": "https://x/?wr_id=29454"}),
        ("out_of_scope", None, None),
    ]
    assert unmatched == 1


def test_with_source_appends_once_and_strips_bold():
    gold = Document(page_content="", metadata={"title": "수강신청 안내", "url": "https://a/1"})
    text = with_source("**9월 1일**에 시작해요.", gold)
    assert text == "9월 1일에 시작해요.\n\n참고: 수강신청 안내 (https://a/1)"
    assert with_source(text, gold) == text
