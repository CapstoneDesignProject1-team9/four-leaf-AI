"""교수자용 AI 어드바이저: 저장된 학생 질문 기록에 대한 자유 질문에 답한다.

1. 보관 기간 안의 최근 학생 질문에 번호를 붙여 Gemini에 보내고, 교수자가 묻는 주제와
   관련된 질문 번호를 유형별로 묶게 한다.
2. 반환된 번호를 실제 기록과 대조해 검증하고, 유형별 건수는 코드가 센다.
3. Gemini가 그 집계만 근거로 교수자에게 답한다. 실패하면 집계를 그대로 문장으로 만든다.
"""

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.schemas import AdvisorAskRequest, AdvisorAskResponse, QuestionGroup
from app.storage.chat_logs import StoredQuestion, fetch_recent_questions, purge_expired_chat_logs

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_EXAMPLES = 3


class _RawGroup(BaseModel):
    label: str = Field(description="질문 유형 이름 (짧은 한국어)")
    question_ids: list[int] = Field(description="이 유형에 속한 학생 질문 번호")


class _Grouping(BaseModel):
    topic: str | None = Field(description="교수자가 묻는 주제. 전체 질문을 묻거나 리포트 요청이면 null")
    groups: list[_RawGroup] = Field(description="관련 질문을 비슷한 내용끼리 묶은 결과")


_GROUP_SYSTEM = """당신은 대학 AI 튜터에 들어온 학생 질문 기록을 분석하는 도우미입니다.
교수자의 질문을 읽고, 학생 질문 목록에서 교수자가 묻는 주제와 관련된 질문만 골라 비슷한 내용끼리 묶으세요.

규칙:
- topic: 교수자가 묻는 주제(과목명이나 개념). 특정 주제 없이 학생 질문 전체를 묻거나 리포트를 요청하면 null.
- topic이 있으면 그 주제와 직접 관련된 질문만 포함하고 나머지는 제외하세요. topic이 null이면 모든 질문을 포함하세요.
- 질문 내용이 비슷한 것끼리 2~8개 묶음으로 나누고, 한 질문은 한 묶음에만 넣으세요.
- label은 교수자가 바로 이해할 수 있는 짧은 유형 이름으로 쓰세요(예: '캐시 매핑 방식').
- question_ids에는 목록에 있는 [번호]만 쓰고, 없는 번호를 만들지 마세요.

{format_instructions}"""

_ANSWER_SYSTEM = """당신은 교수자를 돕는 AI 어드바이저입니다. 아래 집계 결과만 근거로 교수자의 질문에 한국어로 답하세요.
- 건수는 집계 결과의 숫자를 그대로 쓰고, 새로 세거나 추정하지 마세요.
- 많은 유형부터 설명하고, 유형마다 대표 질문을 1~2개 인용하세요.
- 교수자가 리포트를 요청하면 '요약', '주요 질문 유형', '강의 제안' 순서로 작성하세요.
- 집계에 없는 내용은 지어내지 마세요. 마크다운 표는 쓰지 말고 짧은 문단과 '-' 목록만 쓰세요."""


def _gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY or None,
        temperature=0,
    )


async def group_with_gemini(question: str, questions: list[StoredQuestion]) -> dict:
    parser = JsonOutputParser(pydantic_object=_Grouping)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _GROUP_SYSTEM), ("human", "교수자 질문: {question}\n\n학생 질문 목록:\n{questions}")]
    )
    listing = "\n".join(f"[{q.id}] {' '.join(q.question.split())}" for q in questions)
    return await (prompt | _gemini() | parser).ainvoke(
        {
            "question": question,
            "questions": listing,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def build_groups(
    raw: dict, questions: list[StoredQuestion]
) -> tuple[str | None, list[QuestionGroup], int]:
    """Gemini 결과에서 실제로 있는 번호만 남기고(중복 제거) 유형별 건수와 대표 질문을 만든다."""
    by_id = {q.id: q for q in questions}
    topic = raw.get("topic") or None
    if isinstance(topic, str):
        topic = topic.strip() or None
    seen: set[int] = set()
    groups: list[QuestionGroup] = []
    for g in raw.get("groups") or []:
        ids: list[int] = []
        for value in g.get("question_ids") or []:
            try:
                qid = int(value)
            except (TypeError, ValueError):
                continue
            if qid in by_id and qid not in seen:
                seen.add(qid)
                ids.append(qid)
        if not ids:
            continue
        ids.sort(key=lambda i: (by_id[i].created_at, i), reverse=True)
        groups.append(
            QuestionGroup(
                label=str(g.get("label") or "기타").strip() or "기타",
                count=len(ids),
                examples=[by_id[i].question for i in ids[:MAX_EXAMPLES]],
            )
        )
    groups.sort(key=lambda g: g.count, reverse=True)
    return topic, groups, len(seen)


def _period_text(days: int | None) -> str:
    return f"최근 {days}일" if days else f"최근 {settings.CHAT_LOG_RETENTION_MONTHS}개월(보관 기간 전체)"


def _groups_text(groups: list[QuestionGroup]) -> str:
    lines = []
    for i, g in enumerate(groups, 1):
        lines.append(f"{i}. {g.label} ({g.count}건)")
        lines.extend(f"   - {example}" for example in g.examples)
    return "\n".join(lines)


async def write_answer_with_gemini(
    question: str,
    topic: str | None,
    groups: list[QuestionGroup],
    matched: int,
    total: int,
    days: int | None,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _ANSWER_SYSTEM),
            (
                "human",
                "교수자 질문: {question}\n분석 기간: {period}\n기간 내 전체 학생 질문: {total}건\n"
                "주제: {topic}\n관련 질문: {matched}건\n\n질문 유형별 집계:\n{groups}",
            ),
        ]
    )
    return await (prompt | _gemini() | StrOutputParser()).ainvoke(
        {
            "question": question,
            "period": _period_text(days),
            "total": total,
            "topic": topic or "전체",
            "matched": matched,
            "groups": _groups_text(groups),
        }
    )


def fallback_answer(topic: str | None, groups: list[QuestionGroup], matched: int, days: int | None) -> str:
    subject = f"'{topic}' 관련" if topic else "전체"
    return f"{_period_text(days)} 동안 {subject} 학생 질문은 {matched}건이었습니다.\n" + _groups_text(groups)


@router.post("/ask", response_model=AdvisorAskResponse)
async def ask_advisor(request: AdvisorAskRequest):
    """교수자의 자유 질문(예: '컴퓨터구조 관련 학생들 질문은 뭐가 많았어?', '학생들 리포트 써줘')에 답한다."""
    try:
        purge_expired_chat_logs()
        questions = fetch_recent_questions(days=request.days, limit=settings.ADVISOR_MAX_QUESTIONS)
    except Exception:
        logger.exception("학생 질문 기록 조회 실패")
        raise HTTPException(status_code=503, detail="학생 질문 기록을 불러오지 못했습니다.")

    total = len(questions)
    if total == 0:
        return AdvisorAskResponse(
            answer=f"{_period_text(request.days)} 동안 저장된 학생 질문이 없습니다.",
            matched_count=0,
            total_count=0,
            period_days=request.days,
        )

    try:
        raw = await group_with_gemini(request.question, questions)
    except Exception:
        logger.exception("Gemini 질문 분류 실패")
        raise HTTPException(status_code=503, detail="학생 질문을 분석하지 못했습니다. 잠시 후 다시 시도해주세요.")

    topic, groups, matched = build_groups(raw if isinstance(raw, dict) else {}, questions)
    if matched == 0:
        subject = f"'{topic}' 관련 " if topic else ""
        return AdvisorAskResponse(
            answer=f"{_period_text(request.days)} 동안 {subject}학생 질문을 찾지 못했습니다. (전체 {total}건 중)",
            topic=topic,
            matched_count=0,
            total_count=total,
            period_days=request.days,
        )

    try:
        answer = await write_answer_with_gemini(request.question, topic, groups, matched, total, request.days)
    except Exception:
        logger.exception("Gemini 답변 생성 실패 — 집계 결과로 대신 답합니다.")
        answer = fallback_answer(topic, groups, matched, request.days)

    return AdvisorAskResponse(
        answer=answer.strip(),
        topic=topic,
        matched_count=matched,
        total_count=total,
        period_days=request.days,
        groups=groups,
    )
