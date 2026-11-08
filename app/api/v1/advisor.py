"""교수자용 AI 어드바이저: 저장된 학생 질문 기록에 대한 자유 질문에 답한다.

1. 보관 기간 안의 최근 학생 질문에 번호를 붙여 Claude에 보낸다. Claude는 교수자가 원하는
   작업(요약·리포트·어려운 부분·퀴즈·시험문제)과 주제를 정하고, 주제와 관련된 질문 번호를
   유형별로 묶는다.
2. 반환된 번호를 실제 기록과 대조해 검증하고, 유형별 건수는 코드가 센다.
3. Claude가 작업별 지침에 따라 답한다. 분석 작업은 집계만 근거로 쓰고, 퀴즈·시험문제는
   학생 질문에서 드러난 개념을 바탕으로 새 문항을 만든다. 실패하면 집계를 그대로 문장으로 만든다.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from anthropic import AsyncAnthropic, transform_schema
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.schemas import AdvisorAskRequest, AdvisorAskResponse, QuestionGroup
from app.storage.chat_logs import StoredQuestion, fetch_recent_questions, purge_expired_chat_logs

logger = logging.getLogger(__name__)
router = APIRouter()

Task = Literal["summary", "report", "difficulty", "quiz", "exam"]
TASKS: tuple[str, ...] = ("summary", "report", "difficulty", "quiz", "exam")
GENERATION_TASKS = {"quiz", "exam"}
DEFAULT_ITEMS = {"quiz": 5, "exam": 3}
MAX_ITEMS = 20
MAX_EXAMPLES = 3  # 응답에 담을 대표 질문 수
MAX_PROMPT_EXAMPLES = 8  # 답변 생성 프롬프트에 넣을 유형별 질문 수
MAX_TOKENS = 16000
# 안전 분류기가 요청을 거절하면 같은 호출 안에서 대체 모델로 다시 실행한다.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class _RawGroup(BaseModel):
    label: str = Field(description="질문 유형 이름 (짧은 한국어)")
    question_ids: list[int] = Field(description="이 유형에 속한 학생 질문 번호")


class _Grouping(BaseModel):
    task: Task = Field(
        description=(
            "교수자가 원하는 작업. summary=어떤 질문이 많았는지, report=종합 리포트, "
            "difficulty=학생들이 어려워하는 부분, quiz=퀴즈 생성, exam=시험문제 생성"
        )
    )
    item_count: int | None = Field(
        default=None, description="퀴즈·시험문제 요청에서 교수자가 지정한 문항 수. 없으면 null"
    )
    topic: str | None = Field(
        description="교수자가 묻는 주제(과목명이나 개념). 특정 주제 없이 전체를 대상으로 하면 null"
    )
    groups: list[_RawGroup] = Field(description="관련 질문을 비슷한 내용끼리 묶은 결과")


_GROUP_SYSTEM = """당신은 대학 AI 튜터에 들어온 학생 질문 기록을 분석하는 도우미입니다.
교수자의 질문을 읽고 원하는 작업과 주제를 정한 뒤, 학생 질문 목록에서 그 주제와 관련된 질문만 골라 비슷한 내용끼리 묶으세요.

규칙:
- task: 어떤 질문이 많았는지 물으면 summary, 리포트를 요청하면 report, 어려워하는(헷갈려하는) 부분을 물으면 difficulty,
  퀴즈를 만들어 달라고 하면 quiz, 시험문제(중간·기말고사 문제 등)를 만들어 달라고 하면 exam.
- item_count: 퀴즈나 시험문제의 문항 수를 지정했으면 그 숫자, 아니면 null.
- topic: 교수자가 말한 주제(과목명이나 개념). 특정 주제 없이 학생 질문 전체를 대상으로 하면 null.
- topic이 있으면 그 주제와 직접 관련된 질문만 포함하고 나머지는 제외하세요. topic이 null이면 모든 질문을 포함하세요.
- 질문 내용이 비슷한 것끼리 2~8개 묶음으로 나누고, 한 질문은 한 묶음에만 넣으세요.
- label은 교수자가 바로 이해할 수 있는 짧은 유형 이름으로 쓰세요(예: '캐시 매핑 방식').
- question_ids에는 목록에 있는 [번호]만 쓰고, 없는 번호를 만들지 마세요."""

_COMMON_RULES = """- 건수는 집계 결과의 숫자를 그대로 쓰고, 새로 세거나 추정하지 마세요.
- 마크다운 표나 굵은 글씨(**) 없이 짧은 문단, 번호, '-' 목록만 쓰세요."""

_TASK_SYSTEMS: dict[str, str] = {
    "summary": f"""당신은 교수자를 돕는 AI 어드바이저입니다. 아래 집계 결과만 근거로 교수자의 질문에 한국어로 답하세요.
- 많은 유형부터 설명하고, 유형마다 대표 질문을 1~2개 인용하세요.
- 집계에 없는 내용은 지어내지 마세요.
{_COMMON_RULES}""",
    "report": f"""당신은 교수자를 돕는 AI 어드바이저입니다. 아래 집계 결과만 근거로 학생 질문 리포트를 한국어로 쓰세요.
- '요약', '주요 질문 유형', '학생들이 어려워하는 부분', '강의 제안' 순서로 작성하세요.
- 주요 질문 유형은 많은 순으로 건수와 대표 질문을 함께 쓰세요.
- 강의 제안은 질문에서 드러난 어려움을 해소하는 구체적인 수업 활동이나 보충 자료로 2~3개 쓰세요.
- 집계에 없는 사실은 지어내지 마세요.
{_COMMON_RULES}""",
    "difficulty": f"""당신은 교수자를 돕는 AI 어드바이저입니다. 아래 학생 질문을 근거로 학생들이 어려워하는 부분을 한국어로 설명하세요.
- 같은 개념을 반복해서 묻거나, '헷갈려', '모르겠어', '왜'처럼 이해가 막힌 표현이 있는 질문을 어려움의 근거로 보세요.
- 어려운 부분마다 (1) 무엇을 헷갈려하는지, (2) 근거가 된 학생 질문 1~2개와 해당 유형의 건수, (3) 수업에서 보충할 방법을 쓰세요.
- 어려운 순서(근거가 많은 순)로 최대 5개까지 쓰세요.
{_COMMON_RULES}""",
    "quiz": f"""당신은 교수자를 돕는 AI 어드바이저입니다. 학생 질문에서 드러난 헷갈리는 개념을 확인하는 퀴즈를 한국어로 만드세요.
- 문항 수는 정확히 {{item_count}}개, 모두 4지선다 객관식입니다.
- 질문이 많은 유형부터 문항을 배정하고, 각 문항이 학생들이 실제로 헷갈려한 지점을 확인하도록 만드세요.
- 형식(문항마다 반복):
  1. (근거 유형: 유형 이름) 문제
     ① 보기 ② 보기 ③ 보기 ④ 보기
     정답: 번호 / 해설: 왜 정답인지와 흔한 오답의 이유
- 과목의 표준적인 개념으로 문제를 만들고, 확실하지 않은 수치나 학교별 세부 규정은 쓰지 마세요.
- 마지막 줄에 '※ AI가 생성한 문항입니다. 수업에 쓰기 전에 정답과 해설을 검토해 주세요.'를 쓰세요.
{_COMMON_RULES}""",
    "exam": f"""당신은 교수자를 돕는 AI 어드바이저입니다. 학생 질문에서 드러난 핵심 개념을 평가하는 시험문제를 한국어로 만드세요.
- 문항 수는 정확히 {{item_count}}개이며, 서술형과 계산·풀이형을 섞고 개념 이해를 확인하도록 만드세요.
- 질문이 많은 유형부터 문항을 배정하세요.
- 형식(문항마다 반복):
  1. (근거 유형: 유형 이름, 배점: N점) 문제
     모범답안: ...
     채점 기준: 부분 점수 기준을 '-' 목록으로
- 과목의 표준적인 개념으로 문제를 만들고, 확실하지 않은 수치나 학교별 세부 규정은 쓰지 마세요.
- 마지막 줄에 '※ AI가 생성한 문항입니다. 시험에 쓰기 전에 모범답안과 채점 기준을 검토해 주세요.'를 쓰세요.
{_COMMON_RULES}""",
}

TASK_NAMES = {
    "summary": "질문 빈도 분석",
    "report": "학생 질문 리포트",
    "difficulty": "어려운 부분 분석",
    "quiz": "퀴즈",
    "exam": "시험문제",
}


@dataclass
class Analysis:
    task: str
    item_count: int | None
    topic: str | None
    groups: list[QuestionGroup]
    prompt_examples: list[list[str]]  # groups와 같은 순서, 유형별 프롬프트용 질문
    matched: int


@lru_cache(maxsize=1)
def _claude() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY or None)


_GROUPING_FORMAT = {
    "type": "json_schema",
    "schema": transform_schema(_Grouping.model_json_schema()),
}


async def group_with_claude(question: str, questions: list[StoredQuestion]) -> dict:
    listing = "\n".join(f"[{q.id}] {' '.join(q.question.split())}" for q in questions)
    response = await _claude().beta.messages.create(
        model=settings.ADVISOR_MODEL,
        max_tokens=MAX_TOKENS,
        system=_GROUP_SYSTEM,
        messages=[
            {"role": "user", "content": f"교수자 질문: {question}\n\n학생 질문 목록:\n{listing}"}
        ],
        output_config={"format": _GROUPING_FORMAT},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"Claude가 질문 분류를 완료하지 못했습니다: {response.stop_reason}")
    text = next(block.text for block in response.content if block.type == "text")
    return _Grouping.model_validate_json(text).model_dump()


def _clean_task(value: object) -> str:
    return value if isinstance(value, str) and value in TASKS else "summary"


def _clean_items(task: str, value: object) -> int | None:
    if task not in GENERATION_TASKS:
        return None
    try:
        count = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_ITEMS[task]
    return min(max(count, 1), MAX_ITEMS)


def build_analysis(raw: dict, questions: list[StoredQuestion]) -> Analysis:
    """Claude 결과에서 실제로 있는 번호만 남기고(중복 제거) 유형별 건수와 대표 질문을 만든다."""
    by_id = {q.id: q for q in questions}
    task = _clean_task(raw.get("task"))
    topic = raw.get("topic") or None
    topic = topic.strip() or None if isinstance(topic, str) else None
    seen: set[int] = set()
    built: list[tuple[QuestionGroup, list[str]]] = []
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
        texts = [by_id[i].question for i in ids]
        group = QuestionGroup(
            label=str(g.get("label") or "기타").strip() or "기타",
            count=len(ids),
            examples=texts[:MAX_EXAMPLES],
        )
        built.append((group, texts[:MAX_PROMPT_EXAMPLES]))
    built.sort(key=lambda pair: pair[0].count, reverse=True)
    return Analysis(
        task=task,
        item_count=_clean_items(task, raw.get("item_count")),
        topic=topic,
        groups=[g for g, _ in built],
        prompt_examples=[e for _, e in built],
        matched=len(seen),
    )


def _period_text(days: int | None) -> str:
    return (
        f"최근 {days}일"
        if days
        else f"최근 {settings.CHAT_LOG_RETENTION_MONTHS}개월(보관 기간 전체)"
    )


def _groups_text(groups: list[QuestionGroup], examples: list[list[str]] | None = None) -> str:
    lines = []
    for i, g in enumerate(groups, 1):
        lines.append(f"{i}. {g.label} ({g.count}건)")
        shown = examples[i - 1] if examples else g.examples
        lines.extend(f"   - {example}" for example in shown)
    return "\n".join(lines)


async def write_answer_with_claude(
    question: str, analysis: Analysis, total: int, days: int | None
) -> str:
    system = _TASK_SYSTEMS[analysis.task]
    if analysis.item_count is not None:
        system = system.replace("{item_count}", str(analysis.item_count))
    user = (
        f"교수자 질문: {question}\n분석 기간: {_period_text(days)}\n"
        f"기간 내 전체 학생 질문: {total}건\n주제: {analysis.topic or '전체'}\n"
        f"관련 질문: {analysis.matched}건\n\n질문 유형별 집계와 학생 질문:\n"
        + _groups_text(analysis.groups, analysis.prompt_examples)
    )
    response = await _claude().beta.messages.create(
        model=settings.ADVISOR_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude가 답변 생성을 거절했습니다.")
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise RuntimeError(f"Claude 답변이 비어 있습니다: {response.stop_reason}")
    return text


def fallback_answer(analysis: Analysis, days: int | None) -> str:
    subject = f"'{analysis.topic}' 관련" if analysis.topic else "전체"
    head = ""
    if analysis.task in GENERATION_TASKS:
        head = f"{TASK_NAMES[analysis.task]}를 생성하지 못해 집계 결과만 보여드립니다.\n"
    return (
        head
        + f"{_period_text(days)} 동안 {subject} 학생 질문은 {analysis.matched}건이었습니다.\n"
        + _groups_text(analysis.groups)
    )


@router.post("/ask", response_model=AdvisorAskResponse)
async def ask_advisor(request: AdvisorAskRequest):
    """교수자의 자유 질문에 답한다.

    예: '컴퓨터구조 관련 학생들 질문은 뭐가 많았어?', '학생들 리포트 써줘',
    '학생들이 어려워하는 부분 알려줘', '운영체제 퀴즈 5문제 만들어줘', '시험문제 만들어줘'
    """
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
        raw = await group_with_claude(request.question, questions)
    except Exception:
        logger.exception("Claude 질문 분류 실패")
        raise HTTPException(
            status_code=503, detail="학생 질문을 분석하지 못했습니다. 잠시 후 다시 시도해주세요."
        )

    analysis = build_analysis(raw if isinstance(raw, dict) else {}, questions)
    if analysis.matched == 0:
        subject = f"'{analysis.topic}' 관련 " if analysis.topic else ""
        return AdvisorAskResponse(
            answer=(
                f"{_period_text(request.days)} 동안 {subject}학생 질문을 찾지 못했습니다. "
                f"(전체 {total}건 중)"
            ),
            task=analysis.task,
            topic=analysis.topic,
            matched_count=0,
            total_count=total,
            period_days=request.days,
        )

    try:
        answer = await write_answer_with_claude(request.question, analysis, total, request.days)
    except Exception:
        logger.exception("Claude 답변 생성 실패 — 집계 결과로 대신 답합니다.")
        answer = fallback_answer(analysis, request.days)

    return AdvisorAskResponse(
        answer=answer.strip(),
        task=analysis.task,
        item_count=analysis.item_count,
        topic=analysis.topic,
        matched_count=analysis.matched,
        total_count=total,
        period_days=request.days,
        groups=analysis.groups,
    )
