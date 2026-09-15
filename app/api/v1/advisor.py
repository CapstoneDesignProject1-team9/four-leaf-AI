import logging

from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.models.schemas import AdvisorReportRequest, AdvisorReportResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# JSON 파싱을 위한 내부 Pydantic 모델
class ReportFormat(BaseModel):
    summary: str = Field(description="학생 질문 전반적 요약 (3~4문장)")
    keywords: list[str] = Field(description="자주 언급된 핵심 키워드 3~5개")
    recommendations: list[str] = Field(description="교수자를 위한 구체적인 AI 제안 사항 2~3개")


@router.post("/report", response_model=AdvisorReportResponse)
async def generate_advisor_report(request: AdvisorReportRequest):
    """
    [교수자용 AI 어드바이저]
    학생들이 남긴 질문 목록을 분석하여 요약, 키워드, 추천 액션을 생성합니다.
    (HyperCLOVA X 활용)
    """
    if not request.student_questions:
        raise HTTPException(status_code=400, detail="학생 질문 목록이 비어있습니다.")

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        parser = JsonOutputParser(pydantic_object=ReportFormat)

        system_prompt = """당신은 대학교 교수자를 돕는 'AI 어드바이저'입니다.
학생들이 수업 '{course_name}'에 대해 남긴 질문들을 분석해서 다음 JSON 형식으로 리포트를 작성해주세요.
{format_instructions}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "학생들의 질문 목록:\n{questions}"),
            ]
        )

        chain = prompt | llm | parser

        questions_text = "\n".join([f"- {q}" for q in request.student_questions])

        result = chain.invoke(
            {
                "course_name": request.course_name,
                "questions": questions_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        return AdvisorReportResponse(
            summary=result.get("summary", ""),
            keywords=result.get("keywords", []),
            recommendations=result.get("recommendations", []),
        )

    except Exception as e:
        logger.exception("AI 어드바이저 리포트 생성 중 오류: %s", e)
        raise HTTPException(status_code=500, detail="리포트 생성 중 오류가 발생했습니다.")
