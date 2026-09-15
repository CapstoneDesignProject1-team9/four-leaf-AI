from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """채팅 요청 스키마"""

    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="대화 세션 ID (멀티턴 대화용)")


class SourceDocument(BaseModel):
    """RAG 검색 결과 문서"""

    content: str
    source: str
    category: str | None = None


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""

    answer: str = Field(..., description="AI 튜터 답변")
    sources: list[SourceDocument] = Field(default=[], description="참고한 문서 목록")
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class AdvisorReportRequest(BaseModel):
    """교수자용 통계/요약 요청 스키마"""

    course_name: str = Field(..., description="강의명")
    student_questions: list[str] = Field(..., description="수집된 학생들의 질문 목록")


class AdvisorReportResponse(BaseModel):
    """교수자용 통계/요약 응답 스키마"""

    summary: str = Field(..., description="학생 질문 전반적 요약")
    keywords: list[str] = Field(default=[], description="자주 언급된 핵심 키워드")
    recommendations: list[str] = Field(default=[], description="교수자를 위한 AI 제안 사항")
