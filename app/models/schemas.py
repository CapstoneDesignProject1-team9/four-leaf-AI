from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """채팅 요청 스키마"""

    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="대화 세션 ID (멀티턴 대화용)")
    course_name: str | None = Field(None, description="관련 강의명 (선택)", max_length=200)
    model_name: str | None = Field(
        None, description="vLLM 모델 또는 등록된 LoRA 어댑터 이름", max_length=200
    )


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
    course_name: str | None = Field(default=None, description="자동 분류 또는 지정된 과목명")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class AdvisorAskRequest(BaseModel):
    """교수자가 학생 질문 기록에 대해 묻는 요청"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="교수자 질문 (예: 컴퓨터구조 관련 학생들 질문은 뭐가 많았어?)",
    )
    days: int | None = Field(
        default=None, ge=1, le=124, description="최근 며칠 동안의 질문만 볼지. 비우면 보관 기간 전체"
    )


class QuestionGroup(BaseModel):
    """비슷한 학생 질문 묶음"""

    label: str = Field(..., description="질문 유형 이름")
    count: int = Field(..., description="이 유형에 속한 질문 수")
    examples: list[str] = Field(default=[], description="대표 질문(최신순 최대 3개)")


class AdvisorAskResponse(BaseModel):
    """교수자 질문에 대한 답변과 근거 집계"""

    answer: str = Field(..., description="교수자에게 보여줄 답변")
    topic: str | None = Field(default=None, description="질문에서 파악한 주제. 전체 대상이면 null")
    matched_count: int = Field(..., description="주제와 관련된 학생 질문 수")
    total_count: int = Field(..., description="분석 대상 기간의 전체 학생 질문 수")
    period_days: int | None = Field(default=None, description="분석 기간(일). null이면 보관 기간 전체")
    groups: list[QuestionGroup] = Field(default=[], description="질문 유형별 집계(많은 순)")
