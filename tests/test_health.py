"""실제 DB·벡터 초기화·AI 호출 없이 서버 API를 검증한다."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with (
        patch("app.main.initialize_question_store") as mock_init_db,
        patch(
            "app.chains.rag_chain.init_vectorstore",
            new_callable=AsyncMock,
        ) as mock_init_vector,
        patch("app.api.v1.chat.save_student_question") as mock_save,
        patch("app.api.v1.chat.build_rag_chain") as mock_builder,
    ):
        from app.main import app

        mock_builder.return_value.invoke.return_value = {
            "result": "테스트 답변",
            "source_documents": [],
        }

        with TestClient(app) as test_client:
            yield test_client, mock_save, mock_builder

        mock_init_db.assert_called_once_with()
        mock_init_vector.assert_awaited_once_with()


def test_health_endpoint(client):
    test_client, mock_save, mock_builder = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "four-leaf-ai"
    mock_save.assert_not_called()
    mock_builder.assert_not_called()


def test_chat_endpoint_with_mock(client):
    test_client, mock_save, mock_builder = client
    question = "취업 준비는 어떻게 해야 하나요?"
    answer = "학교 취업지원센터를 활용하세요."

    mock_builder.return_value.invoke.return_value = {
        "result": answer,
        "source_documents": [
            MagicMock(
                page_content="취업지원센터에서 이력서 첨삭을 제공합니다.",
                metadata={
                    "url": "https://example.com/notice/1",
                    "source_type": "notice",
                },
            )
        ],
    }

    response = test_client.post(
        "/api/v1/tutor/chat",
        json={"message": question},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == answer
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "https://example.com/notice/1"

    mock_save.assert_called_once()
    assert mock_save.call_args.kwargs["message"] == question
    mock_builder.return_value.invoke.assert_called_once_with(
        {"query": question}
    )


def test_chat_generation_failure(client):
    test_client, mock_save, mock_builder = client
    mock_builder.return_value.invoke.side_effect = RuntimeError(
        "internal-test-error"
    )

    response = test_client.post(
        "/api/v1/tutor/chat",
        json={"message": "테스트 질문"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "AI 튜터 응답 생성 중 오류가 발생했습니다."
    assert "internal-test-error" not in response.text
    mock_save.assert_called_once()


def test_chat_auto_classify_course(client):
    test_client, mock_save, mock_builder = client
    mock_builder.return_value.invoke.return_value = {
        "result": "파이프라인 해저드 설명입니다.",
        "source_documents": [],
    }

    # 사용자가 course_name을 지정하지 않고 질문
    response = test_client.post(
        "/api/v1/tutor/chat",
        json={"message": "컴구 파이프라인 해저드에 대해 알려줘"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["course_name"] == "컴퓨터구조"
    assert mock_save.call_args.kwargs["course_name"] == "컴퓨터구조"


def test_chat_auto_classify_general(client):
    test_client, mock_save, mock_builder = client
    mock_builder.return_value.invoke.return_value = {
        "result": "국가장학금 신청 기간 안내입니다.",
        "source_documents": [],
    }

    # 일반 학사 행정 질문
    response = test_client.post(
        "/api/v1/tutor/chat",
        json={"message": "국가장학금 신청 기간이 언제야?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["course_name"] == "일반"
    assert mock_save.call_args.kwargs["course_name"] == "일반"


def test_advisor_report_no_questions(client):
    test_client, _, _ = client
    with patch("app.api.v1.advisor.get_student_questions", return_value=[]):
        response = test_client.post(
            "/api/v1/advisor/report",
            json={"course_name": "존재하지않는과목"},
        )
        assert response.status_code == 404
        assert "저장된 학생 질문이 없습니다" in response.json()["detail"]


def test_advisor_report_success(client):
    test_client, _, _ = client
    mock_report = {
        "summary": "학생 질문 요약입니다.",
        "keywords": ["파이프라인", "캐시"],
        "recommendations": ["실습 보강"],
    }

    with (
        patch("app.api.v1.advisor.get_student_questions", return_value=["파이프라인 질문1", "캐시 질문2"]),
        patch("app.api.v1.advisor.ChatGoogleGenerativeAI"),
        patch("langchain_core.prompts.ChatPromptTemplate.__or__") as mock_chain_pipe,
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_report
        mock_chain_pipe.return_value.__or__.return_value = mock_chain

        response = test_client.post(
            "/api/v1/advisor/report",
            json={"course_name": "컴퓨터구조"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "학생 질문 요약입니다."
        assert "파이프라인" in data["keywords"]
        assert len(data["recommendations"]) == 1
