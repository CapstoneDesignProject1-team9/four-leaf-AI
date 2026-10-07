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