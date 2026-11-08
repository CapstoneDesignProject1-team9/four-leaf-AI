"""실제 DB·벡터 초기화·AI 호출 없이 서버 API를 검증한다."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.storage.chat_logs import StoredQuestion


@pytest.fixture
def client():
    with (
        patch("app.main.initialize_chat_log_store") as mock_init_db,
        patch(
            "app.chains.rag_chain.init_vectorstore",
            new_callable=AsyncMock,
        ) as mock_init_vector,
        patch("app.api.v1.chat.create_chat_log", return_value=7) as mock_create,
        patch("app.api.v1.chat.complete_chat_log") as mock_complete,
        patch("app.api.v1.chat.fail_chat_log") as mock_fail,
        patch("app.api.v1.chat.build_rag_chain") as mock_builder,
    ):
        from app.main import app

        mock_builder.return_value.invoke.return_value = {
            "result": "테스트 답변",
            "source_documents": [],
        }
        mocks = {
            "create": mock_create,
            "complete": mock_complete,
            "fail": mock_fail,
            "builder": mock_builder,
        }
        with TestClient(app) as test_client:
            yield test_client, mocks

        mock_init_db.assert_called_once_with()
        mock_init_vector.assert_awaited_once_with()


def test_health_endpoint(client):
    test_client, mocks = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "four-leaf-ai"
    mocks["create"].assert_not_called()
    mocks["builder"].assert_not_called()


def test_chat_saves_question_and_answer(client):
    test_client, mocks = client
    question = "취업 준비는 어떻게 해야 하나요?"
    answer = "학교 취업지원센터를 활용하세요."
    mocks["builder"].return_value.invoke.return_value = {
        "result": answer,
        "source_documents": [
            MagicMock(
                page_content="취업지원센터에서 이력서 첨삭을 제공합니다.",
                metadata={"url": "https://example.com/notice/1", "source_type": "notice"},
            )
        ],
    }

    response = test_client.post(
        "/api/v1/tutor/chat", json={"message": question, "session_id": "s1"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == answer
    assert data["sources"][0]["source"] == "https://example.com/notice/1"
    mocks["create"].assert_called_once_with(question=question, session_id="s1")
    mocks["complete"].assert_called_once()
    args, kwargs = mocks["complete"].call_args
    assert args == (7,)
    assert kwargs["answer"] == answer
    assert kwargs["sources"] == ["https://example.com/notice/1"]
    assert kwargs["model_name"]
    mocks["fail"].assert_not_called()
    mocks["builder"].return_value.invoke.assert_called_once_with(
        {"query": question, "model_name": None}
    )


def test_chat_generation_failure_marks_log_failed(client):
    test_client, mocks = client
    mocks["builder"].return_value.invoke.side_effect = RuntimeError("internal-test-error")

    response = test_client.post("/api/v1/tutor/chat", json={"message": "테스트 질문"})

    assert response.status_code == 500
    assert response.json()["detail"] == "AI 튜터 응답 생성 중 오류가 발생했습니다."
    assert "internal-test-error" not in response.text
    mocks["create"].assert_called_once()
    mocks["fail"].assert_called_once()
    assert mocks["fail"].call_args.args == (7,)
    mocks["complete"].assert_not_called()


def test_chat_still_answers_when_storage_is_down(client):
    test_client, mocks = client
    mocks["create"].side_effect = RuntimeError("db down")

    response = test_client.post("/api/v1/tutor/chat", json={"message": "질문"})

    assert response.status_code == 200
    assert response.json()["answer"] == "테스트 답변"
    mocks["complete"].assert_not_called()


def _questions():
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    texts = [
        "캐시 직접 매핑이 뭐야?",
        "캐시 히트율 계산법",
        "파이프라인 데이터 해저드 해결법",
        "데드락 조건이 뭐야?",
    ]
    return [StoredQuestion(i + 1, t, None, now - timedelta(hours=i)) for i, t in enumerate(texts)]


def test_advisor_ask_counts_groups_in_code(client):
    test_client, _ = client
    raw = {
        "topic": "컴퓨터구조",
        # 99는 없는 번호, 1은 중복: 둘 다 무시돼야 한다.
        "groups": [
            {"label": "파이프라인 해저드", "question_ids": [3]},
            {"label": "캐시", "question_ids": [1, 2, 99]},
            {"label": "중복", "question_ids": [1]},
        ],
    }
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=_questions()) as fetch,
        patch("app.api.v1.advisor.group_with_gemini", new=AsyncMock(return_value=raw)),
        patch(
            "app.api.v1.advisor.write_answer_with_gemini",
            new=AsyncMock(return_value="캐시 질문이 가장 많았습니다."),
        ) as write,
    ):
        response = test_client.post(
            "/api/v1/advisor/ask",
            json={"question": "컴퓨터구조 관련 학생들 질문은 뭐가 많았어?", "days": 30},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "캐시 질문이 가장 많았습니다."
    assert data["topic"] == "컴퓨터구조"
    assert data["matched_count"] == 3
    assert data["total_count"] == 4
    assert data["period_days"] == 30
    assert [(g["label"], g["count"]) for g in data["groups"]] == [("캐시", 2), ("파이프라인 해저드", 1)]
    # 대표 질문은 최신순
    assert data["groups"][0]["examples"] == ["캐시 직접 매핑이 뭐야?", "캐시 히트율 계산법"]
    assert fetch.call_args.kwargs["days"] == 30
    assert write.await_args.args[3] == 3  # matched


def test_advisor_ask_without_stored_questions(client):
    test_client, _ = client
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=[]),
        patch("app.api.v1.advisor.group_with_gemini", new=AsyncMock()) as group,
    ):
        response = test_client.post("/api/v1/advisor/ask", json={"question": "학생들 리포트 써줘"})

    assert response.status_code == 200
    assert response.json()["total_count"] == 0
    assert "저장된 학생 질문이 없습니다" in response.json()["answer"]
    group.assert_not_awaited()


def test_advisor_ask_no_matching_questions(client):
    test_client, _ = client
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=_questions()),
        patch(
            "app.api.v1.advisor.group_with_gemini",
            new=AsyncMock(return_value={"topic": "데이터베이스", "groups": []}),
        ),
    ):
        response = test_client.post(
            "/api/v1/advisor/ask", json={"question": "데이터베이스 질문은 뭐가 많았어?"}
        )

    data = response.json()
    assert response.status_code == 200
    assert data["matched_count"] == 0
    assert "'데이터베이스' 관련" in data["answer"]


def test_advisor_ask_falls_back_when_answer_generation_fails(client):
    test_client, _ = client
    raw = {"topic": None, "groups": [{"label": "캐시", "question_ids": [1, 2]}]}
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=_questions()),
        patch("app.api.v1.advisor.group_with_gemini", new=AsyncMock(return_value=raw)),
        patch(
            "app.api.v1.advisor.write_answer_with_gemini",
            new=AsyncMock(side_effect=RuntimeError("quota")),
        ),
    ):
        response = test_client.post("/api/v1/advisor/ask", json={"question": "학생들 리포트 써줘"})

    data = response.json()
    assert response.status_code == 200
    assert "전체 학생 질문은 2건" in data["answer"]
    assert "캐시 (2건)" in data["answer"]


def test_advisor_ask_grouping_failure_returns_503(client):
    test_client, _ = client
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=_questions()),
        patch(
            "app.api.v1.advisor.group_with_gemini",
            new=AsyncMock(side_effect=RuntimeError("invalid key")),
        ),
    ):
        response = test_client.post("/api/v1/advisor/ask", json={"question": "리포트"})

    assert response.status_code == 503
    assert "invalid key" not in response.text
