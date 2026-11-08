"""실제 DB·벡터 초기화·AI 호출 없이 서버 API를 검증한다."""

from datetime import UTC, datetime, timedelta
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
    now = datetime(2026, 11, 8, tzinfo=UTC)
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
        patch("app.api.v1.advisor.group_with_claude", new=AsyncMock(return_value=raw)),
        patch(
            "app.api.v1.advisor.write_answer_with_claude",
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
    assert [(g["label"], g["count"]) for g in data["groups"]] == [
        ("캐시", 2),
        ("파이프라인 해저드", 1),
    ]
    # 대표 질문은 최신순
    assert data["groups"][0]["examples"] == ["캐시 직접 매핑이 뭐야?", "캐시 히트율 계산법"]
    assert fetch.call_args.kwargs["days"] == 30
    assert write.await_args.args[1].matched == 3
    assert data["task"] == "summary"
    assert data["item_count"] is None


def test_advisor_ask_without_stored_questions(client):
    test_client, _ = client
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=[]),
        patch("app.api.v1.advisor.group_with_claude", new=AsyncMock()) as group,
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
            "app.api.v1.advisor.group_with_claude",
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
        patch("app.api.v1.advisor.group_with_claude", new=AsyncMock(return_value=raw)),
        patch(
            "app.api.v1.advisor.write_answer_with_claude",
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
            "app.api.v1.advisor.group_with_claude",
            new=AsyncMock(side_effect=RuntimeError("invalid key")),
        ),
    ):
        response = test_client.post("/api/v1/advisor/ask", json={"question": "리포트"})

    assert response.status_code == 503
    assert "invalid key" not in response.text


def _ask(test_client, raw, answer="생성 결과", question="컴퓨터구조 퀴즈 만들어줘"):
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=_questions()),
        patch("app.api.v1.advisor.group_with_claude", new=AsyncMock(return_value=raw)),
        patch(
            "app.api.v1.advisor.write_answer_with_claude", new=AsyncMock(return_value=answer)
        ) as write,
    ):
        response = test_client.post("/api/v1/advisor/ask", json={"question": question})
    return response, write


@pytest.mark.parametrize(
    ("task", "requested", "expected"),
    [("quiz", None, 5), ("exam", None, 3), ("quiz", 7, 7), ("quiz", 50, 20), ("exam", 0, 1)],
)
def test_advisor_generation_tasks_clamp_item_count(client, task, requested, expected):
    test_client, _ = client
    raw = {
        "task": task,
        "item_count": requested,
        "topic": "컴퓨터구조",
        "groups": [{"label": "캐시", "question_ids": [1, 2]}],
    }
    response, write = _ask(test_client, raw)

    data = response.json()
    assert response.status_code == 200
    assert data["task"] == task
    assert data["item_count"] == expected
    assert write.await_args.args[1].item_count == expected


def test_advisor_unknown_task_falls_back_to_summary(client):
    test_client, _ = client
    raw = {
        "task": "poem",
        "item_count": 3,
        "topic": None,
        "groups": [{"label": "캐시", "question_ids": [1]}],
    }
    response, _ = _ask(test_client, raw, question="학생 질문 알려줘")

    assert response.json()["task"] == "summary"
    assert response.json()["item_count"] is None


def test_advisor_quiz_fallback_says_generation_failed(client):
    test_client, _ = client
    raw = {
        "task": "quiz",
        "topic": "컴퓨터구조",
        "groups": [{"label": "캐시", "question_ids": [1]}],
    }
    with (
        patch("app.api.v1.advisor.purge_expired_chat_logs"),
        patch("app.api.v1.advisor.fetch_recent_questions", return_value=_questions()),
        patch("app.api.v1.advisor.group_with_claude", new=AsyncMock(return_value=raw)),
        patch(
            "app.api.v1.advisor.write_answer_with_claude",
            new=AsyncMock(side_effect=RuntimeError("quota")),
        ),
    ):
        response = test_client.post("/api/v1/advisor/ask", json={"question": "퀴즈 만들어줘"})

    assert response.status_code == 200
    assert response.json()["answer"].startswith("퀴즈를 생성하지 못해 집계 결과만 보여드립니다.")


@pytest.mark.parametrize(
    ("task", "expected_phrases"),
    [
        ("quiz", ["정확히 7개", "4지선다", "검토해 주세요"]),
        ("exam", ["정확히 7개", "모범답안", "채점 기준"]),
        ("difficulty", ["어려워하는 부분", "보충할 방법"]),
        ("report", ["'요약', '주요 질문 유형', '학생들이 어려워하는 부분', '강의 제안'"]),
    ],
)
def test_task_prompts_reach_claude(task, expected_phrases):
    import asyncio
    from types import SimpleNamespace

    from app.api.v1 import advisor

    fake = MagicMock()
    fake.beta.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")]
        )
    )
    analysis = advisor.build_analysis(
        {
            "task": task,
            "item_count": 7,
            "topic": "컴퓨터구조",
            "groups": [{"label": "캐시 {매핑}", "question_ids": [1, 2]}],
        },
        _questions(),
    )
    with patch.object(advisor, "_claude", return_value=fake):
        answer = asyncio.run(advisor.write_answer_with_claude("질문", analysis, 4, None))

    assert answer == "ok"
    kwargs = fake.beta.messages.create.await_args.kwargs
    assert kwargs["model"] == advisor.settings.ADVISOR_MODEL
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == [advisor.FALLBACK_BETA]
    for phrase in expected_phrases:
        assert phrase in kwargs["system"]
    assert "{item_count}" not in kwargs["system"]
    human = kwargs["messages"][0]["content"]
    assert "캐시 {매핑} (2건)" in human
    assert "캐시 직접 매핑이 뭐야?" in human


def test_refused_answer_raises_so_fallback_text_is_used():
    import asyncio
    from types import SimpleNamespace

    from app.api.v1 import advisor

    fake = MagicMock()
    fake.beta.messages.create = AsyncMock(
        return_value=SimpleNamespace(stop_reason="refusal", content=[])
    )
    analysis = advisor.build_analysis(
        {"task": "summary", "groups": [{"label": "캐시", "question_ids": [1]}]}, _questions()
    )
    with patch.object(advisor, "_claude", return_value=fake), pytest.raises(RuntimeError):
        asyncio.run(advisor.write_answer_with_claude("질문", analysis, 4, None))


def test_grouping_request_shape_through_real_sdk():
    """실제 Anthropic SDK가 보내는 HTTP 요청(모델, 대체 모델 베타, JSON 스키마)과 응답 파싱을 확인한다."""
    import asyncio
    import json

    import httpx
    from anthropic import AsyncAnthropic

    from app.api.v1 import advisor

    captured = {}
    reply = {
        "task": "quiz",
        "item_count": 3,
        "topic": "컴퓨터구조",
        "groups": [{"label": "캐시", "question_ids": [1, 2]}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": advisor.settings.ADVISOR_MODEL,
                "content": [{"type": "text", "text": json.dumps(reply, ensure_ascii=False)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 10},
            },
        )

    client = AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with patch.object(advisor, "_claude", return_value=client):
        raw = asyncio.run(advisor.group_with_claude("컴퓨터구조 퀴즈 3개", _questions()))

    assert raw == reply
    body = captured["body"]
    assert body["model"] == advisor.settings.ADVISOR_MODEL
    assert body["fallbacks"] == "default"
    assert advisor.FALLBACK_BETA in captured["headers"]["anthropic-beta"]
    assert "temperature" not in body
    schema = body["output_config"]["format"]["schema"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert set(schema["properties"]) >= {"task", "item_count", "topic", "groups"}
    assert "[1] 캐시 직접 매핑이 뭐야?" in body["messages"][0]["content"]


def test_chat_retries_while_llm_server_wakes_up(client):
    test_client, mocks = client
    mocks["builder"].return_value.invoke.side_effect = [
        RuntimeError('Error code: 500 - {"error":"no upstreams available"}'),
        {"result": "깨어난 뒤 답변", "source_documents": []},
    ]
    with patch("app.api.v1.chat.asyncio.sleep", new=AsyncMock()) as sleep:
        response = test_client.post("/api/v1/tutor/chat", json={"message": "질문"})

    assert response.status_code == 200
    assert response.json()["answer"] == "깨어난 뒤 답변"
    assert mocks["builder"].return_value.invoke.call_count == 2
    sleep.assert_awaited_once()
    mocks["complete"].assert_called_once()


def test_chat_returns_503_when_llm_server_stays_cold(client):
    test_client, mocks = client
    mocks["builder"].return_value.invoke.side_effect = RuntimeError("no upstreams available")
    with (
        patch("app.api.v1.chat.settings.VLLM_COLD_START_WAIT_S", 0.0),
        patch("app.api.v1.chat.asyncio.sleep", new=AsyncMock()),
    ):
        response = test_client.post("/api/v1/tutor/chat", json={"message": "질문"})

    assert response.status_code == 503
    assert response.json()["detail"] == "AI 모델을 준비하고 있어요. 1~2분 후 다시 질문해 주세요."
    mocks["fail"].assert_called_once()
