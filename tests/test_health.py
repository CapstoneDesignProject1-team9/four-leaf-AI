"""
AI 튜터 헬스체크 테스트
- Clova API 실제 호출 없이 FastAPI 앱 동작 검증
- CI 환경에서 외부 의존성(Clova, ChromaDB) 없이 실행 가능
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    테스트용 FastAPI 클라이언트
    벡터스토어 초기화를 mock으로 대체
    """
    with patch("app.chains.rag_chain.init_vectorstore", new_callable=AsyncMock):
        from app.main import app

        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    """GET /health 응답 확인"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "four-leaf-ai"


def test_chat_endpoint_with_mock(client):
    """
    POST /api/v1/chat 동작 확인 (Clova API mock)
    실제 Clova API 키 없이 CI에서 실행 가능
    """
    mock_result = {
        "result": "대학 취업지원센터를 방문하시면 이력서 첨삭 서비스를 받을 수 있습니다.",
        "source_documents": [
            MagicMock(
                page_content="취업 준비를 위해 학교 취업지원센터를 활용하세요.",
                metadata={"source": "sample", "category": "취업"},
            )
        ],
    }

    with patch("app.api.v1.chat.build_rag_chain") as mock_chain_builder:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_chain_builder.return_value = mock_chain

        response = client.post(
            "/api/v1/tutor/chat",
            json={"message": "취업 준비는 어떻게 해야 하나요?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
