from typing import Any

import httpx

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.config import settings


class HyperClovaXChat(BaseChatModel):
    """
    NAVER HyperCLOVA X LangChain 래퍼
    Clova Studio Chat Completions API 연동

    참고: https://api.ncloud-docs.com/docs/ai-naver-clovastudio-chatcompletions
    """

    model_name: str = settings.CLOVA_MODEL
    max_tokens: int = 1024
    temperature: float = 0.5
    top_p: float = 0.8

    @property
    def _llm_type(self) -> str:
        return "hyperclovax"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """동기 호출 (LangChain 기본)"""
        clova_messages = self._convert_messages(messages)

        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": settings.CLOVA_API_KEY_PRIMARY_VAL,
            "X-NCP-APIGW-API-KEY": settings.CLOVA_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "messages": clova_messages,
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
            "topP": self.top_p,
        }

        url = f"{settings.CLOVA_API_HOST}/testapp/v1/chat-completions/{self.model_name}"

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["result"]["message"]["content"]
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """LangChain 메시지 → Clova Studio 메시지 형식 변환"""
        role_map = {
            HumanMessage: "user",
            AIMessage: "assistant",
            SystemMessage: "system",
        }
        return [
            {"role": role_map.get(type(m), "user"), "content": m.content}
            for m in messages
        ]
