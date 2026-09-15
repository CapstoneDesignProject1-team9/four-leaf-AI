import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

class LocalFinetunedLLM(BaseChatModel):
    """
    A100에서 파인튜닝된 로컬 오픈소스 모델(Llama-3, Qwen 등)을 서빙하기 위한 LangChain 래퍼 클래스.
    (실제 프로덕션에서는 vLLM을 별도 서버로 띄우고 OpenAI 호환 API로 연동하는 것이 정석이나,
    여기서는 단일 파이프라인에서 직접 인퍼런스하는 예제 클래스를 구성합니다.)
    """

    model_path: str = "logs/lora_model" # 파인튜닝 저장 경로
    max_tokens: int = 1024
    temperature: float = 0.5

    # 내부 파이프라인 객체 (Singleton 패턴 권장)
    _pipeline: Any = None

    @property
    def _llm_type(self) -> str:
        return "local_finetuned"

    def _initialize_pipeline(self):
        if LocalFinetunedLLM._pipeline is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info(f"로컬 모델 로딩 시작: {self.model_path}")
            # 파인튜닝된 모델(LoRA 병합 또는 직접 로드) 로드
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

            LocalFinetunedLLM._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
            logger.info("로컬 모델 로딩 완료.")
        except Exception as e:
            logger.error(f"로컬 모델 로딩 실패: {e}")
            raise e

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:

        self._initialize_pipeline()

        # LangChain 메시지를 텍스트 프롬프트로 변환
        prompt = self._convert_messages_to_prompt(messages)

        # 모델 추론
        outputs = LocalFinetunedLLM._pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            eos_token_id=LocalFinetunedLLM._pipeline.tokenizer.eos_token_id
        )

        generated_text = outputs[0]["generated_text"][len(prompt):]

        message = AIMessage(content=generated_text.strip())
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages_to_prompt(self, messages: list[BaseMessage]) -> str:
        """
        Llama-3 인스트럭션 포맷 등 모델에 맞게 변환
        (여기서는 단순 예시 포맷 적용)
        """
        prompt = ""
        for m in messages:
            if isinstance(m, SystemMessage):
                prompt += f"<|system|>\n{m.content}\n"
            elif isinstance(m, HumanMessage):
                prompt += f"<|user|>\n{m.content}\n"
            elif isinstance(m, AIMessage):
                prompt += f"<|assistant|>\n{m.content}\n"
        prompt += "<|assistant|>\n"
        return prompt
