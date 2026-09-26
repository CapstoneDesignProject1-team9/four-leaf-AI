from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    score: int = Field(description="1점부터 5점까지의 평가 점수")
    reason: str = Field(description="점수를 부여한 구체적인 이유")


async def evaluate_responses_batch(eval_items: list[dict]) -> list[dict]:
    """
    LLM-as-a-Judge 기법: GPT-4o를 심판으로 사용하여
    파인튜닝된 모델의 응답 품질을 자동 평가합니다. (비동기 병렬 처리)
    """
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=EvalResult)

    prompt_template = """당신은 인공지능 모델의 답변 품질을 평가하는 공정한 심판입니다.
아래의 [질문], [모범 답안], 그리고 인공지능이 생성한 [모델의 답변]을 읽고 평가를 진행해주세요.

[구체적 채점 루브릭]
5점 (매우 우수): 
 - 모범 답안의 사실과 100% 일치함
 - 대학 AI 튜터로서 매우 친절하고 명확한 어조를 완벽히 구사함
 - 환각(Hallucination)이 전혀 없으며, 질문자가 필요한 정보를 충분히 제공함

4점 (우수):
 - 모범 답안의 사실과 대부분 일치하나, 아주 사소한 디테일이 빠져있음
 - 어조가 대체로 적절하고 환각이 없음

3점 (보통):
 - 핵심 사실은 맞으나, 부가 설명이 현저히 부족하거나 어투가 딱딱하고 기계적임
 - 모범 답안의 내용 중 일부를 생략하여 유용성이 다소 떨어짐 (환각은 없음)

2점 (미흡):
 - 모범 답안의 사실과 일부 모순되는 내용이 포함됨
 - 혹은 불필요한 환각(지어낸 내용)이 섞여 있어 사용자에게 혼동을 줄 수 있음

1점 (매우 나쁨):
 - 모범 답안과 완전히 상반된 정보를 제공하여 치명적인 오안내를 함
 - 대학 정보와 무관한 심각한 환각(Hallucination)이 포함됨

위 기준을 바탕으로 1점에서 5점 사이의 점수를 매기고 그 이유를 설명해주세요.
{format_instructions}

[질문]: {question}
[모범 답안]: {ground_truth}
[모델의 답변]: {generated_answer}
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 AI 모델 평가 전문가입니다."), ("human", prompt_template)]
    )

    chain = prompt | evaluator_llm | parser

    # eval_items의 형태: [{"question": "...", "ground_truth": "...", "generated_answer": "...", "format_instructions": ...}, ...]
    try:
        results = await chain.abatch(eval_items)
        return results
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        return [{"score": 0, "reason": f"Error: {e}"} for _ in eval_items]


async def async_main():
    question = "복수전공 신청 기준이 어떻게 되나요?"
    ground_truth = "복수전공은 2학년 1학기부터 신청 가능하며, 직전 학기까지의 평점평균이 3.0 이상이어야 합니다."

    # 예시 모델들의 가상 응답
    responses = {
        "Base Model": "복수전공 신청은 학교마다 다릅니다. 홈페이지를 참고하세요.",
        "Fine-tuned Model": "학생님! 복수전공 신청은 2학년 1학기부터 가능하며, 평점 3.0 이상이어야 신청할 수 있습니다.",
    }

    # 비동기 처리를 위해 입력 데이터 리스트(배치) 생성
    parser = JsonOutputParser(pydantic_object=EvalResult)
    format_instructions = parser.get_format_instructions()

    eval_items = []
    model_names = []

    for model_name, answer in responses.items():
        eval_items.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "format_instructions": format_instructions
        })
        model_names.append(model_name)

    logger.info("GPT-4o(ChatGPT-5급) 기반 비동기 평가 시작...")
    results = await evaluate_responses_batch(eval_items)

    for model_name, res in zip(model_names, results):
        logger.info(f"[{model_name}] 결과: {res}\n")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
