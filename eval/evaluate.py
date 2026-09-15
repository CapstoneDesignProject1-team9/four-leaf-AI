import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    score: int = Field(description="1점부터 5점까지의 평가 점수")
    reason: str = Field(description="점수를 부여한 구체적인 이유")


def evaluate_response_llm_as_a_judge(
    question: str, ground_truth: str, generated_answer: str
) -> dict:
    """
    LLM-as-a-Judge 기법: HyperCLOVA X(또는 GPT-4)를 심판으로 사용하여
    파인튜닝된 모델의 응답 품질을 자동 평가합니다.
    논문 작성 시 ROUGE/BLEU 스코어와 함께 정성적/정량적 지표로 활용할 수 있습니다.
    """
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    parser = JsonOutputParser(pydantic_object=EvalResult)

    prompt_template = """당신은 인공지능 모델의 답변 품질을 평가하는 공정한 심판입니다.
아래의 [질문], [모범 답안], 그리고 인공지능이 생성한 [모델의 답변]을 읽고 평가를 진행해주세요.

평가 기준:
1. 정확성 (모범 답안의 사실과 일치하는가?)
2. 적절성 (대학 AI 튜터로서 친절하고 명확한 어조를 사용했는가?)
3. 환각(Hallucination) 여부 (모범 답안에 없는 내용을 지어내지 않았는가?)

위 기준을 종합하여 1점(매우 나쁨)에서 5점(매우 우수) 사이의 점수를 매기고 그 이유를 설명해주세요.
{format_instructions}

[질문]: {question}
[모범 답안]: {ground_truth}
[모델의 답변]: {generated_answer}
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 AI 모델 평가 전문가입니다."), ("human", prompt_template)]
    )

    chain = prompt | evaluator_llm | parser

    try:
        result = chain.invoke(
            {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result
    except Exception as e:
        logger.error(f"평가 중 오류: {e}")
        return {"score": 0, "reason": "Error during evaluation"}


def main():
    # 실제 실험 시:
    # 1. Base Model 응답 생성
    # 2. Base Model + RAG 응답 생성
    # 3. Fine-tuned Model 응답 생성
    # 4. Fine-tuned Model + RAG 응답 생성
    # 각 모델별로 50~100개 테스트셋에 대해 LLM-as-a-Judge 평가를 수행하고 평균 점수를 구함.

    question = "복수전공 신청 기준이 어떻게 되나요?"
    ground_truth = "복수전공은 2학년 1학기부터 신청 가능하며, 직전 학기까지의 평점평균이 3.0 이상이어야 합니다."

    # 예시 모델들의 가상 응답
    responses = {
        "Base Model": "복수전공 신청은 학교마다 다릅니다. 홈페이지를 참고하세요.",
        "Fine-tuned Model": "학생님! 복수전공 신청은 2학년 1학기부터 가능하며, 평점 3.0 이상이어야 신청할 수 있습니다.",
    }

    results = {}
    for model_name, answer in responses.items():
        logger.info(f"[{model_name}] 평가 중...")
        eval_res = evaluate_response_llm_as_a_judge(question, ground_truth, answer)
        results[model_name] = eval_res
        logger.info(f"결과: {eval_res}\n")


if __name__ == "__main__":
    main()
