import asyncio
import json
import logging
import os
import random

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    score: int = Field(description="1점부터 5점까지의 평가 점수")
    reason: str = Field(description="점수를 부여한 구체적인 이유")


_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def calculate_rouge_l(ground_truth: str, generated_answer: str) -> float:
    """
    모범 답안(Ground Truth)과 생성된 답변(Generated Answer) 간의 ROUGE-L F1 점수를 계산합니다.
    """
    if not ground_truth or not generated_answer or not ground_truth.strip() or not generated_answer.strip():
        return 0.0

    scores = _rouge_scorer.score(target=ground_truth, prediction=generated_answer)
    return round(scores["rougeL"].fmeasure, 4)


async def evaluate_single_item(
    chain,
    item: dict,
    semaphore: asyncio.Semaphore,
    item_index: int,
    total_count: int,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> dict:
    """
    단일 평가 항목을 수행하며, Semaphore로 동시 실행 수를 제한하고
    에러 발생 시 지수 백오프(Exponential Backoff)로 재시도합니다.
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                result = await chain.ainvoke(item)
                logger.info(
                    f"[{item_index}/{total_count}] 평가 완료 - 점수: {result.get('score')}점"
                )
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2**attempt) + random.uniform(0.1, 1.0)
                    logger.warning(
                        f"[{item_index}/{total_count}] 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}. "
                        f"{delay:.1f}초 후 재시도..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[{item_index}/{total_count}] 최대 재시도 횟수({max_retries}) 초과 실패: {e}"
                    )
                    return {"score": 0, "reason": f"Error after {max_retries} retries: {e}"}
        return {"score": 0, "reason": "Unknown evaluation failure"}


async def evaluate_responses_batch(
    eval_items: list[dict],
    max_concurrency: int = 5,
    max_retries: int = 3,
) -> list[dict]:
    """
    LLM-as-a-Judge 기법: Claude 3 Opus(Anthropic)를 심판으로 사용하여
    파인튜닝된 모델의 응답 품질을 자동 평가합니다.
    (asyncio.Semaphore 기반 동시 요청 수 제어 및 지수 백오프 재시도 적용)
    """
    claude_model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    evaluator_llm = ChatAnthropic(model=claude_model, temperature=0.0)
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

    semaphore = asyncio.Semaphore(max_concurrency)
    total_count = len(eval_items)

    tasks = [
        evaluate_single_item(
            chain=chain,
            item=item,
            semaphore=semaphore,
            item_index=idx + 1,
            total_count=total_count,
            max_retries=max_retries,
        )
        for idx, item in enumerate(eval_items)
    ]

    results = await asyncio.gather(*tasks)
    return results


async def async_main():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")

    if not os.path.exists(dataset_path):
        logger.error(f"테스트 데이터셋 파일이 없습니다: {dataset_path}")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    parser = JsonOutputParser(pydantic_object=EvalResult)
    format_instructions = parser.get_format_instructions()

    eval_items = []
    metadata = []  # 결과를 매핑하기 위한 메타데이터

    for item_idx, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        for model_name, answer in item["models"].items():
            eval_items.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": answer,
                    "format_instructions": format_instructions,
                }
            )
            metadata.append({"item_idx": item_idx, "model_name": model_name})

    max_concurrency = int(os.getenv("EVAL_MAX_CONCURRENCY", "5"))
    logger.info(
        f"총 {len(dataset)}개의 질문, {len(eval_items)}개의 응답에 대한 비동기 평가 시작... (동시 실행 제한: {max_concurrency})"
    )
    results = await evaluate_responses_batch(eval_items, max_concurrency=max_concurrency)

    # 평가 결과 출력 및 평균 계산용 변수
    model_scores = {}
    model_rouge_scores = {}

    # CSV 저장을 위한 데이터 구성
    csv_data = []

    for meta, item, res in zip(metadata, eval_items, results):
        q_idx = meta["item_idx"] + 1
        m_name = meta["model_name"]
        score = res.get("score", 0)
        reason = res.get("reason", "")
        rouge_l_score = calculate_rouge_l(
            ground_truth=item["ground_truth"],
            generated_answer=item["generated_answer"],
        )

        logger.info(
            f"[질문 {q_idx} | {m_name}] 점수: {score} | ROUGE-L: {rouge_l_score:.4f} - {reason}"
        )

        if m_name not in model_scores:
            model_scores[m_name] = []
            model_rouge_scores[m_name] = []
        model_scores[m_name].append(score)
        model_rouge_scores[m_name].append(rouge_l_score)

        # CSV 행 추가
        csv_data.append(
            {
                "Question_ID": q_idx,
                "Question": item["question"],
                "Ground_Truth": item["ground_truth"],
                "Model": m_name,
                "Generated_Answer": item["generated_answer"],
                "Opus Score": score,
                "ROUGE-L Score": rouge_l_score,
                "Reason": reason,
            }
        )

    # 평균 점수 집계 및 출력
    logger.info("=" * 40)
    logger.info("📊 모델별 최종 평균 점수")
    logger.info("=" * 40)
    for m_name in model_scores:
        scores = model_scores[m_name]
        rouge_scores = model_rouge_scores[m_name]
        avg_llm = sum(scores) / len(scores) if scores else 0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        logger.info(
            f" - {m_name}: Opus {avg_llm:.2f}점 | ROUGE-L {avg_rouge:.4f} (총 {len(scores)}개)"
        )

    # 결과를 CSV로 저장
    import csv

    output_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    if csv_data:
        keys = csv_data[0].keys()
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(csv_data)
        logger.info(f"✅ 평가 결과가 '{output_path}'에 저장되었습니다.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
