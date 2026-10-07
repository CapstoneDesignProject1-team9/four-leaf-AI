import asyncio
import json
import logging
import os
import sys
from glob import glob

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 1. Pydantic 스키마 정의 ---
class QAPair(BaseModel):
    instruction: str = Field(description="학생이 대학 생활, 학사, 진로 등에 대해 질문하는 프롬프트")
    output: str = Field(description="AI 튜터의 상세하고 친절한 답변")


class QADataset(BaseModel):
    pairs: list[QAPair] = Field(description="생성된 Q&A 쌍의 리스트")


class JudgeResult(BaseModel):
    is_pass: bool = Field(alias="pass", description="품질 검증 통과 여부 (true/false)")
    reason: str = Field(description="판단 이유 한 줄")


# --- 2. 다중 폴더 및 JSON/JSONL/TXT 파일 자동 수집 함수 ---
def get_all_raw_files():
    """knowledge/raw 및 knowledge/raw/notices 경로의 모든 .json, .jsonl, .txt 파일 수집"""
    input_files = []
    
    # 지정된 상대 경로
    target_dirs = [
        os.path.join("knowledge", "raw"),
        os.path.join("knowledge", "raw", "notices")
    ]

    for t_dir in target_dirs:
        if os.path.exists(t_dir):
            for root, _, files in os.walk(t_dir):
                for file in files:
                    if file.endswith((".json", ".jsonl", ".txt")):
                        full_path = os.path.abspath(os.path.join(root, file))
                        if full_path not in input_files:
                            input_files.append(full_path)

    return input_files


# --- 3. 비동기 synthetic 데이터 생성 (Gemini API) ---
semaphore = asyncio.Semaphore(2)  # 429 Rate Limit 방지용 동시 요청 제한


async def generate_synthetic_data_async(context_text: str, num_pairs: int = 5) -> list[dict]:
    async with semaphore:
        await asyncio.sleep(1.5)
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)
        parser = JsonOutputParser(pydantic_object=QADataset)

        system_prompt = """당신은 대학교 학사 규정, 공지사항, 진로 가이드 등을 기반으로
인공지능 모델 파인튜닝을 위한 고품질 Instruction 데이터셋을 생성하는 전문가입니다.

제공된 [참고 문서]를 읽고, 실제 대학생이 할 법한 질문({num_pairs}개)과
그에 대한 친절하고 정확한 AI 튜터의 답변을 생성해주세요.
답변은 반드시 [참고 문서]의 내용만을 기반으로 해야 하며, 친절한 해요체를 사용하세요.

형식 지침에 맞게 정확한 JSON을 출력해주세요.
{format_instructions}
"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "[참고 문서]:\n{context}")]
        )

        chain = prompt | llm | parser

        for attempt in range(3):
            try:
                result = await chain.ainvoke(
                    {
                        "context": context_text,
                        "num_pairs": num_pairs,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
                return result.get("pairs", [])
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    logger.warning("⚠️ API 요청 제한(Rate Limit) 발생! 5초 대기 후 재시도...")
                    await asyncio.sleep(5)
                else:
                    logger.error(f"데이터 생성 에러: {e}")
                    break
        return []


# --- 4. 비동기 LLM-as-a-Judge 품질 검증 ---
async def validate_qa_pair_async(pair: dict, context_text: str) -> tuple[bool, str]:
    async with semaphore:
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)
        parser = JsonOutputParser(pydantic_object=JudgeResult)

        judge_system_prompt = """당신은 AI가 생성한 Q&A 데이터의 품질을 검증하는 엄격한 평가자입니다.
아래 [참고 문서]의 내용만을 근거로 [질문]에 대해 [답변]이 정확하고 충실하게 작성되었는지 평가하세요.

평가 기준:
1. 답변 내용이 참고 문서에 실제로 있는 사실에 기반하는가?
2. 질문과 답변이 논리적으로 상응하는가?
3. 답변에 왜곡되거나 오해의 소지가 있는 정보가 없는가?

형식 지침에 맞게 정확한 JSON을 출력해주세요.
{format_instructions}
"""

        human_prompt = """[참고 문서]:
{context}

[질문]:
{instruction}

[답변]:
{output}
"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", judge_system_prompt), ("human", human_prompt)]
        )

        chain = prompt | llm | parser

        try:
            res = await chain.ainvoke(
                {
                    "context": context_text,
                    "instruction": pair.get("instruction", ""),
                    "output": pair.get("output", ""),
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return res.get("is_pass", False), res.get("reason", "판단 사유 미제공")
        except Exception as e:
            return False, f"검증 에러: {e}"


# --- 5. 메인 비동기 실행 루틴 ---
async def main_async():
    TARGET_COUNT = 10000  # 1차 목표 수량
    current_count = 0

    input_files = get_all_raw_files()

    output_dir = os.path.join("data")
    os.makedirs(output_dir, exist_ok=True)

    validated_output_path = os.path.join(output_dir, "validated_qa_dataset.jsonl")
    rejected_output_path = os.path.join(output_dir, "rejected_qa_dataset.jsonl")

    if not input_files:
        logger.error("knowledge/raw 및 knowledge/raw/notices 에서 크롤링 파일(.json, .jsonl, .txt)을 찾을 수 없습니다.")
        return

    logger.info(f"🎯 목표 수량: {TARGET_COUNT}개 | 감지된 총 크롤링 원본 파일 수: {len(input_files)}개")

    for file_path in input_files:
        if current_count >= TARGET_COUNT:
            break

        contexts = []
        try:
            # 1) .jsonl 파일 읽기
            if file_path.endswith(".jsonl"):
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            text = data.get("content") or data.get("text") or data.get("body") or str(data)
                            if len(str(text).strip()) > 30:
                                contexts.append(str(text).strip())
            
            # 2) .json 파일 읽기
            elif file_path.endswith(".json"):
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            text = item.get("content") or item.get("text") or item.get("body") or str(item) if isinstance(item, dict) else str(item)
                            if len(str(text).strip()) > 30:
                                contexts.append(str(text).strip())
                    elif isinstance(data, dict):
                        text = data.get("content") or data.get("text") or data.get("body") or str(data)
                        if len(str(text).strip()) > 30:
                            contexts.append(str(text).strip())

            # 3) .txt 파일 읽기
            elif file_path.endswith(".txt"):
                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()
                    if len(content) > 30:
                        contexts.append(content)

        except Exception as e:
            logger.error(f"파일 읽기 오류 ({file_path}): {e}")
            continue

        logger.info(f"📄 파일 파싱 중: {os.path.basename(file_path)} (추출 텍스트 블록: {len(contexts)}개)")

        for context_text in contexts:
            if current_count >= TARGET_COUNT:
                break

            pairs = await generate_synthetic_data_async(context_text, num_pairs=5)
            if not pairs:
                continue

            tasks = [validate_qa_pair_async(pair, context_text) for pair in pairs]
            results = await asyncio.gather(*tasks)

            for pair, (is_pass, reason) in zip(pairs, results):
                instruction = pair.get("instruction", "").strip()
                output = pair.get("output", "").strip()

                if not instruction or not output:
                    continue

                if is_pass:
                    current_count += 1
                    with open(validated_output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"instruction": instruction, "output": output}, ensure_ascii=False) + "\n")
                    logger.info(f"[{current_count}/{TARGET_COUNT}] ✅ 통과: {instruction[:25]}...")
                else:
                    with open(rejected_output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"instruction": instruction, "output": output, "reject_reason": reason}, ensure_ascii=False) + "\n")

    logger.info(f"\n🎉 1차 작업 완료! 생성된 고품질 데이터 수: {current_count}개")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
    