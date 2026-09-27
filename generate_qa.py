import json
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# 프로젝트 루트 폴더(four-leaf-AI)를 파이썬 경로에 추가하여 모듈을 찾을 수 있도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
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


# --- 2. 합성 데이터 생성 함수 ---
def generate_synthetic_data(context_text: str, num_pairs: int = 5) -> list[dict]:
    """주어진 텍스트 컨텍스트를 바탕으로 sLLM 파인튜닝용 Instruction-Output 쌍을 생성합니다."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
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

    try:
        logger.info(f"합성 QA 데이터 생성 요청 중... (목표: {num_pairs}개)")
        result = chain.invoke(
            {
                "context": context_text,
                "num_pairs": num_pairs,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        pairs = result.get("pairs", [])
        return pairs

    except Exception as e:
        logger.error(f"데이터 생성 중 오류 발생: {e}")
        return []


# --- 3. LLM-as-a-Judge 품질 검증 함수 ---
def validate_qa_pair(pair: dict, context_text: str) -> tuple[bool, str]:
    """LLM-as-a-Judge: 원본 문서를 근거로 생성된 Q&A의 환각(Hallucination) 및 품질을 검증합니다."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=JudgeResult)

    judge_system_prompt = """당신은 AI가 생성한 Q&A 데이터의 품질을 검증하는 엄격한 평가자입니다.
아래 [참고 문서]의 내용만을 근거로 [질문]에 대해 [답변]이 정확하고 충실하게 작성되었는지 평가하세요.

평가 기준:
1. 답변 내용이 참고 문서에 실제로 있는 사실에 기반하는가? (문서에 없는 내용을 지어냈다면 실패)
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
        res = chain.invoke(
            {
                "context": context_text,
                "instruction": pair.get("instruction", ""),
                "output": pair.get("output", ""),
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return res.get("is_pass", False), res.get("reason", "판단 사유 미제공")
    except Exception as e:
        logger.warning(f"품질 검증 과정 중 예외 발생: {e}")
        return False, f"검증 프로세스 에러: {e}"


# --- 4. 메인 실행 함수 ---
def main():
    input_txt_path = os.path.join("data", "knu_schedule.txt")
    validated_output_path = os.path.join("data", "validated_qa_dataset.jsonl")
    rejected_output_path = os.path.join("data", "rejected_qa_dataset.jsonl")

    try:
        with open(input_txt_path, encoding="utf-8") as f:
            context_text = f.read()
    except FileNotFoundError:
        logger.error(f"입력 파일 {input_txt_path}을 찾을 수 없습니다.")
        return

    # 1) 데이터 생성 (10쌍)
    generated_pairs = generate_synthetic_data(context_text, num_pairs=10)
    if not generated_pairs:
        logger.warning("생성된 데이터가 없습니다.")
        return

    logger.info(f"생성 완료된 {len(generated_pairs)}개 데이터에 대해 LLM-as-a-Judge 품질 검증을 시작합니다...")

    passed_count = 0
    rejected_count = 0

    # 2) 생성된 데이터 품질 검수 및 통과/탈락 분리 저장
    for idx, pair in enumerate(generated_pairs):
        instruction = pair.get("instruction", "").strip()
        output = pair.get("output", "").strip()

        # 기본 형식 검증
        if not instruction or not output:
            logger.warning(f"[{idx+1}/{len(generated_pairs)}] 형식 오류 (빈 값 포함) - 탈락")
            rejected_count += 1
            continue

        # LLM-as-a-Judge 품질 평가
        is_pass, reason = validate_qa_pair(pair, context_text)

        if is_pass:
            passed_count += 1
            logger.info(f"[{idx+1}/{len(generated_pairs)}] 통과 ✅ - 사유: {reason}")
            with open(validated_output_path, "a", encoding="utf-8") as f:
                json_line = json.dumps({"instruction": instruction, "output": output}, ensure_ascii=False)
                f.write(json_line + "\n")
        else:
            rejected_count += 1
            logger.info(f"[{idx+1}/{len(generated_pairs)}] 탈락 ❌ - 사유: {reason}")
            with open(rejected_output_path, "a", encoding="utf-8") as f:
                json_line = json.dumps({"instruction": instruction, "output": output, "reject_reason": reason}, ensure_ascii=False)
                f.write(json_line + "\n")

    logger.info(f"\n[최종 결과 Summary] 총 {len(generated_pairs)}개 중 통과: {passed_count}개 / 탈락: {rejected_count}개")


if __name__ == "__main__":
    main()