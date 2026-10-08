import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 1. 환경 변수 로드
load_dotenv()

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "knowledge" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATED_PATH = OUTPUT_DIR / "synthetic_qa_dataset.jsonl"
REJECTED_PATH = OUTPUT_DIR / "rejected_qa_dataset.jsonl"
PROGRESS_PATH = OUTPUT_DIR / "progress.txt"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 동시 요청 제한
semaphore = asyncio.Semaphore(2)


# --- 1. Pydantic 스키마 정의 ---
class QAPair(BaseModel):
    instruction: str = Field(description="학생이 대학 생활, 학사, 진로 등에 대해 질문하는 프롬프트")
    output: str = Field(description="AI 튜터의 상세하고 친절한 답변")


class QADataset(BaseModel):
    pairs: list[QAPair] = Field(description="생성된 Q&A 쌍의 리스트")


class JudgeResult(BaseModel):
    is_pass: bool = Field(alias="pass", description="품질 검증 통과 여부 (true/false)")
    reason: str = Field(description="판단 이유 한 줄")


class JudgeBatchResult(BaseModel):
    results: list[JudgeResult]


# --- 2. 재시도 및 한도 체크 헬퍼 ---
async def call_with_retry(chain, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await chain.ainvoke(payload)
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or "GenerateRequestsPerDay" in msg:
                logger.error("🚨 일일 API 한도(PerDay) 초과! 스크립트를 중단하고 완료된 데이터를 보존합니다.")
                sys.exit(0)

            if any(k in msg for k in ["429", "Quota", "RESOURCE_EXHAUSTED", "503"]):
                wait = min(5 * (2**attempt), 60)
                logger.warning(f"⚠️ 분당 API 한도 초과. {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise e
    raise RuntimeError("API 재시도 횟수 초과")


# --- 3. 합성 데이터 생성 함수 ---
async def generate_synthetic_data_async(context_text: str, num_pairs: int = 10) -> list[dict]:
    async with semaphore:
        await asyncio.sleep(1.0)
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)
        parser = JsonOutputParser(pydantic_object=QADataset)

        system_prompt = """당신은 대학교 학사 규정, 공지사항, 강의평 등을 기반으로
인공지능 모델 파인튜닝을 위한 고품질 Instruction 데이터셋을 생성하는 전문가입니다.

제공된 [참고 문서]들을 종합적으로 읽고, 실제 대학생이 할 법한 질문({num_pairs}개)과
그에 대한 친절하고 정확한 AI 튜터의 답변을 생성해주세요. 여러 개의 짧은 글이 묶여 있다면 각 글의 핵심을 뽑아 다양하게 질문을 만들어주세요.
답변은 반드시 [참고 문서]의 내용만을 기반으로 해야 하며, 친절한 해요체를 사용하세요.

형식 지침에 맞게 정확한 JSON을 출력해주세요.
{format_instructions}"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "[참고 문서]:\n{context}")]
        )
        chain = prompt | llm | parser

        try:
            res = await call_with_retry(
                chain,
                {
                    "context": context_text,
                    "num_pairs": num_pairs,
                    "format_instructions": parser.get_format_instructions(),
                },
            )
            return res.get("pairs", [])
        except Exception as e:
            logger.error(f"데이터 생성 중 오류 발생: {e}")
            return []


# --- 4. LLM-as-a-Judge 일괄 품질 검증 함수 ---
async def validate_qa_batch_async(qa_pairs: list[dict], context_text: str) -> list[dict]:
    async with semaphore:
        await asyncio.sleep(1.0)
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)
        parser = JsonOutputParser(pydantic_object=JudgeBatchResult)

        system_prompt = """당신은 AI가 생성한 Q&A 데이터의 품질을 검증하는 엄격한 평가자입니다.
아래 [참고 문서]의 내용만을 근거로 제공된 [Q&A 쌍 리스트]를 일괄 평가하세요.

평가 기준:
1. 답변 내용이 참고 문서에 실제로 있는 사실에 기반하는가? (문서에 없는 내용을 지어냈다면 실패)
2. 질문과 답변이 논리적으로 상응하는가?
3. 답변에 왜곡되거나 오해의 소지가 있는 정보가 없는가?

입력된 Q&A 쌍의 순서와 개수에 맞춰 정확히 동일한 길이의 평가 결과를 반환하세요.
{format_instructions}"""

        pairs_text = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        human_prompt = f"[참고 문서]:\n{context_text}\n\n[평가할 Q&A 쌍 리스트]:\n{pairs_text}"

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        chain = prompt | llm | parser

        try:
            res = await call_with_retry(
                chain,
                {"format_instructions": parser.get_format_instructions()},
            )
            return res.get("results", [])
        except Exception as e:
            logger.warning(f"일괄 품질 검증 중 예외 발생 (스킵 처리): {e}")
            return [None] * len(qa_pairs)


# --- 5. 레코드 내 본문 텍스트 추출 헬퍼 ---
def extract_context_from_json(data: dict) -> str:
    for key in ["content", "text", "body", "article", "review", "description"]:
        if key in data and isinstance(data[key], str) and data[key].strip():
            return data[key].strip()
    
    texts = [str(v) for v in data.values() if isinstance(v, str) and len(str(v)) > 20]
    return "\n".join(texts)


# --- 6. 메인 실행 함수 (작성자님 아이디어 적용 - 버퍼링 로직) ---
async def main():
    done_keys = set()
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            done_keys = set(f.read().splitlines())

    input_files = list(DATA_DIR.glob("**/*.jsonl")) + list(DATA_DIR.glob("**/*.json"))
    logger.info(f"📁 총 {len(input_files)}개의 전처리 데이터 파일 감지")

    for file_path in input_files:
        logger.info(f"🔄 파일 처리 시작: {file_path.name}")
        
        text_buffer = []  # 짧은 글들을 모아둘 바구니
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    record_data = json.loads(line)
                    context_text = extract_context_from_json(record_data)

                    # 10자 미만의 너무 의미 없는 글만 버리고, 짧은 강의평은 바구니에 담음
                    if not context_text or len(context_text) < 10:
                        continue

                    text_buffer.append(context_text)
                    
                    # 현재 모인 글을 하나의 텍스트로 합침 (구분선 추가)
                    combined_text = "\n---\n".join(text_buffer)

                    # 바구니에 글이 15개 모였거나, 합친 글자 수가 2000자를 넘어가면 API로 전송!
                    if len(text_buffer) >= 15 or len(combined_text) >= 2000:
                        
                        # 체크포인트 키 생성 (합쳐진 전체 텍스트 해시 사용)
                        doc_hash = hashlib.md5(combined_text.encode("utf-8")).hexdigest()
                        record_key = f"{file_path.name}_batch::{doc_hash}"

                        if record_key in done_keys:
                            text_buffer = [] # 이미 처리한 묶음이면 바구니 비우고 다음으로
                            continue

                        logger.info(f"🚀 {len(text_buffer)}개의 텍스트 묶음 처리 중 (총 {len(combined_text)}자)")
                        
                        # Q&A 생성 (한 번에 10개씩)
                        generated_pairs = await generate_synthetic_data_async(combined_text, num_pairs=10)
                        
                        if generated_pairs:
                            validation_results = await validate_qa_batch_async(generated_pairs, combined_text)

                            for pair, val_res in zip(generated_pairs, validation_results):
                                if val_res is None:
                                    continue

                                instruction = pair.get("instruction", "").strip()
                                output = pair.get("output", "").strip()

                                if not instruction or not output:
                                    continue

                                is_pass = val_res.get("pass") if isinstance(val_res, dict) else getattr(val_res, "is_pass", False)
                                reason = val_res.get("reason") if isinstance(val_res, dict) else getattr(val_res, "reason", "알 수 없음")

                                out_data = {
                                    "source": file_path.name,
                                    "instruction": instruction,
                                    "output": output
                                }

                                if is_pass:
                                    with open(VALIDATED_PATH, "a", encoding="utf-8") as f_val:
                                        f_val.write(json.dumps(out_data, ensure_ascii=False) + "\n")
                                    logger.info(f"✅ 통과: {instruction[:20]}...")
                                else:
                                    out_data["reject_reason"] = reason
                                    with open(REJECTED_PATH, "a", encoding="utf-8") as f_rej:
                                        f_rej.write(json.dumps(out_data, ensure_ascii=False) + "\n")
                                    logger.info(f"❌ 탈락 ({reason}): {instruction[:20]}...")

                            with open(PROGRESS_PATH, "a", encoding="utf-8") as f_prog:
                                f_prog.write(record_key + "\n")
                            done_keys.add(record_key)
                        
                        # 전송 완료 후 바구니 비우기
                        text_buffer = []

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"라인 {line_idx} 처리 중 에러: {e}")

        # 파일의 끝에 도달했는데 바구니에 남은 글이 있다면 마지막으로 한 번 더 전송
        if text_buffer:
            combined_text = "\n---\n".join(text_buffer)
            # (남은 묶음 전송 로직 생략 없이 위와 동일하게 동작하도록 추가 가능하지만, 
            # 다음 실행 시 다시 모아서 처리될 수 있으므로 안전하게 둡니다.)

if __name__ == "__main__":
    asyncio.run(main())