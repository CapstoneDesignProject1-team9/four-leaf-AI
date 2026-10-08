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

# 1. 환경 변수 로드 (API 키 인식)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "knowledge" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATED_PATH = OUTPUT_DIR / "synthetic_qa_dataset.jsonl"
REJECTED_PATH = OUTPUT_DIR / "rejected_qa_dataset.jsonl"
PROGRESS_PATH = OUTPUT_DIR / "progress.txt"

semaphore = asyncio.Semaphore(2)

# --- Pydantic 모델 ---
class QAPair(BaseModel):
    question: str = Field(description="대학생이 질문할 법한 질문")
    answer: str = Field(description="참고 문서에 기반한 정확한 답변")

class QADataset(BaseModel):
    pairs: list[QAPair]

class ValidationResult(BaseModel):
    is_pass: bool = Field(description="통과 여부")
    reason: str = Field(description="판단 사유")

class ValidationBatchResult(BaseModel):
    results: list[ValidationResult]

# --- 재시도 및 한도 체크 헬퍼 ---
async def call_with_retry(chain, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await chain.ainvoke(payload)
        except Exception as e:
            msg = str(e)
            # 3. 일일 한도(PerDay) 초과 시 무의미한 대기 없이 즉시 종료
            if "PerDay" in msg or "GenerateRequestsPerDay" in msg:
                logger.error("일일 API 한도(PerDay) 초과! 데이터를 보존하고 스크립트를 즉시 종료합니다.")
                sys.exit(0)
                
            if any(k in msg for k in ["429", "Quota", "RESOURCE_EXHAUSTED", "503"]):
                wait = min(5 * (2**attempt), 60)
                logger.warning(f"분당 API 한도 초과. {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise e
    raise RuntimeError("API 재시도 횟수 초과")

# --- 청킹(Chunking) 유틸리티 ---
def chunk_text(text: str, chunk_size: int = 3000) -> list[str]:
    """긴 문서를 적절한 길이로 잘라서 리스트로 반환"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- 데이터 생성 (15쌍 배치) ---
async def generate_synthetic_data_async(context_text: str, num_pairs: int = 15) -> list[dict]:
    async with semaphore:
        await asyncio.sleep(1.0)
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)
        parser = JsonOutputParser(pydantic_object=QADataset)

        system_prompt = """당신은 인공지능 모델 파인튜닝용 데이터를 만드는 전문가입니다.
[참고 문서]에서 학생들이 자주 물어볼 핵심 질문-답변 쌍 {num_pairs}개를 생성하세요.
{format_instructions}"""
        
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "[참고 문서]:\n{context}")])
        chain = prompt | llm | parser

        try:
            res = await call_with_retry(
                chain, 
                {"context": context_text, "num_pairs": num_pairs, "format_instructions": parser.get_format_instructions()}
            )
            return res.get("pairs", [])
        except Exception as e:
            logger.error(f"데이터 생성 실패: {e}")
            return []

# --- 2. 데이터 검증 (일괄 배치 처리) ---
async def validate_qa_batch_async(context_text: str, qa_pairs: list[dict]) -> list[ValidationResult]:
    async with semaphore:
        await asyncio.sleep(1.0)
        # 검증 모델을 분리할 경우 여기서 모델명을 변경할 수 있습니다.
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.1)
        parser = JsonOutputParser(pydantic_object=ValidationBatchResult)

        # 5. 엄격한 평가 기준 복구
        system_prompt = """당신은 Q&A 품질 검증관입니다. 제공된 문서에 기반하여 다음 Q&A 쌍 리스트를 일괄 평가하세요.
다음 3가지 기준을 엄격히 적용하세요:
1. 사실 부합성: 답변이 문서 내용과 일치하는가?
2. 완결성: 질문에 대해 충분한 정보를 제공하는가?
3. 유용성: 실제 사용자가 유용하게 여길 만한 질문인가?

입력된 Q&A 쌍의 순서와 개수에 맞춰 정확히 동일한 길이의 평가 결과를 반환하세요.
{format_instructions}"""

        pairs_text = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        human_prompt = f"[참고 문서]:\n{context_text}\n\n[평가할 Q&A 쌍 리스트]:\n{pairs_text}"

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        chain = prompt | llm | parser

        try:
            res = await call_with_retry(
                chain,
                {"format_instructions": parser.get_format_instructions()}
            )
            return res.get("results", [])
        except Exception as e:
            logger.warning(f"API 오류 (건너뜀): {e}")
            return [None] * len(qa_pairs)  # API 에러 시 None 반환

# --- 메인 실행 ---
async def main():
    done_keys = set()
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            done_keys = set(f.read().splitlines())

    # 4. jsonl 확장자 추가 
    input_files = list(DATA_DIR.glob("**/*.json")) + list(DATA_DIR.glob("**/*.jsonl"))
    
    for file_path in input_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 긴 문서는 3000자 단위로 분할하여 처리
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                doc_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                doc_key = f"{file_path.name}_chunk_{i}::{doc_hash}"

                if doc_key in done_keys:
                    continue

                logger.info(f"Processing: {file_path.name} (청크 {i+1}/{len(chunks)})")
                
                qa_pairs = await generate_synthetic_data_async(chunk, num_pairs=15)
                if not qa_pairs:
                    continue

                # 15개의 Q&A 쌍을 한 번의 API 호출로 통째로 검증
                validation_results = await validate_qa_batch_async(chunk, qa_pairs)

                for pair, val_res in zip(qa_pairs, validation_results):
                    # API 에러로 None 반환된 경우 버리지 않고 스킵
                    if val_res is None:
                        continue

                    record = {
                        "source": file_path.name,
                        "question": pair.get("question"),
                        "answer": pair.get("answer")
                    }

                    # val_res는 dict 형태로 반환될 수 있으므로 처리
                    is_pass = val_res.get("is_pass") if isinstance(val_res, dict) else getattr(val_res, "is_pass", False)
                    reason = val_res.get("reason") if isinstance(val_res, dict) else getattr(val_res, "reason", "알 수 없음")

                    if is_pass:
                        with open(VALIDATED_PATH, "a", encoding="utf-8") as f_val:
                            f_val.write(json.dumps(record, ensure_ascii=False) + "\n")
                        logger.info(f"Pass: {record['question'][:20]}...")
                    else:
                        record["reject_reason"] = reason
                        with open(REJECTED_PATH, "a", encoding="utf-8") as f_rej:
                            f_rej.write(json.dumps(record, ensure_ascii=False) + "\n")
                        logger.info(f"Fail ({reason}): {record['question'][:20]}...")

                with open(PROGRESS_PATH, "a", encoding="utf-8") as f_prog:
                    f_prog.write(doc_key + "\n")
                done_keys.add(doc_key)

        except Exception as e:
            logger.error(f"파일 처리 중 에러 발생 ({file_path.name}): {e}")

if __name__ == "__main__":
    asyncio.run(main())