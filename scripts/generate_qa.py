import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "knowledge" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# ⚙️ 핵심 설정
# ---------------------------------------------------------
TEST_MODE = False # True: 각 소스별 1묶음씩만 빠르게 테스트 생성 False: 500개 전체 생성
CONCURRENCY = 5   # 동시 처리 워커(Worker) 수

# 각 소스별 목표량 및 묶음 설정
SOURCES = {
    "syllabi": {"match": ["syllabus"], "target": 100, "group_size": 1, "pairs_per_batch": 3},
    "everytime": {"match": ["reviews"], "target": 200, "group_size": 10, "pairs_per_batch": 5},
    "notices": {"match": ["notices"], "target": 200, "group_size": 1, "pairs_per_batch": 3},
}

prefix = "test_" if TEST_MODE else ""
VALIDATED_PATH = OUTPUT_DIR / f"{prefix}synthetic_qa_dataset.jsonl"
REJECTED_PATH = OUTPUT_DIR / f"{prefix}rejected_qa_dataset.jsonl"
PROGRESS_PATH = OUTPUT_DIR / "progress.txt"

# 본문 앞에 붙여줄 필수 메타데이터 키워드
META_KEYS = ["title", "course_name", "과목명", "강의명", "professor", "교수명", "학기"]
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 각 카테고리별 누적 카운트 저장소
passed_counts = {k: 0 for k in SOURCES.keys()}
attempted = {k: 0 for k in SOURCES.keys()}

class QAPair(BaseModel):
    instruction: str = Field(description="학생이 질문하는 프롬프트")
    output: str = Field(description="AI 튜터의 상세하고 친절한 답변")

class QADataset(BaseModel):
    pairs: list[QAPair]

class JudgeResult(BaseModel):
    is_pass: bool = Field(alias="pass", description="통과 여부 (true/false)")
    reason: str = Field(description="판단 이유 한 줄")

class JudgeBatchResult(BaseModel):
    results: list[JudgeResult]

async def call_with_retry(chain, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await chain.ainvoke(payload)
        except Exception as e:
            msg = str(e).lower()
            if "credit balance" in msg:
                logger.error("🚨 API 크레딧 소진! 충전 후 다시 실행하세요.")
                sys.exit(0)
            if any(k in msg for k in ["429", "529", "503", "overloaded", "rate_limit"]):
                wait = min(5 * (2**attempt), 60)
                logger.warning(f"⚠️ API 과부하. {wait}초 대기 후 재시도...")
                await asyncio.sleep(wait)
            else:
                raise e
    raise RuntimeError("API 재시도 횟수 초과")

async def generate_synthetic_data_async(context_text: str, num_pairs: int) -> list[dict]:
    # 클로드 지적 1번 반영: 모델명 수정 및 temperature 제거(기본값 사용)
    llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=8192)
    parser = JsonOutputParser(pydantic_object=QADataset)
    system_prompt = """당신은 인공지능 파인튜닝을 위한 고품질 Instruction 데이터셋 생성 전문가입니다.
제공된 [참고 문서]를 읽고, 실제 대학생이 할 법한 질문 {num_pairs}개와 그에 대한 AI 튜터의 답변을 생성하세요.
답변은 반드시 문서 내용만을 기반으로 하며, 친절한 해요체를 사용하세요. 여러 짧은 글이 있다면 각 핵심을 뽑아내세요.
{format_instructions}"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "[참고 문서]:\n{context}")])
    chain = prompt | llm | parser

    try:
        res = await call_with_retry(chain, {"context": context_text, "num_pairs": num_pairs, "format_instructions": parser.get_format_instructions()})
        return res.get("pairs", [])
    except Exception as e:
        logger.error(f"생성 중 오류: {e}")
        return []

async def validate_qa_batch_async(qa_pairs: list[dict], context_text: str) -> list[dict]:
    # 클로드 지적 1번 반영: 모델명 수정 및 temperature 제거
    llm = ChatAnthropic(model="claude-haiku-5-5", max_tokens=4096)
    parser = JsonOutputParser(pydantic_object=JudgeBatchResult)
    # 클로드 작은 수정 2번 반영: 검증 프롬프트 기준 복구
    system_prompt = """당신은 Q&A 데이터의 품질을 검증하는 엄격한 평가자입니다.
아래 [참고 문서]만을 근거로 각 Q&A를 평가하세요.
1. 답변이 참고 문서에 실제로 있는 사실에 기반하는가? (없는 내용을 지어냈다면 실패)
2. 질문과 답변이 논리적으로 맞는가?
3. 왜곡되거나 오해의 소지가 있는 정보가 없는가?
입력과 같은 순서와 개수로 평가 결과를 반환하세요.
{format_instructions}"""
    human_prompt = "[참고 문서]:\n{context}\n\n[평가할 Q&A 리스트]:\n{pairs}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain = prompt | llm | parser

    try:
        res = await call_with_retry(chain, {"context": context_text, "pairs": json.dumps(qa_pairs, ensure_ascii=False, indent=2), "format_instructions": parser.get_format_instructions()})
        return res.get("results", [])
    except Exception as e:
        logger.warning(f"검증 중 오류: {e}")
        return [None] * len(qa_pairs)

def get_source_category(file_name: str) -> str:
    low = file_name.lower()
    for cat, conf in SOURCES.items():
        if any(match_word in low for match_word in conf["match"]):
            return cat
    return "unknown"

def extract_and_format_record(record: dict) -> str:
    body = ""
    for key in ["content", "text", "body", "article", "review"]:
        if key in record and isinstance(record[key], str) and record[key].strip():
            body = record[key].strip()
            break
    if not body:
        body = "\n".join([str(v) for v in record.values() if isinstance(v, str) and len(str(v)) > 20])
    if not body.strip():
        return ""
    meta = []
    for k in META_KEYS:
        v = record.get(k)
        if isinstance(v, (str, int)) and v and str(v).strip() not in body:
            meta.append(f"{k}: {str(v).strip()}")
    
    head = f"[{' | '.join(meta)}]\n" if meta else ""
    return head + body

def iter_file_batches():
    files = sorted(DATA_DIR.glob("**/*.jsonl"))
    batches_by_cat = {k: [] for k in SOURCES.keys()}

    for fp in files:
        cat = get_source_category(fp.name)
        if cat == "unknown":
            continue

        conf = SOURCES[cat]
        buf = []
        
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    text = extract_and_format_record(record)
                    if len(text) < 10: continue

                    buf.append(text)
                    if len(buf) >= conf["group_size"]:
                        batches_by_cat[cat].append(("\n---\n".join(buf), fp.name, cat))
                        buf = []
                # 클로드 작은 수정 1번 반영: 구체적인 예외 잡기
                except json.JSONDecodeError:
                    continue
            
            if buf: 
                batches_by_cat[cat].append(("\n---\n".join(buf), fp.name, cat))

    for cat in batches_by_cat:
        random.Random(42).shuffle(batches_by_cat[cat])
    
    return batches_by_cat

# 클로드 수정 1번 반영: 상태 체크 헬퍼 함수
def source_done(cat: str, counts_dict: dict) -> bool:
    if TEST_MODE:
        return attempted[cat] >= 1
    return counts_dict[cat] >= SOURCES[cat]["target"]

async def process_batch(combined_text: str, file_name: str, cat: str, done_keys: set) -> None:
    doc_hash = hashlib.md5(combined_text.encode("utf-8")).hexdigest()
    record_key = f"{file_name}_batch::{doc_hash}"
    
    if not TEST_MODE and record_key in done_keys:
        return
        
    # 클로드 수정 2번 반영: 실행 직전에 한 번 더 목표치 초과 여부 확인
    if source_done(cat, passed_counts):
        return
    
    attempted[cat] += 1
    num_pairs = 1 if TEST_MODE else SOURCES[cat]["pairs_per_batch"]
    
    logger.info(f"🚀 [{cat}] API 요청 중... (목표: {passed_counts[cat]}/{SOURCES[cat]['target']})")
    
    pairs = await generate_synthetic_data_async(combined_text, num_pairs=num_pairs)
    if not pairs: return
    
    results = await validate_qa_batch_async(pairs, combined_text)
    if len(results) != len(pairs) or any(r is None for r in results):
        logger.warning(f"❌ [{cat}] 검증 실패 및 누락 - 묶음 스킵")
        return

    for pair, res in zip(pairs, results):
        instruction, output = (pair.get("instruction") or "").strip(), (pair.get("output") or "").strip()
        if not instruction or not output: continue

        is_pass = res.get("pass", res.get("is_pass", False))
        out_data = {"category": cat, "source": file_name, "instruction": instruction, "output": output}

        if is_pass:
            # 본 실행 중이면서 할당량을 다 채웠으면 파일에 안 씀
            if not TEST_MODE and passed_counts[cat] >= SOURCES[cat]["target"]:
                continue
                
            with open(VALIDATED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_data, ensure_ascii=False) + "\n")
            passed_counts[cat] += 1
            if TEST_MODE: logger.info(f"✅ [테스트 통과]: {instruction[:30]}...")
        else:
            out_data["reject_reason"] = res.get("reason", "알 수 없음")
            with open(REJECTED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_data, ensure_ascii=False) + "\n")

    if not TEST_MODE:
        with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
            f.write(record_key + "\n")
        done_keys.add(record_key)

async def main():
    global passed_counts
    done_keys = set()

    if not TEST_MODE:
        if os.path.exists(PROGRESS_PATH):
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                done_keys = set(f.read().splitlines())
        if VALIDATED_PATH.exists():
            with open(VALIDATED_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        cat = json.loads(line).get("category", "unknown")
                        if cat in passed_counts: passed_counts[cat] += 1
                    except json.JSONDecodeError:
                        pass

    logger.info(f"▶️ 모드: {'테스트(각 파일 1개씩)' if TEST_MODE else '본 실행(500개 목표)'} | 누적: {passed_counts}")

    # 클로드 수정 3번 반영: 묶음별 할당 로직 및 워커 기반 처리로 비용 폭탄 방지
    batches_by_cat = iter_file_batches()
    counts = {c: len(b) for c, b in batches_by_cat.items()}
    logger.info(f"📦 소스별 생성된 전체 묶음 수: {counts}")
    
    empty = [c for c, b in batches_by_cat.items() if not b]
    if empty:
        logger.error(f"🚨 묶음이 0개인 소스: {empty} → SOURCES의 match를 실제 파일명에 맞게 고치세요")
        return

    def interleave():
        queues = {c: iter(b) for c, b in batches_by_cat.items()}
        while queues:
            for cat in list(queues):
                if source_done(cat, passed_counts):
                    del queues[cat]
                    continue
                try:
                    text, fname, _ = next(queues[cat])
                except StopIteration:
                    del queues[cat]
                    continue
                yield text, fname, cat

    stream = interleave()

    async def worker():
        for text, fname, cat in stream:
            await process_batch(text, fname, cat, done_keys)

    n_workers = 1 if TEST_MODE else CONCURRENCY
    await asyncio.gather(*(worker() for _ in range(n_workers)))
    
    logger.info(f"🎉 종료! 최종 누적 개수: {passed_counts}")

if __name__ == "__main__":
    asyncio.run(main())