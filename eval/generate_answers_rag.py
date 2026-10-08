# eval/generate_answers_rag.py
"""
RAG 백엔드 (uvicorn) 로 질문을 보내고, 결과를 eval_dataset.json 의
"Base Model + RAG" 필드에 채워 저장하는 스크립트.
"""

import json
import os
import time
import logging
import argparse
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/qa"   # FastAPI 엔드포인트

def query_rag(question: str) -> str:
    payload = {"question": question}
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # API가 반환하는 키가 "answer" 라는 가정
        return data.get("answer", "").strip()
    except Exception as e:
        logger.error(f"RAG 요청 실패: {e}")
        return f"[ERROR] {e}"

def main():
    parser = argparse.ArgumentParser(
        description="RAG 답변을 eval_dataset.json 에 채워넣기")
    parser.add_argument(
        "--model-name",
        default="Base Model + RAG",
        help="Dataset에 저장할 모델 키 (기본값: 'Base Model + RAG')"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="중단된 인덱스부터 재시작 (0 기반)"
    )
    args = parser.parse_args()

    dataset_path = os.path.join(
        os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset 파일이 없어요: {dataset_path}")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    logger.info(f"총 {total}개 질문, 모델 키: {args.model_name}")
    start = args.start_from

    start_time = time.time()
    for idx in range(start, total):
        item = dataset[idx]
        question = item["question"]
        logger.info(f"[{idx+1}/{total}] 질문: {question[:50]}...")

        answer = query_rag(question)
        item["models"][args.model_name] = answer

        # 매 iteration 마다 파일에 중간 저장 (예기치 않은 중단 대비)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        # 진행 로깅 (시간 예측)
        elapsed = time.time() - start_time
        avg = elapsed / (idx - start + 1)
        remaining = avg * (total - idx - 1)
        logger.info(
            f"  → 답변 길이: {len(answer)}자 | 경과: {elapsed:.0f}s | "
            f"예상 남은 시간: {remaining:.0f}s"
        )

    logger.info("✅ RAG 답변 채우기 완료")
    logger.info(f"파일 저장 위치: {dataset_path}")

if __name__ == "__main__":
    main()
