"""eval_dataset.json의 질문에 대해 Ollama(Llama 3.1 8B)로 답변을 생성하는 스크립트.

사용법:
    # Base Model 답변만 생성
    python eval/generate_answers.py --model-name "Base Model"

    # Fine-tuned Model 답변 생성 (파인튜닝 후)
    python eval/generate_answers.py --model-name "Fine-tuned Model" --ollama-model "fine-tuned-llama:latest"

    # 커스텀 Ollama 모델 지정
    python eval/generate_answers.py --model-name "Base Model" --ollama-model "llama3.1:8b"
"""

import argparse
import json
import logging
import os
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """당신은 경북대학교 컴퓨터학부 학생을 돕는 AI 튜터입니다.
학생의 질문에 대해 한국어로 친절하고 정확하게 답변해주세요.
모르는 내용은 추측하지 말고, 알고 있는 범위 내에서만 답변하세요."""


def query_ollama(question: str, model: str = "llama3.1:8b") -> str:
    """Ollama API를 통해 모델에 질문하고 답변을 반환합니다."""
    payload = {
        "model": model,
        "prompt": question,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error("Ollama 서버에 연결할 수 없습니다. 'ollama serve'를 먼저 실행하세요.")
        raise
    except requests.exceptions.Timeout:
        logger.warning("Ollama 응답 타임아웃 (120초 초과)")
        return "[TIMEOUT] 응답 시간 초과"
    except Exception as e:
        logger.error(f"Ollama 요청 중 오류: {e}")
        return f"[ERROR] {e}"


def main():
    parser = argparse.ArgumentParser(description="Ollama로 eval_dataset.json 답변 생성")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Base Model",
        help="eval_dataset.json에 저장할 모델 이름 (기본값: 'Base Model')",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama에서 사용할 모델 태그 (기본값: 'llama3.1:8b')",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="중단된 지점부터 재개할 질문 인덱스 (0부터 시작)",
    )
    args = parser.parse_args()

    # 데이터셋 로드
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    logger.info(f"총 {total}개 질문, 모델: {args.model_name} ({args.ollama_model})")
    logger.info(f"시작 인덱스: {args.start_from}")

    # Ollama 서버 연결 확인
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        logger.info("✅ Ollama 서버 연결 확인 완료")
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ollama 서버가 실행 중이지 않습니다.")
        logger.error("   다른 터미널에서 'ollama serve'를 실행하세요.")
        return

    # 답변 생성
    start_time = time.time()
    for idx in range(args.start_from, total):
        item = dataset[idx]
        question = item["question"]

        logger.info(f"[{idx + 1}/{total}] 질문: {question[:50]}...")

        answer = query_ollama(question, model=args.ollama_model)

        # 답변 저장
        item["models"][args.model_name] = answer

        # 매 질문마다 중간 저장 (안전)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        # 진행상황
        elapsed = time.time() - start_time
        avg_per_q = elapsed / (idx - args.start_from + 1)
        remaining = avg_per_q * (total - idx - 1)
        logger.info(
            f"  → 답변 길이: {len(answer)}자 | "
            f"경과: {elapsed:.0f}초 | "
            f"예상 남은 시간: {remaining:.0f}초"
        )

    total_time = time.time() - start_time
    logger.info(f"\n✅ 완료! 총 {total}개 답변 생성, 소요 시간: {total_time:.0f}초")
    logger.info(f"결과 저장: {dataset_path}")


if __name__ == "__main__":
    main()
