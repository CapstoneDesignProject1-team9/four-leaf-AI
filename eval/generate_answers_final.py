"""Generate answers for eval_dataset.json.

Usage:
    # Generate *all* four variants (default)
    python -m four_leaf_AI.eval.generate_answers --start-from 0

    # Generate only a specific variant
    python -m four_leaf_AI.eval.generate_answers --mode base
    python -m four_leaf_AI.eval.generate_answers --mode rag
    python -m four_leaf_AI.eval.generate_answers --mode lora
    python -m four_leaf_AI.eval.generate_answers --mode rag_lora
"""

import argparse
import json
import logging
import os
import time
from typing import Callable, Dict

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------- #
#   Model‑specific query APIs #
# --------------------------- #
OLLAMA_API_URL = "http://localhost:11434/api/generate"
SYSTEM_PROMPT = """당신은 경북대학교 컴퓨터학부 학생을 돕는 AI 튜터입니다.
학생의 질문에 대해 한국어로 친절하고 정확하게 답변해주세요.
모르는 내용은 추측하지 말고, 알고 있는 범위 내에서만 답변하세요."""

def query_ollama(question: str, model: str = "llama3.1:8b") -> str:
    """Call Ollama (plain or LoRA‑fine‑tuned) and return the answer."""
    payload = {
        "model": model,
        "prompt": question,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 512},
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama request failed ({model}): {e}")
        return f"[ERROR] {e}"


RAG_API_URL = "http://localhost:8000/qa"   # FastAPI 엔드포인트

def query_rag(question: str) -> str:
    """Send the question to the RAG service and return the answer."""
    try:
        resp = requests.post(RAG_API_URL, json={"question": question}, timeout=120)
        resp.raise_for_status()
        return resp.json().get("answer", "").strip()
    except Exception as e:
        logger.error(f"RAG request failed: {e}")
        return f"[ERROR] {e}"


# --------------------- #
#   Generation runner   #
# --------------------- #
def generate_one(
    question: str,
    mode: str,
) -> str:
    """Dispatch to the correct backend based on `mode`."""
    if mode == "base":
        return query_ollama(question, model="llama3.1:8b")
    if mode == "rag":
        return query_rag(question)
    if mode == "lora":
        # LoRA‑fine‑tuned model tag – adjust if your tag differs
        return query_ollama(question, model="llama3.1:8b‑lora")
    if mode == "rag_lora":
        # Simple approach: use RAG answer as is.
        # If you need a two‑step pipeline (RAG → LoRA), replace the line below.
        return query_rag(question)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate answers for eval_dataset.json (four configs)."
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Resume index (0‑based).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "base", "rag", "lora", "rag_lora"],
        default="all",
        help="Which configuration(s) to generate. 'all' runs the four variants.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    logger.info(f"🧮 {total} questions – mode={args.mode}")

    # -----------------------------------------------------------------
    # Iterate
    # -----------------------------------------------------------------
    start_time = time.time()
    for idx in range(args.start_from, total):
        item = dataset[idx]
        question = item["question"]
        logger.info(f"[{idx+1}/{total}] 질문: {question[:50]}…")

        # Ensure the `models` dict exists
        if "models" not in item or not isinstance(item["models"], dict):
            item["models"] = {}

        modes_to_run = (
            ["base", "rag", "lora", "rag_lora"]
            if args.mode == "all"
            else [args.mode]
        )

        for mode in modes_to_run:
            answer = generate_one(question, mode)
            # Human‑readable key that matches `evaluate.py` expectations
            key_name = {
                "base": "Base",
                "rag": "Base + RAG",
                "lora": "Base + LoRA",
                "rag_lora": "Base + RAG + LoRA",
            }[mode]
            item["models"][key_name] = answer

        # ---------------------------------------------------------
        # Periodic checkpoint (protect against crashes)
        # ---------------------------------------------------------
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        # Simple ETA logging
        elapsed = time.time() - start_time
        avg = elapsed / (idx - args.start_from + 1)
        remaining = avg * (total - idx - 1)
        logger.info(
            f"  → elapsed {elapsed:.0f}s, ETA {remaining:.0f}s"
        )

    logger.info(
        f"\n✅ Finished! Generated answers for {total} items (mode={args.mode})"
    )
    logger.info(f"Dataset saved to {dataset_path}")


if __name__ == "__main__":
    main()