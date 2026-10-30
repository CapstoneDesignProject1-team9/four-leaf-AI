"""Asynchronously generate evaluation answers through vLLM and the AI API."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
VLLM_API_URL = os.getenv("VLLM_GENERATE_URL", "http://localhost:8001/generate")
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1/tutor/chat")
MODE_KEYS = {"base": "Base Model", "rag": "Base Model + RAG", "lora": "Fine-tuned Model (LoRA)", "rag_lora": "Fine-tuned Model (LoRA) + RAG"}


async def _post(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def generate_one(session: aiohttp.ClientSession, question: str, mode: str) -> dict:
    if mode in {"base", "lora"}:
        model_name = os.getenv("VLLM_BASE_MODEL", "base") if mode == "base" else os.getenv("VLLM_LORA_MODEL_NAME", "lora")
        data = await _post(session, VLLM_API_URL, {"question": question, "model_name": model_name})
        return {"answer": data.get("response", "").strip(), "contexts": data.get("contexts", [])}
    model_name = os.getenv("VLLM_BASE_MODEL", "base") if mode == "rag" else os.getenv("VLLM_LORA_MODEL_NAME", "lora")
    data = await _post(session, RAG_API_URL, {"message": question, "model_name": model_name})
    return {"answer": data.get("answer", "").strip(), "contexts": [s.get("content", "") for s in data.get("sources", [])]}


async def main_async(args: argparse.Namespace) -> None:
    dataset_path = Path(__file__).with_name("eval_dataset.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    modes = list(MODE_KEYS) if args.mode == "all" else [args.mode]
    timeout = aiohttp.ClientTimeout(total=180)
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx in range(args.start_from, len(dataset)):
            item = dataset[idx]
            item.setdefault("models", {})
            item.setdefault("contexts", {})

            async def run(mode: str):
                async with semaphore:
                    try:
                        return mode, await generate_one(session, item["question"], mode)
                    except Exception as exc:
                        logger.exception("%s generation failed for question %s", mode, idx + 1)
                        return mode, {"answer": f"[ERROR] {exc}", "contexts": []}

            for mode, result in await asyncio.gather(*(run(mode) for mode in modes)):
                item["models"][MODE_KEYS[mode]] = result["answer"]
                item["contexts"][MODE_KEYS[mode]] = result["contexts"]
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[%d/%d] saved (elapsed %.1fs)", idx + 1, len(dataset), time.monotonic() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers asynchronously.")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--mode", choices=["all", *MODE_KEYS], default="all")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("GENERATION_CONCURRENCY", "4")))
    args = parser.parse_args()
    if args.start_from < 0 or args.concurrency < 1:
        parser.error("--start-from must be >= 0 and --concurrency must be >= 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
