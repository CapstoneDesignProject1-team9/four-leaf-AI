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
MODE_KEYS = {
    "base": "Base Model",
    "rag": "Base Model + RAG",
    "student_lora": "Base Model + Student LoRA",
    "professor_lora": "Base Model + Professor LoRA",
    "student_lora_rag": "Base Model + Student LoRA + RAG",
}


async def _post(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def generate_one(session: aiohttp.ClientSession, question: str, mode: str) -> dict:
    if mode in {"base", "student_lora", "professor_lora"}:
        if mode == "base":
            model_name = os.getenv("VLLM_BASE_MODEL", "base")
        elif mode == "student_lora":
            model_name = os.getenv("VLLM_STUDENT_LORA_MODEL_NAME") or os.getenv("VLLM_LORA_MODEL_NAME")
        else:
            model_name = os.getenv("VLLM_PROFESSOR_LORA_MODEL_NAME")
        if mode != "base" and not model_name:
            raise RuntimeError(f"Set the model name environment variable for {mode} before generation")
        data = await _post(session, VLLM_API_URL, {"question": question, "model_name": model_name})
        return {"answer": data.get("response", "").strip(), "contexts": data.get("contexts", [])}
    if mode == "rag":
        model_name = os.getenv("VLLM_BASE_MODEL", "base")
    else:
        model_name = os.getenv("VLLM_STUDENT_LORA_MODEL_NAME") or os.getenv("VLLM_LORA_MODEL_NAME")
        if not model_name:
            raise RuntimeError("Set VLLM_STUDENT_LORA_MODEL_NAME for student_lora_rag generation")
    data = await _post(session, RAG_API_URL, {"message": question, "model_name": model_name})
    return {"answer": data.get("answer", "").strip(), "contexts": [s.get("content", "") for s in data.get("sources", [])]}


async def main_async(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path
    if not dataset_path.exists():
        raise SystemExit(f"Evaluation dataset not found: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    modes = list(MODE_KEYS) if args.mode == "all" else [args.mode]
    missing = []
    if any(mode in {"student_lora", "student_lora_rag"} for mode in modes):
        if not (os.getenv("VLLM_STUDENT_LORA_MODEL_NAME") or os.getenv("VLLM_LORA_MODEL_NAME")):
            missing.append("VLLM_STUDENT_LORA_MODEL_NAME (or legacy VLLM_LORA_MODEL_NAME)")
    if "professor_lora" in modes and not os.getenv("VLLM_PROFESSOR_LORA_MODEL_NAME"):
        missing.append("VLLM_PROFESSOR_LORA_MODEL_NAME")
    if missing:
        raise SystemExit("Missing required model configuration: " + ", ".join(missing))
    timeout = aiohttp.ClientTimeout(total=180)
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx in range(args.start_from, len(dataset)):
            item = dataset[idx]
            item.setdefault("models", {})
            item.setdefault("contexts", {})
            item.setdefault("generation_latency_s", {})
            item.setdefault("served_model_names", {})

            async def run(mode: str):
                async with semaphore:
                    started = time.perf_counter()
                    try:
                        result = await generate_one(session, item["question"], mode)
                        return mode, result, time.perf_counter() - started
                    except Exception as exc:
                        logger.exception("%s generation failed for question %s", mode, idx + 1)
                        return mode, {"answer": f"[ERROR] {exc}", "contexts": []}, time.perf_counter() - started

            for mode, result, elapsed in await asyncio.gather(*(run(mode) for mode in modes)):
                model_key = MODE_KEYS[mode]
                item["models"][model_key] = result["answer"]
                item["contexts"][model_key] = result["contexts"]
                item["generation_latency_s"][model_key] = round(elapsed, 4)
                if mode in {"base", "rag"}:
                    served_name = os.getenv("VLLM_BASE_MODEL", "base")
                elif mode == "professor_lora":
                    served_name = os.getenv("VLLM_PROFESSOR_LORA_MODEL_NAME", "")
                else:
                    served_name = os.getenv("VLLM_STUDENT_LORA_MODEL_NAME") or os.getenv("VLLM_LORA_MODEL_NAME", "")
                item["served_model_names"][model_key] = served_name
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[%d/%d] saved (elapsed %.1fs)", idx + 1, len(dataset), time.monotonic() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers asynchronously.")
    parser.add_argument("--dataset", default="student_eval_dataset.json", help="Dataset filename or path (default: student_eval_dataset.json)")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--mode", choices=["all", *MODE_KEYS], default="all")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("GENERATION_CONCURRENCY", "4")))
    args = parser.parse_args()
    if args.start_from < 0 or args.concurrency < 1:
        parser.error("--start-from must be >= 0 and --concurrency must be >= 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
