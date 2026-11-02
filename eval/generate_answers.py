"""Generate evaluation answers through vLLM APIs or local Transformers/PEFT."""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
VLLM_CHAT_COMPLETIONS_URL = os.getenv(
    "VLLM_CHAT_COMPLETIONS_URL", "http://localhost:8001/v1/chat/completions"
)
VLLM_MODELS_URL = os.getenv("VLLM_MODELS_URL", "http://localhost:8001/v1/models")
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1/tutor/chat")
MODE_KEYS = {
    "base": "Base Model",
    "rag": "Base Model + RAG",
    "student_lora": "Base Model + Student LoRA",
    "professor_lora": "Base Model + Professor LoRA",
    "student_lora_rag": "Base Model + Student LoRA + RAG",
}
LOCAL_MODES = ("base", "student_lora", "professor_lora")


def build_generation_question(item: dict) -> str:
    """Include the professor scenario's student-question bundle in the model prompt."""
    question = str(item.get("question", ""))
    student_questions = item.get("student_questions") or []
    if student_questions:
        question += "\n\n[제공된 학생 질문 묶음]\n" + "\n".join(
            f"- {student_question}" for student_question in student_questions
        )
    return question


class LocalTransformersBackend:
    """Load one base model and switch between base and two local LoRA adapters."""

    def __init__(self, load_adapters: bool = True) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.getenv("MODEL_DIR")
        if not model_path:
            hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
            snapshots = hf_home / "hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
            candidates = sorted(
                (p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime
            )
            if not candidates:
                raise RuntimeError(
                    f"No cached Llama snapshot found under {snapshots}; set MODEL_DIR"
                )
            model_path = str(candidates[-1])
        if not Path(model_path).is_dir():
            raise RuntimeError(f"MODEL_DIR is not a local model directory: {model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Local Transformers backend requires a CUDA GPU")

        self.torch = torch
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        if load_adapters:
            student_path = os.getenv("STUDENT_LORA_PATH", "adapter_student_llama")
            professor_path = os.getenv("PROFESSOR_LORA_PATH", "adapter_professor_llama")
            if not Path(student_path).is_dir() or not Path(professor_path).is_dir():
                raise RuntimeError(
                    "LoRA folders were not found. Set STUDENT_LORA_PATH and PROFESSOR_LORA_PATH "
                    "or run from the four-leaf-AI directory."
                )
            self.model = PeftModel.from_pretrained(
                base, student_path, adapter_name="student", is_trainable=False
            )
            self.model.load_adapter(professor_path, adapter_name="professor", is_trainable=False)
        else:
            self.model = base
        self.model.eval()
        # Greedy decoding is deterministic; remove sampling settings inherited from the model config.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

    def generate_prompt(self, text: str, mode: str = "base") -> str:
        is_peft_model = hasattr(self.model, "peft_config")
        if mode == "student_lora":
            if not is_peft_model:
                raise RuntimeError("LoRA adapters were not loaded")
            self.model.set_adapter("student")
        elif mode == "professor_lora":
            if not is_peft_model:
                raise RuntimeError("LoRA adapters were not loaded")
            self.model.set_adapter("professor")

        encoded = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "당신은 경북대학교 컴퓨터학부 학생을 돕는 AI 튜터입니다. 한국어로 질문에 직접 답하고, 모르는 내용은 추측하지 마세요.",
                },
                {"role": "user", "content": text},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = self.model.get_input_embeddings().weight.device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with self.torch.inference_mode():
            if mode == "base" and is_peft_model:
                with self.model.disable_adapter():
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(os.getenv("LOCAL_MAX_NEW_TOKENS", "768")),
                        do_sample=False,
                    )
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(os.getenv("LOCAL_MAX_NEW_TOKENS", "768")),
                    do_sample=False,
                )
        return self.tokenizer.decode(
            output[0][input_ids.shape[-1] :], skip_special_tokens=True
        ).strip()

    def generate(self, question: str, mode: str) -> dict:
        answer = self.generate_prompt(question, mode)
        return {"answer": answer, "contexts": []}


class TransformersPromptAdapter:
    """Use a local Transformers model behind the repository's existing RAG chain."""

    def __init__(self, backend: LocalTransformersBackend) -> None:
        self.backend = backend

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self.backend.generate_prompt(str(prompt), mode="base"))


class LocalRAGBackend:
    """Run the repository's Chroma/BGE-M3 RAG chain directly with local Ollama."""

    def __init__(self) -> None:
        from app.chains.rag_chain import build_rag_chain

        self.chain = build_rag_chain()
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, question: str) -> dict:
        result = self.chain.invoke({"query": question})
        answer = str(result.get("result", "")).strip()
        contexts = [doc.page_content for doc in result.get("source_documents", [])]
        return {"answer": answer, "contexts": contexts}


async def _post(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def _generate_vllm(session: aiohttp.ClientSession, question: str, model_name: str) -> dict:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "당신은 경북대학교 컴퓨터학부 학생을 돕는 AI 튜터입니다. 한국어로 질문에 직접 답하고, 모르는 내용은 추측하지 마세요.",
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "768")),
    }
    data = await _post(session, VLLM_CHAT_COMPLETIONS_URL, payload)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM response has no choices: {data}")
    return {"answer": choices[0].get("message", {}).get("content", "").strip(), "contexts": []}


async def _wait_for_models(session: aiohttp.ClientSession, modes: list[str]) -> None:
    expected = set()
    if any(mode in {"base", "rag"} for mode in modes):
        expected.add(os.getenv("VLLM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    if any(mode in {"student_lora", "student_lora_rag"} for mode in modes):
        expected.add(os.getenv("VLLM_STUDENT_LORA_MODEL_NAME", "student-lora"))
    if "professor_lora" in modes:
        expected.add(os.getenv("VLLM_PROFESSOR_LORA_MODEL_NAME", "professor-lora"))

    deadline = time.monotonic() + 180
    last_error = "vLLM has not returned a model list"
    while time.monotonic() < deadline:
        try:
            async with session.get(VLLM_MODELS_URL) as response:
                response.raise_for_status()
                data = await response.json()
            available = {entry.get("id") for entry in data.get("data", [])}
            missing = expected - available
            if not missing:
                return
            last_error = "registered models missing: " + ", ".join(sorted(missing))
        except Exception as exc:
            last_error = str(exc)
        logger.info("Waiting for configured vLLM models: %s", last_error)
        await asyncio.sleep(5)
    raise SystemExit(
        f"vLLM is not ready at {VLLM_MODELS_URL}: {last_error}. Check the base model and adapter folders."
    )


async def generate_one(session: aiohttp.ClientSession, item: dict, mode: str) -> dict:
    question = build_generation_question(item)
    if mode in {"base", "student_lora", "professor_lora"}:
        if mode == "base":
            model_name = os.getenv("VLLM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        elif mode == "student_lora":
            model_name = os.getenv("VLLM_STUDENT_LORA_MODEL_NAME", "student-lora")
        else:
            model_name = os.getenv("VLLM_PROFESSOR_LORA_MODEL_NAME", "professor-lora")
        if mode != "base" and not model_name:
            raise RuntimeError(
                f"Set the model name environment variable for {mode} before generation"
            )
        return await _generate_vllm(session, question, model_name)
    if mode == "rag":
        model_name = os.getenv("VLLM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    else:
        model_name = os.getenv("VLLM_STUDENT_LORA_MODEL_NAME", "student-lora")
        if not model_name:
            raise RuntimeError("Set VLLM_STUDENT_LORA_MODEL_NAME for student_lora_rag generation")
    data = await _post(session, RAG_API_URL, {"message": question, "model_name": model_name})
    return {
        "answer": data.get("answer", "").strip(),
        "contexts": [s.get("content", "") for s in data.get("sources", [])],
    }


async def main_async(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        from_cwd = Path.cwd() / dataset_path
        from_eval_dir = Path(__file__).parent / dataset_path
        dataset_path = from_cwd if from_cwd.exists() else from_eval_dir
    if not dataset_path.exists():
        raise SystemExit(f"Evaluation dataset not found: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.backend == "transformers-rag":
        if args.mode not in {"all", "rag"}:
            raise SystemExit("The transformers-rag backend supports only --mode rag")
        from app.chains.rag_chain import NoticeRagChain, init_vectorstore

        await init_vectorstore()
        backend = LocalTransformersBackend(load_adapters=False)
        rag_chain = NoticeRagChain(TransformersPromptAdapter(backend))
        start = time.monotonic()
        for idx, item in enumerate(dataset[args.start_from :], start=args.start_from):
            item.setdefault("models", {})
            item.setdefault("contexts", {})
            item.setdefault("generation_latency_s", {})
            item.setdefault("served_model_names", {})
            started = time.perf_counter()
            result = rag_chain.invoke({"query": str(item["question"]).strip()})
            elapsed = time.perf_counter() - started
            model_key = MODE_KEYS["rag"]
            item["models"][model_key] = result["result"]
            item["contexts"][model_key] = [
                doc.page_content for doc in result.get("source_documents", [])
            ]
            item["generation_latency_s"][model_key] = round(elapsed, 4)
            item["served_model_names"][model_key] = f"{backend.model_path} + Chroma/BGE-M3"
            dataset_path.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "[%d/%d] saved (elapsed %.1fs)", idx + 1, len(dataset), time.monotonic() - start
            )
        return

    if args.backend == "rag-local":
        if args.mode not in {"all", "rag"}:
            raise SystemExit("The rag-local backend supports only --mode rag")
        from app.chains.rag_chain import init_vectorstore

        await init_vectorstore()
        backend = LocalRAGBackend()
        start = time.monotonic()
        for idx, item in enumerate(dataset[args.start_from :], start=args.start_from):
            item.setdefault("models", {})
            item.setdefault("contexts", {})
            item.setdefault("generation_latency_s", {})
            item.setdefault("served_model_names", {})
            started = time.perf_counter()
            result = backend.generate(build_generation_question(item))
            elapsed = time.perf_counter() - started
            model_key = MODE_KEYS["rag"]
            item["models"][model_key] = result["answer"]
            item["contexts"][model_key] = result["contexts"]
            item["generation_latency_s"][model_key] = round(elapsed, 4)
            item["served_model_names"][model_key] = (
                f"ollama:{backend.model_name} + local Chroma/BGE-M3"
            )
            dataset_path.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "[%d/%d] saved (elapsed %.1fs)", idx + 1, len(dataset), time.monotonic() - start
            )
        return

    if args.backend == "transformers":
        if args.mode == "all":
            modes = list(LOCAL_MODES)
        else:
            modes = [args.mode]
        rag_modes = [mode for mode in modes if mode not in LOCAL_MODES]
        if rag_modes:
            raise SystemExit(
                "The local Transformers backend supports base/student_lora/professor_lora only; RAG needs the AI API."
            )
        backend = LocalTransformersBackend()
        start = time.monotonic()
        for idx in range(args.start_from, len(dataset)):
            item = dataset[idx]
            item.setdefault("models", {})
            # Remove unused legacy slots; canonical condition names below are authoritative.
            for legacy_key in ("Student LoRA", "Professor LoRA"):
                if not item["models"].get(legacy_key):
                    item["models"].pop(legacy_key, None)
            item.setdefault("contexts", {})
            item.setdefault("generation_latency_s", {})
            item.setdefault("served_model_names", {})
            for mode in modes:
                model_key = MODE_KEYS[mode]
                started = time.perf_counter()
                result = backend.generate(build_generation_question(item), mode)
                elapsed = time.perf_counter() - started
                item["models"][model_key] = result["answer"]
                item["contexts"][model_key] = []
                item["generation_latency_s"][model_key] = round(elapsed, 4)
                if mode == "base":
                    served_name = backend.model_path
                elif mode == "student_lora":
                    served_name = f"local:{os.getenv('STUDENT_LORA_PATH', 'adapter_student_llama')}"
                else:
                    served_name = (
                        f"local:{os.getenv('PROFESSOR_LORA_PATH', 'adapter_professor_llama')}"
                    )
                item["served_model_names"][model_key] = served_name
            dataset_path.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "[%d/%d] saved (elapsed %.1fs)", idx + 1, len(dataset), time.monotonic() - start
            )
        return

    modes = list(MODE_KEYS) if args.mode == "all" else [args.mode]
    missing = []
    if any(mode in {"student_lora", "student_lora_rag"} for mode in modes):
        if not os.getenv("VLLM_STUDENT_LORA_MODEL_NAME", "student-lora"):
            missing.append("VLLM_STUDENT_LORA_MODEL_NAME")
    if "professor_lora" in modes and not os.getenv(
        "VLLM_PROFESSOR_LORA_MODEL_NAME", "professor-lora"
    ):
        missing.append("VLLM_PROFESSOR_LORA_MODEL_NAME")
    if missing:
        raise SystemExit("Missing required model configuration: " + ", ".join(missing))
    timeout = aiohttp.ClientTimeout(total=180)
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _wait_for_models(session, modes)
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
                        result = await generate_one(session, item, mode)
                        return mode, result, time.perf_counter() - started
                    except Exception as exc:
                        logger.exception("%s generation failed for question %s", mode, idx + 1)
                        return (
                            mode,
                            {"answer": f"[ERROR] {exc}", "contexts": []},
                            time.perf_counter() - started,
                        )

            for mode, result, elapsed in await asyncio.gather(*(run(mode) for mode in modes)):
                model_key = MODE_KEYS[mode]
                item["models"][model_key] = result["answer"]
                item["contexts"][model_key] = result["contexts"]
                item["generation_latency_s"][model_key] = round(elapsed, 4)
                if mode in {"base", "rag"}:
                    served_name = os.getenv("VLLM_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
                elif mode == "professor_lora":
                    served_name = os.getenv("VLLM_PROFESSOR_LORA_MODEL_NAME", "professor-lora")
                else:
                    served_name = os.getenv("VLLM_STUDENT_LORA_MODEL_NAME", "student-lora")
                item["served_model_names"][model_key] = served_name
            dataset_path.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "[%d/%d] saved (elapsed %.1fs)", idx + 1, len(dataset), time.monotonic() - start
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers asynchronously.")
    parser.add_argument(
        "--dataset",
        default="student_eval_dataset.json",
        help="Dataset filename or path (default: student_eval_dataset.json)",
    )
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--mode", choices=["all", *MODE_KEYS], default="all")
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers", "rag-local", "transformers-rag"],
        default=os.getenv("EVAL_GENERATION_BACKEND", "vllm"),
    )
    parser.add_argument(
        "--concurrency", type=int, default=int(os.getenv("GENERATION_CONCURRENCY", "4"))
    )
    args = parser.parse_args()
    if args.start_from < 0 or args.concurrency < 1:
        parser.error("--start-from must be >= 0 and --concurrency must be >= 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
