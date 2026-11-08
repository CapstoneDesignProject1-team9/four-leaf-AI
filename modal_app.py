"""Serve the Llama 3.1 base model and the student LoRA on Modal with vLLM.

The API is OpenAI-compatible. Pass ``base-model`` or ``student-lora`` as the
request's ``model`` value.
"""

import subprocess
import time

import modal

APP_NAME = "four-leaf-llama"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PORT = 8000
ADAPTERS_PATH = "/models/adapters"
HF_CACHE_PATH = "/root/.cache/huggingface"

app = modal.App(APP_NAME)

# Modal Volumes keep uploads and the gated base model cache out of image layers.
adapter_volume = modal.Volume.from_name("four-leaf-adapters", create_if_missing=True, version=2)
# Reuse the workspace's existing HF cache volume rather than duplicating it.
hf_cache_volume = modal.Volume.from_name("four-leaf-model-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_HOME": HF_CACHE_PATH})
)


@app.server(
    image=vllm_image,
    # Prefer the lower-cost L4; fall back to A10 when L4 capacity is unavailable.
    gpu=["L4", "A10"],
    cpu=4,
    memory=24576,
    startup_timeout=1200,
    scaledown_window=300,
    min_containers=0,
    max_containers=1,
    target_concurrency=1,
    port=PORT,
    volumes={
        ADAPTERS_PATH: adapter_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LlamaServer:
    """Run an authenticated, scale-to-zero OpenAI-compatible vLLM server."""

    @modal.enter()
    def start(self):
        import os

        # Support either common Hugging Face token variable name in the existing
        # Modal secret without printing or exposing its value.
        if "HF_TOKEN" not in os.environ and os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError(
                "Modal secret 'huggingface-secret' must contain HF_TOKEN or HUGGING_FACE_HUB_TOKEN"
            )

        command = [
            "vllm",
            "serve",
            BASE_MODEL,
            "--revision",
            BASE_MODEL_REVISION,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--served-model-name",
            "base-model",
            "--enable-lora",
            "--max-loras",
            "1",
            "--max-lora-rank",
            "16",
            "--lora-modules",
            f"student-lora={ADAPTERS_PATH}/adapter_student_llama",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.90",
            "--dtype",
            "half",
        ]
        self.process = subprocess.Popen(command)

        # Modal considers the server ready when the port accepts connections.
        # Wait for vLLM's model initialization so failed loads exit clearly.
        import httpx

        deadline = time.monotonic() + 1100
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup (code {self.process.returncode})")
            try:
                response = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if response.is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError("vLLM did not become healthy before the startup deadline")

    @modal.exit()
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        hf_cache_volume.commit()
