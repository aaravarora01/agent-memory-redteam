"""Serve Qwen2.5-Instruct on Modal via vLLM's OpenAI-compatible API.

Deploy:
    modal deploy scripts/modal_qwen25_vllm.py

After deploy, Modal prints a URL like:
    https://<workspace>--qwen25-memory-redteam-serve.modal.run

Use that URL with `/v1` in this repo's `.env`:
    QWEN_API_KEY=EMPTY
    QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
    QWEN_BASE_URL=https://<workspace>--qwen25-memory-redteam-serve.modal.run/v1
"""

from __future__ import annotations

import subprocess

import modal


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_PORT = 8000
MINUTES = 60


app = modal.App("qwen25-memory-redteam")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.6",
        "huggingface_hub[hf_transfer]>=0.24.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # vLLM V1 can route sampling through FlashInfer JIT kernels that
            # require nvcc in the runtime image. V0 avoids that cold-start
            # failure on Modal's slim Python image.
            "VLLM_USE_V1": "0",
        }
    )
)

hf_cache = modal.Volume.from_name("qwen25-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen25-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S:1",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "qwen2.5",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "auto",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
    ]

    print("Starting vLLM:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)
