"""Run finance GRPO attacker training on Modal.

Prereqs:
  1. Deploy the target/judge server: modal deploy scripts/modal_qwen25_vllm.py
  2. Keep .env at repo root with QWEN_BASE_URL, QWEN_API_KEY, and QWEN_MODEL.

Run:
    modal run scripts/modal_train_finance_grpo.py --smoke
    modal run scripts/modal_train_finance_grpo.py
    modal run scripts/modal_train_finance_grpo.py --beta0

Checkpoints and metrics persist on the `finance-grpo-outputs` Volume.
"""

from __future__ import annotations

import os
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINUTES = 60

app = modal.App("finance-grpo-train")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.4.0",
        "trl>=0.14.0",
        "transformers>=4.48.0",
        "datasets>=3.0.0",
        "peft>=0.13.0",
        "accelerate>=1.2.0",
        "vllm>=0.10.0",
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.1",
        "huggingface_hub[hf_transfer]>=0.24.0",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_dir(
        ROOT,
        "/root/repo",
        ignore=[
            ".git", "**/__pycache__", "**/*.pyc", "results", "outputs",
            "data/benign_memories.jsonl", "finance_poisoning/results",
            ".env", "*.png",
        ],
    )
)

hf_cache = modal.Volume.from_name("finance-grpo-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("finance-grpo-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=6 * 60 * MINUTES,
    secrets=[modal.Secret.from_dotenv()],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/outputs": outputs,
    },
)
def train(smoke: bool = False, beta0: bool = False) -> None:
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")

    run_name = "finance_grpo" + ("_smoke" if smoke else "") + ("_beta0" if beta0 else "")
    argv = [
        "train_finance_grpo.py",
        "--config", "configs/finance_grpo.yaml",
        "--output-dir", f"/outputs/{run_name}",
    ]
    if smoke:
        argv.append("--smoke")
    if beta0:
        argv.append("--beta0")

    sys.argv = argv
    from finance_poisoning.rl.train_finance_grpo import main

    rc = main()
    outputs.commit()
    if rc != 0:
        raise SystemExit(rc)


@app.local_entrypoint()
def cli(smoke: bool = False, beta0: bool = False) -> None:
    train.remote(smoke=smoke, beta0=beta0)
