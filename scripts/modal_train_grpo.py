"""Run the PISmith GRPO attacker trainer on Modal.

Prereqs:
  1. Deploy the target/judge server:  modal deploy scripts/modal_qwen25_vllm.py
  2. Have a working .env at repo root (QWEN_BASE_URL -> the served /v1 URL,
     QWEN_API_KEY=EMPTY, QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct). It is injected
     into the training container via modal.Secret.from_dotenv().

Run:
    modal run scripts/modal_train_grpo.py --smoke     # ~10 min validation
    modal run scripts/modal_train_grpo.py             # full run
    modal run scripts/modal_train_grpo.py --beta0     # KL=0 ablation

Checkpoints + the embedded corpus persist on the `pismith-outputs` Volume.
"""

from __future__ import annotations

import os
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINUTES = 60

app = modal.App("pismith-grpo-train")

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
    # Mount the repo last; skip caches/artifacts/secrets.
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

hf_cache = modal.Volume.from_name("pismith-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("pismith-outputs", create_if_missing=True)


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

    run_name = "pismith_grpo" + ("_smoke" if smoke else "") + ("_beta0" if beta0 else "")
    argv = [
        "train_pismith_grpo.py",
        "--config", "configs/pismith_grpo.yaml",
        # Persist checkpoints + embedded corpus on the Volume.
        "--output-dir", f"/outputs/{run_name}",
        "--corpus-out", "/outputs/benign_memories.jsonl",
    ]
    if smoke:
        argv.append("--smoke")
    if beta0:
        argv.append("--beta0")

    sys.argv = argv
    from rl.train_pismith_grpo import main

    rc = main()
    outputs.commit()  # flush checkpoints to the Volume
    if rc != 0:
        raise SystemExit(rc)


@app.local_entrypoint()
def cli(smoke: bool = False, beta0: bool = False) -> None:
    train.remote(smoke=smoke, beta0=beta0)
