"""Evaluate a finance GRPO checkpoint on Modal."""

from __future__ import annotations

import os
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINUTES = 60

app = modal.App("finance-grpo-eval")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "peft>=0.13.0",
        "accelerate>=1.2.0",
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.1",
        "huggingface_hub[hf_transfer]>=0.24.0",
        "scikit-learn>=1.4.0",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
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
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_dotenv()],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/outputs": outputs,
    },
)
def evaluate(
    checkpoint: str,
    n: int = 5,
    stage: str | None = None,
    success_signal: str = "judge",
    reward_mode: str = "sparse",
    temperature: float = 0.7,
) -> None:
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")

    checkpoint_path = checkpoint
    if not checkpoint_path.startswith("/"):
        checkpoint_path = f"/outputs/{checkpoint_path}"
    safe_name = checkpoint.strip("/").replace("/", "_")
    argv = [
        "eval_finance_grpo.py",
        "--model", checkpoint_path,
        "--config", "configs/finance_grpo.yaml",
        "--n", str(n),
        "--success-signal", success_signal,
        "--reward-mode", reward_mode,
        "--temperature", str(temperature),
        "--out", f"/outputs/evals/{safe_name}_eval.jsonl",
        "--summary", f"/outputs/evals/{safe_name}_eval_summary.json",
    ]
    if stage:
        argv.extend(["--stage", stage])

    sys.argv = argv
    from finance_poisoning.rl.eval_finance_grpo import main

    rc = main()
    outputs.commit()
    if rc != 0:
        raise SystemExit(rc)


@app.local_entrypoint()
def cli(
    checkpoint: str = "finance_grpo_stage1_scorer_shaped/checkpoint-200",
    n: int = 5,
    stage: str | None = None,
    success_signal: str = "judge",
    reward_mode: str = "sparse",
    temperature: float = 0.7,
) -> None:
    evaluate.remote(
        checkpoint=checkpoint,
        n=n,
        stage=stage,
        success_signal=success_signal,
        reward_mode=reward_mode,
        temperature=temperature,
    )
