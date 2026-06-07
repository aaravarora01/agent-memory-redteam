"""Run finance GRPO attacker training on Modal.

Prereqs:
  1. Deploy the target/judge server: modal deploy scripts/modal_qwen25_vllm.py
  2. Keep .env at repo root with QWEN_BASE_URL, QWEN_API_KEY, and QWEN_MODEL.

Run:
    modal run scripts/modal_train_finance_grpo.py --smoke
    modal run scripts/modal_train_finance_grpo.py --stage stage1
    modal run scripts/modal_train_finance_grpo.py --stage stage1 --max-steps 2 --run-suffix preflight
    modal run scripts/modal_train_finance_grpo.py --stage stage2 --resume-from-checkpoint /outputs/finance_grpo_stage1_scorer_shaped/checkpoint-250
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
def train(
    smoke: bool = False,
    beta0: bool = False,
    stage: str | None = None,
    resume_from_checkpoint: str | None = None,
    max_steps: int | None = None,
    run_suffix: str | None = None,
) -> None:
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")

    run_name = {
        "stage1": "finance_grpo_stage1_scorer_shaped",
        "stage2": "finance_grpo_stage2_judge_shaped",
    }.get(stage, "finance_grpo")
    run_name = run_name + ("_smoke" if smoke else "") + ("_beta0" if beta0 else "")
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"
    argv = [
        "train_finance_grpo.py",
        "--config", "configs/finance_grpo.yaml",
        "--output-dir", f"/outputs/{run_name}",
        "--checkpoint-commit-every", "10",
    ]
    if stage:
        argv.extend(["--stage", stage])
    if resume_from_checkpoint:
        argv.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    if max_steps is not None:
        argv.extend(["--max-steps", str(max_steps)])
    if smoke:
        argv.append("--smoke")
    if beta0:
        argv.append("--beta0")

    sys.argv = argv
    from finance_poisoning.rl.train_finance_grpo import main

    # Let the trainer callback persist checkpoints/metrics mid-run.
    import finance_poisoning.rl.train_finance_grpo as trainer_mod

    old_load_config = trainer_mod.load_config

    def load_config_with_commit(*args, **kwargs):
        cfg = old_load_config(*args, **kwargs)
        cfg["commit_callback"] = outputs.commit
        return cfg

    trainer_mod.load_config = load_config_with_commit
    rc = main()
    outputs.commit()
    if rc != 0:
        raise SystemExit(rc)


@app.local_entrypoint()
def cli(
    smoke: bool = False,
    beta0: bool = False,
    stage: str | None = None,
    resume_from_checkpoint: str | None = None,
    max_steps: int | None = None,
    run_suffix: str | None = None,
    background: bool = False,
) -> None:
    kwargs = {
        "smoke": smoke,
        "beta0": beta0,
        "stage": stage,
        "resume_from_checkpoint": resume_from_checkpoint,
        "max_steps": max_steps,
        "run_suffix": run_suffix,
    }
    if not background:
        train.remote(**kwargs)
        return
    call = train.spawn(**kwargs)
    call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None) or str(call)
    print(
        "Spawned finance GRPO training. The Modal function will continue "
        f"independently; call_id={call_id}",
        flush=True,
    )
