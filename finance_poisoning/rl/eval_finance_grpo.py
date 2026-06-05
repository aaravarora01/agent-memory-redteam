"""Evaluate a finance GRPO attacker checkpoint against the Qwen victim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from finance_poisoning.grpo import FinancePoisonDataset, FinancePoisonReward  # noqa: E402
from finance_poisoning.rl.train_finance_grpo import build_env_config, load_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "finance_grpo.yaml"


def load_attacker(model_path: str, base_model: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            is_trainable=False,
        )
    except Exception:
        model_name = base_model or model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    tokenizer_source = model_path if Path(model_path).exists() else (base_model or model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model or model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prompt: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    import torch

    torch.manual_seed(seed)
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    def mean(key: str) -> float:
        return sum(float(r.get(key) or 0.0) for r in rows) / len(rows)

    by_fact: dict[str, dict[str, Any]] = {}
    for fact in sorted({r["target_fact"] for r in rows}):
        subset = [r for r in rows if r["target_fact"] == fact]
        by_fact[fact] = {
            "n": len(subset),
            "valid_action": sum(float(r.get("valid_action") or 0.0) for r in subset) / len(subset),
            "retrieval_at_5": sum(float(r.get("poison_in_top5") or 0.0) for r in subset) / len(subset),
            "scorer_asr": sum(float(r.get("scorer_success") or 0.0) for r in subset) / len(subset),
            "judge_asr": sum(float(r.get("judge_success") or 0.0) for r in subset) / len(subset),
        }
    return {
        "n": len(rows),
        "valid_action": mean("valid_action"),
        "retrieval_at_5": mean("poison_in_top5"),
        "scorer_asr": mean("scorer_success"),
        "judge_asr": mean("judge_success"),
        "judge_error_rate": mean("judge_error"),
        "reward_mean": mean("reward"),
        "by_fact": by_fact,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Attacker checkpoint/model path")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    ap.add_argument("--n", type=int, default=5, help="Generations per target fact")
    ap.add_argument("--success-signal", choices=["scorer", "judge", "hybrid"], default="judge")
    ap.add_argument("--reward-mode", choices=["sparse", "shaped"], default="sparse")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("finance_poisoning/results/finance_grpo_eval.jsonl"))
    ap.add_argument("--summary", type=Path, default=Path("finance_poisoning/results/finance_grpo_eval_summary.json"))
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=False, beta0=False, stage=args.stage)
    cfg["success_signal"] = args.success_signal
    cfg["reward_mode"] = args.reward_mode
    env_cfg = build_env_config(cfg)
    reward = FinancePoisonReward(env_cfg)
    dataset = FinancePoisonDataset(
        target_facts=env_cfg.target_facts,
        samples_per_fact=args.n,
    )
    model, tokenizer = load_attacker(args.model, base_model=cfg["policy_model"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            sample = dataset[i]
            completion = generate_completion(
                model,
                tokenizer,
                sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed + i,
            )
            score, trace = reward.score_completion(
                completion,
                target_fact=sample["target_fact"],
                sample_idx=i,
            )
            row = {
                "sample_idx": i,
                "target_fact": sample["target_fact"],
                "completion": completion,
                **trace.__dict__,
                "reward": score,
            }
            rows.append(row)
            f.write(json.dumps(row, default=str) + "\n")
            print(
                f"[eval-finance] {i + 1}/{len(dataset)} fact={sample['target_fact']} "
                f"reward={score:.3f} valid={trace.valid_action} "
                f"retr={trace.poison_in_top5} scorer={trace.scorer_success} "
                f"judge={trace.judge_success} err={trace.judge_error}",
                flush=True,
            )

    summary = summarize(rows)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
