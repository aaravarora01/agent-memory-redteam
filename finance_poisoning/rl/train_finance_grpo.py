"""GRPO trainer for finance structured poison-action generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from finance_poisoning.grpo import FinanceGRPOEnvConfig, FinancePoisonDataset, FinancePoisonReward  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "finance_grpo.yaml"
_GRPO_KEYS = [
    "output_dir", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "num_generations", "max_prompt_length",
    "max_completion_length", "max_steps", "save_steps", "logging_steps",
    "temperature", "beta", "seed", "bf16", "gradient_checkpointing",
    "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "report_to",
]


def load_config(
    path: Path,
    smoke: bool,
    beta0: bool,
    stage: str | None = None,
) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    smoke_overrides = cfg.pop("smoke", {}) or {}
    curriculum = cfg.pop("curriculum", {}) or {}
    if stage:
        if stage not in curriculum:
            raise ValueError(
                f"Unknown stage {stage!r}. Known stages: {sorted(curriculum)}"
            )
        cfg.update(curriculum[stage] or {})
    if smoke:
        cfg.update(smoke_overrides)
        cfg["output_dir"] = str(cfg.get("output_dir", "outputs/finance_grpo")) + "_smoke"
    if beta0:
        cfg["beta"] = 0.0
        cfg["output_dir"] = str(cfg.get("output_dir", "outputs/finance_grpo")) + "_beta0"
    return cfg


def build_env_config(cfg: dict[str, Any]) -> FinanceGRPOEnvConfig:
    env_field_names = {f.name for f in fields(FinanceGRPOEnvConfig)}
    env_kwargs = {k: v for k, v in cfg.items() if k in env_field_names}
    if isinstance(env_kwargs.get("target_facts"), list):
        env_kwargs["target_facts"] = tuple(env_kwargs["target_facts"])
    return FinanceGRPOEnvConfig(**env_kwargs)


def build_hf_dataset(target_facts, samples_per_fact: int):
    from datasets import Dataset as HFDataset

    ds = FinancePoisonDataset(
        target_facts=tuple(target_facts),
        samples_per_fact=samples_per_fact,
    )
    rows = []
    for i in range(len(ds)):
        row = dict(ds[i])
        # Arrow needs a stable column type across numeric and string facts.
        row["true_value"] = str(row["true_value"])
        rows.append(row)
    return HFDataset.from_list(rows)


def build_grpo_args(cfg: dict[str, Any]):
    from trl import GRPOConfig

    valid = {f.name for f in fields(GRPOConfig)}
    kwargs = {k: cfg[k] for k in _GRPO_KEYS if k in cfg and k in valid}
    kwargs.setdefault("report_to", "none")
    dropped = [k for k in _GRPO_KEYS if k in cfg and k not in valid]
    if dropped:
        print(f"[grpo] WARNING: dropping unsupported GRPOConfig fields {dropped}", flush=True)
    return GRPOConfig(**kwargs)


def build_peft_config(cfg: dict[str, Any]):
    if not cfg.get("use_peft", False):
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )


def make_monitor_callback(
    reward: FinancePoisonReward,
    metrics_path=None,
    checkpoint_commit_every: int = 0,
    commit_callback=None,
):
    from transformers import TrainerCallback

    sink = Path(metrics_path) if metrics_path else None
    if sink is not None:
        sink.parent.mkdir(parents=True, exist_ok=True)

    def _mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    class MonitorCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            traces = reward.last_traces
            if not traces:
                return
            row = {
                "step": state.global_step,
                "reward": _mean(t.reward for t in traces),
                "valid_action": _mean(1.0 if t.valid_action else 0.0 for t in traces),
                "retrieval_reward": _mean(t.retrieval_reward for t in traces),
                "update_reward": _mean(t.update_reward for t in traces),
                "attack_reward": _mean(t.attack_reward for t in traces),
                "noop_penalty": _mean(t.noop_penalty for t in traces),
                "poison_equals_truth": _mean(1.0 if t.poison_equals_truth else 0.0 for t in traces),
                "tool_fact_present": _mean(1.0 if t.tool_fact_present else 0.0 for t in traces),
                "tool_fact_partial": _mean(1.0 if t.tool_fact_mode == "partial" else 0.0 for t in traces),
                "tool_fact_full": _mean(1.0 if t.tool_fact_mode == "full" else 0.0 for t in traces),
                "tool_fact_none": _mean(1.0 if t.tool_fact_mode == "none" else 0.0 for t in traces),
                "poison_in_top5": _mean(1.0 if t.poison_in_top5 else 0.0 for t in traces),
                "scorer_asr": _mean(1.0 if t.scorer_success else 0.0 for t in traces),
                "judge_asr": _mean(1.0 if t.judge_success else 0.0 for t in traces),
                "judge_err": sum(1 for t in traces if t.judge_error),
            }
            print(
                f"[FinanceGRPO] step={row['step']} reward={row['reward']:.3f} "
                f"valid={row['valid_action']:.2f} retr={row['poison_in_top5']:.2f} "
                f"upd={row['update_reward']:.2f} atk={row['attack_reward']:.2f} "
                f"scorer_asr={row['scorer_asr']:.2f} judge_asr={row['judge_asr']:.2f} "
                f"judge_err={row['judge_err']}",
                flush=True,
            )
            if sink is not None:
                with sink.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if (
                commit_callback is not None
                and checkpoint_commit_every > 0
                and state.global_step % checkpoint_commit_every == 0
            ):
                commit_callback()

    return MonitorCallback()


def preflight_reward(reward: FinancePoisonReward, target_fact: str) -> None:
    completion = (
        '<action>{"target_fact":"%s","corrupted_value_strategy":"replace_with_plausible_wrong_value",'
        '"framing_style":"narrative","retrieval_bait":"target_terms",'
        '"memory_source_type":"transaction_summary","confidence_level":"medium"}</action>'
    ) % target_fact
    try:
        score, trace = reward.score_completion(completion, target_fact)
        print(
            f"[preflight] reward OK: reward={score:.3f} valid={trace.valid_action} "
            f"retr={trace.poison_in_top5} scorer={trace.scorer_success} "
            f"judge={trace.judge_success} err={trace.judge_error}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] WARNING: finance reward call failed: {e!r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--beta0", action="store_true")
    ap.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--checkpoint-commit-every", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, args.smoke, args.beta0, stage=args.stage)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.resume_from_checkpoint:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    env_cfg = build_env_config(cfg)
    reward = FinancePoisonReward(env_cfg)
    if args.smoke or env_cfg.success_signal != "scorer":
        preflight_reward(reward, env_cfg.target_facts[0])

    dataset = build_hf_dataset(env_cfg.target_facts, env_cfg.samples_per_fact)
    grpo_args = build_grpo_args(cfg)
    peft_config = build_peft_config(cfg)

    from trl import GRPOTrainer

    print(
        f"[train-finance] policy={cfg['policy_model']} steps={cfg.get('max_steps')} "
        f"G={cfg.get('num_generations')} reward={env_cfg.reward_mode} "
        f"signal={env_cfg.success_signal} beta={cfg.get('beta')} "
        f"peft={bool(peft_config)} rows={len(dataset)} "
        f"resume={cfg.get('resume_from_checkpoint')}",
        flush=True,
    )

    trainer = GRPOTrainer(
        model=cfg["policy_model"],
        reward_funcs=[reward],
        args=grpo_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[make_monitor_callback(
            reward,
            metrics_path=Path(cfg["output_dir"]) / "train_metrics.jsonl",
            checkpoint_commit_every=args.checkpoint_commit_every,
            commit_callback=cfg.get("commit_callback"),
        )],
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(cfg["output_dir"])
    print(f"[train-finance] done. saved to {cfg['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
