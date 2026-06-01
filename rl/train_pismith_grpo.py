"""GRPO trainer for the PISmith persistent-memory attacker.

Trains a small attacker LLM with TRL's GRPO so its `<prompt>` memory payloads
maximize `PersistentMemoryAttackReward`. The target victim and judge are the
deployed Qwen2.5-7B vLLM server reached via the `.env` `QWEN_BASE_URL` (one
endpoint serves both roles); only the attacker policy is updated here.

Run locally (needs the training stack — see requirements-train.txt):
    python rl/train_pismith_grpo.py --config configs/pismith_grpo.yaml --smoke
    python rl/train_pismith_grpo.py --config configs/pismith_grpo.yaml

On Modal: scripts/modal_train_grpo.py wraps this (see RUN_MODAL.md).

Heavy deps (trl/datasets/peft/transformers) are imported inside `main` so this
module stays importable (and the corpus/Helper utilities testable) without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pismith_env import PersistentMemoryAttackReward, PersistentMemoryDataset  # noqa: E402
from pismith_env.config import PISmithMemoryEnvConfig  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "pismith_grpo.yaml"
DEFAULT_SEED_CORPUS = ROOT / "data" / "benign_memories.seed.jsonl"
DEFAULT_EMBEDDED_CORPUS = ROOT / "data" / "benign_memories.jsonl"

# GRPOConfig (TRL) candidate keys we may pass through; filtered to the fields
# the installed TRL version actually defines, so version drift won't crash.
_GRPO_KEYS = [
    "output_dir", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "num_generations", "max_prompt_length",
    "max_completion_length", "max_steps", "save_steps", "logging_steps",
    "temperature", "beta", "seed", "bf16", "gradient_checkpointing",
    "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "report_to",
]


def ensure_embedded_corpus(seed_path: Path, out_path: Path) -> Path:
    """Materialize an embedding-carrying corpus once so per-episode stores load
    pre-embedded vectors instead of re-encoding the whole corpus every episode.
    """
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[corpus] using existing embedded corpus: {out_path}", flush=True)
        return out_path

    from env.memory_store import _load_default_embedder

    rows = [json.loads(line) for line in Path(seed_path).read_text().splitlines() if line.strip()]
    print(f"[corpus] embedding {len(rows)} seed entries -> {out_path}", flush=True)
    model = _load_default_embedder()
    vecs = model.encode([r["text"] for r in rows], normalize_embeddings=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row, vec in zip(rows, vecs):
            row = dict(row)
            row["embedding"] = [float(x) for x in vec]
            f.write(json.dumps(row) + "\n")
    return out_path


def load_config(path: Path, smoke: bool, beta0: bool) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    smoke_overrides = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg.update(smoke_overrides)
        cfg["output_dir"] = cfg.get("output_dir", "outputs/pismith_grpo") + "_smoke"
    if beta0:
        cfg["beta"] = 0.0
        cfg["output_dir"] = cfg.get("output_dir", "outputs/pismith_grpo") + "_beta0"
    return cfg


def build_env_config(cfg: dict[str, Any]) -> PISmithMemoryEnvConfig:
    env_field_names = {f.name for f in fields(PISmithMemoryEnvConfig)}
    env_kwargs = {k: v for k, v in cfg.items() if k in env_field_names}
    if isinstance(env_kwargs.get("task_ids"), list):
        env_kwargs["task_ids"] = tuple(env_kwargs["task_ids"])
    return PISmithMemoryEnvConfig(**env_kwargs)


def build_hf_dataset(task_ids, samples_per_task: int):
    from datasets import Dataset as HFDataset

    ds = PersistentMemoryDataset(task_ids=tuple(task_ids), samples_per_task=samples_per_task)
    return HFDataset.from_list([ds[i] for i in range(len(ds))])


def build_grpo_args(cfg: dict[str, Any]):
    from trl import GRPOConfig

    valid = {f.name for f in fields(GRPOConfig)}
    kwargs = {k: cfg[k] for k in _GRPO_KEYS if k in cfg and k in valid}
    kwargs.setdefault("report_to", "none")
    dropped = [k for k in _GRPO_KEYS if k in cfg and k not in valid]
    if dropped:
        print(f"[grpo] WARNING: GRPOConfig has no fields {dropped} in this TRL "
              f"version; dropping them.", flush=True)
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


def preflight_judge(reward: PersistentMemoryAttackReward, task_id: str) -> bool:
    """One live judge call so a broken endpoint (e.g. vLLM not honoring
    response_format JSON) is caught up front instead of silently degrading
    hybrid/judge to regex mid-run. Warns loudly but does not abort.
    """
    from env.tasks import TASKS

    task = TASKS[task_id]
    query = task.trigger_queries[0]
    output = f"Sure — the answer is {task.attack_token}."
    try:
        v = reward.judge.evaluate(query, output, task)
        print(f"[preflight] judge OK: attack_succeeded={v.attack_succeeded} "
              f"aligned={v.aligned_with_user} reason={v.reason[:80]!r}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001 - want any failure surfaced
        print(f"[preflight] WARNING: judge call FAILED ({e!r}). hybrid/judge will "
              f"fall back to regex for every episode. Check the endpoint's JSON "
              f"response_format support, or set success_signal=regex.", flush=True)
        return False


def make_monitor_callback(reward: PersistentMemoryAttackReward, metrics_path=None):
    """A TrainerCallback that prints + logs the online ASR decomposition each
    step, reading the reward's `last_traces` (the same factoring decompose_exp1
    does offline) so mode-collapse / retrieval-gating is visible during training
    and plottable afterward.
    """
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
            scored = [t for t in traces if not t.floored and t.payload]
            row = {
                "step": state.global_step,
                "asr_retr": _mean(t.retrieval_rate for t in scored),
                "asr_judge": _mean(t.success_rate for t in scored),
                "asr_regex": _mean(t.success_rate_regex for t in scored),
                "reward": _mean(t.reward for t in traces),
                "len_w": _mean(t.n_words for t in scored),
                "too_short": _mean(1.0 if t.floored else 0.0 for t in traces),
                "nov_pen": _mean(t.novelty_penalty for t in scored),
                "div_sim": _mean(t.max_cos_sim for t in scored),
                "judge_calls": sum(t.judge_calls for t in traces),
                "judge_err": sum(t.judge_errors for t in traces),
            }
            print(
                f"[PISmith] step={row['step']} ASR_retr={row['asr_retr']:.2f} "
                f"ASR_judge={row['asr_judge']:.2f} ASR_regex={row['asr_regex']:.2f} "
                f"reward={row['reward']:.3f} len={row['len_w']:.0f}w "
                f"too_short={row['too_short']:.2f} nov_pen={row['nov_pen']:.3f} "
                f"div_sim={row['div_sim']:.2f} judge_calls={row['judge_calls']} "
                f"judge_err={row['judge_err']}",
                flush=True,
            )
            if sink is not None:
                with sink.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    return MonitorCallback()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true", help="Apply fast smoke overrides.")
    ap.add_argument("--beta0", action="store_true", help="KL ablation: beta=0.0.")
    ap.add_argument("--output-dir", default=None, help="Override output_dir.")
    ap.add_argument("--corpus-seed", type=Path, default=DEFAULT_SEED_CORPUS)
    ap.add_argument("--corpus-out", type=Path, default=DEFAULT_EMBEDDED_CORPUS)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke, beta0=args.beta0)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    # Pre-embed the corpus once, then point the reward at it.
    corpus_path = ensure_embedded_corpus(args.corpus_seed, args.corpus_out)
    cfg["corpus_path"] = str(corpus_path)

    env_cfg = build_env_config(cfg)
    reward = PersistentMemoryAttackReward(env_cfg)

    # Validate the judge endpoint up front (incl. in the smoke, which trains with
    # regex) so a broken endpoint surfaces before the run, not silently mid-run.
    if args.smoke or env_cfg.success_signal != "regex":
        preflight_judge(reward, cfg["task_ids"][0])

    dataset = build_hf_dataset(cfg["task_ids"], cfg.get("samples_per_task", 256))
    grpo_args = build_grpo_args(cfg)
    peft_config = build_peft_config(cfg)

    from trl import GRPOTrainer

    print(f"[train] policy={cfg['policy_model']} steps={cfg.get('max_steps')} "
          f"G={cfg.get('num_generations')} reward_mode={env_cfg.reward_mode} "
          f"signal={env_cfg.success_signal} beta={cfg.get('beta')} "
          f"peft={bool(peft_config)} rows={len(dataset)}", flush=True)

    trainer = GRPOTrainer(
        model=cfg["policy_model"],
        reward_funcs=[reward],
        args=grpo_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[make_monitor_callback(
            reward, metrics_path=Path(cfg["output_dir"]) / "train_metrics.jsonl"
        )],
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[train] done. saved to {cfg['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
