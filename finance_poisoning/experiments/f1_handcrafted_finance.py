"""F1: hand-crafted finance poison sweep over framing styles and facts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from finance_poisoning.env.finance_env import FinanceMemoryPoisonEnv
from finance_poisoning.env.schemas import TargetFact
from finance_poisoning.experiments._logging import log_config, log_episode_row, open_log
from finance_poisoning.experiments._tabulate import aggregate_by_key, build_f1_report
from finance_poisoning.rl.action_space import handcrafted_actions


RESULTS_DIR = ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="F1 handcrafted finance poison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "f1_episodes.jsonl"))
    parser.add_argument("--summary", type=str, default=str(RESULTS_DIR / "f1_summary.json"))
    parser.add_argument("--md", type=str, default=str(RESULTS_DIR / "f1_by_style.md"))
    args = parser.parse_args()

    actions = handcrafted_actions()
    rng = random.Random(args.seed)
    rows: list[dict] = []

    with open_log(args.out) as f:
        log_config(f, {"experiment": "f1_handcrafted", "seed": args.seed, "k": args.k})

        for i, action in enumerate(actions):
            env = FinanceMemoryPoisonEnv(k=args.k, seed=args.seed + i)
            env.reset(target_fact_id=action.target_fact, seed=args.seed + i)
            result = env.run_episode_log(action, reward_mode="shaped")
            row = result.to_log_dict()
            row["poison_rank_advantage"] = (
                (result.truth_rank - result.poison_rank)
                if result.truth_rank and result.poison_rank
                else None
            )
            log_episode_row(f, row)
            rows.append(row)

    summary = {
        "n_episodes": len(rows),
        "by_style": aggregate_by_key(rows, "poison_style"),
        "by_fact": aggregate_by_key(rows, "target_fact"),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.md).write_text(build_f1_report(rows), encoding="utf-8")

    print(f"F1 complete: {len(rows)} episodes")
    for style, m in summary["by_style"].items():
        print(f"  {style}: IRR@5={m['irr_at_5']:.2%} TruthRetention@5={m['truth_retention_at_5']:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
