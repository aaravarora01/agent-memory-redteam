"""F2: sparse-reward random/epsilon-greedy baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from finance_poisoning.env.finance_env import FinanceMemoryPoisonEnv
from finance_poisoning.env.schemas import TargetFact
from finance_poisoning.experiments._logging import log_config, log_episode_row, open_log
from finance_poisoning.experiments.plot_reward_curve import plot_reward_curve
from finance_poisoning.rl.action_space import PoisonActionSpace
from finance_poisoning.rl.random_policy import RandomPolicy


RESULTS_DIR = ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="F2 sparse policy baseline")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "f2_episodes.jsonl"))
    parser.add_argument("--plot", type=str, default=str(RESULTS_DIR / "f2_reward_curve.png"))
    args = parser.parse_args()

    action_space = PoisonActionSpace()
    policy = RandomPolicy(action_space, seed=args.seed)
    facts = [f.value for f in TargetFact]

    with open_log(args.out) as f:
        log_config(f, {
            "experiment": "f2_sparse",
            "episodes": args.episodes,
            "seed": args.seed,
            "reward_mode": "sparse",
        })

        for i in range(args.episodes):
            fact = facts[i % len(facts)]
            env = FinanceMemoryPoisonEnv(k=args.k, reward_mode="sparse", seed=args.seed + i)
            obs = env.reset(target_fact_id=fact, seed=args.seed + i)
            action = policy.act(obs)
            result = env.run_episode_log(action, reward_mode="sparse")
            row = result.to_log_dict()
            row["reward"] = result.retrieval_reward
            log_episode_row(f, row)

    plot_reward_curve(
        args.out,
        args.plot,
        title=f"F2 sparse baseline (n={args.episodes})",
    )
    print(f"F2 complete: {args.episodes} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
