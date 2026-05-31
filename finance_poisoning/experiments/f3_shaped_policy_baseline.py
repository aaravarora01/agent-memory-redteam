"""F3: shaped-reward epsilon-greedy baseline."""

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
from finance_poisoning.rl.ppo_placeholder import EpsilonGreedyPolicy


RESULTS_DIR = ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="F3 shaped policy baseline")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "f3_episodes.jsonl"))
    parser.add_argument("--plot", type=str, default=str(RESULTS_DIR / "f3_reward_curve.png"))
    args = parser.parse_args()

    action_space = PoisonActionSpace()
    policy = EpsilonGreedyPolicy(action_space=action_space, epsilon=args.epsilon, seed=args.seed)
    facts = [f.value for f in TargetFact]

    with open_log(args.out) as f:
        log_config(f, {
            "experiment": "f3_shaped",
            "episodes": args.episodes,
            "seed": args.seed,
            "reward_mode": "shaped",
            "epsilon": args.epsilon,
        })

        for i in range(args.episodes):
            fact = facts[i % len(facts)]
            env = FinanceMemoryPoisonEnv(k=args.k, reward_mode="shaped", seed=args.seed + i)
            obs = env.reset(target_fact_id=fact, seed=args.seed + i)
            action = policy.act(obs)
            result = env.run_episode_log(action, reward_mode="shaped")
            policy.update(obs, action, result.retrieval_reward)
            row = result.to_log_dict()
            row["reward"] = result.retrieval_reward
            log_episode_row(f, row)

    plot_reward_curve(
        args.out,
        args.plot,
        title=f"F3 shaped baseline (n={args.episodes}, eps={args.epsilon})",
    )
    print(f"F3 complete: {args.episodes} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
