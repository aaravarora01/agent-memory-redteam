"""Training launcher for offline finance and live sparse-PPO runs.

The default `finance` mode is no-API and fast. The `live-exp3` mode calls the
LLM-backed target agent once per episode, so it performs the Modal/Qwen readiness
check first and defaults to a small 24-episode smoke run. Pass
`--episodes 2000 --batch-size 64 --wall-clock-cap 7200` for the full Exp 3 run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, args: list[str]) -> int:
    print(f"\n== {label} ==", flush=True)
    print(" ".join([sys.executable, *args]), flush=True)
    proc = subprocess.run([sys.executable, *args], cwd=ROOT)
    if proc.returncode != 0:
        print(f"{label} failed with exit code {proc.returncode}", file=sys.stderr)
    return proc.returncode


def _ready_check(skip: bool) -> int:
    if skip:
        return 0
    return _run(
        "Modal/Qwen readiness check",
        ["scripts/check_modal_ready.py", "--profile", "training"],
    )


def _finance_training(args: argparse.Namespace) -> int:
    steps = [
        (
            "Finance sparse policy baseline",
            [
                "finance_poisoning/experiments/f2_sparse_policy_baseline.py",
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "Finance shaped policy baseline",
            [
                "finance_poisoning/experiments/f3_shaped_policy_baseline.py",
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.seed),
            ],
        ),
        ("Deadline report", ["scripts/deadline_report.py"]),
    ]
    for label, cmd in steps:
        code = _run(label, cmd)
        if code != 0:
            return code
    return 0


def _live_exp3(args: argparse.Namespace) -> int:
    code = _ready_check(args.skip_ready_check)
    if code != 0:
        return code

    cmd = [
        "experiments/exp3_sparse_failure.py",
        "--episodes",
        str(args.episodes),
        "--batch-size",
        str(args.batch_size),
        "--minibatch-size",
        str(args.minibatch_size),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--wall-clock-cap",
        str(args.wall_clock_cap),
        "--task",
        args.task,
        "--seed",
        str(args.seed),
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
        "--episodes-out",
        str(args.episodes_out),
        "--updates-out",
        str(args.updates_out),
    ]
    if args.agent_model:
        cmd.extend(["--agent-model", args.agent_model])
    if args.plot:
        cmd.append("--plot")
    return _run("Live Exp 3 sparse-PPO training", cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finance", "live-exp3"], default="finance")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Defaults: finance=500, live-exp3=24. Use 2000 for the full live sweep.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="T1_brand_hijack")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--wall-clock-cap", type=float, default=7200.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip-ready-check", action="store_true")
    parser.add_argument("--episodes-out", type=Path, default=ROOT / "results" / "exp3_episodes.jsonl")
    parser.add_argument("--updates-out", type=Path, default=ROOT / "results" / "exp3_updates.jsonl")
    args = parser.parse_args()

    if args.mode == "finance":
        if args.episodes is None:
            args.episodes = 500
        return _finance_training(args)

    if args.mode == "live-exp3":
        if args.episodes is None:
            print(
                "live-exp3 defaulting to 24 episodes for a cheap smoke run. "
                "Pass --episodes 2000 for the full training sweep.",
                flush=True,
            )
            args.episodes = 24
            args.batch_size = min(args.batch_size, 8)
            args.minibatch_size = min(args.minibatch_size, 8)
            args.ppo_epochs = min(args.ppo_epochs, 2)
        return _live_exp3(args)

    raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
