"""Orchestrate live/API memory-poisoning simulations.

This is a thin, safe wrapper around the existing experiment drivers. It keeps
the default run small, checks Modal/Qwen readiness before making API calls, and
uses long request timeouts for Modal cold starts.

Examples:
    python scripts/run_live_simulation.py --mode dryrun
    python scripts/run_live_simulation.py --mode exp1 --n 20
    python scripts/run_live_simulation.py --mode baseline --n 20
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
        ["scripts/check_modal_ready.py", "--profile", "simulation"],
    )


def _pismith_live(args: argparse.Namespace) -> int:
    steps = [
        ("PISmith adapter offline smoke", ["experiments/pismith_env_smoke.py"]),
        (
            "PISmith live reward smoke",
            [
                "experiments/pismith_env_smoke.py",
                "--run-episode",
                "--reward-mode",
                args.reward_mode,
                "--request-timeout",
                str(args.request_timeout),
            ],
        ),
    ]
    for label, cmd in steps:
        code = _run(label, cmd)
        if code != 0:
            return code
    return 0


def _smoke_test(args: argparse.Namespace) -> int:
    cmd = [
        "experiments/smoke_test.py",
        "--seeds",
        *[str(s) for s in args.seeds],
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
        "--save",
    ]
    return _run("Live end-to-end smoke test", cmd)


def _finance_qwen(args: argparse.Namespace) -> int:
    cmd = [
        "finance_poisoning/experiments/f5_qwen_victim.py",
        "--limit",
        str(args.finance_limit),
        "--mode",
        args.finance_mode,
        "--reward-mode",
        args.finance_reward_mode,
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
        "--seed",
        str(args.seed),
    ]
    if args.agent_model:
        cmd.extend(["--agent-model", args.agent_model])
    return _run("Finance Qwen victim dry run", cmd)


def _exp1_dryrun(args: argparse.Namespace) -> int:
    cmd = [
        "experiments/exp1_handcrafted.py",
        "--n",
        str(args.n),
        "--pairs",
        *args.pairs,
        "--no-resume",
        "--episodes-out",
        "results/exp1_dryrun.jsonl",
        "--summary-out",
        "results/exp1_dryrun_summary.json",
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
    ]
    return _run("Exp 1 tiny live dry run", cmd)


def _exp1_full(args: argparse.Namespace) -> int:
    code = _run(
        "Exp 1 full live sweep",
        [
            "experiments/exp1_handcrafted.py",
            "--n",
            str(args.n),
            "--request-timeout",
            str(args.request_timeout),
            "--max-retries",
            str(args.max_retries),
        ],
    )
    if code != 0:
        return code
    return _run("Tabulate Exp 1", ["experiments/tabulate_exp1.py"])


def _baseline_full(args: argparse.Namespace) -> int:
    code = _run(
        "Published-baseline live sweep",
        [
            "experiments/exp1_handcrafted.py",
            "--n",
            str(args.n),
            "--seeds",
            "data/baseline_seeds.jsonl",
            "--episodes-out",
            "results/baseline_episodes.jsonl",
            "--summary-out",
            "results/baseline_summary.json",
            "--request-timeout",
            str(args.request_timeout),
            "--max-retries",
            str(args.max_retries),
        ],
    )
    if code != 0:
        return code
    return _run("Tabulate published baseline", ["experiments/tabulate_baselines.py"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dryrun", "smoke", "finance-qwen", "exp1", "baseline", "all"],
        default="dryrun",
        help="dryrun is intentionally small; exp1/baseline/all can spend real API time.",
    )
    parser.add_argument("--n", type=int, default=2, help="Episodes per pair/condition.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", nargs="+", default=["pair_001"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--reward-mode", choices=["terminal", "retrieval", "composite"], default="composite")
    parser.add_argument("--finance-mode", choices=["tool_optional", "tool_forced"], default="tool_optional")
    parser.add_argument("--finance-reward-mode", choices=["sparse", "shaped"], default="sparse")
    parser.add_argument("--finance-limit", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--skip-ready-check", action="store_true")
    args = parser.parse_args()

    code = _ready_check(args.skip_ready_check)
    if code != 0:
        return code

    if args.mode == "dryrun":
        for step in (_pismith_live, _finance_qwen, _exp1_dryrun):
            code = step(args)
            if code != 0:
                return code
        return 0

    if args.mode == "smoke":
        for step in (_pismith_live, _finance_qwen, _smoke_test):
            code = step(args)
            if code != 0:
                return code
        return 0

    if args.mode == "finance-qwen":
        return _finance_qwen(args)

    if args.mode == "exp1":
        return _exp1_full(args)

    if args.mode == "baseline":
        return _baseline_full(args)

    if args.mode == "all":
        for step in (_pismith_live, _finance_qwen, _smoke_test, _exp1_full, _baseline_full):
            code = step(args)
            if code != 0:
                return code
        return 0

    raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
