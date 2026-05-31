"""Run the no-API finance simulator suite in order.

This is the deadline-safe path: it runs unit tests, clean sanity, handcrafted
poisoning, sparse/shaped policy baselines, tool verification, and then refreshes
the one-page deadline report.

Run from the repository root:
    python scripts/run_offline_finance.py --episodes 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, args: list[str]) -> int:
    print(f"\n== {label} ==", flush=True)
    proc = subprocess.run([sys.executable, *args], cwd=ROOT)
    if proc.returncode != 0:
        print(f"{label} failed with exit code {proc.returncode}", file=sys.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_tests:
        steps.append(("Finance unit tests", ["-m", "pytest", "finance_poisoning/tests/", "-q"]))
    steps.extend(
        [
            ("F0 clean sanity", ["finance_poisoning/experiments/f0_clean_sanity.py"]),
            (
                "F1 hand-crafted finance poison",
                ["finance_poisoning/experiments/f1_handcrafted_finance.py", "--seed", str(args.seed)],
            ),
            (
                "F2 sparse policy baseline",
                [
                    "finance_poisoning/experiments/f2_sparse_policy_baseline.py",
                    "--episodes",
                    str(args.episodes),
                    "--seed",
                    str(args.seed),
                ],
            ),
            (
                "F3 shaped policy baseline",
                [
                    "finance_poisoning/experiments/f3_shaped_policy_baseline.py",
                    "--episodes",
                    str(args.episodes),
                    "--seed",
                    str(args.seed),
                ],
            ),
            (
                "F4 tool verification",
                ["finance_poisoning/experiments/f4_tool_optional_answer.py", "--seed", str(args.seed)],
            ),
            ("Deadline report", ["scripts/deadline_report.py"]),
        ]
    )

    for label, cmd in steps:
        code = _run(label, cmd)
        if code != 0:
            return code

    print("\nOffline finance suite complete.", flush=True)
    print(f"Report: {ROOT / 'results' / 'deadline_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
