"""Build a one-page deadline status report from local results.

This script is intentionally offline-only. It does not call the target LLM or
download models; it just summarizes environment readiness and result artifacts
that already exist in the workspace.

Run from the repository root:
    python scripts/deadline_report.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "deadline_report.md"


DEPENDENCIES = [
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("pytest", "pytest"),
    ("matplotlib", "matplotlib"),
    ("sentence-transformers", "sentence_transformers"),
    ("faiss-cpu", "faiss"),
    ("tiktoken", "tiktoken"),
    ("torch", "torch"),
    ("python-dotenv", "dotenv"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_episode_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "episode":
                rows.append(row)
    return rows


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100 * float(value):.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        values.append(float(row[key]))
    if not values:
        return None
    return sum(values) / len(values)


def _dependency_lines() -> list[str]:
    lines = [
        f"- Python: `{sys.version.split()[0]}` at `{sys.executable}`",
        f"- Platform: `{platform.platform()}`",
        f"- Repo `.env`: {'present' if (ROOT / '.env').exists() else 'missing'}",
    ]
    for package, module in DEPENDENCIES:
        ok = importlib.util.find_spec(module) is not None
        lines.append(f"- {package}: {'OK' if ok else 'MISSING'}")
    return lines


def _git_status_line() -> str:
    cmd = [
        "git",
        "-c",
        f"safe.directory={ROOT.as_posix()}",
        "status",
        "--short",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - best effort only
        return f"- Git status: unavailable ({exc})"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return f"- Git status: unavailable ({detail[0] if detail else 'unknown error'})"
    changed = [line for line in proc.stdout.splitlines() if line.strip()]
    return f"- Git status: {len(changed)} changed/untracked paths"


def _exp1_section(lines: list[str]) -> None:
    exp1 = _load_json(ROOT / "results" / "exp1_summary.json")
    baseline = _load_json(ROOT / "results" / "baseline_summary.json")

    lines += ["## Persistent-memory LLM Track", ""]
    if exp1:
        lines.append("Hand-crafted payload sweep (`results/exp1_summary.json`):")
        lines.append("")
        lines.append("| Stealth | Pairs | Survivors | Retrieval@5 | Judge ASR | Benign ASR |")
        lines.append("|---------|-------|-----------|-------------|-----------|------------|")
        labels = {"A": "A overt", "B": "B narrative", "C": "C indirect"}
        for key in ["A", "B", "C"]:
            row = exp1.get("by_stealth", {}).get(key)
            if not row:
                continue
            lines.append(
                f"| {labels[key]} | {row['n_pairs']} | "
                f"{row['survivors']}/{row['n_pairs']} | "
                f"{_pct(row.get('retrieval_at_5_malicious_mean'))} | "
                f"{_pct(row.get('asr_judge_malicious_mean'))} | "
                f"{_pct(row.get('asr_judge_benign_mean'))} |"
            )
        lines.append("")
        lines.append("Interpretation: retrieval is the first gate; T1/T2 are the weak points, while T3/T4 are already saturated in the existing sweep.")
        lines.append("")
    else:
        lines.append("- Missing `results/exp1_summary.json`; run `python experiments/exp1_handcrafted.py --n 20` only after `.env` is set.")
        lines.append("")

    if baseline:
        lines.append("Published-attack baseline (`results/baseline_summary.json`):")
        lines.append("")
        lines.append("| Family | Pairs | Survivors | Retrieval@5 | Judge ASR | Benign ASR |")
        lines.append("|--------|-------|-----------|-------------|-----------|------------|")
        for family, row in sorted(baseline.get("by_stealth", {}).items()):
            lines.append(
                f"| {family} | {row['n_pairs']} | "
                f"{row['survivors']}/{row['n_pairs']} | "
                f"{_pct(row.get('retrieval_at_5_malicious_mean'))} | "
                f"{_pct(row.get('asr_judge_malicious_mean'))} | "
                f"{_pct(row.get('asr_judge_benign_mean'))} |"
            )
        lines.append("")
    else:
        lines.append("- Missing published-attack baseline summary.")
        lines.append("")


def _finance_section(lines: list[str]) -> None:
    lines += ["## Finance Simulator Track", ""]

    f0 = _load_episode_rows(ROOT / "finance_poisoning" / "results" / "f0_sanity.jsonl")
    if f0:
        lines.append(
            f"- F0 clean sanity: {len(f0)} queries, "
            f"TruthRetention@5={_pct(_mean(f0, 'truth_in_top5'))}, "
            f"Tool accuracy={_pct(_mean(f0, 'tool_match'))}"
        )
    else:
        lines.append("- F0 clean sanity: not run yet")

    f1 = _load_json(ROOT / "finance_poisoning" / "results" / "f1_summary.json")
    if f1:
        lines.append("")
        lines.append("F1 hand-crafted poison by framing style:")
        lines.append("")
        lines.append("| Style | N | IRR@5 | PoisonInTop5 | TruthRetention@5 |")
        lines.append("|-------|---|-------|--------------|------------------|")
        for style, row in sorted(f1.get("by_style", {}).items()):
            lines.append(
                f"| {style} | {row['n']} | {_pct(row.get('irr_at_5'))} | "
                f"{_pct(row.get('poison_in_top5'))} | "
                f"{_pct(row.get('truth_retention_at_5'))} |"
            )
    else:
        lines.append("- F1 hand-crafted finance poison: not run yet")

    f2 = _load_episode_rows(ROOT / "finance_poisoning" / "results" / "f2_episodes.jsonl")
    f3 = _load_episode_rows(ROOT / "finance_poisoning" / "results" / "f3_episodes.jsonl")
    if f2 or f3:
        lines.append("")
        lines.append("Policy baselines:")
        lines.append("")
        lines.append("| Run | Episodes | Mean reward | IRR@5 | Last-50 reward |")
        lines.append("|-----|----------|-------------|-------|----------------|")
        for label, rows in [("F2 sparse random", f2), ("F3 shaped epsilon-greedy", f3)]:
            if not rows:
                continue
            tail = rows[-50:]
            lines.append(
                f"| {label} | {len(rows)} | {_mean(rows, 'retrieval_reward') or 0:.2f} | "
                f"{_pct(_mean(rows, 'incorrect_retrieval_dominant'))} | "
                f"{_mean(tail, 'retrieval_reward') or 0:.2f} |"
            )

    f4 = _load_json(ROOT / "finance_poisoning" / "results" / "tool_optional_summary.json")
    if f4:
        lines.append("")
        lines.append("F4 answer-level defense comparison:")
        lines.append("")
        lines.append("| Mode | N | Uses poison | Contradicts tool | Tool use | IRR@5 |")
        lines.append("|------|---|-------------|------------------|----------|-------|")
        for mode, row in sorted(f4.get("by_mode", {}).items()):
            lines.append(
                f"| {mode} | {row['n']} | {_pct(row.get('answer_uses_poison_rate'))} | "
                f"{_pct(row.get('answer_contradiction_rate'))} | "
                f"{_pct(row.get('tool_use_rate'))} | {_pct(row.get('irr_at_5'))} |"
            )
    else:
        lines.append("- F4 tool-optional vs tool-forced: not run yet")
    lines.append("")


def build_report() -> str:
    lines: list[str] = [
        "# Deadline Status Report",
        "",
        "This report is generated offline from local artifacts. It is meant to answer: what can be trusted, what is presentation-ready, and what still needs live API credentials.",
        "",
        "## Environment",
        "",
    ]
    lines.extend(_dependency_lines())
    lines.append(_git_status_line())
    lines.append("")

    _exp1_section(lines)
    _finance_section(lines)

    lines += [
        "## Recommended Close-Deadline Path",
        "",
        "1. Use the finance simulator for guaranteed no-API demonstrations: F0 clean sanity, F1 retrieval poisoning, F2/F3 reward-shaping contrast, and F4 tool verification.",
        "2. Treat the committed Exp 1 and published-baseline summaries as the LLM-backed evidence unless you have a stable `.env` and enough time to rerun.",
        "3. If rerunning live LLM experiments, do a one-pair dry run first, then the full sweep. Keep the JSONL logs; the tabulators can regenerate tables without new API calls.",
        "",
        "## Commands",
        "",
        "Fast offline checks:",
        "",
        "```powershell",
        "python scripts/run_offline_finance.py --episodes 500",
        "```",
        "",
        "Individual offline commands:",
        "",
        "```powershell",
        "python -m pytest finance_poisoning/tests/ -q",
        "python finance_poisoning/experiments/f0_clean_sanity.py",
        "python finance_poisoning/experiments/f1_handcrafted_finance.py --seed 0",
        "python finance_poisoning/experiments/f2_sparse_policy_baseline.py --episodes 500 --seed 0",
        "python finance_poisoning/experiments/f3_shaped_policy_baseline.py --episodes 500 --seed 0",
        "python finance_poisoning/experiments/f4_tool_optional_answer.py --seed 0",
        "python scripts/deadline_report.py",
        "```",
        "",
        "Live LLM dry run after `.env` is configured:",
        "",
        "```powershell",
        "python scripts/check_modal_ready.py --profile simulation",
        "python scripts/run_live_simulation.py --mode dryrun",
        "python scripts/run_live_simulation.py --mode exp1 --n 20",
        "python scripts/check_modal_ready.py --profile training",
        "python scripts/run_training.py --mode live-exp3 --plot",
        "```",
        "",
        "## Things To Be Careful About",
        "",
        "- `.env` is currently required for target-agent and judge calls. Without it, stick to finance/offline simulations and existing summaries.",
        "- For Modal, copy `.env.example` to `.env`, replace the Modal host, keep the `/v1` suffix, and use `QWEN_API_KEY=EMPTY` for the default unauthenticated vLLM server.",
        "- The live LLM/RL track needs `sentence-transformers`, `faiss-cpu`, `tiktoken`, and `torch`; the finance simulator does not.",
        "- Do not commit `data/benign_memories.jsonl`; it is a local embedded cache.",
        "- JSONL and PNG outputs are mostly gitignored. Markdown/JSON summaries are the artifacts to keep for the report.",
        "- On this Windows workspace, Git may report dubious ownership. Use a per-command safe-directory override or explicitly mark this repo safe if you trust the path.",
        "- First MiniLM use can download a model. The finance simulator defaults to TF-IDF, so it avoids that dependency unless `FINANCE_RETRIEVER=minilm` is set.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
