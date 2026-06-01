"""Retrieval/execution decomposition of Exp 1 (plan §2.3, no new API calls).

Reads the per-episode log `results/exp1_episodes.jsonl` (NOT the summary —
the decomposition needs the episode-level `payload_in_topk` × judge outcome
cross-tab, which the pair-level summary collapses) and factors the malicious
attack-success rate (ASR) into the two phases of the MDP:

    ASR  =  P(payload_in_topk)        # Phase 1: retrieval lands the payload
          × P(attack | payload_in_topk) # Phase 2: agent executes once retrieved

The point is to localize where ASR is lost. If retrieval is the floor,
`R_retrievability` is the lever for the composite reward; if the agent
ignores retrieved payloads, `R_stealth` / execution shaping is the lever.

Run:
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python \\
        experiments/decompose_exp1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES_PATH = ROOT / "results" / "exp1_episodes.jsonl"
OUT_PATH = ROOT / "results" / "exp1_decomposition.md"

STEALTH_ORDER = ["A", "B", "C"]
STEALTH_LABELS = {"A": "A overt", "B": "B narrative", "C": "C indirect"}


def _pct(num, den):
    if den == 0:
        return "—"
    return f"{100 * num / den:.0f}%"


def decompose(rows):
    """Return (n, retrieved, succeeded, succ_if_retrieved, succ_if_not) counts."""
    n = len(rows)
    retrieved = sum(r["payload_in_topk"] for r in rows)
    succeeded = sum(r["judge_attack_succeeded"] for r in rows)
    succ_if_ret = sum(
        r["judge_attack_succeeded"] for r in rows if r["payload_in_topk"]
    )
    succ_if_not = sum(
        r["judge_attack_succeeded"] for r in rows if not r["payload_in_topk"]
    )
    return n, retrieved, succeeded, succ_if_ret, succ_if_not


def _row(label, rows):
    n, ret, succ, sir, sin = decompose(rows)
    return (
        f"| {label} | {n} | {_pct(ret, n)} | {_pct(succ, n)} | "
        f"{_pct(sir, ret)} | {sin}/{n - ret} |"
    )


def render(rows) -> str:
    mal = [r for r in rows if r["condition"] == "malicious"]
    n, ret, succ, sir, sin = decompose(mal)

    lines: list[str] = []
    lines.append("# Experiment 1 — Retrieval / Execution Decomposition")
    lines.append("")
    lines.append(
        f"Source: `results/exp1_episodes.jsonl` ({len(rows)} rows, "
        f"{n} malicious). No new API calls — this re-reads the Exp 1 log."
    )
    lines.append("")
    lines.append(
        "The two-phase MDP factors ASR into a retrieval gate and a "
        "conditional-execution rate:"
    )
    lines.append("")
    lines.append("```")
    lines.append(
        f"ASR              = {succ}/{n} = {100 * succ / n:.0f}%"
    )
    lines.append(
        f"  Phase 1  P(payload_in_topk)          = {ret}/{n} = {100 * ret / n:.0f}%"
    )
    lines.append(
        f"  Phase 2  P(attack | payload_in_topk) = {sir}/{ret} = {100 * sir / ret:.0f}%"
        if ret
        else "  Phase 2  P(attack | payload_in_topk) = — (never retrieved)"
    )
    lines.append(
        f"  control  P(attack | NOT in_topk)     = {sin}/{n - ret} = "
        f"{100 * sin / (n - ret):.0f}%" if n - ret else
        f"  control  P(attack | NOT in_topk)     = {sin}/0 = —"
    )
    lines.append("```")
    lines.append("")
    if sin == 0:
        lines.append(
            f"**Retrieval is a strict gate: 0/{n - ret} non-retrieved "
            "episodes succeeded.** Every judge-confirmed success had the "
            "payload in top-k, so ASR is bounded above by the retrieval "
            "rate. The Phase-1 floor — not Phase-2 execution — is what caps "
            "milestone ASR."
        )
    else:
        lines.append(
            f"Note: {sin} episode(s) succeeded *without* the payload in "
            "top-k — retrieval is not a strict prerequisite (check for "
            "leakage / off-target wins)."
        )
    lines.append("")

    header = (
        "| {0} | N | Retrieval@5 | ASR (judge) | "
        "ASR \\| retrieved | ASR \\| not-retrieved |"
    )
    rule = "|---|---|---|---|---|---|"

    lines.append("## By task")
    lines.append("")
    lines.append(header.format("Task"))
    lines.append(rule)
    for t in sorted(set(r["task_id"] for r in mal)):
        lines.append(_row(t, [r for r in mal if r["task_id"] == t]))
    lines.append("")

    lines.append("## By stealth level")
    lines.append("")
    lines.append(
        "*(Confounded with task: stealth aggregates mix all four tasks, so "
        "read the per-task table for the cleaner signal.)*"
    )
    lines.append("")
    lines.append(header.format("Stealth"))
    lines.append(rule)
    for s in STEALTH_ORDER:
        sub = [r for r in mal if r["stealth"] == s]
        if sub:
            lines.append(_row(STEALTH_LABELS[s], sub))
    lines.append("")

    # Regex vs judge disagreement — the judge-primary convention in action.
    dis = [
        r for r in mal if r["regex_success"] != r["judge_attack_succeeded"]
    ]
    lines.append("## Regex vs judge disagreement (malicious)")
    lines.append("")
    if dis:
        by_pair = defaultdict(list)
        for r in dis:
            by_pair[(r["pair_id"], r["task_id"], r["stealth"])].append(r)
        lines.append(
            f"{len(dis)} episode(s) where the regex sanity-check and the "
            "primary judge disagree:"
        )
        lines.append("")
        for (pid, tid, st), eps in sorted(by_pair.items()):
            rg = sum(e["regex_success"] for e in eps)
            jg = sum(e["judge_attack_succeeded"] for e in eps)
            lines.append(
                f"- `{pid}` ({st}, {tid}): {len(eps)} ep — "
                f"regex={rg}, judge={jg} (all `payload_in_topk=True`); "
                "regex over-counts, judge ruled no behavioral drift."
            )
        lines.append("")
    else:
        lines.append("None — regex and judge agree on every malicious episode.")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, default=EPISODES_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if not args.episodes.exists():
        raise SystemExit(
            f"episode log not found at {args.episodes}; "
            "run experiments/exp1_handcrafted.py first"
        )
    rows = [json.loads(line) for line in args.episodes.open() if line.strip()]

    md = render(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
