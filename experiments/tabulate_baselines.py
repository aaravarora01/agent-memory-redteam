"""Tabulate the published-attack baseline (Exp-B) and frame it as a ceiling.

Reads `results/baseline_summary.json` (produced by running
`experiments/exp1_handcrafted.py --seeds data/baseline_seeds.jsonl ...`) and,
if present, `results/exp1_summary.json` (the hand-crafted sweep) so the table
can put the published-attack numbers next to the hand-crafted ones per task.

Writes `results/baseline_table.md`:
  1. By-family headline table (Retrieval@5, ASR-judge mal/benign, ASR-regex, survivors).
  2. Per-task comparison: published vs. best hand-crafted (the "ceiling vs floor" view).
  3. Per-pair detail for the published families.
  4. Flags block: benign false-positives (contamination) and non-survivors.

No API calls. Run:
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python experiments/tabulate_baselines.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_SUMMARY = ROOT / "results" / "baseline_summary.json"
EXP1_SUMMARY = ROOT / "results" / "exp1_summary.json"
OUT_PATH = ROOT / "results" / "baseline_table.md"


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _load(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _by_family_table(summary: dict) -> list[str]:
    rows = [
        "| Family | Pairs | Retrieval@5 (mal) | ASR-judge (mal) | ASR-judge (benign) | ASR-regex (mal) | Survivors |",
        "|--------|-------|-------------------|-----------------|--------------------|-----------------|-----------|",
    ]
    for fam, b in sorted(summary.get("by_stealth", {}).items()):
        rows.append(
            f"| {fam} | {b['n_pairs']} | {_pct(b['retrieval_at_5_malicious_mean'])} "
            f"| {_pct(b['asr_judge_malicious_mean'])} | {_pct(b['asr_judge_benign_mean'])} "
            f"| {_pct(b['asr_regex_malicious_mean'])} | {b['survivors']}/{b['n_pairs']} |"
        )
    return rows


def _per_task_best(summary: dict | None) -> dict[str, dict]:
    """task_id -> best-by-ASR-judge pair record (the strongest payload for that task)."""
    out: dict[str, dict] = {}
    if not summary:
        return out
    for rec in summary.get("per_pair", []):
        tid = rec["task_id"]
        cur = out.get(tid)
        asr = rec.get("asr_judge_malicious") or 0.0
        if cur is None or asr > (cur.get("asr_judge_malicious") or 0.0):
            out[tid] = rec
    return out


def _comparison_table(baseline: dict, exp1: dict | None) -> list[str]:
    base_best = _per_task_best(baseline)
    hand_best = _per_task_best(exp1)
    tasks = sorted(set(base_best) | set(hand_best))
    rows = [
        "| Task | Published retr@5 | Published ASR-judge | Hand-crafted best retr@5 | Hand-crafted best ASR-judge |",
        "|------|------------------|---------------------|--------------------------|-----------------------------|",
    ]
    for tid in tasks:
        b = base_best.get(tid)
        h = hand_best.get(tid)
        rows.append(
            f"| {tid} "
            f"| {_pct(b['retrieval_at_5_malicious']) if b else '—'} "
            f"| {_pct(b['asr_judge_malicious']) if b else '—'} "
            f"| {_pct(h['retrieval_at_5_malicious']) if h else '—'} "
            f"| {_pct(h['asr_judge_malicious']) if h else '—'} |"
        )
    return rows


def _per_pair_table(summary: dict) -> list[str]:
    rows = [
        "| Pair | Family | Task | Method | Retr@5 (mal) | ASR-judge (mal) | ASR-judge (ben) | Survivor |",
        "|------|--------|------|--------|--------------|-----------------|-----------------|----------|",
    ]
    for rec in summary.get("per_pair", []):
        rows.append(
            f"| {rec['pair_id']} | {rec.get('stealth', '?')} | {rec['task_id']} "
            f"| {rec.get('method', '?')} | {_pct(rec['retrieval_at_5_malicious'])} "
            f"| {_pct(rec['asr_judge_malicious'])} | {_pct(rec['asr_judge_benign'])} "
            f"| {'yes' if rec['survivor'] else 'NO'} |"
        )
    return rows


def _flags(summary: dict) -> list[str]:
    fp = [r for r in summary.get("per_pair", []) if (r.get("asr_judge_benign") or 0) > 0]
    non = [r for r in summary.get("per_pair", []) if not r.get("survivor")]
    lines = ["## Flags", ""]
    if fp:
        lines.append("**Benign false-positives (judge fired on the benign sibling — contamination signal):**")
        for r in fp:
            lines.append(f"- {r['pair_id']} ({r['task_id']}): benign ASR-judge {_pct(r['asr_judge_benign'])}")
    else:
        lines.append("- No benign false-positives. ✅")
    lines.append("")
    if non:
        lines.append("**Non-survivors (0 judge-confirmed successes across N malicious episodes):**")
        for r in non:
            lines.append(f"- {r['pair_id']} ({r['task_id']})")
    else:
        lines.append("- All published payloads survived (≥1 judge-confirmed success). ✅")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=BASELINE_SUMMARY)
    ap.add_argument("--exp1", type=Path, default=EXP1_SUMMARY)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    baseline = _load(args.baseline)
    if baseline is None:
        raise SystemExit(
            f"missing {args.baseline}; run the baseline sweep first:\n"
            "  experiments/exp1_handcrafted.py --seeds data/baseline_seeds.jsonl "
            "--episodes-out results/baseline_episodes.jsonl "
            "--summary-out results/baseline_summary.json"
        )
    exp1 = _load(args.exp1)

    out: list[str] = []
    out.append("# Published-attack baseline (Exp-B)")
    out.append("")
    out.append(
        "**Single-payload ports**, not full reproductions. MINJA (Dong et al. 2025) "
        "= retrievability bridging + reasoning-trace framing + reuse indication. "
        "MemoryGraft (Srivastava & He 2025) = trigger-free \"validated/safe "
        "best-practice\" success records exploiting the semantic-imitation heuristic. "
        "Both collapsed into one ingested record and run through the same Exp-1 sweep "
        "as the hand-crafted payloads; the multi-turn / code-execution delivery "
        "mechanisms are NOT reproduced (MemoryGraft's port also omits the paper's "
        "BM25∪FAISS union retrieval — our store is FAISS-only — so it *understates* "
        "MemoryGraft). This is the published-attack **ceiling** the RL attacker must "
        "beat to claim novelty."
    )
    out.append(f"\n_N = {baseline.get('n_target')} episodes per (pair, condition)._\n")

    out.append("## By family")
    out += _by_family_table(baseline)
    out.append("")

    out.append("## Published vs. hand-crafted, per task")
    if exp1 is None:
        out.append("_(hand-crafted summary `results/exp1_summary.json` not found — published-only)_\n")
    out += _comparison_table(baseline, exp1)
    out.append("")

    out.append("## Per-pair detail (published)")
    out += _per_pair_table(baseline)
    out.append("")

    out += _flags(baseline)
    out.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"Wrote {args.out}", file=sys.stderr)
    print("\n".join(out))


if __name__ == "__main__":
    main()
