"""Markdown table generation for finance poisoning results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_by_key(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get(key, "unknown")].append(row)
    out: dict[str, dict[str, float]] = {}
    for k, group in sorted(buckets.items()):
        out[k] = {
            "n": len(group),
            "irr_at_5": _mean([float(r.get("incorrect_retrieval_dominant", 0)) for r in group]),
            "poison_in_top5": _mean([float(r.get("poison_in_top5", 0)) for r in group]),
            "truth_retention_at_5": _mean([float(r.get("truth_in_top5", 0)) for r in group]),
            "poison_rank_advantage": _mean([
                float(r["poison_rank_advantage"])
                for r in group
                if r.get("poison_rank_advantage") is not None
            ]),
            "collateral_damage_rate": _mean([float(r.get("collateral_retrieval", 0)) for r in group]),
            "mean_retrieval_reward": _mean([float(r.get("retrieval_reward", 0)) for r in group]),
        }
    return out


def render_metrics_table(title: str, metrics: dict[str, dict[str, float]]) -> str:
    lines = [f"## {title}", ""]
    lines.append(
        "| Group | N | IRR@5 | PoisonInTop5 | TruthRetention@5 | PoisonRankAdvantage | CollateralDamage |"
    )
    lines.append("|-------|---|-------|--------------|------------------|---------------------|------------------|")
    for group, m in metrics.items():
        lines.append(
            f"| {group} | {m['n']} | {m['irr_at_5']:.2%} | {m['poison_in_top5']:.2%} | "
            f"{m['truth_retention_at_5']:.2%} | {m['poison_rank_advantage']:.2f} | "
            f"{m['collateral_damage_rate']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_f1_report(rows: list[dict]) -> str:
    sections = [
        "# F1 Hand-crafted Finance Poison Results",
        "",
        render_metrics_table("By framing style", aggregate_by_key(rows, "poison_style")),
        render_metrics_table("By target fact", aggregate_by_key(rows, "target_fact")),
    ]
    return "\n".join(sections)


def build_mode_comparison(rows: list[dict]) -> str:
    return render_metrics_table("By mode", aggregate_by_key(rows, "mode"))
