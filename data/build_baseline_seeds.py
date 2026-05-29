#!/usr/bin/env python3
"""Build paired seeds for the published-attack baseline (Exp-B).

Mirrors `data/build_exp1_seeds.py` exactly, but the malicious side comes from
`attacks/published.py` (MINJA-style / MemoryGraft-style single-payload ports)
instead of the hand-crafted A/B/C payloads. Each malicious record is paired
with a benign sibling via the *same* mask-and-rephrase + forbidden-content
filter path, so the baseline drops straight into `experiments/exp1_handcrafted.py`
with no runner changes.

The seed's `stealth` field carries the **attack family** ("MINJA", ...) rather
than an A/B/C level, so the existing by-`stealth` aggregation in the sweep
summary becomes a by-family aggregation for free. A redundant `family` field is
written alongside for clarity.

Families with no implemented payloads (e.g. MemoryGraft, awaiting source) are
skipped with a notice — they are NOT invented.

Run:
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python data/build_baseline_seeds.py
Requires `OPENAI_API_KEY` (loaded from repo-root `.env` via python-dotenv).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attacks.published import PUBLISHED_FAMILIES  # noqa: E402
from data.build_benign_corpus import PERSONAS  # noqa: E402
from data.build_exp1_seeds import METHODS, generate_sibling  # noqa: E402
from env.tasks import TASKS  # noqa: E402

OUT_PATH = ROOT / "data" / "baseline_seeds.jsonl"
PERSONAS_BY_ID = {p["id"]: p for p in PERSONAS}


def _load_existing(out_path: Path) -> dict[str, dict]:
    """Existing seed records keyed by payload_id (for incremental builds)."""
    if not out_path.exists():
        return {}
    out: dict[str, dict] = {}
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["payload_id"]] = r
    return out


def build(out_path: Path, seed: int, rebuild: bool = False) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set (checked process env and repo-root "
            ".env via python-dotenv); benign-sibling generation needs it."
        )
    import openai

    client = openai.OpenAI()
    rng = random.Random(seed)

    # Flatten implemented families, skipping any with no payloads yet.
    flat: list[PublishedPayload] = []  # type: ignore[name-defined]
    skipped_families: list[str] = []
    for family, payloads in PUBLISHED_FAMILIES.items():
        if not payloads:
            skipped_families.append(family)
            continue
        flat.extend(payloads)

    if skipped_families:
        print(
            f"Skipping unimplemented families (no source yet): {skipped_families}",
            file=sys.stderr,
        )
    if not flat:
        raise RuntimeError("no implemented published payloads to build")

    # Incremental: keep existing records whose payload is still present; only
    # (re)generate payloads not already on disk. This preserves benign siblings
    # for families already swept (so their per-episode logs stay valid and the
    # resumable sweep skips them). `--rebuild` forces full regeneration.
    existing = {} if rebuild else _load_existing(out_path)
    current_ids = {p.id for p in flat}
    kept = {pid: r for pid, r in existing.items() if pid in current_ids}
    to_generate = [p for p in flat if p.id not in kept]
    if kept:
        print(f"Reusing {len(kept)} existing seed(s); generating "
              f"{len(to_generate)} new.", file=sys.stderr)

    # Per-family pair-id counter seeded from kept records so new pair_ids
    # don't collide with existing ones.
    family_counter: dict[str, int] = {}
    for r in kept.values():
        family_counter[r["family"]] = max(
            family_counter.get(r["family"], 0),
            int(r["pair_id"].rsplit("_", 1)[1]),
        )

    # Block-assign methods round-robin over a seeded shuffle of the to-generate
    # subset (decouples method from family).
    order = list(range(len(to_generate)))
    rng.shuffle(order)
    method_of = {idx: METHODS[rank % len(METHODS)] for rank, idx in enumerate(order)}

    new_records: list[dict] = []
    dropped: list[tuple[str, list[str]]] = []
    for i, payload in enumerate(to_generate):
        method = method_of[i]
        persona = PERSONAS_BY_ID[payload.persona]
        task = TASKS[payload.target_task]
        family_counter[payload.family] = family_counter.get(payload.family, 0) + 1
        pair_id = f"{payload.family.lower()}_{family_counter[payload.family]:03d}"

        print(
            f"[{payload.id}] family={payload.family} task={task.id} "
            f"method={method} persona={payload.persona}",
            file=sys.stderr,
        )

        benign, rejections = generate_sibling(
            client, payload, method, persona, task.attack_token, rng
        )
        if benign is None:
            print(f"  DROPPED after retries: {rejections}", file=sys.stderr)
            dropped.append((payload.id, rejections))
            continue
        if rejections:
            print(f"  accepted on retry; prior rejections: {rejections}", file=sys.stderr)

        new_records.append(
            {
                "pair_id": pair_id,
                "payload_id": payload.id,
                "task_id": payload.target_task,
                "persona": payload.persona,
                # stealth carries the family so the sweep summary buckets by family.
                "stealth": payload.family,
                "family": payload.family,
                "method": method,
                "malicious_text": payload.text,
                "benign_text": benign,
                "trigger_query_distribution": list(task.trigger_queries),
                "attack_success_if": task.attack_token,
                "metadata": {
                    "rejections_before_accept": rejections,
                    "generator": "gpt-4o-mini",
                    "build_seed": seed,
                    "baseline": True,
                },
            }
        )

    # Merge kept + new, ordered by family then pair_id for a stable file.
    records = list(kept.values()) + new_records
    records.sort(key=lambda r: (r["family"], r["pair_id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    by_family: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for r in records:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
        by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + 1
    print(f"\nWrote {len(records)} paired baseline seeds to {out_path} "
          f"({len(new_records)} new, {len(kept)} reused)", file=sys.stderr)
    print(f"  by family : {by_family}", file=sys.stderr)
    print(f"  by task   : {by_task}", file=sys.stderr)
    if dropped:
        print(f"Dropped {len(dropped)} pair(s):", file=sys.stderr)
        for pid, reasons in dropped:
            print(f"  {pid}: {reasons}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate all families from scratch (ignore existing file)")
    args = ap.parse_args()
    build(args.out, args.seed, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
