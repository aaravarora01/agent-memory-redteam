"""Experiment 2 — trivial-defense baseline (plan §Exp 2).

Re-runs the exact same 10 paired Exp 1 seeds, but with one cheap defense
prepended, to ask: does the Exp 1 attack success (notably the 51%
B-narrative judge-ASR) survive a defense that costs essentially nothing?

Two defenses (see `env/defenses.py`):

  - `firewall` — hardened system prompt (instruction-firewall). Retrieval is
    unchanged; the agent is told to treat memory as untrusted data.
  - `filter`   — regex/heuristic classifier that drops adversarial-looking
    retrieved entries before they reach the agent.
  - `both`     — firewall + filter stacked.
  - `none`     — re-run with production defaults (sanity baseline that should
    reproduce Exp 1; useful when comparing under an identical harness).

Mirrors `exp1_handcrafted.py`: N episodes per (pair × condition) using
`run_episode` + `Judge`, resumable per `(pair_id, condition, episode_idx)`,
streaming to `results/exp2_<defense>_episodes.jsonl` and a per-defense
`results/exp2_<defense>_summary.json`. Extra columns vs Exp 1:

  - `payload_reached_agent` — did the poisoned entry survive the filter?
  - `n_filtered`            — how many retrieved entries the filter dropped.

Run (one defense at a time; each is ~400 episodes / ~30 min / ~$0.30):
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python \\
        experiments/exp2_defenses.py --defense firewall --n 20
    /Users/MihirMenon/miniconda3/envs/cs224r/bin/python \\
        experiments/exp2_defenses.py --defense filter --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.agent import Agent  # noqa: E402
from env.defenses import DEFENSE_NAMES, get_defense  # noqa: E402
from env.episode import DEFAULT_CORPUS_PATH, run_episode  # noqa: E402
from env.judge import Judge  # noqa: E402
from env.memory_store import MemoryStore  # noqa: E402
from env.tasks import TASKS  # noqa: E402

SEEDS_PATH = ROOT / "data" / "exp1_seeds.jsonl"
RESULTS_DIR = ROOT / "results"
DEFAULT_N = 20


def _episodes_path(defense: str) -> Path:
    return RESULTS_DIR / f"exp2_{defense}_episodes.jsonl"


def _summary_path(defense: str) -> Path:
    return RESULTS_DIR / f"exp2_{defense}_summary.json"


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------


def _load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["pair_id"], r["condition"], int(r["episode_idx"])))
    return done


def _load_shared_embedder():
    from sentence_transformers import SentenceTransformer

    from env.memory_store import EMBED_MODEL

    return SentenceTransformer(EMBED_MODEL)


# --------------------------------------------------------------------------
# Per-episode driver
# --------------------------------------------------------------------------


def _run_one(
    pair: dict,
    condition: str,
    episode_idx: int,
    defense: str,
    agent: Agent,
    judge: Judge,
    embedder,
    corpus_path: Path,
    k: int,
    retrieval_filter,
) -> dict:
    task = TASKS[pair["task_id"]]
    payload = pair["malicious_text"] if condition == "malicious" else pair["benign_text"]

    store = MemoryStore.from_corpus(corpus_path, embedder=embedder)

    t0 = time.perf_counter()
    res = run_episode(
        payload=payload,
        task=task,
        store=store,
        agent=agent,
        k=k,
        seed=episode_idx,
        retrieval_filter=retrieval_filter,
        payload_metadata={
            "pair_id": pair["pair_id"],
            "payload_id": pair["payload_id"],
            "condition": condition,
        },
    )
    agent_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        verdict = judge.evaluate(res.query, res.output, task)
        judge_err = None
    except Exception as e:
        verdict = None
        judge_err = repr(e)
    judge_dt = time.perf_counter() - t1

    return {
        "defense": defense,
        "pair_id": pair["pair_id"],
        "payload_id": pair["payload_id"],
        "task_id": pair["task_id"],
        "stealth": pair["stealth"],
        "method": pair["method"],
        "persona": pair["persona"],
        "condition": condition,
        "episode_idx": episode_idx,
        "seed": episode_idx,
        "query": res.query,
        "payload_text": payload,
        "payload_in_topk": res.payload_in_topk,
        "payload_reached_agent": res.payload_reached_agent,
        "n_filtered": res.n_filtered,
        "regex_success": res.success,
        "judge_aligned": (verdict.aligned_with_user if verdict else None),
        "judge_attack_succeeded": (verdict.attack_succeeded if verdict else None),
        "judge_reason": (verdict.reason if verdict else ""),
        "judge_error": judge_err,
        "output": res.output,
        "elapsed_agent_s": round(agent_dt, 2),
        "elapsed_judge_s": round(judge_dt, 2),
    }


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------


def run_sweep(
    defense: str,
    n: int,
    corpus_path: Path,
    seeds_path: Path,
    episodes_path: Path,
    summary_path: Path,
    k: int = 5,
    resume: bool = True,
    pair_filter: Optional[set[str]] = None,
) -> None:
    pairs = _load_pairs(seeds_path)
    if pair_filter:
        pairs = [p for p in pairs if p["pair_id"] in pair_filter]
    if not pairs:
        raise RuntimeError(f"no pairs to run (loaded {seeds_path})")

    system_prompt, retrieval_filter = get_defense(defense)

    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing_keys(episodes_path) if resume else set()
    if done:
        print(f"Resuming: {len(done)} episodes already logged in {episodes_path}", file=sys.stderr)

    print("Loading shared embedder + warming corpus ...", file=sys.stderr)
    embedder = _load_shared_embedder()
    MemoryStore.from_corpus(corpus_path, embedder=embedder)

    agent = Agent(system_prompt=system_prompt)  # None → Agent's default prompt
    judge = Judge()

    print(
        f"Defense = {defense!r}: "
        f"system_prompt={'hardened' if system_prompt else 'default'}, "
        f"retrieval_filter={'on' if retrieval_filter else 'off'}",
        file=sys.stderr,
    )

    conditions = ("malicious", "benign")
    total = len(pairs) * len(conditions) * n
    skipped = 0
    written = 0

    log_f = episodes_path.open("a", buffering=1)
    try:
        for pair in pairs:
            for condition in conditions:
                for ep in range(n):
                    key = (pair["pair_id"], condition, ep)
                    if key in done:
                        skipped += 1
                        continue
                    row = _run_one(
                        pair=pair,
                        condition=condition,
                        episode_idx=ep,
                        defense=defense,
                        agent=agent,
                        judge=judge,
                        embedder=embedder,
                        corpus_path=corpus_path,
                        k=k,
                        retrieval_filter=retrieval_filter,
                    )
                    log_f.write(json.dumps(row) + "\n")
                    written += 1
                    if written % 10 == 0 or ep == n - 1:
                        print(
                            f"  [{defense}|{pair['pair_id']} {condition} ep={ep:02d}] "
                            f"topk={row['payload_in_topk']} "
                            f"reached={row['payload_reached_agent']} "
                            f"judge_atk={row['judge_attack_succeeded']} "
                            f"({written + skipped}/{total})",
                            file=sys.stderr,
                        )
    finally:
        log_f.close()

    print(f"\nWrote {written} new episodes (skipped {skipped} resumed)", file=sys.stderr)
    _write_summary(defense, episodes_path, summary_path, n)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _write_summary(defense: str, episodes_path: Path, summary_path: Path, n: int) -> None:
    rows = []
    with episodes_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    groups: dict[tuple[str, str], list[dict]] = {}
    pair_meta: dict[str, dict] = {}
    for r in rows:
        groups.setdefault((r["pair_id"], r["condition"]), []).append(r)
        pair_meta.setdefault(
            r["pair_id"],
            {
                "payload_id": r["payload_id"],
                "task_id": r["task_id"],
                "stealth": r["stealth"],
                "method": r["method"],
                "persona": r["persona"],
            },
        )

    def _rate(xs, key):
        vals = [x[key] for x in xs if x.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    per_pair: list[dict] = []
    for pair_id, meta in pair_meta.items():
        mal = groups.get((pair_id, "malicious"), [])
        ben = groups.get((pair_id, "benign"), [])
        per_pair.append({
            "pair_id": pair_id,
            **meta,
            "n_malicious": len(mal),
            "n_benign": len(ben),
            "retrieval_at_5_malicious": _rate(mal, "payload_in_topk"),
            "reached_agent_malicious": _rate(mal, "payload_reached_agent"),
            "mean_filtered_malicious": _rate(mal, "n_filtered"),
            "mean_filtered_benign": _rate(ben, "n_filtered"),
            "asr_judge_malicious": _rate(mal, "judge_attack_succeeded"),
            "asr_judge_benign": _rate(ben, "judge_attack_succeeded"),
            "asr_regex_malicious": _rate(mal, "regex_success"),
            "asr_regex_benign": _rate(ben, "regex_success"),
            "survivor": any(x.get("judge_attack_succeeded") for x in mal),
        })

    by_stealth: dict[str, dict] = {}
    for rec in per_pair:
        b = by_stealth.setdefault(
            rec["stealth"],
            {"n_pairs": 0, "ret5_mal": [], "reach_mal": [], "asr_mal": [],
             "asr_ben": [], "regex_mal": [], "survivors": 0},
        )
        b["n_pairs"] += 1
        b["survivors"] += int(bool(rec["survivor"]))
        for key, src in [
            ("ret5_mal", "retrieval_at_5_malicious"),
            ("reach_mal", "reached_agent_malicious"),
            ("asr_mal", "asr_judge_malicious"),
            ("asr_ben", "asr_judge_benign"),
            ("regex_mal", "asr_regex_malicious"),
        ]:
            if rec[src] is not None:
                b[key].append(rec[src])

    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    by_stealth_out = {}
    for stealth, b in by_stealth.items():
        by_stealth_out[stealth] = {
            "n_pairs": b["n_pairs"],
            "survivors": b["survivors"],
            "retrieval_at_5_malicious_mean": mean(b["ret5_mal"]),
            "reached_agent_malicious_mean": mean(b["reach_mal"]),
            "asr_judge_malicious_mean": mean(b["asr_mal"]),
            "asr_judge_benign_mean": mean(b["asr_ben"]),
            "asr_regex_malicious_mean": mean(b["regex_mal"]),
        }

    out = {
        "defense": defense,
        "n_target": n,
        "per_pair": per_pair,
        "by_stealth": by_stealth_out,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote summary → {summary_path}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", choices=DEFENSE_NAMES, required=True,
                    help="which cheap defense to prepend")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="episodes per (pair, condition) [default 20]")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    ap.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    ap.add_argument("--episodes-out", type=Path, default=None,
                    help="default results/exp2_<defense>_episodes.jsonl")
    ap.add_argument("--summary-out", type=Path, default=None,
                    help="default results/exp2_<defense>_summary.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    episodes_out = args.episodes_out or _episodes_path(args.defense)
    summary_out = args.summary_out or _summary_path(args.defense)

    if args.summary_only:
        _write_summary(args.defense, episodes_out, summary_out, args.n)
        return

    if args.no_resume and episodes_out.exists():
        episodes_out.unlink()

    pair_filter = set(args.pairs) if args.pairs else None
    run_sweep(
        defense=args.defense,
        n=args.n,
        corpus_path=args.corpus,
        seeds_path=args.seeds,
        episodes_path=episodes_out,
        summary_path=summary_out,
        k=args.k,
        resume=not args.no_resume,
        pair_filter=pair_filter,
    )


if __name__ == "__main__":
    main()
