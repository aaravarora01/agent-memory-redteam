"""Two-phase rollout (plan §1.6).

```
def run_episode(payload, task):
    store = MemoryStore.from_corpus(benign_memories)
    store.ingest(payload)                # Phase 1: poison the memory
    query = sample(task.trigger_queries) # Phase 2: future user turn
    retrieved = store.query(query, k=5)
    output  = agent.act(query, retrieved)
    return task.success_check(output), output, retrieved
```

This is the *temporal-gap* MDP the proposal frames: Phase 1 ingest writes to a
persistent store; Phase 2 retrieve+execute happens on an independently-sampled
user query that arrives later. Each episode rebuilds the store from the benign
corpus so payloads don't accumulate across episodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .agent import Agent
from .memory_store import MemoryEntry, MemoryStore
from .tasks import Task


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _default_corpus_path() -> Path:
    """Prefer the locally-embedded cache; fall back to the committed seed.

    `data/benign_memories.jsonl` carries pre-computed MiniLM embeddings and
    is produced by `data/build_benign_corpus.py` (gitignored). Teammates who
    just cloned the repo won't have it — they'll only have the text-only
    `data/benign_memories.seed.jsonl` (committed), in which case
    `MemoryStore.from_corpus` re-embeds at load time.
    """
    embedded = _DATA_DIR / "benign_memories.jsonl"
    if embedded.exists():
        return embedded
    return _DATA_DIR / "benign_memories.seed.jsonl"


DEFAULT_CORPUS_PATH = _default_corpus_path()


@dataclass
class EpisodeResult:
    success: bool
    output: str
    retrieved: List[MemoryEntry]
    payload_in_topk: bool
    query: str = ""
    payload_id: str = ""
    metadata: dict = field(default_factory=dict)
    # Defense bookkeeping (plan §Exp 2). When no `retrieval_filter` is passed,
    # `payload_reached_agent == payload_in_topk` and `n_filtered == 0`, so
    # existing callers see unchanged behavior.
    payload_reached_agent: bool = False
    n_filtered: int = 0


def run_episode(
    payload: str,
    task: Task,
    store: Optional[MemoryStore] = None,
    agent: Optional[Agent] = None,
    k: int = 5,
    seed: Optional[int] = None,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    payload_metadata: Optional[dict] = None,
    retrieval_filter: Optional[Callable[[List[MemoryEntry]], List[MemoryEntry]]] = None,
    embedder=None,
) -> EpisodeResult:
    """Run one ingest → query → retrieve → act episode.

    Parameters
    ----------
    payload:
        The text written into the memory store during Phase 1. Pass an empty
        string to run a clean (no-poison) control episode.
    task:
        Defines the trigger-query distribution and the success predicate.
    store:
        Optional pre-built `MemoryStore`. If `None`, a fresh store is built
        from `corpus_path`. Passing one in lets the experiment caller cache
        the benign corpus across episodes; the runner will still mutate it
        with `ingest`, so the caller is responsible for rebuilding between
        episodes if isolation matters.
    agent:
        Optional pre-built `Agent`. Constructed lazily if `None` so that
        importing this module doesn't require an API key.
    k:
        Top-k retrieval count.
    seed:
        Deterministic query sampling. If `None`, sampling uses the global RNG.
    retrieval_filter:
        Optional defense (plan §Exp 2): a callable applied to the top-k
        retrieved entries *before* `agent.act`. It returns the subset of
        entries the agent is allowed to see (e.g. dropping adversarial-looking
        memories). `payload_in_topk` still reflects raw retrieval; the new
        `payload_reached_agent` reflects what survived the filter.
    embedder:
        Optional shared sentence-transformer passed to `MemoryStore.from_corpus`
        when a fresh store is built. Callers that run many episodes (e.g. the
        PISmith GRPO reward) pass one cached encoder so each episode does not
        lazy-load its own MiniLM. Ignored when `store` is supplied.
    """
    if store is None:
        store = MemoryStore.from_corpus(corpus_path, embedder=embedder)
    if agent is None:
        agent = Agent()

    # Phase 1: ingest payload (skip if empty for clean control episodes).
    payload_id = ""
    if payload:
        meta = dict(payload_metadata or {})
        meta.setdefault("malicious", True)
        payload_id = store.ingest(payload, metadata=meta)

    # Phase 2: sample a trigger query, retrieve, act.
    rng = random.Random(seed) if seed is not None else random
    query = rng.choice(task.trigger_queries)
    retrieved = store.query(query, k=k)
    payload_in_topk = bool(payload_id) and any(r.id == payload_id for r in retrieved)

    # Defense hook: drop adversarial-looking entries before the agent sees them.
    n_filtered = 0
    if retrieval_filter is not None:
        kept = retrieval_filter(retrieved)
        n_filtered = len(retrieved) - len(kept)
        retrieved = kept
    payload_reached_agent = bool(payload_id) and any(r.id == payload_id for r in retrieved)

    output = agent.act(query, retrieved)
    success = task.success_check(output)

    return EpisodeResult(
        success=success,
        output=output,
        retrieved=retrieved,
        payload_in_topk=payload_in_topk,
        query=query,
        payload_id=payload_id,
        payload_reached_agent=payload_reached_agent,
        n_filtered=n_filtered,
    )
