"""Unit tests for retrieval scorer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from finance_poisoning.env.schemas import RetrievalHit
from finance_poisoning.env.scorer import score_retrieval


def _hit(memory_id: str, rank: int, fact_id: str, is_poison: bool, score: float) -> RetrievalHit:
    return RetrievalHit(
        memory_id=memory_id,
        rank=rank,
        score=score,
        text=f"text for {memory_id}",
        fact_id=fact_id,
        is_poison=is_poison,
        value=1500 if is_poison else 1850,
    )


def test_poison_ranks_above_truth() -> None:
    hits = [
        _hit("poison_1", 1, "rent_amount", True, 0.9),
        _hit("mem_rent_amount_true", 2, "rent_amount", False, 0.8),
    ]
    score = score_retrieval(hits, "rent_amount", k=5)
    assert score.poison_ranks_above_truth
    assert score.incorrect_retrieval_dominant
    assert score.poison_rank_advantage == 1


def test_truth_displaced() -> None:
    hits = [
        _hit("poison_1", 1, "rent_amount", True, 0.9),
        _hit("mem_filler", 2, "filler", False, 0.5),
    ]
    score = score_retrieval(hits, "rent_amount", k=5)
    assert score.truth_displaced
    assert score.incorrect_retrieval_dominant


def test_clean_retrieval_no_dominant_incorrect() -> None:
    hits = [
        _hit("mem_rent_amount_true", 1, "rent_amount", False, 0.9),
    ]
    score = score_retrieval(hits, "rent_amount", k=5)
    assert not score.incorrect_retrieval_dominant
    assert not score.poison_in_top_k
