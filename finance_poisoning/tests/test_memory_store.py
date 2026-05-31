"""Unit tests for finance memory store."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from finance_poisoning.env.memory_store import FinanceMemoryStore
from finance_poisoning.env.schemas import (
    ConfidenceLevel,
    CorruptedValueStrategy,
    FramingStyle,
    MemorySourceType,
    PoisonAction,
    RetrievalBait,
    load_user,
)
from finance_poisoning.rl.action_space import decode_poison_action


@pytest.fixture
def store() -> FinanceMemoryStore:
    return FinanceMemoryStore.from_clean_corpus()


def test_clean_rent_query_retrieves_true_memory(store: FinanceMemoryStore) -> None:
    hits = store.retrieve("How much do I usually pay for rent?", k=5)
    fact_ids = [h.fact_id for h in hits]
    assert "rent_amount" in fact_ids
    rent_hit = next(h for h in hits if h.fact_id == "rent_amount")
    assert not rent_hit.is_poison
    assert rent_hit.value == 1850


def test_poison_insert_does_not_mutate_clean_snapshot(store: FinanceMemoryStore) -> None:
    user = load_user()
    action = PoisonAction(
        target_fact="rent_amount",
        corrupted_value_strategy=CorruptedValueStrategy.REPLACE_WITH_PLAUSIBLE_WRONG_VALUE.value,
        framing_style=FramingStyle.NARRATIVE.value,
        retrieval_bait=RetrievalBait.TARGET_TERMS.value,
        memory_source_type=MemorySourceType.TRANSACTION_SUMMARY.value,
        confidence_level=ConfidenceLevel.MEDIUM.value,
    )
    poison = decode_poison_action(action, user, memory_id="poison_test")
    poisoned = store.clone_with_poison(poison)
    assert poisoned.poison_count == 1
    assert store.poison_count == 0
    assert len(poisoned) == store.clean_entry_count + 1


def test_reset_to_clean(store: FinanceMemoryStore) -> None:
    user = load_user()
    action = PoisonAction(
        target_fact="rent_amount",
        corrupted_value_strategy=CorruptedValueStrategy.DECREASE_20_PERCENT.value,
        framing_style=FramingStyle.OVERT.value,
        retrieval_bait=RetrievalBait.NONE.value,
        memory_source_type=MemorySourceType.USER_NOTE.value,
        confidence_level=ConfidenceLevel.HIGH.value,
    )
    poison = decode_poison_action(action, user)
    store.add_poison_memory(poison)
    assert store.poison_count == 1
    store.reset_to_clean()
    assert store.poison_count == 0
    assert len(store) == store.clean_entry_count
