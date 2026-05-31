"""Unit tests for poison action decoder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

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


def test_overt_poison_contains_wrong_value_not_true_in_body() -> None:
    user = load_user()
    action = PoisonAction(
        target_fact="rent_amount",
        corrupted_value_strategy=CorruptedValueStrategy.REPLACE_WITH_PLAUSIBLE_WRONG_VALUE.value,
        framing_style=FramingStyle.OVERT.value,
        retrieval_bait=RetrievalBait.TARGET_TERMS.value,
        memory_source_type=MemorySourceType.TRANSACTION_SUMMARY.value,
        confidence_level=ConfidenceLevel.HIGH.value,
    )
    entry = decode_poison_action(action, user)
    assert entry.is_poison
    assert entry.poison_value == 1500
    assert "1500" in entry.text
    assert entry.true_value == 1850
