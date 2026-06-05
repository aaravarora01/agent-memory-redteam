"""Parsing helpers for finance GRPO completions."""

from __future__ import annotations

import json
import re
from typing import Any

from finance_poisoning.env.schemas import (
    ConfidenceLevel,
    CorruptedValueStrategy,
    FramingStyle,
    MemorySourceType,
    PoisonAction,
    RetrievalBait,
    TargetFact,
)


_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    return "" if completion is None else str(completion)


def parse_poison_action(completion: Any, expected_fact: str | None = None) -> tuple[PoisonAction | None, bool, str]:
    """Return `(action, had_tags, error)` from an attacker completion."""
    text = completion_text(completion)
    match = _ACTION_TAG_RE.search(text)
    had_tags = bool(match)
    raw = match.group(1).strip() if match else text.strip()
    if not raw:
        return None, had_tags, "empty completion"

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None, had_tags, "no JSON object found"
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            return None, had_tags, f"invalid JSON: {e}"

    if not isinstance(obj, dict):
        return None, had_tags, "JSON payload is not an object"

    required = {
        "target_fact",
        "corrupted_value_strategy",
        "framing_style",
        "retrieval_bait",
        "memory_source_type",
        "confidence_level",
    }
    missing = sorted(required - set(obj))
    if missing:
        return None, had_tags, f"missing keys: {missing}"

    if expected_fact is not None and obj["target_fact"] != expected_fact:
        return None, had_tags, f"target_fact {obj['target_fact']!r} != expected {expected_fact!r}"

    validators = {
        "target_fact": {v.value for v in TargetFact},
        "corrupted_value_strategy": {v.value for v in CorruptedValueStrategy},
        "framing_style": {v.value for v in FramingStyle},
        "retrieval_bait": {v.value for v in RetrievalBait},
        "memory_source_type": {v.value for v in MemorySourceType},
        "confidence_level": {v.value for v in ConfidenceLevel},
    }
    for key, allowed in validators.items():
        if obj[key] not in allowed:
            return None, had_tags, f"invalid {key}: {obj[key]!r}"

    return PoisonAction(
        target_fact=obj["target_fact"],
        corrupted_value_strategy=obj["corrupted_value_strategy"],
        framing_style=obj["framing_style"],
        retrieval_bait=obj["retrieval_bait"],
        memory_source_type=obj["memory_source_type"],
        confidence_level=obj["confidence_level"],
    ), had_tags, ""
