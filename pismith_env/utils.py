"""Small utilities shared by the PISmith adapter."""

from __future__ import annotations

import re
from typing import Any


_PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.IGNORECASE | re.DOTALL)


def completion_text(completion: Any) -> str:
    """Normalize TRL/OpenAI-style completions to plain text."""
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


def extract_attack_payload(text: str, require_tags: bool = True) -> tuple[str, bool]:
    """Extract the generated memory payload from `<prompt>` tags.

    Returns `(payload, had_tags)`. If `require_tags` is false, untagged text is
    accepted as the payload, matching PISmith's optional format-prompt mode.
    """
    match = _PROMPT_TAG_RE.search(text)
    if match:
        return match.group(1).strip(), True
    if require_tags:
        return "", False
    return text.strip(), False
