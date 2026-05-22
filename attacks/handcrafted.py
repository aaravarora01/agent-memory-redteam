"""10 hand-crafted payloads at 3 stealth levels. Filled in at plan §2.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Payload:
    id: str
    text: str
    stealth: str  # "A" overt | "B" narrative | "C" indirect
    target_tasks: List[str]


PAYLOADS: List[Payload] = []
