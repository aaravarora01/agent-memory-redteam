"""Configuration for the PISmith-style persistent-memory adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PISmithMemoryEnvConfig:
    """Runtime knobs for `PersistentMemoryAttackReward`.

    The names mirror PISmith's benchmark configs where practical, while keeping
    this adapter independent of TRL/PISmith imports.
    """

    task_ids: tuple[str, ...] = (
        "T1_brand_hijack",
        "T2_tool_misdirection",
        "T3_fact_corruption",
        "T4_refusal_bypass",
    )
    episodes_per_sample: int = 1
    k: int = 5
    seed: int = 0
    corpus_path: str | Path | None = None

    # None means defer to QWEN_MODEL / LLM_MODEL from .env.
    target_model_name_or_path: Optional[str] = None
    target_model_temperature: float = 0.2
    target_model_max_tokens: int = 256
    target_request_timeout: float = 300.0

    reward_mode: str = "terminal"
    success_reward_weight: float = 1.0
    retrieval_reward_weight: float = 0.25
    format_reward_weight: float = 0.0
    require_prompt_tags: bool = True
