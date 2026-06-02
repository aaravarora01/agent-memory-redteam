"""Configuration for finance GRPO reward/evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FinanceGRPOEnvConfig:
    target_facts: tuple[str, ...] = (
        "rent_amount",
        "rent_merchant",
        "paycheck_amount",
        "subscription_amount_spotify",
        "budget_goal",
        "rent_account",
    )
    samples_per_fact: int = 64
    mode: str = "tool_optional"
    reward_mode: str = "sparse"  # sparse | shaped
    success_signal: str = "scorer"  # scorer | judge | hybrid
    k: int = 5
    seed: int = 0
    memory_path: str | Path | None = None
    retriever_backend: Optional[str] = None

    target_model_name_or_path: Optional[str] = None
    target_request_timeout: float = 300.0
    target_max_retries: int = 6

    judge_model: Optional[str] = None
    judge_request_timeout: float = 300.0
    judge_max_retries: int = 6

    invalid_action_reward: float = 0.0
    format_reward_weight: float = 0.0
    reward_max_concurrent: int = 1
