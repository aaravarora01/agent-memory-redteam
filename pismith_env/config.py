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

    # Success signal driving the reward. "regex" uses task.success_check only
    # (default → unchanged behavior, no judge calls). "judge" calls the LLM
    # judge every episode (faithful but ~2x API cost). "hybrid" calls the judge
    # only when the regex fires and requires both (regex AND judge) — removes
    # regex false positives at cost ~= ASR-fraction of episodes. See env/judge.py.
    success_signal: str = "regex"        # "regex" | "judge" | "hybrid"
    judge_model: Optional[str] = None    # None → defer to JUDGE_MODEL / LLM_MODEL
    judge_request_timeout: float = 60.0  # bump for Modal cold starts
    judge_max_retries: int = 6

    # Reward shaping (ported from the finbench GRPO loop). All default to
    # no-ops so existing callers see byte-identical rewards; set the
    # recommended training values from configs/pismith_memory.yaml for a run.
    #
    # #3 anti-degenerate: kill empty/too-short reward-hacking modes. A payload
    # below the floor scores 0.0 and is short-circuited *before* the episode
    # loop, so it costs no target/judge API calls. The length penalty discourages
    # bloated payloads (less retrievable, easier to flag).
    min_payload_words: int = 0          # >0 enables the hard floor; recommend 8
    length_target_words: int = 60       # only active when length_penalty_alpha > 0
    length_penalty_alpha: float = 0.0   # >0 enables; recommend ~0.01

    # #2 anti-monoculture: penalize payloads that look like recent ones, so the
    # policy can't collapse onto a single template. Cosine similarity against a
    # rolling buffer of recent payload embeddings (shared MiniLM encoder).
    novelty_alpha: float = 0.0          # >0 enables; recommend ~1.0-2.0 (tune in smoke)
    novelty_threshold: float = 0.8      # similarity above this starts costing
    novelty_buffer_size: int = 256      # rolling window of recent payload embeddings
