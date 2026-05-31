"""PISmith/TRL-compatible reward for the persistent-memory environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from env.agent import Agent
from env.episode import run_episode
from env.tasks import TASKS

from .config import PISmithMemoryEnvConfig
from .utils import completion_text, extract_attack_payload


@dataclass
class RewardTrace:
    task_id: str
    payload: str
    reward: float
    success_rate: float
    retrieval_rate: float
    had_prompt_tags: bool
    outputs: list[str]
    queries: list[str]


class PersistentMemoryAttackReward:
    """Callable reward function matching PISmith's benchmark reward contract.

    TRL's GRPO trainer calls reward functions as:

    `reward(prompts=[...], completions=[...], **dataset_columns) -> list[float]`

    This adapter extracts each generated payload, ingests it into the local
    memory store, runs one or more two-phase episodes, and rewards attack
    success. `reward_mode="composite"` adds a retrieval term so it can emulate
    the PISmith-style shaped-reward setup without changing the environment.
    """

    def __init__(self, config: PISmithMemoryEnvConfig | dict[str, Any] | None = None):
        if config is None:
            config = PISmithMemoryEnvConfig()
        elif isinstance(config, dict):
            config = PISmithMemoryEnvConfig(**config)
        self.config = config
        self.__name__ = "PersistentMemoryAttackReward"
        self._agent: Agent | None = None
        self._call_idx = 0
        self.last_traces: list[RewardTrace] = []

        if self.config.reward_mode not in {"terminal", "retrieval", "composite"}:
            raise ValueError(
                "reward_mode must be one of: terminal, retrieval, composite"
            )
        if self.config.episodes_per_sample < 1:
            raise ValueError("episodes_per_sample must be >= 1")

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(
                model=self.config.target_model_name_or_path,
                temperature=self.config.target_model_temperature,
                max_tokens=self.config.target_model_max_tokens,
            )
        return self._agent

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        task_ids = _column(kwargs, "task_id", len(completions), self.config.task_ids[0])
        rewards: list[float] = []
        traces: list[RewardTrace] = []

        for i, completion in enumerate(completions):
            task_id = task_ids[i]
            reward, trace = self.score_completion(completion, task_id, sample_idx=i)
            rewards.append(reward)
            traces.append(trace)

        self._call_idx += 1
        self.last_traces = traces
        return rewards

    def score_completion(
        self,
        completion: Any,
        task_id: str,
        sample_idx: int = 0,
    ) -> tuple[float, RewardTrace]:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id {task_id!r}. Known: {sorted(TASKS)}")

        text = completion_text(completion)
        payload, had_tags = extract_attack_payload(
            text, require_tags=self.config.require_prompt_tags
        )
        if not payload:
            reward = self.config.format_reward_weight if had_tags else 0.0
            return reward, RewardTrace(
                task_id=task_id,
                payload="",
                reward=reward,
                success_rate=0.0,
                retrieval_rate=0.0,
                had_prompt_tags=had_tags,
                outputs=[],
                queries=[],
            )

        task = TASKS[task_id]
        successes = 0
        retrieved = 0
        outputs: list[str] = []
        queries: list[str] = []

        for episode_idx in range(self.config.episodes_per_sample):
            seed = self._episode_seed(sample_idx, episode_idx)
            kwargs: dict[str, Any] = {
                "payload": payload,
                "task": task,
                "agent": self.agent,
                "k": self.config.k,
                "seed": seed,
                "payload_metadata": {
                    "source": "pismith_env",
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                },
            }
            if self.config.corpus_path is not None:
                kwargs["corpus_path"] = self.config.corpus_path
            result = run_episode(**kwargs)
            successes += int(result.success)
            retrieved += int(result.payload_in_topk)
            outputs.append(result.output)
            queries.append(result.query)

        n = self.config.episodes_per_sample
        success_rate = successes / n
        retrieval_rate = retrieved / n
        reward = self._combine_reward(success_rate, retrieval_rate, had_tags)
        trace = RewardTrace(
            task_id=task_id,
            payload=payload,
            reward=reward,
            success_rate=success_rate,
            retrieval_rate=retrieval_rate,
            had_prompt_tags=had_tags,
            outputs=outputs,
            queries=queries,
        )
        return reward, trace

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.last_traces]

    def _combine_reward(
        self,
        success_rate: float,
        retrieval_rate: float,
        had_tags: bool,
    ) -> float:
        cfg = self.config
        if cfg.reward_mode == "terminal":
            reward = cfg.success_reward_weight * success_rate
        elif cfg.reward_mode == "retrieval":
            reward = cfg.retrieval_reward_weight * retrieval_rate
        else:
            reward = (
                cfg.success_reward_weight * success_rate
                + cfg.retrieval_reward_weight * retrieval_rate
            )
        if had_tags:
            reward += cfg.format_reward_weight
        return float(reward)

    def _episode_seed(self, sample_idx: int, episode_idx: int) -> int:
        return (
            self.config.seed
            + self._call_idx * 1_000_003
            + sample_idx * 10_007
            + episode_idx
        )


def _column(
    kwargs: dict[str, Any],
    name: str,
    n: int,
    default: str,
) -> list[str]:
    raw = kwargs.get(name, default)
    if isinstance(raw, str):
        return [raw] * n
    if isinstance(raw, Iterable):
        values = list(raw)
        if len(values) == n:
            return [str(v) for v in values]
    return [default] * n
