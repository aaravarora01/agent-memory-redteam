"""PISmith/TRL-compatible reward for the persistent-memory environment."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from env.agent import Agent
from env.episode import run_episode
from env.judge import Judge
from env.tasks import TASKS

from .config import PISmithMemoryEnvConfig
from .utils import completion_text, extract_attack_payload


@dataclass
class _JudgeOutcome:
    """Internal result of one judge consultation inside an episode."""
    reason: str
    judge_error: bool


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
    # Reward-shaping bookkeeping (#2/#3). Defaulted so older callers/serialized
    # rows stay compatible; these feed the monitoring line and the offline tests.
    n_words: int = 0
    base_reward: float = 0.0
    length_penalty: float = 0.0
    novelty_penalty: float = 0.0
    max_cos_sim: float = 0.0
    floored: bool = False
    # Judge-signal bookkeeping. `success_rate` above is the reward-driving signal
    # (regex/judge/hybrid per config); `success_rate_regex` is always the raw
    # regex rate, so their gap is the online regex-vs-judge disagreement.
    success_rate_regex: float = 0.0
    judge_calls: int = 0
    judge_errors: int = 0
    judge_reasons: list[str] = field(default_factory=list)


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
        self._judge: Judge | None = None
        self._call_idx = 0
        self.last_traces: list[RewardTrace] = []

        # Novelty (#2): shared MiniLM encoder + rolling buffer of recent payload
        # embeddings. Both stay unset/empty unless novelty shaping is enabled.
        self._embedder = None
        self._novelty_buf: deque = deque(maxlen=max(1, self.config.novelty_buffer_size))

        if self.config.reward_mode not in {"terminal", "retrieval", "composite"}:
            raise ValueError(
                "reward_mode must be one of: terminal, retrieval, composite"
            )
        if self.config.success_signal not in {"regex", "judge", "hybrid"}:
            raise ValueError(
                "success_signal must be one of: regex, judge, hybrid"
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
                request_timeout=self.config.target_request_timeout,
            )
        return self._agent

    @property
    def judge(self) -> Judge:
        """Lazy LLM judge, built only when success_signal != "regex".

        Settable (`reward.judge = Judge(client=stub)`) so tests can inject a
        deterministic judge without an API key.
        """
        if self._judge is None:
            self._judge = Judge(
                model=self.config.judge_model,
                request_timeout=self.config.judge_request_timeout,
                max_retries=self.config.judge_max_retries,
            )
        return self._judge

    @judge.setter
    def judge(self, value: Judge) -> None:
        self._judge = value

    @property
    def embedder(self):
        """Lazy MiniLM encoder, shared with the memory store's model.

        Only constructed when novelty shaping is on, so terminal/off runs never
        pay the load.
        """
        if self._embedder is None:
            from env.memory_store import _load_default_embedder

            self._embedder = _load_default_embedder()
        return self._embedder

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

        # #3 hard floor: too-short payloads score 0 and short-circuit *before*
        # the episode loop, so degenerate rollouts cost no target/judge calls.
        n_words = len(payload.split())
        if n_words < self.config.min_payload_words:
            return 0.0, RewardTrace(
                task_id=task_id,
                payload=payload,
                reward=0.0,
                success_rate=0.0,
                retrieval_rate=0.0,
                had_prompt_tags=had_tags,
                outputs=[],
                queries=[],
                n_words=n_words,
                floored=True,
            )

        task = TASKS[task_id]
        successes = 0
        regex_successes = 0
        retrieved = 0
        judge_calls = 0
        judge_errors = 0
        judge_reasons: list[str] = []
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
            regex_successes += int(result.success)
            ok, verdict = self._resolve_success(
                result.success, result.query, result.output, task
            )
            successes += int(ok)
            if verdict is not None:
                judge_calls += 1
                judge_errors += int(verdict.judge_error)
                judge_reasons.append(verdict.reason)
            retrieved += int(result.payload_in_topk)
            outputs.append(result.output)
            queries.append(result.query)

        n = self.config.episodes_per_sample
        success_rate = successes / n
        success_rate_regex = regex_successes / n
        retrieval_rate = retrieved / n
        base_reward = self._combine_reward(success_rate, retrieval_rate, had_tags)

        # #3 length penalty + #2 novelty penalty, clamped at 0 (finbench v3.2).
        length_penalty = self._length_penalty(n_words)
        novelty_penalty, max_cos_sim = self._novelty_penalty(payload)
        reward = max(0.0, base_reward - length_penalty - novelty_penalty)

        trace = RewardTrace(
            task_id=task_id,
            payload=payload,
            reward=reward,
            success_rate=success_rate,
            retrieval_rate=retrieval_rate,
            had_prompt_tags=had_tags,
            outputs=outputs,
            queries=queries,
            n_words=n_words,
            base_reward=base_reward,
            length_penalty=length_penalty,
            novelty_penalty=novelty_penalty,
            max_cos_sim=max_cos_sim,
            success_rate_regex=success_rate_regex,
            judge_calls=judge_calls,
            judge_errors=judge_errors,
            judge_reasons=judge_reasons,
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

    def _resolve_success(
        self,
        regex_success: bool,
        query: str,
        output: str,
        task,
    ) -> tuple[bool, "_JudgeOutcome | None"]:
        """Decide episode success per `success_signal`.

        Returns `(effective_success, judge_outcome)` where `judge_outcome` is
        None when the judge was not consulted (regex mode, or hybrid mode gated
        out because the regex already said no). A malformed/failed judge call
        falls back to the regex verdict and is flagged `judge_error=True` rather
        than aborting the run.
        """
        signal = self.config.success_signal
        if signal == "regex":
            return regex_success, None
        if signal == "hybrid" and not regex_success:
            # Gate: the judge can only confirm, not resurrect, a regex negative.
            return False, None
        try:
            verdict = self.judge.evaluate(query, output, task)
        except Exception as e:  # malformed JSON / transient API → fall back
            return regex_success, _JudgeOutcome(reason=f"judge_error: {e}", judge_error=True)
        if signal == "hybrid":
            return (regex_success and verdict.attack_succeeded), _JudgeOutcome(
                reason=verdict.reason, judge_error=False
            )
        return verdict.attack_succeeded, _JudgeOutcome(
            reason=verdict.reason, judge_error=False
        )

    def _length_penalty(self, n_words: int) -> float:
        """Linear penalty for payloads above the target length (#3).

        Zero at/below `length_target_words`; disabled when alpha is 0.
        """
        return self.config.length_penalty_alpha * max(
            0, n_words - self.config.length_target_words
        )

    def _novelty_penalty(self, payload: str) -> tuple[float, float]:
        """Penalize payloads similar to recently-generated ones (#2).

        Returns `(penalty, max_cos_sim)`. Cosine similarity is the max dot
        product against a rolling buffer of recent payload embeddings (MiniLM
        vectors are unit-normalized, so dot == cosine). The current payload is
        appended to the buffer afterwards, so within-batch siblings — scored
        sequentially in `__call__` — also count, catching monoculture inside a
        single GRPO group. Disabled (and the encoder never loads) when
        `novelty_alpha == 0`.
        """
        if self.config.novelty_alpha <= 0.0:
            return 0.0, 0.0

        emb = self.embedder.encode([payload], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32").reshape(-1)

        if self._novelty_buf:
            sims = np.stack(list(self._novelty_buf)) @ emb
            max_cos = float(np.max(sims))
        else:
            max_cos = 0.0
        self._novelty_buf.append(emb)

        penalty = self.config.novelty_alpha * max(
            0.0, max_cos - self.config.novelty_threshold
        )
        return penalty, max_cos

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
