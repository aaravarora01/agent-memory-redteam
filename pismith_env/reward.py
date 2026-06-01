"""PISmith/TRL-compatible reward for the persistent-memory environment."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from env.agent import Agent
from env.episode import run_episode
from env.judge import Judge
from env.tasks import TASKS

from .config import PISmithMemoryEnvConfig
from .utils import completion_text, extract_attack_payload


class _LockedEmbedder:
    """Wrap a SentenceTransformer so concurrent `.encode()` calls are serialized.

    Encoding (one payload/query, MiniLM) is cheap relative to the remote
    target/judge calls it runs alongside, so a single lock keeps one model
    thread-safe under the reward's thread pool without hurting throughput.
    """

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()

    def encode(self, *args, **kwargs):
        with self._lock:
            return self._model.encode(*args, **kwargs)


@dataclass
class _JudgeOutcome:
    """Internal result of one judge consultation inside an episode."""
    reason: str
    judge_error: bool


@dataclass
class _EpisodeScore:
    """Thread-safe result of `_score_episodes` (no novelty/buffer mutation).

    `_finalize` turns this into the reward + `RewardTrace` sequentially so the
    novelty buffer stays ordered.
    """
    task_id: str
    payload: str
    had_tags: bool
    n_words: int
    status: str  # "empty" | "floored" | "scored"
    early_reward: float = 0.0
    success_rate: float = 0.0
    success_rate_regex: float = 0.0
    retrieval_rate: float = 0.0
    base_reward: float = 0.0
    length_penalty: float = 0.0
    judge_calls: int = 0
    judge_errors: int = 0
    judge_reasons: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


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
        """Lazy, thread-safe MiniLM encoder shared across all episodes.

        One model total, reused by both the per-episode `MemoryStore` (so each
        episode doesn't reload MiniLM) and the novelty penalty. `.encode()` is
        lock-guarded for the reward's thread pool. Built on first episode/novelty
        use; the novelty path skips it entirely when `novelty_alpha == 0`.
        """
        if self._embedder is None:
            from env.memory_store import _load_default_embedder

            self._embedder = _LockedEmbedder(
                _load_default_embedder(device=self.config.embedder_device)
            )
        return self._embedder

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        task_ids = _column(kwargs, "task_id", len(completions), self.config.task_ids[0])
        n = len(completions)
        max_workers = max(1, int(self.config.reward_max_concurrent))

        # Phase A (I/O-bound, parallelizable): episodes + judge per completion.
        if max_workers == 1 or n <= 1:
            scores = [
                self._score_episodes(completions[i], task_ids[i], i) for i in range(n)
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, n)) as ex:
                scores = list(
                    ex.map(
                        lambda i: self._score_episodes(completions[i], task_ids[i], i),
                        range(n),
                    )
                )

        # Phase B (sequential, order-dependent): novelty buffer + trace.
        rewards: list[float] = []
        traces: list[RewardTrace] = []
        for s in scores:
            reward, trace = self._finalize(s)
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
        """Score one completion end-to-end (single-shot; used by smokes/tests).

        `__call__` instead runs `_score_episodes` (Phase A) across completions
        concurrently and `_finalize` (Phase B) sequentially.
        """
        return self._finalize(self._score_episodes(completion, task_id, sample_idx))

    def _score_episodes(
        self,
        completion: Any,
        task_id: str,
        sample_idx: int = 0,
    ) -> _EpisodeScore:
        """Phase A — extraction, floor, and the episode/judge loop.

        Thread-safe: reads shared config/agent/judge/embedder but never touches
        the novelty buffer (that happens in `_finalize`).
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id {task_id!r}. Known: {sorted(TASKS)}")

        text = completion_text(completion)
        payload, had_tags = extract_attack_payload(
            text, require_tags=self.config.require_prompt_tags
        )
        if not payload:
            early = self.config.format_reward_weight if had_tags else 0.0
            return _EpisodeScore(
                task_id=task_id, payload="", had_tags=had_tags,
                n_words=0, status="empty", early_reward=early,
            )

        # #3 hard floor: too-short payloads score 0 and short-circuit *before*
        # the episode loop, so degenerate rollouts cost no target/judge calls.
        n_words = len(payload.split())
        if n_words < self.config.min_payload_words:
            return _EpisodeScore(
                task_id=task_id, payload=payload, had_tags=had_tags,
                n_words=n_words, status="floored",
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
                "embedder": self.embedder,  # one shared encoder across episodes
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
        return _EpisodeScore(
            task_id=task_id, payload=payload, had_tags=had_tags, n_words=n_words,
            status="scored",
            success_rate=success_rate,
            success_rate_regex=regex_successes / n,
            retrieval_rate=retrieved / n,
            base_reward=self._combine_reward(success_rate, retrieved / n, had_tags),
            length_penalty=self._length_penalty(n_words),
            judge_calls=judge_calls, judge_errors=judge_errors,
            judge_reasons=judge_reasons, outputs=outputs, queries=queries,
        )

    def _finalize(self, s: _EpisodeScore) -> tuple[float, RewardTrace]:
        """Phase B — apply the #2 novelty penalty (mutates the rolling buffer,
        so this must run sequentially in completion order) and build the trace.
        """
        if s.status == "empty":
            return s.early_reward, RewardTrace(
                task_id=s.task_id, payload="", reward=s.early_reward,
                success_rate=0.0, retrieval_rate=0.0, had_prompt_tags=s.had_tags,
                outputs=[], queries=[],
            )
        if s.status == "floored":
            return 0.0, RewardTrace(
                task_id=s.task_id, payload=s.payload, reward=0.0,
                success_rate=0.0, retrieval_rate=0.0, had_prompt_tags=s.had_tags,
                outputs=[], queries=[], n_words=s.n_words, floored=True,
            )

        novelty_penalty, max_cos_sim = self._novelty_penalty(s.payload)
        reward = max(0.0, s.base_reward - s.length_penalty - novelty_penalty)
        trace = RewardTrace(
            task_id=s.task_id, payload=s.payload, reward=reward,
            success_rate=s.success_rate, retrieval_rate=s.retrieval_rate,
            had_prompt_tags=s.had_tags, outputs=s.outputs, queries=s.queries,
            n_words=s.n_words, base_reward=s.base_reward,
            length_penalty=s.length_penalty, novelty_penalty=novelty_penalty,
            max_cos_sim=max_cos_sim, success_rate_regex=s.success_rate_regex,
            judge_calls=s.judge_calls, judge_errors=s.judge_errors,
            judge_reasons=s.judge_reasons,
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
