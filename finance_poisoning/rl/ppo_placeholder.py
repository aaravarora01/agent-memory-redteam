"""Policy interface and epsilon-greedy baseline (PPO-compatible stub)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..env.schemas import PoisonAction
from .action_space import PoisonActionSpace


class Policy(Protocol):
    def act(self, obs: dict) -> PoisonAction: ...


@dataclass
class EpsilonGreedyPolicy:
    action_space: PoisonActionSpace = field(default_factory=PoisonActionSpace)
    epsilon: float = 0.1
    seed: int = 0
    _rng: random.Random = field(init=False)
    _best_actions: dict[str, PoisonAction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, obs: dict) -> PoisonAction:
        key = obs.get("target_fact_id", "default")
        if self._rng.random() < self.epsilon or key not in self._best_actions:
            return self.action_space.sample(self._rng)
        return self._best_actions[key]

    def update(self, obs: dict, action: PoisonAction, reward: float) -> None:
        key = obs.get("target_fact_id", "default")
        if reward > 0:
            self._best_actions[key] = action


@dataclass
class UpdateStats:
    mean_reward: float = 0.0
    n_samples: int = 0


class PPOSparseTrainerStub:
    """Placeholder matching parent rl/ppo_sparse.py interface for future wiring."""

    def __init__(self, policy: Policy, lr: float = 3e-4):
        self.policy = policy
        self.lr = lr

    def act(self, obs: dict) -> tuple[PoisonAction, float, float]:
        action = self.policy.act(obs)
        return action, 0.0, 0.0

    def update(self, batch: list) -> UpdateStats:
        if not batch:
            return UpdateStats()
        rewards = [getattr(s, "reward", s.get("reward", 0.0)) for s in batch]
        return UpdateStats(
            mean_reward=sum(rewards) / len(rewards),
            n_samples=len(rewards),
        )
