"""Random policy over structured poison actions."""

from __future__ import annotations

import random

from ..env.schemas import PoisonAction
from .action_space import PoisonActionSpace


class RandomPolicy:
    def __init__(self, action_space: PoisonActionSpace | None = None, seed: int = 0):
        self.action_space = action_space or PoisonActionSpace()
        self.rng = random.Random(seed)

    def act(self, obs: dict | None = None) -> PoisonAction:
        return self.action_space.sample(self.rng)

    def update(self, *args, **kwargs) -> None:
        pass
