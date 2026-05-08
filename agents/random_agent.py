"""Random-action agent — useful as a baseline opponent and for sanity tests."""

from __future__ import annotations

from typing import Optional

import numpy as np

from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
