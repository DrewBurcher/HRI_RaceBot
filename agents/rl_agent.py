"""Wrappers around stable-baselines3 policies.

Two flavors:
    * `RLAgent`        — lazy SB3 model loader (eval-only)
    * `FrozenRLAgent`  — holds an already-loaded SB3 model in memory

Both implement the BaseAgent.act() contract so the env doesn't have to know
what's behind the wheel.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from agents.base import BaseAgent


def _load_sb3(path: str, algo: str):
    from stable_baselines3 import PPO, SAC
    cls = PPO if algo == "ppo" else SAC
    return cls.load(path, device="cpu")


class RLAgent(BaseAgent):
    """Loads an SB3 model from disk on first use."""

    name = "rl"

    def __init__(self, model_path: str, algo: str = "ppo",
                 deterministic: bool = True):
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.algo = algo
        self.deterministic = deterministic
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _load_sb3(self.model_path, self.algo)

    def act(self, observation: Optional[np.ndarray]) -> np.ndarray:
        if observation is None:
            return np.zeros(2, dtype=np.float32)
        self._ensure_loaded()
        action, _ = self._model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


class FrozenRLAgent(BaseAgent):
    """Adapter for an in-memory SB3 model (used during self-play training)."""

    name = "frozen_rl"

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, observation: Optional[np.ndarray]) -> np.ndarray:
        if observation is None or self.model is None:
            return np.zeros(2, dtype=np.float32)
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)
