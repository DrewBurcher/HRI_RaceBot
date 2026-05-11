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


def _load_vecnormalize(path: str):
    """Load a VecNormalize pickle for inference (training=False, no reward norm).

    SB3's VecNormalize.load needs a venv to wrap; we hand it a tiny dummy
    single-env VecEnv since we only use it to normalize observations.
    """
    if not os.path.exists(path):
        return None
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _Dummy(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            return np.zeros(14, dtype=np.float32), {}
        def step(self, a):
            return np.zeros(14, dtype=np.float32), 0.0, True, False, {}

    venv = DummyVecEnv([lambda: _Dummy()])
    vn = VecNormalize.load(path, venv)
    vn.training = False
    vn.norm_reward = False
    return vn


class RLAgent(BaseAgent):
    """Loads an SB3 model from disk on first use.

    If `vecnormalize_path` is provided and exists, observations are
    normalized with the saved running stats before being passed to the
    policy — matches what the model saw during training.
    """

    name = "rl"

    def __init__(self, model_path: str, algo: str = "ppo",
                 deterministic: bool = True,
                 vecnormalize_path: Optional[str] = None):
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.algo = algo
        self.deterministic = deterministic
        self.vecnormalize_path = vecnormalize_path
        self._model = None
        self._vn = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _load_sb3(self.model_path, self.algo)
        if self._vn is None and self.vecnormalize_path:
            self._vn = _load_vecnormalize(self.vecnormalize_path)

    def act(self, observation: Optional[np.ndarray]) -> np.ndarray:
        if observation is None:
            return np.zeros(2, dtype=np.float32)
        self._ensure_loaded()
        obs = np.asarray(observation, dtype=np.float32)
        if self._vn is not None:
            obs = self._vn.normalize_obs(obs[None, :])[0]
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
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
