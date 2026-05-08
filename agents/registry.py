"""Tiny registry so train.py and debug.py can build agents from strings.

Keeping this in one place makes adding new agent types painless: register
them here and they're available everywhere via `build_agent('mpc', ...)`.
"""

from __future__ import annotations

from typing import Callable, Dict

from agents.base import BaseAgent
from agents.human_agent import HumanKeyboardAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import FrozenRLAgent, RLAgent


_REGISTRY: Dict[str, Callable[..., BaseAgent]] = {
    "random": RandomAgent,
    "human": HumanKeyboardAgent,
    "rl": RLAgent,
    "frozen_rl": FrozenRLAgent,
}


def build_agent(kind: str, **kwargs) -> BaseAgent:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown agent kind: {kind}. "
                         f"Available: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def list_agents() -> list:
    return sorted(_REGISTRY)


def register(kind: str, factory: Callable[..., BaseAgent]) -> None:
    """Plug-in hook — register additional agent kinds at runtime."""
    _REGISTRY[kind] = factory
