"""Agent classes for HRI_RaceBot.

All agents share a single tiny interface (`act(obs) -> np.ndarray`). New agent
types (e.g. a model-predictive controller, an imitation-learning policy) drop
into this package and are picked up automatically by the registry.
"""

from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent, FrozenRLAgent
from agents.human_agent import HumanKeyboardAgent
from agents.registry import build_agent, list_agents

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "RLAgent",
    "FrozenRLAgent",
    "HumanKeyboardAgent",
    "build_agent",
    "list_agents",
]
