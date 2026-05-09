"""Base agent contract.

Intentionally tiny so anything callable can be turned into an agent. The
env only ever calls `.act(observation) -> action` on whatever it's handed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class BaseAgent:
    """Abstract policy used by `SingleAgentRaceWrapper` and friends."""

    name: str = "base"

    def reset(self) -> None:
        """Hook called at the start of each episode (optional)."""

    def act(self, observation: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        """Free any held resources (optional)."""
