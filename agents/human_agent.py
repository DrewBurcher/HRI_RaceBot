"""Keyboard-driven human agent — used by `debug.py`.

Reads WASD (or arrow keys) via PyBullet's keyboard events. Falls back to
`pynput` if PyBullet's keyboard polling isn't available (e.g. headless test).

Bindings:
    W / Up    : forward throttle
    S / Down  : reverse throttle
    A / Left  : steer left
    D / Right : steer right
    SPACE     : brake (zero throttle)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pybullet as p

from agents.base import BaseAgent


class HumanKeyboardAgent(BaseAgent):
    name = "human"

    KEY_W = ord('w')
    KEY_A = ord('a')
    KEY_S = ord('s')
    KEY_D = ord('d')
    KEY_SPACE = ord(' ')

    def __init__(self, client: Optional[int] = None):
        self.client = client
        self._steer = 0.0
        self._throttle = 0.0

    def reset(self) -> None:
        self._steer = 0.0
        self._throttle = 0.0

    def act(self, observation: Optional[np.ndarray]) -> np.ndarray:
        try:
            keys = (p.getKeyboardEvents(physicsClientId=self.client)
                    if self.client is not None else p.getKeyboardEvents())
        except Exception:
            return np.zeros(2, dtype=np.float32)

        # PyBullet special key codes for arrows
        UP, DOWN, LEFT, RIGHT = 65297, 65298, 65295, 65296

        throttle = 0.0
        steer = 0.0
        if self.KEY_W in keys or UP in keys:
            throttle += 1.0
        if self.KEY_S in keys or DOWN in keys:
            throttle -= 1.0
        if self.KEY_A in keys or LEFT in keys:
            steer += 1.0
        if self.KEY_D in keys or RIGHT in keys:
            steer -= 1.0
        if self.KEY_SPACE in keys:
            throttle = 0.0

        # Light smoothing
        self._throttle = 0.6 * self._throttle + 0.4 * throttle
        self._steer = 0.6 * self._steer + 0.4 * steer
        return np.array([self._steer, self._throttle], dtype=np.float32)
