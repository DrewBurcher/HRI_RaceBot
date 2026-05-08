"""
Thin wrapper around the default PyBullet `racecar.urdf`.

Knows how to:
    * load itself at a given pose
    * apply normalized (steer, throttle) commands
    * report kinematic state for the env's observation builder

Deliberately small — race-specific logic (lap counting, rewards, etc.) lives in
`env.py`, not here, so this class can be reused for non-RL purposes.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from config import CAR_CONFIG, CAR_JOINT_PATTERNS


class RaceCar:
    """Single-car PyBullet wrapper."""

    def __init__(
        self,
        client: int,
        position: Sequence[float],
        orientation: Sequence[float],
        car_id: int = 0,
        color: Tuple[float, float, float, float] | None = None,
    ):
        self.client = client
        self.car_id = car_id
        self._cfg = CAR_CONFIG

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf = self._cfg["urdf"]
        self.body = p.loadURDF(
            urdf,
            basePosition=list(position),
            baseOrientation=list(orientation),
            physicsClientId=self.client,
        )

        self.steer_joints, self.drive_joints = self._discover_joints()

        # Tunable physics — exposed as instance state so debug-mode sliders
        # (or a curriculum callback) can change them at runtime without
        # touching CAR_CONFIG.
        self.max_torque = float(self._cfg["max_torque"])
        self.steer_force = float(self._cfg.get("steer_force", 50.0))

        # PyBullet's default joint motor is a velocity controller that resists
        # any imposed torque. Zero its force on the drive joints so our
        # TORQUE_CONTROL commands aren't fought by the default motor.
        self._disable_default_motors(self.drive_joints)

        if color is not None:
            self._tint(color)

    def _disable_default_motors(self, joint_indices: Sequence[int]) -> None:
        for j in joint_indices:
            p.setJointMotorControl2(
                self.body, j, p.VELOCITY_CONTROL,
                targetVelocity=0.0, force=0.0,
                physicsClientId=self.client,
            )

    # ── Joint discovery ────────────────────────────────────────────────
    def _discover_joints(self) -> Tuple[List[int], List[int]]:
        n = p.getNumJoints(self.body, physicsClientId=self.client)
        steer, drive = [], []
        steer_pat = CAR_JOINT_PATTERNS["steer"]
        drive_pat = CAR_JOINT_PATTERNS["drive"]
        for j in range(n):
            info = p.getJointInfo(self.body, j, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            if any(pat in name for pat in steer_pat):
                steer.append(j)
            if any(pat == name for pat in drive_pat):
                drive.append(j)
        return steer, drive

    def _tint(self, rgba: Tuple[float, float, float, float]) -> None:
        n = p.getNumJoints(self.body, physicsClientId=self.client)
        for link in range(-1, n):
            try:
                p.changeVisualShape(self.body, link, rgbaColor=list(rgba),
                                    physicsClientId=self.client)
            except Exception:
                pass

    # ── Control ─────────────────────────────────────────────────────────
    def apply_action(self, steer: float, throttle: float) -> None:
        """Steer and throttle in [-1, 1].

        Drive wheels are torque-controlled: throttle * max_torque is the
        signed torque applied to each drive wheel. Steering stays
        position-controlled (matches how a real car works — you set the
        wheel angle, not the steering torque).
        """
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))
        s_target = steer * self._cfg["max_steer"]
        drive_torque = throttle * self.max_torque

        for j in self.steer_joints:
            p.setJointMotorControl2(
                self.body, j, p.POSITION_CONTROL,
                targetPosition=s_target, force=self.steer_force,
                physicsClientId=self.client,
            )
        for j in self.drive_joints:
            p.setJointMotorControl2(
                self.body, j, p.TORQUE_CONTROL,
                force=drive_torque,
                physicsClientId=self.client,
            )

    def set_max_torque(self, torque: float) -> None:
        """Update the per-drive-wheel torque (N·m) at full throttle."""
        self.max_torque = float(torque)

    def set_traction(self, lateral_friction: float) -> None:
        """Set lateral friction on every wheel link (drive + steer).

        PyBullet uses link index = joint index for a non-fixed joint, so the
        joints discovered as steer/drive double as the wheel links.
        """
        for j in self.drive_joints + self.steer_joints:
            p.changeDynamics(self.body, j,
                              lateralFriction=float(lateral_friction),
                              physicsClientId=self.client)

    def reset(self, position: Sequence[float], orientation: Sequence[float]) -> None:
        p.resetBasePositionAndOrientation(
            self.body, list(position), list(orientation),
            physicsClientId=self.client)
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.client)
        for j in self.steer_joints + self.drive_joints:
            p.resetJointState(self.body, j, 0.0, 0.0,
                              physicsClientId=self.client)

    # ── State ─────────────────────────────────────────────────────────────────────────
    def get_state(self) -> dict:
        pos, orn = p.getBasePositionAndOrientation(self.body,
                                                    physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.body,
                                              physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn)
        return {
            "position": np.array(pos, dtype=np.float32),
            "orientation_quat": np.array(orn, dtype=np.float32),
            "orientation_euler": np.array(euler, dtype=np.float32),
            "linear_velocity": np.array(lin_vel, dtype=np.float32),
            "angular_velocity": np.array(ang_vel, dtype=np.float32),
        }

    def speed(self) -> float:
        lin_vel, _ = p.getBaseVelocity(self.body, physicsClientId=self.client)
        return float(np.linalg.norm(lin_vel[:2]))
