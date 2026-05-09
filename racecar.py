"""
Thin wrapper around the default PyBullet `racecar.urdf`.

Knows how to:
    * load itself at a given pose
    * apply normalized (steer, drive_velocity) commands
    * report kinematic + sensor state for the env's observation builder

Race-specific logic (lap counting, rewards, etc.) lives in `env.py`, not here,
so this class can be reused for non-RL purposes (e.g. manual driving).
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

        self.steer_joints, self.drive_joints, self.rear_joints = self._discover_joints()

        # Tunable physics — exposed as instance state so debug sliders, a
        # curriculum callback, or domain randomization can mutate them at
        # runtime without touching CAR_CONFIG.
        self.max_drive_torque = float(self._cfg["max_drive_torque"])
        self.max_brake_torque = float(self._cfg["max_brake_torque"])
        self.drive_kp = float(self._cfg["drive_kp"])
        self.vel_target_scale = float(self._cfg["vel_target_scale"])
        self.max_steer = float(self._cfg["max_steer"])
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

    # ── Joint discovery ─────────────────────────────────────────────────
    def _discover_joints(self) -> Tuple[List[int], List[int], List[int]]:
        n = p.getNumJoints(self.body, physicsClientId=self.client)
        steer, drive, rear = [], [], []
        steer_pat = CAR_JOINT_PATTERNS["steer"]
        drive_pat = CAR_JOINT_PATTERNS["drive"]
        rear_pat = CAR_JOINT_PATTERNS.get("rear", [])
        for j in range(n):
            info = p.getJointInfo(self.body, j, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            if any(pat in name for pat in steer_pat):
                steer.append(j)
            if any(pat == name for pat in drive_pat):
                drive.append(j)
            if any(pat == name for pat in rear_pat):
                rear.append(j)
        return steer, drive, rear

    def _tint(self, rgba: Tuple[float, float, float, float]) -> None:
        n = p.getNumJoints(self.body, physicsClientId=self.client)
        for link in range(-1, n):
            try:
                p.changeVisualShape(self.body, link, rgbaColor=list(rgba),
                                    physicsClientId=self.client)
            except Exception:
                pass

    # ── Control ──────────────────────────────────────────────────────────
    def apply_action(self, steer: float, vel_cmd: float) -> None:
        """Steer and velocity command, both in [-1, 1].

        steer    → steering angle target (POSITION_CONTROL, real cars work
                   this way: you set the wheel angle, not the torque).
        vel_cmd  → desired drive-wheel angular velocity (rad/s) at full
                   command. A custom PD computes the torque needed to track
                   it, then clamps to:
                       [+max_drive_torque, -max_brake_torque]
                   (asymmetric — brakes can be stronger than the motor).
                   `vel_target_scale` is set high enough that PD saturates
                   the torque clamp at extreme commands, so there's no
                   real top-speed cap.
        """
        steer = float(np.clip(steer, -1.0, 1.0))
        vel_cmd = float(np.clip(vel_cmd, -1.0, 1.0))
        s_target = steer * self.max_steer
        v_target = vel_cmd * self.vel_target_scale

        # Steering: position control.
        for j in self.steer_joints:
            p.setJointMotorControl2(
                self.body, j, p.POSITION_CONTROL,
                targetPosition=s_target, force=self.steer_force,
                physicsClientId=self.client,
            )

        # Drive wheels: PD on velocity → torque, asymmetric clamp.
        for j in self.drive_joints:
            v_curr = p.getJointState(self.body, j,
                                     physicsClientId=self.client)[1]
            torque = self.drive_kp * (v_target - v_curr)
            if torque >= 0.0:
                torque = min(torque, self.max_drive_torque)
            else:
                torque = max(torque, -self.max_brake_torque)
            p.setJointMotorControl2(
                self.body, j, p.TORQUE_CONTROL,
                force=torque,
                physicsClientId=self.client,
            )

    def set_max_drive_torque(self, torque: float) -> None:
        self.max_drive_torque = float(torque)

    def set_max_brake_torque(self, torque: float) -> None:
        self.max_brake_torque = float(torque)

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

    # ── State ────────────────────────────────────────────────────────────────────────
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

    # ── Joint sensors (for the observation builder) ─────────────────────
    def steering_state(self) -> Tuple[float, float]:
        """Return (steering_angle, steering_rate) averaged across steer joints."""
        if not self.steer_joints:
            return 0.0, 0.0
        states = [p.getJointState(self.body, j, physicsClientId=self.client)
                  for j in self.steer_joints]
        angle = float(np.mean([s[0] for s in states]))
        rate = float(np.mean([s[1] for s in states]))
        return angle, rate

    def rear_wheel_velocity(self) -> float:
        """Mean angular velocity (rad/s) of the rear drive wheels."""
        joints = self.rear_joints if self.rear_joints else self.drive_joints
        if not joints:
            return 0.0
        states = [p.getJointState(self.body, j, physicsClientId=self.client)
                  for j in joints]
        return float(np.mean([s[1] for s in states]))

    def forward_unit_vector(self) -> np.ndarray:
        """Car's local +x axis expressed in the world frame (the heading)."""
        _, orn = p.getBasePositionAndOrientation(self.body,
                                                  physicsClientId=self.client)
        m = p.getMatrixFromQuaternion(orn)
        return np.array([m[0], m[3], m[6]], dtype=np.float32)

    def up_z(self) -> float:
        """Z component (in world) of the car's local +z axis.

        Equals 1 when level, 0 when on its side, -1 when fully upside down.
        Used for flip detection.
        """
        _, orn = p.getBasePositionAndOrientation(self.body,
                                                  physicsClientId=self.client)
        m = p.getMatrixFromQuaternion(orn)
        return float(m[8])
