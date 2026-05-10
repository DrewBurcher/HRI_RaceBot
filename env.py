"""
Gymnasium-compatible two-car racing environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from config import (CAR_CONFIG, DR_CONFIG, RACE_CONFIG, REWARD_CONFIG,
                     SIM_CONFIG, TRACK_CONFIG)
from racecar import RaceCar
from track import build_track

AGENT_IDS = [f"car_{i}" for i in range(RACE_CONFIG["num_cars"])]
CAR_COLORS = [
    (0.9, 0.2, 0.2, 1.0),    # red
    (0.2, 0.4, 0.9, 1.0),    # blue
    (0.2, 0.8, 0.3, 1.0),    # green
    (0.9, 0.8, 0.2, 1.0),    # yellow
]

class TwoCarRaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None],
                "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None,
                 num_cars: Optional[int] = None,
                 alternate_lanes: Optional[bool] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.num_cars = num_cars or RACE_CONFIG["num_cars"]
        if self.num_cars > len(AGENT_IDS):
            for k in range(len(AGENT_IDS), self.num_cars):
                AGENT_IDS.append(f"car_{k}")
        self.agent_ids = AGENT_IDS[: self.num_cars]
        self.alternate_lanes = (alternate_lanes if alternate_lanes is not None
                                 else RACE_CONFIG["alternate_lanes"])

        self.client: Optional[int] = None
        self.track: Optional[OvalTrack] = None
        self.cars: Dict[str, RaceCar] = {}

        self._race_index = 0
        self._step_count = 0

        self._prev_progress: Dict[str, float] = {}
        self._lap_progress: Dict[str, float] = {}
        self._flipped: Dict[str, bool] = {}
        self._reward_components: Dict[str, Dict[str, float]] = {}
        self._dr_params: Dict[str, float] = {}
        self._smoothed_action: Dict[str, np.ndarray] = {}

        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                        shape=(2,), dtype=np.float32)
        obs_dim = 2 + 2 + 2 + 1 + 1 + 1 + 1 + 2 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)

    def _connect(self) -> None:
        if self.client is not None:
            try:
                p.getConnectionInfo(self.client)
                return
            except Exception:
                self.client = None
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                        physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=25.0, cameraYaw=0.0, cameraPitch=-65.0,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, SIM_CONFIG["gravity"], physicsClientId=self.client)
        p.setTimeStep(1.0 / SIM_CONFIG["control_freq"],
                       physicsClientId=self.client)

    def reset(self, *, seed: Optional[int] = None, options=None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._connect()

        if self.track is None:
            self.track = build_track(self.client)

        if self.alternate_lanes and self._race_index % 2 == 1:
            lanes = [1, 0] + list(range(2, self.num_cars))
        else:
            lanes = list(range(self.num_cars))

        jitter = self.track.random_start_jitter(self._rng)

        for i, agent_id in enumerate(self.agent_ids):
            pos, quat = self.track.spawn_pose(lanes[i], jitter)
            color = CAR_COLORS[i % len(CAR_COLORS)]
            if agent_id in self.cars:
                self.cars[agent_id].reset(pos, quat)
            else:
                self.cars[agent_id] = RaceCar(self.client, pos, quat,
                                                car_id=i, color=color)

        self._prev_progress = {}
        self._lap_progress = {}
        self._flipped = {a: False for a in self.agent_ids}
        self._reward_components = {a: {} for a in self.agent_ids}
        self._smoothed_action = {a: np.zeros(2, dtype=np.float32)
                                  for a in self.agent_ids}
        self._step_count = 0

        self._dr_params = self._sample_dr_params() if DR_CONFIG.get("enabled") else {}
        if self._dr_params:
            self._apply_dr_params(self._dr_params)

        for a, car in self.cars.items():
            x, y, _ = car.get_state()["position"]
            prog = self.track.centerline_progress(x, y)
            self._prev_progress[a] = prog
            self._lap_progress[a] = 0.0

        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)

        obs = self._build_obs_all()
        info = {a: {"lane": lanes[i], "jitter": jitter,
                    "dr_params": dict(self._dr_params)}
                for i, a in enumerate(self.agent_ids)}
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, float],
                        Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        if self.client is None:
            obs, info = self.reset()
            zeros = {a: 0.0 for a in self.agent_ids}
            term = {a: True for a in self.agent_ids}
            return obs, zeros, term, term, info

        alpha = float(SIM_CONFIG.get("action_lp_alpha", 1.0))
        for a, car in self.cars.items():
            raw = np.asarray(actions.get(a, np.zeros(2, dtype=np.float32)),
                             dtype=np.float32)
            raw = np.clip(raw, -1.0, 1.0)
            prev = self._smoothed_action.get(a, np.zeros(2, dtype=np.float32))
            sm = alpha * raw + (1.0 - alpha) * prev
            self._smoothed_action[a] = sm
            car.apply_action(steer=float(sm[0]), vel_cmd=float(sm[1]))

        sub_steps = SIM_CONFIG["control_freq"] // SIM_CONFIG["policy_freq"]
        try:
            for _ in range(sub_steps):
                p.stepSimulation(physicsClientId=self.client)
        except Exception:
            self.client = None
            obs, info = self.reset()
            zeros = {a: 0.0 for a in self.agent_ids}
            term = {a: True for a in self.agent_ids}
            return obs, zeros, term, term, info

        self._step_count += 1
        rewards = self._compute_rewards()
        terminated = self._check_terminations()
        truncated = {a: self._step_count >= SIM_CONFIG["max_episode_steps"]
                     for a in self.agent_ids}
        obs = self._build_obs_all()
        info = self._build_info()

        # Dynamic winner calculation based on continuous max progress
        current_winner = max(self.agent_ids, key=lambda a: self._lap_progress[a])
        info["__winner__"] = current_winner

        if any(terminated.values()) or any(truncated.values()):
            self._race_index += 1

        return obs, rewards, terminated, truncated, info

    def _build_obs_all(self) -> Dict[str, np.ndarray]:
        obs = {}
        states = {a: c.get_state() for a, c in self.cars.items()}
        for a in self.agent_ids:
            obs[a] = self._build_obs(a, states)
        return obs

    def _world_to_car_xy(self, yaw: float, x: float, y: float) -> Tuple[float, float]:
        c, s = np.cos(yaw), np.sin(yaw)
        return c * x + s * y, -s * x + c * y

    def _build_obs(self, agent_id: str, states: Dict[str, dict]) -> np.ndarray:
        s = states[agent_id]
        pos = s["position"]
        lin = s["linear_velocity"]
        ang = s["angular_velocity"]
        eul = s["orientation_euler"]
        car = self.cars[agent_id]
        yaw = float(eul[2])

        pos_xy = np.array([pos[0], pos[1]], dtype=np.float32)
        fwd_w = car.forward_unit_vector()
        fwd_xy = np.array([fwd_w[0], fwd_w[1]], dtype=np.float32)

        cx, cy = self.track.closest_centerline_point(float(pos[0]), float(pos[1]))
        cx_local, cy_local = self._world_to_car_xy(yaw, cx - float(pos[0]), cy - float(pos[1]))

        steer_angle, steer_rate = car.steering_state()
        rear_wvel = car.rear_wheel_velocity()
        yaw_rate = float(ang[2])

        opp_id = next((b for b in self.agent_ids if b != agent_id), agent_id)
        opp = states[opp_id]
        ox_local, oy_local = self._world_to_car_xy(
            yaw, float(opp["position"][0] - pos[0]), float(opp["position"][1] - pos[1]))

        dvel = (opp["linear_velocity"] - lin).astype(np.float32)
        dvx_local, dvy_local = self._world_to_car_xy(yaw, float(dvel[0]), float(dvel[1]))

        return np.array([
            pos_xy[0], pos_xy[1],
            fwd_xy[0], fwd_xy[1],
            cx_local, cy_local,
            steer_angle, steer_rate,
            rear_wvel, yaw_rate,
            ox_local, oy_local,
            dvx_local, dvy_local,
        ], dtype=np.float32)

    def _signed_lateral(self, x: float, y: float) -> float:
        return self.track.signed_lateral(x, y)

    def _sample_dr_params(self) -> Dict[str, float]:
        cfg = DR_CONFIG
        if not cfg.get("enabled"):
            return {}
        s = float(cfg["std_pct"])
        lo = float(cfg["clip_lo_pct"])
        hi = float(cfg["clip_hi_pct"])
        rng = self._rng

        def clip_around(default: float, value: float) -> float:
            b1 = lo * default
            b2 = hi * default
            return float(np.clip(value, min(b1, b2), max(b1, b2)))

        def sample_around(default: float) -> float:
            v = float(rng.normal(default, s * abs(default)))
            return clip_around(default, v)

        params: Dict[str, float] = {}
        wanted = set(cfg.get("params", []))

        if "max_drive_torque" in wanted:
            params["max_drive_torque"] = sample_around(CAR_CONFIG["max_drive_torque"])
        if "traction" in wanted:
            params["traction"] = clip_around(1.0, float(rng.normal(1.0, s)))
        if "gravity" in wanted:
            params["gravity"] = sample_around(SIM_CONFIG["gravity"])
        if "car_mass" in wanted:
            if not hasattr(self, "_baseline_chassis_mass"):
                first_car = next(iter(self.cars.values()))
                self._baseline_chassis_mass = first_car.chassis_mass()
            params["car_mass"] = sample_around(self._baseline_chassis_mass)
        if "dt" in wanted:
            params["dt"] = sample_around(1.0 / SIM_CONFIG["control_freq"])

        return params

    def _apply_dr_params(self, params: Dict[str, float]) -> None:
        if "gravity" in params:
            p.setGravity(0, 0, float(params["gravity"]), physicsClientId=self.client)
        if "dt" in params:
            p.setTimeStep(float(params["dt"]), physicsClientId=self.client)

        brake_ratio = CAR_CONFIG["max_brake_torque"] / CAR_CONFIG["max_drive_torque"]
        for car in self.cars.values():
            if "max_drive_torque" in params:
                car.set_max_drive_torque(params["max_drive_torque"])
                car.set_max_brake_torque(params["max_drive_torque"] * brake_ratio)
            if "traction" in params:
                car.set_traction(params["traction"])
            if "car_mass" in params:
                car.set_chassis_mass(params["car_mass"])

    def _check_collisions(self, car: RaceCar) -> Tuple[bool, bool]:
        plane_id = self.track.body_ids[0] if self.track else -1
        other_bodies = {c.body for c in self.cars.values() if c is not car}
        contacts = p.getContactPoints(bodyA=car.body, physicsClientId=self.client)
        hit_wall, hit_car = False, False
        for c in contacts:
            bid = c[2]
            if bid == plane_id or bid == car.body:
                continue
            if bid in other_bodies:
                hit_car = True
            else:
                hit_wall = True
        return hit_wall, hit_car

    def _compute_rewards(self) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        rcfg = REWARD_CONFIG
        perim = self.track.perimeter()
        thr_z = float(RACE_CONFIG.get("flip_z_threshold", 0.3))

        progress_delta: Dict[str, float] = {}
        for a, car in self.cars.items():
            x, y, _ = car.get_state()["position"]
            new_prog_raw = self.track.centerline_progress(x, y)
            delta = new_prog_raw - self._prev_progress[a]
            if delta > perim / 2.0:
                delta -= perim
            elif delta < -perim / 2.0:
                delta += perim
            self._lap_progress[a] += delta
            self._prev_progress[a] = new_prog_raw
            progress_delta[a] = delta

        for a, car in self.cars.items():
            state = car.get_state()
            x, y, _ = state["position"]
            roll, pitch, _ = state["orientation_euler"]

            comp: Dict[str, float] = {}
            comp["progress"] = rcfg["progress_reward"] * progress_delta[a]
            comp["speed"] = rcfg["speed_reward"] * car.speed()
            comp["upright"] = rcfg["upright_reward"] * (float(roll) ** 2 + float(pitch) ** 2)
            
            opp_id = next((b for b in self.agent_ids if b != a), a)
            comp["relative"] = rcfg["relative_progress_reward"] * (self._lap_progress[a] - self._lap_progress[opp_id])
            
            lat = self._signed_lateral(x, y)
            dead = float(rcfg.get("centerline_dead_zone", 0.0))
            excess = max(0.0, abs(lat) - dead)
            comp["centerline"] = rcfg["centerline_penalty"] * (excess ** 2)
            
            if self.track.is_off_track(x, y):
                comp["off_track"] = rcfg["off_track_penalty"]

            hit_wall, hit_car = self._check_collisions(car)
            if hit_wall:
                comp["wall_hit"] = rcfg["wall_collision_penalty"]
            if hit_car:
                comp["car_hit"] = rcfg["car_collision_penalty"]

            if not self._flipped[a] and car.up_z() < thr_z:
                self._flipped[a] = True
                comp["flip"] = rcfg["flip_penalty"]

            rewards[a] = float(sum(comp.values()))
            self._reward_components[a] = comp

        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        return {a: bool(self._flipped[a]) for a in self.agent_ids}

    def _build_info(self) -> Dict[str, dict]:
        info: Dict[str, dict] = {}
        for a in self.agent_ids:
            info[a] = {
                "lap_progress": float(self._lap_progress[a]),
                "flipped": self._flipped[a],
                "reward_components": dict(self._reward_components.get(a, {})),
                "dr_params": dict(self._dr_params),
            }
        return info

    def close(self):
        if self.client is not None:
            try:
                p.disconnect(self.client)
            except Exception:
                pass
            self.client = None
            self.track = None
            self.cars = {}