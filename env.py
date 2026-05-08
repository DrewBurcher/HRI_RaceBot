"""
Gymnasium-compatible two-car racing environment.

Designed around a *parallel* multi-agent interface: each `step()` consumes a
dict {agent_id: action} and returns dicts for obs/rewards/etc. This keeps
single- and multi-agent training paths uniform and is trivial to wrap with
stable-baselines3 (one VecEnv per agent in train.py).

The env is intentionally agnostic about *who* is driving — the agent classes
in `agents/` plug in cleanly. That makes it easy to mix RL policies, scripted
opponents, and human drivers (debug mode) without changing this file.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from config import CAR_CONFIG, RACE_CONFIG, SIM_CONFIG, TRACK_CONFIG
from racecar import RaceCar
from track import OvalTrack, build_track

AGENT_IDS = [f"car_{i}" for i in range(RACE_CONFIG["num_cars"])]
CAR_COLORS = [
    (0.9, 0.2, 0.2, 1.0),    # red
    (0.2, 0.4, 0.9, 1.0),    # blue
    (0.2, 0.8, 0.3, 1.0),    # green
    (0.9, 0.8, 0.2, 1.0),    # yellow
]


class TwoCarRaceEnv(gym.Env):
    """Multi-car race — returns dict obs/rew keyed by agent id.

    Action  per car: Box(2,) ∈ [-1, 1] = [steer, throttle]
    Obs     per car: Box of car kinematics + opponent relative pose +
                     vector to the next checkpoint. Cheap to start, easy to
                     extend (LiDAR, camera) later.
    """

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
            # Auto-extend agent ids if config asks for >2 cars.
            for k in range(len(AGENT_IDS), self.num_cars):
                AGENT_IDS.append(f"car_{k}")
        self.agent_ids = AGENT_IDS[: self.num_cars]
        self.alternate_lanes = (alternate_lanes if alternate_lanes is not None
                                 else RACE_CONFIG["alternate_lanes"])

        self.client: Optional[int] = None
        self.track: Optional[OvalTrack] = None
        self.cars: Dict[str, RaceCar] = {}

        self._race_index = 0          # number of completed races
        self._step_count = 0

        # Lap-progress trackers (per car)
        self._prev_progress: Dict[str, float] = {}
        self._lap_progress: Dict[str, float] = {}   # cumulative arc length
        self._last_visited_cp: Dict[str, int] = {}
        self._finished: Dict[str, bool] = {}
        self._winner: Optional[str] = None

        self._rng = np.random.default_rng(seed)

        # Single-agent gym spaces (used when wrapped per-agent)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                        shape=(2,), dtype=np.float32)
        # 3 (pos) + 3 (euler) + 3 (lin vel) + 3 (ang vel)
        # + 3 (relative opponent pose) + 2 (vector to next checkpoint)
        # + 1 (signed lateral distance to centerline)
        # + 1 (lap progress fraction)
        obs_dim = 3 + 3 + 3 + 3 + 3 + 2 + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)

    # ── Reset / connect ─────────────────────────────────────────────────
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

        # Build / rebuild world if first reset or sim died
        if self.track is None:
            self.track = build_track(self.client)

        # Decide lane assignment for this race.
        if self.alternate_lanes and self._race_index % 2 == 1:
            lanes = [1, 0] + list(range(2, self.num_cars))
        else:
            lanes = list(range(self.num_cars))

        jitter = self.track.random_start_jitter(self._rng)

        # Spawn or reposition cars
        for i, agent_id in enumerate(self.agent_ids):
            pos, quat = self.track.spawn_pose(lanes[i], jitter)
            color = CAR_COLORS[i % len(CAR_COLORS)]
            if agent_id in self.cars:
                self.cars[agent_id].reset(pos, quat)
            else:
                self.cars[agent_id] = RaceCar(self.client, pos, quat,
                                                car_id=i, color=color)

        # Initial progress reading (the start straight x is +0 progress at the
        # left edge of the front straight)
        self._prev_progress = {}
        self._lap_progress = {}
        self._last_visited_cp = {}
        self._finished = {a: False for a in self.agent_ids}
        self._winner = None
        self._step_count = 0

        for a, car in self.cars.items():
            x, y, _ = car.get_state()["position"]
            prog = self.track.centerline_progress(x, y)
            self._prev_progress[a] = prog
            self._lap_progress[a] = 0.0
            self._last_visited_cp[a] = -1

        # Settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)

        obs = self._build_obs_all()
        info = {a: {"lane": lanes[i], "jitter": jitter}
                for i, a in enumerate(self.agent_ids)}
        return obs, info

    # ── Step ────────────────────────────────────────────────────────────────────────
    def step(self, actions: Dict[str, np.ndarray]
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, float],
                        Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        if self.client is None:
            obs, info = self.reset()
            zeros = {a: 0.0 for a in self.agent_ids}
            term = {a: True for a in self.agent_ids}
            return obs, zeros, term, term, info

        for a, car in self.cars.items():
            act = actions.get(a, np.zeros(2, dtype=np.float32))
            car.apply_action(steer=float(act[0]), throttle=float(act[1]))

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
        # If a winner is declared, terminate everyone (race is over).
        if self._winner is not None:
            for a in self.agent_ids:
                terminated[a] = True
            self._race_index += 1
            info["__winner__"] = self._winner
            info["__race_index__"] = self._race_index
        return obs, rewards, terminated, truncated, info

    # ── Observations / rewards ─────────────────────────────────────────────
    def _build_obs_all(self) -> Dict[str, np.ndarray]:
        obs = {}
        states = {a: c.get_state() for a, c in self.cars.items()}
        for a in self.agent_ids:
            obs[a] = self._build_obs(a, states)
        return obs

    def _build_obs(self, agent_id: str,
                   states: Dict[str, dict]) -> np.ndarray:
        s = states[agent_id]
        pos = s["position"]
        eul = s["orientation_euler"]
        lin = s["linear_velocity"]
        ang = s["angular_velocity"]

        # Relative opponent pose (use first opponent only for the base obs;
        # extend later for >2 cars without breaking the current shape).
        opp_id = next((b for b in self.agent_ids if b != agent_id),
                      agent_id)
        opp = states[opp_id]
        rel = opp["position"] - pos

        # Vector to next checkpoint
        cp_idx = (self._last_visited_cp[agent_id] + 1) % len(self.track.checkpoints)
        cp = self.track.checkpoints[cp_idx]
        to_cp = np.array([cp.position[0] - pos[0],
                           cp.position[1] - pos[1]], dtype=np.float32)

        # Lateral offset from centerline (positive = outside the curve)
        lat = self._signed_lateral(pos[0], pos[1])

        perim = self.track.perimeter()
        prog_frac = (self._lap_progress[agent_id] / perim) if perim > 0 else 0.0

        return np.concatenate([
            pos.astype(np.float32),
            eul.astype(np.float32),
            lin.astype(np.float32),
            ang.astype(np.float32),
            rel.astype(np.float32),
            to_cp,
            np.array([lat], dtype=np.float32),
            np.array([prog_frac], dtype=np.float32),
        ]).astype(np.float32)

    def _signed_lateral(self, x: float, y: float) -> float:
        """Approx signed distance to the centerline (positive = outside)."""
        sl = self.track.straight_length
        r = self.track.curve_radius
        if -sl / 2.0 <= x <= sl / 2.0:
            return float(abs(y) - 0.0)   # straights run along y=0
        cx = sl / 2.0 if x > 0 else -sl / 2.0
        d = float(np.hypot(x - cx, y))
        return d - r

    def _compute_rewards(self) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        rcfg = RACE_CONFIG
        perim = self.track.perimeter()

        for a, car in self.cars.items():
            x, y, z = car.get_state()["position"]

            # Update cumulative arc-length progress (handle wrap-around).
            new_prog_raw = self.track.centerline_progress(x, y)
            delta = new_prog_raw - self._prev_progress[a]
            if delta > perim / 2.0:           # wrapped backward
                delta -= perim
            elif delta < -perim / 2.0:        # wrapped forward (lap completed)
                delta += perim
            self._lap_progress[a] += delta
            self._prev_progress[a] = new_prog_raw

            r = rcfg["progress_reward"] * delta
            r += rcfg["speed_reward"] * car.speed()

            # Checkpoint reward
            next_cp = (self._last_visited_cp[a] + 1) % len(self.track.checkpoints)
            cp = self.track.checkpoints[next_cp]
            if np.hypot(cp.position[0] - x, cp.position[1] - y) < 2.5:
                r += rcfg["checkpoint_reward"]
                self._last_visited_cp[a] = next_cp

            # Penalties
            if self.track.is_off_track(x, y):
                r += rcfg["off_track_penalty"] * 0.05   # per step

            # Lap completion check
            if (not self._finished[a]
                    and self._lap_progress[a] >= rcfg["laps_to_finish"] * perim):
                self._finished[a] = True
                if self._winner is None:
                    self._winner = a
                    r += rcfg["win_bonus"]
                else:
                    r += rcfg["lose_bonus"]

            rewards[a] = float(r)

        # If someone won this step, the loser(s) take the lose bonus too
        if self._winner is not None:
            for a in self.agent_ids:
                if a != self._winner and not self._finished[a]:
                    rewards[a] += rcfg["lose_bonus"]

        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        return {a: False for a in self.agent_ids}

    def _build_info(self) -> Dict[str, dict]:
        info: Dict[str, dict] = {}
        for a in self.agent_ids:
            info[a] = {
                "lap_progress": float(self._lap_progress[a]),
                "finished": self._finished[a],
                "last_checkpoint": self._last_visited_cp[a],
            }
        return info

    # ── Render / close ─────────────────────────────────────────────────────────
    def close(self):
        if self.client is not None:
            try:
                p.disconnect(self.client)
            except Exception:
                pass
            self.client = None
            self.track = None
            self.cars = {}


# Register a single-agent view of the env so SB3 can train one car at a time
# while the *other* car runs whatever opponent policy is configured.
class SingleAgentRaceWrapper(gym.Env):
    """Wraps `TwoCarRaceEnv` so SB3 sees a single-agent gym env.

    The opponent policy is provided at construction time — it can be a frozen
    snapshot of the *other* learner, a scripted bot, or a human keyboard agent.
    This is the seam train.py uses to train two policies in parallel.
    """

    metadata = {"render_modes": ["human", "rgb_array", None],
                "render_fps": 30}

    def __init__(self, base_env: TwoCarRaceEnv, learner_id: str,
                 opponent_policy):
        super().__init__()
        self.base = base_env
        self.learner_id = learner_id
        self.opp_policy = opponent_policy
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs[self.learner_id], info.get(self.learner_id, {})

    def step(self, action):
        actions = {self.learner_id: np.asarray(action, dtype=np.float32)}
        for a in self.base.agent_ids:
            if a == self.learner_id:
                continue
            obs_a = self._last_obs[a] if self._last_obs is not None else None
            actions[a] = self.opp_policy.act(obs_a)
        obs, rew, term, trunc, info = self.base.step(actions)
        self._last_obs = obs
        info_l = info.get(self.learner_id, {})
        if "__winner__" in info:
            info_l["winner"] = info["__winner__"]
            info_l["is_winner"] = info["__winner__"] == self.learner_id
        return (obs[self.learner_id], rew[self.learner_id],
                term[self.learner_id], trunc[self.learner_id], info_l)

    def render(self):
        return None

    def close(self):
        self.base.close()
