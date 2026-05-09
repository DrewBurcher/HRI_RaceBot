"""
Central configuration for HRI_RaceBot.

All tunable knobs live here so experiments are reproducible and so the rest of
the codebase stays free of magic numbers.
"""

# ── Simulation ─────────────────────────────────────────────────────────────
SIM_CONFIG = {
    "control_freq": 240,           # PyBullet step rate (Hz)
    "policy_freq": 30,             # how often the RL policy acts (Hz)
    "max_episode_steps": 1500,     # ~50 s per race at 30 Hz
    "gravity": -9.81,
}

# ── Track geometry ───────────────────────────────────────────────────────
TRACK_CONFIG = {
    "shape": "oval",               # 'oval' is the only one for now — add more later
    "straight_length": 30.0,       # length of the start/finish straight (m)
    "curve_radius": 12.0,          # radius of the two semicircles (m)
    "track_width": 6.0,            # drivable width (m)
    "wall_height": 0.6,            # outer wall height (m)
    "wall_thickness": 0.3,
    "num_curve_segments": 24,      # higher = smoother curves, more wall bodies
    "lane_offset": 1.5,            # half-distance between inner/outer spawn lanes (m)
    "start_jitter": 8.0,           # random shift along the straight at reset (m)
    "checkpoint_count": 16,        # virtual gates for lap progress
}

# ── Car ─────────────────────────────────────────────────────────────────────────
CAR_CONFIG = {
    "urdf": "racecar/racecar.urdf",   # ships with pybullet_data
    "max_torque": 5.0,                # N·m per drive wheel at full throttle
                                       # (drive joints use TORQUE_CONTROL, top
                                       # speed emerges from friction/slip)
    "max_steer": 0.6,                 # steering range (rad)
    "steer_force": 50.0,              # N·m holding torque on the steer servos
    "spawn_z": 0.05,
}

# Joint name patterns inside the default pybullet racecar URDF
CAR_JOINT_PATTERNS = {
    "steer": ["steering_hinge"],                              # both front wheels
    "drive": ["left_rear_wheel_joint", "right_rear_wheel_joint",
               "left_front_wheel_joint", "right_front_wheel_joint"],
}

# ── Race rules ──────────────────────────────────────────────────────────
RACE_CONFIG = {
    "num_cars": 2,
    "laps_to_finish": 1,
    "win_streak_pause": 3,        # pause a model's training after N consecutive wins
    "alternate_lanes": True,      # swap inside/outside each race
    "collision_penalty": -5.0,
    "off_track_penalty": -10.0,
    "win_bonus": 50.0,
    "lose_bonus": -10.0,
    "checkpoint_reward": 1.0,
    "progress_reward": 1.0,       # weight on per-step forward progress
    "speed_reward": 0.05,         # encourage going fast
}

# ── RL hyperparameters ─────────────────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=[128, 128]),
    "device": "cpu",
    "total_timesteps": 1_000_000,
}

SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 500_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "policy_kwargs": dict(net_arch=[128, 128]),
    "device": "cpu",
    "total_timesteps": 1_000_000,
}
