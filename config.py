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
    # Asymmetric torque limits — brakes can apply more force than the motor.
    "max_drive_torque": 5.0,          # N·m forward, per drive wheel
    "max_brake_torque": 10.0,         # N·m reverse, per drive wheel (~2x)
    "drive_kp": 0.5,                  # PD gain on (v_target - v_curr) → torque
    "vel_target_scale": 500.0,        # action[1]=±1 → ±500 rad/s velocity target
                                       # (high enough that PD saturates the
                                       #  torque clamp; no real top-speed cap)
    "max_steer": 0.6,                 # steering range (rad) — action[0]=±1
    "steer_force": 50.0,              # N·m holding torque on the steer servos
    "spawn_z": 0.05,
}

# Joint name patterns inside the default pybullet racecar URDF
CAR_JOINT_PATTERNS = {
    "steer": ["steering_hinge"],                              # both front wheels
    "drive": ["left_rear_wheel_joint", "right_rear_wheel_joint",
               "left_front_wheel_joint", "right_front_wheel_joint"],
    "rear":  ["left_rear_wheel_joint", "right_rear_wheel_joint"],
}

# ── Race rules ───────────────────────────────────────────────────────────────
RACE_CONFIG = {
    "num_cars": 2,
    "laps_to_finish": 1,
    "win_streak_pause": 3,        # pause a model's training after N consecutive wins
    "alternate_lanes": True,      # swap inside/outside each race
    "flip_z_threshold": 0.3,      # car's local +z dot world +z below this → flipped
}

# ── Reward weights ────────────────────────────────────────────────────────────────
REWARD_CONFIG = {
    # Densely shaped:
    "progress_reward":           5.0,    # × Δ centerline arc-length per step
    "speed_reward":              0.05,   # × forward speed (m/s)
    "upright_reward":           -2.0,    # × (roll² + pitch²) — discourage tilting
    "relative_progress_reward":  0.1,    # × (own_lap_arc - opp_lap_arc)
    "centerline_penalty":       -1.0,    # × max(0, |lateral|-dead_zone)² (m²)
    "centerline_dead_zone":      1.5,    # m of grace; matches lane_offset so
                                          # the racing lanes don't get penalized
    # Per-step penalties:
    "wall_collision_penalty":   -20.0,   # while in contact with any wall body
    "car_collision_penalty":    -20.0,   # while in contact with the other car
    "off_track_penalty":        -5.0,    # while off the drivable ring
    # Sparse / terminal:
    "win_bonus":                 100.0,
    "lose_penalty":             -20.0,
    "flip_penalty":             -100.0,
}

# ── RL hyperparameters ──────────────────────────────────────────────────────────
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
