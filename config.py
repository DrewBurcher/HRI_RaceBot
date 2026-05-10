"""
Central configuration for HRI_RaceBot.
"""

import math

# ── Simulation ─────────────────────────────────────────────
SIM_CONFIG = {
    "control_freq": 240,           
    "policy_freq": 30,             
    "max_episode_steps": 1500,     
    "gravity": -9.81,
    "action_lp_alpha": 0.5,
}

# ── Track geometry ───────────────────────────────────────────────────
TRACK_CONFIG = {
    # ── Shape selector ───────────────────────────────────────────
    # "oval"  → procedural stadium oval (no STL needed)
    # "mesh"  → load Track.STL (Dom's custom SolidWorks design)
    "shape": "mesh",

    # ── Oval track parameters (used when shape="oval") ──────────────────
    "straight_length": 30.0,
    "curve_radius": 12.0,
    # NB: "track_width", "lane_offset", "start_jitter", "checkpoint_count"
    # below are shared between oval and mesh modes.
    "wall_height": 0.6,
    "wall_thickness": 0.3,
    "num_curve_segments": 24,

    # ── Shared params (used by both modes) ──────────────────────────
    "track_width": 2.0,        # drivable channel width (m); STL scaled to match
    "lane_offset": 0.5,        # ≈ track_width / 4
    "start_jitter": 2.0,
    "checkpoint_count": 16,

    # ── Mesh track parameters (used when shape="mesh") ──────────────────
    # Track.STL was modelled in SolidWorks in the XZ plane (extruded thinly
    # along +Y). PyBullet uses Z-up with the ground on the XY plane, so we:
    #   1. mesh_scale       — uniform 3.578× to make the ~0.559-unit drivable
    #                          channel come out to ~2 m in PyBullet
    #   2. mesh_rotation    — rotate +90° around X so SolidWorks XZ → PyBullet XY
    #                          (SW Y "up" → PyBullet Z "up")
    #   3. base_position    — translate so the track is centred on (0, 0)
    # The waypoints below are the extracted centreline AFTER scale + rotation
    # + centring, i.e. already in PyBullet world XY (meters).
    "stl_path": "Track.STL",
    "mesh_scale": [3.578, 3.578, 3.578],
    "mesh_rotation_euler": [math.pi / 2, 0.0, 0.0],
    "base_position": [-49.430, -16.638, 0.0],
    "waypoints": [
        [-31.164,  20.152], [ -6.091,  20.683], [ -4.265,  20.624], [ -0.818,  19.563],
    [ 18.981,  11.432], [ 20.661,  10.283], [ 20.896,   9.163], [ 20.896,   8.191],
    [ 20.631,   6.983], [ 20.071,   6.187], [ 19.070,   3.977], [ 19.070,   2.387],
    [ 19.895,   1.444], [ 20.955,   1.149], [ 27.349,   1.002], [ 28.881,   0.265],
    [ 29.411,  -1.355], [ 29.352,  -5.303], [ 29.735,  -6.835], [ 31.090,  -8.220],
    [ 34.184, -10.813], [ 34.596, -11.785], [ 34.655, -18.002], [ 34.184, -19.210],
    [ 33.330, -19.828], [ 32.063, -20.152], [ 20.130, -20.624], [ 18.215, -20.447],
    [ 15.740, -19.033], [ 12.116, -16.440], [  9.995, -15.821], [  8.168, -16.146],
    [  6.695, -16.912], [  5.723, -17.766], [  4.279, -18.827], [  2.482, -19.357],
    [  0.655, -19.622], [ -1.377, -19.239], [ -2.733, -17.884], [ -3.234, -15.438],
    [ -2.939, -14.466], [ -2.232, -13.995], [ -0.022, -12.875], [  0.891, -12.050],
    [  1.156,  -9.723], [  1.716,  10.253], [  1.480,  11.962], [  0.567,  13.376],
    [ -1.112,  14.083], [ -2.379,  14.584], [-19.320,  14.230], [-22.060,  16.028],
    [-24.741,  16.676], [-27.570,  15.998], [-29.220,  14.466], [-31.989,  13.847],
    [-33.875,  14.908], [-34.670,  16.263], [-34.199,  17.766], [-33.020,  19.298],
    ],
}

# ── Car ───────────────────────────────────────────────────────────────────
CAR_CONFIG = {
    "urdf": "racecar/racecar.urdf",   
    "max_drive_torque": 5.0,          
    "max_brake_torque": 10.0,         
    "drive_kp": 0.5,                  
    "vel_target_scale": 500.0,        
    "max_steer": 0.6,                 
    "steer_force": 50.0,              
    "spawn_z": 0.05,
}

CAR_JOINT_PATTERNS = {
    "steer": ["steering_hinge"],                              
    "drive": ["left_rear_wheel_joint", "right_rear_wheel_joint",
               "left_front_wheel_joint", "right_front_wheel_joint"],
    "rear":  ["left_rear_wheel_joint", "right_rear_wheel_joint"],
}

# ── Race rules ─────────────────────────────────────────────────────────
RACE_CONFIG = {
    "num_cars": 2,
    "alternate_lanes": True,      
    "flip_z_threshold": 0.3,      
}

# ── Reward weights ─────────────────────────────────────────────────────────────
REWARD_CONFIG = {
    "progress_reward":          10.0,    
    "speed_reward":              0.05,   
    "upright_reward":           -2.0,    
    "relative_progress_reward":  0.1,    
    "centerline_penalty":       -0.2,    # Doubled to heavily enforce racing line
    "centerline_dead_zone":      0.0,    
    "wall_collision_penalty":   -20.0,   
    "car_collision_penalty":    -50.0,   # Increased avoidance gradient
    "off_track_penalty":        -5.0,    
    "flip_penalty":             -100.0,
}

# ── RL hyperparameters ──────────────────────────────────────────────────
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
    "policy_kwargs": dict(net_arch=dict(pi=[64], vf=[32])),
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
    "policy_kwargs": dict(net_arch=dict(pi=[64], qf=[32])),
    "device": "cpu",
    "total_timesteps": 1_000_000,
}

# ── Domain randomization ─────────────────────────────────────────────────
DR_CONFIG = {
    "enabled": True,
    "std_pct": 0.25,            # Increased DR variance
    "clip_lo_pct": 0.5,         
    "clip_hi_pct": 1.5,         
    "params": [
        "max_drive_torque",     
        "traction",             
        "gravity",              
        "car_mass",             
        "dt",                   
    ],
}
