"""
Central configuration for HRI_RaceBot.
"""

# ── Simulation ─────────────────────────────────────────────────────────────
SIM_CONFIG = {
    "control_freq": 240,           
    "policy_freq": 30,             
    "max_episode_steps": 1500,     
    "gravity": -9.81,
    "action_lp_alpha": 0.5,
}

# ── Track geometry ───────────────────────────────────────────────────────
TRACK_CONFIG = {
    # ── Shape selector ──────────────────────────────────────────────────
    # "oval"  → procedural stadium oval (no STL needed)
    # "mesh"  → load your SolidWorks STL (fill in the mesh section below)
    "shape": "mesh",

    # ── Oval track parameters (used when shape="oval") ──────────────────
    "straight_length": 30.0,
    "curve_radius": 12.0,
    "track_width": 3.0,
    "wall_height": 0.6,
    "wall_thickness": 0.3,
    "num_curve_segments": 24,
    "lane_offset": 0.75,
    "start_jitter": 8.0,
    "checkpoint_count": 16,

    # ── Mesh track parameters (used when shape="mesh") ──────────────────
    # Step 1: export your SolidWorks part as STL (File → Save As → .stl)
    # Step 2: set stl_path to the path of that file (relative or absolute)
    # Step 3: set mesh_scale — SolidWorks defaults to mm, PyBullet uses m
    #         so [0.001, 0.001, 0.001] converts mm → m
    # Step 4: fill in waypoints — a list of [x, y] points (in meters) that
    #         trace the track centerline counterclockwise as a closed loop.
    #         The first waypoint is the spawn/start position.
    #         Tip: measure key points from your SolidWorks model and convert
    #         to meters using the same scale as mesh_scale.
    # Step 5: set track_width to the drivable width of your track (meters)
    "stl_path": "Track.STL",
    "mesh_scale": [1.0, 1.0, 1.0],         # Track.STL is authored in meters
    # SolidWorks default is Y-up; PyBullet is Z-up. Rotate +π/2 about X.
    "mesh_rpy": [1.5707963267948966, 0.0, 0.0],
    # Mesh-specific overrides (drivable ring in Track.STL is ~0.5 m wide, so
    # the oval defaults above are too generous to spawn cars onto).
    "mesh_track_width": 0.8,
    "mesh_lane_offset": 0.15,
    "mesh_start_jitter": 1.0,
    # After the rotation, recenter the track so its centroid is near the world
    # origin and the top driving surface sits at z=0. The raw STL footprint is
    # X∈[3.87, 23.76], Z∈[-10.67, 1.37] with Y∈[0, 0.2] (thickness).
    "mesh_position": [-14.585, -5.337, -0.18],
    "waypoints": [
        # Centerline waypoints (meters, post-rotation, post-translation) traced
        # CCW from the outer perimeter of Track.STL, offset inward by ~0.3 m.
        [ 4.987,  2.053], [ 3.879,  2.667], [ 2.511,  3.232], [ 1.131,  3.771],
        [-0.306,  4.090], [-1.784,  4.346], [-3.269,  4.602], [-4.732,  4.854],
        [-6.211,  4.884], [-7.710,  4.873], [-9.146,  4.835], [-10.200, 4.320],
        [-9.987,  3.406], [-8.706,  2.920], [-7.443,  3.431], [-6.163,  2.814],
        [-4.591,  2.820], [-3.087,  2.827], [-1.673,  2.825], [-0.791,  2.150],
        [-0.237,  0.758], [-0.283, -0.751], [-0.331, -2.257], [-0.947, -3.621],
        [-1.535, -4.750], [-1.286, -5.814], [-0.105, -5.976], [ 1.185, -5.270],
        [ 2.749, -5.302], [ 4.033, -6.121], [ 5.407, -6.318], [ 6.882, -6.279],
        [ 8.252, -6.221], [ 8.812, -5.255], [ 8.781, -3.942], [ 7.744, -2.909],
        [ 7.378, -1.481], [ 6.811, -0.494], [ 5.317, -0.435], [ 4.665,  0.719],
    ],
    # "track_width" above is shared — update it to match your STL track width
    # "lane_offset" above is shared — set to ~track_width/4 for your track
    # "start_jitter" above is shared — set to safe forward range at the start
    # "checkpoint_count" above is shared
}

# ── Car ─────────────────────────────────────────────────────────────────────────
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

# ── Race rules ──────────────────────────────────────────────────────────────
RACE_CONFIG = {
    "num_cars": 2,
    "alternate_lanes": True,      
    "flip_z_threshold": 0.3,      
}

# ── Reward weights ────────────────────────────────────────────────────────────────
REWARD_CONFIG = {
    "progress_reward":          100.0,    
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

# ── Domain randomization ──────────────────────────────────────────────────────
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