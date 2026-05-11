import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from env import TwoCarRaceEnv

def evaluate(run_dir: str, algo: str, episodes: int, render: bool, seed: int):
    """
    Loads trained models and normalizers to race them head-to-head.
    """
    print(f"Loading run: {run_dir}")
    
    if algo.lower() != "sac":
        print("Warning: Only SAC is officially supported in this configuration.")

    render_mode = "human" if render else None
    env = TwoCarRaceEnv(render_mode=render_mode, seed=seed)
    
    models = {}
    normalizers = {}
    
    for a in env.agent_ids:
        model_path = os.path.join(run_dir, f"{a}_sac_final.zip")
        norm_path = os.path.join(run_dir, f"{a}_vecnormalize.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            env.close()
            return
            
        print(f"Loading model and normalizer for {a}...")
        models[a] = SAC.load(model_path, device="cpu")
        
        # We must attach the normalizer to a dummy env to load it, 
        # but we will manually call it during the loop.
        dummy = DummyVecEnv([lambda: TwoCarRaceEnv(render_mode=None)])
        
        if os.path.exists(norm_path):
            normalizers[a] = VecNormalize.load(norm_path, dummy)
            normalizers[a].training = False      # Freeze running averages
            normalizers[a].norm_reward = False   # Do not normalize rewards during inference
        else:
            print(f"Warning: Normalizer not found at {norm_path}. Using raw inputs.")
            normalizers[a] = None

    print(f"\nStarting evaluation for {episodes} episodes...")
    
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        
        while not done:
            actions = {}
            for a in env.agent_ids:
                # 1. Normalize the observation using the saved statistics
                if normalizers[a]:
                    norm_obs = normalizers[a].normalize_obs(obs[a].reshape(1, -1))
                else:
                    norm_obs = obs[a].reshape(1, -1)
                    
                # 2. Predict using deterministic=True to suppress exploration noise
                action, _ = models[a].predict(norm_obs, deterministic=True)
                actions[a] = action[0] 
                
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            if render:
                # Sleep to maintain a viewable framerate (~60 FPS)
                time.sleep(1.0 / 60.0)
                
            if any(terminated.values()) or any(truncated.values()):
                done = True
                winner = info.get('__winner__')
                print(f"Episode {ep+1}/{episodes} finished. Winner: {winner}")

    env.close()