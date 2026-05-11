import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import DR_CONFIG, RACE_CONFIG, REWARD_CONFIG, SAC_CONFIG
from env import TwoCarRaceEnv

class DummySingleEnv(gym.Env):
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

def _save_resume_state(log_dir: str, models: Dict[str, SAC], normalizers: Dict[str, VecNormalize], 
                       elapsed: int, ep_counts: Dict[str, int]) -> None:
    for a, model in models.items():
        try:
            model.save(os.path.join(log_dir, f"{a}_sac_latest"))
            normalizers[a].save(os.path.join(log_dir, f"{a}_vecnormalize.pkl"))
            if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
                model.save_replay_buffer(os.path.join(log_dir, f"{a}_replay_buffer"))
        except Exception:
            pass

    state = {"elapsed": int(elapsed), "ep_counts": ep_counts}
    try:
        with open(os.path.join(log_dir, "learners_state.json"), "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def _load_resume_state(log_dir: str, base_env: TwoCarRaceEnv, total_timesteps: int) -> Tuple[Dict[str, SAC], Dict[str, VecNormalize], int, Dict[str, int]]:
    state_path = os.path.join(log_dir, "learners_state.json")
    with open(state_path, "r") as f:
        state = json.load(f)

    elapsed = int(state.get("elapsed", 0))
    ep_counts = state.get("ep_counts", {a: 0 for a in base_env.agent_ids})
    models = {}
    normalizers = {}

    for a in base_env.agent_ids:
        dummy = DummySingleEnv(base_env.observation_space, base_env.action_space)
        venv = DummyVecEnv([lambda: dummy])

        vn_path = os.path.join(log_dir, f"{a}_vecnormalize.pkl")
        if os.path.exists(vn_path):
            vec_norm = VecNormalize.load(vn_path, venv)
        else:
            vec_norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        normalizers[a] = vec_norm

        model_path = os.path.join(log_dir, f"{a}_sac_latest.zip")
        model = SAC.load(model_path, env=venv, device="cpu")
        model.tensorboard_log = os.path.join(log_dir, "tb")
        
        buf_path = os.path.join(log_dir, f"{a}_replay_buffer.pkl")
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)

        model._setup_learn(total_timesteps=total_timesteps, tb_log_name=a, reset_num_timesteps=False)
        models[a] = model

    print(f"[resume] Loaded state from step {elapsed}")
    return models, normalizers, elapsed, ep_counts

def train(timesteps: int = 1_000_000, run_name: Optional[str] = None,
          headless: bool = False, dashboard: bool = True, resume_from: Optional[str] = None, seed: int = 0) -> str:
    
    is_resume = resume_from is not None
    if is_resume:
        log_dir = resume_from
        if not os.path.isdir(log_dir):
            raise SystemExit(f"--resume directory does not exist: {log_dir}")
        run_name = os.path.basename(log_dir.rstrip("/\\"))
        print(f"[train] RESUMING from {log_dir}")
    else:
        run_name = run_name or f"sac_parallel_{int(time.time())}"
        log_dir = os.path.join("runs", run_name)
        os.makedirs(log_dir, exist_ok=True)
    
    base_env = TwoCarRaceEnv(render_mode=None if headless else "human", seed=seed)
    obs_dict, _ = base_env.reset()
    
    if is_resume:
        models, normalizers, elapsed, ep_count = _load_resume_state(log_dir, base_env, timesteps)
    else:
        models: Dict[str, SAC] = {}
        normalizers: Dict[str, VecNormalize] = {}
        elapsed = 0
        ep_count = {a: 0 for a in base_env.agent_ids}
        
        for agent_id in base_env.agent_ids:
            dummy = DummySingleEnv(base_env.observation_space, base_env.action_space)
            venv = DummyVecEnv([lambda: dummy])
            vec_norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
            
            cfg = SAC_CONFIG.copy()
            cfg.pop("total_timesteps", None)
            model = SAC("MlpPolicy", venv, verbose=0, tensorboard_log=os.path.join(log_dir, "tb"), **cfg)
            
            model._setup_learn(total_timesteps=timesteps, tb_log_name=agent_id)
            
            models[agent_id] = model
            normalizers[agent_id] = vec_norm

    plot_proc = None
    if dashboard:
        try:
            plot_proc = subprocess.Popen(
                [sys.executable, "live_plot.py", "--run", log_dir],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    metrics = {a: {"reward_components": [], "losses": [], "dr": []} for a in base_env.agent_ids}
    
    if is_resume:
        for a in base_env.agent_ids:
            try:
                with open(os.path.join(log_dir, f"metrics_{a}.json"), "r") as f:
                    metrics[a] = json.load(f)
            except Exception:
                pass

    comps_acc = {a: {} for a in base_env.agent_ids}
    ep_steps = {a: 0 for a in base_env.agent_ids}
    last_dr = {a: {} for a in base_env.agent_ids}
    interrupted = False

    try:
        while elapsed < timesteps:
            actions = {}
            for a in base_env.agent_ids:
                norm_obs = normalizers[a].normalize_obs(obs_dict[a].reshape(1, -1))
                act, _ = models[a].predict(norm_obs, deterministic=False)
                actions[a] = act[0]

            next_obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = base_env.step(actions)

            for a in base_env.agent_ids:
                norm_obs = normalizers[a].normalize_obs(obs_dict[a].reshape(1, -1))
                norm_next_obs = normalizers[a].normalize_obs(next_obs_dict[a].reshape(1, -1))
                norm_reward = normalizers[a].normalize_reward(np.array([rewards_dict[a]]))

                models[a].replay_buffer.add(
                    norm_obs,
                    norm_next_obs,
                    actions[a].reshape(1, -1),
                    norm_reward,
                    np.array([term_dict[a]]),
                    [info_dict[a]]
                )

                for k, v in info_dict[a].get("reward_components", {}).items():
                    comps_acc[a][k] = comps_acc[a].get(k, 0.0) + float(v)
                ep_steps[a] += 1

                dr = info_dict[a].get("dr_params", {})
                if dr and tuple(sorted(dr.items())) != last_dr[a]:
                    last_dr[a] = tuple(sorted(dr.items()))
                    metrics[a]["dr"].append({"timestep": elapsed, **dr})

            obs_dict = next_obs_dict
            elapsed += 1

            for a in base_env.agent_ids:
                models[a].num_timesteps = elapsed
                models[a]._update_current_progress_remaining(elapsed, timesteps)

            if elapsed > SAC_CONFIG.get("learning_starts", 100):
                for a in base_env.agent_ids:
                    models[a].train(batch_size=SAC_CONFIG.get("batch_size", 256), gradient_steps=1)

            if any(term_dict.values()) or any(trunc_dict.values()):
                for a in base_env.agent_ids:
                    entry = {
                        "episode": ep_count[a],
                        "timestep": elapsed,
                        "ep_length": ep_steps[a],
                        "flipped": info_dict[a].get("flipped", False),
                        "is_winner": info_dict.get("__winner__") == a
                    }
                    steps = max(1, ep_steps[a])
                    for k, v in comps_acc[a].items():
                        entry[k] = round(v / steps, 4)
                    metrics[a]["reward_components"].append(entry)
                    comps_acc[a].clear()
                    ep_steps[a] = 0
                    ep_count[a] += 1
                obs_dict, _ = base_env.reset()

            if elapsed % 200 == 0:
                for a in base_env.agent_ids:
                    try:
                        log_dict = models[a].logger.name_to_value
                        loss_entry = {"timestep": elapsed}
                        for key in ["train/actor_loss", "train/critic_loss", "train/ent_coef_loss"]:
                            if key in log_dict:
                                loss_entry[key.split("/")[1]] = round(float(log_dict[key]), 6)
                        if len(loss_entry) > 1:
                            metrics[a]["losses"].append(loss_entry)
                    except Exception:
                        pass
                    
                    try:
                        with open(os.path.join(log_dir, f"metrics_{a}.json"), "w") as f:
                            json.dump(metrics[a], f)
                    except Exception:
                        pass
                        
            if elapsed % 10_000 == 0:
                print(f"[{elapsed}/{timesteps}] Step complete.")

            if elapsed % 50_000 == 0:
                _save_resume_state(log_dir, models, normalizers, elapsed, ep_count)
                
    except KeyboardInterrupt:
        print("\n[train] Ctrl+C detected. Saving state...")
        interrupted = True

    _save_resume_state(log_dir, models, normalizers, elapsed, ep_count)

    if not interrupted:
        for a in base_env.agent_ids:
            models[a].save(os.path.join(log_dir, f"{a}_sac_final"))

    if plot_proc is not None:
        try:
            plot_proc.terminate()
        except Exception:
            pass

    return log_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", dest="resume_from", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-dashboard", dest="dashboard", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    train(timesteps=args.timesteps, run_name=args.name, resume_from=args.resume_from,
          headless=args.headless, dashboard=args.dashboard, seed=args.seed)