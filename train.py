"""
Train two RL policies head-to-head on `TwoCarRaceEnv`.

Design notes
------------
* One stable-baselines3 model per car. Each is wrapped around the same
  underlying `TwoCarRaceEnv`, with a `SingleAgentRaceWrapper` exposing only
  that car's view to its learner.
* Self-play: each learner sees a frozen copy of the *other* learner as its
  opponent. The opponent snapshots refresh every `OPP_REFRESH_STEPS`.
* Win-streak pause: if one model wins `RACE_CONFIG['win_streak_pause']`
  races in a row, its training is paused (skipped during the next chunk)
  while the other one keeps learning until the streak is broken.
* Each training chunk runs N timesteps for the active learner(s); race results
  are tallied between chunks. This keeps the loop tractable and gives the
  win-streak logic a natural place to fire.

This file is a working *starting point*; the chunk size / refresh schedule
are obvious tuning targets once you see how training is going.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.rl_agent import FrozenRLAgent
from agents.random_agent import RandomAgent
from config import PPO_CONFIG, RACE_CONFIG, SAC_CONFIG
from env import SingleAgentRaceWrapper, TwoCarRaceEnv


CHUNK_TIMESTEPS = 25_000          # per learner per chunk
OPP_REFRESH_STEPS = 50_000        # how often each learner sees a fresh snapshot
EVAL_RACES_PER_CHUNK = 5          # races used to count wins between chunks


@dataclass
class LearnerState:
    agent_id: str
    algo: str
    model: object              # SB3 model
    env: object                # SingleAgentRaceWrapper (or VecEnv)
    timesteps: int = 0
    win_streak: int = 0
    paused: bool = False
    wins: int = 0
    races_played: int = 0
    history: List[Dict] = field(default_factory=list)


def _make_algo_class(algo: str):
    return PPO if algo == "ppo" else SAC


def _make_model(algo: str, env, log_dir: str):
    cfg = (PPO_CONFIG if algo == "ppo" else SAC_CONFIG).copy()
    cfg.pop("total_timesteps", None)
    AlgoCls = _make_algo_class(algo)
    return AlgoCls("MlpPolicy", env, verbose=0,
                    tensorboard_log=os.path.join(log_dir, "tb"), **cfg)


def _wrap_for_learner(base_env: TwoCarRaceEnv, learner_id: str,
                      opponent, log_dir: str, learner_tag: str):
    wrapper = SingleAgentRaceWrapper(base_env, learner_id, opponent)
    wrapper = Monitor(wrapper,
                      filename=os.path.join(log_dir, f"monitor_{learner_tag}"))
    return DummyVecEnv([lambda: wrapper])


def _race_once(base_env: TwoCarRaceEnv, learners: List[LearnerState]
               ) -> Optional[str]:
    """Run a single race using each learner's current policy. Returns winner id."""
    obs, _ = base_env.reset()
    done = False
    winner: Optional[str] = None
    while not done:
        actions = {}
        for L in learners:
            o = obs[L.agent_id]
            a, _ = L.model.predict(o, deterministic=True)
            actions[L.agent_id] = a
        obs, rew, term, trunc, info = base_env.step(actions)
        if any(term.values()) or any(trunc.values()):
            done = True
            winner = info.get("__winner__", None)
    return winner


def _refresh_opponents(base_env: TwoCarRaceEnv,
                       learners: List[LearnerState],
                       log_dir: str) -> None:
    """Each learner gets a frozen snapshot of the *other* learner."""
    snapshots = {L.agent_id: copy.deepcopy(L.model) for L in learners}
    for L in learners:
        opp_id = next(o for o in [x.agent_id for x in learners] if o != L.agent_id)
        opp = FrozenRLAgent(snapshots[opp_id], deterministic=True)
        L.env = _wrap_for_learner(base_env, L.agent_id, opp,
                                   log_dir, learner_tag=L.agent_id)
        L.model.set_env(L.env)


def train(algo: str = "ppo",
          total_timesteps: int = 1_000_000,
          run_name: Optional[str] = None,
          render: bool = False,
          seed: int = 0) -> str:
    if run_name is None:
        run_name = f"duo_{algo}_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] log dir = {log_dir}")

    base_env = TwoCarRaceEnv(render_mode="human" if render else None,
                              seed=seed)

    # Bootstrap with random opponents so each learner has a valid env from t=0.
    opp_seed = seed + 1
    init_opp = RandomAgent(seed=opp_seed)

    learners: List[LearnerState] = []
    for car_id in base_env.agent_ids:
        env_l = _wrap_for_learner(base_env, car_id, init_opp,
                                   log_dir, learner_tag=car_id)
        model = _make_model(algo, env_l, log_dir)
        learners.append(LearnerState(agent_id=car_id, algo=algo,
                                      model=model, env=env_l))

    win_streak_cap = RACE_CONFIG["win_streak_pause"]
    elapsed_total = 0
    next_opp_refresh = OPP_REFRESH_STEPS

    print(f"[train] starting head-to-head training of {len(learners)} cars "
          f"({algo.upper()}) for {total_timesteps:,} steps each")

    while elapsed_total < total_timesteps:
        for L in learners:
            if L.paused:
                print(f"[train] {L.agent_id} PAUSED (win streak = {L.win_streak})")
                continue
            print(f"[train] {L.agent_id}: learning {CHUNK_TIMESTEPS:,} steps")
            L.model.learn(total_timesteps=CHUNK_TIMESTEPS,
                           reset_num_timesteps=False, progress_bar=False)
            L.timesteps += CHUNK_TIMESTEPS

        elapsed_total += CHUNK_TIMESTEPS

        # Periodic opponent refresh
        if elapsed_total >= next_opp_refresh:
            print("[train] refreshing self-play opponents")
            _refresh_opponents(base_env, learners, log_dir)
            next_opp_refresh += OPP_REFRESH_STEPS

        # Evaluation block: race policies head-to-head, update streaks/pauses
        wins_this_block: Dict[str, int] = {L.agent_id: 0 for L in learners}
        for _ in range(EVAL_RACES_PER_CHUNK):
            w = _race_once(base_env, learners)
            if w is None:
                continue
            wins_this_block[w] += 1
            for L in learners:
                L.races_played += 1
                if L.agent_id == w:
                    L.wins += 1
                    L.win_streak += 1
                else:
                    L.win_streak = 0

        # Apply pause rule
        for L in learners:
            previously_paused = L.paused
            L.paused = L.win_streak >= win_streak_cap
            if L.paused and not previously_paused:
                print(f"[train] >>> {L.agent_id} PAUSED — won {L.win_streak} "
                      f"races in a row")
            if not L.paused and previously_paused:
                print(f"[train] >>> {L.agent_id} resumed training")

        snapshot = {
            "elapsed_total": elapsed_total,
            "wins_this_block": wins_this_block,
            "learners": [
                {"id": L.agent_id, "wins": L.wins,
                 "races": L.races_played, "streak": L.win_streak,
                 "paused": L.paused, "timesteps": L.timesteps}
                for L in learners
            ],
        }
        for L in learners:
            L.history.append(snapshot)
        with open(os.path.join(log_dir, "history.json"), "w") as f:
            json.dump([L.history[-1] for L in learners], f, indent=2)
        print(f"[train] block summary: {wins_this_block}")

    # Final save
    for L in learners:
        save_path = os.path.join(log_dir, f"{L.agent_id}_{algo}_final")
        L.model.save(save_path)
        print(f"[train] saved {save_path}.zip")

    base_env.close()
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two RL policies head-to-head")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(algo=args.algo, total_timesteps=args.timesteps,
          run_name=args.name, render=args.render, seed=args.seed)
