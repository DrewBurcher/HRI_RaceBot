"""
Train two RL policies head-to-head on `TwoCarRaceEnv`.

Design notes
------------
* One stable-baselines3 model per car. Each is wrapped around the same
  underlying `TwoCarRaceEnv`, with a `SingleAgentRaceWrapper` exposing only
  that car's view to its learner. Each learner's VecEnv is wrapped in
  VecNormalize for obs + (clipped) reward normalization.
* Self-play: each learner sees a frozen copy of the *other* learner as its
  opponent. The opponent snapshots refresh every `OPP_REFRESH_STEPS`.
* Win-streak pause: if one model wins `RACE_CONFIG['win_streak_pause']`
  races in a row, its training is paused (skipped during the next chunk)
  while the other one keeps learning until the streak is broken.
* Each training chunk runs N timesteps for the active learner(s); race results
  are tallied between chunks.
* Reward components AND domain-randomization parameters are logged to
  tensorboard via RewardComponentLogger. A second callback,
  MetricsWriterCallback, dumps per-episode JSON for the live_plot.py
  dashboard subprocess.
* Ctrl+C / KeyboardInterrupt at any point pauses cleanly: model +
  VecNormalize + replay buffer + loop counters are saved to the run
  directory so `--resume runs/<name>` picks up exactly where it left off.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rl_agent import FrozenRLAgent
from agents.random_agent import RandomAgent
from config import DR_CONFIG, PPO_CONFIG, RACE_CONFIG, REWARD_CONFIG, SAC_CONFIG
from env import SingleAgentRaceWrapper, TwoCarRaceEnv


CHUNK_TIMESTEPS = 25_000          # per learner per chunk
OPP_REFRESH_STEPS = 50_000        # how often each learner sees a fresh snapshot
EVAL_RACES_PER_CHUNK = 5          # races used to count wins between chunks


class RewardComponentLogger(BaseCallback):
    """Aggregates per-step reward components into running means and dumps them
    to tensorboard at every algorithm log interval. Also tracks per-episode
    domain-randomization samples (one new sample per reset, deduplicated by
    signature) and dumps their running means under the `dr/*` namespace.
    """

    def __init__(self, learner_id: str, verbose: int = 0):
        super().__init__(verbose)
        self.learner_id = learner_id
        self._sums: Dict[str, float] = {}
        self._count: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._flips: int = 0
        self._episodes: int = 0
        self._dr_sums: Dict[str, float] = {}
        self._dr_count: int = 0
        self._last_dr_signature: tuple = ()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            comp = info.get("reward_components") or {}
            for k, v in comp.items():
                self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._count += 1
            if "winner" in info:
                self._episodes += 1
                if info.get("is_winner"):
                    self._wins += 1
                else:
                    self._losses += 1
            if info.get("flipped"):
                self._flips += 1
            dr = info.get("dr_params") or {}
            sig = tuple(sorted(dr.items()))
            if dr and sig != self._last_dr_signature:
                self._last_dr_signature = sig
                for k, v in dr.items():
                    self._dr_sums[k] = self._dr_sums.get(k, 0.0) + float(v)
                self._dr_count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._count == 0:
            return
        for k, total in self._sums.items():
            self.logger.record(f"reward_components/{k}",
                               total / max(self._count, 1))
        self.logger.record("race/wins_total", self._wins)
        self.logger.record("race/losses_total", self._losses)
        self.logger.record("race/flips_total", self._flips)
        self.logger.record("race/episodes_total", self._episodes)
        if self._dr_count > 0:
            for k, total in self._dr_sums.items():
                self.logger.record(f"dr/{k}_mean", total / self._dr_count)
            self.logger.record("dr/episodes_seen", self._dr_count)
        self._sums.clear()
        self._count = 0
        self._dr_sums.clear()
        self._dr_count = 0


class MetricsWriterCallback(BaseCallback):
    """Per-step + per-episode JSON dump for the live dashboard.

    Writes `runs/<run>/metrics_<learner_id>.json` containing:
        reward_components: per-episode mean of every reward term
        losses:            actor/critic/value/policy_grad/ent_coef snapshots
        dr:                per-episode DR samples (deduplicated by signature)

    The live_plot.py dashboard reads these files at ~1 Hz; missing keys are
    handled gracefully there, so adding a new reward term is automatic.
    """

    def __init__(self, learner_id: str, log_dir: str, save_freq: int = 200,
                 verbose: int = 0):
        super().__init__(verbose)
        self.learner_id = learner_id
        self.metrics_path = os.path.join(log_dir, f"metrics_{learner_id}.json")
        self.save_freq = save_freq
        self._comps: Dict[str, float] = {}
        self._steps_in_ep: int = 0
        self._episode_count: int = 0
        self._data: Dict[str, list] = {
            "reward_components": [],
            "losses": [],
            "dr": [],
        }
        self._last_dr_sig: tuple = ()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", [{}]):
            comp = info.get("reward_components") or {}
            for k, v in comp.items():
                self._comps[k] = self._comps.get(k, 0.0) + float(v)
            self._steps_in_ep += 1

            if "episode" in info:
                steps = max(self._steps_in_ep, 1)
                entry = {
                    "episode": self._episode_count,
                    "timestep": int(self.num_timesteps),
                    "ep_reward": float(info["episode"]["r"]),
                    "ep_length": int(info["episode"]["l"]),
                    "is_winner": bool(info.get("is_winner", False)),
                    "flipped": bool(info.get("flipped", False)),
                }
                for k, v in self._comps.items():
                    entry[k] = round(v / steps, 4)   # per-step mean
                self._data["reward_components"].append(entry)
                self._comps = {}
                self._steps_in_ep = 0
                self._episode_count += 1

            dr = info.get("dr_params") or {}
            sig = tuple(sorted(dr.items()))
            if dr and sig != self._last_dr_sig:
                self._last_dr_sig = sig
                self._data["dr"].append(
                    {"timestep": int(self.num_timesteps), **dr})

        if self.num_timesteps % self.save_freq == 0:
            loss_entry = {"timestep": int(self.num_timesteps)}
            try:
                logger = self.model.logger.name_to_value
                for key in ["train/actor_loss", "train/critic_loss",
                             "train/ent_coef_loss", "train/ent_coef",
                             "train/policy_gradient_loss",
                             "train/value_loss", "train/entropy_loss"]:
                    if key in logger:
                        loss_entry[key.split("/")[1]] = round(float(logger[key]), 6)
            except Exception:
                pass
            if len(loss_entry) > 1:
                self._data["losses"].append(loss_entry)
            try:
                with open(self.metrics_path, "w") as f:
                    json.dump(self._data, f)
            except Exception:
                pass
        return True


def _make_vec_env(base_env: TwoCarRaceEnv, learner_id: str, opponent,
                   log_dir: str, learner_tag: str) -> VecNormalize:
    wrapper = SingleAgentRaceWrapper(base_env, learner_id, opponent)
    wrapper = Monitor(wrapper,
                      filename=os.path.join(log_dir, f"monitor_{learner_tag}"))
    venv = DummyVecEnv([lambda: wrapper])
    return VecNormalize(venv, norm_obs=True, norm_reward=True,
                         clip_obs=10.0, clip_reward=10.0)


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
    return _make_vec_env(base_env, learner_id, opponent,
                          log_dir, learner_tag)


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


def _save_resume_state(log_dir: str,
                        learners: List["LearnerState"],
                        algo: str,
                        elapsed_total: int,
                        next_opp_refresh: int,
                        save_replay_buffer: bool = True) -> None:
    """Snapshot everything needed to resume a paused/interrupted run.

    Per-learner artifacts (under `runs/<run>/`):
        <car>_<algo>_latest.zip       SB3 model
        <car>_vecnormalize.pkl        VecNormalize obs/reward stats
        <car>_replay_buffer.pkl       SAC replay buffer (skipped for PPO)

    Plus a single `learners_state.json` with the loop-level counters
    (timesteps, win streak, paused, wins/races, elapsed_total,
    next_opp_refresh) so resume picks up exactly where Ctrl+C hit.
    """
    for L in learners:
        try:
            L.model.save(os.path.join(log_dir, f"{L.agent_id}_{algo}_latest"))
        except Exception as e:
            print(f"[save] WARN: model save failed for {L.agent_id}: {e}")
        try:
            L.env.save(os.path.join(log_dir, f"{L.agent_id}_vecnormalize.pkl"))
        except Exception as e:
            print(f"[save] WARN: VecNormalize save failed for {L.agent_id}: {e}")
        if save_replay_buffer and hasattr(L.model, "replay_buffer") \
                and L.model.replay_buffer is not None:
            try:
                L.model.save_replay_buffer(
                    os.path.join(log_dir, f"{L.agent_id}_replay_buffer"))
            except Exception as e:
                print(f"[save] WARN: replay buffer save failed for {L.agent_id}: {e}")

    state = {
        "algo": algo,
        "elapsed_total": int(elapsed_total),
        "next_opp_refresh": int(next_opp_refresh),
        "learners": [
            {"id": L.agent_id, "wins": L.wins,
             "races_played": L.races_played, "win_streak": L.win_streak,
             "paused": L.paused, "timesteps": L.timesteps}
            for L in learners
        ],
    }
    try:
        with open(os.path.join(log_dir, "learners_state.json"), "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[save] WARN: learners_state.json write failed: {e}")


def _load_resume_state(log_dir: str, algo: str,
                        base_env: TwoCarRaceEnv
                        ) -> Tuple[List["LearnerState"], int, int]:
    """Load everything _save_resume_state wrote and rebuild the learners.

    Returns (learners, elapsed_total, next_opp_refresh).
    """
    state_path = os.path.join(log_dir, "learners_state.json")
    with open(state_path, "r") as f:
        state = json.load(f)

    if state.get("algo") and state["algo"] != algo:
        print(f"[resume] WARN: saved algo is {state['algo']!r} but --algo "
              f"{algo!r} was requested. Using {algo!r}.")

    AlgoCls = _make_algo_class(algo)
    init_opp = RandomAgent(seed=0)   # placeholder; refreshed on first chunk
    learners: List["LearnerState"] = []

    for st in state["learners"]:
        car_id = st["id"]
        # Build a fresh wrapper around the current base_env, then load
        # VecNormalize stats on top so reward/obs normalization picks up
        # exactly where it left off.
        env_l = _wrap_for_learner(base_env, car_id, init_opp,
                                   log_dir, learner_tag=car_id)
        vn_path = os.path.join(log_dir, f"{car_id}_vecnormalize.pkl")
        if os.path.exists(vn_path):
            try:
                loaded_vn = VecNormalize.load(vn_path, env_l.venv)
                # VecNormalize.load returns a fresh wrapper; copy stats over.
                env_l.obs_rms = loaded_vn.obs_rms
                env_l.ret_rms = loaded_vn.ret_rms
            except Exception as e:
                print(f"[resume] WARN: VecNormalize load failed for {car_id}: {e}")

        model_path = os.path.join(log_dir, f"{car_id}_{algo}_latest.zip")
        model = AlgoCls.load(model_path, env=env_l, device="cpu")

        buf_path = os.path.join(log_dir, f"{car_id}_replay_buffer.pkl")
        if hasattr(model, "load_replay_buffer") and os.path.exists(buf_path):
            try:
                model.load_replay_buffer(buf_path)
            except Exception as e:
                print(f"[resume] WARN: replay buffer load failed for "
                      f"{car_id}: {e}")

        L = LearnerState(
            agent_id=car_id, algo=algo, model=model, env=env_l,
            timesteps=int(st["timesteps"]),
            win_streak=int(st["win_streak"]),
            paused=bool(st["paused"]),
            wins=int(st["wins"]),
            races_played=int(st["races_played"]),
        )
        learners.append(L)
        print(f"[resume] loaded {car_id}: {L.timesteps:,} steps, "
              f"streak={L.win_streak}, paused={L.paused}")

    return (learners, int(state.get("elapsed_total", 0)),
            int(state.get("next_opp_refresh", OPP_REFRESH_STEPS)))


def _refresh_opponents(base_env: TwoCarRaceEnv,
                       learners: List[LearnerState],
                       log_dir: str) -> None:
    """Each learner gets a frozen snapshot of the *other* learner.

    Snapshotting via save→load (not copy.deepcopy) — SB3 models hold
    references to their VecEnv and to PyTorch tensors, so deepcopy
    intermittently chokes on them. Save/load is bulletproof and only
    marginally slower.

    Also preserves VecNormalize obs/reward statistics across the refresh.
    Otherwise we'd reset the running RMS to zero on every refresh, briefly
    feeding the policy un-normalized observations until stats re-converge.
    """
    snapshots: Dict[str, object] = {}
    for L in learners:
        tmp_path = os.path.join(log_dir, f".tmp_snapshot_{L.agent_id}.zip")
        L.model.save(tmp_path)
        AlgoCls = _make_algo_class(L.algo)
        snapshots[L.agent_id] = AlgoCls.load(tmp_path, device="cpu")
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    for L in learners:
        opp_id = next(o for o in [x.agent_id for x in learners] if o != L.agent_id)
        opp = FrozenRLAgent(snapshots[opp_id], deterministic=True)

        old_obs_rms = getattr(L.env, "obs_rms", None)
        old_ret_rms = getattr(L.env, "ret_rms", None)

        new_env = _wrap_for_learner(base_env, L.agent_id, opp,
                                     log_dir, learner_tag=L.agent_id)
        if old_obs_rms is not None:
            new_env.obs_rms = old_obs_rms
        if old_ret_rms is not None:
            new_env.ret_rms = old_ret_rms

        L.env = new_env
        L.model.set_env(L.env)


def train(algo: str = "sac",
          total_timesteps: int = 1_000_000,
          run_name: Optional[str] = None,
          headless: bool = False,
          dashboard: bool = True,
          resume_from: Optional[str] = None,
          seed: int = 0) -> str:
    """Self-play training loop.

    headless=False (default) opens the PyBullet GUI so you can watch both
    cars race during training. headless=True is for unattended/server runs.

    dashboard=True (default) spawns live_plot.py as a subprocess that reads
    the per-learner metrics JSONs and tails the SB3 monitor CSVs into a
    matplotlib window. Pass dashboard=False to skip it.

    resume_from=<run_dir> picks up where a paused/interrupted run left off:
    loads each learner's model + VecNormalize stats + replay buffer (SAC),
    plus the loop counters (elapsed_total, next_opp_refresh, per-learner
    timesteps/streak/paused). Pass `--resume runs/<name>` from the CLI.

    Ctrl+C at any point during training saves all of the above (so the run
    is always resumable) and exits cleanly.
    """
    is_resume = resume_from is not None
    if is_resume:
        log_dir = resume_from
        if not os.path.isdir(log_dir):
            raise SystemExit(f"--resume directory does not exist: {log_dir}")
        run_name = os.path.basename(log_dir.rstrip("/\\"))
        print(f"[train] RESUMING from {log_dir}")
    else:
        if run_name is None:
            run_name = f"duo_{algo}_{int(time.time())}"
        log_dir = os.path.join("runs", run_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[train] log dir = {log_dir}")

    render = not headless
    base_env = TwoCarRaceEnv(render_mode="human" if render else None,
                              seed=seed)

    if is_resume:
        # Need at least one reset so base_env.agent_ids and the cars exist
        # before we rebuild VecNormalize wrappers.
        base_env.reset()
        learners, elapsed_total, next_opp_refresh = _load_resume_state(
            log_dir, algo, base_env)
    else:
        opp_seed = seed + 1
        init_opp = RandomAgent(seed=opp_seed)

        learners: List[LearnerState] = []
        for car_id in base_env.agent_ids:
            env_l = _wrap_for_learner(base_env, car_id, init_opp,
                                       log_dir, learner_tag=car_id)
            model = _make_model(algo, env_l, log_dir)
            learners.append(LearnerState(agent_id=car_id, algo=algo,
                                          model=model, env=env_l))
        elapsed_total = 0
        next_opp_refresh = OPP_REFRESH_STEPS

    win_streak_cap = RACE_CONFIG["win_streak_pause"]

    print(f"[train] {'continuing' if is_resume else 'starting'} head-to-head "
          f"training of {len(learners)} cars ({algo.upper()}) "
          f"target {total_timesteps:,} steps each "
          f"(elapsed: {elapsed_total:,})")

    callbacks = {
        L.agent_id: CallbackList([
            RewardComponentLogger(L.agent_id),
            MetricsWriterCallback(L.agent_id, log_dir),
        ])
        for L in learners
    }

    # ── Live dashboard subprocess ───────────────────────────────────────────
    plot_proc: Optional[subprocess.Popen] = None
    if dashboard:
        try:
            plot_proc = subprocess.Popen(
                [sys.executable, "live_plot.py", "--run", log_dir],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"[train] dashboard launched (pid={plot_proc.pid})")
        except Exception as e:
            print(f"[train] WARN: could not launch dashboard: {e}")

    interrupted = False
    try:
        while elapsed_total < total_timesteps:
            for L in learners:
                if L.paused:
                    print(f"[train] {L.agent_id} PAUSED (win streak = {L.win_streak})")
                    continue
                print(f"[train] {L.agent_id}: learning {CHUNK_TIMESTEPS:,} steps")
                L.model.learn(total_timesteps=CHUNK_TIMESTEPS,
                               reset_num_timesteps=False, progress_bar=False,
                               callback=callbacks[L.agent_id],
                               tb_log_name=L.agent_id)
                L.timesteps += CHUNK_TIMESTEPS

            elapsed_total += CHUNK_TIMESTEPS

            if elapsed_total >= next_opp_refresh:
                print("[train] refreshing self-play opponents")
                _refresh_opponents(base_env, learners, log_dir)
                next_opp_refresh += OPP_REFRESH_STEPS

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

            # Snapshot resume state every chunk. Replay buffer is heavy
            # (~25 MB at 500k transitions) so persist it on opponent-refresh
            # boundaries and at exit, not every chunk.
            save_buf = (elapsed_total % OPP_REFRESH_STEPS == 0)
            _save_resume_state(log_dir, learners, algo,
                                elapsed_total, next_opp_refresh,
                                save_replay_buffer=save_buf)
    except KeyboardInterrupt:
        print("\n[train] Ctrl+C — pausing run; saving model + replay buffer "
              "+ VecNormalize so you can resume with --resume.")
        interrupted = True

    # Final save: model + VecNormalize + replay buffer + learner state.
    # On a clean completion this overwrites the latest checkpoint with the
    # post-final-chunk weights; on Ctrl+C this is the saved-for-resume copy.
    _save_resume_state(log_dir, learners, algo,
                        elapsed_total, next_opp_refresh,
                        save_replay_buffer=True)

    # Also save under the canonical "*_final" name (used by evaluate.py),
    # but only on clean completion — an interrupted run isn't really "final".
    if not interrupted:
        for L in learners:
            save_path = os.path.join(log_dir, f"{L.agent_id}_{algo}_final")
            L.model.save(save_path)
            print(f"[train] saved {save_path}.zip")

    with open(os.path.join(log_dir, "config_snapshot.json"), "w") as f:
        json.dump({
            "algo": algo,
            "ppo_config": PPO_CONFIG if algo == "ppo" else None,
            "sac_config": SAC_CONFIG if algo == "sac" else None,
            "race_config": RACE_CONFIG,
            "reward_config": REWARD_CONFIG,
            "dr_config": DR_CONFIG,
            "chunk_timesteps": CHUNK_TIMESTEPS,
            "opp_refresh_steps": OPP_REFRESH_STEPS,
            "eval_races_per_chunk": EVAL_RACES_PER_CHUNK,
            "interrupted": interrupted,
            "elapsed_total": elapsed_total,
        }, f, indent=2)

    base_env.close()
    if plot_proc is not None:
        try:
            plot_proc.terminate()
        except Exception:
            pass

    if interrupted:
        print(f"[train] run paused. Resume with: "
              f"python main.py train --resume {log_dir}")
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two RL policies head-to-head")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", dest="resume_from", type=str, default=None,
                        help="Resume from runs/<name>; restores model, "
                             "VecNormalize, replay buffer, and loop counters")
    parser.add_argument("--headless", action="store_true",
                        help="Disable PyBullet GUI (default: GUI is on so you "
                             "can watch the cars race during training)")
    parser.add_argument("--no-dashboard", dest="dashboard",
                        action="store_false", default=True,
                        help="Disable the matplotlib live training dashboard")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(algo=args.algo, total_timesteps=args.timesteps,
          run_name=args.name, headless=args.headless,
          dashboard=args.dashboard, resume_from=args.resume_from,
          seed=args.seed)
