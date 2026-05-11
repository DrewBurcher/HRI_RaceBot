"""
Head-to-head evaluation — run N races between two trained policies.

Usage:
    python evaluate.py --run runs/duo_sac_123 --episodes 20 [--render]

Looks for `car_0_<algo>_final.zip` and `car_1_<algo>_final.zip` inside the
run directory.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from agents.rl_agent import RLAgent
from env import TwoCarRaceEnv


def evaluate(run_dir: str, algo: str = "sac", episodes: int = 10,
             render: bool = False, seed: int = 0) -> Dict[str, int]:
    env = TwoCarRaceEnv(render_mode="human" if render else None, seed=seed)
    obs, _ = env.reset()

    agents = {}
    for a in env.agent_ids:
        path = os.path.join(run_dir, f"{a}_{algo}_final")
        if not (os.path.exists(path) or os.path.exists(path + ".zip")):
            # Fall back to the _latest checkpoint (written every chunk, also on
            # Ctrl+C). Lets you race a run that was interrupted before its
            # final timestep budget completed.
            latest = os.path.join(run_dir, f"{a}_{algo}_latest")
            if os.path.exists(latest) or os.path.exists(latest + ".zip"):
                path = latest
            else:
                raise FileNotFoundError(
                    f"Could not find trained model for {a} at "
                    f"{path}.zip or {latest}.zip")
        vn_path = os.path.join(run_dir, f"{a}_vecnormalize.pkl")
        agents[a] = RLAgent(path, algo=algo,
                            vecnormalize_path=vn_path if os.path.exists(vn_path) else None)

    wins = {a: 0 for a in env.agent_ids}
    wins["draw"] = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {a: agents[a].act(obs[a]) for a in env.agent_ids}
            obs, rew, term, trunc, info = env.step(actions)
            if any(term.values()) or any(trunc.values()):
                w = info.get("__winner__", None)
                if w is None:
                    wins["draw"] += 1
                else:
                    wins[w] += 1
                done = True
        print(f"[eval] race {ep + 1}/{episodes}: "
              f"winner = {info.get('__winner__', 'draw')}")

    env.close()
    print("\n[eval] final tally:", wins)
    return wins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run dir (e.g. runs/duo_sac_123)")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.run, args.algo, args.episodes, args.render, args.seed)
