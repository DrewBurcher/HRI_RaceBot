"""
Manual debugging mode — drive a car yourself.

Usage:
    python debug.py                  # drive car_0 with WASD; opponent is random
    python debug.py --opponent random
    python debug.py --opponent rl --opp-model runs/duo_ppo_xxx/car_1_ppo_final --algo ppo
    python debug.py --both           # both cars are driven by the keyboard
                                       (only useful with two people sharing one
                                       keyboard, but exposed for completeness)

The PyBullet GUI window must have keyboard focus for input to register.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agents.human_agent import HumanKeyboardAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from env import TwoCarRaceEnv


def _build_opponent(kind: str, model_path: str | None, algo: str):
    if kind == "random":
        return RandomAgent()
    if kind == "rl":
        if model_path is None:
            raise SystemExit("--opponent rl requires --opp-model")
        return RLAgent(model_path, algo=algo)
    if kind == "human":
        return HumanKeyboardAgent()
    raise SystemExit(f"Unknown opponent kind: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["random", "rl", "human"],
                        default="random")
    parser.add_argument("--opp-model", type=str, default=None)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--both", action="store_true",
                        help="Drive both cars with the keyboard")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = TwoCarRaceEnv(render_mode="human", seed=args.seed)
    obs, _ = env.reset()

    human0 = HumanKeyboardAgent(client=env.client)
    human1 = HumanKeyboardAgent(client=env.client) if args.both else None
    opp = (None if args.both
           else _build_opponent(args.opponent, args.opp_model, args.algo))

    print("=" * 60)
    print("  HRI_RaceBot — manual debugging mode")
    print("  W/Up: throttle  S/Down: reverse  A/Left, D/Right: steer")
    print("  SPACE: brake    Ctrl+C: quit")
    print("=" * 60)

    try:
        race = 0
        while True:
            actions = {}
            actions["car_0"] = human0.act(obs["car_0"])
            if args.both:
                actions["car_1"] = human1.act(obs["car_1"])
            else:
                actions["car_1"] = opp.act(obs["car_1"])
            obs, rew, term, trunc, info = env.step(actions)
            if any(term.values()) or any(trunc.values()):
                winner = info.get("__winner__", "none")
                print(f"[race {race}] over — winner = {winner}")
                race += 1
                obs, _ = env.reset()
                human0.reset()
                if human1 is not None:
                    human1.reset()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n[debug] stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
