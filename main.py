"""
Main CLI entry point for HRI_RaceBot.

Usage:
    python main.py demo                            # random-action sanity demo
    python main.py debug [--opponent random|rl|human] [--opp-model PATH] [--algo ppo|sac]
    python main.py train --algo ppo --timesteps 1000000 [--name my_run]
    python main.py race  --run runs/duo_ppo_xxx --algo ppo [--episodes 10] [--render]
"""

from __future__ import annotations

import argparse
import time


def _cmd_demo(args):
    """Random actions — quickest way to confirm sim & track build correctly."""
    import numpy as np

    from env import TwoCarRaceEnv

    env = TwoCarRaceEnv(render_mode="human")
    obs, _ = env.reset()
    print("[demo] running random actions for both cars; Ctrl+C to stop")
    try:
        race = 0
        while True:
            actions = {a: np.random.uniform(-1.0, 1.0, size=(2,)).astype("float32")
                       for a in env.agent_ids}
            obs, rew, term, trunc, info = env.step(actions)
            if any(term.values()) or any(trunc.values()):
                print(f"[demo] race {race} done, winner={info.get('__winner__')}")
                race += 1
                obs, _ = env.reset()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _cmd_debug(args):
    from debug import main as debug_main
    import sys
    # Forward extra args to debug.main by mutating argv — simplest path.
    sys.argv = ["debug.py"]
    if args.opponent:
        sys.argv += ["--opponent", args.opponent]
    if args.opp_model:
        sys.argv += ["--opp-model", args.opp_model]
    if args.algo:
        sys.argv += ["--algo", args.algo]
    if args.both:
        sys.argv += ["--both"]
    debug_main()


def _cmd_train(args):
    from train import train
    train(algo=args.algo, total_timesteps=args.timesteps,
          run_name=args.name, render=args.render, seed=args.seed)


def _cmd_race(args):
    from evaluate import evaluate
    evaluate(args.run, args.algo, args.episodes, args.render, args.seed)


def main():
    parser = argparse.ArgumentParser(description="HRI_RaceBot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Random-action visual sanity check")
    p_demo.set_defaults(func=_cmd_demo)

    p_debug = sub.add_parser("debug", help="Drive a car manually")
    p_debug.add_argument("--opponent", choices=["random", "rl", "human"],
                          default="random")
    p_debug.add_argument("--opp-model", type=str, default=None)
    p_debug.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p_debug.add_argument("--both", action="store_true")
    p_debug.set_defaults(func=_cmd_debug)

    p_tr = sub.add_parser("train", help="Train two RL policies head-to-head")
    p_tr.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p_tr.add_argument("--timesteps", type=int, default=1_000_000)
    p_tr.add_argument("--name", type=str, default=None)
    p_tr.add_argument("--render", action="store_true")
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.set_defaults(func=_cmd_train)

    p_rc = sub.add_parser("race", help="Race two trained policies")
    p_rc.add_argument("--run", required=True)
    p_rc.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p_rc.add_argument("--episodes", type=int, default=10)
    p_rc.add_argument("--render", action="store_true")
    p_rc.add_argument("--seed", type=int, default=0)
    p_rc.set_defaults(func=_cmd_race)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
