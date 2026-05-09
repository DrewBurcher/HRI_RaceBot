"""
Main CLI entry point for HRI_RaceBot.

Usage:
    python main.py demo                              # random-action sanity demo
    python main.py debug [--opponent random|rl|human] [--opp-model PATH] [--algo sac|ppo]
    python main.py train [--algo sac|ppo] [--timesteps 1000000] [--name my_run]
                          [--headless] [--no-dashboard]
    python main.py race  --run runs/duo_sac_xxx [--algo sac|ppo] [--episodes 10] [--render]

Default algo is SAC: off-policy + replay buffer is much more sample-efficient
than PPO for continuous control with mixed-scale rewards on a single env.
PPO is still available via `--algo ppo`.

During training the PyBullet GUI is open by default (so you can watch the
cars learn) and a matplotlib live dashboard subprocess plots episode reward,
reward-component breakdown, race wins, etc. Pass `--headless` and/or
`--no-dashboard` for unattended/server runs.
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
          run_name=args.name, headless=args.headless,
          dashboard=args.dashboard,
          resume_from=getattr(args, "resume_from", None),
          seed=args.seed)


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
    p_debug.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    p_debug.add_argument("--both", action="store_true")
    p_debug.set_defaults(func=_cmd_debug)

    p_tr = sub.add_parser("train", help="Train two RL policies head-to-head")
    p_tr.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    p_tr.add_argument("--timesteps", type=int, default=1_000_000)
    p_tr.add_argument("--name", type=str, default=None)
    p_tr.add_argument("--resume", dest="resume_from", type=str, default=None,
                       help="Resume from runs/<name> — loads model, "
                            "VecNormalize, replay buffer, win streaks, etc.")
    # GUI + dashboard are on by default — pass these to disable.
    p_tr.add_argument("--headless", action="store_true",
                       help="Disable PyBullet GUI (default: GUI is on)")
    p_tr.add_argument("--no-dashboard", dest="dashboard",
                       action="store_false", default=True,
                       help="Disable the matplotlib live training dashboard")
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.set_defaults(func=_cmd_train)

    p_rc = sub.add_parser("race", help="Race two trained policies")
    p_rc.add_argument("--run", required=True)
    p_rc.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    p_rc.add_argument("--episodes", type=int, default=10)
    p_rc.add_argument("--render", action="store_true")
    p_rc.add_argument("--seed", type=int, default=0)
    p_rc.set_defaults(func=_cmd_race)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
