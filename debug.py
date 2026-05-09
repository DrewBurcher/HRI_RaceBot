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
import pybullet as p

from agents.human_agent import HumanKeyboardAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from config import CAR_CONFIG
from env import TwoCarRaceEnv


# Slider ranges. Defaults match CAR_CONFIG so behavior is unchanged on launch.
_TORQUE_MIN, _TORQUE_MAX = 0.1, 50.0
_TRACTION_MIN, _TRACTION_MAX = 0.1, 3.0
_DEFAULT_TRACTION = 1.0


def _add_sliders(client: int) -> dict:
    """Create the debug sliders and return their PyBullet ids.

    The env disables the PyBullet side panel for headless-style rendering;
    sliders live inside that panel, so we have to re-enable it here. The
    RGB/depth/segmentation preview panels are kept off because they're
    expensive and we don't need them.

    Drive wheels run under TORQUE_CONTROL — Max Torque sets the N·m applied
    per wheel at full throttle. Top speed emerges from friction/slip, not
    from a velocity cap.
    """
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                                physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                                physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                                physicsClientId=client)
    return {
        "torque": p.addUserDebugParameter(
            "Max Torque (N.m)", _TORQUE_MIN, _TORQUE_MAX,
            float(CAR_CONFIG["max_torque"]), physicsClientId=client),
        "traction": p.addUserDebugParameter(
            "Traction (mu)", _TRACTION_MIN, _TRACTION_MAX,
            _DEFAULT_TRACTION, physicsClientId=client),
    }


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

    sliders = _add_sliders(env.client)
    last_traction = -1.0      # force a set_traction on the first frame

    print("=" * 60)
    print("  HRI_RaceBot — manual debugging mode")
    print("  W/Up: throttle  S/Down: reverse  A/Left, D/Right: steer")
    print("  SPACE: brake    Ctrl+C: quit")
    print("  Sliders (PyBullet sidebar): Max Torque, Traction")
    print("=" * 60)

    try:
        race = 0
        while True:
            # Read sliders, push to both cars. Torque is cheap to set every
            # step (just an instance attribute); traction calls
            # changeDynamics so only update on actual change.
            torque_val = float(p.readUserDebugParameter(
                sliders["torque"], physicsClientId=env.client))
            traction_val = float(p.readUserDebugParameter(
                sliders["traction"], physicsClientId=env.client))
            for car in env.cars.values():
                car.set_max_torque(torque_val)
            if abs(traction_val - last_traction) > 1e-3:
                for car in env.cars.values():
                    car.set_traction(traction_val)
                last_traction = traction_val

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
                # Re-apply current slider state to the freshly spawned cars
                last_traction = -1.0
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n[debug] stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
