"""
Manual debugging mode — drive a car yourself.

Usage:
    python debug.py                  # drive car_0 with WASD; opponent is random
    python debug.py --opponent random
    python debug.py --opponent rl --opp-model runs/duo_sac_xxx/car_1_sac_final
    python debug.py --both           # both cars driven by the keyboard

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


_DRIVE_MIN, _DRIVE_MAX = 0.1, 30.0
_BRAKE_MIN, _BRAKE_MAX = 0.1, 60.0
_TRACTION_MIN, _TRACTION_MAX = 0.1, 3.0
_DEFAULT_TRACTION = 1.0


def _add_sliders(client: int) -> dict:
    """Create the debug sliders and return their PyBullet ids.

    The env disables the PyBullet side panel for headless-style rendering;
    sliders live inside that panel, so we have to re-enable it here. The
    RGB/depth/segmentation preview panels are kept off because they're
    expensive and we don't need them.

    Drive wheels run under a custom PD-on-velocity controller with an
    asymmetric torque clamp:
        Drive Torque  → forward acceleration ceiling (N·m / wheel)
        Brake Torque  → backward / braking ceiling   (N·m / wheel, ~2x)
    Steering stays POSITION_CONTROL with a fixed steer_force.
    """
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                                physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                                physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                                physicsClientId=client)
    return {
        "drive_torque": p.addUserDebugParameter(
            "Drive Torque (N.m)", _DRIVE_MIN, _DRIVE_MAX,
            float(CAR_CONFIG["max_drive_torque"]), physicsClientId=client),
        "brake_torque": p.addUserDebugParameter(
            "Brake Torque (N.m)", _BRAKE_MIN, _BRAKE_MAX,
            float(CAR_CONFIG["max_brake_torque"]), physicsClientId=client),
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
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
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
    last_traction = -1.0

    print("=" * 60)
    print("  HRI_RaceBot — manual debugging mode")
    print("  W/Up: throttle  S/Down: reverse  A/Left, D/Right: steer")
    print("  SPACE: brake    Ctrl+C: quit")
    print("  Sliders (PyBullet sidebar): Drive Torque, Brake Torque, Traction")
    print("=" * 60)

    try:
        race = 0
        while True:
            drive_val = float(p.readUserDebugParameter(
                sliders["drive_torque"], physicsClientId=env.client))
            brake_val = float(p.readUserDebugParameter(
                sliders["brake_torque"], physicsClientId=env.client))
            traction_val = float(p.readUserDebugParameter(
                sliders["traction"], physicsClientId=env.client))
            for car in env.cars.values():
                car.set_max_drive_torque(drive_val)
                car.set_max_brake_torque(brake_val)
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
                last_traction = -1.0
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n[debug] stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
