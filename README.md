# HRI_RaceBot

Two-car PyBullet racing built for the Human–Robot Interaction class. Two
independent reinforcement-learning policies race head-to-head on a procedural
oval track; whichever model wins three races in a row has its training paused
so the loser can catch up.

Layout/template inspired by [axelbr/racecar_gym](https://github.com/axelbr/racecar_gym);
training infrastructure (callbacks, monitor logging, run-metadata) ported and
adapted from `ML_Humaniod` in this same workspace. Nothing in `ML_Humaniod`
is modified.

## What's in the box

| Module | What it does |
|---|---|
| `config.py` | Every tunable: sim/control freqs, track geometry, car physics, race rules, **reward weights**, **domain randomization**, PPO/SAC hyperparams. Single source of truth. |
| `track.py` | Procedural stadium oval (two straights + two semicircles). Analytic `centerline_progress`, `closest_centerline_point`, `is_off_track`. Designed to subclass: drop in a new shape behind `build_track()`. |
| `racecar.py` | Wrapper around the default PyBullet `racecar/racecar.urdf`. Drive wheels under custom **PD-on-velocity** with **asymmetric torque clamp** (≈2× braking). Steering under POSITION_CONTROL. |
| `env.py` | Gymnasium two-car race env (multi-agent dict API) + `SingleAgentRaceWrapper` so SB3 sees a single-agent gym env per learner. Domain randomization, low-pass action filter, collision/flip detection, shaped rewards all live here. |
| `agents/` | Pluggable agent registry. `BaseAgent`, `RandomAgent`, `RLAgent` / `FrozenRLAgent`, `HumanKeyboardAgent`. Drop in MPC, scripted, etc. without touching the env. |
| `train.py` | Self-play loop: one SB3 model per car, opponent snapshots refresh every 50k steps, races between chunks tally wins, **3-in-a-row pauses** the leader. Per-learner `VecNormalize`. Tensorboard scalars for reward components and DR samples. |
| `debug.py` | Drive a car yourself with WASD. PyBullet sliders for Drive Torque, Brake Torque, Traction. |
| `evaluate.py` | Head-to-head race tournament between two trained models. |
| `main.py` | CLI dispatcher (`demo` / `debug` / `train` / `race`). |

## Quick start

```bash
pip install -r requirements.txt

# Sim sanity check (random actions, GUI window opens)
python main.py demo

# Drive a car yourself (WASD; click the PyBullet window for keyboard focus)
python main.py debug

# Train two PPO policies head-to-head
python main.py train --algo ppo --timesteps 1000000

# Race the trained models
python main.py race --run runs/<run_name> --algo ppo --episodes 10 --render
```

## Observation, action, reward

### Observation (per car, 15 floats)

The track lies in the world XY plane, so `z` is dropped from every relative
vector — only the car's own `z` position survives, useful for flip/spawn
sanity checks. Vectors marked **CAR FRAME** are rotated by `-yaw` so
forward/back/left/right line up with obs axes, which is much easier for
the policy to interpret than world-aligned vectors.

| Idx | Field | Frame |
|---|---|---|
| 0–2 | position (x, y, z) | world |
| 3–4 | forward unit vector (xy) | world |
| 5–6 | vector to closest centerline point (xy) | **car** |
| 7 | steering angle | rad |
| 8 | steering rate | rad/s |
| 9 | rear-wheel angular velocity | rad/s |
| 10 | yaw angular velocity | rad/s |
| 11–12 | vector to other car (xy) | **car** |
| 13–14 | velocity-difference vector (xy) | **car** |

### Action (per car, 2 floats, both in [-1, 1])

| Idx | Field | What it does |
|---|---|---|
| 0 | steering target | × `max_steer` → POSITION_CONTROL on the steer servos |
| 1 | drive-velocity target | × `vel_target_scale` (default 500 rad/s) → fed to a PD controller; output torque clamped asymmetrically: forward up to `max_drive_torque`, reverse up to `max_brake_torque` (≈2×) |

A first-order **low-pass filter** is applied per car between policy output
and the env: `smoothed = α·new + (1−α)·prev` with `α = SIM_CONFIG["action_lp_alpha"]`
(default 0.5). This stops the car from chattering on raw 30 Hz RL output
without changing what the policy sees in its own rollouts.

### Reward (weights in `config.REWARD_CONFIG`)

Dense (per step):
- `progress` — `+5.0 × Δ centerline arc-length` (the dominant signal)
- `speed` — `+0.05 × forward speed (m/s)`
- `upright` — `−2.0 × (roll² + pitch²)`
- `relative` — `+0.1 × (own_lap_arc − opp_lap_arc)`
- `centerline` — `−1.0 × max(0, |lateral| − 1.5)²` (1.5 m dead zone matches the racing lanes)
- `wall_hit` — `−20.0` while in contact with any wall
- `car_hit` — `−20.0` while in contact with the other car
- `off_track` — `−5.0` while off the drivable ring

Sparse / terminal:
- `win` — `+100.0` for the lap winner
- `lose` — `−20.0` for the loser
- `flip` — `−100.0` and **terminate** when the car's local +z dot world +z
  drops below `flip_z_threshold` (default 0.3 → ~70° tilt)

### Termination

- **Flip:** terminate that car immediately + flip penalty.
- **Race won:** terminate everyone; loser gets the lose penalty.
- **Truncation:** episode hits `max_episode_steps` (default 1500 ≈ 50 s).
- **Stuck:** intentionally not terminated — the policy must learn to recover.

## Self-play training

`train.py` runs both learners against each other in chunks:

1. Each learner runs `CHUNK_TIMESTEPS` (default 25k) of `model.learn()`
   while the other car is driven by a frozen policy.
2. After every `OPP_REFRESH_STEPS` (default 50k), each learner gets a
   fresh frozen snapshot of the other (via `model.save()` → `load()`,
   not `deepcopy`, which gets tripped up by VecEnv references).
3. Between chunks, `EVAL_RACES_PER_CHUNK` (default 5) deterministic head-
   to-head races are run; wins update each learner's streak counter.
4. If a learner wins **`RACE_CONFIG["win_streak_pause"]` = 3** in a row,
   its training is paused for the next chunk while the other one keeps
   learning. Streak resets the moment the leader loses a race.

Each learner's env is wrapped in `VecNormalize(norm_obs=True, norm_reward=True)`,
and the running RMS statistics are preserved across opponent refreshes
(re-attached to the new wrapper) so the policy never briefly sees
un-normalized obs mid-training.

## Domain randomization

At every `env.reset()`, one value per parameter is drawn from
`N(default, std_pct·|default|)` and clipped to `[clip_lo, clip_hi]·default`.
The same sample is applied to **both cars** so a single race stays fair, but
the dynamics vary across episodes. DR values are **not** exposed to the
policy or critic — robustness comes from the network learning to handle
the distribution.

| Param | Default | Configured std (±10%) | Clip (±30%) |
|---|---|---|---|
| `max_drive_torque` | 5.0 N·m | 0.5 N·m | 3.5 – 6.5 N·m |
| `traction` (μ) | 1.0 | 0.10 | 0.7 – 1.3 |
| `gravity` | -9.81 m/s² | 0.98 | -12.75 – -6.87 |
| `car_mass` (chassis) | URDF baseline | 10% | 0.7× – 1.3× |
| `dt` | 1/240 s | 0.000417 s | 0.7× – 1.3× of nominal |

Every per-episode sample is dumped to tensorboard under `dr/*_mean` so you
can confirm the distribution stayed healthy across training. Disable
entirely by setting `DR_CONFIG["enabled"] = False`.

## Tensorboard scalars

`tensorboard --logdir runs/<run_name>/tb` shows:

- **Default SB3 scalars** (PPO loss curves, ep_rew_mean, ep_len_mean, etc.).
- `reward_components/*` — per-step running mean of every reward term, so
  you can see which terms dominate and whether shaping is balanced.
- `race/wins_total`, `race/losses_total`, `race/flips_total`,
  `race/episodes_total` per learner.
- `dr/*_mean` and `dr/episodes_seen` — per-episode DR samples averaged
  over the rollout.

`runs/<name>/config_snapshot.json` saves the full `RACE_CONFIG`,
`REWARD_CONFIG`, `DR_CONFIG`, and algo hyperparams used for the run.

## Branch / PR

Development happens on `claude/setup-hri-racebot-s0Po3`; the active PR is
[#1](https://github.com/DrewBurcher/HRI_RaceBot/pull/1).
