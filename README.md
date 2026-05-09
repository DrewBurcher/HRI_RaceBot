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
| `train.py` | Self-play loop: one SB3 model per car, opponent snapshots refresh every 50k steps, races between chunks tally wins, **3-in-a-row pauses** the leader. Per-learner `VecNormalize`. Tensorboard scalars for reward components and DR samples. Spawns `live_plot.py` as a subprocess. |
| `live_plot.py` | matplotlib live training dashboard — per-learner episode reward, reward-component breakdown, episode length, cumulative race tally, current win streak, run info. Tails `metrics_<car>.json`, monitor CSVs, and `history.json`. |
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

# Train two SAC policies head-to-head — opens the PyBullet viewer (so you
# can watch the cars learn) AND a matplotlib dashboard subprocess
python main.py train --timesteps 1000000

# Same training, no GUI / no dashboard (server / unattended)
python main.py train --timesteps 1000000 --headless --no-dashboard

# Resume a paused / Ctrl+C'd run (model + VecNormalize + replay buffer all
# restored from the run directory; loop counters picked up where you left off)
python main.py train --resume runs/<run_name> --timesteps 1500000

# Race the trained models
python main.py race --run runs/<run_name> --episodes 10 --render
```

**Default algorithm is SAC.** SAC's off-policy replay buffer is much more
sample-efficient than PPO for continuous control on a single env, and it
handles the mixed-scale dense + sparse rewards better in practice. PPO is
still wired up — pass `--algo ppo` to either subcommand.

**Live training UX.** During `python main.py train`, two windows open by
default:
1. The **PyBullet viewer** showing both cars racing in real time as the
   policies learn. Pass `--headless` to skip it (faster training).
2. A **matplotlib dashboard** (`live_plot.py`) tailing per-learner episode
   reward, reward-component breakdown, episode length, cumulative race
   tally, and current win streak. Pass `--no-dashboard` to skip it.

The dashboard reads `runs/<run>/metrics_<car_id>.json` (written ~every 200
sim steps by `MetricsWriterCallback`), the SB3 monitor CSVs, and
`history.json` for race results.

## Observation, action, reward

### Observation (per car, 14 floats)

The track lies in the world XY plane, so `z` is dropped from every
component (it's constant noise on a 2D track). Flip detection still uses
the car's local up vector, independent of the obs. Vectors marked
**CAR FRAME** are rotated by `-yaw` so forward/back/left/right line up
with obs axes — way easier for the policy to interpret than world-aligned
vectors.

| Idx | Field | Frame |
|---|---|---|
| 0–1 | position (x, y) | world |
| 2–3 | forward unit vector (xy) | world |
| 4–5 | vector to closest centerline point (xy) | **car** |
| 6 | steering angle | rad |
| 7 | steering rate | rad/s |
| 8 | rear-wheel angular velocity | rad/s |
| 9 | yaw angular velocity | rad/s |
| 10–11 | vector to other car (xy) | **car** |
| 12–13 | velocity-difference vector (xy) | **car** |

### Action (per car, 2 floats, both in [-1, 1])

| Idx | Field | What it does |
|---|---|---|
| 0 | steering target | × `max_steer` → POSITION_CONTROL on the steer servos |
| 1 | drive-velocity target | × `vel_target_scale` (default 500 rad/s) → fed to a PD controller; output torque clamped asymmetrically: forward up to `max_drive_torque`, reverse up to `max_brake_torque` (≈2×) |

A first-order **low-pass filter** is applied per car between policy output
and the env: `smoothed = α·new + (1−α)·prev` with `α = SIM_CONFIG["action_lp_alpha"]`
(default 0.5). This stops the car from chattering on raw 30 Hz RL output
without changing what the policy sees in its own rollouts.

### Actor / critic architecture

Single hidden layer each, asymmetric:

| Net | Layers | Why |
|---|---|---|
| Actor (`pi`) | `14 → 64 → 2` | More capacity to map a 14-dim obs into a smooth 2-dim policy |
| Critic (SAC `qf` / PPO `vf`) | `14 → 32 → 1` | Scalar output, simpler shape; fewer params reduces critic noise |

Set in `config.py` via `policy_kwargs=dict(net_arch=dict(pi=[64], qf=[32]))`
for SAC and `…dict(pi=[64], vf=[32])` for PPO. These are deliberately
small; the obs is only 14 dims and the action only 2, so wider nets just
slow training without buying capacity. Bump them if you start asking the
policy to do more (camera/LiDAR obs, more cars, more complex tracks).

### Reward (weights in `config.REWARD_CONFIG`)

Dense (per step):
- `progress` — `+10.0 × Δ centerline arc-length` (the dominant signal — at ~4 m/s this is ~1.3/step, which is several × any single penalty so the policy reliably prefers driving forward)
- `speed` — `+0.05 × forward speed (m/s)`
- `upright` — `−2.0 × (roll² + pitch²)`
- `relative` — `+0.1 × (own_lap_arc − opp_lap_arc)`
- `centerline` — `−0.1 × |lateral|²` (continuous quadratic, no dead zone — soft gradient toward the centerline so the policy actively steers away from the walls; weight kept light enough that **progress dominates** during normal driving)
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

## Pause & resume

Hit `Ctrl+C` in the terminal at any point during training. The run snapshots
**model + VecNormalize stats + replay buffer + loop counters** to the run
directory and exits cleanly:

```
runs/<name>/
   car_0_<algo>_latest.zip       car_1_<algo>_latest.zip
   car_0_vecnormalize.pkl        car_1_vecnormalize.pkl
   car_0_replay_buffer.pkl       car_1_replay_buffer.pkl   (SAC only)
   learners_state.json           (timesteps, win streaks, paused, elapsed_total)
   monitor_car_0.monitor.csv     metrics_car_0.json
   monitor_car_1.monitor.csv     metrics_car_1.json
   tb/                           config_snapshot.json
```

Resume with the same `runs/<name>` directory:

```bash
python main.py train --resume runs/<name> --timesteps 1500000
```

Resume re-loads each model + VecNormalize stats + replay buffer (so SAC's
critic doesn't re-bootstrap from scratch) and continues from the saved
`elapsed_total`. The dashboard subprocess restarts automatically and picks
up the existing metrics JSONs / monitor CSVs.

State is also auto-saved every chunk (replay buffer every 50k steps), so
even a hard crash only loses one chunk worth of progress.

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

- **Default SB3 scalars** (algo loss curves, ep_rew_mean, ep_len_mean, etc.).
- `reward_components/*` — per-step running mean of every reward term, so
  you can see which terms dominate and whether shaping is balanced.
- `race/wins_total`, `race/losses_total`, `race/flips_total`,
  `race/episodes_total` per learner.
- `dr/*_mean` and `dr/episodes_seen` — per-episode DR samples averaged
  over the rollout.

`runs/<name>/config_snapshot.json` saves the full `RACE_CONFIG`,
`REWARD_CONFIG`, `DR_CONFIG`, and algo hyperparams used for the run.

## Branch

Active development happens on `claude/setup-hri-racebot-s0Po3`; merged PRs
land on `main`.
