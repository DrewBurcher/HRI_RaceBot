# HRI_RaceBot

Two-car PyBullet racing for the Human-Robot Interaction class.

Template inspired by [axelbr/racecar_gym](https://github.com/axelbr/racecar_gym),
training/visualization patterns ported from `ML_Humaniod`.

## What this project does

* Builds a procedural racetrack (oval with a long straight start section).
* Spawns **two** PyBullet `racecar` URDFs side-by-side at the start line.
* Trains **two independent RL policies** (one per car). Cars alternate between
  the inside and outside lane each race so neither model has a fixed advantage.
* Pauses training for whichever model wins **3 races in a row** (lets the other
  catch up).
* **Debug mode**: drive a car yourself with the keyboard.
* Built for scalability — agents are pluggable (`agents/`), config is centralised,
  and the env supports an arbitrary number of cars.

## Quick start

```bash
pip install -r requirements.txt

# Random-action demo (sanity-check sim)
python main.py demo

# Manual driving with WASD
python main.py debug

# Train two PPO policies head-to-head
python main.py train --algo ppo --timesteps 1000000

# Evaluate trained models
python main.py race --runs runs/duo_ppo_123
```

## Layout

```
config.py          # all tunable knobs
track.py           # builds the racetrack in PyBullet
racecar.py         # car wrapper around the default pybullet racecar
env.py             # Gymnasium two-car racing environment
agents/            # pluggable agent classes (RL, human, scripted)
train.py           # dual-policy training loop with win-streak pause
debug.py           # keyboard driving for one or both cars
evaluate.py        # head-to-head races between trained models
main.py            # CLI dispatcher
```

## Branch

Development happens on `claude/setup-hri-racebot-s0Po3`.
