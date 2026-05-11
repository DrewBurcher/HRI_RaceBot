"""
Render report-quality PNGs from a HRI_RaceBot training run.

Usage:
    # one run
    python plot_run.py --run runs/sac_parallel_1778445890

    # compare two runs side-by-side (overlays + per-run figures for each)
    python plot_run.py --run runs/sac_parallel_1778445890 \
                       --compare runs/sac_parallel_1778443701 \
                       --labels overnight earlier

Reads:
    runs/<run>/monitor_<car>.monitor.csv   per-episode reward, length, time
    runs/<run>/metrics_<car>.json          per-episode reward-component breakdown
    runs/<run>/history.json                per-chunk race winners
    runs/<run>/learners_state.json         final timesteps + win streaks

Writes PNGs into runs/<run>/figures/.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARS = ["car_0", "car_1"]
CAR_COLORS = {"car_0": "#d62728", "car_1": "#1f77b4"}
COMP_ORDER = ["progress", "speed", "relative",
              "upright", "centerline", "off_track", "wall_hit", "car_hit", "flip"]
COMP_COLORS = {
    "progress":   "#2ecc71", "speed":     "#27ae60", "relative":   "#3498db",
    "upright":    "#e67e22", "centerline":"#9b59b6", "off_track":  "#c0392b",
    "wall_hit":   "#e74c3c", "car_hit":   "#d35400", "flip":       "#34495e",
}


def _read_monitor(run_dir: str, car: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, f"monitor_{car}.monitor.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, skiprows=1)


def _read_metrics(run_dir: str, car: str) -> Optional[dict]:
    path = os.path.join(run_dir, f"metrics_{car}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _read_history(run_dir: str) -> Optional[list]:
    path = os.path.join(run_dir, "history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < 2 or window < 2:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().values


def plot_episode_reward(run_dir: str, fig_dir: str, label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for car in CARS:
        df = _read_monitor(run_dir, car)
        if df is None or df.empty:
            continue
        eps = np.arange(len(df))
        ax.plot(eps, df["r"], color=CAR_COLORS[car], alpha=0.2, linewidth=0.6)
        ax.plot(eps, _smooth(df["r"].values, 50),
                color=CAR_COLORS[car], linewidth=2.0,
                label=f"{car} (50-ep mean)")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title(f"Episode reward over training{(' — ' + label) if label else ''}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ep_reward.png"), dpi=150)
    plt.close(fig)


def plot_episode_length(run_dir: str, fig_dir: str, label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for car in CARS:
        df = _read_monitor(run_dir, car)
        if df is None or df.empty:
            continue
        eps = np.arange(len(df))
        ax.plot(eps, df["l"], color=CAR_COLORS[car], alpha=0.2, linewidth=0.6)
        ax.plot(eps, _smooth(df["l"].values, 50),
                color=CAR_COLORS[car], linewidth=2.0,
                label=f"{car} (50-ep mean)")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode length (steps)")
    ax.set_title(f"Episode length over training{(' — ' + label) if label else ''}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ep_length.png"), dpi=150)
    plt.close(fig)


def plot_reward_components(run_dir: str, fig_dir: str, label: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, car in zip(axes, CARS):
        m = _read_metrics(run_dir, car)
        ax.set_title(f"{car} reward components")
        ax.set_xlabel("episode")
        ax.grid(True, alpha=0.3)
        if not m or not m.get("reward_components"):
            ax.text(0.5, 0.5, "(no metrics)", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        rows = m["reward_components"]
        eps = np.arange(len(rows))
        for comp in COMP_ORDER:
            if not any(comp in r for r in rows):
                continue
            y = np.array([float(r.get(comp, 0.0)) for r in rows])
            ax.plot(eps, _smooth(y, 50),
                    color=COMP_COLORS[comp], linewidth=1.5, label=comp)
        ax.legend(loc="best", fontsize=8, ncol=2)
    axes[0].set_ylabel("mean reward per step (smoothed)")
    fig.suptitle(f"Reward component breakdown{(' — ' + label) if label else ''}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "reward_components.png"), dpi=150)
    plt.close(fig)


def plot_race_wins(run_dir: str, fig_dir: str, label: str = "") -> None:
    history = _read_history(run_dir)
    if not history:
        return
    cum: Dict[str, List[int]] = {car: [0] for car in CARS}
    cum["draw"] = [0]
    for entry in history:
        winner = entry.get("winner") if isinstance(entry, dict) else entry
        for key in cum:
            cum[key].append(cum[key][-1] + (1 if winner == key else 0))
    x = np.arange(len(cum["car_0"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for car in CARS:
        ax.plot(x, cum[car], color=CAR_COLORS[car], linewidth=2.0,
                label=f"{car} ({cum[car][-1]} wins)")
    ax.plot(x, cum["draw"], color="gray", linewidth=1.5, linestyle="--",
            label=f"draws ({cum['draw'][-1]})")
    ax.set_xlabel("race index")
    ax.set_ylabel("cumulative wins")
    ax.set_title(f"Head-to-head race tally{(' — ' + label) if label else ''}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "race_wins.png"), dpi=150)
    plt.close(fig)


def plot_compare_episode_reward(runs: List[Tuple[str, str]], out_path: str) -> None:
    """Overlay episode-reward curves across multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, car in zip(axes, CARS):
        ax.set_title(f"{car}")
        ax.set_xlabel("episode")
        ax.grid(True, alpha=0.3)
        for run_dir, label in runs:
            df = _read_monitor(run_dir, car)
            if df is None or df.empty:
                continue
            eps = np.arange(len(df))
            ax.plot(eps, _smooth(df["r"].values, 50), linewidth=2.0, label=label)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("episode reward (50-ep mean)")
    fig.suptitle("Run comparison — episode reward", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_run(run_dir: str, label: str = "") -> str:
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_episode_reward(run_dir, fig_dir, label)
    plot_episode_length(run_dir, fig_dir, label)
    plot_reward_components(run_dir, fig_dir, label)
    plot_race_wins(run_dir, fig_dir, label)
    return fig_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Primary run dir")
    ap.add_argument("--compare", default=None, help="Optional second run dir to overlay")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Display labels for --run and --compare, in order")
    args = ap.parse_args()

    labels = args.labels or []
    label_a = labels[0] if len(labels) > 0 else os.path.basename(args.run.rstrip("/\\"))
    fig_dir_a = render_run(args.run, label_a)
    print(f"[plot_run] wrote per-run figures to {fig_dir_a}")

    if args.compare:
        label_b = labels[1] if len(labels) > 1 else os.path.basename(args.compare.rstrip("/\\"))
        fig_dir_b = render_run(args.compare, label_b)
        print(f"[plot_run] wrote per-run figures to {fig_dir_b}")
        out = os.path.join(args.run, "figures", "compare_ep_reward.png")
        plot_compare_episode_reward([(args.run, label_a), (args.compare, label_b)], out)
        print(f"[plot_run] wrote comparison figure to {out}")


if __name__ == "__main__":
    main()
