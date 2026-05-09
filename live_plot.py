"""
Live training dashboard for HRI_RaceBot.

Spawned as a subprocess by train.py. Tails:
    runs/<run>/monitor_<car_id>.monitor.csv     (episode reward / length)
    runs/<run>/metrics_<car_id>.json            (reward components, losses, DR)
    runs/<run>/history.json                     (per-block win/streak/pause)
    runs/<run>/config_snapshot.json             (run config; written at end)

Layout (3 rows × 3 cols):
    Row 1  | car_0 episode reward | car_1 episode reward | cumulative wins
    Row 2  | car_0 reward parts   | car_1 reward parts   | win-streak history
    Row 3  | car_0 episode length | car_1 episode length | run info / status

Adapted from ML_Humaniod/live_plot.py (single-agent dashboard).

Usage:
    python live_plot.py --run runs/duo_sac_<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
# Prefer TkAgg (ships with Python on Windows) for the interactive dashboard.
# Fall back to whatever backend is available so the smoke test on a headless
# box doesn't blow up.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd


CAR_IDS = ["car_0", "car_1"]
CAR_COLORS = {"car_0": "#d62728", "car_1": "#1f77b4"}   # red, blue
COMPONENT_COLORS = {
    "progress":   "#2ecc71",
    "speed":      "#27ae60",
    "upright":    "#e67e22",
    "relative":   "#3498db",
    "centerline": "#9b59b6",
    "off_track":  "#c0392b",
    "wall_hit":   "#e74c3c",
    "car_hit":    "#d35400",
    "win":        "#16a085",
    "lose":       "#7f8c8d",
    "flip":       "#34495e",
}


# ── Loaders ───────────────────────────────────────────────────────────────────────
def _load_monitor(run_dir: str, car_id: str):
    path = os.path.join(run_dir, f"monitor_{car_id}.monitor.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, skiprows=1)
        return df if len(df) else None
    except Exception:
        return None


def _load_metrics(run_dir: str, car_id: str):
    path = os.path.join(run_dir, f"metrics_{car_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_history(run_dir: str):
    path = os.path.join(run_dir, "history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_config(run_dir: str):
    path = os.path.join(run_dir, "config_snapshot.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _smooth(y, window: int):
    if len(y) < 2 or window < 2:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().values


# ── Dashboard ─────────────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.smooth_window = 20
        self._wins_running = {a: 0 for a in CAR_IDS}

        self.fig = plt.figure(figsize=(18, 11))
        self.fig.canvas.manager.set_window_title(
            f"HRI_RaceBot — {os.path.basename(run_dir)}")

        gs = gridspec.GridSpec(4, 3, figure=self.fig,
                                height_ratios=[1, 1, 1, 0.06],
                                hspace=0.45, wspace=0.32)
        # Row 1: ep reward per car, cumulative wins
        self.ax_rew = {
            "car_0": self.fig.add_subplot(gs[0, 0]),
            "car_1": self.fig.add_subplot(gs[0, 1]),
        }
        self.ax_wins = self.fig.add_subplot(gs[0, 2])
        # Row 2: reward components per car, streak history
        self.ax_comp = {
            "car_0": self.fig.add_subplot(gs[1, 0]),
            "car_1": self.fig.add_subplot(gs[1, 1]),
        }
        self.ax_streak = self.fig.add_subplot(gs[1, 2])
        # Row 3: ep length per car, run info
        self.ax_len = {
            "car_0": self.fig.add_subplot(gs[2, 0]),
            "car_1": self.fig.add_subplot(gs[2, 1]),
        }
        self.ax_info = self.fig.add_subplot(gs[2, 2])

        # Smoothing slider + status text
        ax_slider = self.fig.add_axes([0.06, 0.02, 0.30, 0.018])
        self.smooth_slider = Slider(ax_slider, "Smooth", 1, 100,
                                     valinit=20, valstep=1)
        self.smooth_slider.on_changed(self._on_smooth)

        self.ax_status = self.fig.add_axes([0.40, 0.005, 0.58, 0.030])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(
            0, 0.5, "", fontsize=8, fontfamily="monospace",
            verticalalignment="center")

        self.fig.suptitle(
            f"HRI_RaceBot Training Dashboard — {os.path.basename(run_dir)}",
            fontsize=14, fontweight="bold")

    def _on_smooth(self, val):
        self.smooth_window = int(val)

    # ── Per-car panels ──
    def _draw_episode_reward(self, ax, car_id: str, mon):
        ax.clear()
        if mon is not None and len(mon) > 0:
            r = mon["r"].values
            eps = np.arange(1, len(r) + 1)
            ax.plot(eps, r, alpha=0.18, color=CAR_COLORS[car_id], linewidth=0.5)
            ax.plot(eps, _smooth(r, self.smooth_window),
                    color=CAR_COLORS[car_id], linewidth=2)
        else:
            ax.text(0.5, 0.5, f"Waiting for {car_id} data…",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"{car_id} — Episode Reward")
        ax.grid(True, alpha=0.3)

    def _draw_episode_length(self, ax, car_id: str, mon):
        ax.clear()
        if mon is not None and len(mon) > 0:
            l = mon["l"].values
            eps = np.arange(1, len(l) + 1)
            ax.plot(eps, l, alpha=0.18, color=CAR_COLORS[car_id], linewidth=0.5)
            ax.plot(eps, _smooth(l, self.smooth_window),
                    color=CAR_COLORS[car_id], linewidth=2)
        else:
            ax.text(0.5, 0.5, f"Waiting for {car_id} data…",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title(f"{car_id} — Episode Length")
        ax.grid(True, alpha=0.3)

    def _draw_components(self, ax, car_id: str, metrics):
        ax.clear()
        rcs = (metrics or {}).get("reward_components", []) if metrics else []
        if len(rcs) > 5:
            eps = [e.get("episode", i) for i, e in enumerate(rcs)]
            keys = sorted({k for e in rcs for k in e.keys()
                           if k not in ("episode", "timestep", "ep_reward",
                                         "ep_length", "is_winner", "flipped")})
            for k in keys:
                vals = np.array([e.get(k, 0.0) for e in rcs])
                if not np.any(vals):
                    continue
                color = COMPONENT_COLORS.get(k, None)
                ax.plot(eps, _smooth(vals, self.smooth_window),
                        color=color, linewidth=1.5, label=k)
            ax.legend(fontsize=6, loc="best", ncol=2)
        else:
            ax.text(0.5, 0.5, "Waiting for components…",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean reward / step")
        ax.set_title(f"{car_id} — Reward Components")
        ax.grid(True, alpha=0.3)

    # ── Cross-learner panels ──
    def _draw_wins(self, ax, history):
        ax.clear()
        if history:
            wins = {entry["id"]: entry["wins"] for entry in history}
            races = {entry["id"]: entry["races"] for entry in history}
            ids = list(wins.keys())
            x = np.arange(len(ids))
            ax.bar(x - 0.2, [wins[i] for i in ids], width=0.4,
                   color=[CAR_COLORS.get(i, "gray") for i in ids],
                   label="wins")
            ax.bar(x + 0.2, [races[i] - wins[i] for i in ids], width=0.4,
                   color="lightgray", label="losses")
            ax.set_xticks(x)
            ax.set_xticklabels(ids)
            ax.set_ylabel("count")
            ax.set_title("Cumulative race tally")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Waiting for race data…",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Cumulative race tally")

    def _draw_streak(self, ax, history):
        ax.clear()
        if history:
            ids = [e["id"] for e in history]
            streaks = [e["streak"] for e in history]
            paused = [e["paused"] for e in history]
            x = np.arange(len(ids))
            colors = ["red" if p else CAR_COLORS.get(i, "gray")
                      for i, p in zip(ids, paused)]
            ax.bar(x, streaks, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{i}{'*' if p else ''}"
                                 for i, p in zip(ids, paused)])
            ax.set_ylabel("Win streak")
            ax.set_title("Current win streak (* = paused)")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Waiting for race data…",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Current win streak")

    def _draw_info(self, ax, history, cfg, monitors, metrics_per_car):
        ax.clear()
        ax.axis("off")
        ax.set_title("Run Info", fontsize=11, fontweight="bold")

        lines = [f"Run: {os.path.basename(self.run_dir)}"]
        if cfg:
            lines.append(f"Algo: {cfg.get('algo', '?').upper()}")
            lines.append(f"Chunk: {cfg.get('chunk_timesteps', '?'):,} steps")
            lines.append(f"Eval/chunk: {cfg.get('eval_races_per_chunk', '?')}")
            lines.append("")
        if history:
            lines.append("Latest block:")
            for e in history:
                lines.append(f"  {e['id']}: {e['wins']:>3}/{e['races']:>3} wins"
                             f"  streak={e['streak']}  paused={e['paused']}"
                             f"  steps={e['timesteps']:,}")
            lines.append("")
        for cid in CAR_IDS:
            mon = monitors.get(cid)
            if mon is not None and len(mon) > 0:
                r = mon["r"].values
                last = r[-1]
                avg50 = float(np.mean(r[-50:]))
                lines.append(f"{cid}:  eps={len(r)}  last_r={last:.1f}  "
                             f"avg50={avg50:.1f}")
            else:
                lines.append(f"{cid}:  no episodes yet")
        # DR sample (latest)
        lines.append("")
        for cid in CAR_IDS:
            metrics = metrics_per_car.get(cid)
            dr = (metrics or {}).get("dr", []) if metrics else []
            if dr:
                last = dr[-1]
                pretty = ", ".join(
                    f"{k}={last[k]:.2f}" if isinstance(last.get(k), (int, float))
                    else f"{k}={last[k]}"
                    for k in last if k != "timestep")
                lines.append(f"{cid} DR: {pretty}")
                break  # one is enough — both cars share DR per episode

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=7.5, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

    # ── Tick ──
    def update(self, _frame=None):
        monitors = {cid: _load_monitor(self.run_dir, cid) for cid in CAR_IDS}
        metrics = {cid: _load_metrics(self.run_dir, cid) for cid in CAR_IDS}
        history = _load_history(self.run_dir)
        cfg = _load_config(self.run_dir)

        for cid in CAR_IDS:
            self._draw_episode_reward(self.ax_rew[cid], cid, monitors[cid])
            self._draw_components(self.ax_comp[cid], cid, metrics[cid])
            self._draw_episode_length(self.ax_len[cid], cid, monitors[cid])

        self._draw_wins(self.ax_wins, history)
        self._draw_streak(self.ax_streak, history)
        self._draw_info(self.ax_info, history, cfg, monitors, metrics)

        # Status bar
        parts = []
        for cid in CAR_IDS:
            mon = monitors[cid]
            if mon is not None and len(mon) > 0:
                parts.append(f"{cid}: {len(mon)} eps, "
                              f"last_r={mon['r'].values[-1]:.1f}")
            else:
                parts.append(f"{cid}: 0 eps")
        parts.append(f"smooth={self.smooth_window}")
        parts.append(time.strftime("%H:%M:%S"))
        self.status_text.set_text("  |  ".join(parts))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        self.update()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.update)
        timer.start()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRI_RaceBot live training dashboard")
    parser.add_argument("--run", required=True, help="Run directory under runs/")
    args = parser.parse_args()

    if not os.path.isdir(args.run):
        os.makedirs(args.run, exist_ok=True)

    Dashboard(args.run).run()
