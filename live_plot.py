"""
Live training dashboard for HRI_RaceBot (Parallel SAC Edition).

Spawned as a subprocess by train.py. Tails:
    runs/<run>/metrics_<car_id>.json            (reward components, losses, DR, eps length)
    runs/<run>/config_snapshot.json             (run config; written at end)

Layout (3 rows × 3 cols):
    Row 1  | car_0 episode reward | car_1 episode reward | cumulative wins
    Row 2  | car_0 reward parts   | car_1 reward parts   | domain randomization
    Row 3  | car_0 episode length | car_1 episode length | run info / status

Usage:
    python live_plot.py --run runs/duo_sac_<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
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

def _load_metrics(run_dir: str, car_id: str):
    path = os.path.join(run_dir, f"metrics_{car_id}.json")
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

class Dashboard:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.smooth_window = 20

        self.fig = plt.figure(figsize=(18, 11))
        self.fig.canvas.manager.set_window_title(f"HRI_RaceBot — {os.path.basename(run_dir)}")

        gs = gridspec.GridSpec(4, 3, figure=self.fig,
                                height_ratios=[1, 1, 1, 0.06],
                                hspace=0.45, wspace=0.32)
        
        self.ax_rew = {
            "car_0": self.fig.add_subplot(gs[0, 0]),
            "car_1": self.fig.add_subplot(gs[0, 1]),
        }
        self.ax_wins = self.fig.add_subplot(gs[0, 2])
        self.ax_comp = {
            "car_0": self.fig.add_subplot(gs[1, 0]),
            "car_1": self.fig.add_subplot(gs[1, 1]),
        }
        self.ax_dr = self.fig.add_subplot(gs[1, 2])
        self.ax_len = {
            "car_0": self.fig.add_subplot(gs[2, 0]),
            "car_1": self.fig.add_subplot(gs[2, 1]),
        }
        self.ax_info = self.fig.add_subplot(gs[2, 2])

        ax_slider = self.fig.add_axes([0.06, 0.02, 0.30, 0.018])
        self.smooth_slider = Slider(ax_slider, "Smooth", 1, 100, valinit=20, valstep=1)
        self.smooth_slider.on_changed(self._on_smooth)

        self.ax_status = self.fig.add_axes([0.40, 0.005, 0.58, 0.030])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0, 0.5, "", fontsize=8, fontfamily="monospace", verticalalignment="center")

        self.fig.suptitle(f"HRI_RaceBot Training Dashboard — {os.path.basename(run_dir)}", fontsize=14, fontweight="bold")

    def _on_smooth(self, val):
        self.smooth_window = int(val)

    def _get_reward_keys(self, rcs):
        """Dynamically scan ALL episodes to find every key that has occurred."""
        if not rcs:
            return []
        
        all_keys = set()
        for episode_data in rcs:
            all_keys.update(episode_data.keys())
            
        ignore = {"episode", "timestep", "ep_length", "flipped", "is_winner"}
        return sorted([k for k in all_keys if k not in ignore])

    def _draw_episode_reward(self, ax, car_id: str, metrics):
        ax.clear()
        rcs = (metrics or {}).get("reward_components", [])
        if len(rcs) > 0:
            eps = [e.get("episode", i) for i, e in enumerate(rcs)]
            keys = self._get_reward_keys(rcs)
            
            # Reconstruct total ep_reward from per-step means
            r = []
            for e in rcs:
                ep_len = e.get("ep_length", 1)
                ep_r = sum(e.get(k, 0.0) for k in keys) * ep_len
                r.append(ep_r)

            ax.plot(eps, r, alpha=0.18, color=CAR_COLORS[car_id], linewidth=0.5)
            ax.plot(eps, _smooth(r, self.smooth_window), color=CAR_COLORS[car_id], linewidth=2)
        else:
            ax.text(0.5, 0.5, f"Waiting for {car_id} data…", ha="center", va="center", transform=ax.transAxes)
            
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"{car_id} — Episode Reward")
        ax.grid(True, alpha=0.3)

    def _draw_episode_length(self, ax, car_id: str, metrics):
        ax.clear()
        rcs = (metrics or {}).get("reward_components", [])
        if len(rcs) > 0:
            eps = [e.get("episode", i) for i, e in enumerate(rcs)]
            l = [e.get("ep_length", 0) for e in rcs]
            ax.plot(eps, l, alpha=0.18, color=CAR_COLORS[car_id], linewidth=0.5)
            ax.plot(eps, _smooth(l, self.smooth_window), color=CAR_COLORS[car_id], linewidth=2)
        else:
            ax.text(0.5, 0.5, f"Waiting for {car_id} data…", ha="center", va="center", transform=ax.transAxes)
            
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title(f"{car_id} — Episode Length")
        ax.grid(True, alpha=0.3)

    def _draw_components(self, ax, car_id: str, metrics):
        ax.clear()
        rcs = (metrics or {}).get("reward_components", [])
        
        # Render immediately on episode 1
        if len(rcs) > 0:
            eps = [e.get("episode", i) for i, e in enumerate(rcs)]
            keys = self._get_reward_keys(rcs)
            
            for k in keys:
                vals = np.array([e.get(k, 0.0) for e in rcs])
                
                # Removed the zero-value check so keys always show if they exist in the JSON
                color = COMPONENT_COLORS.get(k, None)
                ax.plot(eps, _smooth(vals, self.smooth_window), color=color, linewidth=1.5, label=k)
                
            if keys:
                ax.legend(fontsize=6, loc="best", ncol=2)
        else:
            ax.text(0.5, 0.5, "Waiting for components…", ha="center", va="center", transform=ax.transAxes)
            
        ax.set_xlim(left=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean reward / step")
        ax.set_title(f"{car_id} — Reward Components")
        ax.grid(True, alpha=0.3)

    def _draw_wins(self, ax, metrics_per_car):
        ax.clear()
        wins = {}
        races = {}
        for cid, metrics in metrics_per_car.items():
            rcs = (metrics or {}).get("reward_components", [])
            races[cid] = len(rcs)
            wins[cid] = sum(1 for e in rcs if e.get("is_winner"))
            
        if any(races.values()):
            ids = list(wins.keys())
            x = np.arange(len(ids))
            ax.bar(x - 0.2, [wins[i] for i in ids], width=0.4, color=[CAR_COLORS.get(i, "gray") for i in ids], label="wins")
            ax.bar(x + 0.2, [races[i] - wins[i] for i in ids], width=0.4, color="lightgray", label="losses")
            ax.set_xticks(x)
            ax.set_xticklabels(ids)
            ax.set_ylabel("count")
            ax.set_title("Cumulative Race Tally")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Waiting for race data…", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Cumulative Race Tally")

    def _draw_dr(self, ax, metrics_per_car):
        ax.clear()
        dr_data = (metrics_per_car.get("car_0") or {}).get("dr", [])
        if dr_data and len(dr_data) > 0:
            steps = [e.get("timestep", 0) for e in dr_data]
            keys = [k for k in dr_data[0].keys() if k != "timestep"]
            
            for k in keys:
                vals = np.array([e.get(k, 0.0) for e in dr_data])
                base = vals[0]
                if abs(base) > 1e-6:
                    pct = ((vals - base) / abs(base)) * 100.0
                else:
                    pct = vals * 100.0
                ax.plot(steps, pct, label=k, linewidth=1.5)
                
            ax.set_title("Domain Randomization (% Deviation)")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("% Change from Start")
            ax.legend(fontsize=6, loc="best", ncol=2)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Waiting for DR data…", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Domain Randomization")

    def _draw_info(self, ax, cfg, metrics_per_car):
        ax.clear()
        ax.axis("off")
        ax.set_title("Run Info", fontsize=11, fontweight="bold")

        lines = [f"Run: {os.path.basename(self.run_dir)}"]
        if cfg:
            lines.append(f"Algo: {cfg.get('algo', 'sac').upper()}")
        lines.append("Mode: Parallel Continuous")
        lines.append("")
        
        for cid in CAR_IDS:
            metrics = metrics_per_car.get(cid)
            rcs = (metrics or {}).get("reward_components", [])
            if rcs:
                eps = len(rcs)
                wins = sum(1 for e in rcs if e.get("is_winner"))
                keys = self._get_reward_keys(rcs)
                
                recent = rcs[-50:]
                avg50 = 0
                for e in recent:
                    ep_len = e.get("ep_length", 1)
                    avg50 += sum(e.get(k, 0.0) for k in keys) * ep_len
                avg50 = avg50 / len(recent) if recent else 0

                last_e = rcs[-1]
                last_len = last_e.get("ep_length", 1)
                last_r = sum(last_e.get(k, 0.0) for k in keys) * last_len

                lines.append(f"{cid}: eps={eps}  wins={wins}")
                lines.append(f"      last_r={last_r:.1f}  avg50={avg50:.1f}")
            else:
                lines.append(f"{cid}: no episodes yet")
        lines.append("")

        dr = (metrics_per_car.get("car_0") or {}).get("dr", [])
        if dr:
            last = dr[-1]
            pretty = ", ".join(f"{k}={last[k]:.2f}" if isinstance(last.get(k), (int, float)) else f"{k}={last[k]}" for k in last if k != "timestep")
            lines.append(f"DR: {pretty}")

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    def update(self, _frame=None):
        metrics = {cid: _load_metrics(self.run_dir, cid) for cid in CAR_IDS}
        cfg = _load_config(self.run_dir)

        for cid in CAR_IDS:
            self._draw_episode_reward(self.ax_rew[cid], cid, metrics[cid])
            self._draw_components(self.ax_comp[cid], cid, metrics[cid])
            self._draw_episode_length(self.ax_len[cid], cid, metrics[cid])

        self._draw_wins(self.ax_wins, metrics)
        self._draw_dr(self.ax_dr, metrics)
        self._draw_info(self.ax_info, cfg, metrics)

        parts = []
        for cid in CAR_IDS:
            rcs = (metrics[cid] or {}).get("reward_components", [])
            if rcs:
                keys = self._get_reward_keys(rcs)
                last_len = rcs[-1].get("ep_length", 1)
                last_r = sum(rcs[-1].get(k, 0.0) for k in keys) * last_len
                parts.append(f"{cid}: {len(rcs)} eps, last_r={last_r:.1f}")
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