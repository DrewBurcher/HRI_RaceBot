"""
Top-down centerline editor for the mesh track.

Loads the STL configured in TRACK_CONFIG, applies the same rotation +
translation that MeshTrack uses, and shows the boundary outlines from above.
Click in order to drop waypoints; the script prints a `waypoints` list ready
to paste back into config.py.

    python draw_centerline.py

Controls:
    left click   add waypoint at cursor
    right click  undo last waypoint
    'r'          reset (clear all waypoints)
    'p'          print current waypoint list
    'q' / close  quit and print final waypoint list
"""

from __future__ import annotations

import struct
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import TRACK_CONFIG


def load_stl_triangles(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    tris = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        base = 84 + i * 50 + 12
        tris[i] = np.frombuffer(data[base:base + 36], dtype=np.float32).reshape(3, 3)
    return tris


def boundary_edges_top_face(tris: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (a, b) 2-D edges that bound the top face of the slab.

    The top face is the set of triangles whose three vertices share the slab's
    maximum Y (SolidWorks Y-up). Edges that appear in only one such triangle
    form the boundary loops (outer perimeter + any interior cutouts).
    """
    y_top = float(tris[:, :, 1].max())
    mask = np.all(np.abs(tris[:, :, 1] - y_top) < 1e-3, axis=1)
    top = tris[mask]

    def key(p, tol=1e-3):
        return (int(round(float(p[0]) / tol)), int(round(float(p[2]) / tol)))

    counts: dict = defaultdict(int)
    verts: dict = {}
    for tri in top:
        ks = [key(tri[k]) for k in range(3)]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            e = tuple(sorted([ks[a], ks[b]]))
            counts[e] += 1
            verts[e] = (tri[a], tri[b])

    edges = []
    for e, c in counts.items():
        if c == 1:
            a, b = verts[e]
            edges.append((np.array([a[0], a[2]]), np.array([b[0], b[2]])))
    return edges


def to_world_xy(p_xz: np.ndarray) -> np.ndarray:
    """Convert STL (x, z) into the world (x, y) used by MeshTrack waypoints.

    MeshTrack scales by `mesh_scale`, rotates +π/2 about X (so y' = -z_stl,
    z' = y_stl), then shifts by the configured base position. Supports both
    `base_position` (Dom's-Track keys) and `mesh_position` (older keys).
    """
    x_stl, z_stl = float(p_xz[0]), float(p_xz[1])
    scale = TRACK_CONFIG.get("mesh_scale", [1.0, 1.0, 1.0])
    pos = TRACK_CONFIG.get("base_position",
                           TRACK_CONFIG.get("mesh_position", [0.0, 0.0, 0.0]))
    return np.array([x_stl * float(scale[0]) + float(pos[0]),
                     -z_stl * float(scale[2]) + float(pos[1])])


def main() -> None:
    stl_path = TRACK_CONFIG.get("stl_path", "Track.STL")
    edges = boundary_edges_top_face(load_stl_triangles(stl_path))
    edges_world = [(to_world_xy(a), to_world_xy(b)) for a, b in edges]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_aspect("equal")
    ax.set_title(
        f"Centerline editor — {stl_path}\n"
        "left-click: add  |  right-click: undo  |  r: reset  |  p: print  |  q: quit"
    )
    ax.set_xlabel("x (m, world frame)")
    ax.set_ylabel("y (m, world frame)")

    for a, b in edges_world:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.4", linewidth=0.6)

    waypoints: List[Tuple[float, float]] = []
    (line,) = ax.plot([], [], color="tab:orange", linewidth=2, marker="o", markersize=5)
    (start_marker,) = ax.plot([], [], color="tab:green", marker="o", markersize=10, linestyle="")

    def redraw() -> None:
        if waypoints:
            xs = [p[0] for p in waypoints] + [waypoints[0][0]]
            ys = [p[1] for p in waypoints] + [waypoints[0][1]]
            line.set_data(xs, ys)
            start_marker.set_data([waypoints[0][0]], [waypoints[0][1]])
        else:
            line.set_data([], [])
            start_marker.set_data([], [])
        fig.canvas.draw_idle()

    def print_waypoints() -> None:
        if not waypoints:
            print("(no waypoints yet)")
            return
        print("\n# paste into TRACK_CONFIG['waypoints']:")
        print('"waypoints": [')
        for i in range(0, len(waypoints), 4):
            chunk = waypoints[i:i + 4]
            row = ", ".join(f"[{x:7.3f}, {y:7.3f}]" for x, y in chunk)
            print(f"    {row},")
        print("],\n")

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 1:
            waypoints.append((float(event.xdata), float(event.ydata)))
        elif event.button == 3 and waypoints:
            waypoints.pop()
        redraw()

    def on_key(event) -> None:
        if event.key == "r":
            waypoints.clear()
            redraw()
        elif event.key == "p":
            print_waypoints()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    print_waypoints()


if __name__ == "__main__":
    main()
