"""
Procedural racetrack builder.

Current shape: a stadium oval. Two parallel straights run along the x axis,
joined by two semicircles on the left (-x) and right (+x) ends. The drivable
area is the ring between an inner wall (radius `curve_radius - track_width/2`)
and an outer wall (radius `curve_radius + track_width/2`).

Centerline traversal order (counter-clockwise looking down the +z axis):
    1. Top straight    : (-sl/2, +r) -> (+sl/2, +r), tangent +x
    2. Right semicircle: around (+sl/2, 0), top -> bottom
    3. Bottom straight : (+sl/2, -r) -> (-sl/2, -r), tangent -x
    4. Left semicircle : around (-sl/2, 0), bottom -> top

Cars start on the top straight, side-by-side, facing +x.

Designed to be subclassed for additional track shapes later (override
`_build_geometry`, `centerline_progress`, `_point_at_arclength`, `spawn_pose`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from config import TRACK_CONFIG


@dataclass
class Checkpoint:
    """A virtual gate the cars must cross in order to score lap progress."""
    index: int
    position: Tuple[float, float]   # (x, y) on the centerline
    tangent: Tuple[float, float]    # unit forward direction at this point


class OvalTrack:
    """Oval racetrack: two parallel straights joined by two semicircles.

    Coordinate frame:
        * x runs along the straights
        * y is lateral
        * Inner radius = curve_radius - track_width/2
        * Outer radius = curve_radius + track_width/2
    """

    def __init__(self, client: int, cfg: dict | None = None):
        self.client = client
        self.cfg = cfg if cfg is not None else TRACK_CONFIG

        self.straight_length = self.cfg["straight_length"]
        self.curve_radius = self.cfg["curve_radius"]
        self.track_width = self.cfg["track_width"]
        self.wall_height = self.cfg["wall_height"]
        self.wall_thickness = self.cfg["wall_thickness"]
        self.num_curve_segments = self.cfg["num_curve_segments"]

        self.inner_radius = self.curve_radius - self.track_width / 2.0
        self.outer_radius = self.curve_radius + self.track_width / 2.0

        self.body_ids: List[int] = []
        self.checkpoints: List[Checkpoint] = self._build_checkpoints(
            self.cfg["checkpoint_count"])

    # 1. Public API
    def build(self) -> None:
        """Load the ground plane, walls, and visuals into the active physics client."""
        self._load_plane()
        self._build_geometry()
        self._draw_centerline_visualization()
        self._build_start_line()

    def _draw_centerline_visualization(self,
                                        color: Tuple[float, float, float] = (1.0, 0.85, 0.0),
                                        width: float = 3.0,
                                        n_segments: int = 96) -> None:
        """Draw the centerline as PyBullet debug lines (visual only)."""
        perim = self.perimeter()
        pts: List[Tuple[float, float, float]] = []
        for i in range(n_segments + 1):
            s = (i % n_segments) * perim / n_segments
            (x, y), _ = self._point_at_arclength(s)
            pts.append((x, y, 0.02))
        for a, b in zip(pts[:-1], pts[1:]):
            try:
                p.addUserDebugLine(list(a), list(b),
                                    lineColorRGB=list(color),
                                    lineWidth=width,
                                    lifeTime=0,
                                    physicsClientId=self.client)
            except Exception:
                break

    def spawn_pose(self, lane: int, jitter: float = 0.0
                   ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Pose for a car at the start of the top straight."""
        from config import CAR_CONFIG

        lane_offset = self.cfg["lane_offset"]
        if lane == 0:
            y = self.curve_radius - lane_offset
        else:
            y = self.curve_radius + lane_offset
        x = -self.straight_length / 2.0 + jitter
        z = CAR_CONFIG["spawn_z"]
        yaw = 0.0
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        return (x, y, z), quat

    def random_start_jitter(self, rng: np.random.Generator) -> float:
        """Sample a forward shift along the start straight."""
        max_jit = min(self.cfg["start_jitter"], self.straight_length * 0.5 - 1.0)
        if max_jit <= 0.0:
            return 0.0
        return float(rng.uniform(-max_jit / 2.0, max_jit / 2.0))

    def centerline_progress(self, x: float, y: float) -> float:
        """Return cumulative arc-length progress around the oval [0, perimeter)."""
        sl = self.straight_length
        r = self.curve_radius
        seg1 = sl
        seg2 = math.pi * r
        seg3 = sl

        if x > sl / 2.0:
            ang = math.atan2(y, x - sl / 2.0)
            arc = (math.pi / 2.0 - ang) * r
            arc = max(0.0, min(seg2, arc))
            return seg1 + arc
        if x < -sl / 2.0:
            ang = math.atan2(y, x + sl / 2.0)
            if ang > 0:
                ang -= 2.0 * math.pi
            arc = (-math.pi / 2.0 - ang) * r
            arc = max(0.0, min(math.pi * r, arc))
            return seg1 + seg2 + seg3 + arc
        if y >= 0:
            return max(0.0, min(seg1, x + sl / 2.0))
        return seg1 + seg2 + max(0.0, min(seg3, sl / 2.0 - x))

    def perimeter(self) -> float:
        return 2.0 * self.straight_length + 2.0 * math.pi * self.curve_radius

    def closest_centerline_point(self, x: float, y: float
                                  ) -> Tuple[float, float]:
        """Project (x, y) onto the stadium centerline analytically."""
        sl = self.straight_length
        r = self.curve_radius
        if x > sl / 2.0:
            dx = x - sl / 2.0
            d = math.hypot(dx, y)
            if d < 1e-6:
                return sl / 2.0 + r, 0.0
            return sl / 2.0 + r * dx / d, r * y / d
        if x < -sl / 2.0:
            dx = x + sl / 2.0
            d = math.hypot(dx, y)
            if d < 1e-6:
                return -sl / 2.0 - r, 0.0
            return -sl / 2.0 + r * dx / d, r * y / d
        return x, (r if y >= 0 else -r)

    def is_off_track(self, x: float, y: float) -> bool:
        """Cheap analytic off-track check."""
        sl = self.straight_length
        r_in = self.inner_radius
        r_out = self.outer_radius
        if -sl / 2.0 <= x <= sl / 2.0:
            return abs(y) < r_in or abs(y) > r_out
        cx = sl / 2.0 if x > 0 else -sl / 2.0
        d = math.hypot(x - cx, y)
        return d < r_in or d > r_out

    # 2. Internals
    def _load_plane(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.body_ids.append(plane)

    def _build_geometry(self) -> None:
        """Build inner and outer wall rings."""
        self._build_wall_ring(self.inner_radius)
        self._build_wall_ring(self.outer_radius)

    def _build_wall_ring(self, radius: float) -> None:
        """Place wall blocks along the two straights and two semicircles."""
        h = self.wall_height
        t = self.wall_thickness
        sl = self.straight_length

        for sign in (+1, -1):
            half_extents = [sl / 2.0, t / 2.0, h / 2.0]
            pos = [0.0, sign * radius, h / 2.0]
            self._add_box(half_extents, pos)

        seg_len = math.pi * radius / self.num_curve_segments
        for cx_sign in (+1, -1):
            cx = cx_sign * sl / 2.0
            ang_start = -math.pi / 2.0 if cx_sign > 0 else math.pi / 2.0
            for i in range(self.num_curve_segments):
                a = ang_start + (i + 0.5) * (math.pi / self.num_curve_segments)
                wx = cx + radius * math.cos(a)
                wy = radius * math.sin(a)
                tan = a + math.pi / 2.0
                half_extents = [seg_len / 2.0, t / 2.0, h / 2.0]
                quat = p.getQuaternionFromEuler([0.0, 0.0, tan])
                self._add_box(half_extents, [wx, wy, h / 2.0], quat)

    def _add_box(self, half_extents, position, orientation=None) -> int:
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.7, 0.7, 0.7, 0.3], physicsClientId=self.client) # Alpha transparency added
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.client,
        )
        self.body_ids.append(body)
        return body
        
    def _build_start_line(self) -> None:
        """Place a visual start line on the top straight."""
        half_extents = [0.1, self.track_width / 2.0, 0.01]
        pos = [-self.straight_length / 2.0, self.curve_radius, 0.01]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=self.client)
        p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, basePosition=pos, physicsClientId=self.client)

    def _build_checkpoints(self, n: int) -> List[Checkpoint]:
        """Sample n evenly-spaced points along the centerline."""
        perim = self.perimeter()
        cps: List[Checkpoint] = []
        for i in range(n):
            s = i * perim / n
            pos, tan = self._point_at_arclength(s)
            cps.append(Checkpoint(index=i, position=pos, tangent=tan))
        return cps

    def _point_at_arclength(self, s: float
                            ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Inverse of centerline_progress: arc-length -> ((x, y), tangent)."""
        sl = self.straight_length
        r = self.curve_radius
        seg1 = sl
        seg2 = math.pi * r
        seg3 = sl
        if s < seg1:
            return (-sl / 2.0 + s, r), (1.0, 0.0)
        s -= seg1
        if s < seg2:
            ang = math.pi / 2.0 - s / r
            x = sl / 2.0 + r * math.cos(ang)
            y = r * math.sin(ang)
            tan_ang = ang - math.pi / 2.0
            return (x, y), (math.cos(tan_ang), math.sin(tan_ang))
        s -= seg2
        if s < seg3:
            return (sl / 2.0 - s, -r), (-1.0, 0.0)
        s -= seg3
        ang = -math.pi / 2.0 - s / r
        x = -sl / 2.0 + r * math.cos(ang)
        y = r * math.sin(ang)
        tan_ang = ang - math.pi / 2.0
        return (x, y), (math.cos(tan_ang), math.sin(tan_ang))


def build_track(client: int) -> OvalTrack:
    """Factory builder for track generation."""
    shape = TRACK_CONFIG.get("shape", "oval")
    if shape == "oval":
        track = OvalTrack(client)
        track.build()
        return track
    raise ValueError(f"Unknown track shape: {shape}")