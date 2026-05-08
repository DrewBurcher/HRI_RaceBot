"""
Procedural racetrack builder.

Current shape: an oval with a long straight section that's used as the
start/finish line. Built from PyBullet primitives (no external assets) so
it runs anywhere the default `pybullet_data` is available.

The straight runs along +x; the two semicircles bulge in +y / -y. The +y curve
is traversed first when driving in the canonical direction (+x along the
straight).

Designed to be subclassed for additional track shapes later (just override
`_build_geometry` and `centerline`).
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

    # ── Public API ─────────────────────────────────────────────────────
    def build(self) -> None:
        """Load the ground plane and walls into the active physics client."""
        self._load_plane()
        self._build_geometry()

    def spawn_pose(self, lane: int, jitter: float = 0.0
                   ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Pose for a car at the start.

        `lane` is 0 for the inside lane and 1 for the outside lane. The
        starting straight runs from -straight_length/2 to +straight_length/2,
        so the start line itself is at x = -straight_length/2 + jitter.
        """
        from config import CAR_CONFIG

        lane_offset = self.cfg["lane_offset"]
        # inside lane sits closer to the center axis; outside sits further out
        y = -lane_offset if lane == 0 else +lane_offset
        x = -self.straight_length / 2.0 + jitter
        z = CAR_CONFIG["spawn_z"]
        # cars face +x at the start
        yaw = 0.0
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        return (x, y, z), quat

    def random_start_jitter(self, rng: np.random.Generator) -> float:
        """Sample a forward shift along the start straight.

        The shift is the same for both cars (so they stay side-by-side) but
        varies between races so the policies don't memorize a fixed start.
        """
        max_jit = min(self.cfg["start_jitter"], self.straight_length * 0.5 - 1.0)
        if max_jit <= 0.0:
            return 0.0
        return float(rng.uniform(-max_jit / 2.0, max_jit / 2.0))

    def centerline_progress(self, x: float, y: float) -> float:
        """Return cumulative arc-length progress around the oval [0, perimeter).

        Used to compute lap progress and detect lap completion. Walks the four
        track segments in order:
            1. front straight  (start → +x end of straight, y ≈ 0)
            2. +y semicircle    (around (+sl/2, 0))
            3. back straight   (returning along -x)
            4. -y semicircle    (around (-sl/2, 0))
        """
        sl = self.straight_length
        r = self.curve_radius
        seg1 = sl                       # front straight
        seg2 = math.pi * r              # right semicircle
        seg3 = sl                       # back straight
        seg4 = math.pi * r              # left semicircle
        perim = seg1 + seg2 + seg3 + seg4

        # Decide which segment (x, y) lies in based on coarse position.
        if -sl / 2.0 <= x <= sl / 2.0 and y >= -1e-3 and abs(y) < r:
            # near the front straight (positive y side of center)
            if y < r * 0.5:  # treat "on straight" liberally
                return (x + sl / 2.0) % perim
        if x > sl / 2.0:
            # right semicircle around (sl/2, 0)
            dx = x - sl / 2.0
            dy = y
            ang = math.atan2(dy, dx)        # 0 along +x, pi/2 along +y
            # Going counter-clockwise on this side means decreasing angle from
            # +pi/2 (entry) through 0 to -pi/2 (exit). Convert to arc length.
            arc = (math.pi / 2.0 - ang) * r
            arc = max(0.0, min(seg2, arc))
            return seg1 + arc
        if -sl / 2.0 <= x <= sl / 2.0 and y < 0:
            # back straight, traversed in -x direction
            return seg1 + seg2 + (sl / 2.0 - x)
        if x < -sl / 2.0:
            # left semicircle around (-sl/2, 0)
            dx = x + sl / 2.0
            dy = y
            ang = math.atan2(dy, dx)        # pi at +x in this frame
            # Entry from below (-pi/2) to top (+pi/2); arc length grows
            arc = (ang + math.pi / 2.0) * r
            # Note: for the lower-left quadrant, atan2 returns negative, so we
            # clamp to keep the value monotonic.
            arc = max(0.0, min(seg4, arc))
            return seg1 + seg2 + seg3 + arc
        # Fallback — shouldn't normally hit
        return 0.0

    def perimeter(self) -> float:
        return 2.0 * self.straight_length + 2.0 * math.pi * self.curve_radius

    def is_off_track(self, x: float, y: float) -> bool:
        """Cheap analytic off-track check (no contact query needed)."""
        sl = self.straight_length
        r_in = self.inner_radius
        r_out = self.outer_radius
        if -sl / 2.0 <= x <= sl / 2.0:
            return abs(y) < r_in or abs(y) > r_out
        cx = sl / 2.0 if x > 0 else -sl / 2.0
        d = math.hypot(x - cx, y)
        return d < r_in or d > r_out

    # ── Internals ───────────────────────────────────────────────────────
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

        # Two straights (north and south)
        for sign in (+1, -1):
            half_extents = [sl / 2.0, t / 2.0, h / 2.0]
            pos = [0.0, sign * radius, h / 2.0]
            self._add_box(half_extents, pos)

        # Two semicircular arcs at +sl/2 and -sl/2
        seg_len = math.pi * radius / self.num_curve_segments
        for cx_sign in (+1, -1):
            cx = cx_sign * sl / 2.0
            # Angles span -pi/2 .. +pi/2 around the +x semicircle, and
            # +pi/2 .. 3pi/2 around the -x one.
            ang_start = -math.pi / 2.0 if cx_sign > 0 else math.pi / 2.0
            for i in range(self.num_curve_segments):
                a = ang_start + (i + 0.5) * (math.pi / self.num_curve_segments)
                wx = cx + radius * math.cos(a)
                wy = radius * math.sin(a)
                # Tangent direction along the wall
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
            rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=self.client)
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
        """Inverse of centerline_progress: arc-length → (xy, tangent)."""
        sl = self.straight_length
        r = self.curve_radius
        seg1 = sl
        seg2 = math.pi * r
        seg3 = sl
        if s < seg1:                          # front straight (y ≈ 0, +x)
            return (-sl / 2.0 + s, 0.0), (1.0, 0.0)
        s -= seg1
        if s < seg2:                          # +x semicircle
            ang = math.pi / 2.0 - s / r       # decreasing from +pi/2
            x = sl / 2.0 + r * math.cos(ang)
            y = r * math.sin(ang)
            tan_ang = ang - math.pi / 2.0     # tangent points around the curve
            return (x, y), (math.cos(tan_ang), math.sin(tan_ang))
        s -= seg2
        if s < seg3:                          # back straight (-x, y < 0)
            return (sl / 2.0 - s, -0.0), (-1.0, 0.0)
        s -= seg3                             # -x semicircle
        ang = -math.pi / 2.0 + s / r
        x = -sl / 2.0 + r * math.cos(ang)
        y = r * math.sin(ang)
        tan_ang = ang - math.pi / 2.0
        return (x, y), (math.cos(tan_ang), math.sin(tan_ang))


def build_track(client: int) -> OvalTrack:
    """Factory — add new shapes by branching on TRACK_CONFIG['shape'] later."""
    shape = TRACK_CONFIG.get("shape", "oval")
    if shape == "oval":
        track = OvalTrack(client)
        track.build()
        return track
    raise ValueError(f"Unknown track shape: {shape}")
