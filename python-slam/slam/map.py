"""
map.py — the persistent state that makes this "SLAM" rather than just "VO".

Visual Odometry (VO) only cares about the *current* camera pose, frame to
frame. SLAM additionally builds and maintains a *map* — a persistent set of
3D landmarks — and, crucially, uses that map to constrain future pose
estimates (tracking against the map, not just against the previous frame).
That's what stops errors from compounding as fast as pure frame-to-frame VO
would: the map acts as a shared, stable reference instead of a chain of
independent guesses.

This file defines the two core data structures:

- MapPoint: a single 3D landmark, plus bookkeeping (id, colour for
  visualization, which keyframes observed it, how many times tracking
  succeeded/failed against it — used to prune bad points).
- KeyFrame: a "landmark" frame we keep around permanently (as opposed to
  the many ordinary frames we track through but discard). Keyframes are
  what new map points get triangulated between, and what a real system
  would run bundle adjustment / loop closure over. This project doesn't
  implement either (see docs/THEORY.md "What's not implemented"), but the
  KeyFrame/MapPoint split is exactly the structure you'd extend to add
  them.
- Map: the container tying it together, plus the logic for merging newly
  triangulated points into the running map.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MapPoint:
    id: int
    position: np.ndarray  # (3,) world coordinates
    color: tuple[int, int, int]  # BGR, sampled from the image at creation time
    observations: int = 1  # how many keyframes have seen this point
    outlier_count: int = 0  # how many times PnP flagged this point as a reprojection outlier


@dataclass
class KeyFrame:
    id: int
    R: np.ndarray  # 3x3, world-to-camera rotation
    t: np.ndarray  # (3,), world-to-camera translation
    # For every tracked 2D point that was alive when this keyframe was
    # created, its pixel location and (if triangulated) the MapPoint id it
    # corresponds to. This is what lets us re-triangulate new points
    # between "the last keyframe" and "this keyframe" using the *same*
    # optical-flow tracks, without re-detecting/re-matching features.
    point_ids: dict[int, np.ndarray] = field(default_factory=dict)  # track_id -> pixel (2,)

    @property
    def camera_center(self) -> np.ndarray:
        """World-space position of the camera (not the same as t, which is world->cam)."""
        return -self.R.T @ self.t


class Map:
    """The running SLAM map: all keyframes + all 3D map points."""

    def __init__(self) -> None:
        self.keyframes: list[KeyFrame] = []
        self.points: dict[int, MapPoint] = {}
        self._next_point_id = itertools.count()

    def add_keyframe(self, kf: KeyFrame) -> None:
        self.keyframes.append(kf)

    def add_point(self, position: np.ndarray, color: tuple[int, int, int]) -> int:
        pid = next(self._next_point_id)
        self.points[pid] = MapPoint(id=pid, position=position, color=color)
        return pid

    def mark_outlier(self, point_id: int) -> None:
        p = self.points.get(point_id)
        if p is None:
            return
        p.outlier_count += 1
        # A point that repeatedly fails to reproject correctly during PnP
        # is most likely a bad triangulation (e.g. from a near-degenerate
        # low-parallax pair, or a moving-object mismatch that slipped past
        # RANSAC). Culling it keeps the map -- and therefore future PnP
        # poses -- from being dragged off by persistently-wrong points.
        if p.outlier_count > 5:
            del self.points[point_id]

    def mark_observed(self, point_id: int) -> None:
        p = self.points.get(point_id)
        if p is not None:
            p.observations += 1

    def point_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: (ids, positions Nx3, colors Nx3) for all live points."""
        if not self.points:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8),
            )
        ids = np.array(list(self.points.keys()), dtype=np.int64)
        positions = np.array([self.points[i].position for i in ids], dtype=np.float64)
        colors = np.array([self.points[i].color for i in ids], dtype=np.uint8)
        return ids, positions, colors
