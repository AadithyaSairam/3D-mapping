"""
synthetic.py — a fake "webcam" for testing/demoing the SLAM pipeline without a camera.

Why this exists: it's genuinely useful to be able to run and demo the
entire pipeline (tracking, mapping, streaming to Unity) in an environment
with no webcam attached (e.g. a CI runner, a cloud sandbox, or just
debugging on a headless machine) -- and it doubles as an integration test
with a *known ground truth*, since we control the camera trajectory and
the 3D scene exactly.

We build a small synthetic "room": a textured floor, four textured walls,
and a scattering of small textured "objects" (as randomly colored/patterned
patches), all as 3D points with fixed colors. A virtual camera moves along
a smooth, slowly-orbiting path through the room. Each call to `read()`
projects the currently-visible 3D points into a rendered 2D image using
the exact same pinhole model the real tracker assumes, adds a little pixel
noise, and returns a BGR frame -- indistinguishable, as far as slam/*.py is
concerned, from a real cv2.VideoCapture frame.
"""

from __future__ import annotations

import numpy as np
import cv2

from .camera import CameraIntrinsics


def _make_textured_room(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (points_xyz (N,3), colors_bgr (N,3) uint8) for a simple textured room."""
    points = []
    colors = []

    def add_plane(origin, u_axis, v_axis, u_extent, v_extent, density, base_color, noise=25):
        n = int(u_extent * v_extent * density)
        us = rng.uniform(-u_extent / 2, u_extent / 2, n)
        vs = rng.uniform(-v_extent / 2, v_extent / 2, n)
        pts = origin[None, :] + us[:, None] * u_axis[None, :] + vs[:, None] * v_axis[None, :]
        cols = np.clip(
            np.array(base_color, dtype=np.float64)[None, :] + rng.normal(0, noise, (n, 3)), 0, 255
        ).astype(np.uint8)
        points.append(pts)
        colors.append(cols)

    room = 6.0  # metres, room is roughly room x room x 3
    # Floor (y = 0)
    add_plane(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), room, room, 60, (90, 110, 90))
    # Ceiling
    add_plane(np.array([0, 2.6, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), room, room, 30, (200, 200, 200))
    # Four walls
    add_plane(np.array([0, 1.3, room / 2]), np.array([1, 0, 0]), np.array([0, 1, 0]), room, 2.6, 40, (150, 130, 110))
    add_plane(np.array([0, 1.3, -room / 2]), np.array([1, 0, 0]), np.array([0, 1, 0]), room, 2.6, 40, (110, 130, 150))
    add_plane(np.array([room / 2, 1.3, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), room, 2.6, 40, (130, 150, 110))
    add_plane(np.array([-room / 2, 1.3, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), room, 2.6, 40, (150, 110, 130))

    # A handful of "furniture" blobs scattered around, as small dense clusters
    for _ in range(6):
        center = np.array(
            [rng.uniform(-room / 2 + 0.5, room / 2 - 0.5), rng.uniform(0.2, 1.2), rng.uniform(-room / 2 + 0.5, room / 2 - 0.5)]
        )
        cluster = center[None, :] + rng.normal(0, 0.25, (150, 3))
        color = rng.integers(40, 220, 3)
        points.append(cluster)
        colors.append(np.tile(color, (150, 1)).astype(np.uint8))

    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


class SyntheticCamera:
    """Drop-in replacement for cv2.VideoCapture, backed by a synthetic scene."""

    def __init__(self, intrinsics: CameraIntrinsics, seed: int = 0, speed: float = 1.0):
        rng = np.random.default_rng(seed)
        self.intr = intrinsics
        self.points, self.colors = _make_textured_room(rng)
        self.t = 0.0
        self.speed = speed
        self.point_radius_px = 1.6

    def read(self) -> tuple[bool, np.ndarray]:
        self.t += 0.02 * self.speed
        R, t = self._camera_pose(self.t)

        img = np.full((self.intr.height, self.intr.width, 3), 30, dtype=np.uint8)
        cam_pts = (R @ self.points.T).T + t[None, :]
        in_front = cam_pts[:, 2] > 0.05
        proj, _ = cv2.projectPoints(
            self.points[in_front].reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3, 1), self.intr.K, None
        )
        proj = proj.reshape(-1, 2)
        cols = self.colors[in_front]

        h, w = img.shape[:2]
        in_view = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
        # depth-sort so nearer points draw on top of farther ones
        depths = cam_pts[in_front][:, 2]
        order = np.argsort(-depths[in_view])
        pts_draw = proj[in_view][order].astype(int)
        cols_draw = cols[in_view][order]

        for (u, v), (b, g, r) in zip(pts_draw, cols_draw):
            cv2.circle(img, (int(u), int(v)), int(self.point_radius_px), (int(b), int(g), int(r)), -1)

        return True, img

    def _camera_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """A smooth orbiting/wandering trajectory through the synthetic room, as (R, cam_pos)."""
        radius = 1.6
        cam_pos = np.array([radius * np.cos(t * 0.4), 1.2 + 0.15 * np.sin(t * 0.9), radius * np.sin(t * 0.4)])
        look_at = np.array([0.3 * np.sin(t * 0.2), 1.0, 0.3 * np.cos(t * 0.2)])

        forward = look_at - cam_pos
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        # Camera-to-world rotation columns are (right, -up, forward) for an
        # OpenCV-style camera (y down, z forward); invert for world->cam.
        R_cam_to_world = np.stack([right, -cam_up, forward], axis=1)
        R_world_to_cam = R_cam_to_world.T
        t_world_to_cam = -R_world_to_cam @ cam_pos
        return R_world_to_cam, t_world_to_cam

    def release(self) -> None:
        pass

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2.VideoCapture's API
        return True
