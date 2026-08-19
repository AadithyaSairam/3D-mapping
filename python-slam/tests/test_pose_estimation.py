"""
Unit tests for pose_estimation.py and triangulation.py against *known*
synthetic ground truth. This is the standard way to validate geometric
vision code: construct a scene + cameras where you know the exact right
answer, project points into synthetic images, and check the recovered
pose/structure matches (up to the ambiguities the algorithm is honestly
subject to -- e.g. monocular scale, see pose_estimation.py's docstring).
"""

from __future__ import annotations

import numpy as np
import cv2

from slam.pose_estimation import estimate_two_view_pose
from slam.triangulation import triangulate_points, cheirality_and_reprojection_mask


def make_K(width=640, height=480, fx=500.0):
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)


def project(K, R, t, points_world):
    proj, _ = cv2.projectPoints(points_world.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3, 1), K, None)
    return proj.reshape(-1, 2)


def random_scene_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # A cloud of points a few metres in front of the origin, spread out
    # enough to give well-conditioned epipolar geometry (not coplanar/degenerate).
    pts = rng.uniform(low=[-2, -2, 3], high=[2, 2, 8], size=(n, 3))
    return pts


def test_two_view_pose_recovers_known_rotation_and_translation_direction():
    K = make_K()
    points = random_scene_points()

    R_true = cv2.Rodrigues(np.array([0.02, 0.15, -0.01]))[0]  # small known rotation
    t_true_direction = np.array([1.0, 0.05, 0.1])
    t_true_direction /= np.linalg.norm(t_true_direction)
    baseline = 0.5  # metres -- irrelevant to recovery, since recoverPose only gives direction

    R_a, t_a = np.eye(3), np.zeros(3)
    R_b, t_b = R_true, t_true_direction * baseline

    pts_a = project(K, R_a, t_a, points)
    pts_b = project(K, R_b, t_b, points)

    # keep only points that land inside the image in both views (a real
    # detector would only ever hand us those anyway)
    in_view = (
        (pts_a[:, 0] >= 0) & (pts_a[:, 0] < 640) & (pts_a[:, 1] >= 0) & (pts_a[:, 1] < 480) &
        (pts_b[:, 0] >= 0) & (pts_b[:, 0] < 640) & (pts_b[:, 1] >= 0) & (pts_b[:, 1] < 480)
    )
    pts_a, pts_b = pts_a[in_view], pts_b[in_view]
    assert len(pts_a) > 20, "test scene not well-conditioned, check the fixture"

    pose = estimate_two_view_pose(pts_a, pts_b, K)
    assert pose is not None

    # Rotation should be recovered almost exactly (rotation isn't subject
    # to the scale ambiguity -- only translation is).
    R_err = pose.R.T @ R_true
    angle_err_deg = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    assert angle_err_deg < 1.0, f"rotation error too large: {angle_err_deg:.3f} deg"

    # Translation direction should match (recoverPose returns unit-norm t,
    # so we can only compare direction, not magnitude -- see module docstring).
    t_recovered = pose.t.reshape(3) / np.linalg.norm(pose.t)
    cos_sim = float(np.dot(t_recovered, t_true_direction))
    assert cos_sim > 0.99, f"translation direction off: cos_sim={cos_sim:.4f}"


def test_triangulation_recovers_known_3d_points():
    K = make_K()
    points = random_scene_points(n=50, seed=1)

    R_a, t_a = np.eye(3), np.zeros(3)
    R_b, t_b = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))[0], np.array([0.3, 0.0, 0.0])

    pts_a = project(K, R_a, t_a, points)
    pts_b = project(K, R_b, t_b, points)

    recovered = triangulate_points(K, R_a, t_a, R_b, t_b, pts_a, pts_b)
    err = np.linalg.norm(recovered - points, axis=1)
    assert np.max(err) < 1e-3, f"triangulation error too large: max={np.max(err):.6f}"


def test_cheirality_mask_rejects_points_behind_camera():
    K = make_K()
    # One point in front, one deliberately placed behind camera B by
    # constructing a bogus "triangulated" point there directly.
    good_point = np.array([[0.1, 0.0, 4.0]])
    behind_point = np.array([[0.0, 0.0, -3.0]])
    points_3d = np.concatenate([good_point, behind_point], axis=0)

    R_a, t_a = np.eye(3), np.zeros(3)
    R_b, t_b = np.eye(3), np.array([0.2, 0.0, 0.0])

    pts_a = project(K, R_a, t_a, points_3d)
    pts_b = project(K, R_b, t_b, points_3d)

    mask = cheirality_and_reprojection_mask(points_3d, K, R_a, t_a, R_b, t_b, pts_a, pts_b)
    assert mask[0] == True  # noqa: E712 - explicit bool comparison reads clearer here
    assert mask[1] == False  # noqa: E712
