"""
triangulation.py — turning a 2D correspondence + two known camera poses into a 3D point.

Once we know both cameras' poses (R, t for each, i.e. their full 3x4
projection matrices P = K [R | t]), a single correspondence p1 <-> p2 no
longer has the one-parameter ambiguity it had during pose estimation:
each 2D point defines a ray through 3D space from its camera centre, and
(in the noise-free case) the two rays intersect at exactly one 3D point.
In practice, due to pixel-measurement noise, the two rays are skew (don't
exactly meet), so we solve for the point that minimizes reprojection error
in a least-squares sense — that's exactly what `cv2.triangulatePoints`
does (via the direct linear transform / DLT method).

This is also where a monocular system establishes its very first notion of
scale: whatever translation magnitude two-view pose estimation guessed
(recall t is unit-norm, see pose_estimation.py) becomes "1 unit" for the
whole map, because every subsequent triangulated point inherits that same
implicit scale. This is why tracker.py deliberately never re-runs
two-view triangulation between arbitrary frame pairs during normal
operation — that would silently introduce a *different* arbitrary scale
each time and make the map internally inconsistent. Scale is fixed once,
at map initialization.
"""

from __future__ import annotations

import cv2
import numpy as np


def triangulate_points(
    K: np.ndarray,
    R_a: np.ndarray,
    t_a: np.ndarray,
    R_b: np.ndarray,
    t_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> np.ndarray:
    """
    Triangulate 3D points (in world coordinates) from matched 2D pixel
    observations in two views with known poses.

    (R_a, t_a), (R_b, t_b): world-to-camera transforms for each view
        (i.e. X_cam = R @ X_world + t — the same convention cv2.recoverPose
        and cv2.solvePnP use).
    pts_a, pts_b: (N, 2) matched pixel coordinates, same convention as
        pose_estimation.estimate_two_view_pose.

    Returns an (N, 3) array of triangulated world points. Points that
    ended up behind either camera are NOT filtered here -- call
    `cheirality_mask` and apply it yourself, since what counts as "valid"
    also usually includes a reprojection-error check that's cheaper to do
    once, after triangulation, together with the cheirality check.
    """
    P_a = K @ np.hstack([R_a, t_a.reshape(3, 1)])
    P_b = K @ np.hstack([R_b, t_b.reshape(3, 1)])

    pts_a_h = pts_a.reshape(-1, 2).T.astype(np.float64)  # 2xN
    pts_b_h = pts_b.reshape(-1, 2).T.astype(np.float64)

    points_4d = cv2.triangulatePoints(P_a, P_b, pts_a_h, pts_b_h)  # 4xN homogeneous
    points_3d = (points_4d[:3] / points_4d[3]).T  # back to Nx3 Euclidean
    return points_3d


def cheirality_and_reprojection_mask(
    points_3d: np.ndarray,
    K: np.ndarray,
    R_a: np.ndarray,
    t_a: np.ndarray,
    R_b: np.ndarray,
    t_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    max_reproj_error_px: float = 4.0,
    min_depth: float = 1e-3,
    max_depth: float = 1000.0,
) -> np.ndarray:
    """
    Decide which triangulated points are actually trustworthy.

    A triangulated point is kept only if ALL of the following hold:
      1. It's in front of camera A (positive depth in camera A's frame).
      2. It's in front of camera B (positive depth in camera B's frame).
      3. Its depth isn't absurd (guards against near-degenerate
         triangulation from two views with almost no parallax between
         them, which blows up numerically).
      4. Projecting it back into both images lands close to the pixels we
         started from (reprojection error) -- this is the single best
         signal for "this correspondence was actually correct" and is the
         same check bundle-adjustment-based SLAM systems use as their loss
         function.

    Returns an (N,) boolean mask.
    """
    n = len(points_3d)
    keep = np.ones(n, dtype=bool)

    for R, t, pts in ((R_a, t_a, pts_a), (R_b, t_b, pts_b)):
        cam_pts = (R @ points_3d.T).T + t.reshape(1, 3)
        depth = cam_pts[:, 2]
        keep &= (depth > min_depth) & (depth < max_depth)

        # reprojection
        proj, _ = cv2.projectPoints(points_3d.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3, 1), K, None)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - pts.reshape(-1, 2), axis=1)
        keep &= err < max_reproj_error_px

    return keep
