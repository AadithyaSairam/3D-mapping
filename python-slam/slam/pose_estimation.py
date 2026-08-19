"""
pose_estimation.py — recovering camera motion from 2D point correspondences.

THE CORE IDEA (epipolar geometry)
----------------------------------
Take two images of the same static scene from two different camera
positions, and one 3D point P visible in both. Its projections p1, p2 (in
*normalized* camera coordinates, i.e. after applying K^-1) are related by
the essential matrix E:

    p2^T * E * p1 = 0

This is the "epipolar constraint": it says that p2 must lie on a specific
line (the epipolar line) determined by p1 and E, *for any* 3D point,
regardless of its distance from the camera. That's the whole trick — E
encodes the relative rotation R and translation t between the two camera
positions (E = [t]_x * R, where [t]_x is the skew-symmetric cross-product
matrix of t), without ever knowing where the 3D points actually are.

Given >= 5 point correspondences, `cv2.findEssentialMat` solves for E
using RANSAC (to reject outlier matches — mismatches, moving objects,
etc.), and `cv2.recoverPose` decomposes E back into R and t via SVD,
picking the one of four mathematically-valid (R, t) combinations where the
triangulated points actually end up in front of both cameras (the
"cheirality check").

WHY THIS ONLY GETS YOU MOTION *UP TO SCALE*
---------------------------------------------
E is defined up to an arbitrary scalar multiple: doubling every 3D point's
distance from the cameras AND doubling the camera translation produces
*identical* image projections. There's no way to tell, from 2D image
motion alone, whether you moved 1cm past objects 1m away or 1m past
objects 100m away. This is the famous monocular scale ambiguity, and
`recoverPose` always returns `t` as a unit vector.

This module only handles *two-view* pose recovery (used once, to
initialize the map). It does NOT fix the scale-drift problem that comes
from chaining many two-view estimates together — that's why tracker.py
switches to PnP (solvePnPRansac) against the existing 3D map for every
frame after initialization: PnP recovers a metrically consistent pose
because it reasons about a 2D point against an *already-scaled* 3D map
point, not two 2D points against each other. See tracker.py's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TwoViewPose:
    R: np.ndarray  # 3x3 rotation, camera A -> camera B
    t: np.ndarray  # 3x1 translation, unit norm (scale is undefined, see module docstring)
    inlier_mask: np.ndarray  # (N,) bool, which input correspondences agreed with the recovered pose
    essential_matrix: np.ndarray


def estimate_two_view_pose(
    pts_a: np.ndarray, pts_b: np.ndarray, K: np.ndarray
) -> TwoViewPose | None:
    """
    Recover relative pose (R, t) between two views from pixel
    correspondences pts_a <-> pts_b (both (N, 2) arrays, same camera K for
    both — true for a single moving webcam).

    Returns None if there weren't enough good correspondences to solve
    reliably (findEssentialMat needs at least 5, we require more in
    practice for a stable estimate).
    """
    if len(pts_a) < 8:
        return None

    E, mask = cv2.findEssentialMat(
        pts_a,
        pts_b,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,  # max reprojection error (px) to count as an inlier
    )
    if E is None or E.shape != (3, 3):
        return None

    # recoverPose re-runs cheirality checking internally and further
    # narrows the inlier mask to correspondences whose triangulated point
    # is in front of *both* cameras -- necessary because RANSAC's inlier
    # set from findEssentialMat can still admit points that satisfy the
    # epipolar constraint but triangulate behind the camera (one of the
    # other 3 SVD solutions).
    inlier_count, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)

    if inlier_count < 8:
        return None

    inlier_mask = pose_mask.reshape(-1).astype(bool)
    return TwoViewPose(R=R, t=t, inlier_mask=inlier_mask, essential_matrix=E)


def pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack (R, t) into a 4x4 homogeneous transform [R | t; 0 0 0 1]."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform cheaply (R^T, -R^T t) instead of a generic matrix inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
