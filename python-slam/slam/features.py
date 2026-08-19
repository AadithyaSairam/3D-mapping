"""
features.py — finding points worth tracking, and following them across frames.

Monocular SLAM has nothing to work with except "this pixel neighbourhood in
frame A corresponds to that pixel neighbourhood in frame B." Everything
else (pose, structure) is derived from those correspondences. So the very
first real algorithmic choice in this project is: how do we find and
maintain correspondences?

There are two broad families:

1. Detect-and-match every frame: run a feature detector (ORB, SIFT, ...) on
   every frame independently, then match descriptors between frames with a
   matcher (brute-force Hamming distance for binary descriptors like ORB).
   Robust to large motion, but relatively slow and the matches are noisier.

2. Detect-once, track-with-optical-flow: run a detector only on "keyframes",
   then use Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) to follow
   those exact pixel locations frame-to-frame. Much cheaper (no descriptor
   computation/matching per frame) and gives smoother, sub-pixel-accurate
   tracks — which is what real-time monocular VO systems (e.g. SVO, and the
   frontend of ORB-SLAM's tracking-only mode) lean on. The tradeoff: a
   tracked point can silently drift onto the wrong object, and long tracks
   need periodic re-detection as points are lost off-screen or through
   occlusion.

This project uses approach (2) for the live per-frame tracking loop (it's
what makes real-time performance possible on a plain webcam + CPU), and
falls back to ORB detect+match (approach 1) whenever we need to (re-)find
correspondences from scratch — e.g. map initialization, and relocalizing
after we've lost too many tracked points.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Tunables. These are the knobs you'd turn first if tracking felt too sparse
# (raise MAX_CORNERS / lower QUALITY_LEVEL) or too jittery/wrong
# (raise QUALITY_LEVEL, raise MIN_DISTANCE).
MAX_CORNERS = 800
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 8
BLOCK_SIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

ORB_N_FEATURES = 2000


def detect_shi_tomasi(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Find "good features to track" (Shi-Tomasi corners) — the points we hand
    to optical flow. Corners are preferred over flat/edge regions because
    optical flow is only well-constrained where the image gradient varies
    in more than one direction (the classic aperture problem: a plain edge
    only pins down motion perpendicular to itself).

    Returns an (N, 2) float32 array of (x, y) pixel coordinates, or an
    (0, 2) array if none were found.
    """
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
        mask=mask,
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_optical_flow(
    prev_gray: np.ndarray, cur_gray: np.ndarray, prev_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Follow `prev_pts` from `prev_gray` into `cur_gray` with pyramidal
    Lucas-Kanade optical flow.

    Returns (cur_pts, valid_mask) where cur_pts has the same shape as
    prev_pts (stale/invalid entries are left as-is — always check
    valid_mask) and valid_mask is a boolean array marking which points were
    tracked successfully.

    Lucas-Kanade assumes brightness constancy and small motion between
    frames — both hold well for a webcam at 30fps unless someone whips the
    camera around, which is exactly the failure mode "quick camera motion
    loses tracking" you may notice while using this.
    """
    if len(prev_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    prev_pts_cv = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    cur_pts_cv, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        prev_pts_cv,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )
    cur_pts = cur_pts_cv.reshape(-1, 2)
    valid = status.reshape(-1).astype(bool)

    # Also reject points that flowed outside the frame — cv2 doesn't do this
    # for us, and a point sitting at a negative pixel coordinate will quietly
    # poison any downstream geometry (essential matrix / PnP) that uses it.
    h, w = cur_gray.shape[:2]
    in_bounds = (cur_pts[:, 0] >= 0) & (cur_pts[:, 0] < w) & (cur_pts[:, 1] >= 0) & (cur_pts[:, 1] < h)
    valid = valid & in_bounds

    return cur_pts, valid


@dataclass
class OrbMatch:
    """A single ORB descriptor match between two frames."""

    query_idx: int
    train_idx: int
    distance: float


def detect_and_match_orb(
    gray_a: np.ndarray, gray_b: np.ndarray, ratio_test: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect ORB keypoints+descriptors independently in both images and match
    them with a brute-force Hamming matcher + Lowe's ratio test.

    Used for "cold start" correspondence finding: map initialization (first
    two keyframes) and relocalization, where we don't have a previous
    optical-flow track to lean on.

    Returns (pts_a, pts_b): matched pixel coordinates, same length, in
    correspondence order (pts_a[i] <-> pts_b[i]).
    """
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        # Lowe's ratio test: a match is only trustworthy if it's clearly
        # better than the *second*-best candidate match. If the best two
        # matches are nearly tied, the match is ambiguous (e.g. a repeating
        # texture) and we'd rather throw it away than feed noise into pose
        # estimation.
        if m.distance < ratio_test * n.distance:
            good.append(m)

    if len(good) < 8:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b
