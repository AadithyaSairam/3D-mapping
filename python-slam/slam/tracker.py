"""
tracker.py — the main loop: ties features + pose_estimation + triangulation
+ map together into an actual (simplified) monocular visual SLAM system.

THE BIG IDEA: two-view init, then PnP tracking (not chained essential matrices)
--------------------------------------------------------------------------------
A tempting-but-wrong design is: for every consecutive frame pair, run
essential-matrix pose estimation (pose_estimation.py) and just chain the
resulting relative poses together. This is called "frame-to-frame VO", and
it drifts badly, for two compounding reasons:
  1. Every relative pose has its own *independent, arbitrary* scale
     (see pose_estimation.py's docstring) — chaining them means the map's
     scale changes randomly every frame instead of staying fixed.
  2. Every relative pose has its own estimation error, and those errors
     accumulate without bound (there's nothing pulling the trajectory back
     towards ground truth).

This project instead does what real monocular SLAM systems (ORB-SLAM and
its relatives) do:

  1. INITIALIZE ONCE: run two-view pose estimation exactly once, between
     the first frame and a later frame with enough parallax (camera
     baseline). Triangulate the matched points -> this fixes the map's
     scale, permanently, as "whatever scale that first triangulation
     produced." Every point added later is triangulated *consistently*
     with that same scale (see triangulation.py).

  2. TRACK AGAINST THE MAP: for every frame after that, don't estimate
     pose from 2D<->2D correspondences at all. Instead, solve the
     Perspective-n-Point problem (`cv2.solvePnPRansac`): given >=6 pairs
     of (already-triangulated 3D map point, its 2D pixel in the current
     frame), directly solve for the current camera pose. This is
     fundamentally more stable, because it's anchored to a fixed set of
     3D points instead of to the (also-moving, also-uncertain) previous
     frame's pose estimate. This is "PnP tracking", and it's the
     workhorse of essentially every real-time visual SLAM/VO system.

  3. GROW THE MAP AT KEYFRAMES: periodically (see `_should_insert_keyframe`),
     triangulate any tracked-but-not-yet-mapped points between the
     previous keyframe and the current one, and add them to the map, so
     tracking has fresh points to work with as old ones leave the frame.

WHAT'S DELIBERATELY NOT HERE (see docs/THEORY.md for the full list)
----------------------------------------------------------------------
- Bundle adjustment (jointly re-optimizing all keyframe poses + map points
  together). Real systems use this constantly to fight drift; this project
  triangulates once and never revisits a point's position.
- Loop closure (recognizing you've returned to a previously-mapped place
  and snapping the accumulated drift shut).
- Relocalization after fully losing tracking (this project just coasts on
  the last known pose and keeps trying).
These are exactly the "next steps if you want to go further" in the docs —
understanding *why* a from-scratch system without them drifts is itself
one of the most useful things this project teaches.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from .camera import CameraIntrinsics
from .features import detect_shi_tomasi, track_optical_flow, detect_and_match_orb
from .map import KeyFrame, Map
from .pose_estimation import estimate_two_view_pose
from .triangulation import triangulate_points, cheirality_and_reprojection_mask

# --- tunables -----------------------------------------------------------
MIN_INIT_PARALLAX_PX = 25.0     # median pixel motion required before we attempt two-view init
MIN_TRACKS_FOR_INIT = 40
MIN_PNP_CORRESPONDENCES = 6      # cv2.solvePnP's hard minimum is effectively this
PNP_REPROJ_ERROR_PX = 8.0
MAX_LIVE_TRACKS = 800
REPLENISH_BELOW = 250            # top the track pool back up once it drops below this
KEYFRAME_MIN_FRAME_GAP = 5
KEYFRAME_MAX_FRAME_GAP = 30
KEYFRAME_MIN_PARALLAX_PX = 14.0
INIT_STALL_FRAMES = 90           # re-anchor initialization if it hasn't succeeded in this many frames


class TrackerState(Enum):
    INITIALIZING = auto()
    TRACKING = auto()
    LOST = auto()


@dataclass
class NewPoint:
    id: int
    position: np.ndarray
    color: tuple[int, int, int]


@dataclass
class FrameResult:
    state: TrackerState
    pose_R: np.ndarray | None  # world-to-camera rotation, or None if no pose yet
    pose_t: np.ndarray | None
    is_keyframe: bool
    new_points: list[NewPoint] = field(default_factory=list)
    num_live_tracks: int = 0
    num_map_correspondences: int = 0


class Tracker:
    def __init__(self, intrinsics: CameraIntrinsics, map_: Map | None = None):
        self.intr = intrinsics
        self.K = intrinsics.K
        self.map = map_ if map_ is not None else Map()

        self.state = TrackerState.INITIALIZING
        self.prev_gray: np.ndarray | None = None

        self._next_track_id = itertools.count()
        self.track_ids = np.zeros((0,), dtype=np.int64)
        self.track_pts = np.zeros((0, 2), dtype=np.float32)
        self.track_map_id: dict[int, int] = {}  # track_id -> MapPoint id, once triangulated

        # Pixel positions of each live track *as observed at the last
        # keyframe* (or, before init succeeds, at the pending init anchor
        # frame). This is the "pts_a" side of the next triangulation.
        self.last_kf_pts: dict[int, np.ndarray] = {}
        self.last_kf_R = np.eye(3)
        self.last_kf_t = np.zeros(3)
        self.frames_since_init_anchor = 0
        self.frame_count = 0

        self.current_R = np.eye(3)
        self.current_t = np.zeros(3)

    # -- public API --------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        if self.prev_gray is None:
            self._start_init_anchor(gray)
            self.prev_gray = gray
            return FrameResult(self.state, None, None, False, num_live_tracks=len(self.track_ids))

        self._advance_tracks_with_optical_flow(gray)

        if self.state == TrackerState.INITIALIZING:
            result = self._try_initialize(gray, frame_bgr)
        else:
            result = self._track_with_pnp(gray, frame_bgr)

        self.prev_gray = gray
        return result

    # -- initialization ------------------------------------------------

    def _start_init_anchor(self, gray: np.ndarray) -> None:
        pts = detect_shi_tomasi(gray)
        ids = np.array([next(self._next_track_id) for _ in range(len(pts))], dtype=np.int64)
        self.track_ids = ids
        self.track_pts = pts
        self.last_kf_pts = {int(i): p.copy() for i, p in zip(ids, pts)}
        self.last_kf_R = np.eye(3)
        self.last_kf_t = np.zeros(3)
        self.frames_since_init_anchor = 0

    def _try_initialize(self, gray: np.ndarray, frame_bgr: np.ndarray) -> FrameResult:
        self.frames_since_init_anchor += 1
        common_ids = [i for i in self.track_ids.tolist() if i in self.last_kf_pts]

        if len(common_ids) < MIN_TRACKS_FOR_INIT:
            if self.frames_since_init_anchor > INIT_STALL_FRAMES:
                self._start_init_anchor(gray)  # stalled -- re-anchor on a fresher frame
            return FrameResult(self.state, None, None, False, num_live_tracks=len(self.track_ids))

        idx_by_id = {int(i): k for k, i in enumerate(self.track_ids.tolist())}
        pts_a = np.array([self.last_kf_pts[i] for i in common_ids], dtype=np.float32)
        pts_b = np.array([self.track_pts[idx_by_id[i]] for i in common_ids], dtype=np.float32)

        parallax = float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))
        if parallax < MIN_INIT_PARALLAX_PX:
            if self.frames_since_init_anchor > INIT_STALL_FRAMES:
                self._start_init_anchor(gray)
            return FrameResult(self.state, None, None, False, num_live_tracks=len(self.track_ids))

        pose = estimate_two_view_pose(pts_a, pts_b, self.K)
        if pose is None or pose.inlier_mask.sum() < MIN_TRACKS_FOR_INIT // 2:
            if self.frames_since_init_anchor > INIT_STALL_FRAMES:
                self._start_init_anchor(gray)
            return FrameResult(self.state, None, None, False, num_live_tracks=len(self.track_ids))

        # Success: kf0 = identity (world origin), kf1 = recovered pose.
        inlier_ids = [common_ids[k] for k in range(len(common_ids)) if pose.inlier_mask[k]]
        inlier_pts_a = pts_a[pose.inlier_mask]
        inlier_pts_b = pts_b[pose.inlier_mask]

        R0, t0 = np.eye(3), np.zeros(3)
        R1, t1 = pose.R, pose.t.reshape(3)

        points_3d = triangulate_points(self.K, R0, t0, R1, t1, inlier_pts_a, inlier_pts_b)
        good = cheirality_and_reprojection_mask(points_3d, self.K, R0, t0, R1, t1, inlier_pts_a, inlier_pts_b)

        new_points: list[NewPoint] = []
        for k, ok in enumerate(good):
            if not ok:
                continue
            tid = inlier_ids[k]
            u, v = inlier_pts_b[k]
            color = tuple(int(c) for c in frame_bgr[int(np.clip(v, 0, frame_bgr.shape[0] - 1)),
                                                       int(np.clip(u, 0, frame_bgr.shape[1] - 1))])
            pid = self.map.add_point(points_3d[k], color)
            self.track_map_id[tid] = pid
            new_points.append(NewPoint(pid, points_3d[k], color))

        if len(new_points) < 8:
            # Geometrically valid but degenerate result (e.g. near-planar
            # scene fooling the cheirality check) -- try a fresh anchor.
            self._start_init_anchor(gray)
            return FrameResult(self.state, None, None, False, num_live_tracks=len(self.track_ids))

        kf0 = KeyFrame(id=len(self.map.keyframes), R=R0, t=t0)
        self.map.add_keyframe(kf0)
        kf1 = KeyFrame(id=len(self.map.keyframes), R=R1, t=t1)
        self.map.add_keyframe(kf1)

        self.current_R, self.current_t = R1, t1
        self.state = TrackerState.TRACKING
        self._reset_keyframe_baseline(R1, t1)
        self._replenish_tracks(gray)

        return FrameResult(
            self.state, R1, t1, True, new_points=new_points, num_live_tracks=len(self.track_ids)
        )

    # -- steady-state tracking ------------------------------------------

    def _track_with_pnp(self, gray: np.ndarray, frame_bgr: np.ndarray) -> FrameResult:
        idx_by_id = {int(i): k for k, i in enumerate(self.track_ids.tolist())}
        corr_ids = [tid for tid in self.track_map_id if tid in idx_by_id]

        object_points = np.array([self.map.points[self.track_map_id[t]].position for t in corr_ids], dtype=np.float64)
        image_points = np.array([self.track_pts[idx_by_id[t]] for t in corr_ids], dtype=np.float64)

        pose_found = False
        inlier_set: set[int] = set()
        if len(corr_ids) >= MIN_PNP_CORRESPONDENCES:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points.reshape(-1, 1, 3),
                image_points.reshape(-1, 1, 2),
                self.K,
                None,
                reprojectionError=PNP_REPROJ_ERROR_PX,
                confidence=0.999,
                iterationsCount=200,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if ok and inliers is not None and len(inliers) >= MIN_PNP_CORRESPONDENCES:
                self.current_R, _ = cv2.Rodrigues(rvec)
                self.current_t = tvec.reshape(3)
                pose_found = True
                inlier_set = {corr_ids[i] for i in inliers.reshape(-1)}

        if pose_found:
            self.state = TrackerState.TRACKING
            for tid in corr_ids:
                pid = self.track_map_id[tid]
                if tid in inlier_set:
                    self.map.mark_observed(pid)
                else:
                    self.map.mark_outlier(pid)
                    if pid not in self.map.points:
                        del self.track_map_id[tid]
        else:
            # Coast on the last known pose rather than crashing; a fuller
            # system would attempt relocalization here (see module docstring).
            self.state = TrackerState.LOST

        new_points: list[NewPoint] = []
        is_keyframe = False
        if pose_found and self._should_insert_keyframe(idx_by_id):
            new_points = self._insert_keyframe(gray, frame_bgr, idx_by_id)
            is_keyframe = True

        if len(self.track_ids) < REPLENISH_BELOW:
            self._replenish_tracks(gray)

        return FrameResult(
            self.state,
            self.current_R if pose_found else None,
            self.current_t if pose_found else None,
            is_keyframe,
            new_points=new_points,
            num_live_tracks=len(self.track_ids),
            num_map_correspondences=len(corr_ids),
        )

    def _should_insert_keyframe(self, idx_by_id: dict[int, int]) -> bool:
        gap = self.frame_count - self._last_keyframe_frame_count
        if gap < KEYFRAME_MIN_FRAME_GAP:
            return False
        if gap >= KEYFRAME_MAX_FRAME_GAP:
            return True
        common = [i for i in self.last_kf_pts if i in idx_by_id]
        if not common:
            return False
        disp = [np.linalg.norm(self.track_pts[idx_by_id[i]] - self.last_kf_pts[i]) for i in common]
        return float(np.median(disp)) > KEYFRAME_MIN_PARALLAX_PX

    def _insert_keyframe(self, gray: np.ndarray, frame_bgr: np.ndarray, idx_by_id: dict[int, int]) -> list[NewPoint]:
        candidate_ids = [
            tid for tid in self.last_kf_pts if tid in idx_by_id and tid not in self.track_map_id
        ]
        new_points: list[NewPoint] = []
        if candidate_ids:
            pts_a = np.array([self.last_kf_pts[i] for i in candidate_ids], dtype=np.float32)
            pts_b = np.array([self.track_pts[idx_by_id[i]] for i in candidate_ids], dtype=np.float32)
            points_3d = triangulate_points(
                self.K, self.last_kf_R, self.last_kf_t, self.current_R, self.current_t, pts_a, pts_b
            )
            good = cheirality_and_reprojection_mask(
                points_3d, self.K, self.last_kf_R, self.last_kf_t, self.current_R, self.current_t, pts_a, pts_b
            )
            for k, ok in enumerate(good):
                if not ok:
                    continue
                tid = candidate_ids[k]
                u, v = pts_b[k]
                color = tuple(int(c) for c in frame_bgr[int(np.clip(v, 0, frame_bgr.shape[0] - 1)),
                                                           int(np.clip(u, 0, frame_bgr.shape[1] - 1))])
                pid = self.map.add_point(points_3d[k], color)
                self.track_map_id[tid] = pid
                new_points.append(NewPoint(pid, points_3d[k], color))

        kf = KeyFrame(id=len(self.map.keyframes), R=self.current_R.copy(), t=self.current_t.copy())
        self.map.add_keyframe(kf)
        self._reset_keyframe_baseline(self.current_R, self.current_t)
        return new_points

    def _reset_keyframe_baseline(self, R: np.ndarray, t: np.ndarray) -> None:
        idx_by_id = {int(i): k for k, i in enumerate(self.track_ids.tolist())}
        self.last_kf_pts = {tid: self.track_pts[idx].copy() for tid, idx in idx_by_id.items()}
        self.last_kf_R = R.copy()
        self.last_kf_t = t.copy()
        self._last_keyframe_frame_count = self.frame_count

    _last_keyframe_frame_count = 0

    # -- track pool maintenance ------------------------------------------

    def _advance_tracks_with_optical_flow(self, gray: np.ndarray) -> None:
        assert self.prev_gray is not None
        cur_pts, valid = track_optical_flow(self.prev_gray, gray, self.track_pts)
        self.track_ids = self.track_ids[valid]
        self.track_pts = cur_pts[valid]
        alive = set(self.track_ids.tolist())
        for tid in list(self.track_map_id):
            if tid not in alive:
                del self.track_map_id[tid]
        for tid in list(self.last_kf_pts):
            if tid not in alive:
                del self.last_kf_pts[tid]

    def _replenish_tracks(self, gray: np.ndarray) -> None:
        if len(self.track_ids) >= MAX_LIVE_TRACKS:
            return
        mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
        for pt in self.track_pts:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 8, 0, -1)
        new_pts = detect_shi_tomasi(gray, mask=mask)
        if len(new_pts) == 0:
            return
        room = MAX_LIVE_TRACKS - len(self.track_ids)
        new_pts = new_pts[:room]
        new_ids = np.array([next(self._next_track_id) for _ in range(len(new_pts))], dtype=np.int64)
        self.track_ids = np.concatenate([self.track_ids, new_ids])
        self.track_pts = np.concatenate([self.track_pts, new_pts], axis=0)
        for tid, pt in zip(new_ids, new_pts):
            self.last_kf_pts[int(tid)] = pt.copy()
