"""
Integration test: run the *actual* Tracker against the synthetic camera for
a couple hundred frames, headless, with no Unity client attached. This
doesn't check geometric accuracy in detail (that's what
test_pose_estimation.py is for) -- it's a smoke test that the whole
pipeline (optical flow tracking -> two-view init -> PnP tracking ->
keyframe-based re-triangulation -> map growth) actually runs end-to-end
without crashing and produces a sane-looking result. This is exactly the
kind of test that would have caught real bugs during development (e.g. an
index mismatch between track_ids and track_pts after a dropped-track
filter) well before ever touching a real webcam or Unity.
"""

from __future__ import annotations

from slam.camera import CameraIntrinsics
from slam.map import Map
from slam.synthetic import SyntheticCamera
from slam.tracker import Tracker, TrackerState


def test_synthetic_pipeline_initializes_and_grows_map():
    intr = CameraIntrinsics.guess_for_resolution(640, 480, hfov_deg=70.0)
    cam = SyntheticCamera(intr, seed=0)
    tracker = Tracker(intr, Map())

    reached_tracking = False
    max_points_seen = 0

    for _ in range(300):
        ok, frame = cam.read()
        assert ok
        result = tracker.process_frame(frame)
        if result.state == TrackerState.TRACKING:
            reached_tracking = True
        max_points_seen = max(max_points_seen, len(tracker.map.points))

    assert reached_tracking, "tracker never left INITIALIZING against the synthetic scene"
    assert max_points_seen > 30, f"map stayed suspiciously small: {max_points_seen} points"
    assert len(tracker.map.keyframes) >= 2


def test_tracker_survives_a_reset_cleanly():
    intr = CameraIntrinsics.guess_for_resolution(320, 240, hfov_deg=70.0)
    cam = SyntheticCamera(intr, seed=1)
    tracker = Tracker(intr, Map())

    for _ in range(60):
        ok, frame = cam.read()
        tracker.process_frame(frame)

    # simulate the 'r' hotkey in main.py: throw away the tracker and map,
    # keep reading frames, make sure nothing about shared state (track id
    # counters, camera object, etc.) leaks in a way that breaks a fresh run.
    tracker = Tracker(intr, Map())
    for _ in range(60):
        ok, frame = cam.read()
        assert ok
        tracker.process_frame(frame)  # must not raise
