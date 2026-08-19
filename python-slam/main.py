#!/usr/bin/env python3
"""
main.py — entry point: capture frames, run the SLAM tracker, stream to Unity.

Usage:
    python main.py --source webcam                 # real webcam, camera 0
    python main.py --source synthetic               # no camera needed, for testing/demos
    python main.py --source video --video path.mp4

Requires camera_intrinsics.json (run calibrate_camera.py first for a real
webcam) -- for --source synthetic, intrinsics are generated to match the
synthetic scene automatically.

While running, a debug window shows the current frame with tracked points
overlaid (green = mapped, yellow = untriangulated) and basic stats. Press
'q' to quit, 'r' to reset the tracker/map (also tells any connected Unity
client to clear its point cloud).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from slam.camera import CameraIntrinsics
from slam.map import Map
from slam.streamer import MapStreamer, StreamerConfig
from slam.synthetic import SyntheticCamera
from slam.tracker import Tracker, TrackerState

INTRINSICS_PATH = Path(__file__).parent / "camera_intrinsics.json"


def build_capture(args, intrinsics_hint: CameraIntrinsics | None):
    if args.source == "synthetic":
        intr = CameraIntrinsics.guess_for_resolution(args.width, args.height, hfov_deg=70.0)
        return SyntheticCamera(intr, seed=args.seed), intr

    if args.source == "video":
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"Could not open video source ({args.source})")

    if intrinsics_hint is not None:
        intr = intrinsics_hint
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        print("WARNING: no camera_intrinsics.json found -- using a rough guess.")
        print("         Run calibrate_camera.py for real results. See docs/THEORY.md.")
        intr = CameraIntrinsics.guess_for_resolution(w, h)
    return cap, intr


def draw_debug_overlay(frame, tracker: Tracker, fps: float) -> np.ndarray:
    vis = frame.copy()
    idx_by_id = {int(i): k for k, i in enumerate(tracker.track_ids.tolist())}
    for tid, k in idx_by_id.items():
        pt = tracker.track_pts[k]
        mapped = tid in tracker.track_map_id
        color = (0, 220, 0) if mapped else (0, 220, 220)
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, color, -1)

    lines = [
        f"state: {tracker.state.name}",
        f"live tracks: {len(tracker.track_ids)}   map points: {len(tracker.map.points)}   keyframes: {len(tracker.map.keyframes)}",
        f"fps: {fps:.1f}",
        "q=quit  r=reset",
    ]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def run(args) -> None:
    intrinsics_hint = None
    if args.source != "synthetic" and INTRINSICS_PATH.exists():
        intrinsics_hint = CameraIntrinsics.load(INTRINSICS_PATH)
        print(f"Loaded calibrated intrinsics from {INTRINSICS_PATH}")

    cap, intr = build_capture(args, intrinsics_hint)

    streamer = None
    if not args.no_stream:
        streamer = MapStreamer(StreamerConfig(port=args.port))
        streamer.start()

    tracker = Tracker(intr, Map())

    frame_idx = 0
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream / camera read failed.")
                break

            if intrinsics_hint is not None and args.source != "synthetic":
                frame = intr.undistort(frame)

            result = tracker.process_frame(frame)
            frame_idx += 1

            if streamer is not None:
                if result.pose_R is not None:
                    streamer.send_pose(frame_idx, result.pose_R, result.pose_t)
                if result.new_points:
                    streamer.send_points([(p.id, p.position, p.color) for p in result.new_points])
                if frame_idx % 15 == 0:
                    streamer.send_status(result.state.name, len(tracker.map.points), len(tracker.map.keyframes))

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            if not args.headless:
                vis = draw_debug_overlay(frame, tracker, fps)
                cv2.imshow("3d-mapping (debug view)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    tracker = Tracker(intr, Map())
                    if streamer is not None:
                        streamer.send_reset()
                    print("--- tracker + map reset ---")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if streamer is not None:
            streamer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["webcam", "synthetic", "video"], default="webcam")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--video", type=str, default=None, help="path to a video file when --source video")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=8765, help="TCP port the Unity client connects to")
    parser.add_argument("--no-stream", action="store_true", help="disable the Unity TCP streamer entirely")
    parser.add_argument("--headless", action="store_true", help="disable the local debug window (e.g. for servers/CI)")
    parser.add_argument("--seed", type=int, default=0, help="synthetic scene seed, --source synthetic only")
    args = parser.parse_args()

    if args.source == "video" and not args.video:
        parser.error("--source video requires --video PATH")

    run(args)


if __name__ == "__main__":
    main()
