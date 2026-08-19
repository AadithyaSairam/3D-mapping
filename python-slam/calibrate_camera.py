#!/usr/bin/env python3
"""
calibrate_camera.py — recover your webcam's intrinsics from a checkerboard.

WHY THIS MATTERS
-----------------
Monocular SLAM recovers 3D structure purely from how points move across the
image plane. That math (the essential matrix, triangulation, PnP) is only
correct in *normalized* camera coordinates — i.e. after undoing your
specific lens's focal length, optical centre, and distortion. Skipping
calibration and using a "close enough" guess is the #1 reason a from-scratch
SLAM pipeline silently produces a warped, drifting map: the errors don't
crash anything, they just quietly bend your geometry.

HOW IT WORKS
------------
Print (or display on a second screen) a standard OpenCV checkerboard
pattern of known square size. A checkerboard is used (rather than any
random textured object) because its corners can be found to sub-pixel
accuracy and its geometry is perfectly known and planar — that lets OpenCV
solve, in closed form, for the K matrix and distortion coefficients that
best explain many observed views of the same known pattern.

USAGE
-----
    python calibrate_camera.py --rows 6 --cols 9 --square-size 0.024

Default checkerboard: 9x6 *internal corners* (i.e. a 10x7 squares board),
24mm squares — this is the printable OpenCV standard pattern:
https://github.com/opencv/opencv/blob/4.x/doc/pattern.png

Controls while running:
    SPACE  - capture the current frame as a calibration sample
             (only works once a checkerboard is detected)
    c      - run calibration once you have >= 10 good samples
    q      - quit without calibrating

Output: python-slam/camera_intrinsics.json (loaded by main.py automatically).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera import CameraIntrinsics

OUTPUT_PATH = Path(__file__).parent / "camera_intrinsics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--rows", type=int, default=6, help="internal corners, short side")
    parser.add_argument("--cols", type=int, default=9, help="internal corners, long side")
    parser.add_argument("--square-size", type=float, default=0.024, help="checkerboard square size in metres")
    parser.add_argument("--min-samples", type=int, default=12)
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)

    # Build the "object points" for one view of the board: the known 3D
    # position of every internal corner in the board's own coordinate
    # frame (Z = 0, since the board is planar), scaled to real-world metres.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    object_points: list[np.ndarray] = []  # 3D points, one array per sample
    image_points: list[np.ndarray] = []  # matching 2D detections, one array per sample

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera_id}")

    print(f"Looking for a {args.cols}x{args.rows} internal-corner checkerboard.")
    print("SPACE = capture sample, c = calibrate, q = quit")

    frame_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(
                gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            display = frame.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, pattern_size, corners, found)

            cv2.putText(
                display,
                f"samples: {len(object_points)}/{args.min_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow("calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and found:
                object_points.append(objp.copy())
                image_points.append(corners)
                print(f"captured sample {len(object_points)}")
            elif key == ord("c"):
                if len(object_points) < args.min_samples:
                    print(f"need at least {args.min_samples} samples, have {len(object_points)}")
                    continue
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(object_points) < 4:
        raise SystemExit("Not enough samples captured; re-run and press SPACE more.")

    assert frame_size is not None
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, frame_size, None, None
    )

    print(f"\nCalibration RMS reprojection error: {rms:.4f} px (lower is better; <1.0 is good)")
    print(f"K =\n{K}")
    print(f"dist_coeffs = {dist.ravel()}")

    intrinsics = CameraIntrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=frame_size[0],
        height=frame_size[1],
        dist_coeffs=dist.ravel(),
    )
    intrinsics.save(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
