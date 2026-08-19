"""
camera.py — camera intrinsics: what they are and why monocular SLAM needs them.

A pinhole camera maps a 3D point (X, Y, Z) in *camera coordinates* to a 2D
pixel (u, v) via the intrinsic matrix K:

    [u]   [fx  0  cx] [X/Z]
    [v] = [ 0 fy  cy] [Y/Z]
    [1]   [ 0  0   1] [ 1 ]

- fx, fy : focal length in pixels (how "zoomed in" the lens is, expressed
  in the same units as pixel coordinates rather than mm — this is what
  lets us mix optics with image measurements).
- cx, cy : the pixel coordinate of the principal point (usually close to
  the image centre, but not exactly, because of small lens/sensor
  misalignment).

Every piece of geometry in this project — the essential matrix, PnP,
triangulation — assumes pixel coordinates have *already* been converted
into "normalized" camera coordinates using K, and that lens distortion has
been removed. If K is wrong, every downstream pose and 3D point is subtly
(or badly) wrong: this is why the very first step of any monocular SLAM
project is calibration, not SLAM itself.

Lens distortion (radial + tangential) is modelled separately by OpenCV as
a 5 (or 8/12/14)-element `dist_coeffs` vector and is corrected with
`cv2.undistort` / `cv2.undistortPoints` before anything else happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics + distortion, with convenience helpers."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        """The 3x3 intrinsic matrix, as used directly by OpenCV."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def guess_for_resolution(cls, width: int, height: int, hfov_deg: float = 70.0) -> "CameraIntrinsics":
        """
        A rough-but-usable intrinsics guess when you haven't calibrated yet.

        Real SLAM needs real calibration (run calibrate_camera.py!), but a
        guess based on a plausible horizontal field-of-view is good enough
        to get the pipeline running end-to-end while you set that up, and
        is exactly what most webcams look like (~60-75 deg HFOV).

            fx = (width / 2) / tan(hfov / 2)

        We assume square pixels (fx == fy), which is true for essentially
        every consumer webcam sensor.
        """
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    def save(self, path: str | Path) -> None:
        data = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "dist_coeffs": np.asarray(self.dist_coeffs).tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        data = json.loads(Path(path).read_text())
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            width=data["width"],
            height=data["height"],
            dist_coeffs=np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64),
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.undistort(image, self.K, self.dist_coeffs)
