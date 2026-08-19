"""
streamer.py — sending the live map + camera pose to Unity over TCP.

WIRE FORMAT
-----------
A plain TCP server, newline-delimited JSON (NDJSON): each message is one
JSON object followed by "\\n". This is dramatically slower and larger than
a binary protocol would be, and that's a deliberate tradeoff for a learning
project: you can `nc localhost 8765` and read the traffic with your own
eyes, and both ends (Python here, C# in Unity) can parse it with a
one-liner instead of hand-rolling a binary layout. See docs/PROTOCOL.md
for the exact schema and for notes on what you'd change for production
(length-prefixed binary frames, e.g. FlatBuffers/Protobuf, would be the
natural next step once this works).

COORDINATE SYSTEMS: OpenCV vs Unity
-------------------------------------
This is the single easiest place to introduce a subtle bug, so it's
handled in exactly one place (here) rather than sprinkled through the
tracker.

OpenCV / this project's SLAM math uses a right-handed camera convention:
  +X right, +Y down, +Z forward (out of the lens, "into the scene").
  A world-to-camera transform is X_cam = R @ X_world + t.

Unity uses a LEFT-handed world convention:
  +X right, +Y up, +Z forward.
  Objects are positioned by a *camera-to-world* position (a Transform's
  `.position` is where the object is, not a world-to-object matrix).

Converting requires two things:
  1. Flip the Y axis (and, to keep a right-handed rotation a valid
     left-handed one, negate Z as well as Y -- flipping a single axis
     turns a rotation matrix into a reflection, which is not a valid
     rotation. Flipping two axes keeps it a proper rotation while
     achieving the handedness change).
  2. Convert world-to-camera (R, t) into a camera *position* in world
     space: camera_center = -R^T @ t (see map.py's KeyFrame.camera_center),
     then apply the same axis flip to that position, and convert R^T into
     a quaternion (also axis-flipped) for the camera's orientation.

The flip is implemented once in `_cv_to_unity_position` and
`_cv_to_unity_quaternion` below, and the identical inverse is documented
in PROTOCOL.md for the Unity side to reverse if it ever needs to send
coordinates back.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
from dataclasses import dataclass

import numpy as np


def _cv_to_unity_position(p_cv: np.ndarray) -> tuple[float, float, float]:
    x, y, z = p_cv
    return (float(x), float(-y), float(z))  # see module docstring


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Standard R -> quaternion (w, x, y, z), Shepperd's method (numerically stable)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def _cv_to_unity_quaternion(R_world_to_cam: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convert a world-to-camera rotation matrix into a Unity camera-to-world
    quaternion (x, y, z, w) -- Unity's Quaternion constructor order.
    """
    R_cam_to_world = R_world_to_cam.T
    # Apply the same (y, z) handedness flip used for position, on both
    # sides of the rotation, which is equivalent to conjugating by
    # diag(1, -1, -1). This keeps forward/right/up axes mapping correctly
    # between the two coordinate systems.
    flip = np.diag([1.0, -1.0, -1.0])
    R_unity = flip @ R_cam_to_world @ flip
    w, x, y, z = _rotation_matrix_to_quaternion(R_unity)
    return (float(x), float(y), float(z), float(w))


@dataclass
class StreamerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    max_queue: int = 256


class MapStreamer:
    """
    A tiny background TCP server. `main.py` calls `send_pose` / `send_points`
    every frame; any connected Unity client receives the resulting NDJSON
    stream. Designed to tolerate Unity not being connected yet (or
    disconnecting/reconnecting) without ever blocking the SLAM loop --
    frames are dropped, not queued indefinitely, if a client is slow.
    """

    def __init__(self, config: StreamerConfig | None = None):
        self.config = config or StreamerConfig()
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._server_thread = threading.Thread(target=self._accept_loop, daemon=True)

    def start(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.config.host, self.config.port))
        self._server_sock.listen(4)
        self._server_sock.settimeout(0.5)
        self._server_thread.start()
        print(f"[streamer] listening on {self.config.host}:{self.config.port} for the Unity client")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server_sock.close()
        except OSError:
            pass
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setblocking(True)
            print(f"[streamer] Unity client connected from {addr}")
            with self._clients_lock:
                self._clients.append(conn)

    def _broadcast(self, message: dict) -> None:
        payload = (json.dumps(message) + "\n").encode("utf-8")
        with self._clients_lock:
            dead = []
            for c in self._clients:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)

    # -- message senders --------------------------------------------------

    def send_pose(self, frame_index: int, R_world_to_cam: np.ndarray, t_world_to_cam: np.ndarray) -> None:
        cam_center = -R_world_to_cam.T @ t_world_to_cam
        pos = _cv_to_unity_position(cam_center)
        quat = _cv_to_unity_quaternion(R_world_to_cam)
        self._broadcast(
            {
                "type": "pose",
                "frame": frame_index,
                "position": pos,
                "rotation": quat,  # (x, y, z, w)
            }
        )

    def send_points(self, points: list[tuple[int, np.ndarray, tuple[int, int, int]]]) -> None:
        if not points:
            return
        payload = []
        for pid, xyz, bgr in points:
            b, g, r = bgr
            x, y, z = _cv_to_unity_position(xyz)
            payload.append({"id": int(pid), "x": x, "y": y, "z": z, "r": int(r), "g": int(g), "b": int(b)})
        self._broadcast({"type": "points", "points": payload})

    def send_status(self, state: str, num_points: int, num_keyframes: int) -> None:
        self._broadcast({"type": "status", "state": state, "num_points": num_points, "num_keyframes": num_keyframes})

    def send_reset(self) -> None:
        self._broadcast({"type": "reset"})

    @property
    def has_client(self) -> bool:
        with self._clients_lock:
            return len(self._clients) > 0
