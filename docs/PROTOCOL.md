# Wire protocol: Python SLAM → Unity

A plain TCP server (`python-slam/slam/streamer.py`) sends newline-delimited
JSON (NDJSON): one JSON object per line, `\n`-terminated. Unity
(`Assets/Scripts/MapStreamClient.cs`) connects as a client and reads it
line by line.

Why NDJSON and not a binary protocol: this project is explicitly a
learning project, and NDJSON means you can debug the entire pipeline with
`nc localhost 8765` and your own eyes, and both ends parse it with a
few lines of standard-library code instead of a hand-rolled binary layout
or an extra dependency (protobuf/FlatBuffers codegen). The tradeoff is
size and parse speed — fine at the point counts (thousands, not millions)
and pose rates (tens of Hz) this project produces; see "Scaling up" below.

## Message types

Every message has a `type` field. Unity's `MapStreamClient` peeks at just
that field first (see `NetworkProtocol.cs`'s `TypeProbe`), then parses the
full message into the matching class.

### `pose` — sent every frame a pose was successfully recovered

```json
{"type": "pose", "frame": 142, "position": [0.12, 1.4, 3.0], "rotation": [0.0, 0.01, 0.0, 0.999]}
```

- `position`: Unity-space `(x, y, z)`, metres (well — SLAM-map units; see
  "Scale" below), the camera's position in the world.
- `rotation`: Unity-space quaternion `(x, y, z, w)` — this is Unity's
  native ctor order (`new Quaternion(x, y, z, w)`), *not* the mathematical
  `(w, x, y, z)` order used internally by `streamer.py`'s own quaternion
  math before it's packed into the message.

### `points` — sent whenever a new keyframe triangulates new map points

```json
{"type": "points", "points": [
  {"id": 17, "x": 0.4, "y": 0.9, "z": 2.1, "r": 120, "g": 140, "b": 90},
  {"id": 18, "x": 0.5, "y": 0.85, "z": 2.0, "r": 110, "g": 130, "b": 100}
]}
```

Point `id`s are assigned once by the Python-side `Map` and are stable —
Unity uses them to avoid adding the same point twice, but currently
**never removes a point** even if the Python map later discards it
internally (see `PointCloudRenderer.cs`'s module comment for why, and
"Possible extensions" below).

### `status` — sent roughly every 15 frames, informational only

```json
{"type": "status", "state": "TRACKING", "num_points": 4213, "num_keyframes": 38}
```

`state` mirrors `slam.tracker.TrackerState`: `INITIALIZING`, `TRACKING`,
or `LOST`.

### `reset` — sent when you press `r` in the Python debug window

```json
{"type": "reset"}
```

Unity clears its point cloud and status display. The Python side has
already thrown away its old `Tracker`/`Map` and started fresh.

## Coordinate systems (read this before touching either side's geometry code)

OpenCV/this project's SLAM math: **right-handed**, `+X` right, `+Y` down,
`+Z` forward (into the scene). A pose is stored as world-to-camera
`(R, t)`: `X_cam = R @ X_world + t`.

Unity: **left-handed**, `+X` right, `+Y` up, `+Z` forward. Objects are
positioned by *where they are* (a `Transform.position`), not a
world-to-object matrix.

`streamer.py` converts once, in one place, on the way out:

1. **Points**: negate Y (`_cv_to_unity_position`). Negating only Y turns a
   right-handed point cloud into the mirror-image left-handed one Unity
   expects — this alone is enough for points, which have no orientation.
2. **Camera pose**: first convert world-to-camera `(R, t)` into a camera
   *position* (`camera_center = -Rᵀt`, computed the same way as
   `KeyFrame.camera_center` in `map.py`), then apply the same Y-negation.
   For the *orientation*, negating a single axis of a rotation matrix
   turns it into an improper rotation (a reflection, determinant −1) —
   not valid for a `Quaternion`. So `_cv_to_unity_quaternion` instead
   conjugates the camera-to-world rotation by `diag(1, -1, -1)` (flipping
   **two** axes), which keeps it a proper rotation while still achieving
   the same handedness change, then converts that to a quaternion.

If you ever need to go the other way (Unity → OpenCV, e.g. to send a
"teleport the virtual camera here" command back to Python), apply the
identical `diag(1, -1, -1)` conjugation and Y-negation again — both
operations are their own inverse.

## Scale

SLAM map units are **not metres** unless you got lucky. Monocular
two-view initialization fixes the map's scale to "whatever the first
essential-matrix translation's unit vector happened to imply" (see
`pose_estimation.py`'s and `triangulation.py`'s docstrings) — it has no
way to know the real-world size of anything. Don't be surprised if the
Unity point cloud comes out at 10x or 0.1x a room's real size; if you want
it metric, the standard fix is a *known* real-world measurement (e.g. you
know two map points are exactly 1m apart) used to rescale the whole map
once, right after initialization.

## Possible extensions (not implemented — see docs/THEORY.md for the full "what's missing" list)

- **Binary framing**: replace NDJSON with length-prefixed binary
  (Protobuf/FlatBuffers, or even just `struct.pack`), if point counts grow
  large enough that JSON parsing becomes a bottleneck.
- **Point removal**: a `{"type": "remove", "ids": [...]}` message, paired
  with a `PointCloudRenderer.RemovePoints(ids)` that either compacts the
  mesh buffers or marks entries dead and skips them at draw time.
- **Bidirectional control**: Unity → Python messages (e.g. "pause",
  "trigger a reset", "recalibrate") over the same socket or a second one.
