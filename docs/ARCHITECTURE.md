# Architecture

## Two processes, one TCP connection

```
 ┌───────────────────────────────────────────┐        NDJSON / TCP        ┌──────────────────────────────────────────┐
 │  python-slam  (Python + OpenCV)             │  ───────────────────────▶ │  Unity  (C#, Assets/Scripts)               │
 │                                              │        :8765               │                                             │
 │  webcam / video / synthetic frame source     │                            │  MapStreamClient   — TCP client,           │
 │        │                                     │                            │       background thread, NDJSON parsing    │
 │        ▼                                     │                            │        │                                    │
 │  Tracker.process_frame()                     │                            │        ├──▶ CameraPoseVisualizer            │
 │    features.py   (optical flow / ORB)        │                            │        │      (ghost camera + smoothing)   │
 │    pose_estimation.py (two-view init)        │                            │        └──▶ PointCloudRenderer              │
 │    triangulation.py                          │                            │               (growing Mesh, Points topo)  │
 │    map.py         (KeyFrame, MapPoint, Map)  │                            │                                             │
 │        │                                     │                            │  MappingSessionController — wires it all   │
 │        ▼                                     │                            │  together + on-screen status (OnGUI)       │
 │  streamer.py  (MapStreamer, TCP server)      │                            │                                             │
 │                                              │                            │  SpectatorController — WASD flycam so you   │
 └───────────────────────────────────────────┘                            │  can look at the reconstruction             │
                                                                              └──────────────────────────────────────────┘
```

The two sides are deliberately independent processes talking over a
socket, not a single in-Unity pipeline (e.g. via OpenCV-for-Unity). That
keeps the SLAM math in ordinary, easily-debuggable/testable Python (see
`python-slam/tests/`) — you can run and iterate on the entire tracking
pipeline with `main.py --source synthetic` and never open Unity — while
Unity stays focused on what it's good at: real-time 3D rendering and
interaction. The cost is the serialization layer (`docs/PROTOCOL.md`);
that's a fair trade for a learning project where being able to poke at
each half independently matters more than end-to-end latency.

## Data flow, frame by frame

1. `main.py` reads one frame (webcam / video file / `synthetic.py`'s fake
   camera).
2. `Tracker.process_frame(frame)`:
   - Advances all live point tracks with optical flow.
   - **If still initializing**: check accumulated parallax; if enough,
     attempt two-view pose estimation + triangulation once (`THEORY.md`
     §2). On success, the map now exists and the tracker switches to
     steady-state tracking.
   - **If tracking**: solve PnP against existing map points to get this
     frame's pose (`THEORY.md` §3); periodically insert a keyframe, which
     triangulates newly-trackable points and replenishes the feature pool
     (`THEORY.md` §4).
   - Returns a `FrameResult`: the pose (if any), any newly-triangulated
     points, and whether this frame was a keyframe.
3. `main.py` hands that `FrameResult` to `MapStreamer`, which serializes
   it to NDJSON and broadcasts to any connected Unity client(s).
4. In Unity, `MapStreamClient`'s background thread reads lines off the
   socket and queues parsed messages; `Update()` drains the queue on the
   main thread and fires C# events.
5. `MappingSessionController` forwards those events to
   `PointCloudRenderer` (append new points, rebuild the mesh) and
   `CameraPoseVisualizer` (smoothly move the ghost camera to the new
   pose).

## Why a scene-builder script instead of a checked-in `.unity` scene

`Assets/Editor/MappingSceneBuilder.cs` (`Tools > 3D Mapping > Build
Scene`) constructs the whole scene — GameObjects, components, a
procedural camera gizmo mesh, the point-cloud material — with plain
Editor API calls at Editor time, rather than shipping a hand-authored
`.unity` YAML file. `.unity` files are normally generated and maintained
entirely by the Unity Editor itself (internal object references are
numeric `fileID`s); hand-writing or hand-patching one outside the Editor
easily produces a file that "looks right" but has a subtly broken
reference. Building the scene from code sidesteps that risk entirely and
is easy to re-run from scratch if you ever want to reset it.

## Extension points, if you keep going

- **Bundle adjustment / loop closure** — see `docs/THEORY.md` §6 for what
  and why; the `KeyFrame`/`MapPoint` split in `slam/map.py` is already
  structured the way you'd hang either onto.
- **Binary wire protocol** — see `docs/PROTOCOL.md`'s "possible
  extensions."
- **Point removal in Unity** — see `PointCloudRenderer.cs`'s module
  comment.
- **Mesh reconstruction from the point cloud** (Poisson surface
  reconstruction, marching cubes, etc.) instead of just rendering raw
  points — a substantial project on its own; `open3d`'s Python bindings
  are the standard starting point if you want to try it as an offline
  post-process on a saved point cloud.
