# 3D Mapping — real-time monocular visual SLAM → Unity

[![CI](https://github.com/AadithyaSairam/3D-mapping/actions/workflows/ci.yml/badge.svg)](https://github.com/AadithyaSairam/3D-mapping/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A plain RGB webcam feeds a from-scratch monocular visual SLAM pipeline
(Python + OpenCV) that tracks the camera's motion and incrementally
triangulates a 3D point cloud of the room around it, streamed live over
TCP into a Unity scene that renders the reconstruction and lets you fly
around it in real time.

This is a learning project first, portfolio piece second: every module
under `python-slam/slam/` is written to be read, not just run — each file
has a docstring explaining the computer-vision idea it implements and
*why* it's built the way it is, and `docs/THEORY.md` walks through the
whole pipeline end to end. See `docs/THEORY.md` §6 for an honest list of
what a from-scratch build like this does *not* do (bundle adjustment,
loop closure, relocalization) compared to a production system like
ORB-SLAM3.

## What it does

- Detects and tracks image features across webcam frames (optical flow +
  ORB, `slam/features.py`).
- Recovers camera motion via epipolar geometry (`slam/pose_estimation.py`)
  to bootstrap a 3D map once, then tracks every subsequent frame against
  that map with PnP (`slam/tracker.py`) — see `docs/THEORY.md` for why
  that two-stage design matters.
- Grows the map over time by triangulating new points at each keyframe
  (`slam/map.py`, `slam/triangulation.py`).
- Streams the live camera pose + growing point cloud to Unity over a
  small TCP/NDJSON protocol (`slam/streamer.py`, `docs/PROTOCOL.md`).
- Renders the result in real time in Unity as a GPU point cloud, with a
  "ghost camera" showing the tracked pose and a free-fly camera so you can
  look around the reconstruction as it builds (`unity-project/`).
- Includes a synthetic-camera mode (`slam/synthetic.py`) so you can run
  and demo the entire pipeline — Python side and Unity side — **without a
  webcam**, and a test suite that validates the geometry against known
  ground truth (`python-slam/tests/`).

## Quickstart (no webcam needed)

```bash
cd python-slam
pip install -r requirements.txt
python main.py --source synthetic
```

This opens a debug window showing a simulated camera wandering through a
generated room, with tracked points overlaid, and starts a TCP server on
`localhost:8765`. Open the Unity project (see below), press Play, and
you'll see the same room being reconstructed live as a point cloud.

## Running against a real webcam

```bash
cd python-slam
pip install -r requirements.txt

# 1. Calibrate your webcam once (see calibrate_camera.py's docstring for
#    what you need — a printed checkerboard pattern).
python calibrate_camera.py

# 2. Run the real thing.
python main.py --source webcam
```

Move the camera with clear sideways translation (not just panning in
place) to get initialization going — see `docs/THEORY.md` §5 for why pure
rotation can't be triangulated from.

Debug window controls: `q` to quit, `r` to reset the tracker and map (also
tells Unity to clear its point cloud).

## Unity setup

The Unity side ships as a set of scripts + a shader (`unity-project/Assets/`),
not a full pre-built project, so you're never stuck with an
unopenable/version-mismatched project file:

1. Create a new Unity project with the **3D (Built-in Render Pipeline)**
   template (any recent 2021/2022 LTS or newer works — the shader targets
   Built-in RP specifically, see `docs/ARCHITECTURE.md` if you want to
   port it to URP).
2. Copy the contents of `unity-project/Assets/` into your new project's
   `Assets/` folder.
3. In the Editor menu bar: **Tools > 3D Mapping > Build Scene**. This
   programmatically builds the whole scene (networking, point cloud
   renderer, tracked-camera gizmo, a WASD/right-click-drag flycam) — see
   `docs/ARCHITECTURE.md` for why it's a builder script rather than a
   checked-in `.unity` file.
4. Press Play, then run `python main.py` (webcam or synthetic) in a
   terminal — order doesn't matter, `MapStreamClient` retries the
   connection until Python is listening.

## Project layout

```
python-slam/
  slam/
    camera.py            camera intrinsics
    features.py            feature detection + tracking
    pose_estimation.py     two-view epipolar geometry
    triangulation.py       2D + known pose -> 3D point
    map.py                  KeyFrame / MapPoint / Map
    tracker.py              the main SLAM loop
    streamer.py             TCP/NDJSON server -> Unity
    synthetic.py            fake camera for testing/demos
  calibrate_camera.py       checkerboard calibration script
  main.py                   entry point
  tests/                    pytest suite (unit + integration)

unity-project/Assets/
  Scripts/                  MapStreamClient, PointCloudRenderer, ...
  Shaders/                  PointCloudPoints.shader
  Editor/                    MappingSceneBuilder.cs (scene setup)

docs/
  THEORY.md                  the "textbook chapter" — read this
  ARCHITECTURE.md            system overview, data flow
  PROTOCOL.md                 exact wire format
```

## Testing

```bash
cd python-slam
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

Includes geometry unit tests against known synthetic ground truth
(pose recovery, triangulation, cheirality filtering) and an end-to-end
integration test that runs the full tracker against the synthetic camera
for hundreds of frames and checks it actually initializes and grows a
map. This whole suite runs headless, no webcam or Unity required — useful
both as a starting point if you extend the pipeline and as evidence
(for yourself, or anyone looking at the repo) that the core geometry is
actually correct, not just "didn't crash when I eyeballed it."

## Known limitations

See `docs/THEORY.md` §5-§6 for the full, honest list (monocular scale
ambiguity, no bundle adjustment, no loop closure, no relocalization, no
homography-vs-essential-matrix degeneracy handling) and why each one is a
reasonable line to draw for a from-scratch learning project rather than an
oversight.

## License

MIT — see [LICENSE](LICENSE).
