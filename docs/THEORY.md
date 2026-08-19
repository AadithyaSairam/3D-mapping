# Theory: how this monocular SLAM system actually works

This is the "textbook chapter" for the project — read it alongside the
code, not instead of it. Every section names the file that implements the
idea; the code's own docstrings go into more line-by-line detail than
this document does. Recommended reading order matches the order things
happen at runtime.

## 0. Vocabulary, up front

- **VO (Visual Odometry)**: estimate the camera's motion frame-to-frame.
  No persistent map, no "have I been here before."
- **SLAM (Simultaneous Localization And Mapping)**: VO, plus a persistent
  map of 3D landmarks that both grows over time *and* is used to correct
  future pose estimates. "Simultaneous" because you need the map to
  localize and you need to know where you are to build the map — this
  project (like all monocular SLAM) bootstraps that chicken-and-egg
  problem with a special-cased initialization step (§2).
- **Monocular**: a single RGB camera, no depth sensor, no second camera
  (stereo), no IMU. The hardest and least-constrained version of this
  problem — see §5 for exactly what that costs you.
- **Keyframe**: a frame we keep around permanently as a reference for
  triangulating new points, as opposed to the many ordinary frames we
  track through and discard. See `slam/map.py`.

## 1. The measurement: feature tracking (`slam/features.py`)

Everything downstream is built from one raw signal: "this pixel
neighbourhood in frame A is the same physical point as that pixel
neighbourhood in frame B." This project gets that signal two ways:

- **Detect once, track with optical flow** (`track_optical_flow`,
  Lucas-Kanade): cheap, smooth, sub-pixel-accurate, used every frame
  during normal operation.
- **Detect-and-match with ORB descriptors** (`detect_and_match_orb`): more
  expensive, more robust to larger motion, used only for the "cold start"
  cases (map initialization) where there's no previous track to lean on.

**Try this**: comment out the optical-flow path and force ORB
detect-and-match every frame instead. Watch the debug window's FPS
counter and the smoothness of tracked-point overlays — this is a very
direct, visual way to feel *why* real-time systems avoid running a
detector on every single frame.

## 2. Bootstrapping: two-view initialization (`slam/pose_estimation.py`, `slam/triangulation.py`)

You cannot triangulate 3D points without knowing two camera poses, and
you cannot know a camera pose (via PnP, see §3) without already having 3D
points. Two-view initialization breaks that cycle exactly once:

1. Wait until the camera has moved enough (`MIN_INIT_PARALLAX_PX` in
   `tracker.py`) that its two views of the scene are genuinely different —
   too little motion and the epipolar geometry (below) is numerically
   degenerate.
2. Compute the **essential matrix** `E` from 2D↔2D correspondences via
   `cv2.findEssentialMat` (RANSAC-robustified), which encodes the
   relative rotation+translation between the two views through the
   epipolar constraint `p2ᵀ E p1 = 0`.
3. Decompose `E` back into `(R, t)` via `cv2.recoverPose`.
4. Triangulate every inlier correspondence into a 3D point
   (`triangulate_points`), using `(R₀,t₀) = identity` for the first view
   as the world origin.

**The key insight to internalize**: `t` from `recoverPose` is a *unit
vector* — motion direction, not distance. There is a fundamental,
irrecoverable ambiguity in monocular 2D image motion between "moved a
little past close objects" and "moved a lot past far objects" (they
produce identical images). Whatever scale this first triangulation
produces becomes the map's permanent scale — see §5.

## 3. Steady-state tracking: PnP against the map (`slam/tracker.py::_track_with_pnp`)

Once there's a map, every subsequent frame is localized with
**Perspective-n-Point** (`cv2.solvePnPRansac`): given ≥6 pairs of (a 3D
point already in the map, its 2D pixel location in the current frame),
solve directly for the camera pose that best explains them.

This is the single most important design decision in the whole project,
and it's worth understanding *why* it beats the "obvious" alternative of
just chaining essential-matrix estimates frame-to-frame:

- PnP's 3D points are already scaled consistently (they came from the
  one-time triangulation in §2, or from later keyframe triangulations
  that reuse that same scale — see §4). Chaining essential matrices
  instead would re-introduce a *new arbitrary scale every single frame*.
- PnP anchors each pose to a fixed external reference (the map), not to
  the *previous frame's own estimate* — so its errors don't compound the
  same way frame-to-frame chaining's do. (They still compound eventually,
  just much more slowly — see §6, bundle adjustment.)

This is exactly the design real systems (ORB-SLAM and its relatives) use,
simplified: no separate "local mapping" thread, no bundle adjustment
refining the map afterward, no loop closure.

## 4. Growing the map: keyframes (`slam/tracker.py::_insert_keyframe`, `slam/map.py`)

Points leave the camera's view as it moves, so the pool of "trackable
things with a known 3D position" needs to be replenished. Periodically
(distance/frame-count heuristics in `_should_insert_keyframe`), the
tracker:

1. Declares the current frame a new **keyframe** and stores its pose.
2. Triangulates any point that's been tracked (via optical flow)
   continuously since the *previous* keyframe but doesn't have a 3D
   position yet — using the same triangulation math as init (§2), just
   between two keyframes instead of the two init frames.
3. Detects fresh Shi-Tomasi corners to top the tracking pool back up.

This is also where **outlier culling** happens: `Map.mark_outlier`
(called from `_track_with_pnp` whenever PnP's own RANSAC flags a
correspondence as an outlier) deletes a map point after it's
misbehaved a few times in a row — a cheap substitute for the "is this
point actually any good" reasoning that a real system does with bundle
adjustment residuals.

## 5. What monocular SLAM structurally cannot give you

- **Absolute scale.** Covered in §2 — without a depth sensor, a second
  camera, an IMU, or a known real-world measurement, "1 unit" in the map
  means nothing outside the map itself.
- **Robustness to almost-purely-rotational motion.** Epipolar geometry
  (§2) degenerates when there's no translation — a camera that only pans
  in place cannot be triangulated from. If initialization is stuck, this
  is usually why: try translating the camera sideways, not just turning
  it.
- **Robustness to a near-planar scene during init.** If everything the
  camera sees during initialization lies roughly on one plane (e.g.
  pointed at a flat wall or table), a *homography* explains the motion
  just as well as an essential matrix does, and the decomposition becomes
  numerically unstable — you may see wildly wrong depths as a result.
  Real systems (ORB-SLAM) explicitly score both an essential-matrix model
  and a homography model and pick whichever fits the data better,
  specifically to detect and reject this case; this project does not do
  that scoring, so seeing occasionally bad-scale initialization against
  planar-ish scenes is expected, not a bug — see §6.

## 6. What's deliberately not implemented (and why that's OK for this project)

- **Bundle adjustment.** A real SLAM backend periodically re-optimizes
  *all* keyframe poses and map points jointly to minimize total
  reprojection error, using something like `g2o` or `ceres-solver`. This
  is the single biggest thing separating "a working monocular VO/SLAM
  pipeline" (this project) from "a production-grade SLAM system." It's a
  substantial chunk of nonlinear-least-squares machinery on its own —
  a natural "part 2" if you want to go further.
- **Loop closure.** Recognizing "I've been here before" (typically via
  place-recognition on a bag-of-visual-words index) and snapping
  accumulated drift shut. Without it, this system's trajectory will drift
  monotonically over a long session — map an entire loop around a room
  and the start/end won't quite line up.
- **Relocalization.** If PnP tracking fails outright (too few map
  correspondences — fast motion, motion blur, pointing at something
  never mapped), this project just coasts on the last known pose and
  keeps trying every frame. A real system would run global relocalization
  (re-detect+match against the whole map, or a place-recognition index)
  to recover.
- **Homography-vs-essential-matrix init scoring.** See §5's planar-scene
  note.

Understanding *why* a from-scratch system without these things behaves
the way it does — where it drifts, when initialization struggles, what a
"good enough for a demo" scale error looks like — is genuinely most of
the value of building this yourself instead of dropping in ORB-SLAM3.

## Further reading

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* — the
  canonical reference for everything in §1-§2.
- Mur-Artal, Montiel, Tardós, *ORB-SLAM: a Versatile and Accurate
  Monocular SLAM System* (2015) — the real system this project's
  structure (init → PnP tracking → keyframes → map) is a simplified
  sketch of.
- Scaramuzza & Fraundorfer, *Visual Odometry* (tutorial, IEEE Robotics &
  Automation Magazine, 2011/2012, two parts) — a very readable overview
  of the whole pipeline end to end.
