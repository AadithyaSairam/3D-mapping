"""
slam — a small, from-scratch monocular visual SLAM / visual odometry pipeline.

This package is deliberately split into one file per concept, in roughly the
order you'd learn them if you were building monocular SLAM for the first
time:

    camera.py           -> what a calibrated camera even means
    features.py          -> finding & tracking points across frames
    pose_estimation.py   -> recovering motion between two views (epipolar geometry)
    triangulation.py     -> turning 2D correspondences into 3D points
    map.py                -> keeping a persistent 3D map (keyframes + map points)
    tracker.py            -> the main loop that ties all of the above together
    streamer.py            -> sending the map + camera pose to Unity over TCP

Read them in that order. Each file has a module docstring that explains the
*why*, not just the *what* — the goal of this project is to actually
understand monocular SLAM, not just call a library that does it for you.
"""
