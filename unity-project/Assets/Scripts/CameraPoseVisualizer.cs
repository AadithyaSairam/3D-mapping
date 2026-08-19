// CameraPoseVisualizer.cs
//
// Moves a small "ghost camera" gizmo to match the live pose streamed from
// the Python tracker, so you can see where the physical webcam thinks it
// is relative to the point cloud it's building.
//
// Smoothing: pose messages arrive at whatever rate main.py's PnP tracking
// succeeds (roughly webcam FPS, so ~15-30Hz), which rarely lines up with
// Unity's render rate, and monocular PnP pose is noisier than you'd want
// for a perfectly steady visual. We exponentially smooth position and
// rotation in Update() towards the latest received target rather than
// snapping directly to it -- this trades a small amount of latency for a
// much less jittery result, which matters a lot for how convincing the
// demo looks/feels.

using UnityEngine;

namespace Mapping3D
{
    public class CameraPoseVisualizer : MonoBehaviour
    {
        [Tooltip("Higher = snappier / more jitter. Lower = smoother / more lag.")]
        [Range(1f, 30f)]
        public float smoothing = 12f;

        [Tooltip("If assigned, this transform is driven to follow the tracked pose too " +
                 "(e.g. point Unity's Main Camera here to 'be' the webcam).")]
        public Transform followTarget;

        private Vector3 _targetPosition;
        private Quaternion _targetRotation = Quaternion.identity;
        private bool _hasTarget;

        public void SetTargetPose(Vector3 position, Quaternion rotation)
        {
            _targetPosition = position;
            _targetRotation = rotation;
            _hasTarget = true;
        }

        private void Update()
        {
            if (!_hasTarget) return;

            float t = 1f - Mathf.Exp(-smoothing * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, _targetPosition, t);
            transform.rotation = Quaternion.Slerp(transform.rotation, _targetRotation, t);

            if (followTarget != null)
            {
                followTarget.position = transform.position;
                followTarget.rotation = transform.rotation;
            }
        }
    }
}
