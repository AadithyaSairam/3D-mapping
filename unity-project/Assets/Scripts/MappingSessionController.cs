// MappingSessionController.cs
//
// The glue object: wires MapStreamClient's events to PointCloudRenderer /
// CameraPoseVisualizer, and draws a simple status overlay (connection
// state, point/keyframe counts, tracker state) with the immediate-mode
// GUI. OnGUI is used deliberately instead of a Canvas/uGUI setup -- it
// needs zero scene wiring (no Canvas, no EventSystem, no TMP import) which
// keeps "Tools > 3D Mapping > Build Scene" (see Assets/Editor/
// MappingSceneBuilder.cs) a one-click, dependency-free scene setup. Swap
// this out for real UI once the project has other reasons to need one.

using UnityEngine;

namespace Mapping3D
{
    public class MappingSessionController : MonoBehaviour
    {
        public MapStreamClient client;
        public PointCloudRenderer pointCloud;
        public CameraPoseVisualizer cameraVisualizer;

        private string _state = "-";
        private int _numPoints;
        private int _numKeyframes;
        private int _lastFrame;

        private void Reset()
        {
            client = GetComponent<MapStreamClient>();
        }

        private void OnEnable()
        {
            if (client == null) return;
            client.OnPose += HandlePose;
            client.OnPoints += HandlePoints;
            client.OnStatus += HandleStatus;
            client.OnReset += HandleReset;
        }

        private void OnDisable()
        {
            if (client == null) return;
            client.OnPose -= HandlePose;
            client.OnPoints -= HandlePoints;
            client.OnStatus -= HandleStatus;
            client.OnReset -= HandleReset;
        }

        private void HandlePose(Vector3 position, Quaternion rotation, int frame)
        {
            _lastFrame = frame;
            if (cameraVisualizer != null)
                cameraVisualizer.SetTargetPose(position, rotation);
        }

        private void HandlePoints(System.Collections.Generic.List<PointData> points)
        {
            if (pointCloud != null)
                pointCloud.AddPoints(points);
        }

        private void HandleStatus(string state, int numPoints, int numKeyframes)
        {
            _state = state;
            _numPoints = numPoints;
            _numKeyframes = numKeyframes;
        }

        private void HandleReset()
        {
            if (pointCloud != null)
                pointCloud.Clear();
            _state = "-";
            _numPoints = 0;
            _numKeyframes = 0;
        }

        private void OnGUI()
        {
            const int pad = 10;
            var rect = new Rect(pad, pad, 340, 110);
            GUI.Box(rect, GUIContent.none);

            var style = new GUIStyle(GUI.skin.label) { fontSize = 14 };
            string connected = client != null && client.isConnected ? "connected" : "waiting for python main.py...";
            int shownPoints = pointCloud != null ? pointCloud.PointCount : _numPoints;

            GUI.Label(new Rect(pad + 10, pad + 5, 320, 20), $"3D Mapping — {connected}", style);
            GUI.Label(new Rect(pad + 10, pad + 27, 320, 20), $"tracker state: {_state}", style);
            GUI.Label(new Rect(pad + 10, pad + 49, 320, 20), $"points: {shownPoints}   keyframes: {_numKeyframes}", style);
            GUI.Label(new Rect(pad + 10, pad + 71, 320, 20), $"last pose frame: {_lastFrame}", style);
            GUI.Label(new Rect(pad + 10, pad + 90, 320, 20), "right-drag look, WASD move, shift = fast", style);
        }
    }
}
