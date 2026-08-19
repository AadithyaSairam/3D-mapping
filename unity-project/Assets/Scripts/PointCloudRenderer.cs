// PointCloudRenderer.cs
//
// Accumulates the map points streamed from Python and draws them as a
// growing point cloud, using a single dynamic Mesh with
// MeshTopology.Points (see Shaders/PointCloudPoints.shader for how each
// point gets its on-screen size).
//
// DESIGN NOTE: append-only, no removal.
// The Python tracker (see python-slam/slam/map.py: Map.mark_outlier) does
// internally cull map points that turn out to be bad triangulations --
// but streamer.py never sends a "remove this point" message, only
// "points" (additions). That's a deliberate scope cut: implementing point
// removal means either rebuilding index-compacted mesh buffers whenever a
// point is deleted (fiddly) or maintaining a "deleted" flag and skipping
// those vertices at draw time (extra complexity for a small visual
// improvement). The practical effect is that a small number of points the
// SLAM backend has internally discarded may linger in the Unity view. If
// you want to fix this, the natural extension is a `{"type": "remove",
// "ids": [...]}` message and a parallel `RemovePoints` here -- see
// docs/PROTOCOL.md's "possible extensions" section.
//
// PERFORMANCE NOTE: mesh rebuilds are O(total points), not O(new points),
// because Mesh.SetVertices/SetColors/SetIndices replace the whole buffer.
// For the point counts this project produces (thousands, not millions)
// that's comfortably fast even rebuilt every time a points message
// arrives; a production system would use append-friendly GraphicsBuffers
// instead.

using System.Collections.Generic;
using UnityEngine;

namespace Mapping3D
{
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class PointCloudRenderer : MonoBehaviour
    {
        [Tooltip("Rebuild the mesh at most this often, even if points are arriving faster.")]
        public float minRebuildIntervalSeconds = 0.05f;

        private readonly List<Vector3> _positions = new List<Vector3>();
        private readonly List<Color32> _colors = new List<Color32>();
        private readonly Dictionary<int, int> _idToIndex = new Dictionary<int, int>();

        private Mesh _mesh;
        private MeshFilter _meshFilter;
        private bool _dirty;
        private float _lastRebuildTime = -1f;

        public int PointCount => _positions.Count;

        private void Awake()
        {
            _meshFilter = GetComponent<MeshFilter>();
            _mesh = new Mesh { name = "SLAM Point Cloud" };
            _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32; // > 65535 points
            _meshFilter.sharedMesh = _mesh;
        }

        private void Update()
        {
            if (_dirty && Time.unscaledTime - _lastRebuildTime >= minRebuildIntervalSeconds)
            {
                RebuildMesh();
            }
        }

        public void AddPoints(List<PointData> points)
        {
            foreach (var p in points)
            {
                if (_idToIndex.ContainsKey(p.id))
                {
                    // Duplicate point id (shouldn't normally happen -- ids
                    // are assigned once by the Python-side Map -- but if
                    // main.py was restarted without Unity reconnecting
                    // cleanly, ids could collide across sessions). Just
                    // update the existing entry's color/position rather
                    // than silently ignoring or throwing.
                    int idx = _idToIndex[p.id];
                    _positions[idx] = new Vector3(p.x, p.y, p.z);
                    _colors[idx] = new Color32((byte)p.r, (byte)p.g, (byte)p.b, 255);
                    continue;
                }
                _idToIndex[p.id] = _positions.Count;
                _positions.Add(new Vector3(p.x, p.y, p.z));
                _colors.Add(new Color32((byte)p.r, (byte)p.g, (byte)p.b, 255));
            }
            _dirty = true;
        }

        public void Clear()
        {
            _positions.Clear();
            _colors.Clear();
            _idToIndex.Clear();
            _mesh.Clear();
            _dirty = false;
            _lastRebuildTime = Time.unscaledTime;
        }

        private void RebuildMesh()
        {
            _mesh.Clear();
            _mesh.SetVertices(_positions);
            _mesh.SetColors(_colors);
            var indices = new int[_positions.Count];
            for (int i = 0; i < indices.Length; i++) indices[i] = i;
            _mesh.SetIndices(indices, MeshTopology.Points, 0);
            // Points have no inherent size, so Unity can't compute a
            // meaningful bounds from geometry alone in all cases; give it
            // a generous fixed bound so the point cloud never gets
            // frustum-culled just because Unity guessed a bad AABB.
            _mesh.RecalculateBounds();
            var b = _mesh.bounds;
            b.Expand(5f);
            _mesh.bounds = b;

            _dirty = false;
            _lastRebuildTime = Time.unscaledTime;
        }
    }
}
