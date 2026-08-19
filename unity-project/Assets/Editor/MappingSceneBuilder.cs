// MappingSceneBuilder.cs
//
// Tools > 3D Mapping > Build Scene
//
// Programmatically builds a working scene instead of shipping a
// hand-authored .unity file. Unity .unity files are YAML with internal
// object references by numeric fileID, generated and maintained by the
// Editor itself; hand-writing or hand-editing one outside the Editor is
// fragile (it's easy to produce a file that "looks right" but has a
// dangling reference Unity silently drops). Building the scene at Editor
// time with plain GameObject/Component API calls is equivalent in the end
// result and can't produce a broken reference, which is why this project
// ships as scripts + this builder rather than as a checked-in scene asset.
//
// Run it once after opening the project; it's safe to re-run (it clears
// its own previous output first).

using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Mapping3D.EditorTools
{
    public static class MappingSceneBuilder
    {
        private const string RootName = "MappingSession (generated)";

        [MenuItem("Tools/3D Mapping/Build Scene")]
        public static void BuildScene()
        {
            var existingRoot = GameObject.Find(RootName);
            if (existingRoot != null)
            {
                Undo.DestroyObjectImmediate(existingRoot);
            }

            var root = new GameObject(RootName);
            Undo.RegisterCreatedObjectUndo(root, "Build 3D Mapping Scene");

            // --- networking + session glue ---------------------------------
            var sessionGO = new GameObject("MappingSession");
            sessionGO.transform.SetParent(root.transform);
            var client = sessionGO.AddComponent<MapStreamClient>();
            var session = sessionGO.AddComponent<MappingSessionController>();
            session.client = client;

            // --- point cloud -------------------------------------------------
            var pointsGO = new GameObject("PointCloud");
            pointsGO.transform.SetParent(root.transform);
            var meshFilter = pointsGO.AddComponent<MeshFilter>();
            var meshRenderer = pointsGO.AddComponent<MeshRenderer>();
            var pointShader = Shader.Find("Mapping3D/PointCloudPoints");
            if (pointShader == null)
            {
                Debug.LogWarning("Mapping3D/PointCloudPoints shader not found -- did Assets/Shaders/PointCloudPoints.shader import correctly?");
            }
            else
            {
                meshRenderer.sharedMaterial = new Material(pointShader) { name = "SLAM Point Cloud Material" };
            }
            var pointCloudRenderer = pointsGO.AddComponent<PointCloudRenderer>();
            session.pointCloud = pointCloudRenderer;

            // --- tracked-camera gizmo ------------------------------------------
            var camVisGO = new GameObject("TrackedCameraGizmo");
            camVisGO.transform.SetParent(root.transform);
            BuildCameraGizmoMesh(camVisGO);
            var camVis = camVisGO.AddComponent<CameraPoseVisualizer>();
            session.cameraVisualizer = camVis;

            // --- Main Camera: spectator flycam, starts a few metres back -------
            var mainCam = Camera.main;
            if (mainCam == null)
            {
                var camGO = new GameObject("Main Camera");
                mainCam = camGO.AddComponent<Camera>();
                camGO.tag = "MainCamera";
            }
            mainCam.transform.position = new Vector3(0f, 2f, -4f);
            mainCam.transform.rotation = Quaternion.Euler(15f, 0f, 0f);
            mainCam.nearClipPlane = 0.05f;
            mainCam.farClipPlane = 500f;
            if (mainCam.GetComponent<SpectatorController>() == null)
            {
                mainCam.gameObject.AddComponent<SpectatorController>();
            }

            // Ambient light so the (currently unlit, so purely cosmetic for
            // any future lit objects you add) scene isn't pitch black.
            RenderSettings.ambientLight = new Color(0.35f, 0.35f, 0.4f);

            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());

            Debug.Log("3D Mapping scene built. Press Play, then run `python main.py --source webcam` (or --source synthetic to try it without a webcam).");
        }

        /// <summary>
        /// A tiny procedural "camera" gizmo -- a body box plus a forward-pointing
        /// wedge, so you can tell which way it's facing at a glance -- built at
        /// Editor time instead of importing a model, to keep the project
        /// dependency-free.
        /// </summary>
        private static void BuildCameraGizmoMesh(GameObject parent)
        {
            var body = GameObject.CreatePrimitive(PrimitiveType.Cube);
            body.name = "Body";
            body.transform.SetParent(parent.transform, false);
            body.transform.localScale = new Vector3(0.12f, 0.08f, 0.08f);
            Object.DestroyImmediate(body.GetComponent<Collider>());

            var nose = GameObject.CreatePrimitive(PrimitiveType.Cube);
            nose.name = "Nose";
            nose.transform.SetParent(parent.transform, false);
            nose.transform.localScale = new Vector3(0.04f, 0.04f, 0.08f);
            nose.transform.localPosition = new Vector3(0f, 0f, 0.08f);
            Object.DestroyImmediate(nose.GetComponent<Collider>());

            var mat = new Material(Shader.Find("Standard")) { color = new Color(1f, 0.6f, 0.1f) };
            body.GetComponent<Renderer>().sharedMaterial = mat;
            nose.GetComponent<Renderer>().sharedMaterial = mat;
        }
    }
}
