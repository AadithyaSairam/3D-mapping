// SpectatorController.cs
//
// A minimal WASD + right-click-drag-to-look flycam, attached to Unity's
// Main Camera so you can fly around and inspect the reconstructed point
// cloud from any angle -- Unity has no built-in equivalent outside the
// (package-based) Editor scene view camera, and this project deliberately
// avoids extra package dependencies for a two-screen-of-code feature.

using UnityEngine;

namespace Mapping3D
{
    public class SpectatorController : MonoBehaviour
    {
        public float moveSpeed = 2.5f;
        public float fastMoveMultiplier = 4f;
        public float lookSensitivity = 2.5f;

        private float _yaw;
        private float _pitch;

        private void Start()
        {
            var euler = transform.eulerAngles;
            _yaw = euler.y;
            _pitch = euler.x;
        }

        private void Update()
        {
            if (Input.GetMouseButton(1)) // right mouse button held = look around
            {
                _yaw += Input.GetAxis("Mouse X") * lookSensitivity;
                _pitch -= Input.GetAxis("Mouse Y") * lookSensitivity;
                _pitch = Mathf.Clamp(_pitch, -89f, 89f);
                transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
            }

            float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? fastMoveMultiplier : 1f);
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.W)) move += transform.forward;
            if (Input.GetKey(KeyCode.S)) move -= transform.forward;
            if (Input.GetKey(KeyCode.D)) move += transform.right;
            if (Input.GetKey(KeyCode.A)) move -= transform.right;
            if (Input.GetKey(KeyCode.E)) move += transform.up;
            if (Input.GetKey(KeyCode.Q)) move -= transform.up;

            transform.position += move * speed * Time.deltaTime;
        }
    }
}
