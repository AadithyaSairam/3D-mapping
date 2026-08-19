// MapStreamClient.cs
//
// TCP client for the Python SLAM process's NDJSON stream (see
// python-slam/slam/streamer.py and docs/PROTOCOL.md for the wire format).
//
// WHY A BACKGROUND THREAD
// ------------------------
// Blocking socket reads (Stream.ReadLine) cannot happen on Unity's main
// thread without freezing the game -- Update()/rendering share that same
// thread, so a blocking call there stalls the whole app until data
// arrives. This script instead runs the connect+read loop on a plain
// System.Threading.Thread, and hands finished messages to the main thread
// through a thread-safe queue that Update() drains every frame. This is
// the standard pattern for any blocking I/O in Unity (sockets, file
// watchers, serial ports, ...); the underlying rule is simply "only touch
// UnityEngine API (GameObjects, Transforms, etc.) from the main thread."
//
// RECONNECTION
// ------------
// The Python process may start after Unity, or the connection may drop
// (e.g. you restarted main.py). This client retries on a fixed interval
// rather than giving up, so pressing Play in the Editor and starting the
// Python process in either order both just work.

using System;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace Mapping3D
{
    public class MapStreamClient : MonoBehaviour
    {
        [Header("Connection")]
        public string host = "127.0.0.1";
        public int port = 8765;
        public float reconnectIntervalSeconds = 2f;

        [Header("Status (read-only)")]
        public bool isConnected;

        public event Action<Vector3, Quaternion, int> OnPose;
        public event Action<System.Collections.Generic.List<PointData>> OnPoints;
        public event Action<string, int, int> OnStatus;
        public event Action OnReset;
        public event Action<bool> OnConnectionChanged;

        private Thread _thread;
        private volatile bool _running;
        private readonly ConcurrentQueue<string> _incomingLines = new ConcurrentQueue<string>();
        private volatile bool _connectedFlag;

        private void Awake()
        {
            _running = true;
            _thread = new Thread(ConnectionLoop) { IsBackground = true };
            _thread.Start();
        }

        private void OnDestroy()
        {
            _running = false;
            if (_thread != null && _thread.IsAlive)
            {
                _thread.Join(500);
            }
        }

        private void Update()
        {
            if (_connectedFlag != isConnected)
            {
                isConnected = _connectedFlag;
                OnConnectionChanged?.Invoke(isConnected);
            }

            // Drain everything currently queued this frame, rather than one
            // message per frame -- the Python side can easily produce many
            // messages per Unity frame (it streams at whatever FPS the
            // webcam runs at, which may exceed Unity's frame rate, or a
            // batch of new keyframe points can arrive as one message).
            int guard = 0;
            while (_incomingLines.TryDequeue(out string line) && guard < 2000)
            {
                guard++;
                DispatchLine(line);
            }
        }

        private void DispatchLine(string line)
        {
            if (string.IsNullOrWhiteSpace(line)) return;

            TypeProbe probe;
            try
            {
                probe = JsonUtility.FromJson<TypeProbe>(line);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[MapStreamClient] failed to parse message: {e.Message}\n{line}");
                return;
            }

            if (probe == null || string.IsNullOrEmpty(probe.type)) return;

            switch (probe.type)
            {
                case "pose":
                {
                    var msg = JsonUtility.FromJson<PoseMessage>(line);
                    if (msg.position == null || msg.position.Length != 3 || msg.rotation == null || msg.rotation.Length != 4)
                        return;
                    var pos = new Vector3(msg.position[0], msg.position[1], msg.position[2]);
                    var rot = new Quaternion(msg.rotation[0], msg.rotation[1], msg.rotation[2], msg.rotation[3]);
                    OnPose?.Invoke(pos, rot, msg.frame);
                    break;
                }
                case "points":
                {
                    var msg = JsonUtility.FromJson<PointsMessage>(line);
                    if (msg.points != null && msg.points.Count > 0)
                        OnPoints?.Invoke(msg.points);
                    break;
                }
                case "status":
                {
                    var msg = JsonUtility.FromJson<StatusMessage>(line);
                    OnStatus?.Invoke(msg.state, msg.num_points, msg.num_keyframes);
                    break;
                }
                case "reset":
                {
                    OnReset?.Invoke();
                    break;
                }
                default:
                    Debug.LogWarning($"[MapStreamClient] unknown message type '{probe.type}'");
                    break;
            }
        }

        // --- background thread -------------------------------------------------

        private void ConnectionLoop()
        {
            while (_running)
            {
                try
                {
                    using (var client = new TcpClient())
                    {
                        var connectTask = client.BeginConnect(host, port, null, null);
                        bool connected = connectTask.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(2));
                        if (!connected || !client.Connected)
                        {
                            throw new SocketException();
                        }
                        client.EndConnect(connectTask);

                        _connectedFlag = true;
                        using (var stream = client.GetStream())
                        using (var reader = new System.IO.StreamReader(stream, System.Text.Encoding.UTF8))
                        {
                            while (_running && client.Connected)
                            {
                                string line = reader.ReadLine(); // blocks -- fine, we're on a background thread
                                if (line == null) break; // server closed the connection
                                _incomingLines.Enqueue(line);
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    // Expected whenever main.py isn't running yet, or the
                    // connection drops -- just retry below.
                }

                _connectedFlag = false;
                SleepIfStillRunning(reconnectIntervalSeconds);
            }
        }

        private void SleepIfStillRunning(float seconds)
        {
            float remaining = seconds;
            const float step = 0.1f;
            while (_running && remaining > 0f)
            {
                Thread.Sleep((int)(step * 1000));
                remaining -= step;
            }
        }
    }
}
