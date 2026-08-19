// NetworkProtocol.cs
//
// Plain-old-data classes mirroring the NDJSON messages sent by the Python
// SLAM process (see python-slam/slam/streamer.py and docs/PROTOCOL.md).
//
// These are deserialized with Unity's built-in JsonUtility, which is why
// every field here is a public, directly-JSON-shaped field rather than a
// property with custom (de)serialization logic -- JsonUtility works by
// reflecting over exactly this kind of simple class.
//
// One JsonUtility quirk worth knowing (it's *why* TypeProbe exists below):
// JsonUtility.FromJson<T>() happily ignores JSON fields that don't exist
// on T, and leaves C# fields at their default value if the JSON doesn't
// have them. That means we can't deserialize straight into one big
// "AnyMessage" class and branch on which fields ended up non-null -- a
// "pose" message actually has valid values for every field of a would-be
// combined class except type-specific ones, and there's no clean way to
// tell "the JSON had position=[0,0,0]" apart from "the JSON didn't have
// a position field at all". So instead we peek at just the "type" field
// first, then re-parse into the specific message class it names.

using System;
using System.Collections.Generic;

namespace Mapping3D
{
    [Serializable]
    public class TypeProbe
    {
        public string type;
    }

    [Serializable]
    public class PoseMessage
    {
        public string type;
        public int frame;
        public float[] position;   // Unity-space (x, y, z), see streamer.py's coordinate conversion
        public float[] rotation;   // Unity-space quaternion (x, y, z, w)
    }

    [Serializable]
    public class PointData
    {
        public int id;
        public float x;
        public float y;
        public float z;
        public int r;
        public int g;
        public int b;
    }

    [Serializable]
    public class PointsMessage
    {
        public string type;
        public List<PointData> points;
    }

    [Serializable]
    public class StatusMessage
    {
        public string type;
        public string state;
        public int num_points;
        public int num_keyframes;
    }

    [Serializable]
    public class ResetMessage
    {
        public string type;
    }
}
