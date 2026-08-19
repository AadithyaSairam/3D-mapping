// PointCloudPoints.shader
//
// Renders a mesh with MeshTopology.Points as fixed-pixel-size, per-vertex
// colored dots. Built-in Render Pipeline, no lighting (the SLAM map has no
// normals or light sources -- it's raw triangulated points).
//
// The interesting bit is the PSIZE semantic on the vertex output: it tells
// the GPU's point rasterizer how many pixels wide each point sprite
// should be, which is otherwise fixed at 1px. This is a standard, if
// slightly obscure, HLSL feature for point-cloud rendering; Unity compiles
// it correctly for Direct3D, Metal, and Vulkan (via HLSLcc), which covers
// desktop Windows/Mac -- the target platforms for this project. If you
// ever port this to a platform where this doesn't render (a stray old
// GLES2 device, say), the fallback is billboarded GPU-instanced quads,
// which is more code but universally supported; see docs/ARCHITECTURE.md.
Shader "Mapping3D/PointCloudPoints"
{
    Properties
    {
        _PointSize ("Point Size (px)", Range(1, 20)) = 5
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            float _PointSize;

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float size : PSIZE;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color;
                o.size = _PointSize;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return i.color;
            }
            ENDCG
        }
    }
}
