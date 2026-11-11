#include <metal_stdlib>
using namespace metal;

vertex float4 frostfield_planar_shadow_vertex(
    uint index [[vertex_id]],
    const device float3 *positions [[buffer(0)]],
    constant float4x4 &clipFromWorld [[buffer(1)]]
) {
    return clipFromWorld * float4(positions[index], 1);
}

fragment float4 frostfield_planar_shadow_fragment(
    float4 position [[position]],
    depth2d<float, access::read> sceneDepth [[texture(0)]],
    constant float4 &premultipliedTint [[buffer(0)]],
    constant float2 &depthControl [[buffer(1)]]
) {
    float depth = sceneDepth.read(uint2(position.xy));
    float gap = depthControl.y > 0 ? depth - position.z : position.z - depth;
    if (!(depth >= 0.0f && depth <= 1.0f) || gap > depthControl.x) {
        discard_fragment();
    }
    return premultipliedTint;
}
