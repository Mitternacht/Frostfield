#include <metal_stdlib>
using namespace metal;

vertex float4 frostfield_grade_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertexID], 0, 1);
}

fragment float4 frostfield_grade_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    constant float &strength [[buffer(0)]]
) {
    float4 color = source.read(uint2(position.xy));
    if (strength == 0.0) {
        return color;
    }

    float3 warm = min(color.rgb * float3(1.125, 1.0, 0.8125), float3(color.a));
    return float4(mix(color.rgb, warm, strength), color.a);
}
