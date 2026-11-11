#include <metal_stdlib>
using namespace metal;

struct TriangleVertex {
    float4 position [[position]];
    float3 color;
};

vertex TriangleVertex frostfield_triangle_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {
        float2(0.0, 0.65), float2(-0.64, -0.52), float2(0.64, -0.52)
    };
    const float3 colors[] = {
        float3(0.07, 0.72, 1.0), float3(0.35, 0.09, 1.0), float3(1.0, 0.24, 0.38)
    };
    return {float4(positions[vertexID], 0.0, 1.0), colors[vertexID]};
}

fragment float4 frostfield_triangle_fragment(TriangleVertex input [[stage_in]]) {
    return float4(input.color, 1.0);
}
