#include <metal_stdlib>
#include "../../FrostfieldShaderTypes/include/FrostfieldShaderTypes.h"

using namespace metal;

// Compile-time checks that MSL sees the same record the host writes.
static_assert(sizeof(FFTriangleUniforms) == 16, "FFTriangleUniforms must be 16 bytes");
static_assert(alignof(FFTriangleUniforms) == 8, "FFTriangleUniforms must be 8-byte aligned");
static_assert(sizeof(FFTransformUniforms) == 64, "FFTransformUniforms must be 64 bytes");
static_assert(alignof(FFTransformUniforms) == 16, "FFTransformUniforms must be 16-byte aligned");

// An equilateral triangle centred on the clip-space origin, apex up, with a
// circumradius of 0.75. Pure red, green, and blue corners make channel-order
// and orientation mistakes visible.
constant float2 kTrianglePositions[3] = {
    float2( 0.0,       0.75),
    float2(-0.6495191, -0.375),
    float2( 0.6495191, -0.375),
};

constant float4 kTriangleColors[3] = {
    float4(1.0, 0.0, 0.0, 1.0),
    float4(0.0, 1.0, 0.0, 1.0),
    float4(0.0, 0.0, 1.0, 1.0),
};

struct TriangleVaryings {
    float4 position [[position]];
    float4 color;
};

vertex TriangleVaryings ff_triangle_vertex(
    uint vertexID [[vertex_id]],
    constant FFTriangleUniforms &uniforms [[buffer(FF_TRIANGLE_UNIFORMS_BUFFER_INDEX)]])
{
    float2 p = kTrianglePositions[vertexID];
    float s = sin(uniforms.rotation);
    float c = cos(uniforms.rotation);
    float2 rotated = float2(c * p.x - s * p.y, s * p.x + c * p.y);

    TriangleVaryings out;
    out.position = float4(rotated * uniforms.scale, 0.0, 1.0);
    out.color = kTriangleColors[vertexID];
    return out;
}

// The same triangle as model-space geometry in the z = 0 plane, placed by a
// model transform and viewed through a perspective camera. Its corners run
// counter-clockwise (apex, lower left, lower right) seen from +z.
vertex TriangleVaryings ff_triangle_camera_vertex(
    uint vertexID [[vertex_id]],
    constant FFTransformUniforms &transform [[buffer(FF_TRANSFORM_UNIFORMS_BUFFER_INDEX)]])
{
    TriangleVaryings out;
    out.position = transform.modelViewProjection * float4(kTrianglePositions[vertexID], 0.0, 1.0);
    out.color = kTriangleColors[vertexID];
    return out;
}

fragment float4 ff_triangle_fragment(TriangleVaryings in [[stage_in]])
{
    return in.color;
}
