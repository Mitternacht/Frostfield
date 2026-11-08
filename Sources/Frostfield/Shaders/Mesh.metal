#include <metal_stdlib>
#include "../../FrostfieldShaderTypes/include/FrostfieldShaderTypes.h"

using namespace metal;

static_assert(sizeof(FFTransformUniforms) == 64, "FFTransformUniforms must be 64 bytes");

// One vertex of an indexed mesh, fetched through the renderer's vertex
// descriptor from two streams (FF_MESH_POSITION_BUFFER_INDEX and
// FF_MESH_NORMAL_BUFFER_INDEX).
struct MeshVertexIn {
    float3 position [[attribute(FF_MESH_ATTRIBUTE_POSITION)]];
    float3 normal [[attribute(FF_MESH_ATTRIBUTE_NORMAL)]];
};

// The vertex stage of every mesh material, built-in or app-supplied. Its
// output, FFMeshVaryings, is the material contract's [[stage_in]].
vertex FFMeshVaryings ff_mesh_vertex(
    MeshVertexIn in [[stage_in]],
    constant FFTransformUniforms &transform [[buffer(FF_TRANSFORM_UNIFORMS_BUFFER_INDEX)]])
{
    FFMeshVaryings out;
    out.position = transform.modelViewProjection * float4(in.position, 1.0);
    out.modelNormal = in.normal;
    out.modelPosition = in.position;
    return out;
}

// Built-in diagnostic material: the model-space normal mapped from [-1, 1] to
// [0, 1] per channel, so each face of an axis-aligned shape has its own color
// (+z is (0.5, 0.5, 1.0), -x is (0.0, 0.5, 0.5), and so on) wherever the
// object is turned. It reads no parameters.
fragment float4 ff_mesh_normal_fragment(FFMeshVaryings in [[stage_in]])
{
    return float4(normalize(in.modelNormal) * 0.5 + 0.5, 1.0);
}
