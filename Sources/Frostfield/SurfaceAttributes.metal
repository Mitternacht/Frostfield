#include <metal_stdlib>
#include "FrostfieldLogDepth.h"
#include "FrostfieldScreenSpace.h"
using namespace metal;

struct FrostfieldAttributeVaryings {
    float4 position [[position]];
    float3 normal [[user(frostfield_normal)]];
};

// Surface attributes for meshes without PBR materials: their geometric normal, with no
// specular reflectance so screen-space reflections leave them unchanged.
static uint2 frostfieldMeshAttributes(FrostfieldAttributeVaryings input, bool front) {
    float3 normal = normalize(input.normal);
    if (!all(isfinite(normal))) {
        normal = float3(0, 0, 1);
    }
    return frostfieldPackSurfaceAttributes(front ? normal : -normal, 1.0f, float3(0));
}

fragment uint2 frostfield_mesh_attributes(
    FrostfieldAttributeVaryings input [[stage_in]], bool front [[front_facing]]
) {
    return frostfieldMeshAttributes(input, front);
}

fragment FrostfieldLogAttributes frostfield_mesh_attributes_log(
    FrostfieldAttributeVaryings input [[stage_in]], bool front [[front_facing]]
    FROSTFIELD_LOG_DEPTH_INPUT
) {
    return {frostfieldMeshAttributes(input, front), frostfieldLogDepth(input.position, logDepthRange)};
}
