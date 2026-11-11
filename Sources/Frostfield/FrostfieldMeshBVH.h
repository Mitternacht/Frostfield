#ifndef FROSTFIELD_MESH_BVH_H
#define FROSTFIELD_MESH_BVH_H

#include "FrostfieldRayTriangle.h"

struct FrostfieldMeshBVHRay {
    float3 origin;
    float3 direction;
    float minimum;
    float maximum;
};

enum class FrostfieldMeshBVHSide : uint {
    front,
    back,
    both,
};

struct FrostfieldMeshBVHHit {
    uint triangleIndex;
    uint3 vertexIndices;
    float parameter;
    float3 barycentric;
    float3 geometricNormal;
    bool frontFacing;
};

constant uint frostfieldMeshBVHMiss = 0xffffffffu;

struct FrostfieldMeshBVHNode {
    float4 minimum, maximum;
    uint4 links;
};

static bool frostfieldMeshBVHBounds(
    device const FrostfieldMeshBVHNode& node, FrostfieldMeshBVHRay ray, float limit
) {
    float entry = ray.minimum, exit = limit;
    for (uint axis = 0; axis < 3; ++axis) {
        float origin = ray.origin[axis];
        float minimum = node.minimum[axis], maximum = node.maximum[axis];
        // Widen only the broad phase to retain grazing candidates despite
        // different rounding in the slab and watertight triangle tests.
        float scale = max(abs(origin), max(abs(minimum), abs(maximum)));
        float padding = scale * 1e-6f;
        float lower = (minimum - origin) - padding;
        float upper = (maximum - origin) + padding;
        float step = ray.direction[axis];
        if (step == 0.0f) {
            if (lower > 0.0f || upper < 0.0f) { return false; }
        } else {
            float a = lower / step, b = upper / step;
            entry = max(entry, min(a, b));
            exit = min(exit, max(a, b));
            if (entry > exit) { return false; }
        }
    }
    return true;
}

// data is the buffer bound by MeshBVHShaderBuffer. All ray values are local
// to the geometry. A hit evaluates origin + parameter * direction, with both
// bounds inclusive. The direction is used as supplied.
static FrostfieldMeshBVHHit frostfieldMeshBVHNearest(
    device const uint4* data, FrostfieldMeshBVHRay ray, FrostfieldMeshBVHSide side = FrostfieldMeshBVHSide::both
) {
    FrostfieldMeshBVHHit hit = {frostfieldMeshBVHMiss, uint3(0), 0.0f, float3(0), float3(0), false};
    if (!all(isfinite(ray.origin)) || !all(isfinite(ray.direction)) || all(ray.direction == 0.0f)
        || !isfinite(ray.minimum) || ray.minimum < 0.0f || !(ray.maximum >= ray.minimum)) {
        return hit;
    }
    FrostfieldShearedRay sheared = frostfieldShearRay(ray.origin, ray.direction);
    if (!all(isfinite(sheared.shear))) { return hit; }

    uint4 layout = data[0];
    device const FrostfieldMeshBVHNode* nodes = reinterpret_cast<device const FrostfieldMeshBVHNode*>(data + 1);
    device const uint4* triangles = data + layout.x;
    device const float4* positions = reinterpret_cast<device const float4*>(data + layout.y);
    float best = ray.maximum;

    uint nodeIndex = 0;
    while (nodeIndex < layout.z) {
        device const FrostfieldMeshBVHNode& node = nodes[nodeIndex];
        uint count = node.links.y;
        if (!frostfieldMeshBVHBounds(node, ray, best)) {
            nodeIndex = node.links.z;
            continue;
        }
        if (count == 0u) {
            ++nodeIndex;
            continue;
        }

        uint first = node.links.x;
        for (uint slot = first; slot < first + count; ++slot) {
            uint4 triangle = triangles[slot];
            float parameter;
            float2 weights;
            bool frontFacing;
            if (!frostfieldRayTriangle<true>(
                positions[triangle.x].xyz, positions[triangle.y].xyz, positions[triangle.z].xyz,
                side == FrostfieldMeshBVHSide::front, sheared, ray.minimum, best,
                parameter, weights, frontFacing
            ) || !isfinite(parameter) || !all(isfinite(weights))
                || (side == FrostfieldMeshBVHSide::back && frontFacing)) {
                continue;
            }
            if (parameter == best && triangle.w >= hit.triangleIndex) { continue; }
            best = parameter;
            hit.triangleIndex = triangle.w;
            hit.vertexIndices = triangle.xyz;
            hit.parameter = parameter;
            hit.barycentric = float3(max(0.0f, 1.0f - weights.x - weights.y), weights);
            hit.frontFacing = frontFacing;
        }
        nodeIndex = node.links.z;
    }

    if (hit.triangleIndex != frostfieldMeshBVHMiss) {
        float3 a = positions[hit.vertexIndices.x].xyz;
        float3 ab = positions[hit.vertexIndices.y].xyz - a;
        float3 ac = positions[hit.vertexIndices.z].xyz - a;
        float3 magnitudes = max(abs(ab), abs(ac));
        float scale = max(magnitudes.x, max(magnitudes.y, magnitudes.z));
        if (scale > 0.0f && isfinite(scale)) {
            float3 normal = cross(ab / scale, ac / scale);
            float3 absolute = abs(normal);
            float normalScale = max(absolute.x, max(absolute.y, absolute.z));
            if (normalScale > 0.0f) { hit.geometricNormal = normalize(normal / normalScale); }
        }
    }
    return hit;
}

#endif
