#ifndef FROSTFIELD_RAY_TRIANGLE_H
#define FROSTFIELD_RAY_TRIANGLE_H

#include <metal_stdlib>
using namespace metal;

// A ray prepared for the watertight test of Woop, Benthin and Wald 2013, "Watertight
// Ray/Triangle Intersection": the dominant axis kz, the other two in winding order, and the
// shear that maps the direction onto +kz.
struct FrostfieldShearedRay {
    float3 origin;
    uint kx, ky, kz;
    float3 shear;
};

static FrostfieldShearedRay frostfieldShearRay(float3 origin, float3 direction) {
    float3 magnitude = abs(direction);
    uint kz = magnitude.x > magnitude.y ? (magnitude.x > magnitude.z ? 0u : 2u) : (magnitude.y > magnitude.z ? 1u : 2u);
    uint kx = kz == 2u ? 0u : kz + 1u;
    uint ky = kx == 2u ? 0u : kx + 1u;
    if (direction[kz] < 0.0f) {
        uint swap = kx;
        kx = ky;
        ky = swap;
    }
    float inverse = 1.0f / direction[kz];
    return {origin, kx, ky, kz, float3(direction[kx] * inverse, direction[ky] * inverse, inverse)};
}

// Each edge function is a difference of two products that the opposite triangle across a
// shared edge evaluates with its operands swapped, so the two results are exact negatives and
// a ray through the edge enters one of the triangles. Fused multiply-add would round the
// products differently on each side, so contraction stays off here.
template<bool inclusiveFar>
static bool frostfieldRayTriangle(
    float3 a, float3 b, float3 c, bool cullsBackFaces, thread const FrostfieldShearedRay& ray, float minimum, float limit,
    thread float& distance, thread float2& barycentric, thread bool& frontFacing
) {
#pragma METAL fp contract(off)
    float3 relativeA = a - ray.origin;
    float3 relativeB = b - ray.origin;
    float3 relativeC = c - ray.origin;
    float ax = relativeA[ray.kx] - ray.shear.x * relativeA[ray.kz];
    float ay = relativeA[ray.ky] - ray.shear.y * relativeA[ray.kz];
    float bx = relativeB[ray.kx] - ray.shear.x * relativeB[ray.kz];
    float by = relativeB[ray.ky] - ray.shear.y * relativeB[ray.kz];
    float cx = relativeC[ray.kx] - ray.shear.x * relativeC[ray.kz];
    float cy = relativeC[ray.ky] - ray.shear.y * relativeC[ray.kz];
    float u = cx * by - cy * bx;
    float v = ax * cy - ay * cx;
    float w = bx * ay - by * ax;
    if ((u < 0.0f || v < 0.0f || w < 0.0f) && (u > 0.0f || v > 0.0f || w > 0.0f)) { return false; }
    float determinant = u + v + w;
    // A positive determinant faces the ray: its counterclockwise corners see the ray arrive.
    if (determinant == 0.0f || (cullsBackFaces && determinant < 0.0f)) { return false; }
    frontFacing = determinant > 0.0f;
    float scaled = ray.shear.z * (u * relativeA[ray.kz] + v * relativeB[ray.kz] + w * relativeC[ray.kz]);
    float inverse = 1.0f / determinant;
    distance = scaled * inverse;
    barycentric = float2(v, w) * inverse;
    return distance >= minimum && (inclusiveFar ? distance <= limit : distance < limit);
}

#endif
