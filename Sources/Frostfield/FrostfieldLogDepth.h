#ifndef FROSTFIELD_LOG_DEPTH_H
#define FROSTFIELD_LOG_DEPTH_H
#include <metal_stdlib>

// Buffer 29 holds the effective near plane, inverse log2 range, log2 near,
// and finite far plane. A zero far field selects linear orthographic depth.
static inline float frostfieldLogDepth(metal::float4 position, metal::float4 range) {
    using namespace metal;
    if (range.w == 0.0f) { return position.z; }
    float distance = 1.0f / position.w;
    // The forward clip matrix's far plane can round to infinity at large ratios.
    if (!(distance <= range.w)) { discard_fragment(); }
    float ratio = distance / range.x;
    float logDistance = isfinite(ratio) ? log2(ratio) : log2(distance) - range.z;
    return saturate(logDistance * range.y);
}

// Inverse of the perspective depth mapping above. Evaluating in the log
// domain keeps the intermediate finite even if far / near overflows Float.
static inline float frostfieldDistanceFromLogarithmicDepth(float depth, float logNear, float inverseLogRange, float farPlane) {
    using namespace metal;
    return min(exp2(logNear + depth / inverseLogRange), farPlane);
}

struct FrostfieldLogColor {
    metal::float4 color [[color(0)]];
    float depth [[depth(any)]];
};

struct FrostfieldLogAttributes {
    metal::uint2 attributes [[color(0)]];
    float depth [[depth(any)]];
};

#define FROSTFIELD_LOG_DEPTH_INPUT , constant float4& logDepthRange [[buffer(29)]]
#endif
