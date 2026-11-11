#ifndef FROSTFIELD_SOFT_PARTICLE_H
#define FROSTFIELD_SOFT_PARTICLE_H
#include <metal_stdlib>

static inline float frostfieldParticleViewZ(float depth, metal::float4 projection) {
    return (projection.y - depth * projection.w) / (depth * projection.z - projection.x);
}

static inline float frostfieldParticleFragmentViewZ(
    metal::float2 windowDepthAndReciprocalW, metal::float2 windowRange, metal::float4 projection
) {
    if (projection.z < 0.0f) {
        return (1.0f / windowDepthAndReciprocalW.y - projection.w) / projection.z;
    }
    float depth = (windowDepthAndReciprocalW.x - windowRange.x) / (windowRange.y - windowRange.x);
    return frostfieldParticleViewZ(depth, projection);
}

static inline float frostfieldParticleContrast(float particleZ, float sceneZ, float distance, float contrast) {
    using namespace metal;
    float gap = saturate((particleZ - sceneZ) / distance);
    float folded = gap > 0.5f ? 1.0f - gap : gap;
    float output = 0.5f * pow(2.0f * folded, contrast);
    return gap > 0.5f ? 1.0f - output : output;
}
#endif
