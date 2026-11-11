#ifndef FROSTFIELD_SCREEN_SPACE_H
#define FROSTFIELD_SCREEN_SPACE_H

#include <metal_stdlib>

// Shared by raster PBR shading, the surface attribute prepass, ground-truth ambient
// occlusion and screen-space reflections.

// Octahedral mapping of a unit vector to [-1, 1]² (Cigolle et al. 2014).
static inline metal::float2 frostfieldOctahedralEncode(metal::float3 n) {
    using namespace metal;
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20f);
    float2 folded = (1.0f - abs(n.yx)) * select(float2(-1.0f), float2(1.0f), n.xy >= 0.0f);
    return n.z >= 0.0f ? n.xy : folded;
}

static inline metal::float3 frostfieldOctahedralDecode(metal::float2 e) {
    using namespace metal;
    float3 n = float3(e, 1.0f - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= 0.0f);
    return normalize(n);
}

// One RG32Uint texel per pixel: the world-space shading normal as two snorm16 octahedral
// coordinates, then the square root of the RGB specular reflectance and the perceptual
// roughness as unorm8 values.
static inline metal::uint2 frostfieldPackSurfaceAttributes(
    metal::float3 normal, float roughness, metal::float3 specularReflectance
) {
    using namespace metal;
    return uint2(
        pack_float_to_snorm2x16(frostfieldOctahedralEncode(normal)),
        pack_float_to_unorm4x8(float4(sqrt(saturate(specularReflectance)), saturate(roughness)))
    );
}

struct FrostfieldSurfaceAttributes {
    metal::float3 normal;    // world space, unit length
    float roughness;         // perceptual
    metal::float3 specular;  // directional specular reflectance at the stored view
};

static inline FrostfieldSurfaceAttributes frostfieldUnpackSurfaceAttributes(metal::uint2 packed) {
    using namespace metal;
    float4 reflectance = unpack_unorm4x8_to_float(packed.y);
    return {frostfieldOctahedralDecode(unpack_snorm2x16_to_float(packed.x)), reflectance.w, reflectance.rgb * reflectance.rgb};
}

// Karis's analytic fit of the split-sum GGX directional albedo terms (A, B), used where
// a material has no environment BRDF lookup table ("Physically Based Shading on Mobile", 2014).
static inline metal::float2 frostfieldEnvironmentBRDFApproximation(float roughness, float noV) {
    using namespace metal;
    const float4 c0 = float4(-1.0f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4(1.0f, 0.0425f, 1.04f, -0.04f);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28f * noV)) * r.x + r.y;
    return float2(-1.04f, 1.04f) * a004 + r.zw;
}

// Jimenez et al. 2016 multiple-bounce fit: occlusion that accounts for light
// interreflected by surrounding surfaces of this albedo. The fit reaches 1.0001 at full
// visibility for white albedo, so it is capped at one.
static inline metal::float3 frostfieldMultipleBounce(float visibility, metal::float3 albedo) {
    using namespace metal;
    float3 a = 2.0404f * albedo - 0.3324f;
    float3 b = -4.7951f * albedo + 0.6417f;
    float3 c = 2.7552f * albedo + 0.6903f;
    return min(max(float3(visibility), ((visibility * a + b) * visibility + c) * visibility), float3(1));
}

// Solid angle shared by two spherical caps with these cosines of their half-angles,
// whose axes are `cosDistance` apart. Oat and Sander 2007, with a smoothstep lune.
static inline float frostfieldCapIntersection(float cosFirst, float cosSecond, float cosDistance) {
    using namespace metal;
    float first = acos(clamp(cosFirst, -1.0f, 1.0f));
    float second = acos(clamp(cosSecond, -1.0f, 1.0f));
    float distance = acos(clamp(cosDistance, -1.0f, 1.0f));
    float smaller = 2.0f * M_PI_F * (1.0f - max(cosFirst, cosSecond));
    if (min(first, second) <= max(first, second) - distance) {
        return smaller;
    }
    if (first + second <= distance) {
        return 0.0f;
    }
    float delta = abs(first - second);
    float x = 1.0f - saturate((distance - delta) / max(first + second - delta, 1e-4f));
    return smaller * x * x * (3.0f - 2.0f * x);
}

// Specular occlusion from a bent normal and its cosine-weighted visibility: the share of
// the specular lobe's cone that lies inside the visibility cone (Jimenez et al. 2016).
// The visibility cone holds `visibility` of the cosine-weighted hemisphere, so its
// half-angle has cos² = 1 − visibility. The lobe cone holds 3/4 of the energy of the
// Phong lobe equivalent to GGX at this roughness. The overlap is divided by the lobe's
// overlap with the open hemisphere around the surface normal, not by the whole lobe: the
// split-sum terms already account for the horizon, and this keeps the result continuous
// as visibility reaches one, where it is exactly one.
static inline float frostfieldSpecularOcclusion(
    metal::float3 bentNormal, metal::float3 normal, float visibility, metal::float3 reflection, float roughness
) {
    using namespace metal;
    if (visibility >= 1.0f) {
        return 1.0f;
    }
    float alpha = max(roughness * roughness, 1e-3f);
    float exponent = max(2.0f / (alpha * alpha) - 2.0f, 0.0f);
    float cosLobe = pow(0.25f, 1.0f / (exponent + 1.0f));
    float cosVisible = sqrt(saturate(1.0f - visibility));
    float open = frostfieldCapIntersection(0.0f, cosLobe, dot(normal, reflection));
    if (!(open > 0.0f)) {
        return 1.0f;
    }
    return saturate(frostfieldCapIntersection(cosVisible, cosLobe, dot(bentNormal, reflection)) / open);
}

// The screen-space occlusion that a PBR fragment at this attachment position receives:
// the world-space bend to add to the shading normal (xyz) and visibility (w), or w = -1
// when `info.x` is zero.
static inline metal::float4 frostfieldScreenOcclusion(
    metal::texture2d<float> map, metal::float4 info, metal::float2 position
) {
    using namespace metal;
    if (info.x == 0.0f) {
        return float4(0, 0, 1, -1);
    }
    uint2 texel = min(uint2(position), uint2(map.get_width() - 1, map.get_height() - 1));
    float4 value = map.read(texel);
    return float4(value.xyz, saturate(value.w));
}

#endif
