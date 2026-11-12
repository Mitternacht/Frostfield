#ifndef FROSTFIELD_SHADOW_SAMPLING_H
#define FROSTFIELD_SHADOW_SAMPLING_H

#include <metal_stdlib>

// The directional light's maps, matching DirectionalShadowMap.shaderUniforms (49 float4 slots).
// A single map has one slice; a cascaded map has one slice per cascade, up to eight.
struct FrostfieldDirectionalShadow {
    metal::float4x4 clipFromWorld[8];
    metal::float4 cascade[8];  // depth per UV, depth bias, model-relative flag in first record,
                               // reciprocal half extent
    metal::float4 splits[2];   // far view distance of each cascade, four per vector
    metal::float4 forward;     // view axis or local split plane normal (xyz), blend fraction (w)
    metal::float4 origin;      // view origin (xyz) or local split plane offset (x), map count (w)
    metal::float4 settings;    // x: 1 for a cascaded map, y: tangent of the light's angular radius,
                               // z: fitted camera near distance, w: filtered 0, basic 1, variance 2
    metal::float4 biasRight;   // unit projection x axis (xyz), normal bias in world units (w)
    metal::float4 biasUp;      // unit projection y axis (xyz), shadow strength (w)
    metal::float4 biasDepth;   // unit projection depth axis (xyz), transmitted layer count (w)
    metal::float4 lightColor;  // linear RGB of the directional light (xyz)
};

static inline bool frostfieldAlreadyPeeled(
    metal::float4 position, uint followsPeel, metal::depth2d<float, metal::access::read> previousDepth
) {
    using namespace metal;
    if (followsPeel == 0) { return false; }
    float previous = previousDepth.read(uint2(position.xy));
    // Reject the already-recorded surface, allowing for interpolation rounding.
    return position.z <= previous + 4.0f * FLT_EPSILON;
}

// Read RG32Float moments with clamped bilinear interpolation on every Metal GPU.
static inline metal::float2 frostfieldShadowMoments(
    metal::texture2d_array<float, metal::access::read> moments, metal::float2 uv, uint slice
) {
    using namespace metal;
    int2 size = int2(moments.get_width(), moments.get_height());
    float2 texel = uv * float2(size) - 0.5f;
    int2 corner = int2(floor(texel));
    float2 weight = fract(texel);
    int2 a = clamp(corner, int2(0), size - 1);
    int2 b = clamp(corner + 1, int2(0), size - 1);
    float2 top = mix(moments.read(uint2(a.x, a.y), slice).rg, moments.read(uint2(b.x, a.y), slice).rg, weight.x);
    float2 bottom = mix(moments.read(uint2(a.x, b.y), slice).rg, moments.read(uint2(b.x, b.y), slice).rg, weight.x);
    return mix(top, bottom, weight.y);
}

static inline float frostfieldVarianceVisibility(
    metal::texture2d_array<float, metal::access::read> moments, metal::float2 uv,
    uint slice, float reference, bool reversed
) {
    using namespace metal;
    float2 distribution = frostfieldShadowMoments(moments, uv, slice);
    float distance = reversed ? distribution.x - reference : reference - distribution.x;
    if (distance <= 0.0f) { return 1.0f; }
    // Native perspective depth compresses distant values. A fixed variance
    // floor there would brighten a blocker whose true depth is constant.
    float variance = max(distribution.y - distribution.x * distribution.x, reversed ? 0.0f : 1e-7f);
    if (variance == 0.0f) { return 0.0f; }
    float upperBound = variance / (variance + distance * distance);
    // Three.js r187 remaps the Chebyshev bound to suppress light bleeding.
    return saturate((upperBound - 0.3f) / 0.65f);
}

// Visibility in one orthographic map. `depthPerUV` converts map UV distances to
// normalized depth; it is 1 when both span the same world distance.
static inline float frostfieldShadowMapVisibility(
    metal::depth2d_array<float> shadowMap, metal::texture2d_array<float, metal::access::read> moments,
    uint slice, metal::float4 lightClip, float facing,
    float depthPerUV, float bias, float tangent, uint sampling
) {
    using namespace metal;
    float3 lightNDC = lightClip.xyz / lightClip.w;
    float2 uv = float2(lightNDC.x * 0.5 + 0.5, 0.5 - lightNDC.y * 0.5);
    if (facing <= 0 || any(uv < 0) || any(uv > 1) || lightNDC.z < 0 || lightNDC.z > 1) {
        return 1;
    }
    if (sampling == 2u) {
        return frostfieldVarianceVisibility(moments, uv, slice, lightNDC.z - bias, false);
    }
    if (sampling == 1u) {
        constexpr sampler nearest(
            coord::normalized, address::clamp_to_edge, filter::nearest, compare_func::less_equal
        );
        return shadowMap.sample_compare(nearest, uv, slice, lightNDC.z - bias);
    }

    constexpr sampler shadowSampler(
        coord::normalized, address::clamp_to_edge, filter::linear, compare_func::less_equal
    );
    // Linear comparisons sample neighboring texels. Account for depth
    // variation across that footprint and cap the grazing-angle offset.
    float slope = sqrt(1 - facing * facing) / max(facing, 0.25);
    float texel = 1.0f / float(shadowMap.get_width());
    float receiver = lightNDC.z - max(bias, M_SQRT2_F * slope * texel * depthPerUV);
    if (tangent <= 0) {
        return shadowMap.sample_compare(shadowSampler, uv, slice, receiver);
    }

    // Search for nearby blockers, then grow the filter with blocker separation,
    // measured in map UV so the penumbra has the same world width in every cascade.
    constexpr sampler blockerSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    constexpr float2 blockers[12] = {
        float2( 0.000,  0.000), float2( 0.310,  0.055), float2(-0.235,  0.270),
        float2( 0.090, -0.410), float2( 0.405,  0.320), float2(-0.544, -0.108),
        float2( 0.421, -0.507), float2(-0.136,  0.661), float2(-0.418, -0.648),
        float2( 0.758, -0.071), float2( 0.449,  0.750), float2(-0.840,  0.394)
    };
    float search = clamp(max(receiver, 0.0f) / depthPerUV * tangent, texel, 16.0f * texel);
    float blockerSum = 0;
    float blockerCount = 0;
    for (uint i = 0; i < 12; ++i) {
        float depth = shadowMap.sample(blockerSampler, uv + blockers[i] * search, slice);
        if (depth < receiver) {
            blockerSum += depth;
            blockerCount += 1;
        }
    }
    if (blockerCount == 0) {
        return shadowMap.sample_compare(shadowSampler, uv, slice, receiver);
    }

    float penumbra = (receiver - blockerSum / blockerCount) / depthPerUV * tangent;
    if (penumbra <= texel) {
        return shadowMap.sample_compare(shadowSampler, uv, slice, receiver);
    }
    float radius = min(penumbra, 16.0f * texel);
    constexpr float2 filterDisk[24] = {
        float2( 0.1443,  0.0000), float2(-0.1843,  0.1689), float2( 0.0282, -0.3215),
        float2( 0.2324,  0.3031), float2(-0.4264, -0.0754), float2( 0.4039, -0.2569),
        float2(-0.1351,  0.5026), float2(-0.2577, -0.4961), float2( 0.5590,  0.2041),
        float2(-0.5816,  0.2401), float2( 0.2803, -0.5991), float2( 0.2072,  0.6605),
        float2(-0.6244, -0.3619), float2( 0.7325, -0.1610), float2(-0.4470,  0.6359),
        float2(-0.1033, -0.7970), float2( 0.6340,  0.5343), float2(-0.8532,  0.0353),
        float2( 0.6223, -0.6193), float2(-0.0416,  0.9004), float2(-0.5922, -0.7096),
        float2( 0.9380,  0.1262), float2(-0.7948,  0.5530), float2( 0.2172, -0.9654)
    };
    // Anchor the rotation to shadow texels so the wider filter has no regular bands
    // and stays stable when the viewing camera moves.
    float2 cell = floor(uv / texel);
    float angle = 2 * M_PI_F * fract(52.9829189f * fract(dot(cell, float2(0.06711056f, 0.00583715f))));
    float cosine;
    float sine = sincos(angle, cosine);
    float2x2 rotation(float2(cosine, sine), float2(-sine, cosine));
    float visible = 0;
    for (uint i = 0; i < 24; ++i) {
        visible += shadowMap.sample_compare(shadowSampler, uv + rotation * filterDisk[i] * radius, slice, receiver);
    }
    return visible / 24.0f;
}

static inline metal::float4 frostfieldDirectionalProjection(
    uint slice, metal::float3 position, metal::float3 worldGeometricNormal,
    constant FrostfieldDirectionalShadow& shadow
) {
    using namespace metal;
    float4 lightClip = shadow.clipFromWorld[slice] * float4(position, 1);
    float4 cascade = shadow.cascade[slice];
    if (shadow.biasRight.w != 0.0f) {
        // The receiver position can be model-relative. Project its world-space
        // normal separately and omit the light projection's translation.
        float3 lightNormal = float3(
            dot(shadow.biasRight.xyz, worldGeometricNormal),
            dot(shadow.biasUp.xyz, worldGeometricNormal),
            dot(shadow.biasDepth.xyz, worldGeometricNormal)
        );
        float3 projectionScale = float3(cascade.w, cascade.w, (0.5f * cascade.x) * cascade.w);
        lightClip.xyz += (lightNormal * shadow.biasRight.w) * projectionScale;
    }
    return lightClip;
}

struct FrostfieldDirectionalSelection {
    metal::float3 position;
    uint index;
    float blend;
};

static inline float frostfieldCascadeSplit(constant FrostfieldDirectionalShadow& shadow, uint index) {
    return shadow.splits[index >> 2][index & 3];
}

// The last part of each cascade cross-fades into the next. The last cascade
// fades to unshadowed, and surfaces beyond its split are fully unshadowed.
static inline FrostfieldDirectionalSelection frostfieldDirectionalSelection(
    metal::float3 worldPosition, metal::float3 localPosition, constant FrostfieldDirectionalShadow& shadow
) {
    bool relative = shadow.cascade[0].z != 0.0f;
    metal::float3 position = relative ? localPosition : worldPosition;
    if (shadow.settings.x == 0.0f) { return { position, 0u, 0.0f }; }
    uint count = uint(shadow.origin.w);
    float distance = relative ? metal::dot(localPosition, shadow.forward.xyz) + shadow.origin.x
                              : metal::dot(worldPosition - shadow.origin.xyz, shadow.forward.xyz);
    uint index = 0u;
    while (index < count && distance > frostfieldCascadeSplit(shadow, index)) {
        ++index;
    }
    if (index >= count) { return { position, index, 0.0f }; }
    float far = frostfieldCascadeSplit(shadow, index);
    float near = index == 0u ? shadow.settings.z : frostfieldCascadeSplit(shadow, index - 1u);
    float band = (far - near) * shadow.forward.w;
    float blend = band > 0 && distance > far - band ? metal::saturate((distance - (far - band)) / band) : 0.0f;
    return { position, index, blend };
}

static inline float frostfieldOpaqueCascadeVisibility(
    metal::depth2d_array<float> shadowMap, metal::texture2d_array<float, metal::access::read> moments,
    uint slice, metal::float4 lightClip, float facing, constant FrostfieldDirectionalShadow& shadow
) {
    return frostfieldShadowMapVisibility(
        shadowMap, moments, slice, lightClip, facing,
        shadow.cascade[slice].x, shadow.cascade[slice].y, shadow.settings.y, uint(shadow.settings.w)
    );
}

static inline float frostfieldShadowVisibility(
    metal::depth2d_array<float> shadowMap, metal::texture2d_array<float, metal::access::read> moments,
    metal::float3 worldPosition, metal::float3 localPosition, metal::float3 worldGeometricNormal, float facing,
    constant FrostfieldDirectionalShadow& shadow
) {
    float strength = shadow.biasUp.w;
    if (strength == 0.0f) { return 1.0f; }
    FrostfieldDirectionalSelection selection = frostfieldDirectionalSelection(worldPosition, localPosition, shadow);
    uint slice = selection.index, count = uint(shadow.origin.w);
    if (slice >= count) { return 1.0f; }
    metal::float4 clip = frostfieldDirectionalProjection(slice, selection.position, worldGeometricNormal, shadow);
    float visible = frostfieldOpaqueCascadeVisibility(shadowMap, moments, slice, clip, facing, shadow);
    if (selection.blend > 0.0f) {
        float next = 1.0f;
        if (slice + 1u < count) {
            clip = frostfieldDirectionalProjection(slice + 1u, selection.position, worldGeometricNormal, shadow);
            next = frostfieldOpaqueCascadeVisibility(shadowMap, moments, slice + 1u, clip, facing, shadow);
        }
        visible = metal::mix(visible, next, selection.blend);
    }
    return strength == 1.0f ? visible : metal::mix(1.0f, visible, strength);
}

static inline metal::float3 frostfieldTransmittedTexel(
    metal::texture2d_array<float, metal::access::read> transmission, metal::uint2 pixel,
    uint firstLayer, uint layerCount, float receiverDepth
) {
    using namespace metal;
    float3 tint = float3(1.0f);
    for (uint i = 0; i < layerCount; ++i) {
        float4 layer = transmission.read(pixel, firstLayer + i);
        if (layer.w >= receiverDepth) { break; }
        tint *= layer.rgb;
    }
    return tint;
}

static inline metal::float3 frostfieldTransmittedCascadeVisibility(
    metal::depth2d_array<float> shadowMap, metal::texture2d_array<float, metal::access::read> moments,
    metal::texture2d_array<float, metal::access::read> transmission,
    uint slice, metal::float3 position, metal::float3 worldGeometricNormal, float facing,
    constant FrostfieldDirectionalShadow& shadow
) {
    using namespace metal;
    float4 clip = frostfieldDirectionalProjection(slice, position, worldGeometricNormal, shadow);
    float visible = frostfieldOpaqueCascadeVisibility(shadowMap, moments, slice, clip, facing, shadow);
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
    if (visible == 0.0f || facing <= 0.0f || any(uv < 0.0f) || any(uv > 1.0f) || ndc.z < 0.0f || ndc.z > 1.0f) {
        return float3(visible);
    }
    // Resolve every tap at the receiver depth before filtering its transmittance.
    // Filtering the stored depths first would pull a shadow across a pane.
    int2 size = int2(transmission.get_width(), transmission.get_height());
    float2 texel = uv * float2(size) - 0.5f;
    int2 corner = int2(floor(texel));
    int2 a = clamp(corner, int2(0), size - 1);
    int2 b = clamp(corner + 1, int2(0), size - 1);
    float2 w = fract(texel);
    float receiver = ndc.z - shadow.cascade[slice].y;
    uint layerCount = uint(shadow.biasDepth.w);
    uint first = slice * layerCount;
    float3 aa = frostfieldTransmittedTexel(transmission, uint2(a.x, a.y), first, layerCount, receiver);
    float3 ba = frostfieldTransmittedTexel(transmission, uint2(b.x, a.y), first, layerCount, receiver);
    float3 ab = frostfieldTransmittedTexel(transmission, uint2(a.x, b.y), first, layerCount, receiver);
    float3 bb = frostfieldTransmittedTexel(transmission, uint2(b.x, b.y), first, layerCount, receiver);
    return visible * mix(mix(aa, ba, w.x), mix(ab, bb, w.x), w.y);
}

static inline metal::float3 frostfieldTransmittedShadowVisibility(
    metal::depth2d_array<float> shadowMap, metal::texture2d_array<float, metal::access::read> moments,
    metal::texture2d_array<float, metal::access::read> transmission,
    metal::float3 worldPosition, metal::float3 localPosition, metal::float3 worldGeometricNormal, float facing,
    constant FrostfieldDirectionalShadow& shadow
) {
    using namespace metal;
    if (shadow.biasDepth.w == 0.0f) {
        return float3(frostfieldShadowVisibility(shadowMap, moments, worldPosition, localPosition, worldGeometricNormal, facing, shadow));
    }
    float strength = shadow.biasUp.w;
    if (strength == 0.0f) { return float3(1.0f); }
    FrostfieldDirectionalSelection selection = frostfieldDirectionalSelection(worldPosition, localPosition, shadow);
    uint slice = selection.index, count = uint(shadow.origin.w);
    if (slice >= count) { return float3(1.0f); }
    float3 visible = frostfieldTransmittedCascadeVisibility(
        shadowMap, moments, transmission, slice, selection.position, worldGeometricNormal, facing, shadow
    );
    if (selection.blend > 0.0f) {
        float3 next = slice + 1u < count
            ? frostfieldTransmittedCascadeVisibility(
                shadowMap, moments, transmission, slice + 1u, selection.position, worldGeometricNormal, facing, shadow
            )
            : float3(1.0f);
        visible = mix(visible, next, selection.blend);
    }
    return strength == 1.0f ? visible : mix(float3(1.0f), visible, strength);
}

#endif
