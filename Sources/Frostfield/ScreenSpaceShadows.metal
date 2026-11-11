#include <metal_stdlib>
#include "FrostfieldLogDepth.h"
using namespace metal;

// Matches ScreenSpaceShadows.Uniforms: five float4 slots and one uint4.
struct FrostfieldScreenShadowView {
    float4 projection; // reciprocal scales and lens shifts
    float4 depth;      // depth coefficients, projection kind, reversed flag
    float4 ray;        // direction toward the light in view space, and maximum distance
    float4 settings;   // thickness, depth bias, intensity, sample quality
    float4 extent;     // trace width and height, near and far
    uint4 sampling;    // maximum steps and caller frame index
};

static bool screenShadowSurface(float raw, bool reversed) {
    return reversed ? raw > 0.0f && raw <= 1.0f : raw >= 0.0f && raw < 1.0f;
}

static float screenShadowDistance(float raw, constant FrostfieldScreenShadowView& view) {
    if (view.depth.z == 2.0f) {
        return frostfieldDistanceFromLogarithmicDepth(raw, view.depth.x, view.depth.y, view.extent.w);
    }
    bool reversed = view.depth.w != 0.0f;
    if (view.depth.z == 1.0f) {
        return fma(reversed ? 1.0f - raw : raw, view.depth.y, view.depth.x);
    }
    return 1.0f / fma(reversed ? raw : 1.0f - raw, view.depth.y, view.depth.x);
}

static float screenShadowOffset(float2 pixel, uint frameIndex) {
    // Interleaved gradient noise decorrelates adjacent rays. Four repeating
    // spatial offsets and an integer hash vary the subpixel phase across frames.
    constexpr float offsets[] = {0.0f, 0.5f, 0.25f, 0.75f};
    float noise = fract(52.9829189f * fract(dot(pixel, float2(0.06711056f, 0.00583715f))));
    uint value = uint(pixel.x) * 747796405u ^ uint(pixel.y) * 2891336453u ^ frameIndex;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    float random = float((value ^ (value >> 22u)) >> 8u) * (1.0f / 16777216.0f);
    return fract(noise + offsets[frameIndex % 4u] + random);
}

fragment float4 frostfield_screen_shadow_visibility(
    float4 screen [[position]], depth2d<float, access::read> depth [[texture(0)]],
    constant FrostfieldScreenShadowView& view [[buffer(0)]]
) {
    constexpr float4 visible(1.0f);
    if (!(view.settings.z > 0.0f && view.settings.w > 0.0f)) { return visible; }
    float2 extent = view.extent.xy;
    uint2 depthSize(depth.get_width(), depth.get_height());
    uint2 address = min(uint2(screen.xy * float2(depthSize) / extent), depthSize - 1u);
    bool reversed = view.depth.w != 0.0f;
    float raw = depth.read(address);
    if (!screenShadowSurface(raw, reversed)) { return visible; }
    float distance = screenShadowDistance(raw, view);
    bool orthographic = view.depth.z == 1.0f;
    if (!isfinite(distance) || (!orthographic && !(distance > 0.0f))) { return visible; }

    // A reduced trace starts from an actual full-resolution depth sample.
    float2 uv = (float2(address) + 0.5f) / float2(depthSize);
    float2 ndc = (uv - 0.5f) * float2(2, -2);
    float2 plane = (ndc + view.projection.zw) * view.projection.xy;
    float3 start = float3(plane * (orthographic ? 1.0f : distance), -distance);
    float travel = view.ray.w;
    if (view.ray.z > 0.0f) { travel = min(travel, (distance - view.extent.z) / view.ray.z); }
    if (view.ray.z < 0.0f) { travel = min(travel, (view.extent.w - distance) / -view.ray.z); }
    if (!(travel > 0.0f)) { return visible; }
    float3 end = start + view.ray.xyz * travel;
    float2 endNDC = end.xy / ((orthographic ? 1.0f : -end.z) * view.projection.xy) - view.projection.zw;
    float2 first = uv * extent;
    float2 delta = (endNDC * float2(0.5f, -0.5f) + 0.5f) * extent - first;
    if (!all(isfinite(delta))) { return visible; }

    // Clip to the viewport before choosing a step count. Off-screen travel
    // cannot consume the budget or reduce the on-screen sample density.
    float2 room = select(first, extent - first, delta > 0.0f);
    float2 exit = select(float2(INFINITY), room / abs(delta), abs(delta) > 0.0f);
    float limit = min(1.0f, min(exit.x, exit.y));
    float pixels = max(abs(delta.x), abs(delta.y)) * limit;
    uint steps = uint(min(float(view.sampling.x), floor(pixels * view.settings.w)));
    float phase = screenShadowOffset(screen.xy, view.sampling.y);
    for (uint i = 0; i < steps; ++i) {
        float fraction = (float(i) + phase) / float(steps) * limit;
        float2 sample = first + delta * fraction;
        if (any(sample < 0.0f) || any(sample >= extent)) { break; }
        uint2 tap = min(uint2(sample * float2(depthSize) / extent), depthSize - 1u);
        if (all(tap == address)) { continue; }
        float tapRaw = depth.read(tap);
        if (!screenShadowSurface(tapRaw, reversed)) { continue; }
        float rayDepth = orthographic ? mix(distance, -end.z, fraction)
            : 1.0f / mix(1.0f / distance, 1.0f / -end.z, fraction);
        float behind = rayDepth - screenShadowDistance(tapRaw, view);
        if (behind > view.settings.y && behind < view.settings.x) { return float4(1.0f - view.settings.z); }
    }
    return visible;
}

fragment float4 frostfield_screen_shadow_composite(
    float4 screen [[position]], texture2d<float, access::read> beauty [[texture(0)]],
    texture2d<float, access::read> directional [[texture(1)]], texture2d<float> visibility [[texture(2)]]
) {
    constexpr sampler maskSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    uint2 pixel = uint2(screen.xy);
    float4 source = beauty.read(pixel);
    float2 uv = screen.xy / float2(beauty.get_width(), beauty.get_height());
    float mask = saturate(visibility.sample(maskSampler, uv, level(0)).r);
    if (mask == 1.0f) { return source; }
    return float4(source.rgb - (1.0f - mask) * directional.read(pixel).rgb, source.a);
}
