#include <metal_stdlib>
using namespace metal;

struct FFMotionBlurParameters {
    uint sampleCount;
    float shutterScale;
};

fragment float4 frostfield_motion_blur_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::read> vectors [[texture(1)]],
    constant FFMotionBlurParameters& p [[buffer(0)]]
) {
    constexpr sampler linearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    uint2 pixel = uint2(position.xy);
    float4 base = source.read(pixel);
    if (p.shutterScale == 0) { return base; }

    float4 motion = vectors.read(pixel);
    // Test the float bit patterns so fast math cannot discard the nonfinite check.
    uint2 exponent = as_type<uint2>(motion.xy) & uint2(0x7f800000u);
    if (as_type<uint>(motion.w) != 0x3f800000u || any(exponent == 0x7f800000u) || all(motion.xy == 0)) { return base; }

    float2 inverseSize = 1.0f / float2(source.get_width(), source.get_height());
    float2 uv = position.xy * inverseSize;
    float2 travel = motion.xy * p.shutterScale * inverseSize;
    // Accumulating differences keeps a constant frame exact.
    float4 sum = 0;
    float span = float(p.sampleCount - 1);
    for (uint tap = 1; tap <= p.sampleCount; ++tap) {
        float offset = float(tap) / span - 0.5f;
        sum += source.sample(linearClamp, uv + travel * offset, level(0)) - base;
    }
    return base + sum / float(p.sampleCount + 1);
}

// Keep the regular fragment's floating-point operations unchanged for exact
// zero-jitter compatibility. This separate variant adds the blue-noise phase.
struct FFMotionBlurJitter {
    float amount;
    uint phase;
};

static float2 motionBlurNoise(uint2 pixel, uint phase, constant ushort* ranks) {
    // The existing periodic 64x64 rank field supplies two channels. Rotate and
    // offset the second so diagonal motion does not share the same rank.
    uint2 offsetX(phase & 63u, (phase >> 6u) & 63u);
    uint2 offsetY((phase >> 12u) & 63u, (phase >> 18u) & 63u);
    uint2 x = (pixel + offsetX) & 63u;
    uint2 y = (uint2(pixel.y, 63u - (pixel.x & 63u)) + uint2(17u, 31u) + offsetY) & 63u;
    return float2(ranks[x.y * 64u + x.x], ranks[y.y * 64u + y.x]) * (1.0f / 4095.0f);
}

fragment float4 frostfield_motion_blur_jittered_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::read> vectors [[texture(1)]],
    constant FFMotionBlurParameters& p [[buffer(0)]],
    constant FFMotionBlurJitter& jitter [[buffer(1)]],
    constant ushort* ranks [[buffer(2)]]
) {
    constexpr sampler linearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    uint2 pixel = uint2(position.xy);
    float4 base = source.read(pixel);
    if (p.shutterScale == 0) { return base; }

    float4 motion = vectors.read(pixel);
    // Test the float bit patterns so fast math cannot discard the nonfinite check.
    uint2 exponent = as_type<uint2>(motion.xy) & uint2(0x7f800000u);
    if (as_type<uint>(motion.w) != 0x3f800000u || any(exponent == 0x7f800000u) || all(motion.xy == 0)) { return base; }

    float2 inverseSize = 1.0f / float2(source.get_width(), source.get_height());
    float2 uv = position.xy * inverseSize;
    float2 travel = motion.xy * p.shutterScale * inverseSize;
    uv += travel * (jitter.amount * motionBlurNoise(pixel, jitter.phase, ranks));
    // Accumulating differences keeps a constant frame exact.
    float4 sum = 0;
    float span = float(p.sampleCount - 1);
    for (uint tap = 1; tap <= p.sampleCount; ++tap) {
        float offset = float(tap) / span - 0.5f;
        sum += source.sample(linearClamp, uv + travel * offset, level(0)) - base;
    }
    return base + sum / float(p.sampleCount + 1);
}
