// Spatial FSR 1 kernels adapted from AMD FidelityFX Super Resolution.
// See LICENSES/AMD-FidelityFX-FSR1-MIT.txt.
#include <metal_stdlib>
using namespace metal;

static float fsrLuma(float3 color) {
    return dot(color, float3(0.5f, 1.0f, 0.5f));
}

static float fsrPeak(float3 color) {
    return max(color.r, max(color.g, color.b));
}

static float4 fsrRead(texture2d<float, access::read> source, int2 pixel, int2 last) {
    return source.read(uint2(clamp(pixel, int2(0), last)));
}

static void fsrEdge(
    thread float2& direction, thread float& energy, float weight,
    float up, float left, float center, float right, float down
) {
    float2 across = float2(right - left, down - up);
    float2 slope = max(abs(float2(right - center, down - center)), abs(float2(center - left, center - up)));
    float2 strength = saturate(abs(across) / max(slope, float2(1.0f / 65536.0f)));
    direction += across * weight;
    energy += dot(strength, strength) * weight;
}

fragment float4 frostfield_fsr1_easu(
    float4 position [[position]], texture2d<float, access::read> source [[texture(0)]],
    constant float2& outputSize [[buffer(0)]]
) {
    int2 size = int2(source.get_width(), source.get_height());
    if (all(float2(size) == outputSize)) { return source.read(uint2(position.xy)); }
    // Keep exact source-texel centers on the intended side of floor. A fast
    // reciprocal can otherwise select a different 12-tap footprint there.
    float2 coordinate = precise::divide(position.xy * float2(size), outputSize) - 0.5f;
    int2 base = int2(floor(coordinate));
    float2 fraction = fract(coordinate);
    int2 last = size - 1;
    // The 12-tap cross omits the four outer corners of the 4 × 4 footprint.
    constexpr int2 offsets[] = {
        int2(0, -1), int2(1, -1), int2(-1, 0), int2(0, 0), int2(1, 0), int2(2, 0),
        int2(-1, 1), int2(0, 1), int2(1, 1), int2(2, 1), int2(0, 2), int2(1, 2)
    };
    float4 taps[12];
    float luma[12];
    for (uint i = 0; i < 12; ++i) {
        taps[i] = fsrRead(source, base + offsets[i], last);
        luma[i] = fsrLuma(taps[i].rgb);
    }

    float2 direction = 0.0f;
    float energy = 0.0f;
    fsrEdge(direction, energy, (1 - fraction.x) * (1 - fraction.y), luma[0], luma[2], luma[3], luma[4], luma[7]);
    fsrEdge(direction, energy, fraction.x * (1 - fraction.y), luma[1], luma[3], luma[4], luma[5], luma[8]);
    fsrEdge(direction, energy, (1 - fraction.x) * fraction.y, luma[3], luma[6], luma[7], luma[8], luma[10]);
    fsrEdge(direction, energy, fraction.x * fraction.y, luma[4], luma[7], luma[8], luma[9], luma[11]);

    float directionSquared = dot(direction, direction);
    direction = directionSquared < 1.0f / 32768.0f ? float2(1, direction.y) : direction * rsqrt(directionSquared);
    float edge = energy * 0.5f;
    edge *= edge;
    float stretch = dot(direction, direction) / max(abs(direction.x), abs(direction.y));
    float2 lengths = float2(1.0f + (stretch - 1.0f) * edge, 1.0f - 0.5f * edge);
    float lobe = 0.5f - 0.29f * edge;
    float limit = 1.0f / lobe;
    float2 perpendicular = float2(-direction.y, direction.x);

    float4 difference = 0.0f;
    float sum = 0.0f;
    float4 anchor = taps[3];
    for (uint i = 0; i < 12; ++i) {
        float2 offset = float2(offsets[i]) - fraction;
        float2 rotated = float2(dot(offset, direction), dot(offset, perpendicular)) * lengths;
        float distance = min(dot(rotated, rotated), limit);
        float a = lobe * distance - 1.0f;
        float b = 0.4f * distance - 1.0f;
        float weight = a * a * (b * b * (25.0f / 16.0f) - (9.0f / 16.0f));
        difference += (taps[i] - anchor) * weight;
        sum += weight;
    }

    float4 low = min(min(taps[3], taps[4]), min(taps[7], taps[8]));
    float4 high = max(max(taps[3], taps[4]), max(taps[7], taps[8]));
    float4 result = clamp(anchor + difference / sum, low, high);
    return result.a > 0.0f ? result : float4(0);
}

static float3 fsrStraight(float4 sample, float3 center) {
    return sample.a > 0.0f ? sample.rgb / sample.a : center;
}

static float3 fsrCompress(float3 color) {
    return color / (1.0f + fsrPeak(color));
}

fragment float4 frostfield_fsr1_rcas(
    float4 position [[position]], texture2d<float, access::read> source [[texture(0)]],
    constant float2& settings [[buffer(0)]]
) {
    int2 pixel = int2(position.xy);
    int2 last = int2(source.get_width(), source.get_height()) - 1;
    float4 center = source.read(uint2(pixel));
    if (center.a <= 0.0f) { return float4(0); }
    float3 current = center.rgb / center.a;
    constexpr int2 offsets[] = {int2(0, -1), int2(-1, 0), int2(1, 0), int2(0, 1)};
    float3 ring[4];
    float peak = fsrPeak(current);
    for (uint i = 0; i < 4; ++i) {
        ring[i] = fsrStraight(fsrRead(source, pixel + offsets[i], last), current);
        peak = max(peak, fsrPeak(ring[i]));
    }

    // Use the reversible HDR compression as a bounded lobe estimator.
    // The reconstruction below remains in the original scene-linear units.
    bool hdr = peak > 1.0f;
    float3 centerProbe = hdr ? fsrCompress(current) : current;
    float3 low = float3(1.0f), high = float3(0.0f);
    float centerLuma = fsrLuma(centerProbe);
    float minLuma = centerLuma, maxLuma = centerLuma, ringLuma = 0.0f;
    float3 difference = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float3 probe = hdr ? fsrCompress(ring[i]) : ring[i];
        low = min(low, probe);
        high = max(high, probe);
        float luma = fsrLuma(probe);
        minLuma = min(minLuma, luma);
        maxLuma = max(maxLuma, luma);
        ringLuma += luma;
        difference += ring[i] - current;
    }
    float3 bottom = -min(low, centerProbe) / max(4.0f * high, float3(1e-6f));
    float3 top = -(1.0f - max(high, centerProbe)) / max(4.0f * (1.0f - low), float3(1e-6f));
    bottom = select(float3(-0.25f), bottom, high > 0.0f);
    top = select(float3(-0.25f), top, low < 1.0f);
    float lobe = clamp(fsrPeak(max(bottom, top)), -0.1875f, 0.0f) * settings.x;
    if (settings.y != 0.0f) {
        float noise = abs(ringLuma * 0.25f - centerLuma);
        lobe *= 1.0f - 0.5f * saturate(noise / max(maxLuma - minLuma, 1.0f / 65536.0f));
    }

    float3 rgb = center.rgb + difference * (center.a * lobe / (1.0f + 4.0f * lobe));
    return float4(clamp(rgb, float3(0), float3(65504.0f)), center.a);
}
