#include <metal_stdlib>
using namespace metal;

constexpr sampler frostfield_screen_sampler(coord::normalized, address::clamp_to_edge, filter::linear);

static float3 brightColor(float3 value, float threshold) {
    float3 rgb = max(value, float3(0));
    float peak = max(rgb.r, max(rgb.g, rgb.b));
    return rgb * (max(peak - threshold, 0.0f) / max(peak, 1e-6f));
}

kernel void frostfield_screen_threshold(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant float2& values [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }

    uint2 origin = pixel * 2u;
    uint2 maximum(source.get_width() - 1u, source.get_height() - 1u);
    float3 bright = float3(0);
    for (uint y = 0; y < 2; ++y) {
        for (uint x = 0; x < 2; ++x) {
            bright += brightColor(source.read(min(origin + uint2(x, y), maximum)).rgb, values.x);
        }
    }
    destination.write(float4(bright * 0.25f, 0), pixel);
}

kernel void frostfield_screen_blur(
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant float2& direction [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }

    float2 size(destination.get_width(), destination.get_height());
    float2 uv = (float2(pixel) + 0.5f) / size;
    float3 color = source.sample(frostfield_screen_sampler, uv, level(0)).rgb * 0.375f;
    color += source.sample(frostfield_screen_sampler, uv + direction, level(0)).rgb * 0.25f;
    color += source.sample(frostfield_screen_sampler, uv - direction, level(0)).rgb * 0.25f;
    color += source.sample(frostfield_screen_sampler, uv + 2.0f * direction, level(0)).rgb * 0.0625f;
    color += source.sample(frostfield_screen_sampler, uv - 2.0f * direction, level(0)).rgb * 0.0625f;
    destination.write(float4(color, 0), pixel);
}

vertex float4 frostfield_screen_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertexID], 0, 1);
}

static float grain(uint2 pixel, uint seed) {
    uint bits = pixel.x * 0x9e3779b9u ^ pixel.y * 0x85ebca6bu ^ seed;
    bits ^= bits >> 16;
    bits *= 0x7feb352du;
    bits ^= bits >> 15;
    bits *= 0x846ca68bu;
    bits ^= bits >> 16;
    return float(bits & 0x00ffffffu) * (2.0f / 16777216.0f) - 1.0f;
}

fragment float4 frostfield_screen_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    texture2d<float> glow [[texture(1)]],
    constant float4& amounts [[buffer(0)]],
    constant uint& seed [[buffer(1)]]
) {
    uint2 pixel = uint2(position.xy);
    float4 center = source.read(pixel);
    if (all(amounts.xyz == 0)) { return center; }
    float alpha = saturate(center.a);
    if (alpha == 0) { return float4(0); }

    float2 size(source.get_width(), source.get_height());
    float2 uv = position.xy / size;
    float3 color = center.rgb;
    if (amounts.y > 0) {
        float2 shift = (uv * 2.0f - 1.0f) * (amounts.y / size);
        float4 red = source.sample(frostfield_screen_sampler, uv + shift);
        float4 blue = source.sample(frostfield_screen_sampler, uv - shift);
        color.r = red.a > 0 ? red.r * (alpha / red.a) : 0;
        color.b = blue.a > 0 ? blue.b * (alpha / blue.a) : 0;
    }
    if (amounts.x > 0) {
        color += glow.sample(frostfield_screen_sampler, uv).rgb * (amounts.x * alpha);
    }
    if (amounts.z > 0) {
        color += float3(grain(pixel, seed) * amounts.z * alpha);
    }
    return float4(clamp(color, float3(0), float3(65504.0f)), alpha);
}
