#include <metal_stdlib>
using namespace metal;

vertex float4 frostfield_dof_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertexID], 0, 1);
}

static float focusOverDistance(float depth, float2 ratios, bool reversed) {
    float nearWeight = reversed ? depth : 1.0f - depth;
    return mix(ratios.x, ratios.y, nearWeight);
}

fragment float4 frostfield_dof_fragment(
    float4 position [[position]],
    texture2d<float, access::read> color [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    constant float4& lens [[buffer(0)]],
    constant uint& reversed [[buffer(1)]]
) {
    uint2 pixel = uint2(position.xy);
    float4 center = color.read(pixel);
    if (lens.w <= 0.5f) { return center; }

    float centerRatio = focusOverDistance(depth.read(pixel), lens.xy, reversed != 0);
    float centerRadius = min(lens.z * abs(1.0f - centerRatio), lens.w);
    if (centerRadius <= 0.5f) { return center; }

    // A fixed disk gather keeps the sample budget independent of blur radius.
    constexpr float2 disk[] = {
        float2( 0.35f,  0.35f), float2( 0.35f, -0.35f),
        float2(-0.35f,  0.35f), float2(-0.35f, -0.35f),
        float2( 1.00f,  0.00f), float2(-1.00f,  0.00f),
        float2( 0.00f,  1.00f), float2( 0.00f, -1.00f),
        float2( 0.71f,  0.71f), float2( 0.71f, -0.71f),
        float2(-0.71f,  0.71f), float2(-0.71f, -0.71f),
    };
    int2 size(color.get_width(), color.get_height());
    float4 sum = center;
    float total = 1.0f;
    for (uint i = 0; i < 12; ++i) {
        int2 offset = int2(round(disk[i] * centerRadius));
        int2 tap = int2(pixel) + offset;
        if (all(offset == 0) || any(tap < 0) || any(tap >= size)) { continue; }

        float tapRatio = focusOverDistance(depth.read(uint2(tap)), lens.xy, reversed != 0);
        if ((tapRatio > 1.0f) != (centerRatio > 1.0f)) { continue; }
        float depthWeight = saturate(1.0f - abs(tapRatio - centerRatio) / max(0.25f * centerRatio, 1e-6f));
        float tapRadius = min(lens.z * abs(1.0f - tapRatio), lens.w);
        float reach = saturate(tapRadius + 0.5f - length(float2(offset)));
        float weight = depthWeight * reach;
        if (weight <= 0) { continue; }
        sum += color.read(uint2(tap)) * weight;
        total += weight;
    }
    // Fade the first subpixel of blur so a moving focus plane has no abrupt onset.
    return mix(center, sum / total, smoothstep(0.5f, 1.5f, centerRadius));
}
