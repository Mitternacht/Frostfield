#include <metal_stdlib>
using namespace metal;

// Decode the base-level texel centers into a CPU-readable linear RGB snapshot.
kernel void frostfieldLightmapReadColor(
    texture2d<float, access::sample> source [[texture(0)]],
    device float4* pixels [[buffer(0)]],
    uint2 texel [[thread_position_in_grid]]
) {
    const uint2 size = uint2(source.get_width(), source.get_height());
    if (any(texel >= size)) return;
    constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float2 uv = (float2(texel) + 0.5f) / float2(size);
    pixels[texel.y * size.x + texel.x] = source.sample(nearest, uv, level(0.0f));
}

static inline bool2 frostfield_height_finite(float2 value) {
    return (as_type<uint2>(value) & uint2(0x7f800000u)) != uint2(0x7f800000u);
}

// Derives a tangent-space normal map from a wrapping height field with the same
// central differences and channel layout as ProceduralSurface (ShaderMaterialMaps.swift):
// slope = (height[left] - height[right], height[above] - height[below]) * slopeScale,
// normal = normalize(slope.x, slope.y, 1) * 0.5 + 0.5, rounded to 8 bits half away from zero.
// slopeScale is heightScale × width × 0.5, so texels are treated as square.
kernel void frostfield_height_to_normal(
    texture2d<float, access::read> heights [[texture(0)]],
    texture2d<float, access::write> normals [[texture(1)]],
    constant float& slopeScale [[buffer(0)]],
    uint2 texel [[thread_position_in_grid]]
) {
    uint width = heights.get_width();
    uint height = heights.get_height();
    if (texel.x >= width || texel.y >= height) {
        return;
    }
    uint left = (texel.x + width - 1u) % width;
    uint right = (texel.x + 1u) % width;
    uint above = (texel.y + height - 1u) % height;
    uint below = (texel.y + 1u) % height;
    float2 slope = float2(
        heights.read(uint2(left, texel.y)).r - heights.read(uint2(right, texel.y)).r,
        heights.read(uint2(texel.x, above)).r - heights.read(uint2(texel.x, below)).r
    ) * slopeScale;
    // Nonfinite differences become flat; bounded slopes keep the normalization finite.
    slope = clamp(select(float2(0), slope, frostfield_height_finite(slope)), float2(-1e6f), float2(1e6f));
    float3 encoded = normalize(float3(slope, 1.0f)) * 0.5f + 0.5f;
    normals.write(float4(round(saturate(encoded) * 255.0f) / 255.0f, 1.0f), texel);
}
