#include <metal_stdlib>
using namespace metal;

fragment float4 frostfield_decompress_texture_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    constant uint4 &options [[buffer(0)]]
) {
    constexpr sampler texel(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
    constexpr sampler scaled(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::nearest);
    const float2 uv = position.xy / float2(options.xy);
    const float mip = float(options.z);
    if (options.w != 0) {
        return source.sample(scaled, uv, level(mip));
    }
    return source.sample(texel, uv, level(mip));
}
