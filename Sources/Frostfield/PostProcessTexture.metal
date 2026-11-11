#include <metal_stdlib>
using namespace metal;

constexpr sampler ff_texture_pass_linear(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler ff_texture_pass_nearest(coord::normalized, address::clamp_to_edge, filter::nearest);

static inline float4 ff_texture_pass_premultiplied_texel(texture2d<float> texture, float2 pixel, float2 size) {
    float4 value = texture.sample(ff_texture_pass_nearest, (pixel + 0.5f) / size, level(0));
    return float4(value.rgb * value.a, value.a);
}

static inline float4 ff_texture_pass_filter_straight(texture2d<float> texture, float2 uv) {
    float2 size = float2(texture.get_width(), texture.get_height());
    float2 pixel = uv * size - 0.5f;
    float2 base = floor(pixel);
    float2 fraction = pixel - base;
    float4 topLeft = ff_texture_pass_premultiplied_texel(texture, base, size);
    float4 topRight = ff_texture_pass_premultiplied_texel(texture, base + float2(1.0f, 0.0f), size);
    float4 bottomLeft = ff_texture_pass_premultiplied_texel(texture, base + float2(0.0f, 1.0f), size);
    float4 bottomRight = ff_texture_pass_premultiplied_texel(texture, base + 1.0f, size);
    return mix(mix(topLeft, topRight, fraction.x), mix(bottomLeft, bottomRight, fraction.x), fraction.y);
}

fragment float4 frostfield_post_process_texture(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float> texture [[texture(1)]],
    constant float4& parameters [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float opacity = parameters.z;
    if (opacity == 0.0f) return source.read(pixel);

    float2 uv = position.xy / parameters.xy;
    float4 overlay = parameters.w == 0.0f
        ? texture.sample(ff_texture_pass_linear, uv, level(0))
        : ff_texture_pass_filter_straight(texture, uv);
    if (opacity == 1.0f) return overlay;

    overlay *= opacity;
    return overlay + source.read(pixel) * (1.0f - overlay.a);
}
