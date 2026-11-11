#include <metal_stdlib>
using namespace metal;

struct FrostfieldEnvironmentBackgroundVaryings {
    float4 position [[position]];
    float2 ndc;
};

vertex FrostfieldEnvironmentBackgroundVaryings frostfield_environment_background_vertex(
    uint id [[vertex_id]]
) {
    const float2 points[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(points[id], 0, 1), points[id]};
}

fragment float4 frostfield_environment_background_color(constant float4& color [[buffer(1)]]) {
    return color;
}

struct FrostfieldFlatBackgroundTransform {
    float4 first, second;
};

fragment float4 frostfield_environment_background_flat(
    FrostfieldEnvironmentBackgroundVaryings input [[stage_in]],
    texture2d<float> image [[texture(0)]], sampler filter [[sampler(0)]],
    constant FrostfieldFlatBackgroundTransform& transform [[buffer(0)]],
    constant float4& parameters [[buffer(1)]]
) {
    float3 uv = float3(input.ndc * float2(0.5f, -0.5f) + 0.5f, 1);
    float4 color = image.sample(filter, float2(dot(transform.first.xyz, uv), dot(transform.second.xyz, uv)));
    float alpha = saturate(color.a);
    float factor = parameters.x * (parameters.y == 0 ? alpha : 1.0f);
    return float4(clamp(color.rgb * factor, 0.0f, 65504.0f), alpha);
}

static float3 frostfieldEnvironmentDirection(float2 ndc, constant float3x3& rayFromNDC) {
    float3 ray = rayFromNDC * float3(ndc, 1);
    float magnitude = max(abs(ray.x), max(abs(ray.y), abs(ray.z)));
    return normalize(ray / magnitude);
}

fragment float4 frostfield_environment_background_cube(
    FrostfieldEnvironmentBackgroundVaryings input [[stage_in]],
    texturecube<float> sky [[texture(0)]],
    constant float3x3& rayFromNDC [[buffer(0)]],
    constant float4& parameters [[buffer(1)]]
) {
    constexpr sampler filter(coord::normalized, filter::linear, mip_filter::linear);
    float3 direction = frostfieldEnvironmentDirection(input.ndc, rayFromNDC);
    float3 radiance = sky.sample(filter, direction, level(parameters.y)).rgb * parameters.x;
    return float4(clamp(radiance, 0.0f, 65504.0f), 1);
}

fragment float4 frostfield_environment_background_equirectangular(
    FrostfieldEnvironmentBackgroundVaryings input [[stage_in]],
    texture2d<float> sky [[texture(0)]],
    constant float3x3& rayFromNDC [[buffer(0)]],
    constant float4& parameters [[buffer(1)]]
) {
    constexpr sampler filter(coord::normalized, s_address::repeat, t_address::clamp_to_edge,
                             filter::linear, mip_filter::linear);
    float3 direction = frostfieldEnvironmentDirection(input.ndc, rayFromNDC);
    // Same orientation as the environment-lighting panorama conversion.
    float2 uv = float2(atan2(direction.z, direction.x) / (2.0f * M_PI_F) + 0.5f,
                       acos(clamp(direction.y, -1.0f, 1.0f)) / M_PI_F);
    float3 radiance = sky.sample(filter, uv, level(parameters.y)).rgb * parameters.x;
    return float4(clamp(radiance, 0.0f, 65504.0f), 1);
}
