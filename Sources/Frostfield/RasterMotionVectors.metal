#include <metal_stdlib>
using namespace metal;

struct FrostfieldRasterMotionVaryings {
    float4 position [[position]];
    float4 previous [[user(frostfield_raster_motion_previous)]];
};

fragment float4 frostfield_raster_motion_fragment(
    FrostfieldRasterMotionVaryings input [[stage_in]],
    constant float4& previousViewport [[buffer(0)]],
    constant float2& previousDepthRange [[buffer(1)]]
) {
    if (!(input.previous.w > 0.0f)) return float4(0);
    float3 ndc = input.previous.xyz / input.previous.w;
    if (!all(abs(ndc.xy) <= 1.0f) || !(ndc.z >= 0.0f && ndc.z <= 1.0f)) return float4(0);
    float2 uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    float2 previousPixel = previousViewport.xy + uv * previousViewport.zw;
    return float4(previousPixel - input.position.xy, mix(previousDepthRange.x, previousDepthRange.y, ndc.z), 1.0f);
}
