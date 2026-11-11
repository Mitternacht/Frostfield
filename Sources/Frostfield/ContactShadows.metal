#include <metal_stdlib>
using namespace metal;

fragment float4 frostfield_contact_shadow_mask(
    float4 screen [[position]], depth2d<float, access::read> depth [[texture(0)]],
    constant float& darkness [[buffer(0)]]
) {
    float coverage = saturate((1.0f - depth.read(uint2(screen.xy))) * darkness);
    return float4(coverage);
}
