#include <metal_stdlib>
using namespace metal;

fragment float4 frostfield_post_process_mask(
    float4 position [[position]],
    texture2d<float, access::read> original [[texture(0)]],
    texture2d<float, access::read> processed [[texture(1)]],
    texture2d<float, access::read> mask [[texture(2)]],
    constant float4& options [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float4 value = mask.read(pixel);
    float channel = options.y == 0.0f ? value.r : value.a;
    bool selected = (channel >= options.x) != (options.z != 0.0f);
    return selected ? processed.read(pixel) : original.read(pixel);
}
