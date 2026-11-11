#include <metal_stdlib>
using namespace metal;

kernel void frostfield_variance_shadow_vertical(
    depth2d_array<float, access::read> depth [[texture(0)]],
    texture2d_array<float, access::write> scratch [[texture(1)]],
    constant uint2* slices [[buffer(0)]],
    uint3 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= depth.get_width() || pixel.y >= depth.get_height()) {
        return;
    }
    uint slice = slices[pixel.z].x;
    uint radius = slices[pixel.z].y;
    float2 moments = 0.0f;
    for (int offset = -int(radius); offset <= int(radius); ++offset) {
        uint row = uint(clamp(int(pixel.y) + offset, 0, int(depth.get_height()) - 1));
        float value = depth.read(uint2(pixel.x, row), slice);
        moments += float2(value, value * value);
    }
    scratch.write(float4(moments / float(2u * radius + 1u), 0, 0), pixel.xy, slice);
}

kernel void frostfield_variance_shadow_horizontal(
    texture2d_array<float, access::read> scratch [[texture(0)]],
    texture2d_array<float, access::write> filtered [[texture(1)]],
    constant uint2* slices [[buffer(0)]],
    uint3 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= scratch.get_width() || pixel.y >= scratch.get_height()) {
        return;
    }
    uint slice = slices[pixel.z].x;
    uint radius = slices[pixel.z].y;
    float2 moments = 0.0f;
    for (int offset = -int(radius); offset <= int(radius); ++offset) {
        uint column = uint(clamp(int(pixel.x) + offset, 0, int(scratch.get_width()) - 1));
        moments += scratch.read(uint2(column, pixel.y), slice).rg;
    }
    filtered.write(float4(moments / float(2u * radius + 1u), 0, 0), pixel.xy, slice);
}
