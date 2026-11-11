#include <metal_stdlib>
using namespace metal;

static bool frostfield_temporal_has_depth(float depth, bool reversed) {
    return reversed ? depth > 0.0f : depth < 1.0f;
}

kernel void frostfield_raster_temporal_prepare(
    depth2d<float, access::read> sourceDepth [[texture(0)]],
    texture2d<float, access::read> sourceMotion [[texture(1)]],
    texture2d<float, access::read> previousDepth [[texture(2)]],
    texture2d<float, access::write> depth [[texture(3)]],
    texture2d<float, access::write> motion [[texture(4)]],
    texture2d<float, access::write> reactive [[texture(5)]],
    constant float4& options [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    uint2 size = uint2(depth.get_width(), depth.get_height());
    if (any(pixel >= size)) { return; }

    float stored = sourceDepth.read(pixel);
    depth.write(float4(stored, 0, 0, 0), pixel);
    float4 vector = sourceMotion.read(pixel);
    uint3 exponent = as_type<uint3>(vector.xyz) & uint3(0x7f800000u);
    bool reversed = options.y != 0;
    bool valid = options.x != 0 && as_type<uint>(vector.w) == 0x3f800000u && all(exponent != 0x7f800000u)
        && vector.z >= 0.0f && vector.z <= 1.0f && frostfield_temporal_has_depth(stored, reversed);
    if (valid) {
        float2 previous = float2(pixel) + 0.5f + vector.xy;
        valid = all(previous >= 0.0f) && all(previous < float2(size));
        if (valid) {
            float old = previousDepth.read(uint2(previous)).r;
            valid = frostfield_temporal_has_depth(old, reversed) && abs(old - vector.z) <= options.z;
        }
    }
    motion.write(valid ? float4(vector.xy, 0, 0) : float4(0), pixel);
    reactive.write(float4(valid ? 0.0f : 1.0f), pixel);
}
