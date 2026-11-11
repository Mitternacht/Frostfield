#include <metal_stdlib>
using namespace metal;

struct TextureDebugVaryings {
    float4 position [[position]];
    float2 uv;
};

// Matches the UInt32 vector in TextureDebugPass.encode.
struct TextureDebugOptions {
    uint slice;
    uint mip;
    uint channel;
    uint reversedDepth;
};

vertex TextureDebugVaryings frostfield_texture_debug_vertex(uint id [[vertex_id]]) {
    float2 uv = float2(id == 1 ? 2.0f : 0.0f, id == 2 ? 2.0f : 0.0f);
    return {float4(uv * float2(2, -2) + float2(-1, 1), 0, 1), uv};
}

static uint2 debugTexel(float2 uv, uint width, uint height) {
    uint2 size(width, height);
    return min(uint2(saturate(uv) * float2(size)), size - 1);
}

static float4 debugColor(float4 value, uint channel) {
    return float4(channel == 0 ? value.rgb : float3(value[channel - 1]), 1);
}

static float4 debugDepth(float value, uint reversed) {
    return float4(float3(reversed != 0 ? value : 1 - value), 1);
}

static float4 debugRead(texture2d<float, access::read> source, float2 uv, uint mip) {
    return source.read(debugTexel(uv, source.get_width(mip), source.get_height(mip)), mip);
}

static float4 debugRead(texture2d_array<float, access::read> source, float2 uv, uint mip, uint slice) {
    return source.read(debugTexel(uv, source.get_width(mip), source.get_height(mip)), slice, mip);
}

static float4 debugRead(texture3d<float, access::read> source, float2 uv, uint mip, uint slice) {
    uint2 xy = debugTexel(uv, source.get_width(mip), source.get_height(mip));
    return source.read(uint3(xy, slice), mip);
}

static float4 debugRead(texturecube<float, access::read> source, float2 uv, uint mip, uint face) {
    uint size = source.get_width(mip);
    return source.read(debugTexel(uv, size, size), face, mip);
}

fragment float4 frostfield_texture_debug_color_2d(
    TextureDebugVaryings in [[stage_in]], texture2d<float, access::read> source [[texture(0)]],
    constant TextureDebugOptions& p [[buffer(0)]]
) {
    return debugColor(debugRead(source, in.uv, p.mip), p.channel);
}

fragment float4 frostfield_texture_debug_color_array(
    TextureDebugVaryings in [[stage_in]], texture2d_array<float, access::read> source [[texture(0)]],
    constant TextureDebugOptions& p [[buffer(0)]]
) {
    return debugColor(debugRead(source, in.uv, p.mip, p.slice), p.channel);
}

fragment float4 frostfield_texture_debug_color_3d(
    TextureDebugVaryings in [[stage_in]], texture3d<float, access::read> source [[texture(0)]],
    constant TextureDebugOptions& p [[buffer(0)]]
) {
    return debugColor(debugRead(source, in.uv, p.mip, p.slice), p.channel);
}

fragment float4 frostfield_texture_debug_color_cube(
    TextureDebugVaryings in [[stage_in]], texturecube<float, access::read> source [[texture(0)]],
    constant TextureDebugOptions& p [[buffer(0)]]
) {
    return debugColor(debugRead(source, in.uv, p.mip, p.slice), p.channel);
}

fragment float4 frostfield_texture_debug_depth_2d(
    TextureDebugVaryings in [[stage_in]], depth2d<float, access::read> source [[texture(0)]],
    constant TextureDebugOptions& p [[buffer(0)]]
) {
    uint2 xy = debugTexel(in.uv, source.get_width(p.mip), source.get_height(p.mip));
    return debugDepth(source.read(xy, p.mip), p.reversedDepth);
}

fragment float4 frostfield_texture_debug_depth_array(
    TextureDebugVaryings in [[stage_in]], depth2d_array<float, access::read> source [[texture(0)]],
    constant TextureDebugOptions& p [[buffer(0)]]
) {
    uint2 xy = debugTexel(in.uv, source.get_width(p.mip), source.get_height(p.mip));
    return debugDepth(source.read(xy, p.slice, p.mip), p.reversedDepth);
}

fragment float4 frostfield_texture_debug_label(
    TextureDebugVaryings in [[stage_in]], texture2d<float, access::read> mask [[texture(0)]],
    constant float4& color [[buffer(0)]]
) {
    float alpha = debugRead(mask, in.uv, 0).r * color.a;
    return float4(color.rgb * alpha, alpha);
}

// The existing mesh vertex stages provide top-left UVs and, on a cube helper,
// the native face index in the secondary UV.
struct TextureHelperVaryings {
    float4 position [[position]];
    float2 uv [[user(frostfield_uv)]];
    float2 uv1 [[user(frostfield_uv1)]];
};

struct TextureHelperOptions {
    uint mip;
    uint sliceCount;
    float opacity;
    uint reserved;
};

static float4 helperColor(float4 sample, float alpha) {
    return float4(sample.rgb * alpha, alpha);
}

static uint helperSlice(float index, uint count) {
    return uint(clamp(round(index), 0.0f, float(count - 1)));
}

fragment float4 frostfield_texture_helper_color_2d(
    TextureHelperVaryings in [[stage_in]], texture2d<float, access::read> source [[texture(0)]],
    constant TextureHelperOptions& options [[buffer(1)]]
) {
    return helperColor(debugRead(source, in.uv, options.mip), options.opacity);
}

fragment float4 frostfield_texture_helper_color_array(
    TextureHelperVaryings in [[stage_in]], texture2d_array<float, access::read> source [[texture(0)]],
    constant float4& parameters [[buffer(0)]], constant TextureHelperOptions& options [[buffer(1)]]
) {
    uint slice = helperSlice(parameters.x, options.sliceCount);
    return helperColor(debugRead(source, in.uv, options.mip, slice), options.opacity);
}

fragment float4 frostfield_texture_helper_color_3d(
    TextureHelperVaryings in [[stage_in]], texture3d<float, access::read> source [[texture(0)]],
    constant float4& parameters [[buffer(0)]], constant TextureHelperOptions& options [[buffer(1)]]
) {
    uint slice = helperSlice(parameters.x, options.sliceCount);
    return helperColor(debugRead(source, in.uv, options.mip, slice), options.opacity);
}

fragment float4 frostfield_texture_helper_color_cube(
    TextureHelperVaryings in [[stage_in]], texturecube<float, access::read> source [[texture(0)]],
    constant TextureHelperOptions& options [[buffer(1)]]
) {
    uint face = helperSlice(in.uv1.x, options.sliceCount);
    return helperColor(debugRead(source, in.uv, options.mip, face), options.opacity);
}

// AxisViewHelper uploads three float4 values for each stroke or endpoint.
struct AxisViewMark {
    float4 points;
    float4 color;
    float4 dimensions;
};

struct AxisViewVaryings {
    float4 position [[position]];
    float2 pixel;
    float4 points [[flat]];
    float4 color [[flat]];
    float radius [[flat]];
    float2 glyphSize [[flat]];
};

vertex AxisViewVaryings frostfield_axis_view_vertex(
    uint vertexID [[vertex_id]], uint markID [[instance_id]], constant AxisViewMark* marks [[buffer(0)]]
) {
    AxisViewMark mark = marks[markID];
    float padding = mark.dimensions.x + 0.5f;
    float2 lower = min(mark.points.xy, mark.points.zw) - padding;
    float2 upper = max(mark.points.xy, mark.points.zw) + padding;
    float2 corner = float2(vertexID & 1, vertexID >> 1);
    float2 pixel = mix(lower, upper, corner);
    float2 clip = pixel * float2(2, -2) / mark.dimensions.y + float2(-1, 1);
    return {float4(clip, 0, 1), pixel, mark.points, mark.color, mark.dimensions.x, mark.dimensions.zw};
}

static float axisViewAlpha(AxisViewVaryings in) {
    float2 offset = in.pixel - in.points.xy;
    float2 segment = in.points.zw - in.points.xy;
    float lengthSquared = dot(segment, segment);
    float factor = lengthSquared > 0 ? saturate(dot(offset, segment) / lengthSquared) : 0;
    float distanceToMark = length(offset - factor * segment);
    return in.color.a * saturate(in.radius + 0.5f - distanceToMark);
}

fragment float4 frostfield_axis_view_fragment(AxisViewVaryings in [[stage_in]]) {
    float alpha = axisViewAlpha(in);
    return float4(in.color.rgb * alpha, alpha);
}

fragment float4 frostfield_axis_view_labeled(
    AxisViewVaryings in [[stage_in]], texture2d<float> glyphs [[texture(0)]],
    constant float4& ink [[buffer(0)]]
) {
    constexpr sampler glyphSampler(coord::normalized, address::clamp_to_zero, filter::linear);
    float coverage = 0;
    if (all(in.glyphSize > 0)) {
        float2 uv = (in.pixel - in.points.xy) / in.glyphSize + 0.5f;
        coverage = glyphs.sample(glyphSampler, uv).r;
    }
    float3 color = mix(in.color.rgb, ink.rgb, coverage * ink.a);
    float alpha = axisViewAlpha(in);
    return float4(color * alpha, alpha);
}
