#include <metal_stdlib>
#include "FrostfieldScreenSpace.h"
using namespace metal;

struct FFPixelationParameters {
    float4 worldBias;
    float2 strengths;  // depth, normal
    uint pixelSize;
    uint reversed;
};

static uint2 ff_pixelation_center(int2 block, int2 lastPixel, int pixelSize) {
    int2 cell = clamp(block, int2(0), lastPixel / pixelSize);
    return uint2(min(cell * pixelSize + pixelSize / 2, lastPixel));
}

struct FFPixelationNeighborhood {
    uint2 center;
    uint2 neighbors[4];
    float depth;
    float4 neighborDepths;
    float4 deltas;
};

static FFPixelationNeighborhood ff_pixelation_neighbors(
    depth2d<float, access::read> depth, int2 pixel, int2 lastPixel, constant FFPixelationParameters& p
) {
    constexpr int2 directions[] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)};
    int pixelSize = int(p.pixelSize);
    int2 cell = pixel / pixelSize;
    FFPixelationNeighborhood neighborhood;
    neighborhood.center = ff_pixelation_center(cell, lastPixel, pixelSize);
    neighborhood.depth = depth.read(neighborhood.center);
    for (uint i = 0; i < 4; ++i) {
        neighborhood.neighbors[i] = ff_pixelation_center(cell + directions[i], lastPixel, pixelSize);
        neighborhood.neighborDepths[i] = depth.read(neighborhood.neighbors[i]);
    }
    float4 delta = neighborhood.neighborDepths - neighborhood.depth;
    neighborhood.deltas = p.reversed != 0 ? -delta : delta;
    return neighborhood;
}

static float ff_pixelation_depth_edge(float4 deltas) {
    float difference = dot(saturate(deltas), float4(1));
    return floor(smoothstep(0.01f, 0.02f, difference) * 2.0f) * 0.5f;
}

static bool ff_pixelation_has_surface(float depth, uint reversed) {
    return reversed != 0 ? depth > 0.0f : depth < 1.0f;
}

static bool ff_pixelation_visible_normal(float storedDepth, float normalDepth, uint reversed) {
    return ff_pixelation_has_surface(storedDepth, reversed) && abs(storedDepth - normalDepth) <= 1e-6f;
}

static float ff_pixelation_normal_edge(
    FFPixelationNeighborhood n, texture2d<uint, access::read> attributes,
    depth2d<float, access::read> normalDepth, constant FFPixelationParameters& p
) {
    if (!ff_pixelation_visible_normal(n.depth, normalDepth.read(n.center), p.reversed)) { return 0; }
    float3 normal = frostfieldUnpackSurfaceAttributes(attributes.read(n.center).xy).normal;
    float indicator = 0;
    for (uint i = 0; i < 4; ++i) {
        if (n.deltas[i] * 0.25f + 0.0025f <= 0) { continue; }
        uint2 neighbor = n.neighbors[i];
        float3 other = ff_pixelation_visible_normal(n.neighborDepths[i], normalDepth.read(neighbor), p.reversed)
            ? frostfieldUnpackSurfaceAttributes(attributes.read(neighbor).xy).normal : float3(0);
        float bias = smoothstep(-0.01f, 0.01f, dot(normal - other, p.worldBias.xyz));
        indicator += (1.0f - dot(normal, other)) * bias;
    }
    return step(0.1f, indicator);
}

static float4 ff_pixelation_gain(float4 color, float gain) {
    if (gain == 1.0f) { return color; }
    return float4(clamp(color.rgb * gain, float3(-65504.0f), float3(65504.0f)), color.a);
}

fragment float4 frostfield_pixelated_color(
    float4 position [[position]], texture2d<float, access::read> source [[texture(0)]],
    constant FFPixelationParameters& p [[buffer(0)]]
) {
    int2 last = int2(source.get_width(), source.get_height()) - 1;
    int size = int(p.pixelSize);
    uint2 center = ff_pixelation_center(int2(position.xy) / size, last, size);
    return source.read(center);
}

fragment float4 frostfield_pixelated_depth(
    float4 position [[position]], texture2d<float, access::read> source [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], constant FFPixelationParameters& p [[buffer(0)]]
) {
    int2 last = int2(source.get_width(), source.get_height()) - 1;
    FFPixelationNeighborhood n = ff_pixelation_neighbors(depth, int2(position.xy), last, p);
    return ff_pixelation_gain(source.read(n.center), 1.0f - p.strengths.x * ff_pixelation_depth_edge(n.deltas));
}

fragment float4 frostfield_pixelated_normals(
    float4 position [[position]], texture2d<float, access::read> source [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    depth2d<float, access::read> normalDepth [[texture(3)]], constant FFPixelationParameters& p [[buffer(0)]]
) {
    int2 last = int2(source.get_width(), source.get_height()) - 1;
    FFPixelationNeighborhood n = ff_pixelation_neighbors(depth, int2(position.xy), last, p);
    float depthEdge = p.strengths.x > 0 ? ff_pixelation_depth_edge(n.deltas) : 0;
    float gain = depthEdge > 0 ? 1.0f - p.strengths.x * depthEdge
        : 1.0f + p.strengths.y * ff_pixelation_normal_edge(n, attributes, normalDepth, p);
    return ff_pixelation_gain(source.read(n.center), gain);
}

fragment float4 frostfield_pixelated_scene_expand(
    float4 position [[position]], texture2d<float, access::read> source [[texture(0)]],
    constant uint4& viewport [[buffer(0)]]
) {
    uint2 local = uint2(position.xy) - viewport.xy;
    return source.read(local / viewport.z);
}
