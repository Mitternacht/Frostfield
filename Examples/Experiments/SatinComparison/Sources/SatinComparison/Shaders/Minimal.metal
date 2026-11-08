// Minimal renderer shader. The host prepends StripesColor.h before compiling
// this source with MTLDevice.makeLibrary(source:options:).
#include <metal_stdlib>
using namespace metal;

struct MinimalVertex {
    float4 position;   // xyz used, w = 1
    float2 texcoord;
    float2 padding;
};

struct MinimalVaryings {
    float4 position [[position]];
    float2 texcoord;
};

vertex MinimalVaryings minimalVertex(
    uint vid [[vertex_id]],
    const device MinimalVertex *vertices [[buffer(0)]],
    constant float4x4 &modelViewProjection [[buffer(1)]]) {
    MinimalVaryings out;
    out.position = modelViewProjection * float4(vertices[vid].position.xyz, 1.0);
    out.texcoord = vertices[vid].texcoord;
    return out;
}

fragment float4 minimalFragment(MinimalVaryings in [[stage_in]]) {
    return stripesColor(in.texcoord);
}
