// Satin custom material shader (SourceMaterial, label "Stripes"). Satin
// injects `Vertex`, `VertexUniforms`, the buffer-index enums, and
// SATIN_SKIN_POSITION before compiling this source at run time, and derives
// the function names from the material label.
#include "../StripesColor.h"

typedef struct {
    float4 position [[position]];
    float2 texcoord;
} StripesVertexData;

vertex StripesVertexData stripesVertex(
    Vertex in [[stage_in]],
    ushort amp_id [[amplification_id]],
    // inject instancing args
    constant VertexUniforms *vertexUniforms [[buffer(VertexBufferVertexUniforms)]]) {
    StripesVertexData out;
    out.position = vertexUniforms[amp_id].modelViewProjectionMatrix * SATIN_SKIN_POSITION(in);
    out.texcoord = in.texcoord;
    return out;
}

fragment float4 stripesFragment(StripesVertexData in [[stage_in]]) {
    return stripesColor(in.texcoord);
}
