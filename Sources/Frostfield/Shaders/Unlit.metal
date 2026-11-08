#include <metal_stdlib>
#include "../../FrostfieldShaderTypes/include/FrostfieldShaderTypes.h"

using namespace metal;

static_assert(sizeof(FFUnlitParameters) == 16, "FFUnlitParameters must be 16 bytes");
static_assert(alignof(FFUnlitParameters) == 16, "FFUnlitParameters must be 16-byte aligned");

// Built-in unlit material: every covered pixel receives the parameter color,
// unchanged. It reads no varyings, only its parameter block.
fragment float4 ff_unlit_fragment(
    constant FFUnlitParameters &parameters [[buffer(FF_MATERIAL_PARAMETERS_BUFFER_INDEX)]])
{
    return parameters.color;
}
