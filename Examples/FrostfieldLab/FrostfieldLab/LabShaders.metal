#include <metal_stdlib>
#include "LabShaderTypes.h"

using namespace metal;

static_assert(sizeof(LabPulseParameters) == 48, "LabPulseParameters must be 48 bytes");

// The varyings Frostfield's mesh vertex stage writes, declared from
// docs/materials.md rather than from Frostfield's header. Metal matches these
// members to the vertex output by their [[user(...)]] names.
struct LabMeshVaryings {
    float4 position [[position]];
    float3 normal [[user(ff_model_normal)]];
    float3 modelPosition [[user(ff_model_position)]];
};

// A time-varying material: bands of warm color travel up the mesh as the app
// advances `time`. A fixed model-space light keeps the faces distinguishable.
fragment float4 lab_pulse_fragment(
    LabMeshVaryings in [[stage_in]],
    constant LabPulseParameters &parameters [[buffer(0)]])
{
    float wave = 0.5 + 0.5 * sin(parameters.time * parameters.speed - in.modelPosition.y * 6.0);
    float3 color = mix(parameters.coolColor.rgb, parameters.warmColor.rgb, wave);
    float light = 0.65 + 0.35 * max(dot(normalize(in.normal), normalize(float3(0.4, 0.8, 0.6))), 0.0);
    return float4(color * light, 1.0);
}
