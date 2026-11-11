#include <metal_stdlib>
#include "FrostfieldMaterialExtensions.h"
#include "FrostfieldShadowSampling.h"
using namespace metal;

struct FrostfieldShadowVertex {
    float4 position [[position]];
    float4 uv [[user(frostfield_shadow_uv)]];
};

vertex FrostfieldShadowVertex frostfield_shadow_vertex(
    uint id [[vertex_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clipFromLocal [[buffer(1)]]
) {
    return {clipFromLocal * vertices[id * 4], vertices[id * 4 + 2]};
}

fragment void frostfield_shadow_mask(
    FrostfieldShadowVertex input [[stage_in]],
    texture2d<float> texture [[texture(0)]],
    sampler filtering [[sampler(0)]],
    constant float4& coverage [[buffer(0)]],
    constant float4* uvTransform [[buffer(1)]]
) {
    // The base color texture transform's two rows, as in PhysicallyBasedMaterial.TextureTransform.
    float2 source = coverage.z > 0 ? input.uv.zw : input.uv.xy;
    float2 uv = float2(dot(uvTransform[0].xy, source) + uvTransform[0].z, dot(uvTransform[1].xy, source) + uvTransform[1].z);
    if (saturate(texture.sample(filtering, uv).a) * coverage.x < coverage.y) {
        discard_fragment();
    }
}

fragment float4 frostfield_transmitted_shadow_fragment(
    FrostfieldShadowVertex input [[stage_in]],
    constant float3& transmittance [[buffer(0)]],
    constant uint& followsPeel [[buffer(1)]],
    depth2d<float, access::read> previousDepth [[texture(1)]]
) {
    if (frostfieldAlreadyPeeled(input.position, followsPeel, previousDepth)) { discard_fragment(); }
    return float4(transmittance, input.position.z);
}

struct FrostfieldTexturedTransmissionShadow {
    float4 tint;
    float4 optical;   // transmission factor, metallic factor, optical depth at full thickness
    float4 attenuation;
    float4 coverage;  // alpha mode, cutoff, has alpha map
    float4 transforms[10];  // color, metal/roughness, transmission, thickness, alpha
};

static inline float2 frostfieldTransmissionShadowUV(
    float4 uv, constant FrostfieldTexturedTransmissionShadow& material, uint slot
) {
    float4 row0 = material.transforms[2u * slot];
    return frostfieldTransformUV(row0.w != 0.0f ? uv.zw : uv.xy, row0, material.transforms[2u * slot + 1u]);
}

fragment float4 frostfield_textured_transmitted_shadow_fragment(
    FrostfieldShadowVertex input [[stage_in]],
    constant FrostfieldTexturedTransmissionShadow& material [[buffer(0)]],
    constant uint& followsPeel [[buffer(1)]],
    texture2d<float> color [[texture(0)]], sampler colorSampler [[sampler(0)]],
    depth2d<float, access::read> previousDepth [[texture(1)]],
    texture2d<float> metalRough [[texture(2)]], sampler metalRoughSampler [[sampler(1)]],
    texture2d<float> transmission [[texture(12)]], sampler transmissionSampler [[sampler(8)]],
    texture2d<float> thickness [[texture(14)]], sampler thicknessSampler [[sampler(9)]],
    texture2d<float> alpha [[texture(15)]]
) {
    // Keep implicit texture gradients available across the peel and alpha edges.
    float4 base = material.tint * saturate(color.sample(colorSampler, frostfieldTransmissionShadowUV(input.uv, material, 0)));
    if (material.coverage.z != 0.0f) {
        base.a *= saturate(alpha.sample(colorSampler, frostfieldTransmissionShadowUV(input.uv, material, 4)).g);
    }
    float metallic = material.optical.y
        * saturate(metalRough.sample(metalRoughSampler, frostfieldTransmissionShadowUV(input.uv, material, 1)).b);
    float factor = material.optical.x
        * saturate(transmission.sample(transmissionSampler, frostfieldTransmissionShadowUV(input.uv, material, 2)).r);
    float exponent = material.optical.z > 0.0f
        ? material.optical.z * saturate(thickness.sample(thicknessSampler, frostfieldTransmissionShadowUV(input.uv, material, 3)).g)
        : 0.0f;
    if (frostfieldAlreadyPeeled(input.position, followsPeel, previousDepth)
        || (material.coverage.x == 2.0f && base.a < material.coverage.y)) { discard_fragment(); }
    float coverage = material.coverage.x == 1.0f || material.coverage.x == 2.0f ? 1.0f : base.a;
    if (coverage == 0.0f) { discard_fragment(); }

    float3 filter = base.rgb * (factor * (1.0f - metallic));
    if (exponent > 0.0f) { filter *= pow(material.attenuation.rgb, float3(exponent)); }
    float3 transmitted = 1.0f - coverage * (1.0f - filter);
    // A clear texel should leave the peel available to a farther colored surface.
    if (all(transmitted == 1.0f)) { discard_fragment(); }
    return float4(transmitted, input.position.z);
}
