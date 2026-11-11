#include <metal_stdlib>
#include "FrostfieldLogDepth.h"
#include "FrostfieldShadowSampling.h"
#include "FrostfieldLightProbe.h"
using namespace metal;

struct FrostfieldMeshVaryings {
    float4 position [[position]];
    float3 normal [[user(frostfield_normal)]];
    float2 uv [[user(frostfield_uv)]];
    float3 worldPosition [[user(frostfield_world_position)]];
    float2 uv1 [[user(frostfield_uv1)]];
    float4 tangent [[user(frostfield_tangent)]];
    // Linear RGBA from the instance record, or one for draws without instances.
    float4 instanceTint [[user(frostfield_instance_tint), flat]];
    // Linear RGBA vertex color from vertex buffer 10, or one for meshes without colors.
    float4 color [[user(frostfield_vertex_color)]];
    float3 normalX [[user(frostfield_object_normal_x), flat]];
    float3 normalY [[user(frostfield_object_normal_y), flat]];
    float3 normalZ [[user(frostfield_object_normal_z), flat]];
    // The vertex in model coordinates after local deformation, for precise shadows.
    float3 shadowLocalPosition [[user(frostfield_shadow_local_position)]];
};

static inline float4 frostfieldMeshTangent(float4 tangent, float4x4 transform) {
    if (tangent.w == 0) {
        return float4(0);
    }
    float determinant = dot(transform[0].xyz, cross(transform[1].xyz, transform[2].xyz));
    float3 direction = (transform * float4(tangent.xyz, 0)).xyz;
    float scale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    if (!isfinite(determinant) || determinant == 0 || !isfinite(scale) || scale == 0) {
        return float4(0);
    }
    return float4(normalize(direction / scale), determinant < 0 ? -tangent.w : tangent.w);
}

// Use the same explicit fused operations in color and motion. A generic
// matrix multiply can contract differently when a stage also exports prior
// clip positions, which would fail the motion pass's exact depth test.
static inline float4 frostfieldRasterClipPosition(constant float4x4& clip, float3 position) {
    return fma(clip[2], float4(position.z),
               fma(clip[1], float4(position.y), clip[0] * position.x)) + clip[3];
}

vertex FrostfieldMeshVaryings frostfield_mesh_vertex(
    uint id [[vertex_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clip [[buffer(1)]],
    constant float4x4& normal [[buffer(2)]],
    constant float4x4& world [[buffer(3)]]
) {
    uint base = id * 4;
    return {
        frostfieldRasterClipPosition(clip, vertices[base].xyz),
        (normal * vertices[base + 1]).xyz,
        vertices[base + 2].xy,
        (world * vertices[base]).xyz,
        vertices[base + 2].zw,
        frostfieldMeshTangent(vertices[base + 3], world),
        float4(1),
        float4(1),
        normal[0].xyz, normal[1].xyz, normal[2].xyz,
        vertices[base].xyz
    };
}

struct FrostfieldSkinInfluence {
    uint4 joints;
    float4 weights;
};

static inline float4x4 frostfieldSkinMatrix(
    uint id,
    const device FrostfieldSkinInfluence* influences,
    constant float4x4* jointMatrices
) {
    FrostfieldSkinInfluence influence = influences[id];
    return jointMatrices[influence.joints.x] * influence.weights.x
        + jointMatrices[influence.joints.y] * influence.weights.y
        + jointMatrices[influence.joints.z] * influence.weights.z
        + jointMatrices[influence.joints.w] * influence.weights.w;
}

// The sign-adjusted cofactor matrix is a positive multiple of the inverse transpose.
static inline float3x3 frostfieldNormalCofactors(float4x4 transform) {
    float3 x = transform[0].xyz;
    float3 y = transform[1].xyz;
    float3 z = transform[2].xyz;
    float3 cofactorX = cross(y, z);
    float determinant = dot(x, cofactorX);
    if (!isfinite(determinant) || determinant == 0) { return float3x3(0.0f); }
    return float3x3(cofactorX, cross(z, x), cross(x, y)) * (determinant < 0 ? -1.0f : 1.0f);
}

static inline float3 frostfieldSkinNormal(float3 normal, float4x4 skin) {
    float3 transformed = frostfieldNormalCofactors(skin) * normal;
    float scale = max(abs(transformed.x), max(abs(transformed.y), abs(transformed.z)));
    if (!isfinite(scale) || scale == 0) {
        return normal;
    }
    return normalize(transformed / scale);
}

vertex FrostfieldMeshVaryings frostfield_skinned_mesh_vertex(
    uint id [[vertex_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clip [[buffer(1)]],
    constant float4x4& normal [[buffer(2)]],
    constant float4x4& world [[buffer(3)]],
    const device FrostfieldSkinInfluence* influences [[buffer(4)]],
    constant float4x4* jointMatrices [[buffer(5)]]
) {
    uint base = id * 4;
    float4x4 skin = frostfieldSkinMatrix(id, influences, jointMatrices);
    float4 position = float4((skin * vertices[base]).xyz, 1);
    float4 skinnedNormal = float4(frostfieldSkinNormal(vertices[base + 1].xyz, skin), 0);
    return {
        frostfieldRasterClipPosition(clip, position.xyz),
        (normal * skinnedNormal).xyz,
        vertices[base + 2].xy,
        (world * position).xyz,
        vertices[base + 2].zw,
        frostfieldMeshTangent(frostfieldMeshTangent(vertices[base + 3], skin), world),
        float4(1),
        float4(1),
        normal[0].xyz, normal[1].xyz, normal[2].xyz,
        position.xyz
    };
}

struct FrostfieldSkinnedShadowVertex {
    float4 position [[position]];
    float4 uv [[user(frostfield_shadow_uv)]];
};

vertex FrostfieldSkinnedShadowVertex frostfield_skinned_shadow_vertex(
    uint id [[vertex_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clipFromLocal [[buffer(1)]],
    const device FrostfieldSkinInfluence* influences [[buffer(4)]],
    constant float4x4* jointMatrices [[buffer(5)]]
) {
    float4x4 skin = frostfieldSkinMatrix(id, influences, jointMatrices);
    uint base = id * 4;
    return {clipFromLocal * float4((skin * vertices[base]).xyz, 1), vertices[base + 2]};
}

// Specialized vertex stages for morph targets and instancing. Raster vertex buffers
// 6 to 9 belong to these features: 6 instance records, 7 morph offsets, 8 the
// draw's nonzero morph weights, and 9 the morph layout. Keep the records in
// lockstep with InstanceSet.PackedInstance and MeshRenderer's morph packing.
constant bool frostfield_vertex_skinned [[function_constant(30)]];
constant bool frostfield_vertex_morphed [[function_constant(31)]];
constant bool frostfield_vertex_instanced [[function_constant(32)]];
// Per-vertex linear RGBA colors at vertex buffer 10, one float4 per vertex.
constant bool frostfield_vertex_colored [[function_constant(33)]];
constant bool frostfield_vertex_object_normals [[function_constant(34)]];

// An affine mesh-to-node transform as three rows, then a linear RGBA tint. 64 bytes.
struct FrostfieldInstance {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 tint;
};

struct FrostfieldMorphWeight {
    uint target;
    float weight;
};

// Element offsets of each attribute's section in the packed_float3 offsets, or
// 0xFFFFFFFF when no target has that attribute. A target's section holds one
// offset per vertex. 20 bytes.
struct FrostfieldMorphLayout {
    uint activeCount;
    uint vertexCount;
    uint positionBase;
    uint normalBase;
    uint tangentBase;
};

struct FrostfieldLocalVertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float4 tint;
};

// Morphs in the bind pose, then skins, then applies the instance transform.
static inline FrostfieldLocalVertex frostfieldSpecializedVertex(
    uint id, uint instance,
    const device float4* vertices,
    const device FrostfieldSkinInfluence* influences,
    constant float4x4* jointMatrices,
    const device FrostfieldInstance* instances,
    const device packed_float3* morphOffsets,
    constant FrostfieldMorphWeight* morphWeights,
    constant FrostfieldMorphLayout& morphLayout
) {
    uint base = id * 4;
    FrostfieldLocalVertex local = {vertices[base].xyz, vertices[base + 1].xyz, vertices[base + 3], float4(1)};
    if (frostfield_vertex_morphed) {
        const uint absent = 0xFFFFFFFFu;
        for (uint index = 0; index < morphLayout.activeCount; ++index) {
            FrostfieldMorphWeight entry = morphWeights[index];
            uint element = entry.target * morphLayout.vertexCount + id;
            if (morphLayout.positionBase != absent) {
                local.position += entry.weight * float3(morphOffsets[morphLayout.positionBase + element]);
            }
            if (morphLayout.normalBase != absent) {
                local.normal += entry.weight * float3(morphOffsets[morphLayout.normalBase + element]);
            }
            if (morphLayout.tangentBase != absent) {
                local.tangent.xyz += entry.weight * float3(morphOffsets[morphLayout.tangentBase + element]);
            }
        }
    }
    if (frostfield_vertex_skinned) {
        float4x4 skin = frostfieldSkinMatrix(id, influences, jointMatrices);
        local.position = (skin * float4(local.position, 1)).xyz;
        local.normal = frostfieldSkinNormal(local.normal, skin);
        local.tangent = frostfieldMeshTangent(local.tangent, skin);
    }
    if (frostfield_vertex_instanced) {
        FrostfieldInstance record = instances[instance];
        float4x4 transform = transpose(float4x4(record.row0, record.row1, record.row2, float4(0, 0, 0, 1)));
        local.position = (transform * float4(local.position, 1)).xyz;
        local.normal = frostfieldSkinNormal(local.normal, transform);
        local.tangent = frostfieldMeshTangent(local.tangent, transform);
        local.tint = record.tint;
    }
    return local;
}

struct FrostfieldRasterMotionVaryings {
    float4 position [[position]];
    float4 previous [[user(frostfield_raster_motion_previous)]];
};

// Replay both poses using the same vertex stage as the mesh's color draw.
// Immutable influences and morph offsets are shared by the two poses.
vertex FrostfieldRasterMotionVaryings frostfield_raster_motion_vertex(
    uint id [[vertex_id]], uint instance [[instance_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4* clip [[buffer(1)]],
    const device FrostfieldSkinInfluence* influences [[buffer(4), function_constant(frostfield_vertex_skinned)]],
    constant float4x4* joints [[buffer(5), function_constant(frostfield_vertex_skinned)]],
    const device FrostfieldInstance* instances [[buffer(6), function_constant(frostfield_vertex_instanced)]],
    const device packed_float3* offsets [[buffer(7), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphWeight* weights [[buffer(8), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphLayout& layout [[buffer(9), function_constant(frostfield_vertex_morphed)]],
    constant float4x4* priorJoints [[buffer(11), function_constant(frostfield_vertex_skinned)]],
    const device FrostfieldInstance* priorInstances [[buffer(12), function_constant(frostfield_vertex_instanced)]],
    constant FrostfieldMorphWeight* priorWeights [[buffer(13), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphLayout& priorLayout [[buffer(14), function_constant(frostfield_vertex_morphed)]],
    const device float4* priorVertices [[buffer(15)]]
) {
    const float3 current = frostfieldSpecializedVertex(
        id, instance, vertices, influences, joints, instances, offsets, weights, layout
    ).position;
    const float3 prior = frostfieldSpecializedVertex(
        id, instance, priorVertices, influences, priorJoints, priorInstances, offsets, priorWeights, priorLayout
    ).position;
    return {frostfieldRasterClipPosition(clip[0], current), frostfieldRasterClipPosition(clip[1], prior)};
}

vertex float4 frostfield_toon_outline_vertex(
    uint id [[vertex_id]],
    uint instance [[instance_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clip [[buffer(1)]],
    const device FrostfieldSkinInfluence* influences [[buffer(4), function_constant(frostfield_vertex_skinned)]],
    constant float4x4* jointMatrices [[buffer(5), function_constant(frostfield_vertex_skinned)]],
    const device FrostfieldInstance* instances [[buffer(6), function_constant(frostfield_vertex_instanced)]],
    const device packed_float3* morphOffsets [[buffer(7), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphWeight* morphWeights [[buffer(8), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphLayout& morphLayout [[buffer(9), function_constant(frostfield_vertex_morphed)]],
    constant float& thickness [[buffer(11)]]
) {
    FrostfieldLocalVertex local = frostfieldSpecializedVertex(
        id, instance, vertices, influences, jointMatrices, instances, morphOffsets, morphWeights, morphLayout
    );
    float4 position = clip * float4(local.position, 1);
    // The difference between the projected position and position minus normal.
    float4 direction = clip * float4(local.normal, 0);
    float lengthSquared = dot(direction, direction);
    if (!isfinite(lengthSquared) || lengthSquared <= FLT_MIN) return position;
    return position + direction * (rsqrt(lengthSquared) * thickness * position.w);
}

fragment float4 frostfield_toon_outline_fragment(constant float4& color [[buffer(0)]]) {
    return color;
}

vertex FrostfieldMeshVaryings frostfield_mesh_vertex_specialized(
    uint id [[vertex_id]],
    uint instance [[instance_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clip [[buffer(1)]],
    constant float4x4& normal [[buffer(2)]],
    constant float4x4& world [[buffer(3)]],
    const device FrostfieldSkinInfluence* influences [[buffer(4), function_constant(frostfield_vertex_skinned)]],
    constant float4x4* jointMatrices [[buffer(5), function_constant(frostfield_vertex_skinned)]],
    const device FrostfieldInstance* instances [[buffer(6), function_constant(frostfield_vertex_instanced)]],
    const device packed_float3* morphOffsets [[buffer(7), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphWeight* morphWeights [[buffer(8), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphLayout& morphLayout [[buffer(9), function_constant(frostfield_vertex_morphed)]],
    const device float4* colors [[buffer(10), function_constant(frostfield_vertex_colored)]]
) {
    FrostfieldLocalVertex local = frostfieldSpecializedVertex(
        id, instance, vertices, influences, jointMatrices, instances, morphOffsets, morphWeights, morphLayout
    );
    float4 position = float4(local.position, 1);
    uint base = id * 4;
    float4 color = float4(1);
    if (frostfield_vertex_colored) {
        color = colors[id];
    }
    float3x3 objectNormal = float3x3(normal[0].xyz, normal[1].xyz, normal[2].xyz);
    if (frostfield_vertex_instanced && frostfield_vertex_object_normals) {
        FrostfieldInstance record = instances[instance];
        float4x4 transform = transpose(float4x4(record.row0, record.row1, record.row2, float4(0, 0, 0, 1)));
        objectNormal = objectNormal * frostfieldNormalCofactors(transform);
    }
    return {
        frostfieldRasterClipPosition(clip, position.xyz),
        (normal * float4(local.normal, 0)).xyz,
        vertices[base + 2].xy,
        (world * position).xyz,
        vertices[base + 2].zw,
        frostfieldMeshTangent(local.tangent, world),
        local.tint,
        color,
        objectNormal[0], objectNormal[1], objectNormal[2],
        local.position
    };
}

struct FrostfieldSpecializedShadowVertex {
    float4 position [[position]];
    float4 uv [[user(frostfield_shadow_uv)]];
    // The instance tint's alpha, which scales masked coverage like the color pass.
    float coverage [[user(frostfield_shadow_coverage), flat]];
};

vertex FrostfieldSpecializedShadowVertex frostfield_shadow_vertex_specialized(
    uint id [[vertex_id]],
    uint instance [[instance_id]],
    const device float4* vertices [[buffer(0)]],
    constant float4x4& clipFromLocal [[buffer(1)]],
    const device FrostfieldSkinInfluence* influences [[buffer(4), function_constant(frostfield_vertex_skinned)]],
    constant float4x4* jointMatrices [[buffer(5), function_constant(frostfield_vertex_skinned)]],
    const device FrostfieldInstance* instances [[buffer(6), function_constant(frostfield_vertex_instanced)]],
    const device packed_float3* morphOffsets [[buffer(7), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphWeight* morphWeights [[buffer(8), function_constant(frostfield_vertex_morphed)]],
    constant FrostfieldMorphLayout& morphLayout [[buffer(9), function_constant(frostfield_vertex_morphed)]]
) {
    FrostfieldLocalVertex local = frostfieldSpecializedVertex(
        id, instance, vertices, influences, jointMatrices, instances, morphOffsets, morphWeights, morphLayout
    );
    return {clipFromLocal * float4(local.position, 1), vertices[id * 4 + 2], local.tint.a};
}

fragment void frostfield_shadow_mask_specialized(
    FrostfieldSpecializedShadowVertex input [[stage_in]],
    texture2d<float> texture [[texture(0)]],
    sampler filtering [[sampler(0)]],
    constant float4& coverage [[buffer(0)]]
) {
    float2 uv = coverage.z > 0 ? input.uv.zw : input.uv.xy;
    if (saturate(texture.sample(filtering, uv).a) * saturate(coverage.x * input.coverage) < coverage.y) {
        discard_fragment();
    }
}

static float4 frostfieldUnlit(
    FrostfieldMeshVaryings input,
    constant float4& parameters
) {
    float4 tint = parameters * input.instanceTint;
    float3 color = (0.22 + 0.78 * abs(normalize(input.normal))) * tint.rgb;
    float alpha = saturate(tint.a);
    return float4(color * alpha, alpha);
}

fragment float4 frostfield_mesh_unlit(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]]
) {
    return frostfieldUnlit(input, parameters);
}

fragment FrostfieldLogColor frostfield_mesh_unlit_log(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]]
    FROSTFIELD_LOG_DEPTH_INPUT
) {
    return {frostfieldUnlit(input, parameters), frostfieldLogDepth(input.position, logDepthRange)};
}

static float4 frostfieldTexturedColor(float4 sampled, float4 tint) {
    float alpha = saturate(sampled.a * tint.a);
    return float4(sampled.rgb * tint.rgb * alpha, alpha);
}

static float4 frostfieldTextured(
    FrostfieldMeshVaryings input,
    constant float4& parameters,
    texture2d<float> colorTexture
) {
    constexpr sampler colorSampler(coord::normalized, address::repeat, filter::linear, mip_filter::linear);
    return frostfieldTexturedColor(colorTexture.sample(colorSampler, input.uv), parameters * input.instanceTint);
}

fragment float4 frostfield_mesh_textured(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]],
    texture2d<float> colorTexture [[texture(0)]]
) {
    return frostfieldTextured(input, parameters, colorTexture);
}

fragment FrostfieldLogColor frostfield_mesh_textured_log(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]],
    texture2d<float> colorTexture [[texture(0)]]
    FROSTFIELD_LOG_DEPTH_INPUT
) {
    return {frostfieldTextured(input, parameters, colorTexture), frostfieldLogDepth(input.position, logDepthRange)};
}

struct FrostfieldIndirectTextureArguments {
    texture2d<float> colorTexture [[id(0)]];
    sampler colorSampler [[id(1)]];
};

fragment float4 frostfield_mesh_textured_indirect(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]],
    constant FrostfieldIndirectTextureArguments& arguments [[buffer(1)]]
) {
    return frostfieldTexturedColor(
        arguments.colorTexture.sample(arguments.colorSampler, input.uv), parameters * input.instanceTint
    );
}

static float4 frostfieldLambert(
    FrostfieldMeshVaryings input,
    constant float4& parameters,
    constant FrostfieldLightProbe& probe
) {
    float maxComponent = max(max(abs(parameters.x), abs(parameters.y)), abs(parameters.z));
    float3 direction = maxComponent > 0 ? normalize(parameters.xyz / maxComponent) : float3(0);
    float diffuse = saturate(dot(normalize(input.normal), direction)) * max(parameters.w, 0.0);
    return float4(float3(diffuse) + frostfieldAmbientIrradiance(probe) * M_1_PI_F, 1);
}

fragment float4 frostfield_mesh_lambert(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]]
) {
    return frostfieldLambert(input, parameters, probe);
}

fragment FrostfieldLogColor frostfield_mesh_lambert_log(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& parameters [[buffer(0)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]]
    FROSTFIELD_LOG_DEPTH_INPUT
) {
    return {frostfieldLambert(input, parameters, probe), frostfieldLogDepth(input.position, logDepthRange)};
}

static float4 frostfieldShadowed(
    FrostfieldMeshVaryings input,
    constant float4& tint,
    constant FrostfieldDirectionalShadow& directionalShadow,
    constant float4& light,
    constant FrostfieldLightProbe& probe,
    depth2d_array<float> shadowMap,
    texture2d_array<float, access::read> shadowMoments
) {
    float3 geometricNormal = normalize(input.normal);
    float facing = saturate(dot(geometricNormal, light.xyz));
    float visibility = frostfieldShadowVisibility(
        shadowMap, shadowMoments, input.worldPosition, input.shadowLocalPosition, geometricNormal, facing, directionalShadow
    );
    float diffuse = facing * light.w * visibility;
    float4 color = tint * input.instanceTint;
    float alpha = saturate(color.a);
    float3 directional = directionalShadow.lightColor.rgb * diffuse;
    float3 irradiance = 0.16 + 0.84 * directional + frostfieldAmbientIrradiance(probe) * M_1_PI_F;
    return float4(color.rgb * irradiance * alpha, alpha);
}

fragment float4 frostfield_mesh_shadowed(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& tint [[buffer(0)]],
    constant FrostfieldDirectionalShadow& directionalShadow [[buffer(1)]],
    constant float4& light [[buffer(2)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]],
    depth2d_array<float> shadowMap [[texture(1)]],
    texture2d_array<float, access::read> shadowMoments [[texture(28)]]
) {
    return frostfieldShadowed(input, tint, directionalShadow, light, probe, shadowMap, shadowMoments);
}

fragment FrostfieldLogColor frostfield_mesh_shadowed_log(
    FrostfieldMeshVaryings input [[stage_in]],
    constant float4& tint [[buffer(0)]],
    constant FrostfieldDirectionalShadow& directionalShadow [[buffer(1)]],
    constant float4& light [[buffer(2)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]],
    depth2d_array<float> shadowMap [[texture(1)]],
    texture2d_array<float, access::read> shadowMoments [[texture(28)]]
    FROSTFIELD_LOG_DEPTH_INPUT
) {
    return {frostfieldShadowed(input, tint, directionalShadow, light, probe, shadowMap, shadowMoments), frostfieldLogDepth(input.position, logDepthRange)};
}
