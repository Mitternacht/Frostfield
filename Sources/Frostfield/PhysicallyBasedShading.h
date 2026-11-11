// Shared by the bundled PBR shaders and generated surface variants.
#include "FrostfieldMaterialExtensions.h"
#ifndef FROSTFIELD_LIGHT_PROBE_H
#include "FrostfieldLightProbe.h"
#endif
constant bool frostfield_has_environment [[function_constant(0)]];
constant bool frostfield_has_clearcoat [[function_constant(1)]];
constant bool frostfield_has_transmission [[function_constant(2)]];
// Specular, sheen, anisotropy and iridescence layers (PhysicallyBasedMaterial.hasLayers).
constant bool frostfield_has_layers [[function_constant(3)]];
// MeshRenderState and vertex features. Each one compiles out when false, so materials
// that do not use them keep their original code. 4: alpha map at texture 15 with its
// UV rows at buffer 23. 5: vertex colors multiply the tint. 6: hashed alpha test.
// 7: alpha to coverage, which writes straight rather than premultiplied color and
// ramps a mask cutoff over one pixel (frostfieldCoverageCutoff).
constant bool frostfield_has_alpha_map [[function_constant(4)]];
constant bool frostfield_has_vertex_colors [[function_constant(5)]];
constant bool frostfield_alpha_hash [[function_constant(6)]];
constant bool frostfield_alpha_to_coverage [[function_constant(7)]];
constant bool frostfield_has_object_normals [[function_constant(8)]];
constant bool frostfield_has_raster_maps [[function_constant(9)]];

struct FrostfieldPBRVaryings {
    float4 position [[position]];
    float3 normal [[user(frostfield_normal)]];
    float2 uv [[user(frostfield_uv)]];
    float3 worldPosition [[user(frostfield_world_position)]];
    float2 uv1 [[user(frostfield_uv1)]];
    float4 tangent [[user(frostfield_tangent)]];
    // An instanced draw's per-instance linear RGBA, multiplied into the base color tint.
    float4 instanceTint [[user(frostfield_instance_tint), flat]];
    // Linear RGBA vertex color, or one when the mesh has none.
    float4 color [[user(frostfield_vertex_color)]];
    float3 normalX [[user(frostfield_object_normal_x), flat]];
    float3 normalY [[user(frostfield_object_normal_y), flat]];
    float3 normalZ [[user(frostfield_object_normal_z), flat]];
    float3 shadowLocalPosition [[user(frostfield_shadow_local_position)]];
};

static uint3 frostfieldPCG3(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

// A uniform value in [0, 1) for a lattice cell, hashed from its coordinates' bits.
static float frostfieldCellHash(float3 cell) {
    return float(frostfieldPCG3(as_type<uint3>(cell)).x >> 8u) * (1.0f / 16777216.0f);
}

// Wyman and McGuire's hashed alpha threshold ("Hashed Alpha Testing", I3D 2017), as in
// Three.js alphaHash: hash world position on two power-of-two lattices around 1/20 of
// a pixel, blend them, and remap the blend so the threshold stays uniform on [0, 1).
static float frostfieldAlphaHashThreshold(float3 position) {
    float derivative = max(length(dfdx(position)), length(dfdy(position)));
    float pixelScale = 1.0f / max(0.05f * derivative, 1e-30f);
    float level = log2(pixelScale);
    float2 scales = float2(exp2(floor(level)), exp2(ceil(level)));
    float2 alpha = float2(frostfieldCellHash(floor(scales.x * position)), frostfieldCellHash(floor(scales.y * position)));
    float blend = fract(level);
    float x = (1.0f - blend) * alpha.x + blend * alpha.y;
    float a = max(min(blend, 1.0f - blend), 1e-6f);
    float3 cases = float3(
        x * x / (2.0f * a * (1.0f - a)),
        (x - 0.5f * a) / (1.0f - a),
        1.0f - (1.0f - x) * (1.0f - x) / (2.0f * a * (1.0f - a))
    );
    float threshold = x < 1.0f - a ? (x < a ? cases.x : cases.y) : cases.z;
    return clamp(threshold, 1e-6f, 1.0f);
}

// Three.js alphaTest under alphaToCoverage: smoothstep(cutoff, cutoff + fwidth(a), a), so
// coverage ramps over about one pixel at the cutout edge and stays full inside. The
// explicit form keeps the ramp defined when fwidth is zero or the cutoff is large.
static float frostfieldCoverageCutoff(float alpha, float cutoff) {
    float width = max(fwidth(alpha), 1e-6f);
    float t = saturate((alpha - cutoff) / width);
    return t * t * (3.0f - 2.0f * t);
}

static float3 unitOr(float3 value, float3 fallback) {
    float largest = max(abs(value.x), max(abs(value.y), abs(value.z)));
    return largest > 0.0f && isfinite(largest) ? normalize(value / largest) : fallback;
}

static float3 pbrView(float4 camera, float3 position, float3 normal) {
    return unitOr(camera.w == 0.0f ? camera.xyz : camera.xyz - position, normal);
}

struct PBRSurface {
    float4 base;
    float metallic;
    float roughness;
    float3 normal;
    float occlusion;
    float3 emission;
    float transmissionMask;
};

// Fragment buffer 21, matching PhysicallyBasedMaterial.PackedExtensions (864 bytes).
struct FrostfieldPBRExtensions {
    float4 specular;       // color factor (rgb), factor (a)
    float4 sheen;          // color (rgb), perceptual roughness (a)
    float4 anisotropy;     // strength, cos(rotation), sin(rotation), unused
    float4 iridescence;    // factor, film IOR, minimum and maximum thickness in nanometers
    float4 options;        // x: 1 for unlit base color, 2 for unlit graph emission; y: light-map intensity; z: retroreflectivity
    uint4 maps;            // x: bound layer maps (bit i is map 10 + i); y, z: three-bit sampler slots
    float4 transforms[44]; // two texture transform rows per map
    float4 translucency;   // diffuse transmission color (rgb), translucent share of the diffuse lobe (a)
    float4 scattering;     // multiple-scattering color (rgb), mode (a): 0 none, 1 thin sheet, 2 diffuse volume
    float4 attenuation;    // raster attenuation per world unit (rgb), volume thickness in mesh units (a)
    float4 diffusion;      // diffusion length in world units (rgb), diffuse roughness (a)
};
static_assert(sizeof(FrostfieldPBRExtensions) == 864, "Raster PBR extension stride");
static_assert(__builtin_offsetof(FrostfieldPBRExtensions, transforms) == 96, "Raster PBR transform offset");
static_assert(__builtin_offsetof(FrostfieldPBRExtensions, translucency) == 800, "Raster PBR translucency offset");
static_assert(__builtin_offsetof(FrostfieldPBRExtensions, diffusion) == 848, "Raster PBR diffusion offset");

// Fragment buffer 22: handles for maps 10 to 20 (specular, specular color, sheen color,
// sheen roughness, anisotropy, iridescence, iridescence thickness, diffuse transmission,
// diffuse transmission color, diffuse roughness, multiple-scattering color), then the light map.
struct FrostfieldPBRLayerMaps {
    array<texture2d<float>, 11> maps [[id(0)]];
    texture2d<float, access::read> lightMap [[id(11)]];
};

// A material's UV set mask and normal encodings (buffer 9), and texture transform rows.
struct PBRTexCoords {
    uint mask;
    constant float4* transforms;
};

static float2 pbrUV(FrostfieldPBRVaryings input, PBRTexCoords coordinates, uint channel) {
    float2 uv = (coordinates.mask & (1u << channel)) != 0 ? input.uv1 : input.uv;
    return frostfieldTransformUV(uv, coordinates.transforms[2u * channel], coordinates.transforms[2u * channel + 1u]);
}

// A map's transform without its offset: (a, b, c, d) with u' = a u + b v and v' = c u + d v.
static float4 pbrUVLinear(PBRTexCoords coordinates, uint channel) {
    return float4(coordinates.transforms[2u * channel].xy, coordinates.transforms[2u * channel + 1u].xy);
}

// A unit tangent (xyz) and handedness (w) for a UV set, from the vertex tangent
// or screen-space derivatives. W is zero when the UVs define no frame. Derivatives
// of transformed UVs already follow a texture transform; a vertex tangent is carried
// to the transformed coordinates through the inverse of `linear` (see pbrUVLinear).
static float4 pbrTangentFrame(
    FrostfieldPBRVaryings input, float2 uv, float3 normal, bool useVertexTangent = true,
    float4 linear = float4(1, 0, 0, 1)
) {
    float3 px = dfdx(input.worldPosition);
    float3 py = dfdy(input.worldPosition);
    float2 ux = dfdx(uv);
    float2 uy = dfdy(uv);
    float3 tangent = input.tangent.xyz - normal * dot(normal, input.tangent.xyz);
    float handedness = input.tangent.w < 0 ? -1.0f : 1.0f;
    if (!useVertexTangent || abs(input.tangent.w) < 0.5f || dot(tangent, tangent) < 1e-12f) {
        float determinant = ux.x * uy.y - ux.y * uy.x;
        if (abs(determinant) < 1e-12f) {
            return float4(0);
        }
        tangent = (px * uy.y - py * ux.y) / determinant;
        tangent -= normal * dot(normal, tangent);
        if (dot(tangent, tangent) < 1e-12f) {
            return float4(0);
        }
        float3 uvY = (py * ux.x - px * uy.x) / determinant;
        handedness = dot(cross(normal, tangent), uvY) < 0 ? -1.0f : 1.0f;
    } else if (any(linear != float4(1, 0, 0, 1))) {
        // dP/du' and dP/dv' for u' = a u + b v and v' = c u + d v.
        float determinant = linear.x * linear.w - linear.y * linear.z;
        if (!(abs(determinant) > 1e-12f)) {
            return float4(0);
        }
        float3 bitangent = cross(normal, tangent) * handedness;
        float3 alongU = (tangent * linear.w - bitangent * linear.z) / determinant;
        float3 alongV = (bitangent * linear.x - tangent * linear.y) / determinant;
        tangent = alongU - normal * dot(normal, alongU);
        if (dot(tangent, tangent) < 1e-12f) {
            return float4(0);
        }
        handedness = dot(cross(normal, tangent), alongV) < 0 ? -1.0f : 1.0f;
    }
    return float4(unitOr(tangent, normal), handedness);
}

static float3 mappedNormal(
    FrostfieldPBRVaryings input, float2 uv, float3 normal, float3 texel, float2 scale, bool useVertexTangent,
    float4 linear, uint encoding, bool objectSpace
) {
    float3 decoded = frostfieldDecodeNormalMap(texel, encoding);
    if (frostfield_has_object_normals && objectSpace) {
        return unitOr(float3x3(input.normalX, input.normalY, input.normalZ) * decoded, normal);
    }
    float4 frame = pbrTangentFrame(input, uv, normal, useVertexTangent, linear);
    if (frame.w == 0.0f) {
        return normal;
    }
    float3 tangent = frame.xyz;
    float3 bitangent = cross(normal, tangent) * frame.w;
    return unitOr(
        tangent * (decoded.x * scale.x) + bitangent * (decoded.y * scale.y) + normal * decoded.z, normal
    );
}

static float3 pbrBumpNormal(
    float3 normal, float3 position, float2 uv, float scale, texture2d<float> heights, sampler heightSampler
) {
    float2 dx = dfdx(uv), dy = dfdy(uv);
    gradient2d lod(dx, dy);
    float base = heights.sample(heightSampler, uv, lod).r;
    float2 delta = float2(
        heights.sample(heightSampler, uv + dx, lod).r, heights.sample(heightSampler, uv + dy, lod).r
    ) - base;
    return frostfieldBumpNormal(normal, dfdx(position), dfdy(position), scale * delta);
}

static float pbrCoverage(
    float alpha, float3 position, float mode, float materialCutoff, float graphCutoff, bool mask, bool graphCoverage
) {
    if (!mask) { discard_fragment(); }
    bool graphTest = graphCutoff >= 0.0f;
    bool ramp = frostfield_alpha_to_coverage && (graphTest || mode == 2.0f);
    bool opaque = mode == 1.0f || mode == 2.0f;
    if (ramp) {
        alpha = frostfieldCoverageCutoff(alpha, graphTest ? graphCutoff : materialCutoff);
        if (alpha == 0.0f) { discard_fragment(); }
    } else {
        if (graphTest ? alpha <= graphCutoff : mode == 2.0f && alpha < materialCutoff) { discard_fragment(); }
        if (opaque && !graphCoverage) { alpha = 1.0f; }
    }
    if (frostfield_alpha_hash) {
        if (alpha < frostfieldAlphaHashThreshold(position)) { discard_fragment(); }
        return 1.0f;
    }
    return opaque && !ramp ? 1.0f : alpha;
}

static PBRSurface samplePBR(
    FrostfieldPBRVaryings input, bool front, float4 parameters, float4 camera, float4 baseColorFactor,
    float4 settings, float4 controls, float4 emission, PBRTexCoords coordinates,
    texture2d<float> colorTexture, sampler colorSampler,
    texture2d<float> metalRoughTexture, sampler metalRoughSampler,
    texture2d<float> normalTexture, sampler normalSampler,
    texture2d<float> occlusionTexture, sampler occlusionSampler,
    texture2d<float> emissiveTexture, sampler emissiveSampler,
    texture2d<float> alphaMapTexture, constant float4* alphaMapInfo
#if defined(FROSTFIELD_GENERATED_SURFACE)
    , thread const FrostfieldGraphResources& graphResources, bool graphShadow = false
    , float graphDirectionalVisibility = 1.0f
#if defined(FROSTFIELD_GRAPH_CAST_SHADOW)
    // Evaluates the authored colored shadow instead of the surface. The returned
    // surface.base then holds the transmitted RGB and a coverage of one.
    , bool graphCastShadow = false
#endif
#endif
) {
#if defined(FROSTFIELD_GENERATED_SURFACE)
    float4 tint = baseColorFactor * input.instanceTint;
#else
    float4 tint = parameters * baseColorFactor * input.instanceTint;
#endif
    if (frostfield_has_vertex_colors) {
        tint *= input.color;
    }
    float4 base = saturate(colorTexture.sample(colorSampler, pbrUV(input, coordinates, 0))) * saturate(tint);
    if (frostfield_has_alpha_map) {
        // Three.js alphaMap: the linear green channel multiplies alpha.
        float2 uv = alphaMapInfo[2].x != 0.0f ? input.uv1 : input.uv;
        base.a *= saturate(alphaMapTexture.sample(colorSampler, frostfieldTransformUV(uv, alphaMapInfo[0], alphaMapInfo[1])).g);
    }
#if !defined(FROSTFIELD_GRAPH_COVERAGE)
#if defined(FROSTFIELD_GRAPH_CAST_SHADOW)
    if (!graphCastShadow)
#endif
        base.a = pbrCoverage(base.a, input.worldPosition, settings.w, controls.z, -1.0f, true, false);
#endif
    float2 metalRough = controls.w > 0 ? saturate(metalRoughTexture.sample(metalRoughSampler, pbrUV(input, coordinates, 1)).bg)
        : float2(1);
    float3 normal = unitOr(input.normal, float3(0, 0, 1));
    bool objectNormal = frostfield_has_object_normals && (coordinates.mask & (1u << 28u)) != 0u;
    if (controls.x != 0 || objectNormal) {
        float2 normalUV = pbrUV(input, coordinates, 2);
        uint normalKind = (coordinates.mask >> 24u) & 3u;
        normal = normalKind == frostfieldBumpMapKind
            ? pbrBumpNormal(normal, input.worldPosition, normalUV, controls.x, normalTexture, normalSampler)
            : mappedNormal(
                input, normalUV, normal, normalTexture.sample(normalSampler, normalUV).rgb, float2(controls.x), true,
                pbrUVLinear(coordinates, 2), normalKind, objectNormal
            );
    }
    float occlusion = controls.y > 0 ? mix(1.0f, saturate(occlusionTexture.sample(occlusionSampler, pbrUV(input, coordinates, 3)).r), controls.y)
        : 1.0f;
    float3 emitted = emission.xyz;
    if (emission.w > 0) {
        emitted *= saturate(emissiveTexture.sample(emissiveSampler, pbrUV(input, coordinates, 4)).rgb);
    }
    PBRSurface surface = {
        base, settings.y * metalRough.x, settings.x * metalRough.y, front ? normal : -normal, occlusion, emitted, 1.0f
    };
#if defined(FROSTFIELD_GENERATED_SURFACE)
    // Graph normals and tangents use the interpolated vertex normal and UV0 frame. As in
    // the path tracer, the bitangent follows +V of UV0: its sign comes from the UV
    // derivatives rather than the vertex tangent's w, whose convention varies by source.
    float3 vertexNormal = unitOr(input.normal, float3(0, 0, 1));
    float4 graphFrame = pbrTangentFrame(input, input.uv, vertexNormal);
    float4 graphUVFrame = pbrTangentFrame(input, input.uv, vertexNormal, false);
    if (graphFrame.w != 0.0f && graphUVFrame.w != 0.0f) {
        float3 alongV = cross(vertexNormal, graphUVFrame.xyz) * graphUVFrame.w;
        graphFrame.w = dot(cross(vertexNormal, graphFrame.xyz), alongV) < 0.0f ? -1.0f : 1.0f;
    }
    FrostfieldSurfaceInput graphInput = {};
    graphInput.uv0 = input.uv;
    graphInput.uv1 = input.uv1;
    graphInput.worldPosition = input.worldPosition;
    graphInput.normal = surface.normal;
    graphInput.vertexNormal = vertexNormal;
    graphInput.tangent = graphFrame.xyz;
    graphInput.viewDirection = pbrView(camera, input.worldPosition, surface.normal);
    graphInput.parameters = parameters;
    graphInput.baseColor = surface.base.rgb;
    graphInput.metallic = surface.metallic;
    graphInput.roughness = surface.roughness;
    graphInput.emission = surface.emission;
    graphInput.bitangent = graphFrame.w != 0.0f ? unitOr(cross(vertexNormal, graphFrame.xyz) * graphFrame.w, float3(0))
        : float3(0);
    constant FrostfieldGraphView& graphView = *graphResources.view;
    graphInput.cameraPosition = graphView.camera.xyz;
    graphInput.cameraClip = float2(graphView.camera.w, graphView.clip.x);
    graphInput.directionalShadowVisibility = graphDirectionalVisibility;
    graphInput.viewMatrix = graphView.viewMatrix;
    graphInput.projectionMatrix = graphView.projectionMatrix;
    graphInput.modelMatrix = graphView.modelMatrix;
    graphInput.normalMatrix = float3x3(
        graphView.normalMatrix[0].xyz, graphView.normalMatrix[1].xyz, graphView.normalMatrix[2].xyz
    );
    graphInput.viewportSize = graphView.viewport.zw;
    graphInput.screenUV = (input.position.xy - graphView.viewport.xy) / graphView.viewport.zw;
    graphInput.screenCoordinate = input.position.xy;
    graphInput.screenSize = graphView.clip.yz;
    graphInput.particleDepth = input.position.zw;
    graphInput.depthRange = graphView.depthRange.xyz;
    FrostfieldSurfaceOutput graphOutput = {
        surface.base.rgb, surface.metallic, surface.roughness, surface.emission,
        surface.transmissionMask, surface.occlusion, float3(0, 0, 1), false, 1.0f, -1.0f, true
    };
#if defined(FROSTFIELD_GRAPH_CAST_SHADOW)
    if (graphCastShadow) {
        float4 authored = frostfield_generated_cast_shadow(graphInput, graphResources, graphOutput);
        // Coverage follows stock glass and the ordinary shadow pass. The base alpha scaled by the graph
        // opacity meets the graph or material mask cutoff. Opaque and mask modes then cover fully;
        // material alpha and blend modes keep that alpha. The authored alpha scales the result.
        float materialAlpha = saturate(surface.base.a * graphOutput.opacity);
        bool cutout = graphOutput.alphaTest >= 0.0f ? materialAlpha <= graphOutput.alphaTest
            : settings.w == 2.0f && materialAlpha < controls.z;
        if (!graphOutput.mask || cutout) { discard_fragment(); }
        float coverage = (settings.w == 1.0f || settings.w == 2.0f ? 1.0f : materialAlpha) * authored.a;
        if (coverage == 0.0f) { discard_fragment(); }
        float3 transmitted = 1.0f - coverage * (1.0f - authored.rgb);
        // A clear texel leaves the peel available to a farther colored surface.
        if (all(transmitted == 1.0f)) { discard_fragment(); }
        surface.base = float4(transmitted, 1.0f);
        return surface;
    }
#endif
#if defined(FROSTFIELD_GRAPH_COVERAGE)
    if (graphShadow) {
        frostfield_generated_coverage(graphInput, graphResources, graphOutput);
    } else
#endif
    {
        frostfield_generated_surface(graphInput, graphResources, graphOutput);
    }
#if defined(FROSTFIELD_GRAPH_COVERAGE)
    surface.base.a = pbrCoverage(
        surface.base.a * graphOutput.opacity, input.worldPosition, settings.w, controls.z, graphOutput.alphaTest, graphOutput.mask, true
    );
#endif
    surface.base.rgb = graphOutput.baseColor;
    surface.metallic = graphOutput.metallic;
    surface.roughness = graphOutput.roughness;
    surface.emission = graphOutput.emission;
    surface.transmissionMask = graphOutput.transmissionMask;
    surface.occlusion = graphOutput.occlusion;
    if (graphOutput.hasNormal && graphFrame.w != 0.0f) {
        // A graph normal replaces the authored normal map in the vertex normal's UV0 frame.
        float3 n = graphOutput.normal;
        float3 bitangent = cross(vertexNormal, graphFrame.xyz) * graphFrame.w;
        float3 world = unitOr(graphFrame.xyz * n.x + bitangent * n.y + vertexNormal * n.z, vertexNormal);
        surface.normal = front ? world : -world;
    }
#endif
    return surface;
}

struct PBRClearcoat {
    float3 normal;
    float roughness;
    float factor;
};

static PBRClearcoat sampleClearcoat(
    FrostfieldPBRVaryings input, bool front, PBRTexCoords coordinates, constant float4& layer,
    texture2d<float> factorTexture, sampler factorSampler,
    texture2d<float> roughnessTexture, sampler roughnessSampler,
    texture2d<float> normalTexture, sampler normalSampler
) {
    if (frostfield_has_clearcoat) {
        float factor = layer.x * saturate(factorTexture.sample(factorSampler, pbrUV(input, coordinates, 5)).r);
        float roughness = layer.y * saturate(roughnessTexture.sample(roughnessSampler, pbrUV(input, coordinates, 6)).g);
        float3 normal = unitOr(input.normal, float3(0, 0, 1));
        bool objectNormal = frostfield_has_object_normals && (coordinates.mask & (1u << 29u)) != 0u;
        if (layer.z != 0 || objectNormal) {
            float2 uv = pbrUV(input, coordinates, 7);
            // A different UV set needs its own frame rather than the base normal map's tangent.
            bool useVertexTangent = layer.w == 0.0f || ((coordinates.mask >> 2u) & 1u) == ((coordinates.mask >> 7u) & 1u);
            normal = mappedNormal(
                input, uv, normal, normalTexture.sample(normalSampler, uv).rgb, float2(layer.z), useVertexTangent,
                pbrUVLinear(coordinates, 7), (coordinates.mask >> 26u) & 3u, objectNormal
            );
        }
        return {front ? normal : -normal, roughness, factor};
    }
    return {float3(0, 0, 1), 0, 0};
}

struct PBRLayers {
    float3 specularColor;  // multiplies dielectric F0
    float specularWeight;  // scales dielectric F0 and F90
    float3 sheenColor;
    float sheenRoughness;
    float3 tangent;        // anisotropy direction, perpendicular to the shading normal
    float3 bitangent;
    float anisotropy;
    float iridescence;
    float thickness;       // film thickness in nanometers
    float filmIOR;
    float diffuseRoughness;  // KHR_materials_diffuse_roughness (EON r)
    float translucentShare;  // diffuse transmission or subsurface share of the diffuse lobe
    float3 scatterColor;     // light that enters and leaves the lit side of a volume
    float3 transmitColor;    // light that crosses to the other side
    float3 wrap;             // per-channel wrap of the scattered light around the terminator
    float3 mediumScatter;    // multiple-scattering color of a specular-transmission volume
    float retroreflectivity;
};

static float4 pbrLayerTexel(
    FrostfieldPBRVaryings input, PBRTexCoords coordinates, constant FrostfieldPBRExtensions& extensions,
    constant FrostfieldPBRLayerMaps& layerMaps, array<sampler, 6> samplers, uint map
) {
    uint slots = map < 7u ? extensions.maps.y >> (3u * map) : extensions.maps.z >> (3u * (map - 7u));
    uint slot = min(slots & 7u, 5u);
    return layerMaps.maps[map].sample(samplers[slot], pbrUV(input, coordinates, 10u + map));
}

// KHR specular, sheen, anisotropy and iridescence values at a fragment. Without the
// layers function constant this returns the neutral layers.
static PBRLayers sampleLayers(
    FrostfieldPBRVaryings input, PBRTexCoords coordinates, PBRSurface surface,
    constant FrostfieldPBRExtensions& extensions, constant FrostfieldPBRLayerMaps& layerMaps, array<sampler, 6> samplers
) {
    PBRLayers layers = {
        float3(1), 1.0f, float3(0), 0.0f, float3(0), float3(0), 0.0f, 0.0f, 0.0f, 1.3f, 0.0f, 0.0f, float3(0), float3(0), float3(0),
        float3(0), extensions.options.z
    };
    if (frostfield_has_layers) {
        uint bound = extensions.maps.x;
        layers.specularColor = extensions.specular.rgb;
        layers.specularWeight = extensions.specular.a;
        if ((bound & 1u) != 0) {
            layers.specularWeight *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 0u).a);
        }
        if ((bound & 2u) != 0) {
            layers.specularColor *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 1u).rgb);
        }
        layers.sheenColor = extensions.sheen.rgb;
        layers.sheenRoughness = extensions.sheen.a;
        if ((bound & 4u) != 0) {
            layers.sheenColor *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 2u).rgb);
        }
        if ((bound & 8u) != 0) {
            layers.sheenRoughness *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 3u).a);
        }
        float strength = extensions.anisotropy.x;
        if (strength > 0.0f) {
            float2 direction = float2(1, 0);
            if ((bound & 16u) != 0) {
                float3 texel = saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 4u).rgb);
                direction = texel.rg * 2.0f - 1.0f;
                strength *= texel.b;
            }
            float2 rotation = extensions.anisotropy.yz;
            direction = float2(
                rotation.x * direction.x - rotation.y * direction.y, rotation.y * direction.x + rotation.x * direction.y
            );
            // The mesh tangent frame: the vertex tangent, else derived from primary UVs.
            float3 vertexNormal = unitOr(input.normal, float3(0, 0, 1));
            float4 frame = pbrTangentFrame(input, input.uv, vertexNormal);
            float3 world = frame.xyz * direction.x + cross(vertexNormal, frame.xyz) * (frame.w * direction.y);
            float3 tangent = world - surface.normal * dot(surface.normal, world);
            if (frame.w != 0.0f && strength > 0.0f && dot(tangent, tangent) > 1e-12f) {
                layers.tangent = normalize(tangent);
                layers.bitangent = cross(surface.normal, layers.tangent);
                layers.anisotropy = strength;
            }
        }
        layers.iridescence = extensions.iridescence.x;
        if (layers.iridescence > 0.0f) {
            if ((bound & 32u) != 0) {
                layers.iridescence *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 5u).r);
            }
            float thickness = (bound & 64u) != 0
                ? saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 6u).g) : 1.0f;
            layers.thickness = mix(extensions.iridescence.z, extensions.iridescence.w, thickness);
            layers.filmIOR = extensions.iridescence.y;
        }
        layers.diffuseRoughness = extensions.diffusion.w;
        if (layers.diffuseRoughness > 0.0f && (bound & 512u) != 0) {
            layers.diffuseRoughness *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 9u).r);
        }
        if (extensions.scattering.a == 0.0f && any(extensions.scattering.rgb > 0.0f)) {
            layers.mediumScatter = extensions.scattering.rgb;
            if ((bound & 1024u) != 0) {
                layers.mediumScatter *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 10u).rgb);
            }
        }
        float share = extensions.translucency.a;
        if (share > 0.0f) {
            if ((bound & 128u) != 0) {
                share *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 7u).a);
            }
            float3 color = extensions.translucency.rgb;
            if ((bound & 256u) != 0) {
                color *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 8u).rgb);
            }
            layers.translucentShare = share;
            if (extensions.scattering.a == 1.0f) {
                // A thin sheet transmits once, with the Lambertian BTDF.
                layers.transmitColor = color;
            } else {
                // A diffuse volume: light crosses the boundary twice. Raster approximates the
                // walk's reflection as the multiple-scattering color and the light that reaches
                // the far side as attenuation through the authored thickness (in world units).
                float3 scatter = extensions.scattering.rgb;
                if ((bound & 1024u) != 0) {
                    scatter *= saturate(pbrLayerTexel(input, coordinates, extensions, layerMaps, samplers, 10u).rgb);
                }
                layers.scatterColor = color * color * scatter;
                float depth = extensions.attenuation.a;
                layers.transmitColor = depth > 0.0f ? color * color * exp(-extensions.attenuation.rgb * depth) : float3(0);
                // Diffusion spreads light over about one diffusion length; relative to the radius
                // of curvature that sets how far it wraps past the terminator.
                float3 vertexNormal = unitOr(input.normal, float3(0, 0, 1));
                float spread = length(fwidth(input.worldPosition));
                float curvature = spread > 0.0f ? length(fwidth(vertexNormal)) / spread : 0.0f;
                layers.wrap = saturate(extensions.diffusion.rgb * curvature);
            }
        }
    }
    return layers;
}

static float3 environmentDirection(float3 direction, float2 rotation) {
    // Sampling uses the inverse of the environment's world-space yaw.
    return float3(
        rotation.x * direction.x - rotation.y * direction.z,
        direction.y,
        rotation.y * direction.x + rotation.x * direction.z
    );
}

// Matches RasterEnvironmentLookup. The w lanes of x and y hold intensity and max LOD.
struct FrostfieldEnvironmentLookup {
    float4 x, y, z;
    float4 minimum, maximum, capture;
};

static float3 environmentDirection(float3 direction, constant FrostfieldEnvironmentLookup& environment) {
    return float3x3(environment.x.xyz, environment.y.xyz, environment.z.xyz) * direction;
}

static float3 environmentReflectionDirection(
    float3 direction, float3 position, constant FrostfieldEnvironmentLookup& environment
) {
    if (environment.capture.w != 0.0f && all(position >= environment.minimum.xyz) && all(position <= environment.maximum.xyz)) {
        // Scaling the dominant ray axis to one keeps the nearest exit finite.
        float scale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
        float3 ray = direction / scale;
        bool3 parallel = ray == 0.0f;
        float3 boundary = select(environment.minimum.xyz, environment.maximum.xyz, ray >= 0.0f);
        float3 distance = (boundary - position) / select(ray, float3(1), parallel);
        distance = select(distance, float3(INFINITY), parallel);
        float exit = min(distance.x, min(distance.y, distance.z));
        float3 corrected = fma(ray, float3(exit), position - environment.capture.xyz);
        corrected = select(corrected, boundary - environment.capture.xyz, distance == exit);
        direction = unitOr(corrected, direction);
    }
    return environmentDirection(direction, environment);
}

// Eight float4 slots with a 128-byte stride, matching MeshRenderer.PackedPunctualLight.
struct FrostfieldPunctualLight {
    float4 positionKind;   // world position or rectangle center; w kind: 0 directional, 1 point, 2 spot, 3 rectangle
    float4 directionRange; // unit emission direction (local -Z); w range in meters or 0
    float4 colorIntensity; // linear color; w lux, candela, or nits by kind
    float4 cone;           // cos inner, cos outer, 1 when ranged, shadow view half-angle tangent
    float4 axisU;          // unit rectangle width axis; w half width in meters. Point and spot: local +X, w 1 + IES profile layer
    float4 axisV;          // unit rectangle height axis; w half height in meters. Spot: x 1 + cookie layer, y tan(outer)
    float4 shadow;         // first slice or -1, depth bias (m), normal bias (texels), filter radius or -1 basic / -2 variance
    float4 shadowReceiver; // shadow strength (x)
};

static float2 pbrGGX(float roughness, float noH, float noV, float noL) {
    constexpr float inversePi = 0.31830988618f;
    // GGX with height-correlated Smith visibility. Square perceptual roughness.
    float a = max(roughness, 0.05f);
    a *= a;
    float a2 = a * a;
    float noH2 = noH * noH;
    float d = (1.0f - noH2) + noH2 * a2;
    float distribution = inversePi * a2 / max(d * d, 1e-12f);
    float visibilityV = noL * sqrt(noV * noV * (1.0f - a2) + a2);
    float visibilityL = noV * sqrt(noL * noL * (1.0f - a2) + a2);
    return float2(distribution, 0.5f / max(visibilityV + visibilityL, 1e-6f));
}

// Microfacet distribution and visibility: anisotropic GGX for an anisotropy layer, else isotropic.
static float2 pbrLobe(
    PBRSurface surface, PBRLayers layers, float3 h, float3 view, float3 directionToLight,
    float noH, float noV, float noL
) {
    if (frostfield_has_layers && layers.anisotropy > 0.0f) {
        float a = max(surface.roughness, 0.05f);
        float2 alpha = frostfieldAnisotropicAlpha(a * a, layers.anisotropy);
        float3 localH = float3(dot(layers.tangent, h), dot(layers.bitangent, h), noH);
        float3 localV = float3(dot(layers.tangent, view), dot(layers.bitangent, view), noV);
        float3 localL = float3(dot(layers.tangent, directionToLight), dot(layers.bitangent, directionToLight), noL);
        return float2(frostfieldAnisotropicGGX(localH, alpha), frostfieldAnisotropicVisibility(localV, localL, alpha));
    }
    return pbrGGX(surface.roughness, noH, noV, noL);
}

// Schlick's Fresnel from F0 to F90, mixed toward the iridescent film reflectance.
static float3 pbrFresnel(float3 f0, float f90, float voH, PBRLayers layers, float3 iridescent) {
    float falloff = 1.0f - voH;
    float squared = falloff * falloff;
    float3 fresnel = f0 + (f90 - f0) * (squared * squared * falloff);
    if (frostfield_has_layers && layers.iridescence > 0.0f) {
        fresnel = mix(fresnel, iridescent, layers.iridescence);
    }
    return fresnel;
}

// Adds the sheen lobe and applies its albedo scaling to the layers beneath.
static float3 pbrSheen(PBRLayers layers, float3 beneath, float noH, float noV, float noL) {
    if (!frostfield_has_layers || all(layers.sheenColor <= 0.0f)) {
        return beneath;
    }
    return beneath * frostfieldSheenScaling(layers.sheenColor, layers.sheenRoughness, noV, noL)
        + layers.sheenColor * (frostfieldCharlie(noH, layers.sheenRoughness) * frostfieldSheenVisibility(noL, noV));
}

// The opaque diffuse BRDF times pi: Lambertian, or EON for a rough diffuse layer.
static float3 pbrDiffuseLobe(float3 albedo, PBRLayers layers, float3 n, float3 view, float3 l, float noV, float noL) {
    if (!frostfield_has_layers || layers.diffuseRoughness <= 0.0f) {
        return albedo;
    }
    float s = dot(l, view) - noL * noV;
    return frostfieldEON(albedo, layers.diffuseRoughness, noL, noV, s) * M_PI_F;
}

// Wrapped and back-lit diffuse light of the translucent share for one light, per unit
// radiance and before the diffuse weight. `cosine` is dot(n, l), which may be negative.
static float3 pbrTranslucentDirect(
    PBRLayers layers, float3 n, float3 view, float3 l, float cosine, float3 f0, float f90, float3 iridescent
) {
    constexpr float inversePi = 0.31830988618f;
    if (!frostfield_has_layers || layers.translucentShare <= 0.0f) {
        return float3(0);
    }
    float3 result = float3(0);
    if (any(layers.scatterColor > 0.0f)) {
        // Energy-conserving wrap: max(0, cos + w) / (1 + w)^2 integrates like the cosine.
        float3 wrapped = max(float3(0), cosine + layers.wrap) / ((1.0f + layers.wrap) * (1.0f + layers.wrap));
        float3 h = unitOr(view + l, n);
        float3 fresnel = pbrFresnel(f0, f90, saturate(dot(view, h)), layers, iridescent);
        result += (1.0f - fresnel) * layers.scatterColor * wrapped;
    }
    if (cosine < 0.0f && any(layers.transmitColor > 0.0f)) {
        // The Fresnel of the mirrored light direction, as the glTF diffuse BTDF uses its half vector.
        float3 mirrored = l - 2.0f * cosine * n;
        float3 h = unitOr(view + mirrored, n);
        float3 fresnel = pbrFresnel(f0, f90, saturate(dot(view, h)), layers, iridescent);
        result += (1.0f - fresnel) * layers.transmitColor * (-cosine);
    }
    return result * (layers.translucentShare * inversePi);
}

static float3 pbrDirect(
    PBRSurface surface, PBRLayers layers, float3 view, float noV, float3 f0, float f90, float3 iridescent,
    float3 diffuseColor, float3 directionToLight, float3 incident, float diffuseWeight = 0.0f
) {
    constexpr float inversePi = 0.31830988618f;
    float3 n = surface.normal;
    float cosine = dot(n, directionToLight);
    float3 translucent = noV > 0.0f
        ? pbrTranslucentDirect(layers, n, view, directionToLight, cosine, f0, f90, iridescent) * (diffuseWeight * incident)
        : float3(0);
    float noL = saturate(cosine);
    if (noL <= 0.0f || noV <= 0.0f) {
        return translucent;
    }

    float3 h = unitOr(view + directionToLight, n);
    float noH = saturate(dot(n, h));
    float voH = saturate(dot(view, h));
    float2 ggx = pbrLobe(surface, layers, h, view, directionToLight, noH, noV, noL);

    float3 fresnel = pbrFresnel(f0, f90, voH, layers, iridescent);
    float3 specular = fresnel * ggx.x * ggx.y;
    if (layers.retroreflectivity > 0.0f) {
        float3 retroView = reflect(-view, n);
        float3 retroHalf = unitOr(retroView + directionToLight, n);
        float2 retroGGX = pbrLobe(
            surface, layers, retroHalf, retroView, directionToLight, saturate(dot(n, retroHalf)), noV, noL
        );
        float3 retroFresnel = pbrFresnel(f0, f90, saturate(dot(retroView, retroHalf)), layers, iridescent);
        specular = mix(specular, retroFresnel * retroGGX.x * retroGGX.y, layers.retroreflectivity);
        fresnel = mix(fresnel, retroFresnel, layers.retroreflectivity);
    }
    float3 lobe = frostfield_has_layers && layers.diffuseRoughness > 0.0f
        ? pbrDiffuseLobe(surface.base.rgb, layers, n, view, directionToLight, noV, noL) * diffuseWeight
        : diffuseColor;
    float3 diffuse = (1.0f - fresnel) * lobe * ((1.0f - layers.translucentShare) * inversePi);
    return pbrSheen(layers, diffuse + specular, noH, noV, noL) * (noL * incident) + translucent;
}

static float3 pbrClearcoatDirect(PBRClearcoat coat, float3 view, float3 directionToLight, float3 incident) {
    float noV = saturate(dot(coat.normal, view));
    float noL = saturate(dot(coat.normal, directionToLight));
    if (noV <= 0.0f || noL <= 0.0f) {
        return float3(0);
    }
    float3 halfVector = unitOr(view + directionToLight, coat.normal);
    float2 ggx = pbrGGX(coat.roughness, saturate(dot(coat.normal, halfVector)), noV, noL);
    return incident * (ggx.x * ggx.y * noL);
}

struct PBRIncidentLight {
    float3 direction;
    float3 radiance;
};

// IES profile and cookie filtering, in lockstep with ptLightTables in PathTracingShading.h.
// toSurface is the unit direction from the light, axis its unit emission direction and right
// its local +X. Profile tables hold 361 rows of vertical angle from the axis (0°–180°) and 144
// columns of horizontal angle from +X toward +Y; layers are stored plus one, zero for none.
// A cookie spans [-scale, scale]² at unit distance, top row toward +Y (LightTextureSet).
static float3 pbrLightTables(
    float3 toSurface, float3 axis, float3 right, float profileLayer, float cookieLayer, float cookieScale,
    texture2d_array<float> profiles, texture2d_array<float> cookies
) {
    float3 result = float3(1);
    float3 up = cross(right, axis);
    float x = dot(toSurface, right);
    float y = dot(toSurface, up);
    float z = dot(toSurface, axis);
    // Both tables filter bilinearly by hand: fragment functions have no sampler slots to spare.
    if (profileLayer > 0.5f) {
        constexpr uint width = 144u;
        constexpr uint height = 361u;
        uint layer = uint(profileLayer - 0.5f);
        float phi = x == 0.0f && y == 0.0f ? 0.0f : atan2(y, x);
        float column = phi * (0.5f * M_1_PI_F) * float(width);
        column -= floor(column / float(width)) * float(width);
        float row = acos(clamp(z, -1.0f, 1.0f)) * M_1_PI_F * float(height - 1u);
        uint c0 = min(uint(column), width - 1u);
        uint c1 = (c0 + 1u) % width;
        uint r0 = min(uint(row), height - 2u);
        float tx = saturate(column - float(c0));
        float ty = saturate(row - float(r0));
        float top = mix(profiles.read(uint2(c0, r0), layer).r, profiles.read(uint2(c1, r0), layer).r, tx);
        float bottom = mix(profiles.read(uint2(c0, r0 + 1u), layer).r, profiles.read(uint2(c1, r0 + 1u), layer).r, tx);
        result *= mix(top, bottom, ty);
    }
    if (cookieLayer > 0.5f) {
        if (!(z > 0.0f) || !(cookieScale > 0.0f)) { return float3(0); }
        uint layer = uint(cookieLayer - 0.5f);
        float2 size = float2(cookies.get_width(), cookies.get_height());
        float2 uv = float2(0.5f + 0.5f * x / (z * cookieScale), 0.5f - 0.5f * y / (z * cookieScale));
        float2 position = clamp(uv * size - 0.5f, float2(0), size - 1.0f);
        uint2 p0 = uint2(position);
        uint2 p1 = min(p0 + 1u, uint2(size) - 1u);
        float2 t = position - float2(p0);
        float3 top = mix(cookies.read(p0, layer).rgb, cookies.read(uint2(p1.x, p0.y), layer).rgb, t.x);
        float3 bottom = mix(cookies.read(uint2(p0.x, p1.y), layer).rgb, cookies.read(p1, layer).rgb, t.x);
        result *= mix(top, bottom, t.y);
    }
    return result;
}

// The fourth components mark an eligible model and an eligible light.
static inline float4 pbrRelativeReceiver(float3 local, constant float4x4& model) {
    return model[3].w != 0.0f ? model * float4(local, 1.0f) : float4(0);
}

static inline float3 pbrRelativeView(float4 camera, float3 position, float3 normal, float4 relativeReceiver) {
    if (relativeReceiver.w != 0.0f && camera.w != 0.0f) {
        return pbrView(float4(0, 0, 0, 1), relativeReceiver.xyz, normal);
    }
    return pbrView(camera, position, normal);
}

static PBRIncidentLight pbrPunctual(
    float3 position, constant FrostfieldPunctualLight& light,
    texture2d_array<float> lightProfiles, texture2d_array<float> lightCookies,
    float4 relativeReceiver, float4 relativeLight
) {
    constexpr float maxHalf = 65504.0f;
    float3 directionToLight = -light.directionRange.xyz;
    float attenuation = 1.0f;
    if (light.positionKind.w > 0.0f) {
        float3 delta = relativeLight.w != 0.0f ? relativeLight.xyz - relativeReceiver.xyz
                                              : light.positionKind.xyz - position;
        float distanceSquared = dot(delta, delta);
        if (distanceSquared <= 0.0f || !isfinite(distanceSquared)) {
            return {float3(0), float3(0)};
        }
        float distance = sqrt(distanceSquared);
        directionToLight = delta / distance;
        attenuation = 1.0f / max(distanceSquared, 0.0001f);
        if (light.cone.z > 0.0f) {
            if (distance >= light.directionRange.w) {
                return {float3(0), float3(0)};
            }
            float ratio = distance / light.directionRange.w;
            float squared = ratio * ratio;
            attenuation *= saturate(1.0f - squared * squared);
        }
        if (light.positionKind.w == 2.0f) {
            float cosine = dot(light.directionRange.xyz, -directionToLight);
            float width = light.cone.x - light.cone.y;
            float cone = width > 0.0f
                ? saturate((cosine - light.cone.y) / width)
                : float(cosine >= light.cone.x);
            attenuation *= cone * cone;
        }
    }
    if (attenuation <= 0.0f) {
        return {float3(0), float3(0)};
    }
    float3 emitted = light.colorIntensity.rgb * light.colorIntensity.w;
    // Point and spot records carry profile and cookie layers in axisU.w and axisV.x.
    if (light.positionKind.w > 0.0f && (light.axisU.w > 0.5f || light.axisV.x > 0.5f)) {
        emitted *= pbrLightTables(
            -directionToLight, light.directionRange.xyz, light.axisU.xyz, light.axisU.w, light.axisV.x, light.axisV.y,
            lightProfiles, lightCookies
        );
    }
    float3 incident = min(emitted, float3(maxHalf / attenuation)) * attenuation;
    return {directionToLight, incident};
}

// Shadow-map visibility for a point, spot, or rectangle light with an assigned slice.
// Views use reversed, infinite-far depth: stored depth is near / distance along the view axis.
static float pbrPunctualShadowDepthVisibility(
    float3 position, float3 geometricNormal, constant FrostfieldPunctualLight& light,
    depth2d_array<float> shadowMaps, texture2d_array<float, access::read> moments, constant float4x4* shadowViews,
    float4 relativeReceiver, float4 relativeLight
) {
    bool preciseLight = relativeLight.w != 0.0f;
    float3 offset = preciseLight ? relativeReceiver.xyz - relativeLight.xyz : position - light.positionKind.xyz;
    float distance = length(offset);
    if (!(distance > 0.0f) || !isfinite(distance)) {
        return 1.0f;
    }
    bool spot = light.positionKind.w == 2.0f;
    float3 axes = abs(offset);
    float depth = spot ? dot(offset, light.directionRange.xyz) : max(axes.x, max(axes.y, axes.z));
    if (depth <= 0.0f) {
        return 1.0f;
    }

    // A plane at angle theta to the light changes depth by tan(theta) texel widths per
    // texel. Offsetting the receiver along its normal by sin(theta) texel widths per
    // texel of footprint (filter radius, one bilinear texel, and the normal bias)
    // clears that slope. The 0.1 floor absorbs depth rounding at normal incidence.
    float resolution = float(shadowMaps.get_width());
    float texelSize = 2.0f * depth * light.cone.w / resolution;
    float facing = saturate(dot(geometricNormal, -offset / distance));
    float sine = max(sqrt(saturate(1.0f - facing * facing)), 0.1f);
    float radius = light.shadow.w;
    float normalBias = radius < 0.0f ? light.shadow.z : light.shadow.z + radius + 1.0f;
    float3 normalOffset = geometricNormal * (texelSize * sine * normalBias);
    float3 receiver = position + normalOffset;

    uint slice = uint(light.shadow.x);
    if (!spot) {
        // Faces are +X, -X, +Y, -Y, +Z, -Z, matching PunctualShadowMaps.faceDirections.
        float3 direction = preciseLight ? offset + normalOffset : receiver - light.positionKind.xyz;
        float3 major = abs(direction);
        slice += major.x >= major.y && major.x >= major.z ? (direction.x >= 0.0f ? 0u : 1u)
            : major.y >= major.z ? (direction.y >= 0.0f ? 2u : 3u) : (direction.z >= 0.0f ? 4u : 5u);
    }
    float4 clip;
    if (preciseLight) {
        // The light-relative offset omits the old world translation. Keep
        // the near-depth constant and add normal bias as a separate vector.
        constant float4x4& projection = shadowViews[slice];
        clip = projection * float4(offset, 0.0f) + float4(0, 0, projection[3].z, 0);
        clip += projection * float4(normalOffset, 0.0f);
    } else {
        clip = shadowViews[slice] * float4(receiver, 1.0f);
    }
    float biased = clip.w - light.shadow.y;
    if (clip.w <= 0.0f || biased <= 0.0f) {
        return 1.0f;
    }
    float2 uv = float2(clip.x / clip.w * 0.5f + 0.5f, 0.5f - clip.y / clip.w * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f)) {
        return 1.0f;
    }
    float reference = clip.z / biased;
    if (radius < -1.0f) {
        return frostfieldVarianceVisibility(moments, uv, slice, reference, true);
    }
    if (radius < 0.0f) {
        constexpr sampler nearest(
            coord::normalized, address::clamp_to_edge, filter::nearest, compare_func::greater_equal
        );
        return shadowMaps.sample_compare(nearest, uv, slice, reference);
    }
    constexpr sampler comparison(
        coord::normalized, address::clamp_to_edge, filter::linear, compare_func::greater_equal
    );
    if (radius <= 0.0f) {
        return shadowMaps.sample_compare(comparison, uv, slice, reference);
    }

    constexpr float2 disk[16] = {
        float2( 0.1768,  0.0000), float2(-0.2258,  0.2068), float2( 0.0346, -0.3938), float2( 0.2846,  0.3712),
        float2(-0.5222, -0.0924), float2( 0.4947, -0.3147), float2(-0.1655,  0.6155), float2(-0.3156, -0.6076),
        float2( 0.6846,  0.2500), float2(-0.7123,  0.2940), float2( 0.3434, -0.7337), float2( 0.2537,  0.8089),
        float2(-0.7647, -0.4432), float2( 0.8971, -0.1972), float2(-0.5475,  0.7788), float2(-0.1265, -0.9761)
    };
    // Rotate the disk per shadow texel so the filter has no regular bands and
    // stays stable when the camera moves.
    float texel = 1.0f / resolution;
    float2 cell = floor(uv * resolution);
    float angle = 2.0f * M_PI_F * fract(52.9829189f * fract(dot(cell, float2(0.06711056f, 0.00583715f))));
    float cosine;
    float sineAngle = sincos(angle, cosine);
    float2x2 rotation(float2(cosine, sineAngle), float2(-sineAngle, cosine));
    float visible = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        visible += shadowMaps.sample_compare(comparison, uv + rotation * disk[i] * (radius * texel), slice, reference);
    }
    return visible / 16.0f;
}

static float pbrPunctualShadow(
    float3 position, float3 geometricNormal, constant FrostfieldPunctualLight& light,
    depth2d_array<float> shadowMaps, texture2d_array<float, access::read> moments, constant float4x4* shadowViews,
    float4 relativeReceiver = float4(0), float4 relativeLight = float4(0)
) {
    float strength = light.shadowReceiver.x;
    if (light.shadow.x < 0.0f || strength == 0.0f) {
        return 1.0f;
    }
    float visible = pbrPunctualShadowDepthVisibility(
        position, geometricNormal, light, shadowMaps, moments, shadowViews, relativeReceiver, relativeLight
    );
    return strength == 1.0f ? visible : mix(1.0f, visible, strength);
}

// Lambertian irradiance per unit radiance from the part of a rectangle above the
// receiver's tangent plane: clip the polygon to the hemisphere, then sum the exact
// edge integrals (Lambert's vector irradiance). Corners are relative to the receiver.
static float pbrRectangleIrradiance(float3 c0, float3 c1, float3 c2, float3 c3, float3 normal) {
    float3 corners[4] = {c0, c1, c2, c3};
    float3 polygon[6];
    uint count = 0;
    for (uint i = 0; i < 4; ++i) {
        float3 a = corners[i];
        float3 b = corners[(i + 1u) & 3u];
        float heightA = dot(a, normal);
        float heightB = dot(b, normal);
        if (heightA >= 0.0f && count < 6u) {
            polygon[count++] = a;
        }
        if (((heightA > 0.0f && heightB < 0.0f) || (heightA < 0.0f && heightB > 0.0f)) && count < 6u) {
            polygon[count++] = mix(a, b, heightA / (heightA - heightB));
        }
    }
    if (count < 3u) {
        return 0.0f;
    }
    float3 sum = float3(0);
    for (uint i = 0; i < count; ++i) {
        float3 a = unitOr(polygon[i], normal);
        float3 b = unitOr(polygon[i + 1u == count ? 0u : i + 1u], normal);
        float3 edge = cross(a, b);
        float sine = length(edge);
        float angle = atan2(sine, dot(a, b));
        sum += sine > 1e-7f ? edge * (angle / sine) : edge;
    }
    // After clipping every point has a nonnegative cosine, so the magnitude is the result.
    return 0.5f * abs(dot(sum, normal));
}

// Van Oosterom and Strackee's solid angle of a triangle seen from the origin.
static float pbrTriangleSolidAngle(float3 a, float3 b, float3 c) {
    float la = length(a);
    float lb = length(b);
    float lc = length(c);
    float numerator = abs(dot(a, cross(b, c)));
    float denominator = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
    return 2.0f * atan2(numerator, denominator);
}

struct PBRRectangleLight {
    float3 corners[4];     // relative to the receiver
    float3 toCenter;
    float solidAngle;
    float3 radiance;
    bool farField;         // small enough to treat as a point for irradiance and solid angle
};

static bool pbrRectangle(float3 toCenter, constant FrostfieldPunctualLight& light, thread PBRRectangleLight& result) {
    // One-sided: only points in front of the emitting face receive light.
    if (!(dot(toCenter, light.directionRange.xyz) < 0.0f)) {
        return false;
    }
    float3 u = light.axisU.xyz * light.axisU.w;
    float3 v = light.axisV.xyz * light.axisV.w;
    result.corners[0] = toCenter - u - v;
    result.corners[1] = toCenter + u - v;
    result.corners[2] = toCenter + u + v;
    result.corners[3] = toCenter - u + v;
    result.toCenter = toCenter;
    // Below a size-to-distance ratio of 1e-3 the exact edge sums lose precision to
    // cancellation, while the point approximation's relative error is about 1e-6.
    float distanceSquared = dot(toCenter, toCenter);
    float extentSquared = light.axisU.w * light.axisU.w + light.axisV.w * light.axisV.w;
    result.farField = extentSquared < 1e-6f * distanceSquared;
    if (result.farField) {
        float cosine = -dot(toCenter, light.directionRange.xyz) * rsqrt(distanceSquared);
        result.solidAngle = 4.0f * light.axisU.w * light.axisV.w * cosine / distanceSquared;
    } else {
        result.solidAngle = pbrTriangleSolidAngle(result.corners[0], result.corners[1], result.corners[2])
            + pbrTriangleSolidAngle(result.corners[0], result.corners[2], result.corners[3]);
    }
    result.radiance = light.colorIntensity.rgb * light.colorIntensity.w;
    return result.solidAngle > 0.0f && isfinite(result.solidAngle);
}

// Karis's representative point: the rectangle point closest to the reflection ray.
// The returned w widens the GGX lobe by the emitter's extent in half-vector space
// ((alpha / alpha') per axis), which reflects the emitter's radiance in the mirror limit.
static float4 pbrRectangleSpecular(
    PBRRectangleLight rectangle, constant FrostfieldPunctualLight& light,
    float3 normal, float3 view, float roughness
) {
    float3 emission = light.directionRange.xyz;
    float3 ray = reflect(-view, normal);
    float along = dot(ray, emission);
    float reach = 1e4f * length(rectangle.toCenter);
    // Rays that miss the plane select the edge toward the reflected direction.
    float travel = along < -1e-4f ? min(dot(rectangle.toCenter, emission) / along, reach) : reach;
    float3 local = ray * travel - rectangle.toCenter;
    float s = clamp(dot(local, light.axisU.xyz), -light.axisU.w, light.axisU.w);
    float t = clamp(dot(local, light.axisV.xyz), -light.axisV.w, light.axisV.w);
    float3 direction = unitOr(rectangle.toCenter + light.axisU.xyz * s + light.axisV.xyz * t, normal);

    float alpha = max(roughness, 0.05f);
    alpha *= alpha;
    float voH = max(saturate(dot(view, unitOr(view + direction, normal))), 0.05f);
    // The emitter spans solidAngle / (4 voH) in half-vector space.
    float spread = rectangle.solidAngle / (4.0f * M_PI_F * voH);
    float aspect = light.axisU.w / light.axisV.w;
    float widthAlpha = min(alpha + sqrt(spread * aspect), 1.0f);
    float heightAlpha = min(alpha + sqrt(spread / aspect), 1.0f);
    return float4(direction, alpha * alpha / (widthAlpha * heightAlpha));
}

// r187 LTC form factor, using Heitz, Dupuy, Hill and Neubelt's GGX fit.
static float3 pbrLTCEdge(float3 a, float3 b) {
    float cosine = clamp(dot(a, b), -1.0f, 1.0f);
    float y = abs(cosine);
    float positive = (0.8543985f + (0.4965155f + 0.0145206f * y) * y)
        / (3.4175940f + (4.1616724f + y) * y);
    float ratio = cosine > 0.0f ? positive : 0.5f * rsqrt(max(1.0f - cosine * cosine, 1e-7f)) - positive;
    return cross(a, b) * ratio;
}

// PBR material bindings already occupy the fragment sampler slots. Interpolate the
// two LUT layers at the r187 texel-center coordinates using direct texel reads.
static float4 pbrLTCTable(texture2d_array<float> tables, float2 texel, uint layer) {
    uint2 lower = uint2(texel);
    uint2 upper = min(lower + 1u, uint2(63));
    float2 weight = texel - float2(lower);
    float4 top = mix(tables.read(lower, layer), tables.read(uint2(upper.x, lower.y), layer), weight.x);
    float4 bottom = mix(tables.read(uint2(lower.x, upper.y), layer), tables.read(upper, layer), weight.x);
    return mix(top, bottom, weight.y);
}

static float3 pbrRectangleLTC(
    PBRRectangleLight rectangle, float3 normal, float3 view, float roughness, float3 f0, float3 f90,
    texture2d_array<float> tables
) {
    float noV = dot(normal, view);
    if (noV <= 0.0f) { return float3(0); }
    float2 texel = float2(saturate(roughness), sqrt(1.0f - saturate(noV))) * 63.0f;
    float4 matrix = pbrLTCTable(tables, texel, 0);
    float2 magnitude = pbrLTCTable(tables, texel, 1).rg;
    float3x3 inverseLTC(float3(matrix.x, 0, matrix.y), float3(0, 1, 0), float3(matrix.z, 0, matrix.w));
    // At normal incidence any unit tangent is valid. Use a stable basis instead of normalizing zero.
    float3 axis = abs(normal.z) < 0.99f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = unitOr(view - normal * noV, unitOr(cross(axis, normal), float3(1, 0, 0)));
    float3x3 transform = inverseLTC * transpose(float3x3(tangent, -cross(normal, tangent), normal));
    // Light vertices wind counterclockwise from the emitting face in the LTC reference.
    float3 a = unitOr(transform * rectangle.corners[1], float3(0, 0, 1));
    float3 b = unitOr(transform * rectangle.corners[0], float3(0, 0, 1));
    float3 c = unitOr(transform * rectangle.corners[3], float3(0, 0, 1));
    float3 d = unitOr(transform * rectangle.corners[2], float3(0, 0, 1));
    float3 form = pbrLTCEdge(a, b) + pbrLTCEdge(b, c) + pbrLTCEdge(c, d) + pbrLTCEdge(d, a);
    float len = length(form);
    float horizonClipped = max((dot(form, form) + form.z) / (len + 1.0f), 0.0f);
    return (f0 * magnitude.x + (f90 - f0) * magnitude.y) * horizonClipped;
}

static float3 pbrRectangleDirect(
    PBRSurface surface, PBRLayers layers, PBRRectangleLight rectangle, float4 specular,
    float3 view, float noV, float3 f0, float f90, float3 iridescent, float3 diffuseColor, float diffuseWeight,
    texture2d_array<float> rectangleLTC
) {
    constexpr float inversePi = 0.31830988618f;
    if (noV <= 0.0f) {
        return float3(0);
    }
    float3 n = surface.normal;
    float3 l = specular.xyz;
    float3 h = unitOr(view + l, n);
    float voH = saturate(dot(view, h));
    float3 fresnel = pbrFresnel(f0, f90, voH, layers, iridescent);
    float irradiance = rectangle.farField
        ? rectangle.solidAngle * saturate(dot(n, unitOr(rectangle.toCenter, n)))
        : pbrRectangleIrradiance(rectangle.corners[0], rectangle.corners[1], rectangle.corners[2], rectangle.corners[3], n);
    float3 lobe = diffuseColor;
    if (frostfield_has_layers && layers.diffuseRoughness > 0.0f) {
        // The rough diffuse BRDF toward the panel's center.
        float3 center = unitOr(rectangle.toCenter, n);
        lobe = pbrDiffuseLobe(surface.base.rgb, layers, n, view, center, noV, saturate(dot(n, center))) * diffuseWeight;
    }
    float3 result = (1.0f - fresnel) * lobe * ((1.0f - layers.translucentShare) * inversePi * irradiance) * rectangle.radiance;
    if (frostfield_has_layers && layers.translucentShare > 0.0f) {
        // The panel's irradiance on the lit side scatters back; the other side's crosses the sheet or volume.
        float behind = rectangle.farField
            ? rectangle.solidAngle * saturate(-dot(n, unitOr(rectangle.toCenter, n)))
            : pbrRectangleIrradiance(rectangle.corners[0], rectangle.corners[1], rectangle.corners[2], rectangle.corners[3], -n);
        result += (1.0f - fresnel) * (layers.scatterColor * irradiance + layers.transmitColor * behind)
            * (layers.translucentShare * diffuseWeight * inversePi) * rectangle.radiance;
    }
    float noL = saturate(dot(n, l));
    if (rectangle.farField || (frostfield_has_layers && layers.anisotropy > 0.0f)) {
        // The isotropic fit does not model anisotropy. The small-source branch avoids cancellation
        // between nearly parallel polygon edges, retaining the native point limit.
        if (noL > 0.0f) {
            float2 ggx = pbrLobe(surface, layers, h, view, l, saturate(dot(n, h)), noV, noL);
            result += fresnel * (ggx.x * specular.w * ggx.y * noL * rectangle.solidAngle) * rectangle.radiance;
        }
    } else {
        float film = frostfield_has_layers ? layers.iridescence : 0.0f;
        result += rectangle.radiance * pbrRectangleLTC(
            rectangle, n, view, surface.roughness, mix(f0, iridescent, film), mix(float3(f90), iridescent, film),
            rectangleLTC
        );
    }
    if (frostfield_has_layers && any(layers.sheenColor > 0.0f)) {
        // The sheen lobe toward the representative point, weighted by the panel's irradiance.
        float noH = saturate(dot(n, h));
        result = result * frostfieldSheenScaling(layers.sheenColor, layers.sheenRoughness, noV, noL)
            + layers.sheenColor * (frostfieldCharlie(noH, layers.sheenRoughness) * frostfieldSheenVisibility(noL, noV)
                * irradiance) * rectangle.radiance;
    }
    return result;
}

struct PBRTransmission {
    float dielectricF0;
    float factor;
    float3 radiance;
    // KHR_materials_volume_scatter: light scattered toward the viewer inside the volume, per unit irradiance.
    float3 haze;
};


static float dielectricF0(float ior) {
    if (ior == 1.5f) { return 0.04f; }
    if (ior == 0.0f) { return 1.0f; }
    float ratio = (ior - 1.0f) / (ior + 1.0f);
    return ratio * ratio;
}

static PBRTransmission sampleTransmission(
    PBRSurface surface, PBRLayers layers, FrostfieldPBRVaryings input, float4 camera, PBRTexCoords coordinates,
    constant float4& optical, constant float4& attenuation, constant float4x4& clipFromWorld,
    constant float4x4& inverseTransposeModel, constant float4& backdropViewport,
    texture2d<float> factorTexture, sampler factorSampler,
    texture2d<float> thicknessTexture, sampler thicknessSampler, texture2d<float> backdrop
) {
    PBRTransmission result = {dielectricF0(optical.y), 0.0f, float3(0), float3(0)};
    if (frostfield_has_transmission) {
        result.factor = optical.x * surface.transmissionMask
            * saturate(factorTexture.sample(factorSampler, pbrUV(input, coordinates, 8)).r);
        float thickness = optical.z > 0.0f
            ? optical.z * saturate(thicknessTexture.sample(thicknessSampler, pbrUV(input, coordinates, 9)).g)
            : 0.0f;
        if (result.factor <= 0.0f || surface.metallic >= 1.0f || result.dielectricF0 >= 1.0f) {
            return result;
        }

        float3 view = pbrView(camera, input.worldPosition, surface.normal);
        float2 inverseSize = 1.0f / float2(backdrop.get_width(), backdrop.get_height());
        float2 uv = input.position.xy * inverseSize;
        float travel = 0.0f;
        // The baked thickness estimates the exit point. Only the opaque scene
        // snapshot is available, so offscreen exits fall back to the entry UV.
        if (thickness > 0.0f) {
            float3 ray = refract(-view, surface.normal, 1.0f / optical.y);
            float3 localRay = (transpose(inverseTransposeModel) * float4(ray, 0)).xyz;
            float rateScale = max(abs(localRay.x), max(abs(localRay.y), abs(localRay.z)));
            if (rateScale > 0.0f && isfinite(rateScale)) {
                float distance = (thickness / rateScale) / length(localRay / rateScale);
                if (distance > 0.0f && isfinite(distance)) {
                    travel = distance;
                    float4 exit = clipFromWorld * float4(input.worldPosition + ray * travel, 1);
                    if (exit.w > 0.0f && all(isfinite(exit))) {
                        float2 refracted = exit.xy / exit.w * float2(0.5f, -0.5f) + 0.5f;
                        if (all(refracted >= 0.0f) && all(refracted <= 1.0f)) {
                            uv = refracted * backdropViewport.xy + backdropViewport.zw;
                        }
                    }
                }
            }
        }

        float2 halfTexel = min(inverseSize, backdropViewport.xy) * 0.5f;
        uv = clamp(uv, backdropViewport.zw + halfTexel, backdropViewport.zw + backdropViewport.xy - halfTexel);
        float blur = surface.roughness * saturate(2.0f * (optical.y - 1.0f));
        float lod = blur * float(backdrop.get_num_mip_levels() - 1u);
        float3 scatterDepth = float3(0);
        if (frostfield_has_layers && any(layers.mediumScatter > 0.0f) && travel > 0.0f && optical.w > 0.0f) {
            // KHR_materials_volume_scatter: the scattered fraction of the extinction no longer arrives
            // from the backdrop directly. Blur the backdrop by the scattering optical depth.
            float3 extinction = -log(max(attenuation.rgb, float3(1e-8f))) / optical.w;
            scatterDepth = extinction * frostfieldSingleScatteringAlbedo(layers.mediumScatter) * travel;
            float depth = max(scatterDepth.x, max(scatterDepth.y, scatterDepth.z));
            lod = max(lod, saturate(depth * 0.25f) * float(backdrop.get_num_mip_levels() - 1u));
        }
        constexpr sampler backdropFilter(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
        float3 behind = max(backdrop.sample(backdropFilter, uv, level(lod)).rgb, float3(0));
        if (travel > 0.0f && optical.w > 0.0f) {
            float pathLength = min(travel / optical.w, 1e20f);
            behind *= pow(attenuation.rgb, float3(pathLength));
        }
        float edge = 1.0f - saturate(dot(surface.normal, view));
        float edge2 = edge * edge;
        float f0 = result.dielectricF0;
        float f90 = 1.0f;
        if (frostfield_has_layers) {
            float tint = max(layers.specularColor.x, max(layers.specularColor.y, layers.specularColor.z));
            f0 = min(f0 * tint, 1.0f) * layers.specularWeight;
            f90 = layers.specularWeight;
        }
        float fresnel = f0 + (f90 - f0) * edge2 * edge2 * edge;
        result.radiance = behind * surface.base.rgb * ((1.0f - fresnel) * (1.0f - surface.metallic) * result.factor);
        // Scattered light leaves toward the viewer with about the multiple-scattering color.
        result.haze = layers.mediumScatter * (1.0f - exp(-scatterDepth))
            * ((1.0f - fresnel) * (1.0f - surface.metallic) * result.factor);
    }
    return result;
}

static float3 frostfieldLightProbeIrradiance(constant FrostfieldLightProbe& probe, float3 normal) {
    constant float4* c = probe.coefficients;
    if (c[0].w != 1.0f) {
        return frostfieldAmbientIrradiance(probe);
    }
    float x = normal.x, y = normal.y, z = normal.z;
    float3 irradiance = c[0].rgb * 0.886227f;
    irradiance += 1.023328f * (c[1].rgb * y + c[2].rgb * z + c[3].rgb * x);
    irradiance += 0.858086f * (c[4].rgb * x * y + c[5].rgb * y * z + c[7].rgb * x * z);
    irradiance += c[6].rgb * (0.743125f * z * z - 0.247708f);
    irradiance += c[8].rgb * (0.429043f * (x * x - y * y));
    return max(irradiance, 0.0f) + frostfieldAmbientIrradiance(probe);
}

// All sixteen fragment sampler slots can already be active on a shadowed
// material. Filter the atlas via texel reads so it needs no sampler binding.
static float3 pbrReadLightMap(texture2d<float, access::read> map, float2 uv, uint mip) {
    int2 size = int2(map.get_width(mip), map.get_height(mip));
    float2 location = saturate(uv) * float2(size) - 0.5f;
    int2 corner = int2(floor(location));
    float2 weight = fract(location);
    uint2 lower = uint2(clamp(corner, int2(0), size - 1));
    uint2 upper = uint2(clamp(corner + 1, int2(0), size - 1));
    float3 a = map.read(lower, mip).rgb;
    float3 b = map.read(uint2(upper.x, lower.y), mip).rgb;
    float3 c = map.read(uint2(lower.x, upper.y), mip).rgb;
    float3 d = map.read(upper, mip).rgb;
    return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

static float3 pbrLightMapIrradiance(
    FrostfieldPBRVaryings input, PBRTexCoords coordinates, float intensity,
    constant FrostfieldPBRLayerMaps& maps
) {
    if (!frostfield_has_raster_maps || intensity <= 0.0f) return float3(0);
    auto map = maps.lightMap;
    float2 uv = pbrUV(input, coordinates, 21);
    uint lastMip = map.get_num_mip_levels() - 1u;
    float3 irradiance;
    if (lastMip == 0u) {
        irradiance = pbrReadLightMap(map, uv, 0u);
    } else {
        float2 pixel = uv * float2(map.get_width(), map.get_height());
        float2 dx = dfdx(pixel), dy = dfdy(pixel);
        float footprint = max(1.0f, max(dot(dx, dx), dot(dy, dy)));
        float lod = min(float(lastMip), 0.5f * log2(footprint));
        uint level = uint(lod);
        irradiance = pbrReadLightMap(map, uv, level);
        if (level < lastMip) {
            irradiance = mix(irradiance, pbrReadLightMap(map, uv, level + 1u), fract(lod));
        }
    }
    irradiance = max(select(float3(0), irradiance, isfinite(irradiance)), 0.0f);
    return min(irradiance * (intensity * 0.31830988618f), float3(1e30f));
}

static float3 pbrDiffuseIndirect(
    PBRLayers layers, PBRTransmission transmission, float3 base, float3 diffuseColor, float diffuseWeight,
    float noV, float3 f0, float3 iridescent, float3 irradiance, float3 behind, bool includeHaze = true
) {
    float3 diffuseFresnel = 1.0f - f0;
    if (frostfield_has_layers && layers.iridescence > 0.0f) {
        diffuseFresnel = 1.0f - mix(f0, iridescent, layers.iridescence);
    }
    float3 diffuseAlbedo = diffuseColor;
    if (frostfield_has_layers && layers.diffuseRoughness > 0.0f) {
        diffuseAlbedo = frostfieldEONAlbedo(base, layers.diffuseRoughness, noV) * diffuseWeight;
    }
    float3 diffuse = irradiance * diffuseFresnel * diffuseAlbedo * (1.0f - layers.translucentShare);
    if (frostfield_has_layers && layers.translucentShare > 0.0f) {
        diffuse += diffuseFresnel * (irradiance * layers.scatterColor + behind * layers.transmitColor)
            * (layers.translucentShare * diffuseWeight);
    }
    if (includeHaze && frostfield_has_layers && any(transmission.haze > 0.0f)) {
        diffuse += irradiance * transmission.haze;
    }
    return diffuse;
}

static float4 shadePBR(
    PBRSurface surface, PBRClearcoat coat, PBRTransmission transmission, PBRLayers layers,
    float3 position, float4 light, float ambientIntensity,
    float4 camera, float3 visibility,
    constant FrostfieldEnvironmentLookup& environment, texturecube<float> diffuseEnvironment,
    texturecube<float> specularEnvironment, texture2d<float> environmentBRDF,
    constant FrostfieldPunctualLight* punctualLights, uint punctualLightCount,
    float4 relativeReceiver, constant float4* relativeLightPositions,
    float3 geometricNormal, depth2d_array<float> punctualShadowMaps,
    texture2d_array<float, access::read> spotShadowMoments, constant float4x4* punctualShadowViews,
    texture2d_array<float> lightProfiles, texture2d_array<float> lightCookies, texture2d_array<float> rectangleLTC,
    constant FrostfieldLightProbe& probe, float3 bakedIrradiance, float4 screenOcclusion, device const float4* probeGrids
) {
    constexpr float maxHalf = 65504.0f;
    // Screen-space ambient occlusion: the world-space bend from the normal it measured to
    // its bent normal (xyz), and cosine-weighted visibility (w). It occludes only image-based
    // and ambient light; w < 0 disables it.
    bool screenOccluded = screenOcclusion.w >= 0.0f;
    float3 base = surface.base.rgb;
    float opacity = surface.base.a;
    float metallic = surface.metallic;
    // KHR_materials_specular scales dielectric F0 and F90; metals keep F90 = 1.
    float3 dielectric = float3(transmission.dielectricF0);
    float f90 = 1.0f;
    if (frostfield_has_layers) {
        dielectric = min(dielectric * layers.specularColor, 1.0f) * layers.specularWeight;
        f90 = mix(layers.specularWeight, 1.0f, metallic);
    }
    float3 f0 = mix(dielectric, base, metallic);
    // The diffuse layer's weight beneath metal and specular transmission.
    float diffuseWeight = transmission.dielectricF0 < 1.0f ? (1.0f - metallic) * (1.0f - transmission.factor) : 0.0f;
    float3 diffuseColor = base * diffuseWeight;
    float3 n = surface.normal;
    float3 v = pbrRelativeView(camera, position, n, relativeReceiver);
    float noV = saturate(dot(n, v));
    // The bend applies to this fragment's shading normal, so normal maps survive. Material
    // occlusion darkens indirect light as without screen occlusion; screen visibility below
    // it darkens further, relative to it, so visibility one leaves shading unchanged.
    float3 bentNormal = n;
    float screenVisibility = 1.0f;
    if (screenOccluded) {
        if (any(screenOcclusion.xyz != 0.0f)) {
            bentNormal = unitOr(n + screenOcclusion.xyz, n);
        }
        screenVisibility = surface.occlusion > 0.0f ? min(1.0f, screenOcclusion.w / surface.occlusion) : 1.0f;
    }
    // The thin-film reflectance is evaluated once at the view angle.
    float3 iridescent = f0;
    if (frostfield_has_layers && layers.iridescence > 0.0f) {
        iridescent = frostfieldIridescence(1.0f, layers.filmIOR, noV, layers.thickness, f0);
    }

    float coatNoV = 0.0f;
    float coatFresnel = 0.04f;
    float coatWeight = 0.0f;
    if (frostfield_has_clearcoat) {
        coatNoV = saturate(abs(dot(coat.normal, v)));
        float edge = 1.0f - coatNoV;
        float edge2 = edge * edge;
        // IOR 1.5 has F0 = 0.04. View-angle Fresnel weights the entire coat layer.
        coatFresnel = 0.04f + 0.96f * edge2 * edge2 * edge;
        coatWeight = coat.factor * coatFresnel;
    }

    float3 direct = float3(0);
    float3 coatRadiance = float3(0);
    if (light.w > 0.0f) {
        float3 incident = visibility * min(light.w, maxHalf);
        direct = pbrDirect(surface, layers, v, noV, f0, f90, iridescent, diffuseColor, light.xyz, incident, diffuseWeight);
        if (frostfield_has_clearcoat && coatWeight > 0.0f) {
            coatRadiance = pbrClearcoatDirect(coat, v, light.xyz, incident);
        }
    }
    for (uint index = 0; index < punctualLightCount; ++index) {
        constant FrostfieldPunctualLight& record = punctualLights[index];
        float4 lightPosition = relativeReceiver.w != 0.0f ? relativeLightPositions[index] : float4(0);
        if (record.positionKind.w == 3.0f) {
            float3 toCenter = lightPosition.w != 0.0f ? lightPosition.xyz - relativeReceiver.xyz
                                                       : record.positionKind.xyz - position;
            PBRRectangleLight rectangle;
            if (!pbrRectangle(toCenter, record, rectangle)) {
                continue;
            }
            float shadow = pbrPunctualShadow(
                position, geometricNormal, record, punctualShadowMaps, spotShadowMoments,
                punctualShadowViews, relativeReceiver, lightPosition
            );
            if (shadow <= 0.0f) {
                continue;
            }
            float4 specular = pbrRectangleSpecular(rectangle, record, n, v, surface.roughness);
            direct += shadow * pbrRectangleDirect(
                surface, layers, rectangle, specular, v, noV, f0, f90, iridescent, diffuseColor, diffuseWeight, rectangleLTC
            );
            if (frostfield_has_clearcoat && coatWeight > 0.0f) {
                if (rectangle.farField) {
                    float4 coatSpecular = pbrRectangleSpecular(rectangle, record, coat.normal, v, coat.roughness);
                    coatRadiance += pbrClearcoatDirect(coat, v, coatSpecular.xyz, rectangle.radiance * rectangle.solidAngle)
                        * (shadow * coatSpecular.w);
                } else {
                    // The final layer mix applies view Fresnel. Remove that factor from the already
                    // Fresnel-integrated LTC lobe so only the clearcoat factor scales its radiance.
                    coatRadiance += (shadow / coatFresnel) * rectangle.radiance * pbrRectangleLTC(
                        rectangle, coat.normal, v, coat.roughness, float3(0.04f), float3(1), rectangleLTC
                    );
                }
            }
            continue;
        }
        PBRIncidentLight incident = pbrPunctual(position, record, lightProfiles, lightCookies, relativeReceiver, lightPosition);
        if (record.shadow.x >= 0.0f && any(incident.radiance > 0.0f)) {
            incident.radiance *= pbrPunctualShadow(
                position, geometricNormal, record, punctualShadowMaps, spotShadowMoments, punctualShadowViews, relativeReceiver, lightPosition
            );
        }
        direct += pbrDirect(
            surface, layers, v, noV, f0, f90, iridescent, diffuseColor, incident.direction, incident.radiance, diffuseWeight
        );
        if (frostfield_has_clearcoat && coatWeight > 0.0f) {
            coatRadiance += pbrClearcoatDirect(coat, v, incident.direction, incident.radiance);
        }
    }

    float3 indirect;
    if (frostfield_has_environment) {
        constexpr sampler cubeFilter(filter::linear, mip_filter::linear);
        constexpr sampler lutFilter(coord::normalized, address::clamp_to_edge, filter::linear);
        float3 reflectionNormal = n;
        if (frostfield_has_layers && layers.anisotropy > 0.0f) {
            // Bend the reflection normal across the anisotropy direction (glTF Sample Viewer).
            float3 across = cross(layers.bitangent, v);
            float3 anisotropicNormal = cross(across, layers.bitangent);
            float bend = 1.0f - layers.anisotropy * (1.0f - surface.roughness);
            bend *= bend;
            reflectionNormal = unitOr(mix(anisotropicNormal, n, bend * bend), n);
        }
        float3 reflection = reflect(-v, reflectionNormal);
        // Unoccluded light arrives around the bent normal.
        float3 irradiance = diffuseEnvironment.sample(cubeFilter, environmentDirection(bentNormal, environment)).rgb;
        float3 radiance = specularEnvironment.sample(
            cubeFilter, environmentReflectionDirection(reflection, position, environment),
            level(surface.roughness * environment.y.w)
        ).rgb;
        if (layers.retroreflectivity > 0.0f) {
            float roughness2 = surface.roughness * surface.roughness;
            float3 retroDirection = unitOr(mix(v, reflectionNormal, roughness2 * roughness2), reflectionNormal);
            float3 retroRadiance = specularEnvironment.sample(
                cubeFilter, environmentDirection(retroDirection, environment), level(surface.roughness * environment.y.w)
            ).rgb;
            radiance = mix(radiance, retroRadiance, layers.retroreflectivity);
        }
        float2 dfg = environmentBRDF.sample(lutFilter, float2(noV, surface.roughness)).rg;
        float3 specularFresnel = f0 * dfg.x + f90 * dfg.y;
        if (frostfield_has_layers && layers.iridescence > 0.0f) {
            specularFresnel = mix(specularFresnel, iridescent * (dfg.x + dfg.y), layers.iridescence);
        }
        float3 specular = radiance * specularFresnel;
        float3 behind = frostfield_has_layers && layers.translucentShare > 0.0f && any(layers.transmitColor > 0.0f)
            ? diffuseEnvironment.sample(cubeFilter, environmentDirection(-n, environment)).rgb : float3(0);
        float3 diffuse = pbrDiffuseIndirect(
            layers, transmission, base, diffuseColor, diffuseWeight, noV, f0, iridescent, irradiance, behind
        );
        float ambient = min(environment.x.w, maxHalf);
        // Multiple-bounce diffuse occlusion and cone-traced specular occlusion.
        float3 diffuseOcclusion = float3(1);
        float3 specularOcclusion = float3(1);
        if (screenOccluded) {
            diffuseOcclusion = frostfieldMultipleBounce(screenVisibility, diffuseColor);
            specularOcclusion = frostfieldMultipleBounce(
                frostfieldSpecularOcclusion(bentNormal, n, screenVisibility, reflection, surface.roughness), f0
            );
        }
        float3 reflected = diffuse * diffuseOcclusion + specular * specularOcclusion;
        if (frostfield_has_layers && any(layers.sheenColor > 0.0f)) {
            // Approximate image-lit sheen as diffuse irradiance times the lobe's albedo.
            float albedo = frostfieldSheenAlbedo(noV, layers.sheenRoughness);
            float peak = max(layers.sheenColor.x, max(layers.sheenColor.y, layers.sheenColor.z));
            reflected = reflected * max(0.0f, 1.0f - peak * albedo)
                + irradiance * layers.sheenColor * albedo * diffuseOcclusion;
        }
        indirect = reflected * ambient * surface.occlusion;
        if (frostfield_has_clearcoat && coatWeight > 0.0f) {
            float3 reflection = reflect(-v, coat.normal);
            float3 radiance = specularEnvironment.sample(
                cubeFilter, environmentReflectionDirection(reflection, position, environment), level(coat.roughness * environment.y.w)
            ).rgb;
            float2 dfg = environmentBRDF.sample(lutFilter, float2(coatNoV, coat.roughness)).rg;
            // A+B integrates unit Fresnel. The view-angle coat factor is applied below.
            float coatOcclusion = surface.occlusion;
            if (screenOccluded) {
                coatOcclusion *= frostfieldSpecularOcclusion(bentNormal, coat.normal, screenVisibility, reflection, coat.roughness);
            }
            coatRadiance += radiance * ((dfg.x + dfg.y) * ambient * coatOcclusion);
        }
    } else {
        float3 diffuseFill = diffuseColor * (1.0f - layers.translucentShare);
        if (frostfield_has_layers) {
            if (layers.diffuseRoughness > 0.0f) {
                diffuseFill = frostfieldEONAlbedo(base, layers.diffuseRoughness, noV) * (diffuseWeight * (1.0f - layers.translucentShare));
            }
            // Uniform fill reaches both sides of a translucent surface.
            diffuseFill += (layers.scatterColor + layers.transmitColor) * (layers.translucentShare * diffuseWeight) + transmission.haze;
        }
        float3 diffuseOcclusion = float3(1);
        float3 specularOcclusion = float3(1);
        if (screenOccluded) {
            diffuseOcclusion = frostfieldMultipleBounce(screenVisibility, diffuseFill);
            specularOcclusion = frostfieldMultipleBounce(
                frostfieldSpecularOcclusion(bentNormal, n, screenVisibility, reflect(-v, n), surface.roughness), f0
            );
        }
        float3 fill = (1.0f - f0) * diffuseFill * diffuseOcclusion + f0 * specularOcclusion;
        if (frostfield_has_layers && any(layers.sheenColor > 0.0f)) {
            float albedo = frostfieldSheenAlbedo(noV, layers.sheenRoughness);
            float peak = max(layers.sheenColor.x, max(layers.sheenColor.y, layers.sheenColor.z));
            fill = fill * max(0.0f, 1.0f - peak * albedo) + layers.sheenColor * albedo * diffuseOcclusion;
        }
        indirect = fill * min(ambientIntensity, maxHalf) * surface.occlusion;
    }
    if (frostfield_has_raster_maps && any(bakedIrradiance > 0.0f)) {
        // The map already integrated incident visibility and cosine. Apply the
        // diffuse response once, independently of ambient and analytic lights.
        indirect += pbrDiffuseIndirect(
            layers, transmission, base, diffuseColor, diffuseWeight, noV, f0, iridescent,
            bakedIrradiance, float3(0), false
        ) * surface.occlusion;
    }
    uint probeLayers = frostfieldProbeLayers(probe);
    if (probe.coefficients[0].w != 0.0f || probeLayers != 0) {
        constexpr float inversePi = 0.31830988618f;
        // The split-sum cube's diffuse texture already includes 1/π; the SH coefficients do not.
        float3 gridIrradiance = frostfieldProbeGridIrradiance(probeGrids, probeLayers, position, n);
        float3 irradiance = (frostfieldLightProbeIrradiance(probe, bentNormal) + gridIrradiance) * inversePi;
        float3 behind = frostfield_has_layers && layers.translucentShare > 0.0f && any(layers.transmitColor > 0.0f)
            ? (frostfieldLightProbeIrradiance(probe, -n)
               + frostfieldProbeGridIrradiance(probeGrids, probeLayers, position, -n)) * inversePi : float3(0);
        float3 diffuse = pbrDiffuseIndirect(
            layers, transmission, base, diffuseColor, diffuseWeight, noV, f0, iridescent, irradiance, behind
        );
        if (frostfield_has_layers && any(layers.sheenColor > 0.0f)) {
            float albedo = frostfieldSheenAlbedo(noV, layers.sheenRoughness);
            float peak = max(layers.sheenColor.x, max(layers.sheenColor.y, layers.sheenColor.z));
            diffuse = diffuse * max(0.0f, 1.0f - peak * albedo) + irradiance * layers.sheenColor * albedo;
        }
        if (screenOccluded) {
            diffuse *= frostfieldMultipleBounce(screenVisibility, diffuseColor);
        }
        indirect += diffuse * surface.occlusion;
    }
    float3 color = direct + indirect + surface.emission + transmission.radiance;
    if (frostfield_has_clearcoat) {
        color = mix(color, coatRadiance, coatWeight);
    }
    if (frostfield_alpha_to_coverage) {
        // Coverage carries alpha, so the resolve blends straight color.
        return float4(min(color, float3(maxHalf)), opacity);
    }
    return float4(min(color, float3(maxHalf)) * opacity, opacity);
}

static float4 pbrUnlit(PBRSurface surface, bool useGraphEmission) {
    constexpr float maxHalf = 65504.0f;
    float opacity = surface.base.a;
    float3 color = min(useGraphEmission ? surface.emission : surface.base.rgb, float3(maxHalf));
    if (frostfield_alpha_to_coverage) {
        return float4(color, opacity);
    }
    return float4(color * opacity, opacity);
}

#ifndef FROSTFIELD_PBR_HELPERS_ONLY

#if defined(FROSTFIELD_PBR_LOG_DEPTH)
#define FROSTFIELD_PBR_COLOR_OUTPUT FrostfieldLogColor
#define FROSTFIELD_PBR_COLOR_RESULT(color, surface) FrostfieldLogColor{(color), frostfieldLogDepth(input.position, logDepthRange)}
#define FROSTFIELD_PBR_ATTRIBUTE_OUTPUT FrostfieldLogAttributes
#define FROSTFIELD_PBR_ATTRIBUTE_RESULT(attributes) FrostfieldLogAttributes{(attributes), frostfieldLogDepth(input.position, logDepthRange)}
#define FROSTFIELD_PBR_DEPTH_INPUT FROSTFIELD_LOG_DEPTH_INPUT
#else
#ifndef FROSTFIELD_PBR_COLOR_OUTPUT
#define FROSTFIELD_PBR_COLOR_OUTPUT float4
#define FROSTFIELD_PBR_COLOR_RESULT(color, surface) (color)
#endif
#define FROSTFIELD_PBR_ATTRIBUTE_OUTPUT uint2
#define FROSTFIELD_PBR_ATTRIBUTE_RESULT(attributes) (attributes)
#define FROSTFIELD_PBR_DEPTH_INPUT
#endif

struct FrostfieldPBRScreenInputs {
    texture2d<float> ambient [[id(0)]];
    texture2d<float, access::read> directional [[id(1)]];
    float4 info [[id(2)]];
};

static inline float pbrScreenDirectionalVisibility(texture2d<float, access::read> map, float4 info, float2 pixel) {
    if (info.y == 0.0f) { return 1.0f; }
    int2 size(map.get_width(), map.get_height());
    if (all(float2(size) == info.zw)) { return saturate(map.read(uint2(pixel)).r); }
    // The shadowed PBR pipeline already uses all 16 sampler slots.
    float2 texel = pixel * float2(size) / info.zw - 0.5f;
    int2 base = int2(floor(texel));
    int2 lower = clamp(base, int2(0), size - 1);
    int2 upper = clamp(base + 1, int2(0), size - 1);
    float2 fraction = fract(texel);
    float top = mix(map.read(uint2(lower)).r, map.read(uint2(upper.x, lower.y)).r, fraction.x);
    float bottom = mix(map.read(uint2(lower.x, upper.y)).r, map.read(uint2(upper)).r, fraction.x);
    return saturate(mix(top, bottom, fraction.y));
}

fragment FROSTFIELD_PBR_COLOR_OUTPUT frostfield_mesh_pbr(
    FrostfieldPBRVaryings input [[stage_in]],
    bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]],
    constant float4& light [[buffer(1)]],
    constant float4& settings [[buffer(2)]],
    constant float4& camera [[buffer(3)]],
    constant float4& controls [[buffer(6)]],
    constant float4& emission [[buffer(7)]],
    constant float4& baseColorFactor [[buffer(8)]],
    constant uint& coordinates [[buffer(9)]],
    constant FrostfieldEnvironmentLookup& environment [[buffer(10), function_constant(frostfield_has_environment)]],
    constant FrostfieldPunctualLight* punctualLights [[buffer(11)]],
    constant uint& punctualLightCount [[buffer(12)]],
    constant float4& clearcoat [[buffer(13), function_constant(frostfield_has_clearcoat)]],
    constant float4& optical [[buffer(14)]],
    constant float4& attenuation [[buffer(15), function_constant(frostfield_has_transmission)]],
    constant float4x4& transmissionClip [[buffer(16), function_constant(frostfield_has_transmission)]],
    constant float4x4& inverseTransposeModel [[buffer(17), function_constant(frostfield_has_transmission)]],
    constant float4& backdropViewport [[buffer(18), function_constant(frostfield_has_transmission)]],
    constant FrostfieldPBRExtensions& extensions [[buffer(21)]],
    constant FrostfieldPBRLayerMaps& layerMaps [[buffer(22), function_constant(frostfield_has_raster_maps)]],
    array<sampler, 6> layerSamplers [[sampler(10), function_constant(frostfield_has_layers)]],
    texture2d<float> colorTexture [[texture(0)]], sampler colorSampler [[sampler(0)]],
    texture2d<float> metalRoughTexture [[texture(2)]], sampler metalRoughSampler [[sampler(1)]],
    texture2d<float> normalTexture [[texture(3)]], sampler normalSampler [[sampler(2)]],
    texture2d<float> occlusionTexture [[texture(4)]], sampler occlusionSampler [[sampler(3)]],
    texture2d<float> emissiveTexture [[texture(5)]], sampler emissiveSampler [[sampler(4)]],
    texturecube<float> diffuseEnvironment [[texture(6), function_constant(frostfield_has_environment)]],
    texturecube<float> specularEnvironment [[texture(7), function_constant(frostfield_has_environment)]],
    texture2d<float> environmentBRDF [[texture(8), function_constant(frostfield_has_environment)]],
    texture2d<float> coatFactorTexture [[texture(9), function_constant(frostfield_has_clearcoat)]],
    sampler coatFactorSampler [[sampler(5), function_constant(frostfield_has_clearcoat)]],
    texture2d<float> coatRoughnessTexture [[texture(10), function_constant(frostfield_has_clearcoat)]],
    sampler coatRoughnessSampler [[sampler(6), function_constant(frostfield_has_clearcoat)]],
    texture2d<float> coatNormalTexture [[texture(11), function_constant(frostfield_has_clearcoat)]],
    sampler coatNormalSampler [[sampler(7), function_constant(frostfield_has_clearcoat)]],
    texture2d<float> transmissionFactorTexture [[texture(12), function_constant(frostfield_has_transmission)]],
    texture2d<float> opaqueBackdrop [[texture(13), function_constant(frostfield_has_transmission)]],
    texture2d<float> thicknessTexture [[texture(14), function_constant(frostfield_has_transmission)]],
    sampler transmissionFactorSampler [[sampler(8), function_constant(frostfield_has_transmission)]],
    sampler thicknessSampler [[sampler(9), function_constant(frostfield_has_transmission)]],
    constant float4x4* punctualShadowViews [[buffer(20)]],
    depth2d_array<float> punctualShadowMaps [[texture(24)]],
    texture2d_array<float, access::read> spotShadowMoments [[texture(29)]],
    texture2d<float> alphaMapTexture [[texture(15), function_constant(frostfield_has_alpha_map)]],
    constant float4* alphaMapInfo [[buffer(23), function_constant(frostfield_has_alpha_map)]],
    texture2d_array<float> lightProfiles [[texture(25)]],
    texture2d_array<float> lightCookies [[texture(26)]],
    texture2d_array<float> rectangleLTC [[texture(30)]],
    constant FrostfieldPBRScreenInputs& screenInputs [[buffer(25)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]],
    device const float4* probeGrids [[buffer(5)]],
    constant float4x4& relativeLightModel [[buffer(27)]],
    constant float4* relativeLightPositions [[buffer(28)]]
#if defined(FROSTFIELD_GENERATED_SURFACE)
    , constant float4* graphUniforms [[buffer(19)]]
    , array<texture2d<float>, 8> graphTextures [[texture(16)]]
    , constant FrostfieldGraphView& graphView [[buffer(24)]]
    FROSTFIELD_GRAPH_VIEWPORT_ARG
#endif
    FROSTFIELD_PBR_DEPTH_INPUT
) {
#if defined(FROSTFIELD_GENERATED_SURFACE)
    FrostfieldGraphResources graphResources = {
        graphUniforms, graphTextures[0], graphTextures[1], graphTextures[2], graphTextures[3],
        graphTextures[4], graphTextures[5], graphTextures[6], graphTextures[7], &graphView
        FROSTFIELD_GRAPH_VIEWPORT_VALUE
    };
#endif
    PBRTexCoords texCoords = {coordinates, extensions.transforms};
    PBRSurface surface = samplePBR(
        input, front, parameters, camera, baseColorFactor, settings, controls, emission, texCoords, colorTexture, colorSampler,
        metalRoughTexture, metalRoughSampler, normalTexture, normalSampler,
        occlusionTexture, occlusionSampler, emissiveTexture, emissiveSampler, alphaMapTexture, alphaMapInfo
#if defined(FROSTFIELD_GENERATED_SURFACE)
        , graphResources
#endif
    );
    if (extensions.options.x != 0.0f) {
        return FROSTFIELD_PBR_COLOR_RESULT(pbrUnlit(surface, extensions.options.x == 2.0f), surface);
    }
    PBRClearcoat coat = sampleClearcoat(
        input, front, texCoords, clearcoat, coatFactorTexture, coatFactorSampler,
        coatRoughnessTexture, coatRoughnessSampler, coatNormalTexture, coatNormalSampler
    );
    PBRLayers layers = sampleLayers(input, texCoords, surface, extensions, layerMaps, layerSamplers);
    PBRTransmission transmission = sampleTransmission(
        surface, layers, input, camera, texCoords, optical, attenuation, transmissionClip, inverseTransposeModel,
        backdropViewport, transmissionFactorTexture, transmissionFactorSampler, thicknessTexture, thicknessSampler,
        opaqueBackdrop
    );
    float3 geometricNormal = unitOr(input.normal, float3(0, 0, 1)) * (front ? 1.0f : -1.0f);
    float3 visibility = float3(pbrScreenDirectionalVisibility(screenInputs.directional, screenInputs.info, input.position.xy));
    float4 color = shadePBR(
        surface, coat, transmission, layers, input.worldPosition, light, settings.z, camera, float3(visibility),
        environment, diffuseEnvironment, specularEnvironment, environmentBRDF, punctualLights, punctualLightCount,
        pbrRelativeReceiver(input.shadowLocalPosition, relativeLightModel), relativeLightPositions,
        geometricNormal, punctualShadowMaps, spotShadowMoments, punctualShadowViews, lightProfiles, lightCookies,
        rectangleLTC, probe, pbrLightMapIrradiance(input, texCoords, extensions.options.y, layerMaps),
        frostfieldScreenOcclusion(screenInputs.ambient, screenInputs.info, input.position.xy), probeGrids
    );
    return FROSTFIELD_PBR_COLOR_RESULT(color, surface);
}

#if defined(FROSTFIELD_GRAPH_DIRECTIONAL_SHADOW)
static inline float pbrDirectionalVisibility(
    FrostfieldPBRVaryings input, float3 geometricNormal, float3 directionToLight,
    depth2d_array<float> shadowMap, texture2d_array<float, access::read> shadowMoments,
    constant FrostfieldDirectionalShadow& directionalShadow
) {
    float facing = saturate(dot(geometricNormal, directionToLight));
    return frostfieldShadowVisibility(
        shadowMap, shadowMoments, input.worldPosition, input.shadowLocalPosition, geometricNormal, facing, directionalShadow
    );
}
#endif

fragment FROSTFIELD_PBR_COLOR_OUTPUT frostfield_mesh_pbr_shadowed(
    FrostfieldPBRVaryings input [[stage_in]],
    bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]],
    constant float4& light [[buffer(1)]],
    constant float4& settings [[buffer(2)]],
    constant float4& camera [[buffer(3)]],
    constant FrostfieldDirectionalShadow& directionalShadow [[buffer(4)]],
    constant float4& controls [[buffer(6)]],
    constant float4& emission [[buffer(7)]],
    constant float4& baseColorFactor [[buffer(8)]],
    constant uint& coordinates [[buffer(9)]],
    constant FrostfieldEnvironmentLookup& environment [[buffer(10), function_constant(frostfield_has_environment)]],
    constant FrostfieldPunctualLight* punctualLights [[buffer(11)]],
    constant uint& punctualLightCount [[buffer(12)]],
    constant float4& clearcoat [[buffer(13), function_constant(frostfield_has_clearcoat)]],
    constant float4& optical [[buffer(14)]],
    constant float4& attenuation [[buffer(15), function_constant(frostfield_has_transmission)]],
    constant float4x4& transmissionClip [[buffer(16), function_constant(frostfield_has_transmission)]],
    constant float4x4& inverseTransposeModel [[buffer(17), function_constant(frostfield_has_transmission)]],
    constant float4& backdropViewport [[buffer(18), function_constant(frostfield_has_transmission)]],
    constant FrostfieldPBRExtensions& extensions [[buffer(21)]],
    constant FrostfieldPBRLayerMaps& layerMaps [[buffer(22), function_constant(frostfield_has_raster_maps)]],
    array<sampler, 6> layerSamplers [[sampler(10), function_constant(frostfield_has_layers)]],
    texture2d<float> colorTexture [[texture(0)]], sampler colorSampler [[sampler(0)]],
    depth2d_array<float> shadowMap [[texture(1)]],
    texture2d_array<float, access::read> shadowMoments [[texture(28)]],
    texture2d_array<float, access::read> directionalTransmission [[texture(27)]],
    texture2d<float> metalRoughTexture [[texture(2)]], sampler metalRoughSampler [[sampler(1)]],
    texture2d<float> normalTexture [[texture(3)]], sampler normalSampler [[sampler(2)]],
    texture2d<float> occlusionTexture [[texture(4)]], sampler occlusionSampler [[sampler(3)]],
    texture2d<float> emissiveTexture [[texture(5)]], sampler emissiveSampler [[sampler(4)]],
    texturecube<float> diffuseEnvironment [[texture(6), function_constant(frostfield_has_environment)]],
    texturecube<float> specularEnvironment [[texture(7), function_constant(frostfield_has_environment)]],
    texture2d<float> environmentBRDF [[texture(8), function_constant(frostfield_has_environment)]],
    texture2d<float> coatFactorTexture [[texture(9), function_constant(frostfield_has_clearcoat)]],
    sampler coatFactorSampler [[sampler(5), function_constant(frostfield_has_clearcoat)]],
    texture2d<float> coatRoughnessTexture [[texture(10), function_constant(frostfield_has_clearcoat)]],
    sampler coatRoughnessSampler [[sampler(6), function_constant(frostfield_has_clearcoat)]],
    texture2d<float> coatNormalTexture [[texture(11), function_constant(frostfield_has_clearcoat)]],
    sampler coatNormalSampler [[sampler(7), function_constant(frostfield_has_clearcoat)]],
    texture2d<float> transmissionFactorTexture [[texture(12), function_constant(frostfield_has_transmission)]],
    texture2d<float> opaqueBackdrop [[texture(13), function_constant(frostfield_has_transmission)]],
    texture2d<float> thicknessTexture [[texture(14), function_constant(frostfield_has_transmission)]],
    sampler transmissionFactorSampler [[sampler(8), function_constant(frostfield_has_transmission)]],
    sampler thicknessSampler [[sampler(9), function_constant(frostfield_has_transmission)]],
    constant float4x4* punctualShadowViews [[buffer(20)]],
    depth2d_array<float> punctualShadowMaps [[texture(24)]],
    texture2d_array<float, access::read> spotShadowMoments [[texture(29)]],
    texture2d<float> alphaMapTexture [[texture(15), function_constant(frostfield_has_alpha_map)]],
    constant float4* alphaMapInfo [[buffer(23), function_constant(frostfield_has_alpha_map)]],
    texture2d_array<float> lightProfiles [[texture(25)]],
    texture2d_array<float> lightCookies [[texture(26)]],
    texture2d_array<float> rectangleLTC [[texture(30)]],
    constant FrostfieldPBRScreenInputs& screenInputs [[buffer(25)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]],
    device const float4* probeGrids [[buffer(5)]],
    constant float4x4& relativeLightModel [[buffer(27)]],
    constant float4* relativeLightPositions [[buffer(28)]]
#if defined(FROSTFIELD_GENERATED_SURFACE)
    , constant float4* graphUniforms [[buffer(19)]]
    , array<texture2d<float>, 8> graphTextures [[texture(16)]]
    , constant FrostfieldGraphView& graphView [[buffer(24)]]
    FROSTFIELD_GRAPH_VIEWPORT_ARG
#endif
    FROSTFIELD_PBR_DEPTH_INPUT
) {
#if defined(FROSTFIELD_GENERATED_SURFACE)
    FrostfieldGraphResources graphResources = {
        graphUniforms, graphTextures[0], graphTextures[1], graphTextures[2], graphTextures[3],
        graphTextures[4], graphTextures[5], graphTextures[6], graphTextures[7], &graphView
        FROSTFIELD_GRAPH_VIEWPORT_VALUE
    };
#endif
#if defined(FROSTFIELD_GRAPH_DIRECTIONAL_SHADOW)
    float3 geometricNormal = unitOr(input.normal, float3(0, 0, 1)) * (front ? 1.0f : -1.0f);
    float graphVisibility = pbrDirectionalVisibility(input, geometricNormal, light.xyz, shadowMap, shadowMoments, directionalShadow)
        * pbrScreenDirectionalVisibility(screenInputs.directional, screenInputs.info, input.position.xy);
#endif
    PBRTexCoords texCoords = {coordinates, extensions.transforms};
    PBRSurface surface = samplePBR(
        input, front, parameters, camera, baseColorFactor, settings, controls, emission, texCoords, colorTexture, colorSampler,
        metalRoughTexture, metalRoughSampler, normalTexture, normalSampler,
        occlusionTexture, occlusionSampler, emissiveTexture, emissiveSampler, alphaMapTexture, alphaMapInfo
#if defined(FROSTFIELD_GENERATED_SURFACE)
        , graphResources
#if defined(FROSTFIELD_GRAPH_DIRECTIONAL_SHADOW)
        , false, graphVisibility
#endif
#endif
    );
    if (extensions.options.x != 0.0f) {
        return FROSTFIELD_PBR_COLOR_RESULT(pbrUnlit(surface, extensions.options.x == 2.0f), surface);
    }
#if !defined(FROSTFIELD_GRAPH_DIRECTIONAL_SHADOW)
    float3 geometricNormal = unitOr(input.normal, float3(0, 0, 1)) * (front ? 1.0f : -1.0f);
#endif
    float facing = saturate(dot(geometricNormal, light.xyz));
    float3 visibility = frostfieldTransmittedShadowVisibility(
        shadowMap, shadowMoments, directionalTransmission, input.worldPosition, input.shadowLocalPosition,
        geometricNormal, facing, directionalShadow
    ) * pbrScreenDirectionalVisibility(screenInputs.directional, screenInputs.info, input.position.xy);
    PBRClearcoat coat = sampleClearcoat(
        input, front, texCoords, clearcoat, coatFactorTexture, coatFactorSampler,
        coatRoughnessTexture, coatRoughnessSampler, coatNormalTexture, coatNormalSampler
    );
    PBRLayers layers = sampleLayers(input, texCoords, surface, extensions, layerMaps, layerSamplers);
    PBRTransmission transmission = sampleTransmission(
        surface, layers, input, camera, texCoords, optical, attenuation, transmissionClip, inverseTransposeModel,
        backdropViewport, transmissionFactorTexture, transmissionFactorSampler, thicknessTexture, thicknessSampler,
        opaqueBackdrop
    );
    float4 color = shadePBR(
        surface, coat, transmission, layers, input.worldPosition, light, settings.z, camera,
        visibility * directionalShadow.lightColor.rgb,
        environment, diffuseEnvironment, specularEnvironment, environmentBRDF, punctualLights, punctualLightCount,
        pbrRelativeReceiver(input.shadowLocalPosition, relativeLightModel), relativeLightPositions,
        geometricNormal, punctualShadowMaps, spotShadowMoments, punctualShadowViews, lightProfiles, lightCookies,
        rectangleLTC, probe, pbrLightMapIrradiance(input, texCoords, extensions.options.y, layerMaps),
        frostfieldScreenOcclusion(screenInputs.ambient, screenInputs.info, input.position.xy), probeGrids
    );
    return FROSTFIELD_PBR_COLOR_RESULT(color, surface);
}

// The surface attribute prepass (SceneGraph.encodeSurfaceAttributes): the shading normal,
// perceptual roughness and split-sum specular reflectance at the view angle, which
// screen-space occlusion and reflections read. Masked materials discard as in color.
fragment FROSTFIELD_PBR_ATTRIBUTE_OUTPUT frostfield_mesh_pbr_attributes(
    FrostfieldPBRVaryings input [[stage_in]],
    bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]],
    constant float4& settings [[buffer(2)]],
    constant float4& camera [[buffer(3)]],
    constant float4& controls [[buffer(6)]],
    constant float4& emission [[buffer(7)]],
    constant float4& baseColorFactor [[buffer(8)]],
    constant uint& coordinates [[buffer(9)]],
    constant float4& optical [[buffer(14)]],
    constant FrostfieldPBRExtensions& extensions [[buffer(21)]],
    constant FrostfieldPBRLayerMaps& layerMaps [[buffer(22), function_constant(frostfield_has_layers)]],
    array<sampler, 6> layerSamplers [[sampler(10), function_constant(frostfield_has_layers)]],
    texture2d<float> colorTexture [[texture(0)]], sampler colorSampler [[sampler(0)]],
    texture2d<float> metalRoughTexture [[texture(2)]], sampler metalRoughSampler [[sampler(1)]],
    texture2d<float> normalTexture [[texture(3)]], sampler normalSampler [[sampler(2)]],
    texture2d<float> occlusionTexture [[texture(4)]], sampler occlusionSampler [[sampler(3)]],
    texture2d<float> emissiveTexture [[texture(5)]], sampler emissiveSampler [[sampler(4)]],
    texture2d<float> environmentBRDF [[texture(8), function_constant(frostfield_has_environment)]],
    texture2d<float> alphaMapTexture [[texture(15), function_constant(frostfield_has_alpha_map)]],
    constant float4* alphaMapInfo [[buffer(23), function_constant(frostfield_has_alpha_map)]]
#if defined(FROSTFIELD_GENERATED_SURFACE)
    , constant float4* graphUniforms [[buffer(19)]]
    , array<texture2d<float>, 8> graphTextures [[texture(16)]]
    , constant FrostfieldGraphView& graphView [[buffer(24)]]
    FROSTFIELD_GRAPH_VIEWPORT_ARG
#endif
    FROSTFIELD_PBR_DEPTH_INPUT
) {
#if defined(FROSTFIELD_GENERATED_SURFACE)
    FrostfieldGraphResources graphResources = {
        graphUniforms, graphTextures[0], graphTextures[1], graphTextures[2], graphTextures[3],
        graphTextures[4], graphTextures[5], graphTextures[6], graphTextures[7], &graphView
        FROSTFIELD_GRAPH_VIEWPORT_VALUE
    };
#endif
    PBRTexCoords texCoords = {coordinates, extensions.transforms};
    PBRSurface surface = samplePBR(
        input, front, parameters, camera, baseColorFactor, settings, controls, emission, texCoords, colorTexture, colorSampler,
        metalRoughTexture, metalRoughSampler, normalTexture, normalSampler,
        occlusionTexture, occlusionSampler, emissiveTexture, emissiveSampler, alphaMapTexture, alphaMapInfo
#if defined(FROSTFIELD_GENERATED_SURFACE)
        , graphResources
#endif
    );
    if (extensions.options.x != 0.0f) {
        return FROSTFIELD_PBR_ATTRIBUTE_RESULT(frostfieldPackSurfaceAttributes(surface.normal, 1.0f, float3(0)));
    }
    PBRLayers layers = sampleLayers(input, texCoords, surface, extensions, layerMaps, layerSamplers);
    float3 dielectric = float3(dielectricF0(optical.y));
    float f90 = 1.0f;
    if (frostfield_has_layers) {
        dielectric = min(dielectric * layers.specularColor, 1.0f) * layers.specularWeight;
        f90 = mix(layers.specularWeight, 1.0f, surface.metallic);
    }
    float3 f0 = mix(dielectric, surface.base.rgb, surface.metallic);
    float noV = saturate(dot(surface.normal, pbrView(camera, input.worldPosition, surface.normal)));
    float2 dfg;
    if (frostfield_has_environment) {
        constexpr sampler lutFilter(coord::normalized, address::clamp_to_edge, filter::linear);
        dfg = environmentBRDF.sample(lutFilter, float2(noV, surface.roughness)).rg;
    } else {
        dfg = frostfieldEnvironmentBRDFApproximation(surface.roughness, noV);
    }
    // Shading scales the image-based reflection by material occlusion, so the stored
    // reflectance does too: reflections then replace exactly what shading added.
    return FROSTFIELD_PBR_ATTRIBUTE_RESULT(frostfieldPackSurfaceAttributes(
        surface.normal, surface.roughness, (f0 * dfg.x + f90 * dfg.y) * surface.occlusion
    ));
}

#if defined(FROSTFIELD_GRAPH_COVERAGE)
// Directional and punctual maps share this depth-only fragment. Only the graph's
// coverage outputs run, with the same material inputs as the color pass.
fragment void frostfield_mesh_graph_shadow(
    FrostfieldPBRVaryings input [[stage_in]], bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]], constant float4& settings [[buffer(2)]],
    constant float4& controls [[buffer(6)]], constant float4& emission [[buffer(7)]],
    constant float4& baseColorFactor [[buffer(8)]], constant uint& coordinates [[buffer(9)]],
    constant float4* graphUniforms [[buffer(19)]], constant FrostfieldPBRExtensions& extensions [[buffer(21)]],
    constant float4* alphaMapInfo [[buffer(23), function_constant(frostfield_has_alpha_map)]],
    constant FrostfieldGraphView& graphView [[buffer(24)]],
    texture2d<float> colorTexture [[texture(0)]], sampler colorSampler [[sampler(0)]],
    texture2d<float> metalRoughTexture [[texture(2)]], sampler metalRoughSampler [[sampler(1)]],
    texture2d<float> normalTexture [[texture(3)]], sampler normalSampler [[sampler(2)]],
    texture2d<float> occlusionTexture [[texture(4)]], sampler occlusionSampler [[sampler(3)]],
    texture2d<float> emissiveTexture [[texture(5)]], sampler emissiveSampler [[sampler(4)]],
    texture2d<float> alphaMapTexture [[texture(15), function_constant(frostfield_has_alpha_map)]],
    array<texture2d<float>, 8> graphTextures [[texture(16)]]
    FROSTFIELD_GRAPH_VIEWPORT_ARG
) {
    FrostfieldGraphResources resources = {
        graphUniforms, graphTextures[0], graphTextures[1], graphTextures[2], graphTextures[3],
        graphTextures[4], graphTextures[5], graphTextures[6], graphTextures[7], &graphView
        FROSTFIELD_GRAPH_VIEWPORT_VALUE
    };
    PBRTexCoords texCoords = {coordinates, extensions.transforms};
    (void)samplePBR(
        input, front, parameters, float4(0), baseColorFactor, settings, controls, emission, texCoords,
        colorTexture, colorSampler, metalRoughTexture, metalRoughSampler, normalTexture, normalSampler,
        occlusionTexture, occlusionSampler, emissiveTexture, emissiveSampler, alphaMapTexture, alphaMapInfo, resources, true
    );
}
#endif
#if defined(FROSTFIELD_GRAPH_CAST_SHADOW)
fragment float4 frostfield_mesh_graph_transmitted_shadow(
    FrostfieldPBRVaryings input [[stage_in]], bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]], constant uint& followsPeel [[buffer(1)]],
    constant float4& settings [[buffer(2)]],
    constant float4& controls [[buffer(6)]], constant float4& emission [[buffer(7)]],
    constant float4& baseColorFactor [[buffer(8)]], constant uint& coordinates [[buffer(9)]],
    constant float4* graphUniforms [[buffer(19)]], constant FrostfieldPBRExtensions& extensions [[buffer(21)]],
    constant float4* alphaMapInfo [[buffer(23), function_constant(frostfield_has_alpha_map)]],
    constant FrostfieldGraphView& graphView [[buffer(24)]],
    texture2d<float> colorTexture [[texture(0)]], sampler colorSampler [[sampler(0)]],
    depth2d<float, access::read> previousDepth [[texture(1)]],
    texture2d<float> metalRoughTexture [[texture(2)]], sampler metalRoughSampler [[sampler(1)]],
    texture2d<float> normalTexture [[texture(3)]], sampler normalSampler [[sampler(2)]],
    texture2d<float> occlusionTexture [[texture(4)]], sampler occlusionSampler [[sampler(3)]],
    texture2d<float> emissiveTexture [[texture(5)]], sampler emissiveSampler [[sampler(4)]],
    texture2d<float> alphaMapTexture [[texture(15), function_constant(frostfield_has_alpha_map)]],
    array<texture2d<float>, 8> graphTextures [[texture(16)]]
    FROSTFIELD_GRAPH_VIEWPORT_ARG
) {
    FrostfieldGraphResources resources = {
        graphUniforms, graphTextures[0], graphTextures[1], graphTextures[2], graphTextures[3],
        graphTextures[4], graphTextures[5], graphTextures[6], graphTextures[7], &graphView
        FROSTFIELD_GRAPH_VIEWPORT_VALUE
    };
    PBRTexCoords texCoords = {coordinates, extensions.transforms};
    PBRSurface cast = samplePBR(
        input, front, parameters, float4(0), baseColorFactor, settings, controls, emission, texCoords,
        colorTexture, colorSampler, metalRoughTexture, metalRoughSampler, normalTexture, normalSampler,
        occlusionTexture, occlusionSampler, emissiveTexture, emissiveSampler, alphaMapTexture, alphaMapInfo,
        resources, false, 1.0f, true
    );
    // Sampling and graph derivatives run before the per-pixel peel rejection.
    if (frostfieldAlreadyPeeled(input.position, followsPeel, previousDepth)) { discard_fragment(); }
    return float4(cast.base.rgb, input.position.z);
}
#endif
#endif // FROSTFIELD_PBR_HELPERS_ONLY
