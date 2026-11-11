#include <metal_stdlib>
#include "FrostfieldShadowSampling.h"
using namespace metal;

// Reuse the PBR varyings, light records, punctual shadows and the hashed alpha
// threshold without compiling a second copy of the PBR fragment functions.
#define FROSTFIELD_PBR_HELPERS_ONLY 1
#include "FrostfieldScreenSpace.h"
#include "PhysicallyBasedShading.h"

// ClassicMaterial.modelCode: 0 basic, 1 Lambert, 2 Phong, 3 toon, 4 matcap,
// 5 normal, 6 depth, 7 distance, 8 shadow.
constant uint frostfield_classic_model [[function_constant(40)]];
// True when the render state blends, so the fragment writes premultiplied color.
constant bool frostfield_classic_blends [[function_constant(41)]];
constant bool frostfield_classic_directional [[function_constant(42)]];
// ClassicMaterial.depthOutputCode: 0 grayscale, 1 red distance, 3200...3203 Three.js depth packing.
constant uint frostfield_classic_depth_output [[function_constant(43)]];
constant bool frostfield_classic_surface_detail [[function_constant(44)]];

// Fragment buffer 1, matching ClassicMaterial.Packed (368 bytes).
struct FrostfieldClassicUniforms {
    float4 color;        // linear rgb, opacity
    float4 emissive;      // emissive radiance (rgb), environment refraction ratio (w)
    float4 specular;     // Phong specular rgb, shininess
    float4 ambient;      // ambient irradiance rgb, alpha test cutoff
    float4 intensities;  // occlusion, light map, reflectivity, environment intensity
    float4 reference;    // distance reference xyz, cosine of the environment yaw
    float4 range;        // near, far, 1 for reversed depth, sine of the environment yaw
    float4 normal;       // signed XY scale, encoding (3 for height), object-space flag
    uint4 flags;         // bound maps, secondary UV bits, environment blend, options (refraction, flat shading)
    float4 transforms[14];
};

// The world-space right, up and backward axes of the view (fragment buffer 6).
struct FrostfieldViewBasis {
    float4 right;
    float4 up;
    float4 back;
};

static float2 classicUV(FrostfieldPBRVaryings input, constant FrostfieldClassicUniforms& material, uint map) {
    float2 uv = ((material.flags.y >> map) & 1u) != 0u ? input.uv1 : input.uv;
    return frostfieldTransformUV(uv, material.transforms[2u * map], material.transforms[2u * map + 1u]);
}

static bool classicHasMap(constant FrostfieldClassicUniforms& material, uint map) {
    return ((material.flags.x >> map) & 1u) != 0u;
}

static float3 classicMappedNormal(
    FrostfieldPBRVaryings input, float3 normal, float3 position, bool front,
    constant FrostfieldClassicUniforms& material, texture2d<float> detail, sampler detailSampler
) {
    float2 uv = classicUV(input, material, 6u);
    float side = front ? 1.0f : -1.0f;
    uint encoding = uint(material.normal.z);
    if (encoding == frostfieldBumpMapKind) {
        return side * pbrBumpNormal(normal * side, position, uv, material.normal.x, detail, detailSampler);
    }
    input.worldPosition = position;
    float4 linear = float4(material.transforms[12].xy, material.transforms[13].xy);
    bool primary = (material.flags.y & (1u << 6u)) == 0u;
    bool useTangent = primary && (material.flags.w & 2u) == 0u;
    return side * mappedNormal(
        input, uv, normal * side, detail.sample(detailSampler, uv).rgb, material.normal.xy,
        useTangent, linear, encoding, material.normal.w != 0.0f
    );
}

struct ClassicLighting {
    float3 diffuse;
    float3 specular;
};

// One light's direct contribution, with `irradiance` the light's irradiance at normal incidence.
static void classicDirect(
    thread ClassicLighting& lighting, float3 n, float3 v, float3 l, float3 irradiance,
    constant FrostfieldClassicUniforms& material, texture2d<float> gradient, sampler gradientSampler,
    bool hasGradient
) {
    constexpr float inversePi = 0.31830988618f;
    if (frostfield_classic_model == 3u) {
        // Three.js getGradientIrradiance: N·L is not clamped, so the far side reads the low end.
        float coordinate = dot(n, l) * 0.5f + 0.5f;
        float band;
        if (hasGradient) {
            band = gradient.sample(gradientSampler, float2(coordinate, 0.5f)).r;
        } else {
            float width = max(fwidth(coordinate) * 0.5f, 1e-5f);
            band = mix(0.7f, 1.0f, smoothstep(0.7f - width, 0.7f + width, coordinate));
        }
        lighting.diffuse += irradiance * (band * inversePi);
        return;
    }
    float noL = saturate(dot(n, l));
    lighting.diffuse += irradiance * (noL * inversePi);
    if (frostfield_classic_model == 2u && noL > 0.0f) {
        float3 h = unitOr(l + v, n);
        float noH = saturate(dot(n, h));
        float voH = saturate(dot(v, h));
        float falloff = 1.0f - voH;
        float squared = falloff * falloff;
        float3 fresnel = material.specular.rgb + (1.0f - material.specular.rgb) * (squared * squared * falloff);
        float distribution = inversePi * (material.specular.w * 0.5f + 1.0f) * pow(noH, material.specular.w);
        lighting.specular += irradiance * noL * fresnel * (0.25f * distribution);
    }
}

static float4 classicSurfaceColor(
    FrostfieldPBRVaryings input, float4 parameters, constant FrostfieldClassicUniforms& material,
    texture2d<float> colorMap, sampler colorSampler, texture2d<float> alphaMap, sampler alphaSampler
) {
    float4 diffuse = material.color * parameters * input.instanceTint;
    if (frostfield_has_vertex_colors) {
        diffuse *= input.color;
    }
    if (classicHasMap(material, 0u)) {
        diffuse *= colorMap.sample(colorSampler, classicUV(input, material, 0u));
    }
    if (classicHasMap(material, 1u)) {
        diffuse.a *= alphaMap.sample(alphaSampler, classicUV(input, material, 1u)).g;
    }
    diffuse = max(diffuse, 0.0f);
    diffuse.a = saturate(diffuse.a);
    if (frostfield_alpha_to_coverage && material.ambient.w > 0.0f) {
        // Three.js alphaTest with alphaToCoverage: ramp coverage at the cutoff, discard only zero.
        diffuse.a = frostfieldCoverageCutoff(diffuse.a, material.ambient.w);
        if (diffuse.a == 0.0f) {
            discard_fragment();
        }
    } else if (material.ambient.w > 0.0f && diffuse.a < material.ambient.w) {
        discard_fragment();
    }
    if (frostfield_alpha_hash) {
        if (diffuse.a < frostfieldAlphaHashThreshold(input.worldPosition)) {
            discard_fragment();
        }
        diffuse.a = 1.0f;
    }
    return diffuse;
}

fragment uint2 frostfield_mesh_classic_attributes(
    FrostfieldPBRVaryings input [[stage_in]],
    bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]],
    constant FrostfieldClassicUniforms& material [[buffer(1)]],
    constant float4& camera [[buffer(3)]],
    texture2d<float> colorMap [[texture(0)]], sampler colorSampler [[sampler(0)]],
    texture2d<float> alphaMap [[texture(2)]], sampler alphaSampler [[sampler(2)]],
    texture2d<float> detail [[texture(8), function_constant(frostfield_classic_surface_detail)]],
    sampler detailSampler [[sampler(8), function_constant(frostfield_classic_surface_detail)]]
) {
    classicSurfaceColor(input, parameters, material, colorMap, colorSampler, alphaMap, alphaSampler);
    float3 normal;
    if ((material.flags.w & 2u) != 0u) {
        normal = unitOr(cross(dfdx(input.worldPosition), dfdy(input.worldPosition)), float3(0, 0, 1));
        if (dot(normal, pbrView(camera, input.worldPosition, normal)) < 0.0f) {
            normal = -normal;
        }
    } else {
        normal = unitOr(input.normal, float3(0, 0, 1)) * (front ? 1.0f : -1.0f);
    }
    if (frostfield_classic_surface_detail) {
        normal = classicMappedNormal(input, normal, input.worldPosition, front, material, detail, detailSampler);
    }
    return frostfieldPackSurfaceAttributes(normal, 1.0f, float3(0));
}

static float4 classicDepthOutput(float depth, float opacity) {
    const uint packing = frostfield_classic_depth_output;
    if (packing == 3200u) {
        return float4(float3(1.0f - depth), opacity);
    }
    if (depth <= 0.0f || depth >= 1.0f) {
        const float endpoint = depth >= 1.0f ? 1.0f : 0.0f;
        if (packing == 3201u) { return float4(endpoint); }
        if (packing == 3202u) { return float4(float3(endpoint), 1.0f); }
        return float4(endpoint, endpoint, 0.0f, 1.0f);
    }

    // The most significant digit comes first. Whole bytes use 1/255,
    // while the last stored channel keeps the remaining base-256 fraction.
    constexpr float inverseByte = 1.0f / 255.0f;
    float digit;
    float fraction = modf(depth * 256.0f, digit);
    const float red = digit * inverseByte;
    if (packing == 3203u) { return float4(red, fraction, 0.0f, 1.0f); }
    fraction = modf(fraction * 256.0f, digit);
    const float green = digit * inverseByte;
    if (packing == 3202u) { return float4(red, green, fraction, 1.0f); }
    fraction = modf(fraction * 256.0f, digit);
    return float4(red, green, digit * inverseByte, fraction);
}

fragment float4 frostfield_mesh_classic(
    FrostfieldPBRVaryings input [[stage_in]],
    bool front [[front_facing]],
    constant float4& parameters [[buffer(0)]],
    constant FrostfieldClassicUniforms& material [[buffer(1)]],
    constant float4& shadowLight [[buffer(2)]],
    constant float4& camera [[buffer(3)]],
    constant FrostfieldDirectionalShadow& directionalShadow [[buffer(4)]],
    constant FrostfieldViewBasis& view [[buffer(6)]],
    constant FrostfieldPunctualLight* punctualLights [[buffer(11)]],
    constant uint& punctualLightCount [[buffer(12)]],
    constant float4x4* punctualShadowViews [[buffer(20)]],
    constant FrostfieldLightProbe& probe [[buffer(26)]],
    device const float4* probeGrids [[buffer(5)]],
    constant float4x4& relativeLightModel [[buffer(27)]],
    constant float4* relativeLightPositions [[buffer(28)]],
    texture2d<float> colorMap [[texture(0)]], sampler colorSampler [[sampler(0)]],
    depth2d_array<float> shadowMap [[texture(1)]],
    texture2d<float> alphaMap [[texture(2)]], sampler alphaSampler [[sampler(2)]],
    texture2d<float> occlusionMap [[texture(3)]], sampler occlusionSampler [[sampler(3)]],
    texture2d<float> lightMap [[texture(4)]], sampler lightSampler [[sampler(4)]],
    texture2d<float> emissiveMap [[texture(5)]], sampler emissiveSampler [[sampler(5)]],
    texture2d<float> lookup [[texture(6)]], sampler lookupSampler [[sampler(6)]],
    texturecube<float> environmentMap [[texture(7)]], sampler environmentSampler [[sampler(7)]],
    texture2d<float> detail [[texture(8), function_constant(frostfield_classic_surface_detail)]],
    sampler detailSampler [[sampler(8), function_constant(frostfield_classic_surface_detail)]],
    depth2d_array<float> punctualShadowMaps [[texture(24)]],
    texture2d_array<float, access::read> spotShadowMoments [[texture(29)]],
    texture2d_array<float> lightProfiles [[texture(25)]],
    texture2d_array<float> lightCookies [[texture(26)]],
    texture2d_array<float, access::read> shadowMoments [[texture(28), function_constant(frostfield_classic_directional)]]
) {
    constexpr float inversePi = 0.31830988618f;
    constexpr float maxHalf = 65504.0f;
    float4 diffuse = classicSurfaceColor(input, parameters, material, colorMap, colorSampler, alphaMap, alphaSampler);

    float4 relativeReceiver = pbrRelativeReceiver(input.shadowLocalPosition, relativeLightModel);
    float3 geometryPosition = relativeReceiver.w != 0.0f ? relativeReceiver.xyz : input.worldPosition;
    float3 geometric = unitOr(cross(dfdx(geometryPosition), dfdy(geometryPosition)), float3(0, 0, 1));
    float3 v = pbrRelativeView(camera, input.worldPosition, geometric, relativeReceiver);
    if (dot(geometric, v) < 0.0f) {
        geometric = -geometric;
    }
    float3 n = (material.flags.w & 2u) != 0u ? geometric
        : unitOr(input.normal, float3(0, 0, 1)) * (front ? 1.0f : -1.0f);
    if (frostfield_classic_surface_detail) {
        n = classicMappedNormal(input, n, geometryPosition, front, material, detail, detailSampler);
    }
    float3 color;
    float alpha = diffuse.a;

    if (frostfield_classic_model == 4u) {
        float3 viewNormal = float3(dot(n, view.right.xyz), dot(n, view.up.xyz), dot(n, view.back.xyz));
        float3 toCamera = unitOr(
            float3(dot(v, view.right.xyz), dot(v, view.up.xyz), dot(v, view.back.xyz)), float3(0, 0, 1)
        );
        float3 x = unitOr(float3(toCamera.z, 0.0f, -toCamera.x), float3(1, 0, 0));
        float3 y = cross(toCamera, x);
        float2 uv = float2(0.5f + 0.495f * dot(x, viewNormal), 0.5f - 0.495f * dot(y, viewNormal));
        color = diffuse.rgb * lookup.sample(lookupSampler, uv).rgb;
    } else if (frostfield_classic_model == 5u) {
        float3 viewNormal = float3(dot(n, view.right.xyz), dot(n, view.up.xyz), dot(n, view.back.xyz));
        color = normalize(viewNormal) * 0.5f + 0.5f;
    } else if (frostfield_classic_model == 6u) {
        float depth = input.position.z;
        if (frostfield_classic_depth_output != 0u) {
            return classicDepthOutput(depth, material.color.a);
        }
        color = float3(material.range.z > 0.0f ? depth : 1.0f - depth);
    } else if (frostfield_classic_model == 7u) {
        float distance = length(input.worldPosition - material.reference.xyz);
        float normalized = saturate((distance - material.range.x) / (material.range.y - material.range.x));
        if (frostfield_classic_depth_output == 1u) {
            return float4(normalized, 0.0f, 0.0f, 1.0f);
        }
        color = float3(normalized);
    } else if (frostfield_classic_model == 8u) {
        float visibility = 1.0f;
        if (frostfield_classic_directional && directionalShadow.origin.w > 0.0f) {
            // The catcher visualizes shadow depth on either side of the surface.
            // Use absolute incidence only for the directional filter's slope bias.
            float facing = max(saturate(abs(dot(geometric, shadowLight.xyz))), 1e-4f);
            visibility *= frostfieldShadowVisibility(
                shadowMap, shadowMoments, input.worldPosition, input.shadowLocalPosition, geometric, facing, directionalShadow
            );
        }
        for (uint index = 0; index < punctualLightCount; ++index) {
            constant FrostfieldPunctualLight& record = punctualLights[index];
            if (record.shadow.x >= 0.0f) {
                float4 lightPosition = relativeReceiver.w != 0.0f ? relativeLightPositions[index] : float4(0);
                visibility *= pbrPunctualShadow(
                    input.worldPosition, geometric, record, punctualShadowMaps, spotShadowMoments,
                    punctualShadowViews, relativeReceiver, lightPosition
                );
            }
        }
        color = diffuse.rgb;
        alpha *= 1.0f - saturate(visibility);
    } else {
        float occlusion = 1.0f;
        if (classicHasMap(material, 2u)) {
            float texel = occlusionMap.sample(occlusionSampler, classicUV(input, material, 2u)).r;
            occlusion = (texel - 1.0f) * material.intensities.x + 1.0f;
        }
        float3 baked = float3(0);
        if (classicHasMap(material, 3u)) {
            baked = max(lightMap.sample(lightSampler, classicUV(input, material, 3u)).rgb, 0.0f) * material.intensities.y;
        }
        if (frostfield_classic_model == 0u) {
            float3 indirect = classicHasMap(material, 3u) ? baked * inversePi : float3(1);
            color = diffuse.rgb * indirect * occlusion;
        } else {
            ClassicLighting lighting = {float3(0), float3(0)};
            bool hasGradient = classicHasMap(material, 5u);
            if (frostfield_classic_directional && shadowLight.w > 0.0f) {
                float facing = saturate(dot(geometric, shadowLight.xyz));
                float visibility = frostfieldShadowVisibility(
                    shadowMap, shadowMoments, input.worldPosition, input.shadowLocalPosition, geometric, facing, directionalShadow
                );
                classicDirect(
                    lighting, n, v, shadowLight.xyz, directionalShadow.lightColor.rgb * (visibility * min(shadowLight.w, maxHalf)),
                    material, lookup, lookupSampler, hasGradient
                );
            }
            for (uint index = 0; index < punctualLightCount; ++index) {
                constant FrostfieldPunctualLight& record = punctualLights[index];
                // Three.js applies rectangle lights only to physically based materials.
                if (record.positionKind.w == 3.0f) {
                    continue;
                }
                float4 lightPosition = relativeReceiver.w != 0.0f ? relativeLightPositions[index] : float4(0);
                PBRIncidentLight incident = pbrPunctual(
                    input.worldPosition, record, lightProfiles, lightCookies, relativeReceiver, lightPosition
                );
                if (record.shadow.x >= 0.0f) {
                    incident.radiance *= pbrPunctualShadow(
                        input.worldPosition, geometric, record, punctualShadowMaps, spotShadowMoments,
                        punctualShadowViews, relativeReceiver, lightPosition
                    );
                }
                classicDirect(
                    lighting, n, v, incident.direction, incident.radiance, material, lookup, lookupSampler, hasGradient
                );
            }
            float3 grid = frostfieldProbeGridIrradiance(probeGrids, frostfieldProbeLayers(probe), input.worldPosition, n);
            float3 indirect = (material.ambient.rgb + baked + frostfieldLightProbeIrradiance(probe, n) + grid) * inversePi * occlusion;
            float3 emitted = material.emissive.rgb;
            if (classicHasMap(material, 4u)) {
                emitted *= emissiveMap.sample(emissiveSampler, classicUV(input, material, 4u)).rgb;
            }
            color = diffuse.rgb * (lighting.diffuse + indirect) + lighting.specular + emitted;
        }
        if (classicHasMap(material, 7u) && frostfield_classic_model <= 2u) {
            float3 direction = (material.flags.w & 1u) != 0u
                ? refract(-v, n, material.emissive.w) : reflect(-v, n);
            float2 yaw = float2(material.reference.w, material.range.w);
            float3 environment = environmentMap.sample(environmentSampler, environmentDirection(direction, yaw)).rgb
                * material.intensities.w;
            float reflectivity = material.intensities.z;
            if (material.flags.z == 0u) {
                color = mix(color, color * environment, reflectivity);
            } else if (material.flags.z == 1u) {
                color = mix(color, environment, reflectivity);
            } else {
                color += environment * reflectivity;
            }
        }
    }
    color = min(max(color, 0.0f), float3(maxHalf));
    if (frostfield_alpha_to_coverage) {
        return float4(color, alpha);
    }
    if (frostfield_classic_blends) {
        return float4(color * alpha, alpha);
    }
    return float4(color, 1.0f);
}
