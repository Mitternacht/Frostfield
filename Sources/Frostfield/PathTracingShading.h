#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#include "FrostfieldMaterialExtensions.h"

// One entry of the scene's texture table, matching PathTracingScene.TextureTableEntry. A
// directly sampled texture stores its handle and the sampler state built from its binding's
// TextureSampling, with layer 0xffffffff; a fallback entry names its layer of the baked array.
struct PTTextureBinding {
    texture2d<half> texture;
    sampler state;
    uint layer;
    uint flags;
};

// FROSTFIELD_GENERATED_SURFACE_FUNCTIONS

constant bool frostfield_path_has_guides [[function_constant(0)]];
// True when the scene traces with Metal acceleration structures instead of the software BVH.
constant bool frostfield_path_metal_rt [[function_constant(1)]];
constant bool frostfield_path_has_aovs [[function_constant(2)]];
constant bool frostfield_path_has_motion [[function_constant(3)]];
// The current material snapshot selects the shadow traversal at pipeline creation.
constant bool frostfield_path_has_null_boundaries [[function_constant(4)]];
constant bool frostfield_path_has_mesh_emitters [[function_constant(5)]];
constant uint ptMissing = 0xffffffffu;
constant float ptPi = 3.141592653589793f;
constant float ptInvPi = 0.318309886183791f;

// One triangle's corners for shading, gathered from a bottom level's vertices. meta holds the
// material index, 1 when back faces are culled, the surface graph and 1 for vertex colors.
struct PTTriangle {
    float4 p0, p1, p2;
    float4 n0, n1, n2;
    float4 uv0, uv1, uv2;
    uint4 meta;
};

// Keep in lockstep with PathTracingScene.Vertex: object-space position, unit normal, and
// UV0 in xy and UV1 in zw. Colored geometry packs half RG and BA in position.w and normal.w.
struct PTVertex {
    float4 position, normal, uv;
};

struct PTNode {
    float4 minimum, maximum;
    uint4 meta;
};

// Keep in lockstep with PathTracingScene.InstanceRecord. Rows of the affine
// object-to-world and world-to-object transforms; meta holds the material index,
// bottom-level node count (zero under Metal ray tracing), bottom-level triangle count and
// flags; surface.x is the surface graph and surface.y is the object ID. The pointers address the bottom level's vertices,
// three vertex indices per triangle, software BVH nodes, and, for the software BVH, each
// stored triangle's original number, which Metal reports as its primitive ID.
struct PTInstance {
    float4 objectToWorld[3];
    float4 worldToObject[3];
    uint4 meta;
    uint4 surface;
    device const PTVertex* vertices;
    device const uint* indices;
    device const PTNode* nodes;
    device const uint* primitives;
};

constant uint ptInstanceMirrored = 1u;
constant uint ptInstanceCullsBackFaces = 2u;
constant uint ptInstanceShadowFront = 8u;
constant uint ptInstanceShadowBack = 16u;
constant uint ptInstanceColored = 4u;

constant float ptStochasticCoverage = -2.0f;
constant float ptWrittenCoverage = -3.0f;

// Sampling modes from PTMaterialExtensions.coordinates; zero repeats with linear filtering.
constant uint ptAddressRepeat = 0u;
constant uint ptAddressMirror = 2u;
constant uint ptNearestMode = 16u;

// Keep in lockstep with PathTracer.Dynamics. counts holds the instance count, the sampler
// (0 independent, 1 Owen-scrambled Sobol, 2 blue dither), and texture and emitter counts.
// limits stores the legacy peak limit, direct mean limit and indirect mean limit.
struct PTDynamics {
    uint4 counts;
    float4 limits;
};

// KHR material extension data, matching PathTracingScene.MaterialExtensions (592 bytes).
struct PTMaterialExtensions {
    float4 specular;       // color factor (rgb), factor (a)
    float4 sheen;          // color (rgb), perceptual roughness (a)
    float4 anisotropy;     // strength, cos(rotation), sin(rotation), unused
    float4 iridescence;    // factor, film IOR, minimum and maximum thickness in nanometers
    float4 options;        // unlit mode (1 base, 2 graph emission), dispersion, specular modulation, saturate tint
    // Clearcoat factor, clearcoat roughness, transmission, specular, specular color, sheen color,
    // sheen roughness, anisotropy, iridescence and iridescence thickness layers, then two unused.
    uint textures[12];
    // x: bit i selects the secondary UV set for texture slot i. For map m (0 base color,
    // 1 normal, 2 metallic-roughness, 3 emissive, then 4 + texture slot), bits 2m and 2m + 1
    // of y and z hold the U and V address modes, and bit m of w selects nearest filtering.
    uint4 coordinates;
    // Two rows each for base color, normal, metallic-roughness and emissive, then each texture slot.
    float4 transforms[28];
};

// Diffuse transmission, diffuse roughness, clearcoat normal and medium data, matching
// PathTracingScene.MaterialLobes (384 bytes).
struct PTMaterialLobes {
    float4 translucency;  // diffuse transmission color (rgb), translucent share of the diffuse lobe (a)
    float4 scattering;    // multiple-scattering color before the entry texture (rgb), Henyey-Greenstein g (a)
    float4 extinction;    // per world unit (rgb); w is the authored transmission thickness
    float4 surface;       // diffuse roughness, clearcoat normal scale, light-map intensity, retroreflectivity
    uint4 flags;          // x: ptLobe flags
    // Clearcoat normal, diffuse transmission, diffuse transmission color, diffuse roughness and
    // multiple-scattering color, baked irradiance, then two unused.
    uint textures[8];
    // x: bit i selects the secondary UV set for slot i; bits 2i and 2i + 1 of y and z hold the
    // U and V address modes, and bit i of w selects nearest filtering.
    uint4 coordinates;
    float4 transforms[16]; // two rows per slot
};

constant uint ptLobeMedium = 1u;           // a closed volume with a homogeneous medium
constant uint ptLobeDiffuseBoundary = 2u;  // light enters and leaves the volume diffusely
constant uint ptLobeNullBoundary = 4u;     // an index-matched, fully transmissive boundary
constant uint ptLobeScattering = 8u;       // the medium scatters
constant uint ptLobeSubsurfaceColor = 16u; // a zero-radius subsurface: the diffuse lobe takes its color

// emission.w is the alpha test, matching PathTracingScene.Material: a mask cutoff of zero
// or more, -1 for no test, ptStochasticCoverage, or ptWrittenCoverage (opaque to rays, with
// the alpha scaling a camera ray's radiance and coverage).
struct PTMaterial {
    float4 baseColor, factors, emission, optical, layers;
    uint4 textureIndices, textureUVs;
    float4 graphParameters;
    // Uniform offset in float4 slots and texture-table offset for the material's surface graph.
    uint4 graphResources;
    PTMaterialExtensions extensions;
    PTMaterialLobes lobes;
};

// Keep in lockstep with PathTracer.MediaUniforms. Fog regions add their
// extinction outside material-bounded optical volumes.
struct PTFogVolume {
    float4 fogExtinction;  // per world unit (rgb)
    float4 fogAlbedo;      // single-scattering albedo (rgb), Henyey-Greenstein g (w)
    float4 fogMinimum, fogMaximum;
    uint4 fogResolution;  // xyz dimensions, w: active record count in the first record
    device const float* fogDensity;
    device const uint4* fogBoundary;
    float4x4 fogLocalFromWorld;
};
static_assert(sizeof(PTFogVolume) == 160, "Fog uniform array stride");

// The two-level acceleration structure and texture table for one dispatch. Material texture
// indices address `textures`; fallback entries sample their layer of `maps`. Under Metal ray
// tracing, `accelerationStructure` holds the instances in `instances` order and `topLevel`
// is unused.
struct PTScene {
    device const PTInstance* instances;
    device const PTNode* topLevel;
    device const PTMaterial* materials;
    texture2d_array<half> maps;
    device const PTTextureBinding* textures;
    uint textureCount;
    uint topLevelCount;
    uint instanceCount;
    instance_acceleration_structure accelerationStructure;
#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
    constant float4* graphUniforms;
    device const uint* graphTable;
#endif
};

struct PTAreaLight {
    float4 center, halfU, halfV, radiance; // center.w is camera visibility; halfU.w selects a disk.
};

// Six float4 slots, matching PathTracer.PackedAnalyticLight. The first three slots
// and cone.xyz use FrostfieldPunctualLight's raster packing; cone.w holds
// 1 - cos(half angle) for a directional disk and zero for a delta light. frame holds
// the light's local +X and 1 + its IES profile layer; tables holds 1 + its cookie
// layer, outer cone tangent, aperture radius and axial offset. Zero layers mean none.
struct PTAnalyticLight {
    float4 positionKind, directionRange, colorIntensity, cone, frame, tables;
};

// Matches EnvironmentLookup. The x.w lane holds intensity.
struct PTSkyLookup {
    float4 x, y, z;
};

// Keep in lockstep with PathTracer.Uniforms. skyInfo holds map width, map height,
// analytic light count and flags.
// cameraRight.w and cameraUp.w hold the half window; projection holds near, far, the
// projection mode and half the stereo eye separation; lens holds the aperture blade
// count, rotation and horizontal and vertical scales; shift.xy moves the perspective window.
struct PTUniforms {
    float4 cameraRight, cameraUp, cameraForward, cameraPosition, environment, projection;
    uint4 image, controls;
    PTSkyLookup sky;
    uint4 skyInfo;
    float4 lens, shift;
};

static_assert(sizeof(PTSkyLookup) == 48, "Path sky lookup stride");
static_assert(sizeof(PTUniforms) == 224, "Path uniforms stride");

// Projection modes, matching PathTracer.CameraModel.mode.
constant uint ptPerspective = 0u;
constant uint ptOrthographic = 1u;
constant uint ptEquirectangular = 2u;
constant uint ptStereoEquirectangular = 3u;

constant uint ptSkyMap = 1u;
constant uint ptSkySampled = 2u;
constant uint ptSkyHidden = 4u;
constant uint ptSkyDirectionalDisks = 8u;
constant uint ptDisableMIS = 16u;

struct PTRay {
    float3 origin, direction;
    float minimum, maximum;
    float width, spread;
    // Pixel coordinate of the camera sample, then the image size. Bounces keep the camera pixel.
    float4 screen;
};

// index is the triangle within the instance's bottom level.
struct PTHit {
    float distance;
    uint index;
    float2 barycentric;
    uint instance;
};

struct PTEmissiveTriangle {
    uint instance, primitive;
    float cumulative, area;
};

struct PTLightHit {
    float distance, area, cosine;
    uint index;
};

static float ptMax(float3 value) {
    return max(value.x, max(value.y, value.z));
}

static float3 ptUnit(float3 value, float3 fallback) {
    float scale = ptMax(abs(value));
    return isfinite(scale) && scale > 0.0f ? normalize(value / scale) : fallback;
}

static uint ptHash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

static float ptRandom(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint value = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    value ^= value >> 22u;
    return float(value >> 8u) * 0x1p-24f;
}

static float2 ptRandom2(thread uint& state) {
    float x = ptRandom(state);
    return float2(x, ptRandom(state));
}

// One camera sample's random numbers. The independent mode draws the original PCG
// stream in call order. The Sobol mode reads one Owen-scrambled dimension per call,
// starting at `dimension`, with a per-pixel, per-dimension index shuffle and
// scramble (Burley 2020, "Practical Hash-based Owen Scrambling"). Blue dithering
// shares the scrambled sequence across image pixels and rotates it by a spatial rank.
struct PTSampler {
    uint state;
    uint seed;
    uint index;
    uint dimension;
    bool sobol;
    uint ditherOffset;
};

// Bits flow only from higher to lower positions of x, which nests the scramble.
static uint ptOwen(uint x, uint seed) {
    x = reverse_bits(x);
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16u) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return reverse_bits(x);
}

// The second Sobol dimension: the Pascal matrix modulo two.
static uint ptSobolSecond(uint index) {
    uint result = 0u;
    for (uint direction = 0x80000000u; index != 0u; index >>= 1u, direction ^= direction >> 1u) {
        if ((index & 1u) != 0u) { result ^= direction; }
    }
    return result;
}

static float2 ptSobol2(thread const PTSampler& sampler, uint dimension) {
    uint seed = ptHash(sampler.seed ^ ptHash(dimension + 0x9e3779b9u));
    uint shuffled = ptOwen(sampler.index, seed);
    uint x = ptOwen(reverse_bits(shuffled), ptHash(seed ^ 0xa511e9b3u));
    uint y = ptOwen(ptSobolSecond(shuffled), ptHash(seed ^ 0x63d83595u));
    uint2 values = uint2(x, y) >> 8u;
    // Add the rank modulo one on the same 24-bit grid as the sampler. This
    // preserves the exact circular shift without rounding across the wrap.
    if (sampler.ditherOffset != 0u) { values = (values + sampler.ditherOffset) & 0xffffffu; }
    return float2(values) * 0x1p-24f;
}

static float ptRandom(thread PTSampler& sampler) {
    if (!sampler.sobol) { return ptRandom(sampler.state); }
    return ptSobol2(sampler, sampler.dimension++).x;
}

static float2 ptRandom2(thread PTSampler& sampler) {
    if (!sampler.sobol) { return ptRandom2(sampler.state); }
    return ptSobol2(sampler, sampler.dimension++);
}

// Dimension slots per path vertex. Independent mode ignores them.
constant uint ptCameraDimensions = 2u;
constant uint ptVertexDimensions = 24u;
constant uint ptAreaLightSlot = 0u;     // light choice, then panel point
constant uint ptSkySlot = 2u;
constant uint ptScatterSlot = 3u;       // lobe, direction, then reflection choice
constant uint ptSurvivalSlot = 6u;
constant uint ptAnalyticSlot = 7u;      // one per analytic light, up to 16
constant uint ptMediumSlot = 23u;       // fog free-path channel and distance
// A disjoint pair per bounce keeps all existing dimensions and their sequences intact.
constant uint ptEmitterDimensionBase = 0x10000u;
constant uint ptFogDimensionBase = 0x20000u;
constant uint ptMaximumFogVolumes = 8u;  // PathTracer.Lighting.maximumFogVolumes
// Boundary faces one ray may cross inside one mesh fog region: PathTracer.Fog.MeshBoundary.maximumRayCrossings.
constant uint ptMaximumFogCrossings = 256u;
// Re-queries after skipping a group of faces that share a ray-edge or ray-vertex hit.
constant uint ptMaximumFogCoincidentGroups = 32u;

static void ptDimension(thread PTSampler& sampler, uint bounce, uint slot) {
    sampler.dimension = ptCameraDimensions + bounce * ptVertexDimensions + slot;
}

// Scales a contribution so its largest channel is at most the limit; zero disables it.
static float3 ptClamp(float3 value, float limit) {
    float peak = max(value.x, max(value.y, value.z));
    return limit > 0.0f && peak > limit ? value * (limit / peak) : value;
}

static float3 ptLimitContribution(float3 value, uint pathDepth, float3 limits) {
    float limit = pathDepth <= 1u ? limits.y : limits.z;
    if (limit < 0.0f) {
        return pathDepth == 0u ? value : ptClamp(value, limits.x);
    }
    if (limit == 0.0f) { return value; }
    // The reference caps the mean absolute RGB, scaling all channels together.
    float strength = dot(abs(value), float3(1.0f / 3.0f));
    return strength > limit ? value * (limit / strength) : value;
}

static float2 ptDisk(float2 random) {
    float angle = 2.0f * ptPi * random.y;
    return sqrt(random.x) * float2(cos(angle), sin(angle));
}

static float3x3 ptFrame(float3 normal) {
    float3 axis = abs(normal.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(axis, normal));
    return float3x3(tangent, cross(normal, tangent), normal);
}

static float ptOffset(float3 position) {
    return max(1e-5f, ptMax(abs(position)) * 2e-7f);
}

static float3 ptOrigin(float3 position, float3 normal, float3 direction) {
    return position + copysign(ptOffset(position), dot(normal, direction)) * normal;
}

static float4 ptUV(thread const PTTriangle& triangle, float2 barycentric) {
    return triangle.uv0 * (1.0f - barycentric.x - barycentric.y)
        + triangle.uv1 * barycentric.x + triangle.uv2 * barycentric.y;
}

static float2 ptUVSet(float4 uv, uint coordinateSet) {
    return coordinateSet == 1 ? uv.zw : uv.xy;
}

// The world-space footprint of a ray cone where it meets a surface, after the anisotropic
// ray cones of Akenine-Möller et al. 2021, "Improved Shader and Texture Level of Detail Using
// Ray Cones". Both axes lie in the surface plane. The major axis follows the ray's projection
// onto the plane and is stretched by 1 / |cos| of the incidence angle; the minor axis crosses
// it. Each spans the cone's full width, one pixel for camera rays, so its image in texture
// space plays the role of a raster derivative.
struct PTFootprint {
    float3 major, minor;
};

static PTFootprint ptFootprint(float3 direction, float3 faceNormal, float width) {
    float3 along = direction - dot(faceNormal, direction) * faceNormal;
    float length2 = dot(along, along);
    along = length2 > 1e-12f ? along * rsqrt(length2) : ptFrame(faceNormal)[0];
    float cosine = abs(dot(faceNormal, direction));
    return {along * (width / max(cosine, 1e-4f)), cross(faceNormal, along) * width};
}

// The texture-coordinate offsets of a footprint's axes in one UV set: major (xy), minor (zw).
// Each axis is written in the triangle's edges, whose UV differences then give its offset.
static float4 ptUVGradients(thread const PTTriangle& triangle, uint coordinateSet, PTFootprint footprint) {
    float3 e1 = triangle.p1.xyz - triangle.p0.xyz;
    float3 e2 = triangle.p2.xyz - triangle.p0.xyz;
    float3 normal = cross(e1, e2);
    float area2 = dot(normal, normal);
    if (!(area2 > 0.0f) || !isfinite(area2)) { return float4(0); }
    float inverse = 1.0f / area2;
    float2 first = ptUVSet(triangle.uv1, coordinateSet) - ptUVSet(triangle.uv0, coordinateSet);
    float2 second = ptUVSet(triangle.uv2, coordinateSet) - ptUVSet(triangle.uv0, coordinateSet);
    float2 major = (dot(normal, cross(footprint.major, e2)) * first + dot(normal, cross(e1, footprint.major)) * second);
    float2 minor = (dot(normal, cross(footprint.minor, e2)) * first + dot(normal, cross(e1, footprint.minor)) * second);
    float4 result = float4(major, minor) * inverse;
    return select(float4(0), result, isfinite(result));
}

// The texture transform's linear part applied to both gradients.
static float4 ptTransformGradients(float4 gradients, float4 row0, float4 row1) {
    return float4(
        dot(row0.xy, gradients.xy), dot(row1.xy, gradients.xy), dot(row0.xy, gradients.zw), dot(row1.xy, gradients.zw)
    );
}

// Interpolated coordinates of both UV sets at a hit and the footprint gradients of each.
struct PTTexturing {
    float4 uv;
    float4 gradients[2];
    PTFootprint footprint;
};

static PTTexturing ptTexturing(thread const PTTriangle& triangle, float2 barycentric, float3 direction, float width) {
    float3 face = ptUnit(cross(triangle.p1.xyz - triangle.p0.xyz, triangle.p2.xyz - triangle.p0.xyz), float3(0, 0, 1));
    PTTexturing texturing;
    texturing.uv = ptUV(triangle, barycentric);
    texturing.footprint = ptFootprint(direction, face, max(width, 0.0f));
    texturing.gradients[0] = ptUVGradients(triangle, 0u, texturing.footprint);
    texturing.gradients[1] = ptUVGradients(triangle, 1u, texturing.footprint);
    return texturing;
}

// Folds a mirrored-repeat coordinate into [0, 1]. Clamp-to-edge sampling of the folded
// coordinate then matches mirrored-repeat addressing at every mip level.
static float ptMirror(float coordinate) {
    return 1.0f - abs(fract(coordinate * 0.5f) * 2.0f - 1.0f);
}

// Samples one baked layer with repeat or clamp-to-edge addressing on each axis and linear or
// nearest filtering, at the level the gradients select. Mip levels always blend linearly.
static half4 ptSampleLayer(
    texture2d_array<half> maps, float2 uv, uint layer, float4 gradients, bool clampU, bool clampV, bool nearest
) {
    constexpr sampler filtering(coord::normalized, address::repeat, filter::linear, mip_filter::linear);
    constexpr sampler clampURepeatV(
        coord::normalized, s_address::clamp_to_edge, t_address::repeat, filter::linear, mip_filter::linear
    );
    constexpr sampler repeatUClampV(
        coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear, mip_filter::linear
    );
    constexpr sampler clampLinear(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
    constexpr sampler repeatNearest(coord::normalized, address::repeat, filter::nearest, mip_filter::linear);
    constexpr sampler clampURepeatVNearest(
        coord::normalized, s_address::clamp_to_edge, t_address::repeat, filter::nearest, mip_filter::linear
    );
    constexpr sampler repeatUClampVNearest(
        coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::nearest, mip_filter::linear
    );
    constexpr sampler clampNearest(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::linear);
    gradient2d lod(gradients.xy, gradients.zw);
    switch (uint(clampU) | (uint(clampV) << 1u) | (uint(nearest) << 2u)) {
        case 1u: return maps.sample(clampURepeatV, uv, layer, lod);
        case 2u: return maps.sample(repeatUClampV, uv, layer, lod);
        case 3u: return maps.sample(clampLinear, uv, layer, lod);
        case 4u: return maps.sample(repeatNearest, uv, layer, lod);
        case 5u: return maps.sample(clampURepeatVNearest, uv, layer, lod);
        case 6u: return maps.sample(repeatUClampVNearest, uv, layer, lod);
        case 7u: return maps.sample(clampNearest, uv, layer, lod);
        default: return maps.sample(filtering, uv, layer, lod);
    }
}

// Samples a table entry with its texture transform rows. A directly sampled texture uses its
// own sampler state, which carries the binding's addressing per axis, filters and anisotropy,
// and its own mip chain; a baked fallback uses the sampling modes (bits 0-1 U address, bits 2-3
// V address, bit 4 nearest). Missing maps read as one.
static float4 ptMap(
    thread const PTScene& scene, uint index, uint coordinateSet, thread const PTTexturing& texturing,
    float4 row0, float4 row1, uint modes, float2 offset = float2(0)
) {
    if (index == ptMissing || index >= scene.textureCount) { return float4(1); }
    float2 coordinate = frostfieldTransformUV(ptUVSet(texturing.uv, coordinateSet), row0, row1) + offset;
    float4 gradients = ptTransformGradients(texturing.gradients[coordinateSet & 1u], row0, row1);
    device const PTTextureBinding& binding = scene.textures[index];
    if (binding.layer == ptMissing) {
        texture2d<half> texture = binding.texture;
        return float4(texture.sample(binding.state, coordinate, gradient2d(gradients.xy, gradients.zw)));
    }
    if (binding.layer >= scene.maps.get_array_size()) { return float4(1); }
    uint addressU = modes & 3u;
    uint addressV = (modes >> 2u) & 3u;
    if (addressU == ptAddressMirror) { coordinate.x = ptMirror(coordinate.x); }
    if (addressV == ptAddressMirror) { coordinate.y = ptMirror(coordinate.y); }
    return float4(ptSampleLayer(
        scene.maps, coordinate, binding.layer, gradients, addressU != ptAddressRepeat, addressV != ptAddressRepeat,
        (modes & ptNearestMode) != 0u
    ));
}

// The sampling modes of material map `map`, in ptMap's layout.
static uint ptMapModes(device const PTMaterial& material, uint map) {
    uint4 packed = material.extensions.coordinates;
    uint shift = 2u * map;
    return ((packed.y >> shift) & 3u) | (((packed.z >> shift) & 3u) << 2u) | (((packed.w >> map) & 1u) << 4u);
}

// Base maps: 0 base color, 1 normal, 2 metallic-roughness, 3 emissive. Texture slots follow from 4.
static float4 ptMaterialMap(
    thread const PTScene& scene, device const PTMaterial& material, uint map, uint index, uint coordinateSet,
    thread const PTTexturing& texturing, float2 offset = float2(0)
) {
    return ptMap(
        scene, index, coordinateSet, texturing,
        material.extensions.transforms[2u * map], material.extensions.transforms[2u * map + 1u],
        ptMapModes(material, map), offset
    );
}

static float4 ptLayerMap(
    thread const PTScene& scene, device const PTMaterial& material, uint slot, thread const PTTexturing& texturing
) {
    return ptMaterialMap(
        scene, material, 4u + slot, material.extensions.textures[slot], (material.extensions.coordinates.x >> slot) & 1u,
        texturing
    );
}

// Samples lobe slot `slot` of PTMaterialLobes: 0 clearcoat normal, 1 diffuse transmission,
// 2 diffuse transmission color, 3 diffuse roughness, 4 multiple-scattering color, 5 baked irradiance.
static float4 ptLobeMap(
    thread const PTScene& scene, device const PTMaterial& material, uint slot, thread const PTTexturing& texturing
) {
    device const PTMaterialLobes& lobes = material.lobes;
    uint4 packed = lobes.coordinates;
    uint shift = 2u * slot;
    uint modes = ((packed.y >> shift) & 3u) | (((packed.z >> shift) & 3u) << 2u) | (((packed.w >> slot) & 1u) << 4u);
    return ptMap(
        scene, lobes.textures[slot], (packed.x >> slot) & 1u, texturing,
        lobes.transforms[2u * slot], lobes.transforms[2u * slot + 1u], modes
    );
}

static bool ptBounds(
    device const PTNode& node, PTRay ray, float limit, float3 inverse, bool3 parallel
) {
    float start = ray.minimum;
    float end = limit;
    for (uint axis = 0; axis < 3; ++axis) {
        float origin = ray.origin[axis];
        if (parallel[axis]) {
            if (origin < node.minimum[axis] || origin > node.maximum[axis]) { return false; }
        } else {
            float a = (node.minimum[axis] - origin) * inverse[axis];
            float b = (node.maximum[axis] - origin) * inverse[axis];
            start = max(start, min(a, b));
            end = min(end, max(a, b));
            if (end < start) { return false; }
        }
    }
    return true;
}

#include "FrostfieldMeshBVH.h"

static float3 ptToWorld(device const PTInstance& instance, float3 position) {
    float4 point = float4(position, 1.0f);
    return float3(
        dot(instance.objectToWorld[0], point), dot(instance.objectToWorld[1], point), dot(instance.objectToWorld[2], point)
    );
}

static float3 ptToObject(device const PTInstance& instance, float4 value) {
    return float3(
        dot(instance.worldToObject[0], value), dot(instance.worldToObject[1], value), dot(instance.worldToObject[2], value)
    );
}

// Transforms by the inverse transpose of the object-to-world matrix.
static float4 ptNormalToWorld(device const PTInstance& instance, float4 normal, float3 fallback = float3(0, 0, 1)) {
    float3 world = instance.worldToObject[0].xyz * normal.x + instance.worldToObject[1].xyz * normal.y
        + instance.worldToObject[2].xyz * normal.z;
    return float4(ptUnit(world, fallback), 0.0f);
}

// Gathers triangle `index` of the instance's bottom level in object space.
static PTTriangle ptLocalTriangle(device const PTInstance& instance, uint index) {
    device const uint* corners = instance.indices + 3u * index;
    PTVertex a = instance.vertices[corners[0]];
    PTVertex b = instance.vertices[corners[1]];
    PTVertex c = instance.vertices[corners[2]];
    PTTriangle triangle;
    triangle.p0 = a.position;
    triangle.p1 = b.position;
    triangle.p2 = c.position;
    triangle.n0 = a.normal;
    triangle.n1 = b.normal;
    triangle.n2 = c.normal;
    triangle.uv0 = a.uv;
    triangle.uv1 = b.uv;
    triangle.uv2 = c.uv;
    triangle.meta = uint4(
        instance.meta.x, (instance.meta.w & ptInstanceCullsBackFaces) != 0u ? 1u : 0u, instance.surface.x,
        (instance.meta.w & ptInstanceColored) != 0u ? 1u : 0u
    );
    return triangle;
}

// The world-space triangle that shading uses, carrying the instance's material index.
// Mirrored instances swap the second and third vertices, and the barycentric pair
// that addresses them, so the world winding still faces along the transformed normals.
static PTTriangle ptWorldTriangle(
    device const PTInstance& instance, thread const PTTriangle& local, thread float2& barycentric
) {
    bool mirrored = (instance.meta.w & ptInstanceMirrored) != 0u;
    PTTriangle world;
    // The w lanes carry packed vertex colors when local.meta.w is set (see ptVertexColor).
    world.p0 = float4(ptToWorld(instance, local.p0.xyz), local.p0.w);
    float4 p1 = float4(ptToWorld(instance, local.p1.xyz), local.p1.w);
    float4 p2 = float4(ptToWorld(instance, local.p2.xyz), local.p2.w);
    world.n0 = float4(ptNormalToWorld(instance, local.n0).xyz, local.n0.w);
    float4 n1 = float4(ptNormalToWorld(instance, local.n1).xyz, local.n1.w);
    float4 n2 = float4(ptNormalToWorld(instance, local.n2).xyz, local.n2.w);
    world.uv0 = local.uv0;
    world.p1 = mirrored ? p2 : p1;
    world.p2 = mirrored ? p1 : p2;
    world.n1 = mirrored ? n2 : n1;
    world.n2 = mirrored ? n1 : n2;
    world.uv1 = mirrored ? local.uv2 : local.uv1;
    world.uv2 = mirrored ? local.uv1 : local.uv2;
    world.meta = local.meta;
    if (mirrored) { barycentric = barycentric.yx; }
    return world;
}

// One vertex's linear RGBA from the halves packed in its position and normal w lanes.
static float4 ptPackedColor(float4 position, float4 normal) {
    half2 redGreen = as_type<half2>(as_type<uint>(position.w));
    half2 blueAlpha = as_type<half2>(as_type<uint>(normal.w));
    return float4(float2(redGreen), float2(blueAlpha));
}

// The interpolated vertex color of a triangle, or one when it has none (meta.w == 0).
static float4 ptVertexColor(thread const PTTriangle& triangle, float2 barycentric) {
    if (triangle.meta.w == 0u) { return float4(1); }
    return ptPackedColor(triangle.p0, triangle.n0) * (1.0f - barycentric.x - barycentric.y)
        + ptPackedColor(triangle.p1, triangle.n1) * barycentric.x
        + ptPackedColor(triangle.p2, triangle.n2) * barycentric.y;
}

static float4 ptMaterialTint(device const PTMaterial& material, thread const PTTriangle& triangle, float2 barycentric) {
    if (material.extensions.options.w == 0.0f) { return material.baseColor; }
    return saturate(material.baseColor * ptVertexColor(triangle, barycentric));
}

// A uniform number that decides stochastic coverage for one ray and triangle. It depends
// only on the world ray's bits, the instance and the triangle's original number, so
// closest-hit and any-hit traversals of a ray agree in any visiting order, and the software
// BVH and Metal ray tracing make the same decision, while camera, bounce and shadow rays,
// whose origins or directions differ, decide independently.
static float ptCoverageSample(PTRay ray, uint instance, uint triangle) {
    uint value = ptHash(as_type<uint>(ray.origin.x) ^ 0x9e3779b9u);
    value = ptHash(value ^ as_type<uint>(ray.origin.y));
    value = ptHash(value ^ as_type<uint>(ray.origin.z));
    value = ptHash(value ^ as_type<uint>(ray.direction.x));
    value = ptHash(value ^ as_type<uint>(ray.direction.y));
    value = ptHash(value ^ as_type<uint>(ray.direction.z));
    value = ptHash(value ^ instance);
    value = ptHash(value ^ triangle);
    return float(value >> 8u) * 0x1p-24f;
}

#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
// Alpha, explicit graph cutoff (-1 for unset), and mask predicate.
static float3 ptGraphCoverage(
    thread const PTScene& scene, device const PTInstance& instance, thread const PTTriangle& triangle,
    float2 barycentric, float distance, PTRay ray, PTTexturing texturing
);
#endif

// The alpha test of a candidate hit on triangle `index`, whose original number is `primitive`.
// Masked materials reject hits below their cutoff. Stochastic coverage accepts a hit with
// probability alpha, so the expected visibility through the surface is 1 - alpha. The guide
// pass sets dominantCoverage, which accepts a stochastic hit exactly when alpha is at least
// one half, so its guides show the layer that dominates a pixel rather than per-pixel noise.
static bool ptAccepts(
    thread const PTScene& scene, PTRay ray, uint instanceIndex, uint index, uint primitive, float2 barycentric,
    float distance, bool dominantCoverage
) {
    device const PTInstance& instance = scene.instances[instanceIndex];
    device const PTMaterial& material = scene.materials[instance.meta.x];
    bool stochastic = material.emission.w == ptStochasticCoverage;
    bool testAlpha = material.emission.w >= 0.0f || stochastic;
#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
    uint flags = material.graphResources.w;
    bool graphCoverage = (flags & 2u) != 0u || (testAlpha && (flags & 1u) != 0u);
    if (!testAlpha && !graphCoverage) { return true; }
#else
    if (!testAlpha) { return true; }
#endif
    float2 worldBarycentric = barycentric;
    PTTriangle world = ptWorldTriangle(instance, ptLocalTriangle(instance, index), worldBarycentric);
    PTTexturing texturing = ptTexturing(world, worldBarycentric, ray.direction, ray.width + ray.spread * distance);
    float alpha;
    float graphCutoff = -1.0f;
#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
    if (graphCoverage) {
        float3 coverage = ptGraphCoverage(scene, instance, world, worldBarycentric, distance, ray, texturing);
        if (coverage.z == 0.0f) { return false; }
        alpha = coverage.x;
        graphCutoff = coverage.y;
    } else
#endif
    {
        alpha = ptMaterialTint(material, world, worldBarycentric).a * ptMaterialMap(
            scene, material, 0u, material.textureIndices.x, material.textureUVs.x, texturing
        ).a;
    }
    if (graphCutoff >= 0.0f ? !(alpha > graphCutoff) : (testAlpha && !stochastic && !(alpha >= material.emission.w))) {
        return false;
    }
    return !stochastic || (dominantCoverage ? alpha >= 0.5f : ptCoverageSample(ray, instanceIndex, primitive) < alpha);
}

struct PTBoundsDiagnostic {
    uint count;
    uint remaining;
    uint levels;
    uint saturation;
    bool truncated;
    bool intersects;
};

// Nil shadow side inherits the camera cull. A set side accepts only the named faces.
static bool ptShadowAccepts(uint flags, bool frontFacing, bool cullsBackFaces) {
    bool front = (flags & ptInstanceShadowFront) != 0u;
    bool back = (flags & ptInstanceShadowBack) != 0u;
    if (!front && !back) { return !cullsBackFaces || frontFacing; }
    return frontFacing ? front : back;
}

// Traverses the top level in world space and each entered bottom level in object
// space. Object rays keep the unnormalized transformed direction, so distances stay
// in world units. Both builders bound depth at 32, and one stack holds both levels.
template<bool diagnostic, bool countBounds>
static PTHit ptGeometrySoftwareTraversal(
    PTRay ray, bool anyHit, thread const PTScene& scene, bool dominantCoverage, thread PTBoundsDiagnostic& bounds,
    bool shadowRay = false
) {
    PTHit hit = {ray.maximum, ptMissing, float2(0), ptMissing};
    if (scene.topLevelCount == 0u) { return hit; }
    bool3 worldParallel = abs(ray.direction) < 1e-20f;
    float3 worldInverse = 1.0f / select(ray.direction, float3(1), worldParallel);
    PTRay local = ray;
    bool3 parallel = worldParallel;
    float3 inverse = worldInverse;
    FrostfieldShearedRay sheared = frostfieldShearRay(ray.origin, ray.direction);
    device const PTNode* nodes = scene.topLevel;
    uint nodeCount = scene.topLevelCount;
    uint instanceIndex = ptMissing;
    uint instanceFloor = 0u;
    uint stack[68];
    uint size = 1;
    stack[0] = 0;
    while (size != 0u) {
        if (diagnostic) {
            if (bounds.remaining == 0u) { bounds.truncated = true; break; }
            --bounds.remaining;
        }
        if (instanceIndex != ptMissing && size == instanceFloor) {
            instanceIndex = ptMissing;
            local = ray;
            parallel = worldParallel;
            inverse = worldInverse;
            nodes = scene.topLevel;
            nodeCount = scene.topLevelCount;
        }
        uint index = stack[--size];
        if (index >= nodeCount) { continue; }
        device const PTNode& node = nodes[index];
        if (!ptBounds(node, local, hit.distance, inverse, parallel)) { continue; }
        if (diagnostic && !countBounds) { bounds.intersects = true; }
        if (countBounds) {
            uint level = instanceIndex == ptMissing ? 1u : 2u;
            if ((bounds.levels & level) != 0u && ++bounds.count == bounds.saturation) { break; }
        }

        uint count = node.meta.w;
        if (count == 0u) {
            uint firstChild = node.meta.x;
            uint secondChild = node.meta.y;
            if (size + 2u <= 68u && firstChild < nodeCount && secondChild < nodeCount) {
                float3 firstCenter = nodes[firstChild].minimum.xyz * 0.5f + nodes[firstChild].maximum.xyz * 0.5f;
                float3 secondCenter = nodes[secondChild].minimum.xyz * 0.5f + nodes[secondChild].maximum.xyz * 0.5f;
                bool firstNearer = dot(secondCenter - firstCenter, local.direction) >= 0.0f;
                stack[size++] = firstNearer ? secondChild : firstChild;
                stack[size++] = firstNearer ? firstChild : secondChild;
            }
            continue;
        }
        if (instanceIndex == ptMissing) {
            if (countBounds && (bounds.levels & 2u) == 0u) { continue; }
            // A top-level leaf holds one instance.
            uint candidate = node.meta.z;
            if (candidate >= scene.instanceCount || size + 1u > 68u) { continue; }
            device const PTInstance& instance = scene.instances[candidate];
            if (instance.meta.y == 0u) { continue; }
            instanceIndex = candidate;
            instanceFloor = size;
            local.origin = ptToObject(instance, float4(ray.origin, 1.0f));
            local.direction = ptToObject(instance, float4(ray.direction, 0.0f));
            parallel = abs(local.direction) < 1e-20f;
            inverse = 1.0f / select(local.direction, float3(1), parallel);
            sheared = frostfieldShearRay(local.origin, local.direction);
            nodes = instance.nodes;
            nodeCount = instance.meta.y;
            stack[size++] = 0;
            continue;
        }
        if (countBounds) { continue; }
        device const PTInstance& instance = scene.instances[instanceIndex];
        bool cullsBackFaces = (instance.meta.w & ptInstanceCullsBackFaces) != 0u;
        uint first = node.meta.z;
        for (uint offset = 0; offset < count; ++offset) {
            uint candidate = first + offset;
            if (candidate >= instance.meta.z) { break; }
            device const uint* corners = instance.indices + 3u * candidate;
            float distance;
            float2 barycentric;
            bool frontFacing;
            if (!frostfieldRayTriangle<false>(
                instance.vertices[corners[0]].position.xyz, instance.vertices[corners[1]].position.xyz,
                instance.vertices[corners[2]].position.xyz, shadowRay ? false : cullsBackFaces, sheared, ray.minimum,
                hit.distance, distance, barycentric, frontFacing
            )) { continue; }
            if (shadowRay && !ptShadowAccepts(instance.meta.w, frontFacing, cullsBackFaces)) { continue; }
            if (!ptAccepts(
                scene, ray, instanceIndex, candidate, instance.primitives[candidate], barycentric, distance, dominantCoverage
            )) { continue; }
            hit = {distance, candidate, barycentric, instanceIndex};
            if (anyHit) { return hit; }
        }
    }
    return hit;
}

static PTHit ptGeometrySoftware(
    PTRay ray, bool anyHit, thread const PTScene& scene, bool dominantCoverage, bool shadowRay
) {
    PTBoundsDiagnostic unused = {};
    return ptGeometrySoftwareTraversal<false, false>(ray, anyHit, scene, dominantCoverage, unused, shadowRay);
}

// Metal ray tracing over the instance acceleration structure. Instances whose material has an
// alpha test are non-opaque, so their candidates return here for the same test as the
// software path; opaque instances commit in the intersector. Back faces cull per instance.
// Metal's winding names the order as the ray sees the corners in its own left-handed frame,
// so the triangles that face a ray with counterclockwise corners, as the software test and
// raster define front faces, are the ones Metal calls clockwise.
static PTHit ptGeometryMetal(PTRay ray, bool anyHit, thread const PTScene& scene, bool dominantCoverage, bool shadowRay) {
    PTHit hit = {ray.maximum, ptMissing, float2(0), ptMissing};
    if (scene.instanceCount == 0u || !(ray.maximum > ray.minimum)) { return hit; }
    intersection_params parameters;
    parameters.set_triangle_cull_mode(triangle_cull_mode::back);
    parameters.set_triangle_front_facing_winding(winding::clockwise);
    parameters.assume_geometry_type(geometry_type::triangle);
    parameters.accept_any_intersection(anyHit);
    intersection_query<triangle_data, instancing> query;
    query.reset(raytracing::ray(ray.origin, ray.direction, ray.minimum, ray.maximum), scene.accelerationStructure, 0xffu, parameters);
    while (query.next()) {
        if (query.get_candidate_intersection_type() != intersection_type::triangle) { continue; }
        uint primitive = query.get_candidate_primitive_id();
        uint instanceID = query.get_candidate_instance_id();
        device const PTInstance& instance = scene.instances[instanceID];
        bool customShadow = (instance.meta.w & (ptInstanceShadowFront | ptInstanceShadowBack)) != 0u;
        if (customShadow) {
            device const uint* corners = instance.indices + 3u * primitive;
            float3 origin = ptToObject(instance, float4(ray.origin, 1.0f));
            float3 direction = ptToObject(instance, float4(ray.direction, 0.0f));
            FrostfieldShearedRay sheared = frostfieldShearRay(origin, direction);
            float distance;
            float2 barycentric;
            bool frontFacing = false;
            if (!frostfieldRayTriangle<false>(
                instance.vertices[corners[0]].position.xyz, instance.vertices[corners[1]].position.xyz,
                instance.vertices[corners[2]].position.xyz, false, sheared, 0.0f, 1.0e20f,
                distance, barycentric, frontFacing
            )) { continue; }
            bool cullsBackFaces = (instance.meta.w & ptInstanceCullsBackFaces) != 0u;
            bool keep = shadowRay ? ptShadowAccepts(instance.meta.w, frontFacing, cullsBackFaces)
                : (!cullsBackFaces || frontFacing);
            if (!keep) { continue; }
        }
        if (ptAccepts(
            scene, ray, instanceID, primitive, primitive,
            query.get_candidate_triangle_barycentric_coord(), query.get_candidate_triangle_distance(), dominantCoverage
        )) {
            query.commit_triangle_intersection();
        }
    }
    if (query.get_committed_intersection_type() == intersection_type::triangle) {
        hit = {
            query.get_committed_distance(), query.get_committed_primitive_id(),
            query.get_committed_triangle_barycentric_coord(), query.get_committed_instance_id()
        };
    }
    return hit;
}

static PTHit ptGeometry(
    PTRay ray, bool anyHit, thread const PTScene& scene, bool dominantCoverage = false, bool shadowRay = false
) {
    if (frostfield_path_metal_rt) { return ptGeometryMetal(ray, anyHit, scene, dominantCoverage, shadowRay); }
    return ptGeometrySoftware(ray, anyHit, scene, dominantCoverage, shadowRay);
}

static PTLightHit ptAreaLights(
    PTRay ray, bool primary, device const PTAreaLight* lights, uint lightCount
) {
    PTLightHit hit = {ray.maximum, 0, 0, ptMissing};
    for (uint index = 0; index < lightCount; ++index) {
        device const PTAreaLight& light = lights[index];
        if (primary && light.center.w <= 0.0f) { continue; }
        float3 u = light.halfU.xyz;
        float3 v = light.halfV.xyz;
        float3 crossAxes = cross(u, v);
        float areaQuarter = length(crossAxes);
        if (!(areaQuarter > 0.0f)) { continue; }
        float3 normal = crossAxes / areaQuarter;
        float cosine = -dot(normal, ray.direction);
        if (abs(cosine) < 1e-8f) { continue; }
        float distance = dot(ray.origin - light.center.xyz, normal) / cosine;
        if (!(distance >= ray.minimum && distance < hit.distance)) { continue; }
        float3 relative = ray.origin + distance * ray.direction - light.center.xyz;
        float inverseArea = 1.0f / areaQuarter;
        float3 basis = normal * inverseArea;
        float s = dot(cross(relative, v), basis);
        float t = dot(cross(u, relative), basis);
        bool disk = light.halfU.w != 0.0f;
        if (disk ? s * s + t * t <= 1.0f : abs(s) <= 1.0f && abs(t) <= 1.0f) {
            hit = {distance, (disk ? ptPi : 4.0f) * areaQuarter, cosine, index};
        }
    }
    return hit;
}

struct PTSurface {
    float3 position, normal, geometricNormal, baseColor, emission, attenuation;
    // alpha is the base color alpha: factor, tint and map.
    float metallic, roughness, transmission, ior, attenuationDistance, coat, coatRoughness, alpha;
    bool front, thin;
    uint material;
    // KHR material extension layers.
    float3 specularColor, sheenColor, anisotropyDirection;
    float specularWeight, sheenRoughness, anisotropy, iridescence, iridescenceThickness, iridescenceIOR, dispersion;
    bool specularModulated, unlit;
    // The clearcoat normal facing the ray, diffuse transmission, diffuse roughness and the
    // multiple-scattering color where light enters the volume.
    float3 coatNormal, translucencyColor, scatterColor;
    float translucency, diffuseRoughness;
    uint lobeFlags;
    float3 bakedIrradiance;
    float retroreflectivity;
};

// A unit tangent (xyz) and bitangent sign (w) from the triangle's UV set after a texture
// transform; w is zero without a frame.
static float4 ptTangentFrame(
    float3 normal, thread const PTTriangle& triangle, uint coordinateSet,
    float4 row0 = float4(1, 0, 0, 0), float4 row1 = float4(0, 1, 0, 0)
) {
    float2 uv0 = frostfieldTransformUV(ptUVSet(triangle.uv0, coordinateSet), row0, row1);
    float2 duv1 = frostfieldTransformUV(ptUVSet(triangle.uv1, coordinateSet), row0, row1) - uv0;
    float2 duv2 = frostfieldTransformUV(ptUVSet(triangle.uv2, coordinateSet), row0, row1) - uv0;
    float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(determinant) < 1e-12f) { return float4(0); }

    float3 e1 = triangle.p1.xyz - triangle.p0.xyz;
    float3 e2 = triangle.p2.xyz - triangle.p0.xyz;
    float3 tangent = (e1 * duv2.y - e2 * duv1.y) / determinant;
    tangent = ptUnit(tangent - normal * dot(normal, tangent), float3(0));
    if (dot(tangent, tangent) == 0.0f) { return float4(0); }
    float3 axisV = (e2 * duv1.x - e1 * duv2.x) / determinant;
    return float4(tangent, copysign(1.0f, dot(cross(normal, tangent), axisV)));
}

// Maps a tangent-space normal. `normal` faces the ray; on back faces (front false) the
// frame is built from the front-side normal and the result negated, as raster and glTF do,
// so the tangential tilt keeps its sign instead of mirroring.
static float3 ptNormalMap(
    float3 normal, float3 encoded, float scale, thread const PTTriangle& triangle, uint coordinateSet,
    bool front, float4 row0 = float4(1, 0, 0, 0), float4 row1 = float4(0, 1, 0, 0), uint encoding = 0u
) {
    float side = front ? 1.0f : -1.0f;
    float3 outward = normal * side;
    float4 frame = ptTangentFrame(outward, triangle, coordinateSet, row0, row1);
    if (frame.w == 0.0f) { return normal; }
    float3 tangent = frame.xyz;
    float3 bitangent = cross(outward, tangent) * frame.w;
    float3 local = frostfieldDecodeNormalMap(encoded, encoding);
    float3 mapped = tangent * (local.x * scale) + bitangent * (local.y * scale) + outward * local.z;
    return ptUnit(mapped * side, normal);
}

static float3 ptObjectNormalMap(device const PTInstance& instance, float3 texel, uint encoding, bool front, float3 fallback) {
    float3 decoded = frostfieldDecodeNormalMap(texel, encoding) * (front ? 1.0f : -1.0f);
    return ptNormalToWorld(instance, float4(decoded, 0.0f), fallback).xyz;
}

static float3 ptBumpNormal(
    thread const PTScene& scene, device const PTMaterial& material, thread const PTTexturing& texturing,
    float3 normal, bool front
) {
    uint index = material.textureIndices.y, coordinateSet = material.textureUVs.y;
    float4 gradients = ptTransformGradients(
        texturing.gradients[coordinateSet & 1u], material.extensions.transforms[2], material.extensions.transforms[3]
    );
    float height = ptMaterialMap(scene, material, 1u, index, coordinateSet, texturing).r;
    float2 delta = float2(
        ptMaterialMap(scene, material, 1u, index, coordinateSet, texturing, gradients.xy).r,
        ptMaterialMap(scene, material, 1u, index, coordinateSet, texturing, gradients.zw).r
    ) - height;
    float side = front ? 1.0f : -1.0f;
    return side * frostfieldBumpNormal(
        normal * side, texturing.footprint.major, texturing.footprint.minor, material.layers.x * delta
    );
}

struct PTSampledSurface {
    float3 geometric, normal, vertexNormal;
    float4 base;
    float metallic, roughness;
    float3 emission;
    bool front;
};

// Coverage and shading see the same sampled channels and oriented normals.
static PTSampledSurface ptSampleSurface(
    thread const PTScene& scene, device const PTInstance& instance, device const PTMaterial& material,
    thread const PTTriangle& triangle, float2 bary, PTRay ray, PTTexturing texturing
) {
    float3 geometric = ptUnit(cross(triangle.p1.xyz - triangle.p0.xyz, triangle.p2.xyz - triangle.p0.xyz), -ray.direction);
    bool front = dot(geometric, ray.direction) < 0.0f;
    geometric *= front ? 1.0f : -1.0f;
    float3 normal = ptUnit(
        triangle.n0.xyz * (1.0f - bary.x - bary.y) + triangle.n1.xyz * bary.x + triangle.n2.xyz * bary.y,
        geometric
    );
    if (dot(normal, geometric) < 0.0f) { normal = -normal; }
    if (dot(normal, -ray.direction) < 1e-4f) { normal = geometric; }
    float4 baseColor = ptMaterialTint(material, triangle, bary) * ptMaterialMap(
        scene, material, 0u, material.textureIndices.x, material.textureUVs.x, texturing
    );
    float2 metalRough = ptMaterialMap(
        scene, material, 2u, material.textureIndices.z, material.textureUVs.z, texturing
    ).bg;
    float3 emission = material.emission.rgb;
    if (ptMax(emission) > 0.0f) {
        emission *= ptMaterialMap(
            scene, material, 3u, material.textureIndices.w, material.textureUVs.w, texturing
        ).rgb;
    }
    float3 vertexNormal = normal;
    bool objectNormal = (material.lobes.flags.w & 1u) != 0u;
    if (material.textureIndices.y != ptMissing && (material.layers.x != 0.0f || objectNormal)) {
        uint normalKind = material.lobes.flags.y;
        float3 mapped;
        if (normalKind == frostfieldBumpMapKind) {
            mapped = ptBumpNormal(scene, material, texturing, normal, front);
        } else {
            float3 map = ptMaterialMap(
                scene, material, 1u, material.textureIndices.y, material.textureUVs.y, texturing
            ).rgb;
            mapped = objectNormal ? ptObjectNormalMap(instance, map, normalKind, front, normal) : ptNormalMap(
                normal, map, material.layers.x, triangle, material.textureUVs.y, front,
                material.extensions.transforms[2], material.extensions.transforms[3], normalKind
            );
        }
        if (dot(mapped, geometric) > 0.0f && dot(mapped, -ray.direction) > 1e-4f) { normal = mapped; }
    }
    float metallic = saturate(material.factors.x * metalRough.x);
    float roughness = saturate(material.factors.y * metalRough.y);
    return {geometric, normal, vertexNormal, baseColor, metallic, roughness, emission, front};
}

#ifdef FROSTFIELD_HAS_SURFACE_GRAPHS
static FrostfieldSurfaceInput ptMakeGraphInput(
    device const PTInstance& instance, device const PTMaterial& material, thread const PTSampledSurface& sample,
    float4 uv, float4 graphFrame, float3 position, PTRay ray, constant float4* graphUniforms
) {
    FrostfieldSurfaceInput graphInput = {};
    graphInput.uv0 = uv.xy;
    graphInput.uv1 = uv.zw;
    graphInput.worldPosition = position;
    graphInput.normal = sample.normal;
    graphInput.vertexNormal = sample.front ? sample.vertexNormal : -sample.vertexNormal;
    graphInput.tangent = graphFrame.xyz;
    graphInput.viewDirection = -ray.direction;
    graphInput.parameters = material.graphParameters;
    graphInput.baseColor = sample.base.rgb;
    graphInput.metallic = sample.metallic;
    graphInput.roughness = sample.roughness;
    graphInput.materialScalars = float4(sample.base.a, material.factors.w, material.factors.z, material.lobes.extinction.w);
    graphInput.emission = sample.emission;
    graphInput.bitangent = graphFrame.w != 0.0f ? ptUnit(cross(sample.vertexNormal, graphFrame.xyz) * graphFrame.w, float3(0))
        : float3(0);
    // The dispatch stores the camera in the first ten graph uniform slots (PathTracingScene.graphViewSlotCount).
    graphInput.cameraPosition = graphUniforms[0].xyz;
    graphInput.cameraClip = float2(graphUniforms[0].w, graphUniforms[1].x);
    graphInput.viewMatrix = float4x4(graphUniforms[2], graphUniforms[3], graphUniforms[4], graphUniforms[5]);
    graphInput.projectionMatrix = float4x4(graphUniforms[6], graphUniforms[7], graphUniforms[8], graphUniforms[9]);
    graphInput.modelMatrix = transpose(float4x4(
        instance.objectToWorld[0], instance.objectToWorld[1], instance.objectToWorld[2], float4(0, 0, 0, 1)
    ));
    graphInput.normalMatrix = float3x3(
        instance.worldToObject[0].xyz, instance.worldToObject[1].xyz, instance.worldToObject[2].xyz
    );
    float2 imageSize = max(ray.screen.zw, float2(1));
    graphInput.screenUV = ray.screen.xy / imageSize;
    graphInput.screenCoordinate = ray.screen.xy;
    graphInput.viewportSize = imageSize;
    graphInput.screenSize = imageSize;
    return graphInput;
}

static FrostfieldGraphResources ptMakeGraphResources(
    thread const PTScene& scene, device const PTMaterial& material, PTTexturing texturing,
    constant float4* graphUniforms, device const uint* graphTable
) {
    return {
        graphUniforms, graphTable, scene.maps, scene.textures, scene.textureCount,
        material.graphResources.x, material.graphResources.y,
        {texturing.gradients[0], texturing.gradients[1]}, texturing.footprint.major, texturing.footprint.minor
    };
}
#endif

#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
static float3 ptGraphCoverage(
    thread const PTScene& scene, device const PTInstance& instance, thread const PTTriangle& triangle,
    float2 barycentric, float distance, PTRay ray, PTTexturing texturing
) {
    device const PTMaterial& material = scene.materials[triangle.meta.x];
    PTSampledSurface sample = ptSampleSurface(scene, instance, material, triangle, barycentric, ray, texturing);
    float4 frame = ptTangentFrame(sample.vertexNormal, triangle, 0u);
    FrostfieldSurfaceInput input = ptMakeGraphInput(
        instance, material, sample, texturing.uv, frame, ray.origin + distance * ray.direction, ray, scene.graphUniforms
    );
    FrostfieldSurfaceOutput output = {};
    output.opacity = 1.0f;
    output.alphaTest = -1.0f;
    output.mask = true;
    FrostfieldGraphResources resources = ptMakeGraphResources(scene, material, texturing, scene.graphUniforms, scene.graphTable);
    ptCoverageGraph(triangle.meta.z, input, resources, output);
    return float3(sample.base.a * output.opacity, output.alphaTest, output.mask ? 1.0f : 0.0f);
}
#endif

static PTSurface ptSurface(
    PTHit hit, PTRay ray, thread const PTScene& scene, constant float4* graphUniforms, device const uint* graphTable,
    bool readLightMap = false
) {
    device const PTInstance& instance = scene.instances[hit.instance];
    float2 bary = hit.barycentric;
    PTTriangle triangle = ptWorldTriangle(instance, ptLocalTriangle(instance, hit.index), bary);
    uint materialIndex = triangle.meta.x;
    device const PTMaterial& material = scene.materials[materialIndex];
    PTTexturing texturing = ptTexturing(triangle, bary, ray.direction, ray.width + ray.spread * hit.distance);
    PTSampledSurface sample = ptSampleSurface(scene, instance, material, triangle, bary, ray, texturing);
    float3 position = ray.origin + hit.distance * ray.direction;
    float3 geometric = sample.geometric, normal = sample.normal, vertexNormal = sample.vertexNormal;
    bool front = sample.front;
    float3 base = sample.base.rgb, emission = sample.emission;
    float alpha = sample.base.a, metallic = sample.metallic, roughness = sample.roughness;
    float transmissionMask = 1.0f;
#ifdef FROSTFIELD_HAS_SURFACE_GRAPHS
    if (triangle.meta.z != 0u) {
        float4 graphFrame = ptTangentFrame(vertexNormal, triangle, 0u);
        FrostfieldSurfaceInput graphInput = ptMakeGraphInput(
            instance, material, sample, texturing.uv, graphFrame, position, ray, graphUniforms
        );
        FrostfieldSurfaceOutput graphOutput = {
            base, metallic, roughness, emission, transmissionMask, 1.0f, float3(0, 0, 1), false, 1.0f, -1.0f, true
        };
        FrostfieldGraphResources resources = ptMakeGraphResources(scene, material, texturing, graphUniforms, graphTable);
        ptSurfaceGraph(triangle.meta.z, graphInput, resources, graphOutput);
        base = graphOutput.baseColor;
        metallic = graphOutput.metallic;
        roughness = graphOutput.roughness;
        emission = graphOutput.emission;
        transmissionMask = graphOutput.transmissionMask;
        alpha *= graphOutput.opacity;
        if (graphOutput.hasNormal && graphFrame.w != 0.0f) {
            // Path tracing derives occlusion from geometry; a graph normal replaces the authored map.
            float3 mapped = ptNormalMap(vertexNormal, graphOutput.normal * 0.5f + 0.5f, 1.0f, triangle, 0u, front);
            if (dot(mapped, geometric) > 0.0f && dot(mapped, -ray.direction) > 1e-4f) { normal = mapped; }
        }
    }
#endif
    device const PTMaterialExtensions& layers = material.extensions;
    PTSurface surface = {
        position, normal, geometric, saturate(base), max(emission, float3(0)),
        saturate(material.optical.rgb), metallic, roughness,
        saturate(material.factors.z * transmissionMask * ptLayerMap(scene, material, 2u, texturing).r),
        material.factors.w, material.optical.w,
        saturate(material.layers.y * ptLayerMap(scene, material, 0u, texturing).r),
        saturate(material.layers.z * ptLayerMap(scene, material, 1u, texturing).g),
        saturate(alpha), front, material.layers.w > 0.5f, materialIndex
    };
    surface.unlit = layers.options.x > 0.5f;
    surface.coatNormal = vertexNormal;
    if (surface.unlit) {
        if (layers.options.x == 1.0f) { surface.emission = surface.baseColor; }
        return surface;
    }
    device const PTMaterialLobes& lobes = material.lobes;
    surface.lobeFlags = lobes.flags.x;
    surface.retroreflectivity = lobes.surface.w;
    bool objectCoat = (lobes.flags.w & 2u) != 0u;
    if (surface.coat > 0.0f && (lobes.surface.y != 0.0f || objectCoat) && lobes.textures[0] != ptMissing) {
        // Clearcoat maps have their own tangent frame or use the object's normal matrix.
        float3 texel = ptLobeMap(scene, material, 0u, texturing).rgb;
        float3 mapped = objectCoat ? ptObjectNormalMap(instance, texel, lobes.flags.z, front, vertexNormal) : ptNormalMap(
            vertexNormal, texel, lobes.surface.y, triangle, lobes.coordinates.x & 1u, front,
            lobes.transforms[0], lobes.transforms[1], lobes.flags.z
        );
        if (dot(mapped, geometric) > 0.0f && dot(mapped, -ray.direction) > 1e-4f) { surface.coatNormal = mapped; }
    }
    surface.diffuseRoughness = saturate(lobes.surface.x * ptLobeMap(scene, material, 3u, texturing).r);
    if (readLightMap && lobes.textures[5] != ptMissing) {
        float3 irradiance = ptLobeMap(scene, material, 5u, texturing).rgb;
        irradiance = max(select(float3(0), irradiance, isfinite(irradiance)), 0.0f);
        surface.bakedIrradiance = min(irradiance * (lobes.surface.z * ptInvPi), float3(1e30f));
    }
    surface.translucency = saturate(lobes.translucency.a * ptLobeMap(scene, material, 1u, texturing).a);
    surface.translucencyColor = saturate(lobes.translucency.rgb * ptLobeMap(scene, material, 2u, texturing).rgb);
    surface.scatterColor = saturate(lobes.scattering.rgb * ptLobeMap(scene, material, 4u, texturing).rgb);
    if ((surface.lobeFlags & ptLobeSubsurfaceColor) != 0u) {
        // A zero mean free path scatters at the entry point, so the share shades as diffuse with the subsurface color.
        surface.baseColor = mix(surface.baseColor, surface.scatterColor, surface.translucency);
        surface.translucency = 0.0f;
    }
    surface.dispersion = layers.options.y;
    surface.specularModulated = layers.options.z > 0.5f;
    surface.specularWeight = saturate(layers.specular.a * ptLayerMap(scene, material, 3u, texturing).a);
    surface.specularColor = layers.specular.rgb * saturate(ptLayerMap(scene, material, 4u, texturing).rgb);
    surface.sheenColor = saturate(layers.sheen.rgb * ptLayerMap(scene, material, 5u, texturing).rgb);
    surface.sheenRoughness = saturate(layers.sheen.a * ptLayerMap(scene, material, 6u, texturing).a);
    float strength = layers.anisotropy.x;
    if (strength > 0.0f) {
        float2 direction = float2(1, 0);
        if (layers.textures[7] != ptMissing) {
            float3 texel = saturate(ptLayerMap(scene, material, 7u, texturing).rgb);
            direction = texel.rg * 2.0f - 1.0f;
            strength *= texel.b;
        }
        float2 rotation = layers.anisotropy.yz;
        direction = float2(
            rotation.x * direction.x - rotation.y * direction.y, rotation.y * direction.x + rotation.x * direction.y
        );
        // The mesh tangent frame from primary UVs, as for surface graphs.
        float4 frame = ptTangentFrame(vertexNormal, triangle, 0u);
        float3 world = frame.xyz * direction.x + cross(vertexNormal, frame.xyz) * (frame.w * direction.y);
        float3 tangent = world - normal * dot(normal, world);
        if (frame.w != 0.0f && strength > 0.0f && dot(tangent, tangent) > 1e-12f) {
            surface.anisotropyDirection = normalize(tangent);
            surface.anisotropy = strength;
        }
    }
    surface.iridescence = saturate(layers.iridescence.x * ptLayerMap(scene, material, 8u, texturing).r);
    if (surface.iridescence > 0.0f) {
        float thickness = saturate(ptLayerMap(scene, material, 9u, texturing).g);
        surface.iridescenceThickness = mix(layers.iridescence.z, layers.iridescence.w, thickness);
        surface.iridescenceIOR = layers.iridescence.y;
    }
    return surface;
}

struct PTBSDF {
    float3 base, tint, outgoing;
    float3x3 frame;
    // GGX alpha along the frame's tangent and bitangent; equal unless anisotropic.
    float2 alpha;
    float metal, transmission, coat, coatAlpha, eta, coatScale;
    float diffuseProbability, baseProbability, coatProbability, sheenProbability;
    bool thin, infiniteIOR;
    // KHR material extension layers.
    float3 specularColor, sheenColor;
    float specularWeight, sheenRoughness, iridescence, iridescenceThickness, iridescenceIOR;
    bool specularModulated;
    // The clearcoat lobe's own frame and view direction, which a coat normal map tilts.
    float3x3 coatFrame;
    float3 coatOutgoing;
    // KHR_materials_diffuse_transmission (or a subsurface walk's entry) and diffuse roughness.
    float3 translucencyColor;
    float translucency, diffuseRoughness, translucencyProbability;
    // The authored BRDF blend and the fraction of the base proposal assigned to retroreflection.
    float retroreflectivity, retroProbability;
};

// value includes the cosine of each lobe's own normal.
struct PTEvaluation {
    float3 value;
    float pdf;
};

struct PTScatter {
    float3 direction, weight;
    float pdf;
    bool transmitted;
    // Sampled from the diffuse transmission lobe, which crosses without refracting.
    bool diffuse;
    // The straight-through delta of an index-matched boundary, which leaves the path's direction unchanged.
    bool straight;
};

static float ptFresnel(float cosine, float eta, bool infiniteIOR) {
    if (infiniteIOR) { return 1.0f; }
    if (eta == 1.0f) { return 0.0f; }
    float incident = saturate(cosine);
    float ratio = 1.0f / eta;
    float sineTransmitted2 = (1.0f - incident * incident) * ratio * ratio;
    if (sineTransmitted2 >= 1.0f) { return 1.0f; }
    float transmitted = sqrt(max(0.0f, 1.0f - sineTransmitted2));
    float a = (incident - eta * transmitted) / max(incident + eta * transmitted, 1e-12f);
    float b = (eta * incident - transmitted) / max(eta * incident + transmitted, 1e-12f);
    return saturate(0.5f * (a * a + b * b));
}

static float ptSchlickWeight(float cosine) {
    float x = 1.0f - saturate(cosine);
    float square = x * x;
    return square * square * x;
}

static float ptGGXDistribution(float cosine, float alpha) {
    float a2 = alpha * alpha;
    float denominator = (1.0f - cosine * cosine) + a2 * cosine * cosine;
    return ptInvPi * a2 / max(denominator * denominator, 1e-20f);
}

static float ptGGXRoot(float cosine, float alpha) {
    float a2 = alpha * alpha;
    return sqrt(a2 + (1.0f - a2) * cosine * cosine);
}

static float ptGGXVisibility(float outgoing, float incoming, float alpha) {
    return 0.5f / max(incoming * ptGGXRoot(outgoing, alpha) + outgoing * ptGGXRoot(incoming, alpha), 1e-12f);
}

static float ptReflectionPDF(float distribution, float outgoing, float alpha) {
    return distribution * 0.5f / max(outgoing + ptGGXRoot(outgoing, alpha), 1e-12f);
}

// GGX for the base lobe, anisotropic when the frame's alphas differ.
static float ptGGXD(float3 halfway, float2 alpha) {
    if (alpha.x == alpha.y) { return ptGGXDistribution(saturate(halfway.z), alpha.x); }
    return frostfieldAnisotropicGGX(halfway, alpha);
}

static float ptGGXV(float3 outgoing, float3 incoming, float2 alpha) {
    if (alpha.x == alpha.y) { return ptGGXVisibility(outgoing.z, abs(incoming.z), alpha.x); }
    return frostfieldAnisotropicVisibility(outgoing, incoming, alpha);
}

// The reflected-direction density of visible-normal sampling, D G1(o) / (4 o.z).
static float ptReflectionPDF2(float distribution, float3 outgoing, float2 alpha) {
    if (alpha.x == alpha.y) { return ptReflectionPDF(distribution, outgoing.z, alpha.x); }
    return distribution * 0.5f / max(outgoing.z + frostfieldAnisotropicRoot(outgoing, alpha), 1e-12f);
}

// Sample the projected, stretched GGX visible-normal distribution.
static float3 ptVisibleNormal(float3 outgoing, float2 alpha, float2 random) {
    float3 stretched = normalize(float3(alpha * outgoing.xy, outgoing.z));
    float3 x = ptUnit(cross(float3(0, 0, 1), stretched), float3(1, 0, 0));
    float3 y = cross(stretched, x);
    float2 disk = ptDisk(random);
    float horizon = sqrt(max(0.0f, 1.0f - disk.x * disk.x));
    disk.y = mix(horizon, disk.y, (1.0f + stretched.z) * 0.5f);
    float z = sqrt(max(0.0f, 1.0f - dot(disk, disk)));
    float3 halfway = disk.x * x + disk.y * y + z * stretched;
    return normalize(float3(alpha * halfway.xy, max(halfway.z, 1e-6f)));
}

// Dielectric reflectance. KHR_materials_specular remaps the exact Fresnel curve from
// [F0, 1] to [min(F0 color, 1) weight, weight]; iridescence then mixes in the thin film.
static float3 ptDielectric(PTBSDF bsdf, float cosine) {
    float exact = ptFresnel(cosine, bsdf.eta, bsdf.infiniteIOR);
    float3 result = float3(exact);
    if (!bsdf.specularModulated && bsdf.iridescence <= 0.0f) { return result; }
    float ratio = (bsdf.eta - 1.0f) / (bsdf.eta + 1.0f);
    float f0 = bsdf.infiniteIOR ? 1.0f : ratio * ratio;
    float3 baseF0 = float3(f0);
    if (bsdf.specularModulated) {
        float along = f0 < 1.0f ? saturate((exact - f0) / (1.0f - f0)) : 1.0f;
        baseF0 = min(f0 * bsdf.specularColor, 1.0f) * bsdf.specularWeight;
        result = bsdf.specularWeight * mix(min(f0 * bsdf.specularColor, 1.0f), float3(1.0f), along);
    }
    if (bsdf.iridescence > 0.0f) {
        float3 film = frostfieldIridescence(1.0f, bsdf.iridescenceIOR, cosine, bsdf.iridescenceThickness, baseF0);
        result = mix(result, film, bsdf.iridescence);
    }
    return result;
}

// Schlick conductor reflectance, mixed toward a thin film over the base color.
static float3 ptConductor(PTBSDF bsdf, float cosine) {
    float3 conductor = mix(bsdf.base, float3(1), ptSchlickWeight(cosine));
    if (bsdf.iridescence > 0.0f) {
        float3 film = frostfieldIridescence(1.0f, bsdf.iridescenceIOR, cosine, bsdf.iridescenceThickness, bsdf.base);
        conductor = mix(conductor, film, bsdf.iridescence);
    }
    return conductor;
}

static void ptConfigureFresnel(thread PTBSDF& bsdf, PTSurface surface, float eta) {
    bsdf.base = surface.baseColor;
    bsdf.eta = eta;
    bsdf.infiniteIOR = surface.ior == 0.0f;
    bsdf.specularModulated = surface.specularModulated;
    bsdf.specularColor = surface.specularColor;
    bsdf.specularWeight = surface.specularWeight;
    bsdf.iridescence = surface.iridescence;
    bsdf.iridescenceThickness = surface.iridescenceThickness;
    bsdf.iridescenceIOR = surface.iridescenceIOR;
}

static float ptCoatFresnel(float cosine) {
    return 0.04f + 0.96f * ptSchlickWeight(cosine);
}

static PTBSDF ptBSDF(PTSurface surface, float3 outgoing, float eta) {
    PTBSDF bsdf;
    float alpha = max(surface.roughness * surface.roughness, 0.001f);
    float3x3 frame = ptFrame(surface.normal);
    bsdf.alpha = float2(alpha);
    if (surface.anisotropy > 0.0f) {
        // Align the frame with the anisotropy direction (KHR_materials_anisotropy).
        frame = float3x3(surface.anisotropyDirection, cross(surface.normal, surface.anisotropyDirection), surface.normal);
        bsdf.alpha = frostfieldAnisotropicAlpha(alpha, surface.anisotropy);
    }
    float3 localOutgoing = transpose(frame) * outgoing;
    ptConfigureFresnel(bsdf, surface, eta);
    bsdf.tint = surface.thin || surface.front ? surface.baseColor : float3(1);
    bsdf.outgoing = localOutgoing;
    bsdf.frame = frame;
    bsdf.metal = surface.metallic;
    bsdf.transmission = bsdf.infiniteIOR ? 0.0f : surface.transmission * (1.0f - surface.metallic);
    bsdf.coat = surface.coat;
    bsdf.coatAlpha = max(surface.coatRoughness * surface.coatRoughness, 0.001f);
    bsdf.thin = surface.thin;
    bsdf.sheenColor = surface.sheenColor;
    bsdf.sheenRoughness = surface.sheenRoughness;
    bsdf.coatFrame = ptFrame(surface.coatNormal);
    bsdf.coatOutgoing = transpose(bsdf.coatFrame) * outgoing;
    bsdf.coatScale = 1.0f - surface.coat * ptCoatFresnel(bsdf.coatOutgoing.z);
    bsdf.translucency = surface.translucency;
    bsdf.translucencyColor = surface.translucencyColor;
    bsdf.diffuseRoughness = surface.diffuseRoughness;
    bsdf.retroreflectivity = surface.retroreflectivity;
    // Reserve an ordinary VNDF proposal for refraction even at full
    // retroreflectivity. Null boundaries have no reflected contribution.
    bool straight = (surface.lobeFlags & ptLobeNullBoundary) != 0u && abs(eta - 1.0f) < 1e-5f;
    bsdf.retroProbability = straight ? 0.0f : surface.retroreflectivity * (bsdf.transmission > 0.0f ? 0.5f : 1.0f);

    float dielectric = ptMax(ptDielectric(bsdf, localOutgoing.z));
    float reflection = mix(dielectric, ptMax(surface.baseColor), surface.metallic);
    float opaque = (1.0f - surface.metallic) * (1.0f - surface.transmission) * (1.0f - dielectric);
    float diffuseWeight = ptMax(surface.baseColor) * opaque * (1.0f - surface.translucency);
    float translucencyWeight = ptMax(surface.translucencyColor) * opaque * surface.translucency;
    float baseWeight = max(reflection + bsdf.transmission * (1.0f - dielectric), 0.03f);
    float coatWeight = surface.coat * 0.25f;
    // Sheen shares cosine-weighted sampling with the diffuse lobe.
    float sheenWeight = ptMax(surface.sheenColor) > 0.0f
        ? min(ptMax(surface.sheenColor) * frostfieldSheenAlbedo(localOutgoing.z, surface.sheenRoughness), 1.0f) : 0.0f;
    float total = diffuseWeight + translucencyWeight + baseWeight + coatWeight + sheenWeight;
    bsdf.diffuseProbability = diffuseWeight / total;
    bsdf.translucencyProbability = translucencyWeight / total;
    bsdf.baseProbability = baseWeight / total;
    bsdf.coatProbability = coatWeight / total;
    bsdf.sheenProbability = sheenWeight / total;
    return bsdf;
}

// A noise-free specular color guide at the view angle, before lighting. It
// uses the same Fresnel curves, thin-film material and coat attenuation as the BSDF.
static float3 ptSpecularAlbedo(PTSurface surface, float3 outgoing) {
    if (surface.unlit) { return float3(0); }
    bool startsInside = !surface.front && !surface.thin && surface.metallic < 1.0f
        && (surface.transmission > 0.0f || (surface.lobeFlags & ptLobeMedium) != 0u);
    float eta = surface.ior > 0.0f ? clamp(startsInside ? 1.0f / surface.ior : surface.ior, 1e-6f, 1e6f) : 1.0f;
    PTBSDF bsdf = {};
    ptConfigureFresnel(bsdf, surface, eta);
    float noV = saturate(dot(surface.normal, outgoing));
    float3 base = mix(ptDielectric(bsdf, noV), ptConductor(bsdf, noV), surface.metallic);
    float coat = surface.coat * ptCoatFresnel(dot(surface.coatNormal, outgoing));
    return saturate(base * (1.0f - coat) + coat);
}

// The opaque diffuse lobe without its weight: Lambertian, or EON for a rough diffuse layer.
static float3 ptDiffuseLobe(float3 albedo, float roughness, float3 incoming, float3 outgoing) {
    if (roughness <= 0.0f) { return albedo * ptInvPi; }
    float noL = incoming.z;
    float noV = outgoing.z;
    return frostfieldEON(albedo, roughness, noL, noV, dot(incoming, outgoing) - noL * noV);
}

// A diffuse-only irradiance map has no incident-direction information. Apply
// the hemispherical albedo of rough diffuse and normal-incidence dielectric
// reflectance, as for a raster indirect diffuse term.
static float3 ptBakedDiffuse(PTBSDF bsdf, float3 irradiance) {
    float diffuse = (1.0f - bsdf.metal - bsdf.transmission) * (1.0f - bsdf.translucency);
    float3 reflectance = 1.0f - ptDielectric(bsdf, 1.0f);
    float3 albedo = bsdf.diffuseRoughness > 0.0f
        ? frostfieldEONAlbedo(bsdf.base, bsdf.diffuseRoughness, bsdf.outgoing.z) : bsdf.base;
    return irradiance * albedo * reflectance * (diffuse * bsdf.coatScale);
}

// Sheen above the base layer: scales what lies beneath and adds the Charlie lobe.
static float3 ptSheen(PTBSDF bsdf, float3 beneath, float noH, float noV, float noL) {
    if (ptMax(bsdf.sheenColor) <= 0.0f) { return beneath; }
    return beneath * frostfieldSheenScaling(bsdf.sheenColor, bsdf.sheenRoughness, noV, noL)
        + bsdf.sheenColor * (frostfieldCharlie(noH, bsdf.sheenRoughness) * frostfieldSheenVisibility(noL, noV));
}

static float ptChooseReflection(PTBSDF bsdf, float dielectric, float cosine) {
    float metal = bsdf.iridescence > 0.0f ? ptMax(ptConductor(bsdf, cosine))
        : mix(ptMax(bsdf.base), 1.0f, ptSchlickWeight(cosine));
    float reflection = mix(dielectric, metal, bsdf.metal);
    float transmission = bsdf.transmission * (1.0f - dielectric) * ptMax(bsdf.tint);
    float total = reflection + transmission;
    return total > 0.0f ? reflection / total : 1.0f;
}

// The mirrored outgoing direction in the shading frame, including anisotropy.
static float3 ptRetroView(PTBSDF bsdf) {
    return float3(-bsdf.outgoing.xy, bsdf.outgoing.z);
}

static float ptOrdinaryBaseProbability(PTBSDF bsdf) {
    return bsdf.baseProbability * (1.0f - bsdf.retroProbability);
}

static PTEvaluation ptReflection(PTBSDF bsdf, float3 incoming) {
    PTEvaluation result = {float3(0), 0};
    float noL = incoming.z;
    float noV = bsdf.outgoing.z;
    if (noL > 0.0f && noV > 0.0f) {
        float3 halfway = ptUnit(bsdf.outgoing + incoming, float3(0, 0, 1));
        float voH = saturate(dot(bsdf.outgoing, halfway));
        float noH = saturate(halfway.z);
        float3 dielectricColor = ptDielectric(bsdf, voH);
        float dielectric = ptMax(dielectricColor);
        float3 conductor = ptConductor(bsdf, voH);
        float3 fresnel = mix(dielectricColor, conductor, bsdf.metal);
        float distribution = ptGGXD(halfway, bsdf.alpha);
        float specular = distribution * ptGGXV(bsdf.outgoing, incoming, bsdf.alpha);
        float diffuse = (1.0f - bsdf.metal - bsdf.transmission) * (1.0f - dielectric) * (1.0f - bsdf.translucency);
        float3 diffuseLobe = ptDiffuseLobe(bsdf.base, bsdf.diffuseRoughness, incoming, bsdf.outgoing);
        float3 layered = specular * fresnel + diffuseLobe * diffuse;
        float basePDF = bsdf.baseProbability * ptChooseReflection(bsdf, dielectric, voH)
            * ptReflectionPDF2(distribution, bsdf.outgoing, bsdf.alpha);
        if (bsdf.retroreflectivity > 0.0f) {
            float3 retroView = ptRetroView(bsdf);
            float3 retroHalf = ptUnit(retroView + incoming, float3(0, 0, 1));
            float retroVoH = saturate(dot(retroView, retroHalf));
            float3 retroDielectric = ptDielectric(bsdf, retroVoH);
            float3 retroFresnel = mix(retroDielectric, ptConductor(bsdf, retroVoH), bsdf.metal);
            float retroD = ptGGXD(retroHalf, bsdf.alpha);
            float retroSpecular = retroD * ptGGXV(retroView, incoming, bsdf.alpha);
            float retroDiffuse = (1.0f - bsdf.metal - bsdf.transmission)
                * (1.0f - ptMax(retroDielectric)) * (1.0f - bsdf.translucency);
            layered = (1.0f - bsdf.retroreflectivity) * layered
                + bsdf.retroreflectivity * (retroSpecular * retroFresnel + diffuseLobe * retroDiffuse);
            // MIS uses the same mixture as the sampler, including both proposals
            // where their reflected directions overlap.
            float retroPDF = bsdf.baseProbability * ptReflectionPDF2(retroD, retroView, bsdf.alpha);
            basePDF = basePDF * (1.0f - bsdf.retroProbability) + retroPDF * bsdf.retroProbability;
        }
        result.value = bsdf.coatScale * ptSheen(bsdf, layered, noH, noV, noL) * noL;
        result.pdf = (bsdf.diffuseProbability + bsdf.sheenProbability) * noL * ptInvPi + basePDF;
    }

    if (bsdf.coat > 0.0f) {
        // The coat lobe in its own frame, with its own cosine.
        float3 coatIncoming = transpose(bsdf.coatFrame) * (bsdf.frame * incoming);
        float coatL = coatIncoming.z;
        float coatV = bsdf.coatOutgoing.z;
        if (coatL > 0.0f && coatV > 0.0f) {
            float3 halfway = ptUnit(bsdf.coatOutgoing + coatIncoming, float3(0, 0, 1));
            float coatDistribution = ptGGXDistribution(saturate(halfway.z), bsdf.coatAlpha);
            float coatFresnel = ptCoatFresnel(saturate(dot(bsdf.coatOutgoing, halfway)));
            result.value += bsdf.coat * coatFresnel * coatDistribution * ptGGXVisibility(coatV, coatL, bsdf.coatAlpha) * coatL;
            result.pdf += bsdf.coatProbability * ptReflectionPDF(coatDistribution, coatV, bsdf.coatAlpha);
        }
    }
    return result;
}

// Microfacet transmission (thin or refracting) and diffuse transmission toward a direction below the surface.
static PTEvaluation ptTransmission(PTBSDF bsdf, float3 incoming) {
    PTEvaluation result = {float3(0), 0};
    float noL = -incoming.z;
    float noV = bsdf.outgoing.z;
    if (!(noL > 0.0f && noV > 0.0f)) { return result; }
    float sheenScale = ptMax(bsdf.sheenColor) > 0.0f
        ? frostfieldSheenScaling(bsdf.sheenColor, bsdf.sheenRoughness, noV, noL) : 1.0f;

    if (bsdf.translucency > 0.0f) {
        // KHR_materials_diffuse_transmission: the diffuse BTDF with the Fresnel of the mirrored direction.
        float3 mirrored = float3(incoming.xy, -incoming.z);
        float3 halfway = ptUnit(bsdf.outgoing + mirrored, float3(0, 0, 1));
        float dielectric = ptMax(ptDielectric(bsdf, saturate(dot(bsdf.outgoing, halfway))));
        float weight = bsdf.coatScale * (1.0f - bsdf.metal - bsdf.transmission) * bsdf.translucency * (1.0f - dielectric);
        result.value += ptDiffuseLobe(bsdf.translucencyColor, bsdf.diffuseRoughness, mirrored, bsdf.outgoing)
            * (weight * sheenScale * noL);
        result.pdf += bsdf.translucencyProbability * noL * ptInvPi;
    }
    if (bsdf.transmission == 0.0f || (!bsdf.thin && abs(bsdf.eta - 1.0f) < 1e-5f)) {
        // Index-matched volumes transmit only along the straight-through delta.
        return result;
    }

    if (bsdf.thin) {
        // Reflect the sample across the sheet so a smooth thin wall keeps its ray direction.
        float3 opposite = float3(incoming.xy, -incoming.z);
        float3 halfway = ptUnit(bsdf.outgoing + opposite, float3(0, 0, 1));
        float voH = saturate(dot(bsdf.outgoing, halfway));
        float dielectric = ptMax(ptDielectric(bsdf, voH));
        float distribution = ptGGXD(halfway, bsdf.alpha);
        result.value += bsdf.tint * (bsdf.coatScale * bsdf.transmission * (1.0f - dielectric)
            * distribution * ptGGXV(bsdf.outgoing, incoming, bsdf.alpha) * sheenScale * noL);
        result.pdf += ptOrdinaryBaseProbability(bsdf) * (1.0f - ptChooseReflection(bsdf, dielectric, voH))
            * ptReflectionPDF2(distribution, bsdf.outgoing, bsdf.alpha);
        return result;
    }

    float3 halfway = ptUnit(bsdf.outgoing + incoming * bsdf.eta, float3(0));
    if (halfway.z < 0.0f) { halfway = -halfway; }
    float voH = dot(bsdf.outgoing, halfway);
    float liH = dot(incoming, halfway);
    if (!(voH > 0.0f && liH < 0.0f)) { return result; }

    float denominator = liH + voH / bsdf.eta;
    float denominator2 = denominator * denominator;
    if (denominator2 < 1e-16f) { return result; }
    float dielectric = ptMax(ptDielectric(bsdf, voH));
    float distribution = ptGGXD(halfway, bsdf.alpha);
    float jacobian = -liH / denominator2;
    float visiblePDF = 4.0f * voH * ptReflectionPDF2(distribution, bsdf.outgoing, bsdf.alpha);
    result.pdf += ptOrdinaryBaseProbability(bsdf) * (1.0f - ptChooseReflection(bsdf, dielectric, voH)) * visiblePDF * jacobian;
    float inverseEta = 1.0f / bsdf.eta;
    result.value += bsdf.tint * (bsdf.coatScale * bsdf.transmission * (1.0f - dielectric)
        * distribution * ptGGXV(bsdf.outgoing, incoming, bsdf.alpha) * (4.0f * voH * jacobian * inverseEta * inverseEta)
        * sheenScale * noL);
    return result;
}

static PTScatter ptSampleBSDF(PTBSDF bsdf, thread PTSampler& state) {
    PTScatter scatter = {float3(0), float3(0), 0, false, false, false};
    if (bsdf.outgoing.z <= 0.0f) { return scatter; }

    float select = ptRandom(state);
    float2 random = ptRandom2(state);
    float3 incoming;
    float diffuseEnd = bsdf.diffuseProbability + bsdf.sheenProbability;
    float translucentEnd = diffuseEnd + bsdf.translucencyProbability;
    float coatEnd = translucentEnd + bsdf.coatProbability;
    if (select < diffuseEnd) {
        // Cosine-weighted sampling serves both the diffuse and the sheen lobes.
        float2 disk = ptDisk(random);
        incoming = float3(disk, sqrt(max(0.0f, 1.0f - dot(disk, disk))));
    } else if (select < translucentEnd) {
        // The diffuse transmission lobe: cosine-weighted below the surface.
        float2 disk = ptDisk(random);
        incoming = float3(disk, -sqrt(max(0.0f, 1.0f - dot(disk, disk))));
        scatter.transmitted = true;
        scatter.diffuse = true;
    } else if (select < coatEnd) {
        float3 halfway = ptVisibleNormal(bsdf.coatOutgoing, float2(bsdf.coatAlpha), random);
        float3 coatIncoming = reflect(-bsdf.coatOutgoing, halfway);
        if (coatIncoming.z <= 0.0f) { return scatter; }
        incoming = transpose(bsdf.frame) * (bsdf.coatFrame * coatIncoming);
    } else if (bsdf.retroProbability > 0.0f && select < mix(coatEnd, 1.0f, bsdf.retroProbability)) {
        float3 retroView = ptRetroView(bsdf);
        float3 halfway = ptVisibleNormal(retroView, bsdf.alpha, random);
        incoming = reflect(-retroView, halfway);
        if (incoming.z <= 0.0f) { return scatter; }
    } else {
        float3 halfway = ptVisibleNormal(bsdf.outgoing, bsdf.alpha, random);
        float cosine = saturate(dot(bsdf.outgoing, halfway));
        float dielectric = ptMax(ptDielectric(bsdf, cosine));
        float reflectProbability = ptChooseReflection(bsdf, dielectric, cosine);
        if (ptRandom(state) < reflectProbability) {
            incoming = reflect(-bsdf.outgoing, halfway);
            if (incoming.z <= 0.0f) { return scatter; }
        } else {
            scatter.transmitted = true;
            if (!bsdf.thin && abs(bsdf.eta - 1.0f) < 1e-5f) {
                // Index matching is a straight-through delta, even when roughness is authored.
                scatter.direction = -(bsdf.frame * bsdf.outgoing);
                scatter.weight = bsdf.tint * (bsdf.coatScale * bsdf.transmission * (1.0f - dielectric)
                    / max(ptOrdinaryBaseProbability(bsdf) * (1.0f - reflectProbability), 1e-12f));
                scatter.straight = true;
                return scatter;
            }
            if (bsdf.thin) {
                incoming = reflect(-bsdf.outgoing, halfway);
                incoming.z = -incoming.z;
            } else {
                incoming = refract(-bsdf.outgoing, halfway, 1.0f / bsdf.eta);
            }
            if (incoming.z >= 0.0f) { return scatter; }
        }
    }

    // A coat sample below the base's shading hemisphere still reflects.
    PTEvaluation evaluation = scatter.transmitted ? ptTransmission(bsdf, incoming) : ptReflection(bsdf, incoming);
    if (evaluation.pdf > 0.0f && isfinite(evaluation.pdf)) {
        scatter.direction = bsdf.frame * incoming;
        scatter.weight = evaluation.value / evaluation.pdf;
        scatter.pdf = evaluation.pdf;
    }
    return scatter;
}

static float ptMIS(float first, float second) {
    float scale = max(first, second);
    if (!(scale > 0.0f) || !isfinite(scale)) { return 0.0f; }
    float a = first / scale;
    float b = second / scale;
    return a * a / (a * a + b * b);
}

static bool ptOccluded(
    PTRay shadow, thread const PTScene& scene, device const PTAreaLight* lights, uint lightCount
) {
    return ptGeometry(shadow, true, scene, false, true).index != ptMissing
        || ptAreaLights(shadow, false, lights, lightCount).index != ptMissing;
}

constant uint ptScatterSurface = 0u;  // a surface BSDF
constant uint ptScatterMedium = 1u;   // a medium's phase function
constant uint ptScatterExit = 2u;     // a subsurface walk leaving the volume diffusely

// One closed volume on a path's medium stack.
struct PTMedium {
    uint instance;
    float ior;
    float3 extinction;  // per world unit
    float3 albedo;      // single-scattering albedo
    float anisotropy;
    uint flags;         // ptLobe flags of the volume's material
};

// The ray starts in the first count entries of the path's current volume stack.
struct PTShadowMedium {
    thread const PTMedium* volumes;
    uint count;
};

// A point that gathers next-event light.
struct PTScatterer {
    float3 position;
    // Surface: the geometric normal on the arriving ray's side. Exit: the outward geometric normal.
    float3 normal;
    // Medium: the arriving propagation direction. Exit: the outward shading normal.
    float3 direction;
    // Exit: the diffuse transmission color.
    float3 tint;
    float anisotropy;
    uint kind;
    // Surface: whether directions on the other side are gathered through its transmission lobes.
    bool transmits;
    // The media of shadow rays on the arriving side and beyond a transmitting surface.
    PTShadowMedium near, far;
};

// The scattered radiance per unit incident radiance toward `direction` (cosine included) and its sampling density.
static PTEvaluation ptScatterEvaluate(thread const PTScatterer& point, thread const PTBSDF& bsdf, float3 direction) {
    PTEvaluation none = {float3(0), 0.0f};
    if (point.kind == ptScatterMedium) {
        float phase = frostfieldHenyeyGreenstein(dot(point.direction, direction), point.anisotropy);
        return {float3(phase), phase};
    }
    if (point.kind == ptScatterExit) {
        float cosine = dot(point.direction, direction);
        if (!(cosine > 0.0f) || dot(point.normal, direction) <= 0.0f) { return none; }
        return {point.tint * (cosine * ptInvPi), cosine * ptInvPi};
    }
    float side = dot(point.normal, direction);
    float3 local = transpose(bsdf.frame) * direction;
    if (side > 0.0f) { return ptReflection(bsdf, local); }
    if (side < 0.0f && point.transmits) { return ptTransmission(bsdf, local); }
    return none;
}

// Whether a light toward `direction` can be gathered at all, before evaluating the BSDF.
static bool ptGathers(thread const PTScatterer& point, float3 direction) {
    if (point.kind == ptScatterMedium) { return true; }
    float side = dot(point.normal, direction);
    return side > 0.0f || (side < 0.0f && point.kind == ptScatterSurface && point.transmits);
}

static float3 ptShadowOrigin(thread const PTScatterer& point, float3 direction) {
    return point.kind == ptScatterMedium ? point.position : ptOrigin(point.position, point.normal, direction);
}

static PTShadowMedium ptShadowStart(thread const PTScatterer& point, float3 direction) {
    return point.kind == ptScatterSurface && dot(point.normal, direction) < 0.0f ? point.far : point.near;
}

// The entry and exit distances of a ray through an axis-aligned box; empty when x > y.
static float2 ptBoxInterval(float3 origin, float3 direction, float3 minimum, float3 maximum) {
    float entry = -INFINITY;
    float exit = INFINITY;
    for (uint axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) < 1e-20f) {
            if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis]) { return float2(1, 0); }
            continue;
        }
        float inverse = 1.0f / direction[axis];
        float a = (minimum[axis] - origin[axis]) * inverse;
        float b = (maximum[axis] - origin[axis]) * inverse;
        entry = max(entry, min(a, b));
        exit = min(exit, max(a, b));
    }
    return float2(entry, exit);
}

static float3 ptExtinguish(float3 extinction, float distance) {
    return exp(-extinction * min(max(distance, 0.0f), 1e30f));
}

struct PTFogIntegral {
    float opticalDistance;
    float worldDistance;
    float densityAtEvent;
    bool scattered;
};

// Traverse a clipped ray through piecewise constant voxels. Each iteration
// crosses at least one grid plane or ends the segment. Simultaneous crossings
// step all tied axes, so even a corner ray visits at most the sum of dimensions.
static PTFogIntegral ptFogIntegral(
    PTRay ray, float start, float end, constant PTFogVolume& media, float opticalTarget
) {
    int3 dimensions = int3(media.fogResolution.xyz);
    float3 minimum = media.fogMinimum.xyz;
    float3 extent = media.fogMaximum.xyz - minimum;
    float3 position = fma(ray.direction, float3(start), ray.origin);
    float3 coordinate = saturate((position - minimum) / extent) * float3(dimensions);
    int3 cell = clamp(int3(floor(coordinate)), int3(0), dimensions - 1);
    bool3 positive = ray.direction >= 0.0f;
    int3 step = select(int3(-1), int3(1), positive);
    int3 boundaryOffset = select(int3(0), int3(1), positive);
    bool3 parallel = abs(ray.direction) < 1e-20f;
    float3 distancePerCell = (extent / float3(dimensions)) / select(ray.direction, float3(1), parallel);
    float segmentLimit = end - start;
    float distance = 0.0f, optical = 0.0f;
    int crossingLimit = dimensions.x + dimensions.y + dimensions.z;
    for (int crossing = 0; crossing < crossingLimit; ++crossing) {
        float3 plane = float3(cell + boundaryOffset);
        float3 next = select((plane - coordinate) * distancePerCell, float3(INFINITY), parallel);
        float segmentEnd = max(distance, min(segmentLimit, min(next.x, min(next.y, next.z))));
        uint index = uint((cell.z * dimensions.y + cell.y) * dimensions.x + cell.x);
        float density = media.fogDensity[index];
        float cellOptical = density * (segmentEnd - distance);
        float remaining = opticalTarget - optical;
        if (remaining < cellOptical && density > 0.0f) {
            return {opticalTarget, distance + max(remaining, 0.0f) / density, density, true};
        }
        optical += cellOptical;
        if (segmentEnd >= segmentLimit) { break; }
        distance = segmentEnd;
        cell += select(int3(0), step, next <= segmentEnd);
        if (any(cell < 0) || any(cell >= dimensions)) { break; }
    }
    return {optical, segmentLimit, 0.0f, false};
}

static PTFogIntegral ptFogSegment(
    PTRay ray, float start, float end, constant PTFogVolume& volume, float target
) {
    if (volume.fogResolution.x != 0u) { return ptFogIntegral(ray, start, end, volume, target); }
    float length = end - start;
    return target < length ? PTFogIntegral{target, target, 1.0f, true}
        : PTFogIntegral{length, length, 0.0f, false};
}

// The nearest boundary face at or beyond `query.minimum`, and the parameter at which to
// resume after it. A ray through an edge or vertex hits every face that shares it at nearly
// the same parameter. Faces on the same side count as one crossing, so `resume` steps past
// the whole group. If faces on both sides share the hit, the ray touches a convex edge, or
// slides through a reflex one, and stays on its current side, so the group is skipped. The
// group window spans a few ulps because neighbors evaluate a shared edge with swapped
// operands. A pathological fan of such groups ends after `ptMaximumFogCoincidentGroups`
// queries and keeps the face it reached.
static FrostfieldMeshBVHHit ptFogBoundaryCrossing(
    device const uint4* boundary, FrostfieldMeshBVHRay query, thread float& resume
) {
    FrostfieldMeshBVHHit hit = frostfieldMeshBVHNearest(boundary, query);
    resume = nextafter(hit.parameter, INFINITY);
    for (uint group = 0u; group < ptMaximumFogCoincidentGroups; ++group) {
        if (hit.triangleIndex == frostfieldMeshBVHMiss || all(hit.barycentric > 1e-6f)) { return hit; }
        float scale = max(abs(hit.parameter), 1.0f);
        float reach = hit.parameter + 16.0f * (nextafter(scale, INFINITY) - scale);
        resume = nextafter(reach, INFINITY);
        FrostfieldMeshBVHRay coincident = query;
        coincident.minimum = max(query.minimum, hit.parameter);
        coincident.maximum = min(query.maximum, reach);
        FrostfieldMeshBVHSide opposite = hit.frontFacing ? FrostfieldMeshBVHSide::back : FrostfieldMeshBVHSide::front;
        if (frostfieldMeshBVHNearest(boundary, coincident, opposite).triangleIndex == frostfieldMeshBVHMiss) { return hit; }
        query.minimum = resume;
        hit = frostfieldMeshBVHNearest(boundary, query);
        resume = nextafter(hit.parameter, INFINITY);
    }
    return hit;
}

// Returns the integrated scalar density, or the absolute world ray parameter
// at a collision. The unnormalized local ray preserves that parameter, so a
// non-uniform placement keeps extinction per world distance. After
// `ptMaximumFogCrossings` boundary crossings the region stops contributing along
// the ray: the rest of the segment counts as vacuum for this region.
static PTFogIntegral ptFogVolumeIntegral(
    PTRay ray, float start, float end, constant PTFogVolume& volume, float target
) {
    if (volume.fogBoundary != nullptr) {
        ray.origin = (volume.fogLocalFromWorld * float4(ray.origin, 1.0f)).xyz;
        ray.direction = (volume.fogLocalFromWorld * float4(ray.direction, 0.0f)).xyz;
    }
    float2 interval = ptBoxInterval(ray.origin, ray.direction, volume.fogMinimum.xyz, volume.fogMaximum.xyz);
    float begin = max(start, interval.x), stop = min(end, interval.y);
    if (!(stop > begin)) { return {0.0f, stop, 0.0f, false}; }
    if (volume.fogBoundary == nullptr) {
        PTFogIntegral result = ptFogSegment(ray, begin, stop, volume, target);
        result.worldDistance += begin;
        return result;
    }

    // The next front face enters the volume, so a back face means this ray
    // already starts inside. Query past the clip limit to classify short rays.
    FrostfieldMeshBVHRay query = {ray.origin, ray.direction, begin, INFINITY};
    float resume;
    FrostfieldMeshBVHHit hit = ptFogBoundaryCrossing(volume.fogBoundary, query, resume);
    bool inside = hit.triangleIndex != frostfieldMeshBVHMiss && !hit.frontFacing;
    float optical = 0.0f;
    float cursor = begin;
    for (uint crossings = 0u; cursor < stop; ++crossings) {
        bool missed = hit.triangleIndex == frostfieldMeshBVHMiss;
        if (!missed && crossings >= ptMaximumFogCrossings) { break; }
        float crossing = missed ? INFINITY : hit.parameter;
        float segmentEnd = min(crossing, stop);
        if (inside && segmentEnd > cursor) {
            PTFogIntegral result = ptFogSegment(ray, cursor, segmentEnd, volume, max(target - optical, 0.0f));
            if (result.scattered) { return {target, cursor + result.worldDistance, result.densityAtEvent, true}; }
            optical += result.opticalDistance;
        }
        if (crossing >= stop) { break; }
        inside = hit.frontFacing;
        cursor = max(cursor, crossing);
        query.minimum = resume;
        hit = ptFogBoundaryCrossing(volume.fogBoundary, query, resume);
    }
    return {optical, stop, 0.0f, false};
}

static float3 ptFogVolumeTransmittance(PTRay ray, float start, float end, constant PTFogVolume& volume) {
    float optical = ptFogVolumeIntegral(ray, start, end, volume, INFINITY).opticalDistance;
    return ptExtinguish(volume.fogExtinction.rgb, optical);
}

// Transmittance multiplies across all fog traversed outside closed volumes.
static float3 ptFogTransmittance(PTRay ray, float start, float end, constant PTFogVolume* media) {
    uint count = media[0].fogResolution.w;
    if (count == 0u) { return float3(1); }
    float3 transmittance = ptFogVolumeTransmittance(ray, start, end, media[0]);
    for (uint index = 1u; index < count; ++index) {
        transmittance *= ptFogVolumeTransmittance(ray, start, end, media[index]);
    }
    return transmittance;
}

// The transmittance of a shadow ray: zero when geometry or an area light blocks it, otherwise
// the extinction of the media it crosses. Index-matched boundaries let it pass, tinted by their
// base color where it enters, into or out of their volume.
static float3 ptShadow(
    PTRay shadow, PTShadowMedium medium, constant PTFogVolume* media, thread const PTScene& scene,
    device const PTAreaLight* lights, uint lightCount
) {
    if (!frostfield_path_has_null_boundaries) {
        if (ptOccluded(shadow, scene, lights, lightCount)) { return float3(0); }
        return medium.count > 0u ? ptExtinguish(medium.volumes[medium.count - 1u].extinction, shadow.maximum - shadow.minimum)
            : ptFogTransmittance(shadow, shadow.minimum, shadow.maximum, media);
    }
    struct ShadowLayer {
        float3 extinction;
        float ior;
        uint instance;
    };
    ShadowLayer volumes[4];
    uint count = medium.count;
    for (uint index = 0u; index < count; ++index) {
        thread const PTMedium& volume = medium.volumes[index];
        volumes[index] = {volume.extinction, volume.ior, volume.instance};
    }
    float3 transmittance = float3(1);
    for (uint crossing = 0u; crossing <= 8u; ++crossing) {
        PTHit hit = ptGeometry(shadow, false, scene, false, true);
        PTRay lightRay = shadow;
        lightRay.maximum = hit.distance;
        if (ptAreaLights(lightRay, false, lights, lightCount).index != ptMissing) { return float3(0); }
        transmittance *= count > 0u ? ptExtinguish(volumes[count - 1u].extinction, hit.distance - shadow.minimum)
            : ptFogTransmittance(shadow, shadow.minimum, hit.distance, media);
        if (hit.index == ptMissing) { return transmittance; }
        if (crossing == 8u) { return float3(0); }
        device const PTInstance& instance = scene.instances[hit.instance];
        device const PTMaterial& material = scene.materials[instance.meta.x];
        if ((material.lobes.flags.x & ptLobeNullBoundary) == 0u) { return float3(0); }
        float2 barycentric = hit.barycentric;
        PTTriangle triangle = ptWorldTriangle(instance, ptLocalTriangle(instance, hit.index), barycentric);
        float3 geometric = ptUnit(
            cross(triangle.p1.xyz - triangle.p0.xyz, triangle.p2.xyz - triangle.p0.xyz), -shadow.direction
        );
        bool front = dot(geometric, shadow.direction) < 0.0f;
        if (front) {
            if (count == 4u || (count > 0u && volumes[count - 1u].ior != 1.0f)) { return float3(0); }
            // The straight-through delta of an index-matched surface carries its base color inward.
            PTTexturing texturing = ptTexturing(
                triangle, barycentric, shadow.direction, shadow.width + shadow.spread * hit.distance
            );
            float3 tint = ptMaterialTint(material, triangle, barycentric).rgb * ptMaterialMap(
                scene, material, 0u, material.textureIndices.x, material.textureUVs.x, texturing
            ).rgb;
            transmittance *= saturate(tint);
            volumes[count++] = {material.lobes.extinction.rgb, 1.0f, hit.instance};
        } else {
            if (count == 0u || volumes[count - 1u].instance != hit.instance
                || (count > 1u && volumes[count - 2u].ior != 1.0f)) { return float3(0); }
            --count;
        }
        if (ptMax(transmittance) <= 0.0f) { return float3(0); }
        float3 position = shadow.origin + hit.distance * shadow.direction;
        float3 origin = ptOrigin(position, geometric, shadow.direction);
        float travelled = dot(origin - shadow.origin, shadow.direction);
        shadow.origin = origin;
        shadow.maximum = isfinite(shadow.maximum) ? shadow.maximum - travelled : INFINITY;
        shadow.minimum = 0.0f;
        if (!(shadow.maximum > 0.0f)) { return transmittance; }
    }
    return float3(0);
}

static float3 ptDirect(
    thread const PTScatterer& point, thread const PTBSDF& bsdf, float footprint, thread PTSampler& state,
    thread const PTScene& scene, device const PTAreaLight* lights, uint lightCount, constant PTFogVolume* media
) {
    if (lightCount == 0u) { return float3(0); }
    uint index = min(uint(ptRandom(state) * float(lightCount)), lightCount - 1u);
    device const PTAreaLight& light = lights[index];
    bool disk = light.halfU.w != 0.0f;
    float2 random = ptRandom2(state);
    float2 sample = disk ? ptDisk(random) : 2.0f * random - 1.0f;
    float3 sampled = light.center.xyz + light.halfU.xyz * sample.x + light.halfV.xyz * sample.y;
    float3 delta = sampled - point.position;
    float distance = length(delta);
    if (!(distance > 2e-5f)) { return float3(0); }
    float3 direction = delta / distance;
    if (!ptGathers(point, direction)) { return float3(0); }

    float3 crossAxes = cross(light.halfU.xyz, light.halfV.xyz);
    float areaCosine = -(disk ? ptPi : 4.0f) * dot(crossAxes, direction);
    if (!(areaCosine > 0.0f)) { return float3(0); }
    PTEvaluation evaluation = ptScatterEvaluate(point, bsdf, direction);
    if (ptMax(evaluation.value) <= 0.0f) { return float3(0); }

    float epsilon = point.kind == ptScatterMedium ? 0.0f : ptOffset(point.position);
    float3 origin = ptShadowOrigin(point, direction);
    float3 shadowDelta = sampled - origin;
    float shadowLength = length(shadowDelta);
    if (shadowLength <= epsilon) { return float3(0); }
    PTRay shadow = {origin, shadowDelta / shadowLength, 0, shadowLength - epsilon, footprint, 0, float4(0)};
    float3 transmittance = ptShadow(shadow, ptShadowStart(point, direction), media, scene, lights, lightCount);
    if (ptMax(transmittance) <= 0.0f) { return float3(0); }

    float pdf = (distance * distance) / (areaCosine * float(lightCount));
    float weight = ptMIS(pdf, evaluation.pdf);
    return light.radiance.rgb * evaluation.value * transmittance * (weight / max(pdf, 1e-12f));
}

// There must be room for the shadow origin offset and an endpoint gap. Both
// estimators exclude shorter segments so a nearby path hit keeps its full weight.
static float ptEmitterDistance2(float3 start, float3 end) {
    float3 delta = end - start;
    float distance2 = dot(delta, delta);
    float startOffset = ptOffset(start);
    float minimum = startOffset + max(startOffset, ptOffset(end));
    return distance2 > minimum * minimum ? distance2 : 0.0f;
}

// Each interval's width, including its Float rounding, is the same probability
// used for both explicit selection and a BSDF ray that hits this primitive.
static float ptEmitterPDF(device const PTEmissiveTriangle* triangles, uint index, float distance2, float cosine) {
    if (!(cosine > 0.0f) || !(distance2 > 0.0f)) { return 0.0f; }
    float previous = index == 0u ? 0.0f : triangles[index - 1u].cumulative;
    float mass = triangles[index].cumulative - previous;
    float pdf = (mass / triangles[index].area) * (distance2 / cosine);
    return isfinite(pdf) ? pdf : 0.0f;
}

static float ptEmitterHitPDF(
    device const PTEmissiveTriangle* triangles, uint count, PTHit hit,
    float3 previousPosition, float3 position, float3 normal
) {
    uint first = 0u, end = count;
    while (first < end) {
        uint middle = first + (end - first) / 2u;
        device const PTEmissiveTriangle& row = triangles[middle];
        bool before = row.instance < hit.instance || (row.instance == hit.instance && row.primitive < hit.index);
        if (before) { first = middle + 1u; } else { end = middle; }
    }
    if (first == count || triangles[first].instance != hit.instance || triangles[first].primitive != hit.index) {
        return 0.0f;
    }
    float3 delta = position - previousPosition;
    float distance2 = ptEmitterDistance2(previousPosition, position);
    if (!(distance2 > 0.0f)) { return 0.0f; }
    float cosine = abs(dot(normal, delta * rsqrt(distance2)));
    return ptEmitterPDF(triangles, first, distance2, cosine);
}

static float3 ptEmitterRadiance(
    thread const PTScene& scene, device const PTMaterial& material, thread const PTTriangle& triangle,
    float2 barycentric, float3 direction, float width
) {
    float3 emission = material.emission.rgb;
    if (material.textureIndices.w != ptMissing) {
        PTTexturing texturing = ptTexturing(triangle, barycentric, direction, width);
        emission *= ptMaterialMap(scene, material, 3u, material.textureIndices.w, material.textureUVs.w, texturing).rgb;
    }
    return max(emission, float3(0));
}

static float3 ptEmitterDirect(
    thread const PTScatterer& point, thread const PTBSDF& bsdf, float footprint, float spread,
    thread PTSampler& state, uint bounce,
    thread const PTScene& scene, device const PTAreaLight* lights, uint lightCount, constant PTFogVolume* media,
    device const PTEmissiveTriangle* triangles, uint count
) {
    state.dimension = ptEmitterDimensionBase + 2u * bounce;
    float choice = ptRandom(state);
    uint first = 0u, end = count;
    while (first < end) {
        uint middle = first + (end - first) / 2u;
        if (choice >= triangles[middle].cumulative) { first = middle + 1u; } else { end = middle; }
    }

    float2 random = ptRandom2(state);
    float radial = sqrt(random.x);
    float2 bary = float2(radial * (1.0f - random.y), radial * random.y);
    device const PTEmissiveTriangle& row = triangles[first];
    device const PTInstance& instance = scene.instances[row.instance];
    PTTriangle triangle = ptWorldTriangle(instance, ptLocalTriangle(instance, row.primitive), bary);
    float3 ab = triangle.p1.xyz - triangle.p0.xyz, ac = triangle.p2.xyz - triangle.p0.xyz;
    float3 sampled = triangle.p0.xyz + bary.x * ab + bary.y * ac;
    float3 delta = sampled - point.position;
    float distance2 = ptEmitterDistance2(point.position, sampled);
    if (!(distance2 > 0.0f)) { return float3(0); }
    float3 direction = delta * rsqrt(distance2);
    if (!ptGathers(point, direction)) { return float3(0); }
    float3 normal = ptUnit(cross(ab, ac), float3(0));
    float cosine = -dot(normal, direction);
    if ((instance.meta.w & ptInstanceCullsBackFaces) == 0u) { cosine = abs(cosine); }
    float pdf = ptEmitterPDF(triangles, first, distance2, cosine);
    if (!(pdf > 0.0f)) { return float3(0); }

    PTEvaluation evaluation = ptScatterEvaluate(point, bsdf, direction);
    if (ptMax(evaluation.value) <= 0.0f) { return float3(0); }
    float epsilon = point.kind == ptScatterMedium ? 0.0f : ptOffset(point.position);
    float3 origin = ptShadowOrigin(point, direction);
    float3 shadowDelta = sampled - origin;
    float shadowLength = length(shadowDelta);
    if (!(shadowLength > epsilon)) { return float3(0); }
    // The endpoint belongs to the emitter. Keep a small gap even when scattering in
    // a medium with no origin offset so rounding cannot shadow a sampled triangle by itself.
    float limit = shadowLength - max(epsilon, ptOffset(sampled));
    if (!(limit > 0.0f)) { return float3(0); }
    PTRay shadow = {origin, shadowDelta / shadowLength, 0.0f, limit, footprint, 0.0f, float4(0)};
    float3 transmission = ptShadow(shadow, ptShadowStart(point, direction), media, scene, lights, lightCount);
    if (ptMax(transmission) <= 0.0f) { return float3(0); }

    float3 emission = ptEmitterRadiance(
        scene, scene.materials[instance.meta.x], triangle, bary, shadow.direction, footprint + spread * shadowLength
    );
    return emission * evaluation.value * transmission * (ptMIS(pdf, evaluation.pdf) / pdf);
}

// A proper rotation preserves solid angles. Look up radiance and PDF with
// world-to-map columns, and transpose them to rotate samples into world space.
static float3 ptSkyLocal(float3 world, constant PTSkyLookup& sky) {
    return float3x3(sky.x.xyz, sky.y.xyz, sky.z.xyz) * world;
}

static float3 ptSkyWorld(float3 local, constant PTSkyLookup& sky) {
    return float3(dot(sky.x.xyz, local), dot(sky.y.xyz, local), dot(sky.z.xyz, local));
}

// The equirectangular convention of EnvironmentLighting.metal: the top row faces +Y,
// the center column faces +X, and increasing U turns toward +Z.
static float2 ptSkyUV(float3 local) {
    return float2(atan2(local.z, local.x) * (0.5f * ptInvPi) + 0.5f, acos(clamp(local.y, -1.0f, 1.0f)) * ptInvPi);
}

static float3 ptSkyRadiance(texture2d<float> map, float3 direction, constant PTSkyLookup& sky) {
    constexpr sampler panorama(coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear);
    float3 color = map.sample(panorama, ptSkyUV(ptSkyLocal(direction, sky)), level(0.0f)).rgb;
    return select(float3(0), color, isfinite(color)) * sky.x.w;
}

// Returns the first entry of an inclusive CDF that exceeds the value.
static uint ptSearch(device const float* cdf, uint count, float value) {
    uint low = 0u;
    uint high = count - 1u;
    while (low < high) {
        uint middle = (low + high) >> 1u;
        if (cdf[middle] > value) { high = middle; } else { low = middle + 1u; }
    }
    return low;
}

// The table holds height marginal CDF entries, then height rows of width conditional entries.
static float ptSkyDensity(device const float* table, uint width, uint height, uint row, uint column, float sine) {
    device const float* conditional = table + height + row * width;
    float rowMass = table[row] - (row == 0u ? 0.0f : table[row - 1u]);
    float columnMass = conditional[column] - (column == 0u ? 0.0f : conditional[column - 1u]);
    return rowMass * float(height) * columnMass * float(width) / (2.0f * ptPi * ptPi * sine);
}

static float ptSkyPDF(device const float* table, uint width, uint height, constant PTSkyLookup& sky, float3 direction) {
    float3 local = ptSkyLocal(direction, sky);
    float sine = sqrt(max(0.0f, 1.0f - local.y * local.y));
    if (!(sine > 1e-6f)) { return 0.0f; }
    float2 uv = ptSkyUV(local);
    uint column = min(uint(uv.x * float(width)), width - 1u);
    uint row = min(uint(uv.y * float(height)), height - 1u);
    return ptSkyDensity(table, width, height, row, column, sine);
}

struct PTSkySample {
    float3 direction;
    float pdf;
};

static PTSkySample ptSampleSky(device const float* table, uint width, uint height, constant PTSkyLookup& sky, float2 random) {
    PTSkySample result = {float3(0), 0.0f};
    uint row = ptSearch(table, height, random.y);
    float rowBefore = row == 0u ? 0.0f : table[row - 1u];
    float rowMass = table[row] - rowBefore;
    device const float* conditional = table + height + row * width;
    uint column = ptSearch(conditional, width, random.x);
    float columnBefore = column == 0u ? 0.0f : conditional[column - 1u];
    float columnMass = conditional[column] - columnBefore;
    if (!(rowMass > 0.0f && columnMass > 0.0f)) { return result; }
    float u = (float(column) + saturate((random.x - columnBefore) / columnMass)) / float(width);
    float v = (float(row) + saturate((random.y - rowBefore) / rowMass)) / float(height);
    float theta = v * ptPi;
    float phi = (u - 0.5f) * 2.0f * ptPi;
    float sine = sin(theta);
    if (!(sine > 1e-6f)) { return result; }
    result.direction = ptSkyWorld(float3(sine * cos(phi), cos(theta), sine * sin(phi)), sky);
    result.pdf = rowMass * float(height) * columnMass * float(width) / (2.0f * ptPi * ptPi * sine);
    return result;
}

static float3 ptSkyDirect(
    thread const PTScatterer& point, thread const PTBSDF& bsdf, float footprint, thread PTSampler& state,
    constant PTUniforms& uniforms, device const float* skyTable, texture2d<float> skyMap, thread const PTScene& scene,
    device const PTAreaLight* lights, uint lightCount, constant PTFogVolume* media
) {
    PTSkySample sample = ptSampleSky(skyTable, uniforms.skyInfo.x, uniforms.skyInfo.y, uniforms.sky, ptRandom2(state));
    if (!(sample.pdf > 0.0f) || !isfinite(sample.pdf)) { return float3(0); }
    float3 direction = sample.direction;
    if (!ptGathers(point, direction)) { return float3(0); }
    PTEvaluation evaluation = ptScatterEvaluate(point, bsdf, direction);
    if (ptMax(evaluation.value) <= 0.0f) { return float3(0); }
    PTRay shadow = {ptShadowOrigin(point, direction), direction, 0.0f, INFINITY, footprint, 0.0f, float4(0)};
    float3 transmittance = ptShadow(shadow, ptShadowStart(point, direction), media, scene, lights, lightCount);
    if (ptMax(transmittance) <= 0.0f) { return float3(0); }
    float weight = ptMIS(sample.pdf, evaluation.pdf);
    return ptSkyRadiance(skyMap, direction, uniforms.sky) * evaluation.value * transmittance * (weight / sample.pdf);
}

// IES profile and cookie filtering, in lockstep with pbrLightTables in PhysicallyBasedShading.h.
// toSurface is the unit direction from the light, axis its unit emission direction and right
// its local +X. Profile tables hold 361 rows of vertical angle from the axis (0°–180°) and 144
// columns of horizontal angle from +X toward +Y; layers are stored plus one, zero for none.
// A cookie spans [-scale, scale]² at unit distance, top row toward +Y (LightTextureSet).
static float3 ptLightTables(
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
        float column = phi * (0.5f * ptInvPi) * float(width);
        column -= floor(column / float(width)) * float(width);
        float row = acos(clamp(z, -1.0f, 1.0f)) * ptInvPi * float(height - 1u);
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

static float ptDirectionalDiskPDF(float spread) {
    return 1.0f / (2.0f * ptPi * spread);
}

static bool ptVisibleDirectionalDisk(PTAnalyticLight light) {
    return light.positionKind.w == 0.0f && light.cone.x >= 0.5f;
}

static float3 ptDirectionalDiskRadiance(PTAnalyticLight light, float3 direction, float previousPDF) {
    if (!ptVisibleDirectionalDisk(light)) { return float3(0); }
    float spread = light.cone.w;
    // |a - b|² = 2 (1 - cos θ). This stays precise on small disks
    // where subtracting a dot product from one loses the edge.
    float3 offset = direction + light.directionRange.xyz;
    if (dot(offset, offset) > 2.0f * spread) { return float3(0); }
    float weight = previousPDF > 0.0f ? ptMIS(previousPDF, ptDirectionalDiskPDF(spread)) : 1.0f;
    float projectedArea = ptPi * spread * (2.0f - spread);
    return light.colorIntensity.rgb * light.colorIntensity.w * (weight / projectedArea);
}

static float3 ptAnalyticDirect(
    thread const PTScatterer& point, thread const PTBSDF& bsdf, float footprint, thread PTSampler& state,
    PTAnalyticLight light, thread const PTScene& scene, device const PTAreaLight* lights, uint lightCount,
    constant PTFogVolume* media,
    texture2d_array<float> lightProfiles, texture2d_array<float> lightCookies
) {
    constexpr float maxHalf = 65504.0f;
    float3 emitted = light.colorIntensity.rgb * light.colorIntensity.w;
    bool positioned = light.positionKind.w > 0.0f;
    float3 toLight;
    float3 incident;
    float diskPDF = 0.0f;
    float3 samplePosition = light.positionKind.xyz;
    if (positioned) {
        if (light.tables.z > 0.0f) {
            float2 disk = light.tables.z * ptDisk(ptRandom2(state));
            float3 axis = light.directionRange.xyz;
            float3 right = light.frame.xyz;
            samplePosition += axis * light.tables.w + right * disk.x + cross(right, axis) * disk.y;
        }
        // Mirrors pbrPunctual in PhysicallyBasedShading.h.
        float3 delta = samplePosition - point.position;
        float distanceSquared = dot(delta, delta);
        if (!(distanceSquared > 0.0f) || !isfinite(distanceSquared)) { return float3(0); }
        float distance = sqrt(distanceSquared);
        toLight = delta / distance;
        float attenuation = 1.0f / max(distanceSquared, 0.0001f);
        if (light.cone.z > 0.0f) {
            if (distance >= light.directionRange.w) { return float3(0); }
            float ratio = distance / light.directionRange.w;
            float squared = ratio * ratio;
            attenuation *= saturate(1.0f - squared * squared);
        }
        if (light.positionKind.w == 2.0f) {
            float cosine = dot(light.directionRange.xyz, -toLight);
            float width = light.cone.x - light.cone.y;
            float cone = width > 0.0f
                ? saturate((cosine - light.cone.y) / width)
                : float(cosine >= light.cone.x);
            attenuation *= cone * cone;
        }
        if (attenuation <= 0.0f) { return float3(0); }
        if (light.frame.w > 0.5f || light.tables.x > 0.5f) {
            emitted *= ptLightTables(
                -toLight, light.directionRange.xyz, light.frame.xyz, light.frame.w, light.tables.x, light.tables.y,
                lightProfiles, lightCookies
            );
        }
        incident = min(emitted, float3(maxHalf / attenuation)) * attenuation;
    } else {
        float3 axis = -light.directionRange.xyz;
        float spread = light.cone.w;
        toLight = axis;
        float scale = 1.0f;
        if (spread > 0.0f) {
            // A uniform disk of half angle a has radiance E / (pi sin^2 a) over the
            // solid angle 2 pi (1 - cos a), so each cone sample carries 2 E / (1 + cos a).
            float2 random = ptRandom2(state);
            float offset = random.x * spread;
            float sine = sqrt(max(0.0f, offset * (2.0f - offset)));
            float angle = 2.0f * ptPi * random.y;
            toLight = ptFrame(axis) * float3(sine * cos(angle), sine * sin(angle), 1.0f - offset);
            scale = 2.0f / (2.0f - spread);
            if (light.cone.x > 0.5f) { diskPDF = ptDirectionalDiskPDF(spread); }
        }
        // A visible disk's light and scatter samples must share the same
        // radiance before the final half-float destination is written.
        float3 directional = emitted * scale;
        incident = diskPDF > 0.0f ? directional : min(directional, float3(maxHalf));
    }
    if (!ptGathers(point, toLight)) { return float3(0); }
    PTEvaluation evaluation = ptScatterEvaluate(point, bsdf, toLight);
    if (ptMax(evaluation.value) <= 0.0f) { return float3(0); }

    float3 origin = ptShadowOrigin(point, toLight);
    PTRay shadow = {origin, toLight, 0.0f, INFINITY, footprint, 0.0f, float4(0)};
    if (positioned) {
        float epsilon = point.kind == ptScatterMedium ? 0.0f : ptOffset(point.position);
        float3 shadowDelta = samplePosition - origin;
        float shadowLength = length(shadowDelta);
        if (shadowLength <= epsilon) { return float3(0); }
        shadow.direction = shadowDelta / shadowLength;
        shadow.maximum = shadowLength - epsilon;
    }
    float3 transmittance = ptShadow(shadow, ptShadowStart(point, toLight), media, scene, lights, lightCount);
    if (diskPDF > 0.0f) { incident *= ptMIS(diskPDF, evaluation.pdf); }
    return incident * evaluation.value * transmittance;
}

// The most scattering events and inner boundary hits of one walk through scattering or
// diffuse-boundary volumes. A walk that reaches it ends, losing its remaining energy.
constant uint ptMaximumWalk = 256u;

// Samples where a path segment of length `limit` first scatters in a homogeneous medium.
// The channel that sets the free path is chosen in proportion to the path throughput, and
// the weight divides by the mixture of every channel's density (one-sample MIS over the
// channels), so chromatic media stay unbiased. Returns true with `distance` at a scattering
// event, or false after crossing the segment. Updates the throughput by the event's weight.
static bool ptSampleFreePath(
    float3 extinction, float3 albedo, float limit, float2 random, thread float3& throughput, thread float& distance
) {
    float total = throughput.x + throughput.y + throughput.z;
    float3 weights = total > 0.0f ? throughput / total : float3(1.0f / 3.0f);
    uint channel = random.x < weights.x ? 0u : (random.x < weights.x + weights.y ? 1u : 2u);
    float sigma = extinction[channel];
    float travel = sigma > 0.0f ? -log(1.0f - random.y) / sigma : INFINITY;
    if (travel < limit) {
        float3 transmittance = ptExtinguish(extinction, travel);
        float pdf = dot(weights, extinction * transmittance);
        throughput = pdf > 0.0f ? throughput * (extinction * albedo * transmittance / pdf) : float3(0);
        distance = travel;
        return true;
    }
    float3 transmittance = ptExtinguish(extinction, limit);
    float probability = dot(weights, transmittance);
    throughput = probability > 0.0f ? throughput * (transmittance / probability) : float3(0);
    distance = limit;
    return false;
}

// In optical distance the heterogeneous field has constant channel extinction.
// At a collision its local scalar density multiplies both the numerator and the
// mixture PDF, so it cancels from the throughput weight.
static bool ptSampleGridFog(
    PTRay ray, float start, float end, constant PTFogVolume& media, float2 random,
    thread float3& throughput, thread float& distance
) {
    float3 extinction = media.fogExtinction.rgb;
    float total = throughput.x + throughput.y + throughput.z;
    float3 weights = total > 0.0f ? throughput / total : float3(1.0f / 3.0f);
    uint channel = random.x < weights.x ? 0u : (random.x < weights.x + weights.y ? 1u : 2u);
    float sigma = extinction[channel];
    float target = sigma > 0.0f ? -log(1.0f - random.y) / sigma : INFINITY;
    PTFogIntegral integral = ptFogIntegral(ray, start, end, media, target);
    float3 transmittance = ptExtinguish(extinction, integral.opticalDistance);
    distance = integral.worldDistance;
    if (integral.scattered) {
        float pdf = dot(weights, extinction * transmittance);
        throughput = pdf > 0.0f ? throughput * (extinction * media.fogAlbedo.rgb * transmittance / pdf) : float3(0);
        return true;
    }
    float probability = dot(weights, transmittance);
    throughput = probability > 0.0f ? throughput * (transmittance / probability) : float3(0);
    return false;
}

// Condition on one spectral channel and sample a free path in each region.
// The first event belongs to that region. Its joint PDF is the region's
// local extinction times combined transmittance.
static bool ptSampleFogVolumes(
    PTRay ray, float end, constant PTFogVolume* media, uint bounce, thread PTSampler& state,
    thread float3& throughput, thread float& distance, thread float& anisotropy
) {
    float total = throughput.x + throughput.y + throughput.z;
    float3 weights = total > 0.0f ? throughput / total : float3(1.0f / 3.0f);
    ptDimension(state, bounce, ptMediumSlot);
    float channelChoice = ptRandom(state);
    uint channel = channelChoice < weights.x ? 0u : (channelChoice < weights.x + weights.y ? 1u : 2u);
    state.dimension = ptFogDimensionBase + bounce * ptMaximumFogVolumes;
    float nearest = end;
    float3 eventExtinction = float3(0);
    uint selected = ptMissing;
    for (uint index = 0u; index < media[0].fogResolution.w; ++index) {
        constant PTFogVolume& volume = media[index];
        float sigma = volume.fogExtinction[channel];
        float random = ptRandom(state);
        if (sigma <= 0.0f) { continue; }
        float opticalTarget = -log(1.0f - random) / sigma;
        PTFogIntegral candidate = ptFogVolumeIntegral(ray, ray.minimum, nearest, volume, opticalTarget);
        if (candidate.scattered) {
            nearest = candidate.worldDistance;
            selected = index;
            eventExtinction = volume.fogExtinction.rgb * candidate.densityAtEvent;
        }
    }
    float3 transmittance = ptFogTransmittance(ray, ray.minimum, nearest, media);
    if (selected != ptMissing) {
        constant PTFogVolume& volume = media[selected];
        float3 density = transmittance * eventExtinction;
        float pdf = dot(weights, density);
        throughput = pdf > 0.0f ? throughput * (density * volume.fogAlbedo.rgb / pdf) : float3(0);
        distance = nearest;
        anisotropy = volume.fogAlbedo.w;
        return true;
    }
    float probability = dot(weights, transmittance);
    throughput = probability > 0.0f ? throughput * (transmittance / probability) : float3(0);
    return false;
}

// Russian roulette for walk events once the path carries little energy.
static bool ptWalkSurvives(thread float3& throughput, float etaScale, thread PTSampler& state) {
    float peak = ptMax(throughput) * etaScale;
    if (peak >= 0.25f) { return true; }
    float survival = max(peak * 4.0f, 0.05f);
    if (ptRandom(state) > survival) { return false; }
    throughput /= survival;
    return true;
}

// Returns linear radiance and camera-ray coverage. Limits apply to each term
// according to the depth of the path segment that reaches its light source.
// The stratum rotates the dispersion wavelength through red, green and blue across
// a pixel's consecutive samples.
//
// Inside a scattering volume, or one that light enters diffusely, scattering events and
// hits on the volume's own boundary form a walk: they do not count as bounces and draw
// independent random numbers instead of the vertex's Sobol dimensions. Fog scattering
// events count as bounces.
static float4 ptTrace(
    PTRay ray, thread PTSampler& state, constant PTUniforms& uniforms, thread const PTScene& scene,
    device const PTAreaLight* lights, device const PTAnalyticLight* analyticLights,
    device const float* skyTable, texture2d<float> skyMap,
    constant float4* graphUniforms, device const uint* graphTable, float3 radianceLimits, uint stratum,
    constant PTFogVolume* media,
    texture2d_array<float> lightProfiles, texture2d_array<float> lightCookies,
    device const PTEmissiveTriangle* emitters, uint emitterCount
) {
    uint lightCount = uniforms.controls.z;
    uint bounces = uniforms.controls.x;
    uint analyticCount = min(uniforms.skyInfo.z, 16u);
    uint skyFlags = uniforms.skyInfo.w;
    bool multipleImportanceSampling = (skyFlags & ptDisableMIS) == 0u;
    bool sampleSky = multipleImportanceSampling && (skyFlags & ptSkySampled) != 0u;
    bool sobol = state.sobol;
    uint fogCount = media[0].fogResolution.w;
    float coverage = 1.0f;
    float3 radiance = float3(0);
    float3 throughput = float3(1);
    float previousPDF = 0.0f;
    float3 previousPosition = ray.origin;
    PTRay emitterRay = ray;
    float etaScale = 1.0f;
    PTMedium media_[4];
    uint mediumCount = 0u;
    // The dispersion channel chosen for this path: 0 red, 1 green, 2 blue, 3 none yet.
    uint wavelength = 3u;
    uint bounce = 0u;
    uint walk = 0u;

    while (bounce <= bounces) {
        bool inside = mediumCount != 0u;
        PTMedium current = inside ? media_[mediumCount - 1u] : PTMedium{0u, 1.0f, float3(0), float3(0), 0.0f, 0u};
        bool walking = inside && (current.flags & (ptLobeScattering | ptLobeDiffuseBoundary)) != 0u;
        if (walking && ++walk > ptMaximumWalk) { break; }
        state.sobol = sobol && !walking;

        PTHit hit = ptGeometry(ray, false, scene);
        PTRay lightRay = ray;
        lightRay.maximum = hit.distance;
        PTLightHit panel = ptAreaLights(lightRay, bounce == 0u, lights, lightCount);
        uint nextDepth = bounce + 1u;
        float end = panel.index != ptMissing ? panel.distance : hit.distance;

        // The segment's medium: the innermost closed volume, else the fog box.
        bool scattered = false;
        float scatterDistance = 0.0f;
        float anisotropy = 0.0f;
        if (inside) {
            if ((current.flags & ptLobeScattering) != 0u) {
                scattered = ptSampleFreePath(
                    current.extinction, current.albedo, end, ptRandom2(state), throughput, scatterDistance
                );
                anisotropy = current.anisotropy;
            } else if (isfinite(end)) {
                throughput *= ptExtinguish(current.extinction, end);
            }
        } else if (fogCount == 1u && media[0].fogBoundary == nullptr) {
            constant PTFogVolume& volume = media[0];
            float2 interval = ptBoxInterval(ray.origin, ray.direction, volume.fogMinimum.xyz, volume.fogMaximum.xyz);
            float start = max(interval.x, ray.minimum);
            float stop = min(interval.y, end);
            if (stop > start) {
                ptDimension(state, bounce, ptMediumSlot);
                float travelled;
                float2 random = ptRandom2(state);
                scattered = volume.fogResolution.x == 0u ? ptSampleFreePath(
                    volume.fogExtinction.rgb, volume.fogAlbedo.rgb, stop - start, random, throughput, travelled
                ) : ptSampleGridFog(ray, start, stop, volume, random, throughput, travelled);
                scatterDistance = start + travelled;
                anisotropy = volume.fogAlbedo.w;
            }
        } else if (fogCount != 0u) {
            // Several regions, or a mesh region, whether the ray is the camera's or a scattered one.
            scattered = ptSampleFogVolumes(ray, end, media, bounce, state, throughput, scatterDistance, anisotropy);
        }
        if (!all(isfinite(throughput)) || ptMax(throughput) <= 0.0f) { break; }

        if (scattered) {
            if (!walking && bounce == bounces) { break; }
            PTShadowMedium here = {media_, mediumCount};
            PTScatterer point = {
                ray.origin + scatterDistance * ray.direction, float3(0), ray.direction, float3(0), anisotropy,
                ptScatterMedium, false, here, here
            };
            PTBSDF unused = {};
            float footprint = ray.width + ray.spread * scatterDistance;
            // Shadow rays leave a closed volume only through an index-matched boundary.
            bool gathers = !inside || ((current.flags & ptLobeNullBoundary) != 0u && frostfield_path_has_null_boundaries);
            if (gathers) {
                if (multipleImportanceSampling) {
                    ptDimension(state, bounce, ptAreaLightSlot);
                    radiance += ptLimitContribution(
                        throughput * ptDirect(point, unused, footprint, state, scene, lights, lightCount, media),
                        nextDepth, radianceLimits
                    );
                    if (frostfield_path_has_mesh_emitters) {
                        radiance += ptLimitContribution(throughput * ptEmitterDirect(
                            point, unused, footprint, ray.spread, state, bounce, scene, lights, lightCount, media,
                            emitters, emitterCount
                        ), nextDepth, radianceLimits);
                    }
                }
                for (uint index = 0u; index < analyticCount; ++index) {
                    if (!multipleImportanceSampling && ptVisibleDirectionalDisk(analyticLights[index])) { continue; }
                    ptDimension(state, bounce, ptAnalyticSlot + index);
                    radiance += ptLimitContribution(throughput * ptAnalyticDirect(
                        point, unused, footprint, state, analyticLights[index], scene, lights, lightCount, media,
                        lightProfiles, lightCookies
                    ), nextDepth, radianceLimits);
                }
                if (sampleSky) {
                    ptDimension(state, bounce, ptSkySlot);
                    radiance += ptLimitContribution(throughput * ptSkyDirect(
                        point, unused, footprint, state, uniforms, skyTable, skyMap, scene, lights, lightCount, media
                    ), nextDepth, radianceLimits);
                }
            }
            ptDimension(state, bounce, ptScatterSlot);
            float3 direction = frostfieldSampleHenyeyGreenstein(ray.direction, anisotropy, ptRandom2(state));
            // The phase function samples itself exactly, so the weight is one.
            previousPosition = point.position;
            previousPDF = gathers ? frostfieldHenyeyGreenstein(dot(ray.direction, direction), anisotropy) : 0.0f;
            ray = {point.position, direction, 0.0f, INFINITY, footprint, ray.spread, ray.screen};
            if (frostfield_path_has_mesh_emitters) { emitterRay = ray; }
            if (walking) {
                if (!ptWalkSurvives(throughput, etaScale, state)) { break; }
            } else {
                if (bounce >= 3u) {
                    float survival = clamp(ptMax(throughput) * etaScale, 0.05f, 0.95f);
                    ptDimension(state, bounce, ptSurvivalSlot);
                    if (ptRandom(state) > survival) { break; }
                    throughput /= survival;
                }
                ++bounce;
            }
            continue;
        }

        if (panel.index != ptMissing) {
            if (panel.cosine > 0.0f) {
                float weight = 1.0f;
                if (multipleImportanceSampling && previousPDF > 0.0f) {
                    float pdf = panel.distance * panel.distance / (panel.area * panel.cosine * float(lightCount));
                    weight = ptMIS(previousPDF, pdf);
                }
                radiance += ptLimitContribution(throughput * lights[panel.index].radiance.rgb * weight, bounce, radianceLimits);
            }
            break;
        }
        if (hit.index == ptMissing) {
            if (bounce == 0u && (skyFlags & ptSkyHidden) != 0u) {
                coverage = 0.0f;
                break;
            }
            if ((skyFlags & ptSkyMap) != 0u) {
                float weight = 1.0f;
                if (previousPDF > 0.0f && sampleSky) {
                    float pdf = ptSkyPDF(skyTable, uniforms.skyInfo.x, uniforms.skyInfo.y, uniforms.sky, ray.direction);
                    weight = ptMIS(previousPDF, pdf);
                }
                radiance += ptLimitContribution(
                    throughput * ptSkyRadiance(skyMap, ray.direction, uniforms.sky) * weight, bounce, radianceLimits
                );
            } else {
                radiance += ptLimitContribution(throughput * uniforms.environment.rgb, bounce, radianceLimits);
            }
            if ((skyFlags & ptSkyDirectionalDisks) != 0u) {
                for (uint index = 0u; index < analyticCount; ++index) {
                    radiance += ptLimitContribution(throughput * ptDirectionalDiskRadiance(
                        analyticLights[index], ray.direction, multipleImportanceSampling ? previousPDF : 0.0f
                    ), bounce, radianceLimits);
                }
            }
            break;
        }

        PTSurface surface = ptSurface(hit, ray, scene, graphUniforms, graphTable, bounce < bounces);
        // A frosted part of the boundary can have zero transmission while the ray is still inside the medium.
        bool exitingMedium = !surface.front && !surface.thin && inside && current.instance == hit.instance;
        if (walking && !exitingMedium) {
            // Another surface inside the volume is an ordinary, counted vertex.
            walking = false;
            state.sobol = sobol;
        }
        // A material-alpha surface writes its alpha over whatever lies behind it, as raster does.
        if (bounce == 0u && scene.materials[surface.material].emission.w == ptWrittenCoverage) {
            coverage = surface.alpha;
        }
        // A walk reaches its boundary from inside, where the surface does not emit.
        if (!walking) {
            float weight = 1.0f;
            float3 emission = surface.emission;
            if (frostfield_path_has_mesh_emitters && multipleImportanceSampling && previousPDF > 0.0f) {
                device const PTMaterial& material = scene.materials[surface.material];
                bool mapped = material.textureIndices.w != ptMissing && ptMax(material.emission.rgb) > 0.0f;
                if (mapped || ptMax(emission) > 0.0f) {
                    float pdf = ptEmitterHitPDF(
                        emitters, emitterCount, hit, previousPosition, surface.position, surface.geometricNormal
                    );
                    if (pdf > 0.0f) {
                        weight = ptMIS(previousPDF, pdf);
                        // Straight index-matched boundaries retain the previous vertex's
                        // PDF. Use its cone for the same emitter lookup as the light ray.
                        if (mapped) {
                            device const PTInstance& instance = scene.instances[hit.instance];
                            PTTriangle triangle = ptWorldTriangle(instance, ptLocalTriangle(instance, hit.index), hit.barycentric);
                            float distance = max(dot(surface.position - emitterRay.origin, emitterRay.direction), 0.0f);
                            emission = ptEmitterRadiance(
                                scene, material, triangle, hit.barycentric, emitterRay.direction,
                                emitterRay.width + emitterRay.spread * distance
                            );
                        }
                    }
                }
            }
            radiance += ptLimitContribution(throughput * emission * weight, bounce, radianceLimits);
        }
        // Unlit surfaces emit their selected color and scatter nothing.
        if ((!walking && bounce == bounces) || surface.unlit) { break; }
        float footprint = ray.width + ray.spread * hit.distance;
        PTShadowMedium outside = {media_, mediumCount > 0u ? mediumCount - 1u : 0u};
        PTShadowMedium here = {media_, mediumCount};

        if (walking && (current.flags & ptLobeDiffuseBoundary) != 0u) {
            // The walk leaves a diffuse-boundary volume through a Lambertian BTDF tinted by the
            // diffuse transmission color, without internal reflection.
            float3 outward = -surface.geometricNormal;
            float3 shading = -surface.normal;
            PTScatterer exit = {
                surface.position, outward, shading, surface.translucencyColor, 0.0f, ptScatterExit, false, outside, outside
            };
            PTBSDF unused = {};
            if (multipleImportanceSampling) {
                radiance += ptLimitContribution(
                    throughput * ptDirect(exit, unused, footprint, state, scene, lights, lightCount, media),
                    nextDepth, radianceLimits
                );
                if (frostfield_path_has_mesh_emitters) {
                    radiance += ptLimitContribution(throughput * ptEmitterDirect(
                        exit, unused, footprint, ray.spread, state, bounce, scene, lights, lightCount, media,
                        emitters, emitterCount
                    ), nextDepth, radianceLimits);
                }
            }
            for (uint index = 0u; index < analyticCount; ++index) {
                if (!multipleImportanceSampling && ptVisibleDirectionalDisk(analyticLights[index])) { continue; }
                radiance += ptLimitContribution(throughput * ptAnalyticDirect(
                    exit, unused, footprint, state, analyticLights[index], scene, lights, lightCount, media,
                    lightProfiles, lightCookies
                ), nextDepth, radianceLimits);
            }
            if (sampleSky) {
                radiance += ptLimitContribution(throughput * ptSkyDirect(
                    exit, unused, footprint, state, uniforms, skyTable, skyMap, scene, lights, lightCount, media
                ), nextDepth, radianceLimits);
            }
            float2 disk = ptDisk(ptRandom2(state));
            float cosine = sqrt(max(0.0f, 1.0f - dot(disk, disk)));
            float3 direction = ptFrame(shading) * float3(disk, cosine);
            if (dot(outward, direction) <= 0.0f || cosine <= 0.0f) { break; }
            throughput *= surface.translucencyColor;
            previousPosition = exit.position;
            previousPDF = cosine * ptInvPi;
            --mediumCount;
            ray = {ptOrigin(surface.position, surface.geometricNormal, direction), direction, 0.0f, INFINITY, footprint, ray.spread, ray.screen};
            if (frostfield_path_has_mesh_emitters) { emitterRay = ray; }
            if (!ptWalkSurvives(throughput, etaScale, state)) { break; }
            continue;
        }

        float surfaceIOR = surface.ior;
        if (surface.dispersion > 0.0f && !surface.thin && surface.transmission > 0.0f && surface.ior > 0.0f) {
            // KHR_materials_dispersion: the first dispersive volume picks one color channel for
            // the rest of the path and weights it by three; later volumes use the same wavelength.
            // Consecutive samples of a pixel take the channels in turn, so every three cover all.
            if (wavelength == 3u) {
                wavelength = stratum % 3u;
                float3 selected = float3(0);
                selected[wavelength] = 3.0f;
                throughput *= selected;
            }
            surfaceIOR = frostfieldDispersedIOR(surface.ior, surface.dispersion, frostfieldDispersionWavelengths[wavelength]);
        }
        float from = mediumCount == 0u ? 1.0f : media_[mediumCount - 1u].ior;
        float to = surfaceIOR > 0.0f ? surfaceIOR : 1.0f;
        bool volumetric = (surface.lobeFlags & ptLobeMedium) != 0u;
        bool thick = !surface.thin && (surface.transmission > 0.0f || volumetric) && surface.metallic < 1.0f && surface.ior > 0.0f;
        if (exitingMedium) {
            to = mediumCount > 1u ? media_[mediumCount - 2u].ior : 1.0f;
        } else if (!surface.front && thick && mediumCount == 0u) {
            // A camera may start inside a dielectric without a previously crossed boundary.
            from = to;
            to = 1.0f;
        }
        float eta = clamp(to / from, 1e-6f, 1e6f);
        PTBSDF bsdf = ptBSDF(surface, -ray.direction, eta);
        if (!walking && ptMax(surface.bakedIrradiance) > 0.0f) {
            // The map evaluates a diffuse reflection at the hit, independently of
            // live light sampling and of the authored-emission MIS weight.
            radiance += ptLimitContribution(
                throughput * ptBakedDiffuse(bsdf, surface.bakedIrradiance), nextDepth, radianceLimits
            );
        }
        // Light on the other side is gathered through thin sheets and through a volume's boundary
        // from inside; entering a volume, shadow rays would cross its far side.
        bool transmits = surface.thin || exitingMedium;
        PTScatterer point = {
            surface.position, surface.geometricNormal, -ray.direction, float3(0), 0.0f, ptScatterSurface, transmits,
            here, surface.thin ? here : outside
        };
        if (multipleImportanceSampling) {
            if (!walking) { ptDimension(state, bounce, ptAreaLightSlot); }
            radiance += ptLimitContribution(
                throughput * ptDirect(point, bsdf, footprint, state, scene, lights, lightCount, media),
                nextDepth, radianceLimits
            );
            if (frostfield_path_has_mesh_emitters) {
                radiance += ptLimitContribution(throughput * ptEmitterDirect(
                    point, bsdf, footprint, max(ray.spread, 0.04f * surface.roughness), state, bounce, scene, lights,
                    lightCount, media, emitters, emitterCount
                ), nextDepth, radianceLimits);
            }
        }
        for (uint index = 0u; index < analyticCount; ++index) {
            if (!multipleImportanceSampling && ptVisibleDirectionalDisk(analyticLights[index])) { continue; }
            if (!walking) { ptDimension(state, bounce, ptAnalyticSlot + index); }
            radiance += ptLimitContribution(throughput * ptAnalyticDirect(
                point, bsdf, footprint, state, analyticLights[index], scene, lights, lightCount, media,
                lightProfiles, lightCookies
            ), nextDepth, radianceLimits);
        }
        if (sampleSky) {
            if (!walking) { ptDimension(state, bounce, ptSkySlot); }
            radiance += ptLimitContribution(throughput * ptSkyDirect(
                point, bsdf, footprint, state, uniforms, skyTable, skyMap, scene, lights, lightCount, media
            ), nextDepth, radianceLimits);
        }

        if (!walking) { ptDimension(state, bounce, ptScatterSlot); }
        PTScatter scatter = ptSampleBSDF(bsdf, state);
        float side = dot(surface.geometricNormal, scatter.direction);
        if (ptMax(scatter.weight) <= 0.0f || !all(isfinite(scatter.weight))
            || (scatter.transmitted ? side >= 0.0f : side <= 0.0f)) {
            break;
        }
        throughput *= scatter.weight;
        if (!all(isfinite(throughput)) || ptMax(throughput) <= 0.0f) { break; }

        if (scatter.transmitted && thick) {
            if (surface.front) {
                if (mediumCount == 4u) { break; }
                device const PTMaterialLobes& lobes = scene.materials[surface.material].lobes;
                // The multiple-scattering color, with its texture, is read where light enters.
                media_[mediumCount++] = {
                    hit.instance, surfaceIOR, lobes.extinction.rgb, frostfieldSingleScatteringAlbedo(surface.scatterColor),
                    lobes.scattering.a, surface.lobeFlags
                };
            } else if (exitingMedium) {
                --mediumCount;
            }
            // Refraction compresses radiance by eta squared; diffuse transmission does not refract.
            if (!scatter.diffuse) { etaScale = clamp(etaScale * eta * eta, 1e-12f, 1e12f); }
        }

        if (walking) {
            if (!ptWalkSurvives(throughput, etaScale, state)) { break; }
        } else if (bounce >= 3u) {
            float survival = clamp(ptMax(throughput) * etaScale, 0.05f, 0.95f);
            ptDimension(state, bounce, ptSurvivalSlot);
            if (ptRandom(state) > survival) { break; }
            throughput /= survival;
        }
        // The light strategy samples transmitted directions only where next-event estimation gathers them.
        // A straight pass through an index-matched boundary keeps the previous vertex's density, as its
        // shadow rays cross the boundary too.
        bool changesSample = !scatter.straight || (surface.lobeFlags & ptLobeNullBoundary) == 0u || from != 1.0f || to != 1.0f;
        if (changesSample) {
            previousPosition = point.position;
            previousPDF = scatter.transmitted && !transmits ? 0.0f : scatter.pdf;
        }
        ray = {
            ptOrigin(surface.position, surface.geometricNormal, scatter.direction),
            scatter.direction, 0.0f, INFINITY, footprint, max(ray.spread, 0.04f * surface.roughness), ray.screen
        };
        if (frostfield_path_has_mesh_emitters && changesSample) { emitterRay = ray; }
        if (!walking) { ++bounce; }
    }
    // Radiance is premultiplied by coverage; a camera ray that misses a hidden sky gathers none.
    return float4(select(float3(0), max(radiance, float3(0)), isfinite(radiance)) * coverage, coverage);
}

// A uniform point on the lens aperture with unit circumradius: a disk, or a regular
// polygon of lens.x blades rotated by lens.y, then scaled by lens.zw for anamorphic lenses.
// The polygon picks a sector with the first number and fills its triangle uniformly.
static float2 ptAperture(float2 random, float4 lens) {
    uint blades = uint(lens.x + 0.5f);
    float2 point;
    if (blades < 3u) {
        point = ptDisk(random);
    } else {
        float scaled = random.x * float(blades);
        float sector = min(floor(scaled), float(blades) - 1.0f);
        float along = saturate(scaled - sector);
        float step = 2.0f * ptPi / float(blades);
        float angle = lens.y + sector * step;
        float2 first = float2(cos(angle), sin(angle));
        float2 second = float2(cos(angle + step), sin(angle + step));
        point = sqrt(random.y) * mix(first, second, along);
    }
    return point * lens.zw;
}

static PTRay ptCamera(
    uint2 pixel, float2 offset, float2 lens, constant PTUniforms& uniforms
) {
    float2 uv = (float2(pixel) + offset) / float2(uniforms.image.xy);
    float3 forward = uniforms.cameraForward.xyz;
    float3 right = uniforms.cameraRight.xyz;
    float3 up = uniforms.cameraUp.xyz;
    float3 origin = uniforms.cameraPosition.xyz;
    uint mode = uint(uniforms.projection.z + 0.5f);
    if (mode == ptEquirectangular || mode == ptStereoEquirectangular) {
        // Longitude 0 looks forward and turns toward +X; the top row looks up.
        float eyeHeight = float(uniforms.image.y);
        if (mode == ptStereoEquirectangular) {
            // ODS: the left eye fills the top half. Each ray starts on the eye circle, tangent to it.
            bool left = uv.y < 0.5f;
            uv.y = left ? uv.y * 2.0f : uv.y * 2.0f - 1.0f;
            eyeHeight *= 0.5f;
            float longitude = (uv.x - 0.5f) * 2.0f * ptPi;
            origin += (left ? -1.0f : 1.0f) * uniforms.projection.w * (cos(longitude) * right - sin(longitude) * forward);
        }
        float longitude = (uv.x - 0.5f) * 2.0f * ptPi;
        float latitude = (0.5f - uv.y) * ptPi;
        float3 horizontal = sin(longitude) * right + cos(longitude) * forward;
        float3 direction = ptUnit(cos(latitude) * horizontal + sin(latitude) * up, forward);
        float spread = max(2.0f * ptPi / float(uniforms.image.x), ptPi / eyeHeight);
        return {origin, direction, uniforms.projection.x, uniforms.projection.y, 0, spread,
            float4((float2(pixel) + offset).x, (float2(pixel) + offset).y, uniforms.image.x, uniforms.image.y)};
    }
    float2 screen = uv * float2(2, -2) + float2(-1, 1);
    if (mode == ptOrthographic) {
        origin += right * (screen.x * uniforms.cameraRight.w) + up * (screen.y * uniforms.cameraUp.w);
        float width = 2.0f * max(
            uniforms.cameraRight.w / float(uniforms.image.x), uniforms.cameraUp.w / float(uniforms.image.y)
        );
        return {origin, forward, uniforms.projection.x, uniforms.projection.y, width, 0,
            float4((float2(pixel) + offset).x, (float2(pixel) + offset).y, uniforms.image.x, uniforms.image.y)};
    }
    screen += uniforms.shift.xy;
    float3 direction = normalize(
        forward + right * (screen.x * uniforms.cameraRight.w) + up * (screen.y * uniforms.cameraUp.w)
    );
    if (uniforms.cameraForward.w > 0.0f) {
        float focus = uniforms.cameraPosition.w / max(dot(direction, forward), 1e-6f);
        float3 target = origin + direction * focus;
        origin += uniforms.cameraForward.w * (right * lens.x + up * lens.y);
        direction = ptUnit(target - origin, direction);
    }
    float projection = max(dot(direction, forward), 1e-6f);
    float spread = max(
        uniforms.cameraRight.w / float(uniforms.image.x), uniforms.cameraUp.w / float(uniforms.image.y)
    );
    return {origin, direction, uniforms.projection.x / projection, uniforms.projection.y / projection, 0, 2.0f * spread,
        float4((float2(pixel) + offset).x, (float2(pixel) + offset).y, uniforms.image.x, uniforms.image.y)};
}

// Keep in lockstep with PathTracer.MotionUniforms.
struct PTMotionUniforms {
    float4 right, up, forward, position, clip, shiftAndJitter;
};

// Both backends report barycentrics in the original object-space corner order.
// The software BVH can reorder triangles after posing, so fetch corner IDs
// from the current hit and read the corresponding previous vertices by ID.
static float3 ptPreviousHitRelative(
    PTHit hit, device const PTInstance& current, device const PTInstance& previous, float3 previousCamera
) {
    device const uint* indices = current.indices + hit.index * 3u;
    float3 p0 = previous.vertices[indices[0]].position.xyz;
    float3 p1 = previous.vertices[indices[1]].position.xyz;
    float3 p2 = previous.vertices[indices[2]].position.xyz;
    float3 local = p0 + (p1 - p0) * hit.barycentric.x + (p2 - p0) * hit.barycentric.y;
    return float3(
        dot(previous.objectToWorld[0].xyz, local) + (previous.objectToWorld[0].w - previousCamera.x),
        dot(previous.objectToWorld[1].xyz, local) + (previous.objectToWorld[1].w - previousCamera.y),
        dot(previous.objectToWorld[2].xyz, local) + (previous.objectToWorld[2].w - previousCamera.z)
    );
}

static float2 ptMotionVector(
    float3 relative, bool hit, float2 currentPixel, float2 extent, constant PTMotionUniforms& prior
) {
    if (prior.position.w == 0.0f) { return float2(0); }
    float2 outside = extent * 2.0f;
    bool perspective = prior.clip.z == ptPerspective;
    // A parallel sky ray has no unique image position. A camera that looks in
    // the same direction can reuse the corresponding pixel's sky history.
    if (!hit && !perspective) {
        return all(relative == prior.forward.xyz) ? float2(0) : outside;
    }
    float z = dot(relative, prior.forward.xyz);
    if ((hit && (z < prior.clip.x || z > prior.clip.y)) || (perspective && !(z > 0.0f))) {
        return outside;
    }
    float2 screen = float2(dot(relative, prior.right.xyz), dot(relative, prior.up.xyz));
    if (perspective) { screen /= z; }
    screen = screen / float2(prior.right.w, prior.up.w) - prior.shiftAndJitter.xy;
    // Clamp far-outside positions before conversion to half so motion remains
    // finite without making off-screen history look like it is valid.
    screen = clamp(screen, float2(-4.0f), float2(4.0f));
    float2 previousPixel = (screen * float2(0.5f, -0.5f) + 0.5f) * extent;
    return all(isfinite(previousPixel)) ? previousPixel - currentPixel : outside;
}

kernel void frostfield_path_bvh_heatmap(
    device const PTMaterial* materials [[buffer(2)]],
    constant PTUniforms& uniforms [[buffer(4)]],
    constant float4* graphUniforms [[buffer(5)]],
    device const uint* graphTable [[buffer(6)]],
    device const PTTextureBinding* textures [[buffer(9)]],
    device const PTInstance* instances [[buffer(11)]],
    device const PTNode* topLevel [[buffer(12)]],
    // Instance count, level flags, first-surface flag and saturation count.
    constant uint4& options [[buffer(13)]],
    // Top-left region origin, visit budget and texture count.
    constant uint4& window [[buffer(14)]],
    texture2d_array<half> maps [[texture(0)]],
    texture2d<half, access::write> destination [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (any(gid >= uint2(destination.get_width(), destination.get_height()))) { return; }
    PTScene scene = {
        instances, topLevel, materials, maps, textures, window.w, uniforms.controls.w, options.x, {},
#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
        graphUniforms, graphTable,
#endif
    };
    PTRay ray = ptCamera(gid + window.xy, float2(0.5f), float2(0), uniforms);
    PTBoundsDiagnostic diagnostic = {0u, window.z, options.y, options.w, false, false};
    if (options.y != 0u) {
        if (options.z != 0u) {
            PTHit first = ptGeometrySoftwareTraversal<true, false>(ray, false, scene, false, diagnostic);
            ray.maximum = first.distance;
        }
        if (!diagnostic.truncated && (options.z == 0u || diagnostic.intersects)) {
            ptGeometrySoftwareTraversal<true, true>(ray, false, scene, false, diagnostic);
        }
    }
    float t = saturate(float(diagnostic.count) / float(options.w));
    float3 heat = saturate(1.5f - abs(4.0f * t - float3(3, 2, 1)));
    destination.write(half4(half3(diagnostic.truncated ? float3(1, 0, 1) : heat), 1.0h), gid);
}

kernel void frostfield_path_trace(
    device const PTMaterial* materials [[buffer(2)]],
    device const PTAreaLight* lights [[buffer(3)]],
    constant PTUniforms& uniforms [[buffer(4)]],
    constant float4* graphUniforms [[buffer(5)]],
    device const uint* graphTable [[buffer(6)]],
    device const PTAnalyticLight* analyticLights [[buffer(7)]],
    device const float* skyTable [[buffer(8)]],
    device const PTInstance* instances [[buffer(11)]],
    device const PTNode* topLevel [[buffer(12)]],
    constant PTDynamics& dynamics [[buffer(13)]],
    constant uint2& regionOrigin [[buffer(14)]],
    constant uint& aovMask [[buffer(15), function_constant(frostfield_path_has_aovs)]],
    device const PTTextureBinding* textures [[buffer(9)]],
    instance_acceleration_structure accelerationStructure [[buffer(10), function_constant(frostfield_path_metal_rt)]],
    constant PTFogVolume* media [[buffer(16)]],
    constant PTMotionUniforms& motion [[buffer(17), function_constant(frostfield_path_has_motion)]],
    device const PTInstance* priorInstances [[buffer(18), function_constant(frostfield_path_has_motion)]],
    device const ushort* blueNoiseRanks [[buffer(19)]],
    device const PTEmissiveTriangle* emitters [[buffer(20), function_constant(frostfield_path_has_mesh_emitters)]],
    texture2d_array<half> maps [[texture(0)]],
    texture2d<float, access::read> previous [[texture(1)]],
    texture2d<float, access::write> next [[texture(2)]],
    texture2d<half, access::write> destination [[texture(3)]],
    texture2d<half, access::write> normalGuide [[texture(4), function_constant(frostfield_path_has_guides)]],
    texture2d<half, access::write> albedoGuide [[texture(5), function_constant(frostfield_path_has_guides)]],
    texture2d<float> skyMap [[texture(6)]],
    texture2d_array<float> lightProfiles [[texture(7)]],
    texture2d_array<float> lightCookies [[texture(8)]],
    texture2d<float, access::write> albedoAOV [[texture(9), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> normalAOV [[texture(10), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> depthAOV [[texture(11), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> roughnessAOV [[texture(12), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> emissionAOV [[texture(13), function_constant(frostfield_path_has_aovs)]],
    texture2d<uint, access::write> objectAOV [[texture(14), function_constant(frostfield_path_has_aovs)]],
    texture2d<uint, access::write> materialAOV [[texture(15), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> specularAlbedoAOV [[texture(16), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> projectedDepthAOV [[texture(17), function_constant(frostfield_path_has_aovs)]],
    texture2d<float, access::write> motionAOV [[texture(18), function_constant(frostfield_path_has_motion)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (any(gid >= uint2(destination.get_width(), destination.get_height()))) { return; }
    // Keep texture accesses local, while rays and random sequences use the full image.
    uint2 imagePixel = gid + regionOrigin;
    uint sampleStart = uniforms.image.z;
    uint samples = uniforms.image.w;
    PTScene scene = {
        instances, topLevel, materials, maps, textures, dynamics.counts.z, uniforms.controls.w, dynamics.counts.x, {},
#ifdef FROSTFIELD_HAS_COVERAGE_GRAPHS
        graphUniforms, graphTable,
#endif
    };
    if (frostfield_path_metal_rt) { scene.accelerationStructure = accelerationStructure; }

    float2 firstHitOffset = float2(0.5f);
    if (frostfield_path_has_motion) { firstHitOffset += motion.shiftAndJitter.zw; }
    bool writeGuides = frostfield_path_has_guides && sampleStart == 0u;
    if (writeGuides || frostfield_path_has_aovs) {
        float3 normal = float3(0);
        float3 albedo = float3(0);
        float3 diffuseAlbedo = float3(0);
        float3 specularAlbedo = float3(0);
        float3 emission = float3(0);
        float depth = 0.0f;
        float roughness = 0.0f;
        float coverage = 0.0f;
        uint objectID = 0u;
        uint materialID = 0u;
        // Stills use a centered pinhole sample. A motion frame uses the same
        // frame-wide offset as its beauty ray. Stochastic surfaces appear
        // where their alpha is at least one half.
        PTRay ray = ptCamera(imagePixel, firstHitOffset, float2(0), uniforms);
        PTHit hit = ptGeometry(ray, false, scene, true);
        PTRay lightRay = ray;
        lightRay.maximum = hit.distance;
        PTLightHit panel = ptAreaLights(lightRay, true, lights, uniforms.controls.z);
        if (panel.index != ptMissing) {
            device const PTAreaLight& light = lights[panel.index];
            normal = ptUnit(cross(light.halfU.xyz, light.halfV.xyz), -ray.direction);
            if (panel.cosine < 0.0f) { normal = -normal; }
            albedo = float3(1);
            diffuseAlbedo = float3(1);
            emission = panel.cosine > 0.0f ? light.radiance.rgb : float3(0);
            coverage = 1.0f;
            depth = panel.distance;
        } else if (hit.index != ptMissing) {
            PTSurface surface = ptSurface(hit, ray, scene, graphUniforms, graphTable);
            normal = surface.normal;
            albedo = surface.baseColor;
            diffuseAlbedo = surface.unlit ? albedo : albedo * (1.0f - surface.metallic) * (1.0f - surface.transmission);
            if (frostfield_path_has_aovs && (aovMask & 128u)) {
                specularAlbedo = ptSpecularAlbedo(surface, -ray.direction);
            }
            emission = surface.emission;
            roughness = surface.roughness;
            depth = hit.distance;
            coverage = 1.0f;
            objectID = scene.instances[hit.instance].surface.y;
            materialID = surface.material + 1u;
        }
        if (frostfield_path_has_motion) {
            bool covered = coverage > 0.0f;
            // A sky miss uses direction only. A camera-visible area light is
            // stationary as long as the validated lighting rig is unchanged.
            float3 relative = covered ? (ray.origin - motion.position.xyz) + depth * ray.direction : ray.direction;
            if (motion.position.w == 2.0f && panel.index == ptMissing && hit.index != ptMissing) {
                relative = ptPreviousHitRelative(
                    hit, instances[hit.instance], priorInstances[hit.instance], motion.position.xyz
                );
            }
            float2 velocity = ptMotionVector(
                relative, covered, float2(imagePixel) + firstHitOffset, float2(uniforms.image.xy), motion
            );
            motionAOV.write(float4(velocity, 0, 0), gid);
        }
        if (writeGuides) {
            normalGuide.write(half4(half3(normal), half(min(depth, 65504.0f))), gid);
            albedoGuide.write(half4(half3(albedo), half(roughness)), gid);
        }
        if (frostfield_path_has_aovs) {
            // Keep slot and mask order in sync with PathTracer.AOVs.
            float depthScale = uniforms.projection.z >= ptEquirectangular
                ? 1.0f : dot(ray.direction, uniforms.cameraForward.xyz);
            if (aovMask & 1u) { albedoAOV.write(float4(diffuseAlbedo, coverage), gid); }
            if (aovMask & 2u) { normalAOV.write(float4(normal, coverage), gid); }
            if (aovMask & 4u) { depthAOV.write(float4(coverage > 0.0f ? depth * depthScale : INFINITY), gid); }
            if (aovMask & 8u) { roughnessAOV.write(float4(roughness), gid); }
            if (aovMask & 16u) { emissionAOV.write(float4(emission, coverage), gid); }
            if (aovMask & 32u) { objectAOV.write(uint4(objectID), gid); }
            if (aovMask & 64u) { materialAOV.write(uint4(materialID), gid); }
            if (aovMask & 128u) { specularAlbedoAOV.write(float4(specularAlbedo, coverage), gid); }
            if (aovMask & 256u) {
                float projectedDepth = 1.0f;
                if (coverage > 0.0f) {
                    float z = depth * depthScale;
                    float near = uniforms.projection.x;
                    float far = uniforms.projection.y;
                    float span = far - near;
                    projectedDepth = uniforms.projection.z == ptOrthographic
                        ? (z - near) / span
                        : (far / span) * (1.0f - near / z);
                }
                projectedDepthAOV.write(float4(saturate(projectedDepth)), gid);
            }
        }
    }

    float4 sum = float4(0);
    uint pixel = imagePixel.x + imagePixel.y * uniforms.image.x;
    bool blueDither = dynamics.counts.y == 2u;
    uint pixelSeed = ptHash((blueDither ? 0u : pixel) ^ ptHash(uniforms.controls.y));
    uint2 rankPixel = imagePixel & 63u;
    uint ditherOffset = blueDither ? uint(blueNoiseRanks[rankPixel.y * 64u + rankPixel.x]) << 12u : 0u;
    for (uint sample = 0u; sample < samples; ++sample) {
        PTSampler state = {
            ptHash(pixel ^ ptHash(uniforms.controls.y) ^ ptHash(sampleStart + sample)), pixelSeed,
            sampleStart + sample, 0u, dynamics.counts.y != 0u, ditherOffset
        };
        float2 offset = ptRandom2(state);
        if (frostfield_path_has_motion) { offset = firstHitOffset; }
        state.dimension = 1u;
        float2 lens = uniforms.cameraForward.w > 0.0f ? ptAperture(ptRandom2(state), uniforms.lens) : float2(0);
        sum += ptTrace(
            ptCamera(imagePixel, offset, lens, uniforms), state, uniforms, scene, lights,
            analyticLights, skyTable, skyMap, graphUniforms, graphTable, dynamics.limits.xyz,
            pixel + sampleStart + sample, media, lightProfiles, lightCookies,
            frostfield_path_has_mesh_emitters ? emitters : nullptr, dynamics.counts.w
        );
    }
    // The mean texture stores premultiplied radiance and camera-ray coverage.
    float total = float(sampleStart) + float(samples);
    float4 before = sampleStart == 0u ? float4(0) : previous.read(gid);
    float4 mean = total > 0.0f ? before + (sum - before * float(samples)) / total : float4(0);
    next.write(mean, gid);
    destination.write(half4(half3(clamp(mean.rgb, float3(0), float3(65504.0f))), half(saturate(mean.a))), gid);
}

kernel void frostfield_path_bake_texture(
    texture2d<float> source [[texture(0)]],
    texture2d_array<half, access::write> destination [[texture(1)]],
    constant uint& slice [[buffer(0)]],
    // 0 filters linearly. 1, for nearest-magnified maps, keeps the nearest texel on each axis
    // the bake magnifies and filters linearly on each axis it shrinks, so texel edges stay hard
    // without dropping texels. 2 point-samples 32-bit float sources on GPUs that cannot filter them.
    constant uint& mode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 size = uint2(destination.get_width(), destination.get_height());
    if (any(gid >= size) || slice >= destination.get_array_size()) { return; }
    constexpr sampler filtering(coord::normalized, address::clamp_to_edge, filter::linear);
    constexpr sampler point(coord::normalized, address::clamp_to_edge, filter::nearest);
    float2 uv = (float2(gid) + 0.5f) / float2(size);
    float4 value;
    if (mode == 1u) {
        // Reads work for every format, so this filters by hand: the weight is zero on nearest axes.
        uint2 sourceSize = uint2(source.get_width(), source.get_height());
        float2 position = uv * float2(sourceSize);
        bool2 shrinks = sourceSize > size;
        float2 filtered = position - 0.5f;
        float2 first = select(floor(position), floor(filtered), shrinks);
        float2 weight = select(float2(0), filtered - floor(filtered), shrinks);
        float2 last = float2(sourceSize - 1u);
        uint2 a = uint2(clamp(first, float2(0), last));
        uint2 b = uint2(clamp(first + 1.0f, float2(0), last));
        value = mix(
            mix(source.read(a), source.read(uint2(b.x, a.y)), weight.x),
            mix(source.read(uint2(a.x, b.y)), source.read(b), weight.x),
            weight.y
        );
    } else {
        value = mode == 2u ? source.sample(point, uv, level(0.0f)) : source.sample(filtering, uv, level(0.0f));
    }
    destination.write(half4(value), gid, slice);
}
