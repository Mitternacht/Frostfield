// KHR material extension math shared by the raster PBR and path-tracing shaders.
// Runtime-compiled variants splice this file in place of its #include line.
#ifndef FROSTFIELD_MATERIAL_EXTENSIONS_H
#define FROSTFIELD_MATERIAL_EXTENSIONS_H
#include <metal_stdlib>
using namespace metal;

// KHR_texture_transform: two rows per map, u' = dot(r0.xy, uv) + r0.z and
// v' = dot(r1.xy, uv) + r1.z, matching PhysicallyBasedMaterial.TextureTransform.
static inline float2 frostfieldTransformUV(float2 uv, float4 row0, float4 row1) {
    return float2(dot(row0.xy, uv) + row0.z, dot(row1.xy, uv) + row1.z);
}

// Stock normal-map encodings, matching PhysicallyBasedMaterial.NormalEncoding.
static inline float3 frostfieldDecodeNormalMap(float3 texel, uint encoding) {
    if (encoding == 0u) return texel * 2.0f - 1.0f;
    float2 xy = clamp(encoding == 1u ? texel.xy * 2.0f - 1.0f : texel.xy, -1.0f, 1.0f);
    return float3(xy, sqrt(max(0.0f, 1.0f - dot(xy, xy))));
}

// The base normal slot can hold a scalar height map instead of a stored vector.
constant uint frostfieldBumpMapKind = 3u;

// Surface-gradient bump mapping. The two height differences span the raster
// pixel or ray footprint, and the matching surface directions are normalized.
static inline float3 frostfieldBumpNormal(float3 normal, float3 dx, float3 dy, float2 heightDelta) {
    float x2 = dot(dx, dx), y2 = dot(dy, dy);
    if (!(x2 > 0.0f && y2 > 0.0f) || !all(isfinite(heightDelta))) return normal;
    float3 x = dx * rsqrt(x2), y = dy * rsqrt(y2);
    float3 rx = cross(y, normal), ry = cross(normal, x);
    float determinant = dot(x, rx);
    if (!(abs(determinant) > 1e-6f)) return normal;
    float3 bumped = abs(determinant) * normal - sign(determinant) * (heightDelta.x * rx + heightDelta.y * ry);
    float magnitude = max(max(abs(bumped.x), abs(bumped.y)), abs(bumped.z));
    return magnitude > 0.0f && all(isfinite(bumped)) ? normalize(bumped / magnitude) : normal;
}

// KHR_materials_sheen. Perceptual roughness below this floor keeps the Charlie
// exponent finite; the albedo table uses the same floor.
constant float frostfieldMinimumSheenRoughness = 0.07f;

// The Charlie distribution (Estevez and Kulla 2017) with alpha = roughness squared.
static inline float frostfieldCharlie(float cosine, float roughness) {
    float clamped = max(roughness, frostfieldMinimumSheenRoughness);
    float inverseAlpha = 1.0f / (clamped * clamped);
    float sine2 = max(1.0f - cosine * cosine, 0.0078125f);
    return (2.0f + inverseAlpha) * pow(sine2, 0.5f * inverseAlpha) * (0.5f * M_1_PI_F);
}

// Neubelt and Pettineo's sheen visibility, 1 / (4 (NoL + NoV - NoL NoV)).
static inline float frostfieldSheenVisibility(float noL, float noV) {
    return 1.0f / (4.0f * max(noL + noV - noL * noV, 1e-6f));
}

// Directional albedo of the unit-color Charlie and Neubelt sheen lobe, tabulated by
// numerical integration over the hemisphere (800 x 800 cells). Rows are perceptual
// roughness i / 15 with the 0.07 floor; columns are cos(theta) = (j / 15)^2.
// The visibility term is not energy conserving, so grazing albedo exceeds one.
constant float frostfieldSheenAlbedoTable[256] = {
    5.68618f, 4.86671f, 3.50752f, 2.22768f, 1.23562f, 0.58286f, 0.22435f, 0.06644f, 0.01394f, 0.00184f, 0.00013f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f,
    5.68618f, 4.86671f, 3.50752f, 2.22768f, 1.23562f, 0.58286f, 0.22435f, 0.06644f, 0.01394f, 0.00184f, 0.00013f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f,
    3.02142f, 2.75718f, 2.27395f, 1.75292f, 1.26676f, 0.85358f, 0.53060f, 0.29946f, 0.15003f, 0.06462f, 0.02282f, 0.00613f, 0.00111f, 0.00010f, 0.00000f, 0.00000f,
    2.04890f, 1.91880f, 1.67116f, 1.38974f, 1.10858f, 0.84722f, 0.61756f, 0.42620f, 0.27553f, 0.16432f, 0.08839f, 0.04139f, 0.01589f, 0.00444f, 0.00066f, 0.00001f,
    1.57120f, 1.49144f, 1.33666f, 1.15629f, 0.97011f, 0.78954f, 0.62191f, 0.47209f, 0.34313f, 0.23656f, 0.15256f, 0.09007f, 0.04693f, 0.02009f, 0.00590f, 0.00042f,
    1.29137f, 1.23592f, 1.12726f, 0.99896f, 0.86418f, 0.73040f, 0.60238f, 0.48343f, 0.37585f, 0.28124f, 0.20060f, 0.13435f, 0.08237f, 0.04405f, 0.01826f, 0.00340f,
    1.11019f, 1.06832f, 0.98588f, 0.88789f, 0.78396f, 0.67940f, 0.57753f, 0.48063f, 0.39031f, 0.30781f, 0.23402f, 0.16960f, 0.11500f, 0.07043f, 0.03590f, 0.01118f,
    0.98504f, 0.95152f, 0.88541f, 0.80658f, 0.72254f, 0.63734f, 0.55343f, 0.47242f, 0.39545f, 0.32338f, 0.25687f, 0.19646f, 0.14259f, 0.09565f, 0.05594f, 0.02370f,
    0.89458f, 0.86656f, 0.81130f, 0.74533f, 0.67484f, 0.60306f, 0.53191f, 0.46259f, 0.39589f, 0.33240f, 0.27255f, 0.21671f, 0.16518f, 0.11826f, 0.07622f, 0.03935f,
    0.82696f, 0.80274f, 0.75504f, 0.69812f, 0.63724f, 0.57513f, 0.51334f, 0.45278f, 0.39404f, 0.33750f, 0.28344f, 0.23205f, 0.18349f, 0.13791f, 0.09548f, 0.05635f,
    0.77505f, 0.75358f, 0.71136f, 0.66102f, 0.60721f, 0.55228f, 0.49752f, 0.44367f, 0.39118f, 0.34028f, 0.29112f, 0.24379f, 0.19833f, 0.15476f, 0.11311f, 0.07340f,
    0.73436f, 0.71494f, 0.67680f, 0.63141f, 0.58293f, 0.53346f, 0.48411f, 0.43550f, 0.38797f, 0.34166f, 0.29664f, 0.25291f, 0.21043f, 0.16912f, 0.12890f, 0.08967f,
    0.70189f, 0.68402f, 0.64902f, 0.60743f, 0.56308f, 0.51785f, 0.47275f, 0.42829f, 0.38475f, 0.34221f, 0.30068f, 0.26010f, 0.22037f, 0.18135f, 0.14287f, 0.10474f,
    0.67558f, 0.65894f, 0.62639f, 0.58778f, 0.54668f, 0.50481f, 0.46310f, 0.42199f, 0.38169f, 0.34227f, 0.30369f, 0.26585f, 0.22861f, 0.19178f, 0.15515f, 0.11845f,
    0.65399f, 0.63832f, 0.60772f, 0.57151f, 0.53300f, 0.49385f, 0.45487f, 0.41649f, 0.37887f, 0.34205f, 0.30597f, 0.27050f, 0.23548f, 0.20070f, 0.16590f, 0.13077f,
    0.63607f, 0.62118f, 0.59217f, 0.55788f, 0.52150f, 0.48456f, 0.44783f, 0.41170f, 0.37631f, 0.34168f, 0.30772f, 0.27432f, 0.24127f, 0.20836f, 0.17530f, 0.14176f
};

static inline float frostfieldSheenAlbedo(float cosine, float roughness) {
    float column = sqrt(saturate(cosine)) * 15.0f;
    float row = saturate(roughness) * 15.0f;
    uint c = min(uint(column), 14u);
    uint r = min(uint(row), 14u);
    float u = column - float(c);
    float v = row - float(r);
    float lower = mix(frostfieldSheenAlbedoTable[r * 16u + c], frostfieldSheenAlbedoTable[r * 16u + c + 1u], u);
    float upper = mix(frostfieldSheenAlbedoTable[r * 16u + 16u + c], frostfieldSheenAlbedoTable[r * 16u + 17u + c], u);
    return mix(lower, upper, v);
}

// The specification's base-layer scaling, min(1 - max(color) E(NoV), 1 - max(color) E(NoL)),
// clamped at zero where the lobe's albedo exceeds one.
static inline float frostfieldSheenScaling(float3 color, float roughness, float noV, float noL) {
    float peak = max(color.x, max(color.y, color.z));
    return max(0.0f, min(
        1.0f - peak * frostfieldSheenAlbedo(noV, roughness), 1.0f - peak * frostfieldSheenAlbedo(noL, roughness)
    ));
}

// KHR_materials_anisotropy: alpha along the anisotropy direction and along its bitangent.
static inline float2 frostfieldAnisotropicAlpha(float alpha, float strength) {
    return float2(mix(alpha, 1.0f, strength * strength), alpha);
}

// Anisotropic GGX for a unit half vector in the (direction, bitangent, normal) frame.
static inline float frostfieldAnisotropicGGX(float3 halfway, float2 alpha) {
    float3 scaled = float3(halfway.x / alpha.x, halfway.y / alpha.y, halfway.z);
    float squared = dot(scaled, scaled);
    return M_1_PI_F / max(alpha.x * alpha.y * squared * squared, 1e-20f);
}

// sqrt(alpha_x^2 x^2 + alpha_y^2 y^2 + z^2), the anisotropic Smith term for a local direction.
static inline float frostfieldAnisotropicRoot(float3 direction, float2 alpha) {
    return length(float3(alpha * direction.xy, direction.z));
}

// Height-correlated anisotropic Smith visibility, G2 / (4 NoL NoV).
static inline float frostfieldAnisotropicVisibility(float3 view, float3 light, float2 alpha) {
    float noV = abs(view.z);
    float noL = abs(light.z);
    return 0.5f / max(noL * frostfieldAnisotropicRoot(view, alpha) + noV * frostfieldAnisotropicRoot(light, alpha), 1e-12f);
}

// KHR_materials_iridescence: Belcour and Barla's thin-film Fresnel (2017), following
// the Khronos glTF Sample Viewer formulation. Thickness is in nanometers; the base
// reflectance converts to a per-channel index of refraction.
static inline float3 frostfieldIridescenceSensitivity(float opticalPath, float3 shift) {
    float phase = 2.0f * M_PI_F * opticalPath * 1.0e-9f;
    const float3 value = float3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    const float3 position = float3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    const float3 variance = float3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
    float3 xyz = value * sqrt(2.0f * M_PI_F * variance) * precise::cos(position * phase + shift)
        * precise::exp(-(phase * phase) * variance);
    xyz.x += 9.7470e-14f * sqrt(2.0f * M_PI_F * 4.5282e+09f) * precise::cos(2.2399e+06f * phase + shift.x)
        * precise::exp(-4.5282e+09f * phase * phase);
    xyz /= 1.0685e-7f;
    return float3(
        3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
        -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
        0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z
    );
}

static inline float frostfieldSchlickWeight(float cosine) {
    float x = 1.0f - saturate(cosine);
    float squared = x * x;
    return squared * squared * x;
}

static inline float3 frostfieldIridescence(
    float outsideIOR, float filmIOR, float cosine, float thickness, float3 baseF0
) {
    // The film fades to the base interface as its thickness approaches zero.
    float film = mix(outsideIOR, filmIOR, smoothstep(0.0f, 0.03f, thickness));
    float cosine1 = saturate(cosine);
    float ratio = outsideIOR / film;
    float cosine2Squared = 1.0f - ratio * ratio * (1.0f - cosine1 * cosine1);
    if (cosine2Squared < 0.0f) { return float3(1.0f); }
    float cosine2 = sqrt(cosine2Squared);

    float r0 = (film - outsideIOR) / (film + outsideIOR);
    float r12 = r0 * r0 + (1.0f - r0 * r0) * frostfieldSchlickWeight(cosine1);
    float t121 = 1.0f - r12;
    float phi12 = film < outsideIOR ? M_PI_F : 0.0f;
    float phi21 = M_PI_F - phi12;

    float3 root = sqrt(clamp(baseF0, float3(0.0f), float3(0.9999f)));
    float3 baseIOR = (1.0f + root) / (1.0f - root);
    float3 r1 = (baseIOR - film) / (baseIOR + film);
    r1 *= r1;
    float3 r23 = r1 + (1.0f - r1) * frostfieldSchlickWeight(cosine2);
    float3 phi23 = select(float3(0.0f), float3(M_PI_F), baseIOR < film);

    float opticalPath = 2.0f * film * thickness * cosine2;
    float3 phi = phi21 + phi23;
    float3 r123 = clamp(r12 * r23, float3(1e-5f), float3(0.9999f));
    float3 r123Root = sqrt(r123);
    float3 rs = t121 * t121 * r23 / (1.0f - r123);
    float3 result = r12 + rs;
    float3 cm = rs - t121;
    for (uint m = 1u; m <= 2u; ++m) {
        cm *= r123Root;
        result += cm * 2.0f * frostfieldIridescenceSensitivity(float(m) * opticalPath, float(m) * phi);
    }
    // Out-of-gamut spectra can produce negative linear components.
    return max(result, float3(0.0f));
}

// KHR_materials_dispersion: n(lambda) = max(n + (n - 1) (dispersion / 20) (523655 / lambda^2 - 1.5168), 1)
// with lambda in nanometers; dispersion is 20 divided by the Abbe number.
static inline float frostfieldDispersedIOR(float ior, float dispersion, float wavelength) {
    return max(ior + (ior - 1.0f) * dispersion * 0.05f * (523655.0f / (wavelength * wavelength) - 1.5168f), 1.0f);
}

// The path tracer's red, green and blue wavelengths: the Fraunhofer C, d and F lines, in nanometers.
constant float3 frostfieldDispersionWavelengths = float3(656.27f, 587.56f, 486.13f);

// KHR_materials_volume_scatter and subsurface media. The Henyey-Greenstein phase function in
// solid angle for mu, the cosine between the incoming and outgoing propagation directions;
// positive g scatters forward.
static inline float frostfieldHenyeyGreenstein(float mu, float g) {
    float denominator = max(1.0f + g * g - 2.0f * g * mu, 1e-12f);
    return (1.0f - g * g) / (4.0f * M_PI_F * denominator * sqrt(denominator));
}

// Inverts the Henyey-Greenstein CDF: the scattering cosine for a uniform number.
static inline float frostfieldSampleHenyeyGreensteinCosine(float g, float random) {
    if (abs(g) < 1e-3f) { return clamp(1.0f - 2.0f * random, -1.0f, 1.0f); }
    float ratio = (1.0f - g * g) / (1.0f - g + 2.0f * g * random);
    return clamp((1.0f + g * g - ratio * ratio) / (2.0f * g), -1.0f, 1.0f);
}

// A propagation direction scattered from `direction` (unit) by the phase function.
static inline float3 frostfieldSampleHenyeyGreenstein(float3 direction, float g, float2 random) {
    float mu = frostfieldSampleHenyeyGreensteinCosine(g, random.x);
    float sine = sqrt(max(0.0f, 1.0f - mu * mu));
    float angle = 2.0f * M_PI_F * random.y;
    float3 axis = abs(direction.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(axis, direction));
    float3 bitangent = cross(direction, tangent);
    return normalize(tangent * (sine * cos(angle)) + bitangent * (sine * sin(angle)) + direction * mu);
}

// The single-scattering albedo whose index-matched, semi-infinite, isotropic medium reflects
// the multiple-scattering albedo `color` under diffuse light (Kulla and Conty 2017, as
// KHR_materials_volume_scatter specifies).
static inline float3 frostfieldSingleScatteringAlbedo(float3 color) {
    float3 albedo = saturate(color);
    float3 root = 4.09712f + 4.20863f * albedo - sqrt(9.59217f + 41.6808f * albedo + 17.7126f * albedo * albedo);
    return saturate(1.0f - root * root);
}

// KHR_materials_diffuse_roughness: the energy-preserving Oren-Nayar BRDF (EON) of Portsmouth,
// Kutz and Hill (2025), which OpenPBR uses, with roughness r in [0, 1]. Cosines are measured
// from the shading normal; s = dot(l, v) - NoL NoV. Reference: github.com/portsmouth/EON-diffuse (LICENSES/EON-diffuse.txt).
constant float frostfieldFON1 = 0.5f - 2.0f / (3.0f * M_PI_F);
constant float frostfieldFON2 = 2.0f / 3.0f - 28.0f / (15.0f * M_PI_F);

// The directional albedo of the single-scattering Fujii Oren-Nayar lobe, exact.
static inline float frostfieldFONAlbedo(float mu, float r) {
    float a = 1.0f / (1.0f + frostfieldFON1 * r);
    float b = r * a;
    mu = clamp(mu, -1.0f, 1.0f);
    float sine = sqrt(max(0.0f, 1.0f - mu * mu));
    float g = sine * (acos(mu) - sine * mu)
        + (2.0f / 3.0f) * (sine * mu * (1.0f + sine + sine * sine) / (1.0f + sine) - sine);
    return a + b * M_1_PI_F * g;
}

// The EON BRDF value (without the cosine) for albedo rho.
static inline float3 frostfieldEON(float3 rho, float r, float noL, float noV, float s) {
    float sOverT = s > 0.0f ? s / max(max(noL, noV), 1e-6f) : s;
    float a = 1.0f / (1.0f + frostfieldFON1 * r);
    float3 single = rho * (M_1_PI_F * a * (1.0f + r * sOverT));
    float albedoOut = frostfieldFONAlbedo(noV, r);
    float albedoIn = frostfieldFONAlbedo(noL, r);
    float average = a * (1.0f + frostfieldFON2 * r);
    float3 multipleAlbedo = rho * rho * average / max(1.0f - rho * (1.0f - average), float3(1e-6f));
    const float epsilon = 1e-7f;
    float3 multiple = multipleAlbedo * (M_1_PI_F * max(epsilon, 1.0f - albedoOut) * max(epsilon, 1.0f - albedoIn)
        / max(epsilon, 1.0f - average));
    return max(single + multiple, float3(0.0f));
}

// The EON directional albedo for albedo rho, which is exact under uniform illumination.
static inline float3 frostfieldEONAlbedo(float3 rho, float r, float mu) {
    float a = 1.0f / (1.0f + frostfieldFON1 * r);
    float single = frostfieldFONAlbedo(mu, r);
    float average = a * (1.0f + frostfieldFON2 * r);
    float3 multipleAlbedo = rho * rho * average / max(1.0f - rho * (1.0f - average), float3(1e-6f));
    return rho * single + multipleAlbedo * (1.0f - single);
}

#endif
