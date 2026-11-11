#include <metal_stdlib>
using namespace metal;

#include "FrostfieldDithering.h"

// Operator codes. They match HDRDisplay.ToneMapping.shaderCode.
constant constexpr uint toneNone = 0;
constant constexpr uint toneLinear = 1;
constant constexpr uint toneReinhard = 2;
constant constexpr uint toneReinhardExtended = 3;
constant constexpr uint toneCineon = 4;
constant constexpr uint toneACESFilmic = 5;
constant constexpr uint toneACESNarkowicz = 6;
constant constexpr uint toneAgX = 7;
constant constexpr uint toneNeutral = 8;

// Matches HDRDisplay.ShaderOptions: 28 bytes of 32-bit fields.
struct HDRDisplayOptions {
    float exposure;
    uint toneMapping;
    uint look;
    // 1 / W² for extended Reinhard, computed in double precision on the CPU and
    // limited to the largest finite float, so it never overflows here.
    float inverseWhiteSquared;
    float headroom;
    uint outputColorSpace;
    // 0 off. 1 adds the r187 dithering shift to the stored color, for an 8-bit target without
    // a transfer function. 2 adds it to the sRGB-encoded color, for an sRGB target.
    uint ditherMode;
};

// Implements the equations in the Khronos PBR Neutral specification (CC BY 4.0),
// as Three.js r187 NeutralToneMapping. The peak generalizes the specification's
// 1.0 to the display headroom: below the shoulder start the curve is unchanged,
// and the shoulder approaches `peak` instead of 1. With peak 1 it is the reference.
// https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral
static float3 pbrNeutral(float3 color, float peak) {
    constexpr float reflectance = 0.04f;
    constexpr float shoulderStart = 0.8f - reflectance;
    constexpr float desaturation = 0.15f;
    float low = min(color.r, min(color.g, color.b));
    color -= low < 2.0f * reflectance
        ? low * (1.0f - low / (4.0f * reflectance))
        : reflectance;
    float high = max(color.r, max(color.g, color.b));
    if (high < shoulderStart) { return color; }

    float shoulderWidth = peak - shoulderStart;
    float compressed = peak - shoulderWidth * shoulderWidth / (high + shoulderWidth - shoulderStart);
    float saturation = 1.0f / (1.0f + desaturation * (high - compressed));
    return mix(float3(compressed), color * (compressed / high), saturation);
}

// Three.js r187 CineonToneMapping: Jim Hejl and Richard Burgess-Dawson's filmic
// operator, which includes a 1/2.2 display gamma that the final pow removes.
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
static float3 cineon(float3 color) {
    color = max(float3(0), color - 0.004f);
    return pow((color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f), float3(2.2f));
}

// Stephen Hill's RRT and ODT fit, as Three.js r187 ACESFilmicToneMapping, including
// its 1/0.6 exposure for brighter viewing environments (three.js issue #19621).
// https://github.com/selfshadow/ltc_code/blob/master/webgl/shaders/ltc/ltc_blit.fs
static float3 acesFilmic(float3 color) {
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT, as columns.
    const float3x3 input = float3x3(
        float3(0.59719f, 0.07600f, 0.02840f),
        float3(0.35458f, 0.90834f, 0.13383f),
        float3(0.04823f, 0.01566f, 0.83777f)
    );
    // ODT_SAT => XYZ => D60_2_D65 => sRGB, as columns.
    const float3x3 output = float3x3(
        float3(1.60475f, -0.10208f, -0.00327f),
        float3(-0.53108f, 1.10813f, -0.07276f),
        float3(-0.07367f, -0.00605f, 1.07602f)
    );
    color = input * (color / 0.6f);
    float3 a = color * (color + 0.0245786f) - 0.000090537f;
    float3 b = color * (0.983729f * color + 0.4329510f) + 0.238081f;
    return saturate(output * (a / b));
}

// Krzysztof Narkowicz's luminance fit of the ACES RRT and ODT, applied per channel.
// The fit is pre-exposed so that input 1 maps to about 0.8.
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
static float3 acesNarkowicz(float3 x) {
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

// AgX as Three.js r187 AgXToneMapping: Blender's Rec.2020 inset and outset
// matrices (via Filament) with the iolite sixth-order sigmoid. Looks apply the
// ASC CDL values from Troy Sobotka's AgX configuration (sobotka/AgX config.ocio)
// in the sigmoid's display-encoded space, where Filament applies its looks, with
// luma taken after slope and power as the ASC CDL specifies. Negative sigmoid
// output (at most 0.00232 below zero) clamps to zero before the power.
// https://github.com/google/filament/pull/7236
// https://iolite-engine.com/blog_posts/minimal_agx_implementation
static float3 agx(float3 color, uint look) {
    const float3x3 srgbToRec2020 = float3x3(
        float3(0.6274f, 0.0691f, 0.0164f),
        float3(0.3293f, 0.9195f, 0.0880f),
        float3(0.0433f, 0.0113f, 0.8956f)
    );
    const float3x3 rec2020ToSRGB = float3x3(
        float3(1.6605f, -0.1246f, -0.0182f),
        float3(-0.5876f, 1.1329f, -0.1006f),
        float3(-0.0728f, -0.0083f, 1.1187f)
    );
    const float3x3 inset = float3x3(
        float3(0.856627153315983f, 0.137318972929847f, 0.11189821299995f),
        float3(0.0951212405381588f, 0.761241990602591f, 0.0767994186031903f),
        float3(0.0482516061458583f, 0.101439036467562f, 0.811302368396859f)
    );
    const float3x3 outset = float3x3(
        float3(1.1271005818144368f, -0.1413297634984383f, -0.14132976349843826f),
        float3(-0.11060664309660323f, 1.157823702216272f, -0.11060664309660294f),
        float3(-0.016493938717834573f, -0.016493938717834257f, 1.2519364065950405f)
    );
    // log2(2^-10 * 0.18) and log2(2^6.5 * 0.18).
    constexpr float minEV = -12.47393f;
    constexpr float maxEV = 4.026069f;

    color = inset * (srgbToRec2020 * color);
    color = clamp((log2(max(color, float3(1e-10f))) - minEV) / (maxEV - minEV), 0.0f, 1.0f);
    float3 x2 = color * color;
    float3 x4 = x2 * x2;
    color = 15.5f * x4 * x2 - 40.14f * x4 * color + 31.96f * x4
        - 6.868f * x2 * color + 0.4298f * x2 + 0.1191f * color - 0.00232f;

    if (look != 0) {
        // 1 Punchy: power 1.35, saturation 1.4. 2 Golden: slope (1, 0.9, 0.5), power 0.8, saturation 1.3.
        float3 slope = look == 2 ? float3(1.0f, 0.9f, 0.5f) : float3(1.0f);
        float power = look == 2 ? 0.8f : 1.35f;
        float saturation = look == 2 ? 1.3f : 1.4f;
        color = pow(max(color * slope, float3(0)), float3(power));
        float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
        color = luma + saturation * (color - luma);
    }

    color = pow(max(outset * color, float3(0)), float3(2.2f));
    return saturate(rec2020ToSRGB * color);
}

// The SDR curves (codes 1 to 7), mapping exposed scene-linear color to 0...1.
static float3 sdrToneMap(float3 color, constant HDRDisplayOptions& options) {
    switch (options.toneMapping) {
        case toneLinear: return saturate(color);
        case toneReinhard: return color / (1.0f + color);
        case toneReinhardExtended:
            // Reinhard et al. 2002, equation 4, per channel. With a finite 1 / W², a zero
            // channel stays zero and any overflow is +infinity, which clamps to white.
            return min(color * (1.0f + color * options.inverseWhiteSquared) / (1.0f + color), float3(1.0f));
        case toneCineon: return cineon(color);
        case toneACESFilmic: return acesFilmic(color);
        case toneACESNarkowicz: return acesNarkowicz(color);
        default: return agx(color, options.look);
    }
}

static inline float maxComponent(float3 v) { return max(v.r, max(v.g, v.b)); }

// Extends an SDR curve's highlights into an EDR headroom H > 1 without changing
// shadows or mid-tones. With m the pixel's largest exposed scene channel and w the
// curve's largest output at exposed white (1, 1, 1), the pixel's largest display
// channel rises by
//     E = (H - 1) · Δ² / ((Δ + H - w) · (Δ + 1 - w)),   Δ = w · max(m - 1, 0),
// and RGB scales with it, so the SDR curve's hue and saturation are kept. E is the
// difference between two PBR Neutral shoulders that leave the knee at the knee's
// gain w, one approaching H and one approaching 1. It is zero with zero slope at
// the knee (exposed white), increases with m, and approaches H - 1, so a gray
// input maps to T(x) + E, which stays below H.
static float3 extendHighlights(float3 mapped, float sceneHigh, float knee, float peak) {
    float delta = knee * (sceneHigh - 1.0f);
    float high = maxComponent(mapped);
    if (!(delta > 0.0f) || !(high > 0.0f)) { return mapped; }
    // Written as two ratios in 0...1, which cannot overflow for any finite delta.
    float extension = (peak - 1.0f) * (delta / (delta + peak - knee)) * (delta / (delta + 1.0f - knee));
    return mapped * ((high + extension) / high);
}

// Maps exposed, straight, nonnegative scene-linear color to display-linear color in
// 0...headroom. At headroom 1 each curve is its reference formula. Above 1, `.none`
// and `.linear` are unchanged apart from the clamp, Neutral aims its shoulder at the
// peak, and the other curves keep their SDR output for scene values up to exposed
// white and extend brighter pixels with `extendHighlights`. `knee` is the SDR
// curve's largest output at exposed white.
static float3 toneMap(float3 color, constant HDRDisplayOptions& options, float knee) {
    float peak = options.headroom;
    switch (options.toneMapping) {
        case toneNone: return min(color, float3(65504.0f));
        case toneLinear: return min(color, float3(peak));
        case toneNeutral: return clamp(pbrNeutral(color, peak), 0.0f, peak);
        default: break;
    }
    float3 mapped = sdrToneMap(color, options);
    if (peak > 1.0f) {
        mapped = min(extendHighlights(mapped, maxComponent(color), knee, peak), float3(peak));
    }
    return mapped;
}

struct HDRDisplayVertex {
    float4 position [[position]];
    // The SDR curve's largest output at exposed white, the knee of the highlight
    // extension. It depends only on the options, so each of the three vertices
    // evaluates it once instead of every fragment.
    float knee [[flat]];
};

vertex HDRDisplayVertex frostfield_hdr_vertex(
    uint vertexID [[vertex_id]],
    constant HDRDisplayOptions& options [[buffer(0)]]
) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    HDRDisplayVertex out;
    out.position = float4(positions[vertexID], 0, 1);
    out.knee = maxComponent(sdrToneMap(float3(1.0f), options));
    return out;
}

// Linear D65 transforms derived from the sRGB, Display P3 and Rec. 2020 primaries.
static float3 outputRGB(float3 rgb, uint space) {
    switch (space) {
        case 1:
            return float3x3(
                float3(0.82246197f, 0.03319420f, 0.01708263f),
                float3(0.17753803f, 0.96680580f, 0.07239744f),
                float3(0.0f, 0.0f, 0.91051993f)
            ) * rgb;
        case 2:
            return float3x3(
                float3(0.62740390f, 0.06909729f, 0.01639144f),
                float3(0.32928304f, 0.91954040f, 0.08801331f),
                float3(0.04331307f, 0.01136232f, 0.89559525f)
            ) * rgb;
        default:
            return rgb;
    }
}

static float4 displayColor(float4 sample, constant HDRDisplayOptions& options, float knee) {
    float alpha = saturate(sample.a);
    if (!(alpha > 0)) { return float4(0); }
    float scale = options.exposure / alpha;
    float3 mapped;
    if (options.outputColorSpace != 0 && options.toneMapping == toneNone) {
        mapped = clamp(outputRGB(sample.rgb * scale, options.outputColorSpace), -65504.0f, 65504.0f);
    } else {
        float3 straightRGB = max(sample.rgb, float3(0)) * scale;
        mapped = outputRGB(toneMap(straightRGB, options, knee), options.outputColorSpace);
    }
    // Nonfinite intermediate values become black rather than spreading through blending.
    mapped = select(mapped, float3(0), isnan(mapped));
    return float4(mapped * alpha, alpha);
}

static float luminance(float4 sample, uint space) {
    switch (space) {
        case 1: return sqrt(max(0.0f, dot(sample.rgb, float3(0.22897456f, 0.69173852f, 0.07928691f))));
        case 2: return sqrt(max(0.0f, dot(sample.rgb, float3(0.26270021f, 0.67799807f, 0.05930172f))));
        default: return sqrt(dot(sample.rgb, float3(0.2126f, 0.7152f, 0.0722f)));
    }
}

static float4 displaySample(
    texture2d<float> source, texture2d<float> preview, uint2 pixel, uint2 outputSize, bool resamples, float weight
) {
    constexpr sampler filter(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(pixel) + 0.5f) / float2(outputSize);
    float4 value = resamples ? source.sample(filter, uv) : source.read(pixel);
    return weight < 1.0f ? mix(preview.sample(filter, uv), value, weight) : value;
}

// Three.js r187 dithering of the final premultiplied color, as `dithering_fragment` runs
// after premultiplication. Fully transparent pixels stay exactly zero. `position` is the
// pixel center from the top-left corner. gl_FragCoord counts from the bottom-left.
static float4 ditherOutput(float4 color, constant HDRDisplayOptions& options, float2 position, uint2 outputSize) {
    if (options.ditherMode == 0u || !(color.a > 0.0f)) { return color; }
    float2 fragmentCoordinate = float2(position.x, float(outputSize.y) - position.y);
    return float4(frostfieldR187DitherOutput(color.rgb, fragmentCoordinate, options.ditherMode), color.a);
}

fragment float4 frostfield_hdr_fragment(
    HDRDisplayVertex in [[stage_in]],
    texture2d<float> source [[texture(0)]],
    texture2d<float> preview [[texture(1)]],
    constant HDRDisplayOptions& options [[buffer(0)]],
    constant float& antiAliasing [[buffer(1)]],
    constant uint2& outputSize [[buffer(2)]],
    constant float& weight [[buffer(3)]]
) {
    uint2 pixel = uint2(in.position.xy);
    float knee = in.knee;
    bool resamples = any(outputSize != uint2(source.get_width(), source.get_height()));
    float4 center = displayColor(displaySample(source, preview, pixel, outputSize, resamples, weight), options, knee);
    if (antiAliasing == 0) { return ditherOutput(center, options, in.position.xy, outputSize); }

    uint2 maximum = outputSize - 1u;
    uint left = pixel.x == 0 ? 0 : pixel.x - 1u;
    uint up = pixel.y == 0 ? 0 : pixel.y - 1u;
    float4 west = displayColor(displaySample(source, preview, uint2(left, pixel.y), outputSize, resamples, weight), options, knee);
    float4 east = displayColor(displaySample(source, preview, uint2(min(pixel.x + 1u, maximum.x), pixel.y), outputSize, resamples, weight), options, knee);
    float4 north = displayColor(displaySample(source, preview, uint2(pixel.x, up), outputSize, resamples, weight), options, knee);
    float4 south = displayColor(displaySample(source, preview, uint2(pixel.x, min(pixel.y + 1u, maximum.y)), outputSize, resamples, weight), options, knee);

    // Straight horizontal and vertical edges stay sharp. Mix display-linear
    // samples near stair steps where the gradient has both components.
    uint space = options.outputColorSpace;
    float2 lumaGradient(luminance(east, space) - luminance(west, space),
                        luminance(south, space) - luminance(north, space));
    float2 alphaGradient(east.a - west.a, south.a - north.a);
    float2 gradient = abs(dot(alphaGradient, alphaGradient) > dot(lumaGradient, lumaGradient)
        ? alphaGradient : lumaGradient);
    float high = max(gradient.x, gradient.y);
    float low = min(gradient.x, gradient.y);
    if (low < 0.0625f) { return ditherOutput(center, options, in.position.xy, outputSize); }
    return ditherOutput(
        mix(center, (west + east + north + south) * 0.25f, 0.5f * low / high), options, in.position.xy, outputSize
    );
}
