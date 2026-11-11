#include <metal_stdlib>
using namespace metal;

vertex float4 frostfield_aa_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertexID], 0, 1);
}

// MARK: - FXAA 3.11 quality
//
// A port of the PC quality path of Timothy Lottes' FXAA 3.11 (NVIDIA, 2011), with
// the luma that FXAA expects in alpha written to a separate texture by a first pass,
// so premultiplied alpha passes through. Luma is the square root of Rec.709
// luminance, FXAA's suggested perceptual approximation. Color and luma are sampled
// bilinearly with clamp-to-edge addressing, as FxaaTexTop and FxaaTexOff do.

fragment float frostfield_fxaa_luma_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]]
) {
    float3 rgb = max(source.read(uint2(position.xy)).rgb, float3(0));
    return sqrt(dot(rgb, float3(0.2126f, 0.7152f, 0.0722f)));
}

// Matches FXAA.ShaderOptions: 72 bytes.
struct FXAAOptions {
    float2 texelSize;
    float subpixel;
    float edgeThreshold;
    float edgeThresholdMinimum;
    uint stepCount;
    float steps[12];
};

static inline float fxaaLuma(texture2d<float> luma, int2 pixel, int2 maximum) {
    return luma.read(uint2(clamp(pixel, int2(0), maximum))).r;
}

fragment float4 frostfield_fxaa_fragment(
    float4 position [[position]],
    texture2d<float> color [[texture(0)]],
    texture2d<float> luma [[texture(1)]],
    constant FXAAOptions& o [[buffer(0)]]
) {
    constexpr sampler bilinear(coord::normalized, address::clamp_to_edge, filter::linear);
    int2 pixel = int2(position.xy);
    int2 maximum = int2(luma.get_width(), luma.get_height()) - 1;
    float2 rcpFrame = o.texelSize;
    float2 posM = position.xy * rcpFrame;
    float4 rgbaM = color.read(uint2(pixel));

    float lumaM = fxaaLuma(luma, pixel, maximum);
    float lumaS = fxaaLuma(luma, pixel + int2(0, 1), maximum);
    float lumaE = fxaaLuma(luma, pixel + int2(1, 0), maximum);
    float lumaN = fxaaLuma(luma, pixel + int2(0, -1), maximum);
    float lumaW = fxaaLuma(luma, pixel + int2(-1, 0), maximum);

    float maxSM = max(lumaS, lumaM);
    float minSM = min(lumaS, lumaM);
    float maxESM = max(lumaE, maxSM);
    float minESM = min(lumaE, minSM);
    float maxWN = max(lumaN, lumaW);
    float minWN = min(lumaN, lumaW);
    float rangeMax = max(maxWN, maxESM);
    float rangeMin = min(minWN, minESM);
    float rangeMaxScaled = rangeMax * o.edgeThreshold;
    float range = rangeMax - rangeMin;
    float rangeMaxClamped = max(o.edgeThresholdMinimum, rangeMaxScaled);
    if (range < rangeMaxClamped) { return rgbaM; }

    float lumaNW = fxaaLuma(luma, pixel + int2(-1, -1), maximum);
    float lumaSE = fxaaLuma(luma, pixel + int2(1, 1), maximum);
    float lumaNE = fxaaLuma(luma, pixel + int2(1, -1), maximum);
    float lumaSW = fxaaLuma(luma, pixel + int2(-1, 1), maximum);

    float lumaNS = lumaN + lumaS;
    float lumaWE = lumaW + lumaE;
    float subpixRcpRange = 1.0f / range;
    float subpixNSWE = lumaNS + lumaWE;
    float edgeHorz1 = (-2.0f * lumaM) + lumaNS;
    float edgeVert1 = (-2.0f * lumaM) + lumaWE;
    float lumaNESE = lumaNE + lumaSE;
    float lumaNWNE = lumaNW + lumaNE;
    float edgeHorz2 = (-2.0f * lumaE) + lumaNESE;
    float edgeVert2 = (-2.0f * lumaN) + lumaNWNE;
    float lumaNWSW = lumaNW + lumaSW;
    float lumaSWSE = lumaSW + lumaSE;
    float edgeHorz4 = (abs(edgeHorz1) * 2.0f) + abs(edgeHorz2);
    float edgeVert4 = (abs(edgeVert1) * 2.0f) + abs(edgeVert2);
    float edgeHorz3 = (-2.0f * lumaW) + lumaNWSW;
    float edgeVert3 = (-2.0f * lumaS) + lumaSWSE;
    float edgeHorz = abs(edgeHorz3) + edgeHorz4;
    float edgeVert = abs(edgeVert3) + edgeVert4;
    float subpixNWSWNESE = lumaNWSW + lumaNESE;
    float lengthSign = rcpFrame.x;
    bool horzSpan = edgeHorz >= edgeVert;
    float subpixA = subpixNSWE * 2.0f + subpixNWSWNESE;
    if (!horzSpan) { lumaN = lumaW; }
    if (!horzSpan) { lumaS = lumaE; }
    if (horzSpan) { lengthSign = rcpFrame.y; }
    float subpixB = (subpixA * (1.0f / 12.0f)) - lumaM;
    float gradientN = lumaN - lumaM;
    float gradientS = lumaS - lumaM;
    float lumaNN = lumaN + lumaM;
    float lumaSS = lumaS + lumaM;
    bool pairN = abs(gradientN) >= abs(gradientS);
    float gradient = max(abs(gradientN), abs(gradientS));
    if (pairN) { lengthSign = -lengthSign; }
    float subpixC = saturate(abs(subpixB) * subpixRcpRange);

    float2 posB = posM;
    float2 offNP = horzSpan ? float2(rcpFrame.x, 0.0f) : float2(0.0f, rcpFrame.y);
    if (!horzSpan) { posB.x += lengthSign * 0.5f; }
    if (horzSpan) { posB.y += lengthSign * 0.5f; }

    float2 posN = posB - offNP * o.steps[0];
    float2 posP = posB + offNP * o.steps[0];
    float subpixD = ((-2.0f) * subpixC) + 3.0f;
    float lumaEndN = luma.sample(bilinear, posN).r;
    float subpixE = subpixC * subpixC;
    float lumaEndP = luma.sample(bilinear, posP).r;

    if (!pairN) { lumaNN = lumaSS; }
    float gradientScaled = gradient * (1.0f / 4.0f);
    float lumaMM = lumaM - lumaNN * 0.5f;
    float subpixF = subpixD * subpixE;
    bool lumaMLTZero = lumaMM < 0.0f;

    lumaEndN -= lumaNN * 0.5f;
    lumaEndP -= lumaNN * 0.5f;
    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    if (!doneN) { posN -= offNP * o.steps[1]; }
    if (!doneP) { posP += offNP * o.steps[1]; }

    // FXAA 3.11 unrolls these steps as nested blocks; each one samples the ends
    // that have not found the edge's end and then advances them by the next step.
    for (uint step = 2; step < o.stepCount && (!doneN || !doneP); ++step) {
        if (!doneN) { lumaEndN = luma.sample(bilinear, posN).r - lumaNN * 0.5f; }
        if (!doneP) { lumaEndP = luma.sample(bilinear, posP).r - lumaNN * 0.5f; }
        doneN = abs(lumaEndN) >= gradientScaled;
        doneP = abs(lumaEndP) >= gradientScaled;
        if (!doneN) { posN -= offNP * o.steps[step]; }
        if (!doneP) { posP += offNP * o.steps[step]; }
    }

    float dstN = horzSpan ? posM.x - posN.x : posM.y - posN.y;
    float dstP = horzSpan ? posP.x - posM.x : posP.y - posM.y;
    bool goodSpanN = (lumaEndN < 0.0f) != lumaMLTZero;
    float spanLength = dstP + dstN;
    bool goodSpanP = (lumaEndP < 0.0f) != lumaMLTZero;
    float spanLengthRcp = 1.0f / spanLength;

    bool directionN = dstN < dstP;
    float dst = min(dstN, dstP);
    bool goodSpan = directionN ? goodSpanN : goodSpanP;
    float subpixG = subpixF * subpixF;
    float pixelOffset = (dst * (-spanLengthRcp)) + 0.5f;
    float subpixH = subpixG * o.subpixel;

    float pixelOffsetGood = goodSpan ? pixelOffset : 0.0f;
    float pixelOffsetSubpix = max(pixelOffsetGood, subpixH);
    if (!horzSpan) { posM.x += pixelOffsetSubpix * lengthSign; }
    if (horzSpan) { posM.y += pixelOffsetSubpix * lengthSign; }
    // Premultiplied RGBA filters as one value, so alpha edges smooth with color.
    return color.sample(bilinear, posM);
}

// MARK: - Temporal anti-aliasing

// Matches TemporalAntiAliasing.ShaderOptions: 224 bytes.
struct TAAOptions {
    float4x4 worldFromJitteredClip;
    float4x4 currentClipFromWorld;
    float4x4 previousClipFromWorld;
    float currentWeight;
    uint flags;
    float2 jitterDelta;
    float motionDepthThreshold;
    float maxVelocityLength;
    float edgeDepthDifference;
};

constant constexpr uint taaHistoryValid = 1;
constant constexpr uint taaClipHistory = 2;
constant constexpr uint taaReversedDepth = 4;
// RGBA32Float history (no clipping), read at the nearest texel rather than filtered:
// 32-bit float filtering is optional on Apple GPUs.
constant constexpr uint taaExactHistory = 8;
constant constexpr uint taaReferenceResolve = 16;
constant constexpr uint taaSubpixelCorrection = 32;

struct TAAOutput {
    float4 history [[color(0)]];
    float4 resolved [[color(1)]];
};

static inline float3 rgbToYCoCg(float3 c) {
    return float3(
        0.25f * c.r + 0.5f * c.g + 0.25f * c.b,
        0.5f * c.r - 0.5f * c.b,
        -0.25f * c.r + 0.5f * c.g - 0.25f * c.b
    );
}

static inline float3 yCoCgToRGB(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Moves the history toward the box center until it lies inside the box
// (Playdead's INSIDE TAA, Pedersen 2016), rather than clamping each axis.
static inline float3 clipToBox(float3 low, float3 high, float3 history) {
    float3 center = 0.5f * (high + low);
    float3 extent = 0.5f * (high - low) + 1e-6f;
    float3 offset = history - center;
    float3 units = abs(offset / extent);
    float largest = max(units.x, max(units.y, units.z));
    return largest > 1.0f ? center + offset / largest : history;
}

// Bicubic Catmull-Rom history fetch from nine bilinear taps (Pettineo 2016). At a
// texel center it returns that texel exactly, so a still camera never blurs history.
static float4 sampleCatmullRom(texture2d<float> texture, float2 uv) {
    constexpr sampler bilinear(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 size = float2(texture.get_width(), texture.get_height());
    float2 samplePosition = uv * size;
    float2 texel1 = floor(samplePosition - 0.5f) + 0.5f;
    float2 f = samplePosition - texel1;
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);
    float2 w12 = w1 + w2;
    float2 texel0 = (texel1 - 1.0f) / size;
    float2 texel3 = (texel1 + 2.0f) / size;
    float2 texel12 = (texel1 + w2 / w12) / size;

    float4 result = texture.sample(bilinear, float2(texel0.x, texel0.y)) * (w0.x * w0.y);
    result += texture.sample(bilinear, float2(texel12.x, texel0.y)) * (w12.x * w0.y);
    result += texture.sample(bilinear, float2(texel3.x, texel0.y)) * (w3.x * w0.y);
    result += texture.sample(bilinear, float2(texel0.x, texel12.y)) * (w0.x * w12.y);
    result += texture.sample(bilinear, float2(texel12.x, texel12.y)) * (w12.x * w12.y);
    result += texture.sample(bilinear, float2(texel3.x, texel12.y)) * (w3.x * w12.y);
    result += texture.sample(bilinear, float2(texel0.x, texel3.y)) * (w0.x * w3.y);
    result += texture.sample(bilinear, float2(texel12.x, texel3.y)) * (w12.x * w3.y);
    result += texture.sample(bilinear, float2(texel3.x, texel3.y)) * (w3.x * w3.y);
    return result;
}

static float4 taaBlend(
    float4 previous, float4 center, float3 low, float3 high, float alphaLow, float alphaHigh,
    constant TAAOptions& o
) {
    if ((o.flags & taaClipHistory) != 0) {
        previous.rgb = max(yCoCgToRGB(clipToBox(low, high, rgbToYCoCg(previous.rgb))), float3(0));
        previous.a = clamp(previous.a, alphaLow, alphaHigh);
    }
    float4 resolved = mix(previous, center, o.currentWeight);
    return select(resolved, center, isnan(resolved));
}

fragment TAAOutput frostfield_taa_fragment(
    float4 position [[position]],
    texture2d<float, access::read> current [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float> history [[texture(2)]],
    constant TAAOptions& o [[buffer(0)]]
) {
    int2 size = int2(current.get_width(), current.get_height());
    int2 pixel = int2(position.xy);
    float4 center = current.read(uint2(pixel));
    TAAOutput output;
    output.history = center;
    output.resolved = center;
    if ((o.flags & taaHistoryValid) == 0) { return output; }

    // The current 3×3 neighborhood's YCoCg and alpha bounds, and its nearest surface.
    bool reversed = (o.flags & taaReversedDepth) != 0;
    float3 low = rgbToYCoCg(center.rgb);
    float3 high = low;
    float alphaLow = center.a;
    float alphaHigh = center.a;
    float nearestDepth = depth.read(uint2(pixel));
    int2 nearestPixel = pixel;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) { continue; }
            int2 neighbor = clamp(pixel + int2(x, y), int2(0), size - 1);
            float4 sample = current.read(uint2(neighbor));
            float3 ycocg = rgbToYCoCg(sample.rgb);
            low = min(low, ycocg);
            high = max(high, ycocg);
            alphaLow = min(alphaLow, sample.a);
            alphaHigh = max(alphaHigh, sample.a);
            float d = depth.read(uint2(neighbor));
            if (reversed ? d > nearestDepth : d < nearestDepth) {
                nearestDepth = d;
                nearestPixel = neighbor;
            }
        }
    }

    // Reproject the nearest surface with the unjittered current and previous
    // views, so a still camera reads history at this exact texel. Homogeneous
    // coordinates keep points at infinity (reversed infinite depth 0) valid.
    float2 nearestUV = (float2(nearestPixel) + 0.5f) / float2(size);
    float4 clip = float4(nearestUV.x * 2.0f - 1.0f, 1.0f - nearestUV.y * 2.0f, nearestDepth, 1.0f);
    float4 world = o.worldFromJitteredClip * clip;
    float4 now = o.currentClipFromWorld * world;
    float4 before = o.previousClipFromWorld * world;
    if (!(now.w > 0.0f) || !(before.w > 0.0f)) { return output; }
    float2 motion = (now.xy / now.w - before.xy / before.w) * float2(0.5f, -0.5f);
    float2 historyUV = position.xy / float2(size) - motion;
    if (any(!isfinite(historyUV)) || any(historyUV < 0.0f) || any(historyUV > 1.0f)) { return output; }

    float4 previous;
    if ((o.flags & taaExactHistory) != 0) {
        int2 texel = clamp(int2(floor(historyUV * float2(size))), int2(0), size - 1);
        previous = max(history.read(uint2(texel)), float4(0));
    } else {
        previous = max(sampleCatmullRom(history, historyUV), float4(0));
    }
    output.history = taaBlend(previous, center, low, high, alphaLow, alphaHigh, o);
    output.resolved = output.history;
    return output;
}

struct TAAMotionOutput {
    float4 history [[color(0)]];
    float4 resolved [[color(1)]];
    float storedDepth [[color(2)]];
};

static bool taaHasSurface(float depth, bool reversed) {
    return reversed ? depth > 0.0f : depth < 1.0f;
}

static bool taaDepthMatches(float stored, float expected, bool reversed, float tolerance) {
    return taaHasSurface(stored, reversed) && abs(stored - expected) <= tolerance;
}

static inline float taaFootprintWeight(float2 fraction, int x, int y) {
    return (x == 0 ? 1.0f - fraction.x : fraction.x) * (y == 0 ? 1.0f - fraction.y : fraction.y);
}

static bool taaAtDepthEdge(depth2d<float, access::read> depth, uint2 pixel, int2 size, float limit) {
    float nearest = 1.0f;
    float farthest = 0.0f;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            uint2 tap = uint2(clamp(int2(pixel) + int2(x, y), int2(0), size - 1));
            float stored = depth.read(tap);
            nearest = min(nearest, stored);
            farthest = max(farthest, stored);
        }
    }
    return farthest - nearest > limit;
}

static bool taaDepthFootprintMatches(
    texture2d<float, access::read> previousDepth, float2 position, int2 size,
    float expected, bool reversed, float tolerance
) {
    float2 grid = position - 0.5f;
    int2 base = int2(floor(grid));
    float2 fraction = fract(grid);
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            if (taaFootprintWeight(fraction, x, y) == 0.0f) { continue; }
            uint2 tap = uint2(clamp(base + int2(x, y), int2(0), size - 1));
            if (taaDepthMatches(previousDepth.read(tap).r, expected, reversed, tolerance)) { return true; }
        }
    }
    return false;
}

static float taaCompressedLuminance(float3 color) {
    float peak = max(color.r, max(color.g, color.b));
    return dot(color / (1.0f + peak), float3(0.2126f, 0.7152f, 0.0722f));
}

static float4 taaReferenceBlend(
    texture2d<float, access::read> current, uint2 pixel, int2 size, float4 center,
    float4 previous, float2 motionPixels, constant TAAOptions& o
) {
    float factor = saturate(length(motionPixels) / o.maxVelocityLength);
    float currentWeight = 0.05f + factor;
    if ((o.flags & taaSubpixelCorrection) != 0) {
        float2 phase = fract(motionPixels);
        float2 phaseWeight = max(phase, 1.0f - phase);
        currentWeight += (1.0f - phaseWeight.x * phaseWeight.y) * (0.25f / 0.75f);
    }
    currentWeight = saturate(currentWeight);
    if (currentWeight == 1.0f) { return center; }

    float4 firstMoment = 0.0f;
    float4 secondMoment = 0.0f;
    float alphaLow = center.a;
    float alphaHigh = center.a;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            uint2 tap = uint2(clamp(int2(pixel) + int2(x, y), int2(0), size - 1));
            float4 sample = max(current.read(tap), float4(0));
            firstMoment += sample;
            secondMoment += sample * sample;
            alphaLow = min(alphaLow, sample.a);
            alphaHigh = max(alphaHigh, sample.a);
        }
    }

    float4 mean = firstMoment / 9.0f;
    float still = 1.0f - factor;
    float gamma = mix(0.5f, 1.0f, still * still);
    float3 sigma = sqrt(max(secondMoment.rgb / 9.0f - mean.rgb * mean.rgb, float3(0))) * gamma;
    float4 history = max(previous, float4(0));
    float4 offset = history - mean;
    float3 unit = abs(offset.rgb) / (sigma + 1e-7f);
    float largest = max(unit.x, max(unit.y, unit.z));
    if (largest > 1.0f) { history = mean + offset / largest; }

    // Keep clipped coverage in the current neighborhood and transparent texels black.
    history.a = clamp(history.a, alphaLow, alphaHigh);
    history.rgb = history.a > 0.0f ? max(history.rgb, float3(0)) : float3(0);
    float currentLuma = taaCompressedLuminance(max(center.rgb, float3(0)));
    float historyLuma = taaCompressedLuminance(history.rgb);
    float a = currentWeight / (1.0f + currentLuma);
    float b = (1.0f - currentWeight) / (1.0f + historyLuma);
    float4 resolved = (center * a + history * b) / max(a + b, 1e-5f);
    return select(resolved, center, isnan(resolved));
}

fragment TAAMotionOutput frostfield_taa_motion_fragment(
    float4 position [[position]],
    texture2d<float, access::read> current [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> history [[texture(2)]],
    texture2d<float, access::read> motion [[texture(3)]],
    texture2d<float, access::read> previousDepth [[texture(4)]],
    constant TAAOptions& o [[buffer(0)]]
) {
    int2 size = int2(current.get_width(), current.get_height());
    uint2 pixel = uint2(position.xy);
    float4 center = current.read(pixel);
    float storedDepth = depth.read(pixel);
    TAAMotionOutput output = {center, center, storedDepth};
    bool reversed = (o.flags & taaReversedDepth) != 0;
    if ((o.flags & taaHistoryValid) == 0 || !taaHasSurface(storedDepth, reversed)) { return output; }

    float4 vector = motion.read(pixel);
    // Check the float bit patterns because fast math can remove nonfinite tests.
    uint3 exponent = as_type<uint3>(vector.xyz) & uint3(0x7f800000u);
    if (as_type<uint>(vector.w) != 0x3f800000u || any(exponent == 0x7f800000u) ||
        vector.z < 0.0f || vector.z > 1.0f) { return output; }

    float2 rawPrevious = position.xy + vector.xy;
    float2 historyPosition = rawPrevious - o.jitterDelta;
    if (any(rawPrevious < 0.0f) || any(rawPrevious >= float2(size)) ||
        any(historyPosition < 0.0f) || any(historyPosition >= float2(size))) { return output; }
    float projectedDepth = vector.z;
    float tolerance = o.motionDepthThreshold;
    bool reference = (o.flags & taaReferenceResolve) != 0;
    if (!taaDepthMatches(previousDepth.read(uint2(rawPrevious)).r, projectedDepth, reversed, tolerance)) {
        if (!reference || !taaAtDepthEdge(depth, pixel, size, o.edgeDepthDifference) ||
            !taaDepthFootprintMatches(previousDepth, rawPrevious, size, projectedDepth, reversed, tolerance)) {
            return output;
        }
    }

    float4 previous = 0.0f;
    if ((o.flags & taaExactHistory) != 0) {
        uint2 tap = uint2(historyPosition);
        if (!taaDepthMatches(previousDepth.read(tap).r, projectedDepth, reversed, tolerance)) { return output; }
        previous = history.read(tap);
    } else {
        // Depth-gated bilinear color taps keep the filter from pulling in history
        // across an old depth boundary. Normalize the surviving weights.
        float2 colorGrid = historyPosition - 0.5f;
        int2 base = int2(floor(colorGrid));
        float2 fraction = fract(colorGrid);
        float total = 0.0f;
        for (int y = 0; y <= 1; ++y) {
            for (int x = 0; x <= 1; ++x) {
                float weight = taaFootprintWeight(fraction, x, y);
                if (weight == 0.0f) { continue; }
                uint2 tap = uint2(clamp(base + int2(x, y), int2(0), size - 1));
                if (!taaDepthMatches(previousDepth.read(tap).r, projectedDepth, reversed, tolerance)) { continue; }
                previous += history.read(tap) * weight;
                total += weight;
            }
        }
        if (total == 0.0f) { return output; }
        previous /= total;
    }

    if (reference) {
        output.history = taaReferenceBlend(current, pixel, size, center, previous, vector.xy - o.jitterDelta, o);
        output.resolved = output.history;
        return output;
    }

    float3 low = rgbToYCoCg(center.rgb);
    float3 high = low;
    float alphaLow = center.a;
    float alphaHigh = center.a;
    if ((o.flags & taaClipHistory) != 0) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int2 neighbor = clamp(int2(pixel) + int2(x, y), int2(0), size - 1);
                float4 sample = current.read(uint2(neighbor));
                float3 ycocg = rgbToYCoCg(sample.rgb);
                low = min(low, ycocg);
                high = max(high, ycocg);
                alphaLow = min(alphaLow, sample.a);
                alphaHigh = max(alphaHigh, sample.a);
            }
        }
    }

    output.history = taaBlend(max(previous, float4(0)), center, low, high, alphaLow, alphaHigh, o);
    output.resolved = output.history;
    return output;
}
