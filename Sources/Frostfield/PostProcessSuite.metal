#include <metal_stdlib>
using namespace metal;

// Full-screen effects for color lookup, blurs, mip-chain bloom, frame history,
// resampling and supersample accumulation. Every color texture holds
// premultiplied linear RGBA.

constexpr sampler ff_linear_clamp(coord::normalized, address::clamp_to_edge, filter::linear);

vertex float4 frostfield_fx_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertexID], 0, 1);
}

static inline float4 ff_finite(float4 c) {
    return float4(clamp(select(float3(0), c.rgb, isfinite(c.rgb)), float3(0), float3(65504.0f)),
                  isfinite(c.a) ? saturate(c.a) : 0.0f);
}

static inline float3 ff_linear_to_srgb(float3 c) {
    c = max(c, float3(0));
    return select(1.055f * pow(c, float3(1.0f / 2.4f)) - 0.055f, c * 12.92f, c <= 0.0031308f);
}

static inline float3 ff_srgb_to_linear(float3 c) {
    return select(pow((c + 0.055f) / 1.055f, float3(2.4f)), c / 12.92f, c <= 0.04045f);
}

static inline float ff_view_depth(float stored, float4 d) {
    float forward = d.z > 0.5f ? 1.0f - stored : stored;
    return d.x * d.y / (d.y - forward * (d.y - d.x));
}

fragment float4 frostfield_copy_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]]
) {
    return source.read(uint2(position.xy));
}

// MARK: - Color lookup

struct FFLookupParameters {
    float4 domainMinimum;
    float4 domainScale;
    float size;
    float intensity;
    uint interpolation;
    uint encoding;
};

static float3 ff_lookup_tetrahedral(texture3d<float> lut, float3 coordinate, float size) {
    float3 p = saturate(coordinate) * (size - 1.0f);
    float3 base = min(floor(p), float3(size - 2.0f));
    float3 f = p - base;
    uint3 b = uint3(base);
    float3 c000 = lut.read(b).rgb;
    float3 c111 = lut.read(b + uint3(1, 1, 1)).rgb;
    if (f.x >= f.y) {
        if (f.y >= f.z) {
            return (1 - f.x) * c000 + (f.x - f.y) * lut.read(b + uint3(1, 0, 0)).rgb
                + (f.y - f.z) * lut.read(b + uint3(1, 1, 0)).rgb + f.z * c111;
        }
        if (f.x >= f.z) {
            return (1 - f.x) * c000 + (f.x - f.z) * lut.read(b + uint3(1, 0, 0)).rgb
                + (f.z - f.y) * lut.read(b + uint3(1, 0, 1)).rgb + f.y * c111;
        }
        return (1 - f.z) * c000 + (f.z - f.x) * lut.read(b + uint3(0, 0, 1)).rgb
            + (f.x - f.y) * lut.read(b + uint3(1, 0, 1)).rgb + f.y * c111;
    }
    if (f.z >= f.y) {
        return (1 - f.z) * c000 + (f.z - f.y) * lut.read(b + uint3(0, 0, 1)).rgb
            + (f.y - f.x) * lut.read(b + uint3(0, 1, 1)).rgb + f.x * c111;
    }
    if (f.z >= f.x) {
        return (1 - f.y) * c000 + (f.y - f.z) * lut.read(b + uint3(0, 1, 0)).rgb
            + (f.z - f.x) * lut.read(b + uint3(0, 1, 1)).rgb + f.x * c111;
    }
    return (1 - f.y) * c000 + (f.y - f.x) * lut.read(b + uint3(0, 1, 0)).rgb
        + (f.x - f.z) * lut.read(b + uint3(1, 1, 0)).rgb + f.z * c111;
}

fragment float4 frostfield_lut_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture3d<float> lut [[texture(1)]],
    constant FFLookupParameters& p [[buffer(0)]]
) {
    float4 color = source.read(uint2(position.xy));
    if (!(color.a > 0)) { return color; }
    float3 straight = max(color.rgb / color.a, float3(0));
    float3 encoded = p.encoding != 0 ? ff_linear_to_srgb(straight) : straight;
    float3 coordinate = saturate((encoded - p.domainMinimum.xyz) * p.domainScale.xyz);
    float3 graded;
    if (p.interpolation != 0) {
        graded = ff_lookup_tetrahedral(lut, coordinate, p.size);
    } else {
        float3 uvw = (coordinate * (p.size - 1.0f) + 0.5f) / p.size;
        graded = lut.sample(ff_linear_clamp, uvw).rgb;
    }
    if (p.encoding != 0) { graded = ff_srgb_to_linear(graded); }
    float3 result = mix(straight, max(graded, float3(0)), p.intensity);
    return ff_finite(float4(result * color.a, color.a));
}

// MARK: - Gaussian and bilateral blur

struct FFBlurParameters {
    int radius;
    int axis;
    float depthSigma;
    float colorSigma;
    float4 depth;
};

static float4 ff_blur(
    texture2d<float, access::read> source, depth2d<float, access::read> depth, bool useDepth, int2 pixel,
    constant FFBlurParameters& p, constant float* weights
) {
    int2 last = int2(source.get_width(), source.get_height()) - 1;
    int2 direction = p.axis == 0 ? int2(1, 0) : int2(0, 1);
    float4 center = source.read(uint2(pixel));
    float centerDepth = useDepth ? ff_view_depth(depth.read(uint2(pixel)), p.depth) : 0;
    float4 sum = 0;
    float total = 0;
    for (int i = -p.radius; i <= p.radius; ++i) {
        uint2 tap = uint2(clamp(pixel + direction * i, int2(0), last));
        float4 color = source.read(tap);
        float weight = weights[i + p.radius];
        if (useDepth && p.depthSigma > 0) {
            float difference = (ff_view_depth(depth.read(tap), p.depth) - centerDepth) / (p.depthSigma * centerDepth);
            weight *= exp(-0.5f * difference * difference);
        }
        if (p.colorSigma > 0) {
            float difference = length(color - center) / p.colorSigma;
            weight *= exp(-0.5f * difference * difference);
        }
        sum += color * weight;
        total += weight;
    }
    return total > 0 ? sum / total : center;
}

kernel void frostfield_blur_kernel(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    depth2d<float, access::read> depth [[texture(2)]],
    constant FFBlurParameters& p [[buffer(0)]],
    constant float* weights [[buffer(1)]],
    constant uint& useDepth [[buffer(2)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    destination.write(ff_blur(source, depth, useDepth != 0, int2(pixel), p, weights), pixel);
}

fragment float4 frostfield_blur_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    depth2d<float, access::read> depth [[texture(2)]],
    constant FFBlurParameters& p [[buffer(0)]],
    constant float* weights [[buffer(1)]],
    constant uint& useDepth [[buffer(2)]]
) {
    return ff_finite(ff_blur(source, depth, useDepth != 0, int2(position.xy), p, weights));
}

// MARK: - Kawase blur

struct FFKawaseParameters {
    float4 taps;    // x offset, yz blur-grid texel size, w tilt-shift full-blur distance
    float4 output;  // xy one over the output size
};

static float4 ff_kawase(texture2d<float> source, float2 uv, float2 o) {
    return 0.25f * (source.sample(ff_linear_clamp, uv + float2(-o.x, -o.y), level(0))
        + source.sample(ff_linear_clamp, uv + float2(o.x, -o.y), level(0))
        + source.sample(ff_linear_clamp, uv + float2(-o.x, o.y), level(0))
        + source.sample(ff_linear_clamp, uv + o, level(0)));
}

kernel void frostfield_kawase_kernel(
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant FFKawaseParameters& p [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    destination.write(ff_kawase(source, (float2(pixel) + 0.5f) * p.output.xy, (p.taps.x + 0.5f) * p.taps.yz), pixel);
}

fragment float4 frostfield_kawase_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    constant FFKawaseParameters& p [[buffer(0)]]
) {
    return ff_finite(ff_kawase(source, position.xy * p.output.xy, (p.taps.x + 0.5f) * p.taps.yz));
}

// Band xy is the screen normal with its X component scaled by the frame
// aspect ratio. Band z is the signed offset and w is the sharp half-width.
static float ff_tilt_shift_amount(float2 uv, float4 band, float fullBlur) {
    float distance = abs(dot(band.xy, uv * 2.0f - 1.0f) - band.z);
    if (distance <= band.w) { return 0.0f; }
    return fullBlur > band.w ? smoothstep(band.w, fullBlur, distance) : 1.0f;
}

kernel void frostfield_tilt_shift_kernel(
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant FFKawaseParameters& p [[buffer(0)]],
    constant float4& band [[buffer(1)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    float2 uv = (float2(pixel) + 0.5f) * p.output.xy;
    float amount = ff_tilt_shift_amount(uv, band, p.taps.w);
    destination.write(ff_kawase(source, uv, (p.taps.x + 0.5f) * p.taps.yz * amount), pixel);
}

fragment float4 frostfield_tilt_shift_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::read> original [[texture(1)]],
    constant FFKawaseParameters& p [[buffer(0)]],
    constant float4& band [[buffer(1)]]
) {
    float2 uv = position.xy * p.output.xy;
    float amount = ff_tilt_shift_amount(uv, band, p.taps.w);
    if (amount == 0.0f) { return ff_finite(original.read(uint2(position.xy))); }
    return ff_finite(ff_kawase(source, uv, (p.taps.x + 0.5f) * p.taps.yz * amount));
}

// MARK: - Mip-chain bloom

struct FFBloomPrefilter {
    float threshold;
    float knee;
    uint useThreshold;
    uint karis;
};

static inline float ff_luma(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }

static float3 ff_bloom_threshold(float3 c, constant FFBloomPrefilter& p) {
    c = max(c, float3(0));
    if (p.useThreshold == 0) { return c; }
    float brightness = max(c.r, max(c.g, c.b));
    float soft = clamp(brightness - p.threshold + p.knee, 0.0f, 2.0f * p.knee);
    soft = soft * soft / (4.0f * p.knee + 1e-5f);
    // A pixel never scatters more light than it has, even where a knee wider than the threshold overshoots.
    float contribution = min(max(soft, brightness - p.threshold) / max(brightness, 1e-5f), 1.0f);
    return c * contribution;
}

// A bilinear tap of the prefiltered frame. With a threshold, each texel is thresholded
// before the bilinear weights are applied, so the bloom chain stays a linear filter of
// the thresholded frame and the composite can remove exactly the light it scatters.
static float3 ff_tap(texture2d<float> source, float2 uv, constant FFBloomPrefilter& p) {
    if (p.useThreshold == 0) { return ff_bloom_threshold(source.sample(ff_linear_clamp, uv, level(0)).rgb, p); }
    int2 last = int2(source.get_width(), source.get_height()) - 1;
    float2 t = uv * float2(last + 1) - 0.5f;
    float2 base = floor(t);
    float2 f = t - base;
    int2 b = int2(base);
    float3 c00 = ff_bloom_threshold(source.read(uint2(clamp(b, int2(0), last))).rgb, p);
    float3 c10 = ff_bloom_threshold(source.read(uint2(clamp(b + int2(1, 0), int2(0), last))).rgb, p);
    float3 c01 = ff_bloom_threshold(source.read(uint2(clamp(b + int2(0, 1), int2(0), last))).rgb, p);
    float3 c11 = ff_bloom_threshold(source.read(uint2(clamp(b + int2(1, 1), int2(0), last))).rgb, p);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014):
// thirteen bilinear taps form five overlapping 2 × 2 boxes weighted 0.5 and 4 × 0.125.
static float3 ff_downsample13(texture2d<float> source, float2 uv, float2 t, constant FFBloomPrefilter& p) {
    float3 a = ff_tap(source, uv + t * float2(-2, -2), p);
    float3 b = ff_tap(source, uv + t * float2(0, -2), p);
    float3 c = ff_tap(source, uv + t * float2(2, -2), p);
    float3 d = ff_tap(source, uv + t * float2(-2, 0), p);
    float3 e = ff_tap(source, uv, p);
    float3 f = ff_tap(source, uv + t * float2(2, 0), p);
    float3 g = ff_tap(source, uv + t * float2(-2, 2), p);
    float3 h = ff_tap(source, uv + t * float2(0, 2), p);
    float3 i = ff_tap(source, uv + t * float2(2, 2), p);
    float3 j = ff_tap(source, uv + t * float2(-1, -1), p);
    float3 k = ff_tap(source, uv + t * float2(1, -1), p);
    float3 l = ff_tap(source, uv + t * float2(-1, 1), p);
    float3 m = ff_tap(source, uv + t * float2(1, 1), p);
    float3 boxes[5] = {(j + k + l + m) * 0.25f, (a + b + d + e) * 0.25f, (b + c + e + f) * 0.25f,
                       (d + e + g + h) * 0.25f, (e + f + h + i) * 0.25f};
    float weights[5] = {0.5f, 0.125f, 0.125f, 0.125f, 0.125f};
    float3 sum = 0;
    float total = 0;
    for (int n = 0; n < 5; ++n) {
        // Karis average: weighting each box by 1 / (1 + luma) suppresses single-pixel fireflies.
        float w = weights[n] * (p.karis != 0 ? 1.0f / (1.0f + ff_luma(boxes[n])) : 1.0f);
        sum += boxes[n] * w;
        total += w;
    }
    return sum / total;
}

static float3 ff_tent(texture2d<float> source, float2 uv, float2 t) {
    float3 sum = source.sample(ff_linear_clamp, uv, level(0)).rgb * 4.0f;
    sum += (source.sample(ff_linear_clamp, uv + float2(-t.x, 0), level(0)).rgb
        + source.sample(ff_linear_clamp, uv + float2(t.x, 0), level(0)).rgb
        + source.sample(ff_linear_clamp, uv + float2(0, -t.y), level(0)).rgb
        + source.sample(ff_linear_clamp, uv + float2(0, t.y), level(0)).rgb) * 2.0f;
    sum += source.sample(ff_linear_clamp, uv + float2(-t.x, -t.y), level(0)).rgb
        + source.sample(ff_linear_clamp, uv + float2(t.x, -t.y), level(0)).rgb
        + source.sample(ff_linear_clamp, uv + float2(-t.x, t.y), level(0)).rgb
        + source.sample(ff_linear_clamp, uv + float2(t.x, t.y), level(0)).rgb;
    return sum / 16.0f;
}

kernel void frostfield_bloom_downsample(
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant FFBloomPrefilter& p [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    float2 uv = (float2(pixel) + 0.5f) / float2(destination.get_width(), destination.get_height());
    float2 texel = 1.0f / float2(source.get_width(), source.get_height());
    destination.write(float4(ff_downsample13(source, uv, texel, p), 0), pixel);
}

kernel void frostfield_bloom_upsample(
    texture2d<float> lower [[texture(0)]],
    texture2d<float, access::read> current [[texture(1)]],
    texture2d<float, access::write> destination [[texture(2)]],
    constant float& radius [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    float2 uv = (float2(pixel) + 0.5f) / float2(destination.get_width(), destination.get_height());
    float3 up = ff_tent(lower, uv, 1.0f / float2(lower.get_width(), lower.get_height()));
    destination.write(float4(mix(current.read(pixel).rgb, up, radius), 0), pixel);
}

struct FFBloomComposite {
    float strength;
    uint additive;
    float2 inverseSize;
    FFBloomPrefilter prefilter;
};

// Energy-conserving blending removes the light the prefilter sent into the chain and
// adds the scattered result: `scene + strength · (bloom − prefiltered(scene))`.
fragment float4 frostfield_bloom_composite(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float> bloom [[texture(1)]],
    constant FFBloomComposite& p [[buffer(0)]]
) {
    float4 color = source.read(uint2(position.xy));
    float3 glow = ff_tent(bloom, position.xy * p.inverseSize, 1.0f / float2(bloom.get_width(), bloom.get_height()));
    float3 rgb = p.additive != 0
        ? color.rgb + glow * p.strength
        : color.rgb + p.strength * (glow - ff_bloom_threshold(color.rgb, p.prefilter));
    return ff_finite(float4(rgb, color.a));
}

// The independent bloom field used by graph consumers such as lens dirt.
fragment float4 frostfield_bloom_glow(
    float4 position [[position]],
    texture2d<float> bloom [[texture(0)]],
    constant float4& parameters [[buffer(0)]]
) {
    float3 glow = ff_tent(bloom, position.xy * parameters.zw, 1.0f / float2(bloom.get_width(), bloom.get_height()));
    return ff_finite(float4(glow * parameters.x, 1.0f));
}

// MARK: - Frame history

struct FFTwoTargets {
    float4 color [[color(0)]];
    float4 history [[color(1)]];
};

// x: damp or persistence, y: 0 Three.js maximum, 1 linear blend, z: 1 when history is valid.
fragment FFTwoTargets frostfield_afterimage_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::read> previous [[texture(1)]],
    constant float4& p [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float4 now = source.read(pixel);
    float4 result = now;
    if (p.z > 0.5f) {
        float4 old = previous.read(pixel);
        if (p.y < 0.5f) {
            // AfterimageShader: texelOld *= damp * when_gt(texelOld, 0.1); max(texelNew, texelOld).
            old *= p.x * float4(old > 0.1f);
            result = max(now, old);
        } else {
            result = mix(now, old, p.x);
        }
    }
    result = ff_finite(result);
    return {result, result};
}

// MARK: - Resampling and accumulation

// The average of the source texels under one destination texel's footprint, each
// weighted by its overlap area. Footprints tile the source, so the mean is preserved
// exactly at any ratio.
static float4 ff_area_average(texture2d<float> source, float2 pixel, float2 destinationSize) {
    float2 sourceSize = float2(source.get_width(), source.get_height());
    float2 scale = sourceSize / destinationSize;
    float2 low = pixel * scale;
    float2 high = min(low + scale, sourceSize);
    int2 first = int2(floor(low));
    int2 last = min(int2(ceil(high)) - 1, int2(sourceSize) - 1);
    float4 sum = 0;
    float total = 0;
    for (int y = first.y; y <= last.y; ++y) {
        float wy = min(high.y, float(y + 1)) - max(low.y, float(y));
        for (int x = first.x; x <= last.x; ++x) {
            float w = (min(high.x, float(x + 1)) - max(low.x, float(x))) * wy;
            sum += source.read(uint2(x, y)) * w;
            total += w;
        }
    }
    return total > 0 ? sum / total : 0;
}

// x, y: the output size; z: 1 to average each output texel's source footprint, else one bilinear tap.
fragment float4 frostfield_resample_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    constant float4& p [[buffer(0)]]
) {
    if (p.z > 0.5f) { return ff_area_average(source, floor(position.xy), p.xy); }
    return source.sample(ff_linear_clamp, position.xy / p.xy, level(0));
}

kernel void frostfield_area_downsample(
    texture2d<float> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    destination.write(ff_area_average(source, float2(pixel), float2(destination.get_width(), destination.get_height())), pixel);
}

// x: scale applied to the result; y: 1 when `previous` holds a running sum to add.
fragment float4 frostfield_accumulate_fragment(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::read> previous [[texture(1)]],
    constant float2& p [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float4 sum = source.read(pixel);
    if (p.y > 0.5f) { sum += previous.read(pixel); }
    return sum * p.x;
}

// MARK: - Scene transition

fragment float4 frostfield_transition_fade(
    float4 position [[position]],
    texture2d<float, access::read> from [[texture(0)]],
    texture2d<float, access::read> to [[texture(1)]],
    constant float4& settings [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float progress = settings.x;
    if (progress == 0.0f) return from.read(pixel);
    if (progress == 1.0f) return to.read(pixel);
    return mix(from.read(pixel), to.read(pixel), progress);
}

fragment float4 frostfield_transition_masked(
    float4 position [[position]],
    texture2d<float, access::read> from [[texture(0)]],
    texture2d<float, access::read> to [[texture(1)]],
    texture2d<float> mask [[texture(2)]],
    constant float4& settings [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float progress = settings.x;
    if (progress == 0.0f) return from.read(pixel);
    if (progress == 1.0f) return to.read(pixel);

    float2 uv = position.xy / float2(from.get_width(), from.get_height());
    float red = saturate(mask.sample(ff_linear_clamp, uv, level(0)).r);
    if (settings.z > 0.0f) red = 1.0f - red;
    float threshold = settings.y;
    // Algebraically the r187 feathered ramp with A-to-B progress.
    float weight = threshold > 0.0f
        ? saturate((progress * (1.0f + 2.0f * threshold) - red) / threshold)
        : step(red, progress);
    return mix(from.read(pixel), to.read(pixel), weight);
}

// MARK: - Radial blur

struct FFRadialBlurParameters {
    float2 center;
    float radius;
    float weight;
    float decay;
    float exposure;
    uint sampleCount;
};

fragment float4 frostfield_radial_blur_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    constant FFRadialBlurParameters& settings [[buffer(0)]]
) {
    float4 base = source.read(uint2(position.xy));
    if (settings.radius == 0 || settings.exposure == 0 || settings.weight == 0) { return base; }

    float2 uv = position.xy / float2(source.get_width(), source.get_height());
    float2 step = (settings.center - uv) * (settings.radius / float(settings.sampleCount));
    float noise = fract(52.9829189f * fract(dot(position.xy, float2(0.06711056f, 0.00583715f))));
    uv += step * noise;

    float weight = settings.weight;
    float4 sum = 0;
    for (uint i = 0; i < settings.sampleCount; ++i) {
        uv += step;
        sum += source.sample(ff_linear_clamp, uv, level(0)) * weight;
        weight *= settings.decay;
    }

    float4 result = base + sum * (0.5f * settings.exposure / float(settings.sampleCount));
    return float4(clamp(result.rgb, float3(-65504.0f), float3(65504.0f)), saturate(result.a));
}

// MARK: - Halftone

struct FFHalftoneParameters {
    float4 rotationsAndRadius;
    float2 scatterAndBlend;
    uint shape;
    uint blendMode;
    uint grayscale;
    uint enabled;
};

static float ff_halftone_phase(float2 pixel) {
    float2 cell = floor(pixel);
    float seed = fma(cell.y, 78.233f, cell.x * 12.9898f);
    return fract(precise::sin(seed) * 43758.5453f) * 6.283185307f;
}

static float4 ff_halftone_tap(texture2d<float> source, float2 point, float2 size) {
    // HalftoneShader measures its grid and rotations from the bottom left.
    float2 uv = float2(point.x / size.x, 1.0f - point.y / size.y);
    return source.sample(ff_linear_clamp, uv, level(0));
}

static float ff_halftone_density(
    texture2d<float> source, float2 center, float2 size, float radius, uint channel
) {
    float phase = ff_halftone_phase(center);
    float4 sum = ff_halftone_tap(source, center, size);
    for (uint i = 0; i < 8; ++i) {
        float angle = phase + float(i) * 0.785398163f;
        float2 tap = center + float2(cos(angle), sin(angle)) * (radius * 0.66f);
        sum += ff_halftone_tap(source, tap, size);
    }
    return sum.a > 0 ? saturate(sum[channel] / sum.a) : 0.0f;
}

static float ff_halftone_edge(float2 delta, float2 normal, float angle, float density, float radius, uint shape) {
    float distance = length(delta);
    float extent;
    if (shape == 3) {
        distance = abs(dot(delta, normal));
        extent = pow(density, 1.5f) * radius;
    } else if (shape == 4 || shape == 5) {
        float tilt = angle + (shape == 5 ? 0.785398163f : 0.0f);
        float azimuth = distance > 0 ? atan2(delta.y, delta.x) - tilt : 0.0f;
        float alignment = max(abs(cos(azimuth)), abs(sin(azimuth)));
        extent = pow(density, 1.4f) * radius * (2.0f - alignment);
    } else {
        if (shape == 2) {
            distance = distance * 0.79289322f + abs(dot(delta, normal)) * 0.41421356f;
        }
        extent = pow(density, 1.125f) * radius;
    }
    return saturate((extent - distance) / min(radius * 0.5f, 1.25f));
}

static float ff_halftone_channel(
    texture2d<float> source, float2 pixel, float2 size, uint channel, constant FFHalftoneParameters& settings
) {
    float angle = settings.rotationsAndRadius[channel];
    float radius = settings.rotationsAndRadius.w;
    float2 normal = float2(cos(angle), sin(angle));
    float2 tangent = float2(-normal.y, normal.x);
    float2 grid = float2(dot(pixel, normal), dot(pixel, tangent)) / radius;
    float2 nearest = round(grid);
    float2 residual = grid - nearest;
    // Select the opposite corners, using the r187 outward tie rule on grid lines.
    float2 direction = select(residual, grid, residual == 0.0f);
    float2 neighbor = select(float2(1), float2(-1), direction < 0.0f);
    float2 corner = (normal * nearest.x + tangent * nearest.y) * radius;
    float scatter = settings.scatterAndBlend.x;
    if (scatter > 0) {
        float phase = ff_halftone_phase(corner);
        corner += float2(cos(phase), sin(phase)) * (scatter * radius * 0.25f);
    }
    float2 across = normal * (radius * neighbor.x);
    float2 along = tangent * (radius * neighbor.y);
    float2 centers[] = {corner, corner + across, corner + along, corner + across + along};
    float result = 0;
    for (uint i = 0; i < 4; ++i) {
        float density = ff_halftone_density(source, centers[i], size, radius, channel);
        result += ff_halftone_edge(pixel - centers[i], normal, angle, density, radius, settings.shape);
    }
    return saturate(result);
}

fragment float4 frostfield_halftone_fragment(
    float4 position [[position]],
    texture2d<float> source [[texture(0)]],
    constant FFHalftoneParameters& settings [[buffer(0)]]
) {
    float4 original = source.read(uint2(position.xy));
    float blend = settings.scatterAndBlend.y;
    if (settings.enabled == 0 || original.a <= 0 ||
        (settings.blendMode == 1 && blend == 0 && settings.grayscale == 0)) { return original; }

    float2 size = float2(source.get_width(), source.get_height());
    float2 pixel = float2(position.x, size.y - position.y);
    float3 screened;
    for (uint channel = 0; channel < 3; ++channel) {
        screened[channel] = ff_halftone_channel(source, pixel, size, channel, settings);
    }

    float3 straight = original.rgb / original.a;
    float3 bounded = saturate(straight);
    float3 color;
    switch (settings.blendMode) {
        case 2: color = mix(screened, screened * bounded, blend); break;
        case 3: color = mix(screened, saturate(screened + bounded), blend); break;
        case 4: color = mix(screened, max(screened, bounded), blend); break;
        case 5: color = mix(screened, min(screened, bounded), blend); break;
        default: color = mix(bounded, screened, blend); break;
    }
    color += straight - bounded;
    if (settings.grayscale != 0) { color = float3((color.r + color.g + color.b) / 3.0f); }
    return float4(clamp(color * original.a, float3(-65504.0f), float3(65504.0f)), original.a);
}

// MARK: - Depth-aware planar reflection blur

struct FFPlanarBlurParameters {
    float4 filter;    // tap offsets and inverse capture dimensions
    float4 response;  // proximity thresholds, depth scale, reversed depth flag
};

static float ff_planar_proximity(
    depth2d<float, access::read> depth, uint2 pixel, constant FFPlanarBlurParameters& p
) {
    if (p.response.z == 0) { return 0; }
    float stored = depth.read(pixel);
    float proximity = p.response.w > 0 ? stored : 1 - stored;
    return smoothstep(p.response.x, p.response.y, proximity) * p.response.z;
}

static float4 ff_planar_blur(
    texture2d<float> source, float2 pixel, float proximity, constant FFPlanarBlurParameters& p
) {
    float2 uv = pixel * p.filter.zw;
    float contraction = p.response.z == 0 ? 1 : 1 - saturate(proximity + 0.25f);
    float2 tap = p.filter.xy * contraction;
    return (source.sample(ff_linear_clamp, uv + tap, level(0))
            + source.sample(ff_linear_clamp, uv - tap, level(0))
            + source.sample(ff_linear_clamp, uv + float2(tap.x, -tap.y), level(0))
            + source.sample(ff_linear_clamp, uv + float2(-tap.x, tap.y), level(0))) * 0.25f;
}

fragment float4 frostfield_planar_blur_fragment(
    float4 position [[position]], texture2d<float> source [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], constant FFPlanarBlurParameters& p [[buffer(0)]]
) {
    float proximity = ff_planar_proximity(depth, uint2(position.xy), p);
    return ff_planar_blur(source, position.xy, proximity, p);
}

struct FFPlanarBlurOutputs {
    float4 color [[color(0)]];
    float depthFactor [[color(1)]];
};

fragment FFPlanarBlurOutputs frostfield_planar_blur_depth_fragment(
    float4 position [[position]], texture2d<float> source [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], constant FFPlanarBlurParameters& p [[buffer(0)]]
) {
    float proximity = ff_planar_proximity(depth, uint2(position.xy), p);
    float factor = p.response.z == 0 ? 1 : clamp(proximity, 0.0001f, 1.0f);
    return {ff_planar_blur(source, position.xy, proximity, p), factor};
}
