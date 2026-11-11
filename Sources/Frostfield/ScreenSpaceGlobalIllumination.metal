#include <metal_stdlib>
#include "FrostfieldScreenSpace.h"
using namespace metal;

// Screen-space indirect lighting with visibility bitmasks (Therrien et al.).
// In each view slice, depth samples are finite-thickness angular intervals.
// Only the first surface to cover a sector contributes its radiance.
struct FFSSGIView {
    float4 projection; // reciprocal focal scales, projection center
    float4 depth;      // far inverse distance, inverse distance range, near plane, inverse far for linear thickness
    float4 options;    // world radius, thickness, maximum pixel radius, step exponent
    float4 sampling;   // slices, steps per side, normal source, reversed depth
    float4 light;      // intensity, AO power, edge fade, backface share
    float4 viewFromWorld[3];
    float4 trace;      // trace width and height, screen-relative search, linear thickness
    float4 phase;      // temporal slice rotation and step offset
};

constant float3 ssgiLuminance = float3(0.2126f, 0.7152f, 0.0722f);

static bool ssgiSurface(float stored, constant FFSSGIView& view) {
    return view.sampling.w != 0.0f ? stored > 0.0f && stored <= 1.0f : stored >= 0.0f && stored < 1.0f;
}

static float ssgiDistance(float stored, constant FFSSGIView& view) {
    float nearWeight = view.sampling.w != 0.0f ? stored : 1.0f - stored;
    return 1.0f / fma(nearWeight, view.depth.y, view.depth.x);
}

static float3 ssgiPosition(float2 screen, float distance, float2 inverseSize, constant FFSSGIView& view) {
    float2 uv = screen * inverseSize;
    float2 ndc = float2(2.0f * uv.x - 1.0f, 1.0f - 2.0f * uv.y);
    return float3((ndc + view.projection.zw) * view.projection.xy, -1.0f) * distance;
}

static float3 ssgiUnit(float3 vector) {
    float length2 = dot(vector, vector);
    return length2 > 1e-16f && isfinite(length2) ? vector * rsqrt(length2) : float3(0.0f);
}

// One-sided closest-depth derivatives avoid normals spanning an occlusion edge.
static float3 ssgiDerivative(
    int2 pixel, int2 axis, int2 size, float3 center, float2 inverseSize,
    depth2d<float, access::read> depth, constant FFSSGIView& view
) {
    float best = INFINITY;
    float3 result = float3(0.0f);
    for (int side = -1; side <= 1; side += 2) {
        int2 p = pixel + axis * side;
        if (any(p < 0) || any(p >= size)) { continue; }
        float stored = depth.read(uint2(p));
        if (!ssgiSurface(stored, view)) { continue; }
        float distance = ssgiDistance(stored, view);
        float difference = abs(distance + center.z);
        if (difference < best && difference < max(0.015f, -center.z * 0.08f)) {
            best = difference;
            result = (ssgiPosition(float2(p) + 0.5f, distance, inverseSize, view) - center) * float(side);
        }
    }
    return result;
}

static float3 ssgiNormal(
    int2 pixel, int2 size, float3 center, float2 inverseSize, depth2d<float, access::read> depth,
    texture2d<uint, access::read> attributes, texture2d<float, access::read> normals, constant FFSSGIView& view
) {
    if (view.sampling.z == 0.0f) {
        float3 right = ssgiDerivative(pixel, int2(1, 0), size, center, inverseSize, depth, view);
        float3 down = ssgiDerivative(pixel, int2(0, 1), size, center, inverseSize, depth, view);
        return ssgiUnit(cross(down, right));
    }
    float3 normal = view.sampling.z == 1.0f
        ? frostfieldUnpackSurfaceAttributes(attributes.read(uint2(pixel)).xy).normal : normals.read(uint2(pixel)).xyz;
    float3x3 rotation = float3x3(view.viewFromWorld[0].xyz, view.viewFromWorld[1].xyz, view.viewFromWorld[2].xyz);
    return ssgiUnit(rotation * normal);
}

static float2 ssgiAcos(float2 value) {
    value = clamp(value, -1.0f, 1.0f);
    float2 a = abs(value);
    float2 angle = (M_PI_2_F - 0.156583f * a) * sqrt(1.0f - a);
    return select(M_PI_F - angle, angle, value >= 0.0f);
}

static uint ssgiSectorMask(float2 interval) {
    uint start = min(32u, uint(interval.x * 32.0f));
    uint count = min(32u - start, uint(ceil(max(0.0f, interval.y - interval.x) * 32.0f)));
    if (count == 0u) { return 0u; }
    return (count == 32u ? 0xffffffffu : (1u << count) - 1u) << start;
}

// Snap the reduced trace to the closest visible depth around the mapped center.
static int2 ssgiGuidePixel(
    int2 tracePixel, int2 size, depth2d<float, access::read> depth, constant FFSSGIView& view
) {
    if (all(int2(view.trace.xy) == size)) { return tracePixel; }
    float2 center = (float2(tracePixel) + 0.5f) * float2(size) / view.trace.xy;
    int2 corner = int2(floor(center - 0.5f));
    int2 best = min(int2(center), size - 1);
    float nearest = INFINITY;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 p = clamp(corner + int2(x, y), int2(0), size - 1);
            float stored = depth.read(uint2(p));
            if (!ssgiSurface(stored, view)) { continue; }
            float distance = ssgiDistance(stored, view);
            if (distance < nearest) {
                nearest = distance;
                best = p;
            }
        }
    }
    return best;
}

fragment float4 frostfield_ssgi_gather(
    float4 screen [[position]], texture2d<float, access::read> beauty [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> normals [[texture(3)]], constant FFSSGIView& view [[buffer(0)]]
) {
    int2 size = int2(depth.get_width(), depth.get_height());
    int2 tracePixel = int2(screen.xy);
    int2 pixel = ssgiGuidePixel(tracePixel, size, depth, view);
    float2 centerScreen = float2(pixel) + 0.5f;
    float2 inverseSize = 1.0f / float2(size);
    float stored = depth.read(uint2(pixel));
    if (!ssgiSurface(stored, view)) { return float4(0, 0, 0, 1); }
    float distance = ssgiDistance(stored, view);
    if (!(distance > 0.0f && isfinite(distance))) { return float4(0, 0, 0, 1); }
    float3 center = ssgiPosition(centerScreen, distance, inverseSize, view);
    float3 normal = ssgiNormal(pixel, size, center, inverseSize, depth, attributes, normals, view);
    if (!any(normal != 0.0f)) { return float4(0, 0, 0, 1); }
    float3 toEye = ssgiUnit(-center);
    if (!any(toEye != 0.0f)) { return float4(0, 0, 0, 1); }
    float focal = max(0.5f * float(size.x) / view.projection.x, 0.5f * float(size.y) / view.projection.y);
    float worldSearch = focal * view.options.x / max(distance - view.options.x, view.depth.z);
    float screenSearch = view.options.x * float(size.x) / 32.0f;
    float search = clamp(view.trace.z != 0.0f ? screenSearch : worldSearch, 1.0f, view.options.z);
    float noise = fract(52.9829189f * fract(dot(float2(tracePixel), float2(0.06711056f, 0.00583715f))));
    float phase = 0.25f + 0.5f * fract(noise + float((tracePixel.x - tracePixel.y) & 3) * 0.25f + view.phase.y);
    int slices = int(view.sampling.x), steps = int(view.sampling.y);
    float3 indirect = float3(0.0f);
    float occluded = 0.0f;
    for (int slice = 0; slice < slices; ++slice) {
        float theta = (float(slice) + noise + view.phase.x) * (M_PI_F / float(slices));
        float2 axis = float2(cos(theta), sin(theta));
        float3 plane = ssgiUnit(cross(float3(axis, 0.0f), toEye));
        float3 tangent = cross(toEye, plane);
        float3 projectedNormal = ssgiUnit(normal - plane * dot(normal, plane));
        if (!any(projectedNormal != 0.0f)) { continue; }
        float normalAngle = -sign(dot(projectedNormal, tangent)) * acos(clamp(dot(projectedNormal, toEye), -1.0f, 1.0f));
        uint occupied = 0u;
        for (int side = -1; side <= 1; side += 2) {
            int2 previous = pixel;
            for (int step = 0; step < steps; ++step) {
                float t = (float(step) + phase) / float(steps);
                float offset = 1.0f + (search - 1.0f) * pow(t, view.options.w);
                float2 screenSample = centerScreen + float2(axis.x, -axis.y) * (offset * float(side));
                if (any(screenSample < 0.0f) || any(screenSample >= float2(size))) { break; }
                int2 p = int2(screenSample);
                if (all(p == previous)) { continue; }
                previous = p;
                float z = depth.read(uint2(p));
                if (!ssgiSurface(z, view)) { continue; }
                float sampleDistance = ssgiDistance(z, view);
                float3 samplePoint = ssgiPosition(float2(p) + 0.5f, sampleDistance, inverseSize, view);
                float3 delta = samplePoint - center;
                float length = metal::length(delta);
                if (!(length > 1e-5f && length < view.options.x)) { continue; }
                float thicknessScale = view.trace.w != 0.0f ? 100.0f * saturate(sampleDistance * view.depth.w) : 1.0f;
                float3 backDelta = delta - toEye * (view.options.y * thicknessScale);
                // A slab entirely below the receiving hemisphere covers no light.
                float surfaceBias = max(1e-5f, length * 0.001f);
                if (max(dot(normal, delta), dot(normal, backDelta)) <= surfaceBias) { continue; }
                float3 direction = delta / length;
                float3 back = ssgiUnit(backDelta);
                float2 horizon = ssgiAcos(float2(dot(direction, toEye), dot(back, toEye)));
                horizon = saturate((-float(side) * horizon - (normalAngle - M_PI_2_F)) / M_PI_F);
                uint mask = ssgiSectorMask(side > 0 ? horizon.yx : horizon);
                uint visible = mask & ~occupied;
                occupied |= mask;
                if (visible == 0u) { continue; }
                float receiving = saturate(dot(normal, direction));
                float3 color = max(beauty.read(uint2(p)).rgb, 0.0f);
                if (!(receiving > 0.001f && dot(color, ssgiLuminance) > 0.001f)) { continue; }
                float3 lightNormal = ssgiNormal(p, size, samplePoint, inverseSize, depth, attributes, normals, view);
                float emission = dot(lightNormal, -direction);
                float outgoing = emission >= 0.0f ? emission
                    : (dot(lightNormal, toEye) > 0.0f ? -emission * view.light.w : 0.0f);
                float2 uv = screenSample * inverseSize;
                float2 edge = min(uv, 1.0f - uv);
                float fade = view.light.z > 0.0f ? smoothstep(0.0f, view.light.z, min(edge.x, edge.y)) : 1.0f;
                float reach = 1.0f - smoothstep(0.7f * view.options.x, view.options.x, length);
                indirect += color * (float(popcount(visible)) / 32.0f * receiving * outgoing * fade * reach);
            }
        }
        occluded += float(popcount(occupied)) / 32.0f;
    }
    indirect *= view.light.x / float(slices);
    float luminance = dot(indirect, ssgiLuminance);
    if (luminance > 7.0f) { indirect *= 7.0f / luminance; }
    float visibility = view.light.y == 0.0f ? 1.0f : pow(saturate(1.0f - occluded / float(slices)), view.light.y);
    return float4(indirect, visibility);
}

fragment float4 frostfield_ssgi_resolve(
    float4 screen [[position]], texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> normals [[texture(3)]], constant FFSSGIView& view [[buffer(0)]]
) {
    int2 size = int2(depth.get_width(), depth.get_height());
    int2 pixel = int2(screen.xy);
    float stored = depth.read(uint2(pixel));
    if (!ssgiSurface(stored, view)) { return float4(0, 0, 0, 1); }
    float2 inverseSize = 1.0f / float2(size);
    float distance = ssgiDistance(stored, view);
    if (!(distance > 0.0f && isfinite(distance))) { return float4(0, 0, 0, 1); }
    float3 center = ssgiPosition(screen.xy, distance, inverseSize, view);
    float3 normal = ssgiNormal(pixel, size, center, inverseSize, depth, attributes, normals, view);
    int2 traceSize = int2(view.trace.xy);
    if (any(traceSize != size)) {
        float2 traceCenter = screen.xy * view.trace.xy * inverseSize;
        int2 base = int2(traceCenter);
        float4 sum = float4(0.0f);
        float weights = 0.0f;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int2 tap = base + int2(x, y);
                if (any(tap < 0) || any(tap >= traceSize)) { continue; }
                int2 p = ssgiGuidePixel(tap, size, depth, view);
                float z = depth.read(uint2(p));
                if (!ssgiSurface(z, view)) { continue; }
                float sampleDistance = ssgiDistance(z, view);
                if (!(sampleDistance > 0.0f && isfinite(sampleDistance))) { continue; }
                float3 samplePoint = ssgiPosition(float2(p) + 0.5f, sampleDistance, inverseSize, view);
                float3 sampleNormal = ssgiNormal(p, size, samplePoint, inverseSize, depth, attributes, normals, view);
                float alignment = saturate(dot(sampleNormal, normal));
                // Same-plane samples can be farther along the camera axis on a sloped surface.
                float gap = abs(dot(normal, samplePoint - center));
                float continuity = saturate(1.0f - gap / max(0.015f, distance * 0.02f));
                float2 offset = float2(tap) + 0.5f - traceCenter;
                float weight = continuity * pow(alignment, 8.0f) / (1.0f + dot(offset, offset) * 2.0f);
                sum += raw.read(uint2(tap)) * weight;
                weights += weight;
            }
        }
        return weights > 0.001f ? sum / weights : float4(0, 0, 0, 1);
    }
    float4 sum = raw.read(uint2(pixel));
    float weights = 1.0f;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 p = pixel + int2(x, y);
            if ((x == 0 && y == 0) || any(p < 0) || any(p >= size)) { continue; }
            float z = depth.read(uint2(p));
            if (!ssgiSurface(z, view)) { continue; }
            float sampleDistance = ssgiDistance(z, view);
            float relative = abs(sampleDistance - distance) / max(0.015f, distance * 0.025f);
            if (relative >= 1.0f) { continue; }
            float3 samplePoint = ssgiPosition(float2(p) + 0.5f, sampleDistance, inverseSize, view);
            float3 sampleNormal = ssgiNormal(p, size, samplePoint, inverseSize, depth, attributes, normals, view);
            float alignment = saturate(dot(sampleNormal, normal));
            float weight = (1.0f - relative) * pow(alignment, 8.0f) * ((x == 0 || y == 0) ? 0.5f : 0.25f);
            sum += raw.read(uint2(p)) * weight;
            weights += weight;
        }
    }
    return sum / weights;
}

struct FFSSGIReprojection {
    float4x4 previousViewFromCurrent;
    float4x4 previousClipFromCurrent;
    float4 options; // history weight, uses motion, previous far inverse distance and range
};

struct FFSSGIHistoryOutput {
    float4 light [[color(0)]];
    float4 history [[color(1)]];
    uint2 geometry [[color(2)]];
};

fragment FFSSGIHistoryOutput frostfield_ssgi_temporal(
    float4 screen [[position]], texture2d<float, access::read> current [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> normals [[texture(3)]], texture2d<float, access::read> previous [[texture(4)]],
    texture2d<uint, access::read> previousGeometry [[texture(5)]], texture2d<float, access::read> motionTexture [[texture(6)]],
    constant FFSSGIView& view [[buffer(0)]], constant FFSSGIReprojection& history [[buffer(1)]]
) {
    int2 size = int2(depth.get_width(), depth.get_height());
    int2 pixel = int2(screen.xy);
    float stored = depth.read(uint2(pixel));
    float4 clear = float4(0, 0, 0, 1);
    uint2 invalidGeometry = uint2(as_type<uint>(-1.0f), 0u);
    if (!ssgiSurface(stored, view)) { return {clear, clear, invalidGeometry}; }
    float distance = ssgiDistance(stored, view);
    if (!(distance > 0.0f && isfinite(distance))) { return {clear, clear, invalidGeometry}; }
    float2 inverseSize = 1.0f / float2(size);
    float3 center = ssgiPosition(screen.xy, distance, inverseSize, view);
    float3 normal = ssgiNormal(pixel, size, center, inverseSize, depth, attributes, normals, view);
    float4 result = current.read(uint2(pixel));
    if (!any(normal != 0.0f)) { return {result, result, invalidGeometry}; }
    uint2 geometry = uint2(as_type<uint>(stored), pack_float_to_snorm2x16(frostfieldOctahedralEncode(normal)));
    if (history.options.x <= 0.0f) { return {result, result, geometry}; }

    bool usesMotion = history.options.y != 0.0f;
    float expectedDepth = 0.0f;
    float expectedDistance = 0.0f;
    float2 previousPixel;
    if (usesMotion) {
        float4 motion = motionTexture.read(uint2(pixel));
        if (!(all(isfinite(motion)) && motion.w > 0.5f && motion.z >= 0.0f && motion.z <= 1.0f)) {
            return {result, result, geometry};
        }
        // RasterMotionVectors already includes the difference in projection jitter.
        previousPixel = screen.xy - 0.5f + motion.xy;
        if (any(previousPixel < -0.5f) || any(previousPixel >= float2(size) - 0.5f)) {
            return {result, result, geometry};
        }
        expectedDepth = motion.z;
    } else {
        float4 clip = history.previousClipFromCurrent * float4(center, 1.0f);
        if (!(all(isfinite(clip)) && clip.w > 0.0f && clip.z >= 0.0f && clip.z <= clip.w)) { return {result, result, geometry}; }
        float2 ndc = clip.xy / clip.w;
        if (any(abs(ndc) >= 1.0f)) { return {result, result, geometry}; }
        expectedDistance = -(history.previousViewFromCurrent * float4(center, 1.0f)).z;
        if (!(expectedDistance > 0.0f && isfinite(expectedDistance))) { return {result, result, geometry}; }
        previousPixel = (ndc * float2(0.5f, -0.5f) + 0.5f) * float2(size) - 0.5f;
    }
    float3 expectedNormal = (history.previousViewFromCurrent * float4(normal, 0.0f)).xyz;
    int2 corner = int2(floor(previousPixel));
    float2 fraction = fract(previousPixel);
    float4 sum = float4(0.0f);
    float weights = 0.0f;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 p = corner + int2(x, y);
            if (any(p < 0) || any(p >= size)) { continue; }
            uint2 sample = previousGeometry.read(uint2(p)).xy;
            float previousDepth = as_type<float>(sample.x);
            if (!ssgiSurface(previousDepth, view)) { continue; }
            float nearWeight = view.sampling.w != 0.0f ? previousDepth : 1.0f - previousDepth;
            float reciprocal = fma(nearWeight, history.options.w, history.options.z);
            float previousDistance = 1.0f / reciprocal;
            if (!(previousDistance > 0.0f && isfinite(previousDistance))) { continue; }
            if (usesMotion) {
                // Motion stores raw depth in Float16. Compare equal quantized depths directly,
                // with a small view-distance tolerance for a sloped bilinear footprint.
                float depthTolerance = max(0.015f, previousDistance * 0.02f) * reciprocal * reciprocal / history.options.w;
                if (float(half(previousDepth)) != expectedDepth && abs(previousDepth - expectedDepth) > depthTolerance) { continue; }
            } else if (abs(previousDistance - expectedDistance) >= max(0.015f, expectedDistance * 0.02f)) {
                continue;
            }
            float3 previousNormal = frostfieldOctahedralDecode(unpack_snorm2x16_to_float(sample.y));
            if (dot(expectedNormal, previousNormal) <= 0.9f) { continue; }
            float weight = (x == 0 ? 1.0f - fraction.x : fraction.x) * (y == 0 ? 1.0f - fraction.y : fraction.y);
            sum += previous.read(uint2(p)) * weight;
            weights += weight;
        }
    }
    result = result * (1.0f - weights * history.options.x) + sum * history.options.x;
    return {result, result, geometry};
}

fragment float4 frostfield_ssgi_composite(
    float4 screen [[position]], texture2d<float, access::read> beauty [[texture(0)]],
    texture2d<float, access::read> illumination [[texture(1)]], texture2d<float, access::read> albedo [[texture(2)]],
    constant uint& hasAlbedo [[buffer(0)]]
) {
    uint2 pixel = uint2(screen.xy);
    float4 source = beauty.read(pixel);
    float3 reflectance = hasAlbedo != 0u ? saturate(albedo.read(pixel).rgb) : float3(1.0f);
    float3 diffuse = max(illumination.read(pixel).rgb, 0.0f) * reflectance * saturate(source.a);
    return float4(min(source.rgb + diffuse, float3(65504.0f)), source.a);
}
