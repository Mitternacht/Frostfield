#include <metal_stdlib>
#include "FrostfieldScreenSpace.h"
#include "FrostfieldLogDepth.h"
using namespace metal;

// Ground-truth ambient occlusion (Jimenez et al. 2016, "Practical Real-Time Strategies for
// Accurate Indirect Occlusion"), following the structure of Intel's XeGTAO. Camera space is
// right-handed and looks down -Z; screen Y grows downward.

// Matches AmbientOcclusion.viewConstants. A view point (x, y, -d) has
// NDC (x / (d·projection.x) - projection.z, y / (d·projection.y) - projection.w), and 1/d is
// depth.x + depth.y · w, where w runs from 0 at the cleared depth to 1 at the near plane.
// In orthographic mode XY is independent of view distance and depth stores near and span.
struct FrostfieldOcclusionView {
    float4 projection;  // reciprocal projection scales in x and y, then the projection's x and y centers
    float4 depth;       // perspective inverse distance and range, orthographic near/span, or log2 near/inverse log2 range; near and focal ratio in zw
    float4 options;     // world radius, vertical focal length in pixels, 1 for reversed depth, falloff fraction
    float4 sampling;    // slices, steps per side, 1 with surface attributes, largest radius in pixels
    float4 viewFromWorld[3];  // rotation columns; w holds strength, orthographic flag, and far for logarithmic depth
    float4 controls;    // view-axis thickness, normal bias, visibility exponent, fixed pixel radius
    float4 phase;       // temporal slice rotation, step offset, trace width and height
    float4 clipMinimum; // world-space lower corner; w enables clipping
    float4 clipMaximum; // world-space upper corner; w is the sample-distance exponent
    float4 cameraOrigin;
};

constant bool aoOrthographic [[function_constant(0)]];
constant bool aoTemporalMotion [[function_constant(1)]];
constant bool aoLogarithmic [[function_constant(2)]];
constant bool aoTemporalClip [[function_constant(3)]];
// Resolve shaders specialize clipping and gather shaders specialize tap spacing.
constant bool aoSpatialExtraControl [[function_constant(4)]];

vertex float4 frostfield_ao_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertexID], 0, 1);
}

static bool aoHasSurface(float depth, bool reversed) {
    return reversed ? depth > 0.0f : depth < 1.0f;
}

static inline bool aoLogView(constant FrostfieldOcclusionView& view) {
    return aoLogarithmic && view.viewFromWorld[2].w > 0.0f;
}

// Distance along the view axis.
static inline float aoViewDepth(float depth, constant FrostfieldOcclusionView& view, bool orthographic) {
    if (orthographic) {
        float farWeight = view.options.z != 0.0f ? 1.0f - depth : depth;
        return fma(farWeight, view.depth.y, view.depth.x);
    }
    if (aoLogView(view)) {
        return frostfieldDistanceFromLogarithmicDepth(depth, view.depth.x, view.depth.y, view.viewFromWorld[2].w);
    }
    float nearWeight = view.options.z != 0.0f ? depth : 1.0f - depth;
    return 1.0f / fma(view.depth.y, nearWeight, view.depth.x);
}

static inline float3 aoViewPositionAtNDC(float2 ndc, float viewDepth, constant FrostfieldOcclusionView& view, bool orthographic) {
    if (orthographic) { return float3((ndc + view.projection.zw) * view.projection.xy, -viewDepth); }
    return float3((ndc + view.projection.zw) * view.projection.xy, -1.0f) * viewDepth;
}

static inline float3 aoViewPosition(
    float2 pixel, float viewDepth, float2 inverseSize, constant FrostfieldOcclusionView& view, bool orthographic
) {
    float2 uv = (pixel + 0.5f) * inverseSize;
    return aoViewPositionAtNDC(float2(2.0f * uv.x - 1.0f, 1.0f - 2.0f * uv.y), viewDepth, view, orthographic);
}

// A surface point on the slice line at continuous screen position `screen` (pixel units,
// texel centers at +0.5). Metal depth is affine in screen space across a plane, so the
// bilinear blend of a continuous 2×2 neighborhood places the point exactly on planar
// surfaces and exactly on the slice. Across a depth jump the nearest texel is used.
static inline bool aoSample(
    float2 screen, int2 size, depth2d<float> depth, float2 inverseSize,
    constant FrostfieldOcclusionView& view, bool orthographic, thread float3& position
) {
    constexpr sampler corners(coord::normalized, address::clamp_to_edge, filter::nearest);
    bool reversed = view.options.z != 0.0f;
    float2 ndc = float2(2.0f * screen.x * inverseSize.x - 1.0f, 1.0f - 2.0f * screen.y * inverseSize.y);
    float2 texel = screen - 0.5f;
    float2 base = floor(texel);
    float2 fraction = texel - base;
    // Gather at the shared corner of the 2×2 block so the footprint is unambiguous:
    // w is the top-left texel, z top-right, x bottom-left and y bottom-right.
    float4 quad = depth.gather(corners, (base + 1.0f) * inverseSize);
    float d00 = quad.w;
    float d10 = quad.z;
    float d01 = quad.x;
    float d11 = quad.y;
    if (aoHasSurface(d00, reversed) && aoHasSurface(d10, reversed)
        && aoHasSurface(d01, reversed) && aoHasSurface(d11, reversed)) {
        // View depth falls as Metal depth rises (or the reverse), so the extremes bound it.
        float nearest = aoViewDepth(reversed ? max(max(d00, d10), max(d01, d11)) : min(min(d00, d10), min(d01, d11)), view, orthographic);
        float farthest = aoViewDepth(reversed ? min(min(d00, d10), min(d01, d11)) : max(max(d00, d10), max(d01, d11)), view, orthographic);
        float tolerance = orthographic ? 2.0f / min(view.options.y, view.options.y * view.depth.w) : 0.1f * nearest;
        if (farthest - nearest <= tolerance) {
            if (aoLogView(view)) {
                // Reciprocal view distance is affine on a projected plane. Stored log depth is not.
                float top = mix(1.0f / aoViewDepth(d00, view, false), 1.0f / aoViewDepth(d10, view, false), fraction.x);
                float bottom = mix(1.0f / aoViewDepth(d01, view, false), 1.0f / aoViewDepth(d11, view, false), fraction.x);
                position = aoViewPositionAtNDC(ndc, 1.0f / mix(top, bottom, fraction.y), view, false);
            } else {
                float blended = mix(mix(d00, d10, fraction.x), mix(d01, d11, fraction.x), fraction.y);
                position = aoViewPositionAtNDC(ndc, aoViewDepth(blended, view, orthographic), view, orthographic);
            }
            return true;
        }
    }
    // The covering texel's depth, kept on the slice at the exact screen position.
    int2 pixel = clamp(int2(floor(screen)), int2(0), size - 1);
    float sampled = depth.read(uint2(pixel));
    if (!aoHasSurface(sampled, reversed)) {
        return false;
    }
    position = aoViewPositionAtNDC(ndc, aoViewDepth(sampled, view, orthographic), view, orthographic);
    return true;
}

static float3x3 aoViewFromWorld(constant FrostfieldOcclusionView& view) {
    return float3x3(view.viewFromWorld[0].xyz, view.viewFromWorld[1].xyz, view.viewFromWorld[2].xyz);
}

static float aoClipWeight(
    int2 pixel, float storedDepth, float2 inverseSize, constant FrostfieldOcclusionView& view, bool orthographic
) {
    if (view.clipMinimum.w == 0.0f) { return 1.0f; }
    float distance = aoViewDepth(storedDepth, view, orthographic);
    float3 center = aoViewPosition(float2(pixel), distance, inverseSize, view, orthographic);
    float3 point = transpose(aoViewFromWorld(view)) * center + view.cameraOrigin.xyz;
    if (!all(isfinite(point))) { return 0.0f; }
    float3 outside = max(float3(0), max(view.clipMinimum.xyz - point, point - view.clipMaximum.xyz));
    if (all(outside == 0.0f)) { return 1.0f; }
    float radius = view.options.x;
    if (view.controls.w > 0.0f) {
        // Convert the fixed pixel radius through the vertical focal scale at this depth.
        radius = min(view.controls.w, view.sampling.w) / view.options.y * (orthographic ? 1.0f : distance);
    }
    if (!(radius > 0.0f)) { return 0.0f; }
    return 1.0f - smoothstep(0.0f, 1.0f, length(outside / radius));
}

static float aoResolvedStrength(
    int2 pixel, depth2d<float> depth, constant FrostfieldOcclusionView& view, bool orthographic
) {
    float strength = view.viewFromWorld[0].w;
    if (strength == 0.0f) { return 0.0f; }
    float stored = depth.read(uint2(pixel));
    if (!aoHasSurface(stored, view.options.z != 0.0f)) { return 0.0f; }
    if (!aoSpatialExtraControl) { return strength; }
    return strength * aoClipWeight(pixel, stored, 1.0f / float2(depth.get_width(), depth.get_height()), view, orthographic);
}

// Choose the nearer depth derivative on each axis so normals do not span silhouettes.
static bool aoUseFirst(float first, float second, float center, bool reversed, constant FrostfieldOcclusionView& view) {
    if (aoLogView(view)) {
        if (!aoHasSurface(first, reversed)) { return false; }
        if (!aoHasSurface(second, reversed)) { return true; }
        // Choose the same side as the perspective depth derivative.
        float centerInverse = 1.0f / aoViewDepth(center, view, false);
        return abs(1.0f / aoViewDepth(first, view, false) - centerInverse)
            < abs(1.0f / aoViewDepth(second, view, false) - centerInverse);
    }
    return aoHasSurface(first, reversed)
        && (!aoHasSurface(second, reversed) || abs(first - center) < abs(second - center));
}

// The view-space normal from depth alone, or zero where neighbors cannot define one.
// Neighbors past the screen edge count as empty, so border pixels use one-sided differences.
static inline float3 aoReconstructedNormal(
    int2 pixel, float centerDepth, int2 size, depth2d<float> depth,
    float2 inverseSize, constant FrostfieldOcclusionView& view, bool orthographic
) {
    bool reversed = view.options.z != 0.0f;
    float empty = reversed ? 0.0f : 1.0f;
    float west = pixel.x > 0 ? depth.read(uint2(pixel + int2(-1, 0))) : empty;
    float east = pixel.x < size.x - 1 ? depth.read(uint2(pixel + int2(1, 0))) : empty;
    float north = pixel.y > 0 ? depth.read(uint2(pixel + int2(0, -1))) : empty;
    float south = pixel.y < size.y - 1 ? depth.read(uint2(pixel + int2(0, 1))) : empty;
    if ((!aoHasSurface(west, reversed) && !aoHasSurface(east, reversed))
        || (!aoHasSurface(north, reversed) && !aoHasSurface(south, reversed))) {
        return float3(0);
    }
    bool westward = aoUseFirst(west, east, centerDepth, reversed, view);
    bool northward = aoUseFirst(north, south, centerDepth, reversed, view);
    float3 center = aoViewPosition(float2(pixel), aoViewDepth(centerDepth, view, orthographic), inverseSize, view, orthographic);
    float3 horizontal = aoViewPosition(
        float2(pixel + int2(westward ? -1 : 1, 0)), aoViewDepth(westward ? west : east, view, orthographic), inverseSize, view, orthographic
    ) - center;
    float3 vertical = aoViewPosition(
        float2(pixel + int2(0, northward ? -1 : 1)), aoViewDepth(northward ? north : south, view, orthographic), inverseSize, view, orthographic
    ) - center;
    float3 right = horizontal * (westward ? -1.0f : 1.0f);
    float3 up = vertical * (northward ? 1.0f : -1.0f);
    float3 normal = cross(right, up);
    float length2 = dot(normal, normal);
    return length2 > 0.0f && isfinite(length2) ? normal * rsqrt(length2) : float3(0);
}

static inline float3 aoNormal(
    int2 pixel, float centerDepth, int2 size, depth2d<float> depth,
    texture2d<uint, access::read> attributes, float2 inverseSize, constant FrostfieldOcclusionView& view, bool orthographic
) {
    if (view.sampling.z != 0.0f) {
        float3 world = frostfieldUnpackSurfaceAttributes(attributes.read(uint2(pixel)).xy).normal;
        return normalize(aoViewFromWorld(view) * world);
    }
    return aoReconstructedNormal(pixel, centerDepth, size, depth, inverseSize, view, orthographic);
}

constant float aoBayer[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};

// The cosine-weighted visibility of one slice side between the view vector and horizon h,
// relative to the projected normal angle n (the GTAO inner integral).
static float aoArc(float h, float n, float cosN, float sinN) {
    return 0.25f * (cosN + 2.0f * h * sinN - cos(2.0f * h - n));
}

// Jimenez et al. 2016: the cosine-weighted integral of unoccluded directions in one slice
// between horizons h0 and h1, as components along the slice direction and the view vector.
static float3 aoSliceBentNormal(float h0, float h1, float n, float cosN, float sinN, float3 direction, float3 toCamera) {
    float t0 = (6.0f * sin(h0 - n) - sin(3.0f * h0 - n) + 6.0f * sin(h1 - n) - sin(3.0f * h1 - n)
        + 16.0f * sinN - 3.0f * (sin(h0 + n) + sin(h1 + n))) / 12.0f;
    float t1 = (-cos(3.0f * h0 - n) - cos(3.0f * h1 - n) + 8.0f * cosN - 3.0f * (cos(h0 + n) + cos(h1 + n))) / 12.0f;
    return direction * t0 + toCamera * t1;
}

// The nearest visible depth around each mapped trace center gives reduced
// samples a consistent full-size position and surface normal.
static int2 aoGuidePixel(int2 tracePixel, int2 size, int2 traceSize, depth2d<float> depth, bool reversed) {
    float2 center = (float2(tracePixel) + 0.5f) * float2(size) / float2(traceSize);
    int2 corner = int2(floor(center - 0.5f));
    int2 result = min(int2(center), size - 1);
    float nearest = reversed ? 0.0f : 1.0f;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 p = clamp(corner + int2(x, y), int2(0), size - 1);
            float sample = depth.read(uint2(p));
            if (reversed ? sample > nearest : sample < nearest) {
                nearest = sample;
                result = p;
            }
        }
    }
    return result;
}

// Raw occlusion, one minus visibility (w), and the view-space bend from the normal to the
// bent normal (xyz), before spatial filtering. Storing occlusion and the bend keeps open
// surfaces at exactly zero in half precision, and pixels without a usable normal read zero.
static inline float4 aoHorizons(
    float2 position, depth2d<float> depth, texture2d<uint, access::read> attributes,
    constant FrostfieldOcclusionView& view, bool controlled, bool orthographic
) {
    int2 tracePixel = int2(position);
    int2 size(depth.get_width(), depth.get_height());
    int2 traceSize = int2(view.phase.zw);
    bool reversed = view.options.z != 0.0f;
    int2 pixel = controlled && any(traceSize != size) ? aoGuidePixel(tracePixel, size, traceSize, depth, reversed) : tracePixel;
    float centerDepth = depth.read(uint2(pixel));
    if (!aoHasSurface(centerDepth, reversed)) {
        return float4(0);
    }
    float2 inverseSize = 1.0f / float2(size);
    float viewDepth = aoViewDepth(centerDepth, view, orthographic);
    float3 center = aoViewPosition(float2(pixel), viewDepth, inverseSize, view, orthographic);
    float3 normal = aoNormal(pixel, centerDepth, size, depth, attributes, inverseSize, view, orthographic);
    float radius = view.options.x;
    bool fixedScreenRadius = controlled && view.controls.w > 0.0f;
    float screenRadius = min(fixedScreenRadius ? view.controls.w
        : (orthographic ? radius * view.options.y : radius * view.options.y / viewDepth), view.sampling.w);
    if (all(normal == 0.0f) || screenRadius < 1.0f) {
        return float4(0);
    }
    float3 toCamera = orthographic ? float3(0, 0, 1) : normalize(-center);
    float falloffRange = view.options.w * radius;
    uint slices = uint(view.sampling.x);
    uint steps = uint(view.sampling.y);
    // A 4×4 pattern of slice rotations and step offsets that the 4×4 filter averages.
    uint cell = uint(tracePixel.x & 3) + 4u * uint(tracePixel.y & 3);
    float sliceNoise = (aoBayer[cell] + 0.5f) / 16.0f;
    float stepNoise = (aoBayer[(cell * 7u + 5u) & 15u] + 0.5f) / 16.0f;
    if (controlled) { stepNoise = fract(stepNoise + view.phase.y); }
    float minimumStep = 1.3f / screenRadius;

    float visibility = 0.0f;
    float reference = 0.0f;
    float3 bent = float3(0);
    float3 reference3 = float3(0);
    bool anyRaised = false;
    for (uint slice = 0; slice < slices; ++slice) {
        float phi = (float(slice) + sliceNoise) * M_PI_F / float(slices);
        if (controlled) { phi += view.phase.x; }
        float2 direction2 = float2(cos(phi), sin(phi));
        // Screen steps whose back-projection runs along this view-space direction.
        float2 screenDirection = float2(direction2.x * view.depth.w, -direction2.y);
        float2 omega = screenDirection * screenRadius;
        float sliceRadius = radius;
        float sliceFalloff = falloffRange;
        if (fixedScreenRadius) {
            float inverseAspect = 1.0f / length(screenDirection);
            omega *= inverseAspect;
            sliceRadius = screenRadius * inverseAspect * (orthographic ? 1.0f : viewDepth) / view.options.y;
            if (!isfinite(sliceRadius)) { continue; }
            sliceFalloff = view.options.w * sliceRadius;
        }
        float3 direction = float3(direction2, 0.0f);
        float3 orthoDirection = direction - dot(direction, toCamera) * toCamera;
        float3 axis = normalize(cross(orthoDirection, toCamera));
        float3 projectedNormal = normal - axis * dot(normal, axis);
        float projectedLength = length(projectedNormal);
        if (projectedLength < 1e-6f) {
            continue;
        }
        float signN = dot(orthoDirection, projectedNormal) >= 0.0f ? 1.0f : -1.0f;
        float cosN = saturate(dot(projectedNormal, toCamera) / projectedLength);
        float n = signN * acos(cosN);
        float sinN = sin(n);
        // The tangent-plane horizons cos(n ± π/2) = ∓sin(n), taken from the same vectors the
        // samples use so that coplanar samples never raise them through rounding.
        float tangentSine = dot(projectedNormal, normalize(orthoDirection)) / projectedLength;
        float lowCos0 = -tangentSine;
        float lowCos1 = tangentSine;
        float horizonCos0 = lowCos0;
        float horizonCos1 = lowCos1;
        bool raised = false;
        for (uint step = 0; step < steps; ++step) {
            float s = (float(step) + stepNoise) / float(steps);
            s = aoSpatialExtraControl ? pow(s, view.clipMaximum.w) + minimumStep : s * s + minimumStep;
            if (fixedScreenRadius) { s = min(s, 1.0f); }
            float2 offset = omega * s;
            for (uint side = 0; side < 2u; ++side) {
                float2 screen = float2(pixel) + 0.5f + (side == 0u ? offset : -offset);
                float3 tapPosition;
                // Past the outermost texel centers the 2×2 footprint would leave the texture
                // and clamp, placing the point off its surface.
                if (any(screen < 0.5f) || any(screen > float2(size) - 0.5f)
                    || !aoSample(screen, size, depth, inverseSize, view, orthographic, tapPosition)) {
                    continue;
                }
                float3 delta = tapPosition - center;
                if (controlled && ((view.controls.x > 0.0f && abs(delta.z) >= view.controls.x)
                    || (view.controls.y > 0.0f && dot(normal, delta) <= view.controls.y))) { continue; }
                float distance = length(delta);
                if (!(distance > 0.0f)) {
                    continue;
                }
                float weight = sliceFalloff > 0.0f ? saturate((sliceRadius - distance) / sliceFalloff)
                    : float(distance <= sliceRadius);
                // Raises below 1e-5 in cosine are rounding on coplanar samples, not occluders.
                float low = side == 0u ? lowCos0 : lowCos1;
                float horizonCos = mix(low, dot(delta / distance, toCamera), weight);
                if (horizonCos > low + 1e-5f) {
                    raised = true;
                    if (side == 0u) {
                        horizonCos0 = max(horizonCos0, horizonCos);
                    } else {
                        horizonCos1 = max(horizonCos1, horizonCos);
                    }
                }
            }
        }
        // The open slice integrates to cos n + n sin n with mean direction
        // (4/3 sin n, 2/3 cos n) along the slice direction and the view vector; these
        // references normalize away the slice discretization.
        float3 unitDirection = normalize(orthoDirection);
        float open = cosN + n * sinN;
        float3 openBent = unitDirection * (4.0f / 3.0f * sinN) + toCamera * (2.0f / 3.0f * cosN);
        reference += projectedLength * open;
        reference3 += projectedLength * openBent;
        if (!raised) {
            visibility += projectedLength * open;
            bent += projectedLength * openBent;
            continue;
        }
        anyRaised = true;
        float h0 = -acos(clamp(horizonCos1, -1.0f, 1.0f));
        float h1 = acos(clamp(horizonCos0, -1.0f, 1.0f));
        h0 = n + clamp(h0 - n, -M_PI_2_F, M_PI_2_F);
        h1 = n + clamp(h1 - n, -M_PI_2_F, M_PI_2_F);
        visibility += projectedLength * (aoArc(h0, n, cosN, sinN) + aoArc(h1, n, cosN, sinN));
        // The cosine-weighted mean unoccluded direction in this slice.
        bent += projectedLength * aoSliceBentNormal(h0, h1, n, cosN, sinN, unitDirection, toCamera);
    }
    if (!anyRaised) {
        return float4(0);
    }
    float occlusion = reference > 0.0f && visibility < reference ? saturate(1.0f - visibility / reference) : 0.0f;
    if (!(occlusion > 8.0f * FLT_EPSILON)) {
        // Roundoff in the weighted sums can create tiny positive occlusion on a plane.
        // Zero it before the half-float visibility write, which truncates to the next lower value.
        return float4(0);
    }
    // Fade in over the first pixel of radius so tiny radii do not pop.
    if (controlled && view.controls.z != 1.0f) { occlusion = 1.0f - pow(1.0f - occlusion, view.controls.z); }
    occlusion *= smoothstep(1.0f, 2.0f, screenRadius);
    // Replace the open slices' discrete sum by the exact normal, so open surfaces bend nothing.
    bent += normal * length(reference3) - reference3;
    float bentLength = length(bent);
    float3 bentNormal = bentLength > 0.0f && isfinite(bentLength) ? bent / bentLength : normal;
    return float4(bentNormal - normal, occlusion);
}

// A depth- and normal-aware 4×4 mean of bend and occlusion that removes the rotation pattern.
fragment float4 frostfield_gtao_fragment(
    float4 position [[position]], depth2d<float> depth [[texture(0)]],
    texture2d<uint, access::read> attributes [[texture(1)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]]
) {
    return aoHorizons(position.xy, depth, attributes, view, false, aoOrthographic);
}

fragment float4 frostfield_gtao_controlled(
    float4 position [[position]], depth2d<float> depth [[texture(0)]],
    texture2d<uint, access::read> attributes [[texture(1)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]]
) {
    return aoHorizons(position.xy, depth, attributes, view, true, aoOrthographic);
}

static inline float4 aoFiltered(
    int2 pixel, texture2d<float, access::read> raw, depth2d<float> depth,
    texture2d<uint, access::read> attributes, constant FrostfieldOcclusionView& view, bool filter, bool orthographic
) {
    float4 centerValue = raw.read(uint2(pixel));
    if (!filter) {
        return centerValue;
    }
    int2 size(depth.get_width(), depth.get_height());
    bool reversed = view.options.z != 0.0f;
    float centerDepth = depth.read(uint2(pixel));
    float2 inverseSize = 1.0f / float2(size);
    float viewDepth = aoViewDepth(centerDepth, view, orthographic);
    float3 center = aoViewPosition(float2(pixel), viewDepth, inverseSize, view, orthographic);
    bool hasAttributes = view.sampling.z != 0.0f;
    float3 normal = hasAttributes
        ? frostfieldUnpackSurfaceAttributes(attributes.read(uint2(pixel)).xy).normal
        : aoReconstructedNormal(pixel, centerDepth, size, depth, inverseSize, view, orthographic);
    float3 viewNormal = hasAttributes ? aoViewFromWorld(view) * normal : normal;
    // Neighbors more than two pixel footprints off the center's tangent plane are excluded.
    float tolerance = 2.0f * (orthographic ? 1.0f : viewDepth) / view.options.y;
    float4 sum = float4(0);
    float total = 0.0f;
    for (int y = -1; y <= 2; ++y) {
        for (int x = -1; x <= 2; ++x) {
            int2 tap = pixel + int2(x, y);
            if (any(tap < 0) || any(tap >= size)) {
                continue;
            }
            float tapDepth = depth.read(uint2(tap));
            if (!aoHasSurface(tapDepth, reversed)) {
                continue;
            }
            float weight = 1.0f;
            if (x != 0 || y != 0) {
                float3 position = aoViewPosition(float2(tap), aoViewDepth(tapDepth, view, orthographic), inverseSize, view, orthographic);
                weight = all(viewNormal == 0.0f) ? float(abs(position.z - center.z) < tolerance)
                    : saturate(1.0f - abs(dot(position - center, viewNormal)) / tolerance);
                if (hasAttributes) {
                    float3 tapNormal = frostfieldUnpackSurfaceAttributes(attributes.read(uint2(tap)).xy).normal;
                    float facing = saturate(dot(tapNormal, normal));
                    facing *= facing;
                    facing *= facing;
                    weight *= facing * facing;
                }
            }
            sum += weight * raw.read(uint2(tap));
            total += weight;
        }
    }
    float4 mean = sum / max(total, 1e-6f);
    return float4(mean.xyz, sum.w > 0.0f ? mean.w : 0.0f);
}

// The world-space bend (xyz) and visibility (w) for PBR shading, both scaled by strength.
fragment float4 frostfield_gtao_visibility(
    float4 position [[position]],
    texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]],
    texture2d<uint, access::read> attributes [[texture(2)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]],
    constant float4& filter [[buffer(1)]]
) {
    int2 pixel = int2(position.xy);
    float strength = aoResolvedStrength(pixel, depth, view, aoOrthographic);
    if (strength == 0.0f) {
        return float4(0, 0, 0, 1);
    }
    float4 value = aoFiltered(pixel, raw, depth, attributes, view, filter.x != 0.0f, aoOrthographic);
    // The view rotation is orthonormal, so its transpose maps view directions to world.
    float3 world = transpose(aoViewFromWorld(view)) * (strength * value.xyz);
    return float4(world, 1.0f - strength * value.w);
}

// Scene color darkened by the filtered visibility, preserving alpha.
fragment float4 frostfield_gtao_apply(
    float4 position [[position]],
    texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]],
    texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> color [[texture(3)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]],
    constant float4& filter [[buffer(1)]]
) {
    int2 pixel = int2(position.xy);
    float4 scene = color.read(uint2(pixel));
    float strength = aoResolvedStrength(pixel, depth, view, aoOrthographic);
    if (strength == 0.0f) {
        return scene;
    }
    float occlusion = aoFiltered(pixel, raw, depth, attributes, view, filter.x != 0.0f, aoOrthographic).w;
    return float4(scene.rgb * (1.0f - strength * occlusion), scene.a);
}

static inline float4 aoReduced(
    int2 pixel, texture2d<float, access::read> raw, depth2d<float> depth,
    texture2d<uint, access::read> attributes, constant FrostfieldOcclusionView& view, bool filter, bool orthographic
) {
    int2 size(depth.get_width(), depth.get_height());
    int2 traceSize(raw.get_width(), raw.get_height());
    bool reversed = view.options.z != 0.0f;
    float stored = depth.read(uint2(pixel));
    float2 inverseSize = 1.0f / float2(size);
    float distance = aoViewDepth(stored, view, orthographic);
    if (!(isfinite(distance) && (orthographic ? distance >= 0.0f : distance > 0.0f))) { return float4(0.0f); }
    float3 center = aoViewPosition(float2(pixel), distance, inverseSize, view, orthographic);
    float3 normal = aoNormal(pixel, stored, size, depth, attributes, inverseSize, view, orthographic);
    if (all(normal == 0.0f)) { return float4(0.0f); }
    float2 location = (float2(pixel) + 0.5f) * float2(traceSize) * inverseSize - 0.5f;
    int2 base = int2(floor(location));
    float2 fraction = fract(location);
    int start = filter ? -1 : 0;
    int end = filter ? 2 : 1;
    float tolerance = 2.0f * (orthographic ? 1.0f : distance) / view.options.y;
    float4 sum = float4(0.0f);
    float weights = 0.0f;
    for (int y = start; y <= end; ++y) {
        for (int x = start; x <= end; ++x) {
            int2 tap = base + int2(x, y);
            if (any(tap < 0) || any(tap >= traceSize)) { continue; }
            int2 guide = aoGuidePixel(tap, size, traceSize, depth, reversed);
            float z = depth.read(uint2(guide));
            if (!aoHasSurface(z, reversed)) { continue; }
            float3 sampleNormal = aoNormal(guide, z, size, depth, attributes, inverseSize, view, orthographic);
            float alignment = saturate(dot(normal, sampleNormal));
            if (alignment < 0.5f) { continue; }
            float3 point = aoViewPosition(float2(guide), aoViewDepth(z, view, orthographic), inverseSize, view, orthographic);
            float continuity = saturate(1.0f - abs(dot(normal, point - center)) / tolerance);
            float2 offset = float2(tap) - location;
            float spatial = filter ? 1.0f / (1.0f + 2.0f * dot(offset, offset))
                : (x == 0 ? 1.0f - fraction.x : fraction.x) * (y == 0 ? 1.0f - fraction.y : fraction.y);
            float weight = spatial * continuity * pow(alignment, 8.0f);
            sum += raw.read(uint2(tap)) * weight;
            weights += weight;
        }
    }
    return weights > 0.001f ? sum / weights : float4(0.0f);
}

fragment float4 frostfield_gtao_visibility_reduced(
    float4 position [[position]], texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]], constant float4& filter [[buffer(1)]]
) {
    int2 pixel = int2(position.xy);
    float strength = aoResolvedStrength(pixel, depth, view, aoOrthographic);
    if (strength == 0.0f) {
        return float4(0, 0, 0, 1);
    }
    float4 value = aoReduced(pixel, raw, depth, attributes, view, filter.x != 0.0f, aoOrthographic);
    float3 world = transpose(aoViewFromWorld(view)) * (strength * value.xyz);
    return float4(world, 1.0f - strength * value.w);
}

fragment float4 frostfield_gtao_apply_reduced(
    float4 position [[position]], texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> color [[texture(3)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]], constant float4& filter [[buffer(1)]]
) {
    int2 pixel = int2(position.xy);
    float4 scene = color.read(uint2(pixel));
    float strength = aoResolvedStrength(pixel, depth, view, aoOrthographic);
    if (strength == 0.0f) { return scene; }
    float occlusion = aoReduced(pixel, raw, depth, attributes, view, filter.x != 0.0f, aoOrthographic).w;
    return float4(scene.rgb * (1.0f - strength * occlusion), scene.a);
}

struct FrostfieldOcclusionPoisson {
    uint sampleCount;
    uint sampleIndex;
    float lumaPhi;
    float depthPhi;
    float normalPhi;
};

static float4 aoPoisson(
    int2 pixel, texture2d<float, access::read> raw, depth2d<float> depth,
    texture2d<uint, access::read> attributes, constant FrostfieldOcclusionView& view,
    constant FrostfieldOcclusionPoisson& settings, constant float2* offsets
) {
    int2 size(depth.get_width(), depth.get_height());
    int2 traceSize(raw.get_width(), raw.get_height());
    bool reduced = any(size != traceSize);
    bool reversed = view.options.z != 0.0f;
    float stored = depth.read(uint2(pixel));
    float2 inverseSize = 1.0f / float2(size);
    float distance = aoViewDepth(stored, view, aoOrthographic);
    float3 center = aoViewPosition(float2(pixel), distance, inverseSize, view, aoOrthographic);
    float3 normal = aoNormal(pixel, stored, size, depth, attributes, inverseSize, view, aoOrthographic);
    float4 centerValue = reduced ? aoReduced(pixel, raw, depth, attributes, view, false, aoOrthographic)
                                 : raw.read(uint2(pixel));
    if (all(normal == 0.0f)) { return centerValue; }

    uint cell = uint(pixel.x & 3) + 4u * uint(pixel.y & 3);
    float noise = (aoBayer[cell] + 0.5f) / 16.0f;
    float angle = (noise + float(settings.sampleIndex) * 0.25f) * (2.0f * M_PI_F) + view.phase.x;
    float2 turn(cos(angle), sin(angle));
    // A reduced texel may cover two depth layers. Keep its guide within the
    // same tangent-plane footprint used by the regular reduced resolve.
    float footprint = 2.0f * (aoOrthographic ? 1.0f : distance) / view.options.y;
    float depthTolerance = reduced ? min(settings.depthPhi, footprint) : settings.depthPhi;

    float4 sum = float4(0.0f);
    float weights = 1.0f;
    for (uint tap = 0; tap < settings.sampleCount; ++tap) {
        float2 offset = offsets[tap];
        float2 screen = float2(pixel) + 0.5f + float2(dot(offset, float2(turn.x, -turn.y)), dot(offset, turn.yx));
        if (any(screen < 0.0f) || any(screen >= float2(size))) { continue; }
        int2 sample = reduced ? min(int2(screen * inverseSize * float2(traceSize)), traceSize - 1) : int2(screen);
        int2 guide = reduced ? aoGuidePixel(sample, size, traceSize, depth, reversed) : sample;
        float z = depth.read(uint2(guide));
        if (!aoHasSurface(z, reversed)) { continue; }
        float3 sampleNormal = aoNormal(guide, z, size, depth, attributes, inverseSize, view, aoOrthographic);
        float alignment = saturate(dot(normal, sampleNormal));
        if (!(alignment > 0.0f)) { continue; }
        float3 point = aoViewPosition(float2(guide), aoViewDepth(z, view, aoOrthographic), inverseSize, view, aoOrthographic);
        float plane = saturate(1.0f - abs(dot(point - center, normal)) / depthTolerance);
        float4 neighbor = raw.read(uint2(sample));
        // Luminance of grayscale visibility differs by the same amount as occlusion.
        float luma = saturate(1.0f - abs(neighbor.w - centerValue.w) / settings.lumaPhi);
        float weight = luma * plane * pow(alignment, settings.normalPhi);
        sum += (neighbor - centerValue) * weight;
        weights += weight;
    }
    return centerValue + sum / weights;
}

fragment float4 frostfield_gtao_visibility_poisson(
    float4 position [[position]], texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]],
    constant FrostfieldOcclusionPoisson& settings [[buffer(1)]], constant float2* offsets [[buffer(2)]]
) {
    int2 pixel = int2(position.xy);
    float strength = aoResolvedStrength(pixel, depth, view, aoOrthographic);
    if (strength == 0.0f) {
        return float4(0, 0, 0, 1);
    }
    float4 value = aoPoisson(pixel, raw, depth, attributes, view, settings, offsets);
    float3 world = transpose(aoViewFromWorld(view)) * (strength * value.xyz);
    return float4(world, 1.0f - strength * value.w);
}

fragment float4 frostfield_gtao_apply_poisson(
    float4 position [[position]], texture2d<float, access::read> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> color [[texture(3)]], constant FrostfieldOcclusionView& view [[buffer(0)]],
    constant FrostfieldOcclusionPoisson& settings [[buffer(1)]], constant float2* offsets [[buffer(2)]]
) {
    int2 pixel = int2(position.xy);
    float4 scene = color.read(uint2(pixel));
    float strength = aoResolvedStrength(pixel, depth, view, aoOrthographic);
    if (strength == 0.0f) { return scene; }
    float occlusion = aoPoisson(pixel, raw, depth, attributes, view, settings, offsets).w;
    return float4(scene.rgb * (1.0f - strength * occlusion), scene.a);
}

static bool aoMatchesMotionDepth(
    float stored, float expectedRaw, constant FrostfieldOcclusionView& prior, bool orthographic
) {
    // B is Float16. First compare the quantized values, then allow the same
    // view-distance tolerance as camera reprojection for sloped bilinear taps.
    if (float(half(stored)) == expectedRaw) { return true; }
    float distance = aoViewDepth(stored, prior, orthographic);
    float distanceTolerance = max(0.015f, distance * 0.02f);
    float rawTolerance = orthographic ? distanceTolerance / prior.depth.y
        : distanceTolerance / (distance * distance * prior.depth.y);
    return abs(stored - expectedRaw) <= rawTolerance;
}

// Blend in world space so the stored bend follows the receiver across camera rotations.
fragment float4 frostfield_gtao_temporal(
    float4 position [[position]],
    texture2d<float, access::read> current [[texture(0)]], depth2d<float> depth [[texture(1)]],
    texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> previous [[texture(3)]], depth2d<float> priorDepth [[texture(4)]],
    texture2d<uint, access::read> priorAttributes [[texture(5)]],
    texture2d<float, access::read> motionTexture [[texture(6), function_constant(aoTemporalMotion)]],
    constant FrostfieldOcclusionView& view [[buffer(0)]], constant FrostfieldOcclusionView& prior [[buffer(1)]],
    constant float4x4& clipFromCurrent [[buffer(2)]], constant float4& options [[buffer(3)]]
) {
    int2 pixel = int2(position.xy);
    uint2 address = uint2(pixel);
    float4 value = current.read(address);
    float stored = depth.read(address);
    if (!(options.y > 0.0f) || view.viewFromWorld[0].w == 0.0f ||
        !aoHasSurface(stored, view.options.z != 0.0f)) { return value; }

    int2 size = int2(depth.get_width(), depth.get_height());
    float2 inverseSize = 1.0f / float2(size);
    bool orthographic = view.viewFromWorld[1].w != 0.0f;
    bool priorOrthographic = prior.viewFromWorld[1].w != 0.0f;
    float currentClip = 1.0f;
    if (aoTemporalClip) {
        currentClip = aoClipWeight(pixel, stored, inverseSize, view, orthographic);
        if (currentClip == 0.0f) { return value; }
    }
    float2 oldPixel;
    float expected;
    if (aoTemporalMotion) {
        float4 motion = motionTexture.read(address);
        if (!(all(isfinite(motion)) && motion.w > 0.5f && motion.z >= 0.0f && motion.z <= 1.0f)) { return value; }
        // The raster vector includes the difference in projection jitter.
        oldPixel = position.xy - 0.5f + motion.xy;
        if (any(oldPixel < -0.5f) || any(oldPixel >= float2(size) - 0.5f)) { return value; }
        expected = motion.z;
    } else {
        float distance = aoViewDepth(stored, view, orthographic);
        float3 point = aoViewPosition(float2(pixel), distance, inverseSize, view, orthographic);
        float4 clip = clipFromCurrent * float4(point, 1.0f);
        if (!(all(isfinite(clip)) && clip.w > 0.0f && clip.z >= 0.0f && clip.z <= clip.w)) { return value; }
        float2 ndc = clip.xy / clip.w;
        if (any(abs(ndc) >= 1.0f)) { return value; }
        expected = priorOrthographic ? aoViewDepth(clip.z / clip.w, prior, true) : clip.w * options.x;
        if (!(isfinite(expected) && (priorOrthographic ? expected >= 0.0f : expected > 0.0f))) { return value; }
        if (aoLogView(prior) && expected > prior.viewFromWorld[2].w) { return value; }
        oldPixel = (ndc * float2(0.5f, -0.5f) + 0.5f) * float2(size) - 0.5f;
    }
    float3 currentNormal = transpose(aoViewFromWorld(view)) * aoNormal(pixel, stored, size, depth, attributes, inverseSize, view, orthographic);
    if (all(currentNormal == 0.0f)) { return value; }
    int2 corner = int2(floor(oldPixel));
    float2 fraction = fract(oldPixel);
    float4 sum = value;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 p = corner + int2(x, y);
            if (any(p < 0) || any(p >= size)) { continue; }
            float z = priorDepth.read(uint2(p));
            if (!aoHasSurface(z, prior.options.z != 0.0f)) { continue; }
            if (aoTemporalMotion) {
                if (!aoMatchesMotionDepth(z, expected, prior, priorOrthographic)) { continue; }
            } else if (abs(aoViewDepth(z, prior, priorOrthographic) - expected) >= max(0.015f, expected * 0.02f)) {
                continue;
            }
            float3 normal = transpose(aoViewFromWorld(prior)) * aoNormal(p, z, size, priorDepth, priorAttributes, inverseSize, prior, priorOrthographic);
            if (dot(currentNormal, normal) <= 0.9f) { continue; }
            float weight = (x == 0 ? 1.0f - fraction.x : fraction.x) * (y == 0 ? 1.0f - fraction.y : fraction.y);
            if (aoTemporalClip) {
                float priorClip = aoClipWeight(p, z, inverseSize, prior, priorOrthographic);
                // Compare occlusion in the current mask and weight by the prior coverage.
                // Cross-multiplying avoids recovering unmasked values from very small weights.
                float4 open(0, 0, 0, 1);
                sum += ((previous.read(uint2(p)) - open) * currentClip - (value - open) * priorClip) * (weight * options.y);
            } else {
                sum += (previous.read(uint2(p)) - value) * (weight * options.y);
            }
        }
    }
    return sum;
}
