#include <metal_stdlib>
#include "FrostfieldScreenSpace.h"
using namespace metal;

// Matches ScreenSpaceReflections.encode's constants: ten float4 slots. A view point
// (x, y, -d) has NDC (x / (d·projection.x) - projection.z, y / (d·projection.y) - projection.w),
// and 1/d is depth.x + depth.y · w, where w runs from 0 at the cleared depth to 1 at the near plane.
struct FrostfieldReflectionView {
    float4 projection;   // reciprocal projection scales in x and y, then the projection's x and y centers
    float4 depth;        // 1/d at the cleared depth, its increase to the near plane, near distance
    float4 march;        // maximum distance, thickness, step count, refinement count
    float4 fade;         // maximum roughness, edge fade fraction, intensity, 1 for reversed depth
    float4 viewFromWorld[3];  // columns of the world-to-view rotation
    float4 environment;  // cos and sin of the environment yaw, intensity, maximum specular LOD
    float4 options;      // environment, receiver mask, blur radius, stochastic sampling
    float4 trace;        // trace width and height, low and high halves of the seed
};

struct FrostfieldReflectionHistory {
    float4x4 viewFromCurrent;
    float4x4 clipFromCurrent;
    float4 depth; // prior far inverse distance and range, reversed flag, reprojection mode
};

static bool ssrHasSurface(float depth, constant FrostfieldReflectionView& view) {
    return view.fade.w != 0.0f ? depth > 0.0f : depth < 1.0f;
}

static float ssrViewDepth(float depth, constant FrostfieldReflectionView& view) {
    float nearWeight = view.fade.w != 0.0f ? depth : 1.0f - depth;
    return 1.0f / fma(view.depth.y, nearWeight, view.depth.x);
}

// Camera-space position at a continuous screen position in pixels (texel centers at +0.5).
static float3 ssrViewPosition(float2 screen, float viewDepth, float2 size, constant FrostfieldReflectionView& view) {
    float2 ndc = float2(2.0f * screen.x / size.x - 1.0f, 1.0f - 2.0f * screen.y / size.y);
    return float3((ndc + view.projection.zw) * view.projection.xy, -1.0f) * viewDepth;
}

static float2 ssrScreen(float3 position, float2 size, constant FrostfieldReflectionView& view) {
    float2 ndc = position.xy / (-position.z * view.projection.xy) - view.projection.zw;
    return float2((ndc.x + 1.0f) * 0.5f * size.x, (1.0f - ndc.y) * 0.5f * size.y);
}

// The scene depth behind a point on the ray, or infinity where nothing was drawn.
static float ssrSceneDepth(float2 screen, int2 size, depth2d<float, access::read> depth, constant FrostfieldReflectionView& view) {
    uint2 texel = uint2(clamp(int2(floor(screen)), int2(0), size - 1));
    float stored = depth.read(texel);
    return ssrHasSurface(stored, view) ? ssrViewDepth(stored, view) : INFINITY;
}

static float3 ssrEnvironmentDirection(float3 direction, float2 rotation) {
    // The inverse of the environment's yaw, as in PBR shading.
    return float3(
        rotation.x * direction.x - rotation.y * direction.z,
        direction.y,
        rotation.y * direction.x + rotation.x * direction.z
    );
}

static float ssrMask(uint2 pixel, texture2d<float, access::read> mask, constant FrostfieldReflectionView& view) {
    if (view.options.y == 0.0f) { return 1.0f; }
    float value = mask.read(pixel).x;
    return isfinite(value) ? saturate(value) : 0.0f;
}

// PCG hash for deterministic per-pixel ray directions.
static uint ssrHash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint mixed = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (mixed >> 22u) ^ mixed;
}

static float2 ssrRandom(uint2 pixel, constant FrostfieldReflectionView& view) {
    uint seed = uint(view.trace.z) | (uint(view.trace.w) << 16u);
    uint sequence = ssrHash(pixel.x ^ ssrHash(pixel.y ^ ssrHash(seed)));
    return float2(sequence >> 8u, ssrHash(sequence) >> 8u) * (1.0f / 16777216.0f);
}

// Isotropic bounded GGX visible-normal sampling, using the spherical cap of
// Eto and Tokuyoshi (2023). Energy remains the receiver's split-sum reflectance.
static float3 ssrGGXDirection(float3 normal, float3 toCamera, float roughness, float2 sample) {
    float3 axis = abs(normal.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(axis, normal));
    float3 bitangent = cross(normal, tangent);
    float3 localView = float3(dot(toCamera, tangent), dot(toCamera, bitangent), dot(toCamera, normal));
    float alpha = max(0.001f, roughness * roughness);
    float3 standardView = normalize(float3(alpha * localView.xy, localView.z));
    float scale = 1.0f + length(localView.xy);
    float denominator = scale * scale + pow(alpha * localView.z, 2.0f);
    float cap = standardView.z * (1.0f - alpha * alpha) * scale * scale / denominator;
    float z = mix(1.0f, -cap, sample.y);
    float radial = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PI_F * sample.x;
    float3 stretched = standardView + float3(radial * cos(phi), radial * sin(phi), z);
    float3 halfVector = normalize(
        alpha * (tangent * stretched.x + bitangent * stretched.y) + normal * max(0.0f, stretched.z)
    );
    return reflect(-toCamera, halfVector);
}

static bool ssrHistoryDepthMatches(float stored, float expected, constant FrostfieldReflectionHistory& history) {
    bool reversed = history.depth.z != 0.0f;
    if (!(reversed ? stored > 0.0f && stored <= 1.0f : stored >= 0.0f && stored < 1.0f)) { return false; }
    float nearWeight = reversed ? stored : 1.0f - stored;
    float distance = 1.0f / fma(history.depth.y, nearWeight, history.depth.x);
    return abs(distance - expected) <= max(0.01f, 0.005f * expected);
}

static float3 ssrHitRadiance(
    uint2 hit, float2 extent, texture2d<float, access::read> color, depth2d<float, access::read> depth,
    texture2d<float, access::read> previousColor, depth2d<float, access::read> previousDepth,
    constant FrostfieldReflectionView& view, constant FrostfieldReflectionHistory& history
) {
    float3 radiance = color.read(hit).rgb;
    if (history.depth.w == 0.0f) { return radiance; }
    float currentDistance = ssrViewDepth(depth.read(hit), view);
    if (history.depth.w == 1.0f) {
        return ssrHistoryDepthMatches(previousDepth.read(hit), currentDistance, history) ? previousColor.read(hit).rgb : radiance;
    }

    float3 point = ssrViewPosition(float2(hit) + 0.5f, currentDistance, extent, view);
    float4 clip = history.clipFromCurrent * float4(point, 1);
    if (!(all(isfinite(clip)) && clip.w > 0.0f && clip.z >= 0.0f && clip.z <= clip.w)) { return radiance; }
    float2 ndc = clip.xy / clip.w;
    if (any(abs(ndc) >= 1.0f)) { return radiance; }
    float expected = -(history.viewFromCurrent * float4(point, 1)).z;
    if (!(expected > 0.0f && isfinite(expected))) { return radiance; }

    float2 previousPixel = (ndc * float2(0.5f, -0.5f) + 0.5f) * extent - 0.5f;
    int2 corner = int2(floor(previousPixel));
    float2 fraction = fract(previousPixel);
    float3 result = radiance;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 sample = corner + int2(x, y);
            if (any(sample < 0) || any(sample >= int2(extent))) { continue; }
            uint2 pixel = uint2(sample);
            if (!ssrHistoryDepthMatches(previousDepth.read(pixel), expected, history)) { continue; }
            float weight = (x == 0 ? 1.0f - fraction.x : fraction.x) * (y == 0 ? 1.0f - fraction.y : fraction.y);
            result += (previousColor.read(pixel).rgb - radiance) * weight;
        }
    }
    return result;
}

// McGuire and Mara style linear march, with perspective-correct depth and a refined first crossing.
// The return value is hit radiance premultiplied by confidence, plus confidence in alpha.
static float4 ssrTrace(
    float2 screenPosition, texture2d<float, access::read> color, depth2d<float, access::read> depth,
    texture2d<uint, access::read> attributes, texture2d<float, access::read> mask,
    texture2d<float, access::read> previousColor, depth2d<float, access::read> previousDepth,
    constant FrostfieldReflectionView& view, constant FrostfieldReflectionHistory& history
) {
    int2 pixel = int2(screenPosition);
    int2 size(color.get_width(), color.get_height());
    float2 extent = float2(size);
    float stored = depth.read(uint2(pixel));
    if (!ssrHasSurface(stored, view) || ssrMask(uint2(pixel), mask, view) <= 0.0f) {
        return float4(0);
    }
    FrostfieldSurfaceAttributes surface = frostfieldUnpackSurfaceAttributes(attributes.read(uint2(pixel)).xy);
    float maximumRoughness = view.fade.x;
    float roughnessFade = 1.0f - smoothstep(0.5f * maximumRoughness, maximumRoughness, surface.roughness);
    if (roughnessFade <= 0.0f || all(surface.specular <= 0.0f)) {
        return float4(0);
    }

    float3x3 viewFromWorld(view.viewFromWorld[0].xyz, view.viewFromWorld[1].xyz, view.viewFromWorld[2].xyz);
    float3 origin = ssrViewPosition(screenPosition, ssrViewDepth(stored, view), extent, view);
    float3 normal = normalize(viewFromWorld * surface.normal);
    float3 toCamera = normalize(-origin);
    if (dot(toCamera, normal) <= 0.0f) { return float4(0); }
    float3 ray = view.options.w != 0.0f && surface.roughness > 0.0f
        ? ssrGGXDirection(normal, toCamera, surface.roughness, ssrRandom(uint2(pixel), view))
        : reflect(-toCamera, normal);
    if (!(dot(ray, normal) > 0.0f)) {
        return float4(0);
    }

    // Clip the ray in front of the near plane, then march it across the screen.
    float travel = view.march.x;
    float near = view.depth.z;
    if (origin.z + ray.z * travel > -near) {
        travel = (-near - origin.z) / ray.z * 0.99f;
    }
    float3 end = origin + ray * travel;
    float2 start = screenPosition;
    float2 finish = ssrScreen(end, extent, view);
    float inverseStart = 1.0f / -origin.z;
    float inverseEnd = 1.0f / -end.z;
    float span = length(finish - start);
    if (!(span >= 1.0f) || !(travel > 0.0f)) {
        return float4(0);
    }
    // Never step finer than one pixel.
    uint steps = uint(min(view.march.z, ceil(span)));
    float thickness = view.march.y;
    float previous = 0.0f;
    float hit = -1.0f;
    for (uint step = 1; step <= steps; ++step) {
        float t = float(step) / float(steps);
        float2 screen = mix(start, finish, t);
        if (any(screen < 0.0f) || any(screen >= extent)) {
            break;
        }
        float rayDepth = 1.0f / mix(inverseStart, inverseEnd, t);
        float behind = rayDepth - ssrSceneDepth(screen, size, depth, view);
        if (behind > 0.0f && behind < thickness) {
            // Binary refinement toward the first crossing between the last two steps.
            float low = previous;
            float high = t;
            for (uint refinement = 0; refinement < uint(view.march.w); ++refinement) {
                float middle = 0.5f * (low + high);
                float2 point = mix(start, finish, middle);
                float pointDepth = 1.0f / mix(inverseStart, inverseEnd, middle);
                if (pointDepth > ssrSceneDepth(point, size, depth, view)) {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            hit = high;
            break;
        }
        previous = t;
    }
    if (hit < 0.0f) {
        return float4(0);
    }
    float2 hitScreen = mix(start, finish, hit);
    uint2 hitTexel = uint2(clamp(int2(floor(hitScreen)), int2(0), size - 1));
    // Reject back faces: the reflected ray must arrive at the hit surface's front.
    float3 hitNormal = viewFromWorld * frostfieldUnpackSurfaceAttributes(attributes.read(hitTexel).xy).normal;
    if (dot(hitNormal, ray) >= 0.0f) {
        return float4(0);
    }
    float2 uv = hitScreen / extent;
    float edge = view.fade.y > 0.0f
        ? saturate(min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y)) / view.fade.y) : 1.0f;
    // `hit` runs along the screen; the matching fraction of the view-space ray follows from
    // the perspective-correct depth, 1/z = mix(1/z0, 1/z1, hit).
    float rayFraction = hit * inverseEnd / mix(inverseStart, inverseEnd, hit);
    float distanceFade = 1.0f - smoothstep(0.8f, 1.0f, rayFraction * travel / view.march.x);
    float confidence = roughnessFade * edge * distanceFade * view.fade.z;

    float3 radiance = ssrHitRadiance(hitTexel, extent, color, depth, previousColor, previousDepth, view, history);
    return float4(radiance * confidence, confidence);
}

static float4 ssrCompose(
    float2 position, float4 hit, texture2d<float, access::read> color, depth2d<float, access::read> depth,
    texture2d<uint, access::read> attributes, texturecube<float> specularEnvironment,
    texture2d<float, access::read> mask, constant FrostfieldReflectionView& view
) {
    uint2 pixel = uint2(position);
    float4 scene = color.read(pixel);
    if (hit.a <= 0.0f) { return scene; }
    float selection = ssrMask(pixel, mask, view);
    if (selection <= 0.0f) { return scene; }
    FrostfieldSurfaceAttributes surface = frostfieldUnpackSurfaceAttributes(attributes.read(pixel).xy);
    if (surface.roughness >= view.fade.x || all(surface.specular <= 0.0f)) { return scene; }
    float3 replaced = float3(0);
    if (view.options.x != 0.0f) {
        constexpr sampler cubeFilter(filter::linear, mip_filter::linear);
        float3x3 viewFromWorld(view.viewFromWorld[0].xyz, view.viewFromWorld[1].xyz, view.viewFromWorld[2].xyz);
        float2 size = float2(color.get_width(), color.get_height());
        float3 origin = ssrViewPosition(position, ssrViewDepth(depth.read(pixel), view), size, view);
        float3 ray = reflect(normalize(origin), normalize(viewFromWorld * surface.normal));
        float3 worldRay = transpose(viewFromWorld) * ray;
        replaced = specularEnvironment.sample(
            cubeFilter, ssrEnvironmentDirection(worldRay, view.environment.xy),
            level(surface.roughness * view.environment.w)
        ).rgb * view.environment.z;
    }
    float3 result = scene.rgb + selection * surface.specular * (hit.rgb - hit.a * replaced) * scene.a;
    return float4(max(result, float3(0)), scene.a);
}

fragment float4 frostfield_ssr_fragment(
    float4 position [[position]], texture2d<float, access::read> color [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texturecube<float> specularEnvironment [[texture(3)]], texture2d<float, access::read> mask [[texture(4)]],
    texture2d<float, access::read> previousColor [[texture(6)]], depth2d<float, access::read> previousDepth [[texture(7)]],
    constant FrostfieldReflectionView& view [[buffer(0)]], constant FrostfieldReflectionHistory& history [[buffer(1)]]
) {
    float4 hit = ssrTrace(position.xy, color, depth, attributes, mask, previousColor, previousDepth, view, history);
    return ssrCompose(position.xy, hit, color, depth, attributes, specularEnvironment, mask, view);
}

static int2 ssrGuidePixel(int2 tracePixel, int2 size, depth2d<float, access::read> depth,
                          constant FrostfieldReflectionView& view) {
    if (all(int2(view.trace.xy) == size)) { return tracePixel; }
    float2 center = (float2(tracePixel) + 0.5f) * float2(size) / view.trace.xy;
    int2 corner = int2(floor(center - 0.5f));
    int2 nearest = min(int2(center), size - 1);
    float nearestDistance = INFINITY;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 pixel = clamp(corner + int2(x, y), int2(0), size - 1);
            float stored = depth.read(uint2(pixel));
            if (!ssrHasSurface(stored, view)) { continue; }
            float distance = ssrViewDepth(stored, view);
            if (distance < nearestDistance) {
                nearest = pixel;
                nearestDistance = distance;
            }
        }
    }
    return nearest;
}

fragment float4 frostfield_ssr_trace(
    float4 position [[position]], texture2d<float, access::read> color [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texture2d<float, access::read> mask [[texture(4)]],
    texture2d<float, access::read> previousColor [[texture(6)]], depth2d<float, access::read> previousDepth [[texture(7)]],
    constant FrostfieldReflectionView& view [[buffer(0)]], constant FrostfieldReflectionHistory& history [[buffer(1)]]
) {
    int2 size = int2(color.get_width(), color.get_height());
    int2 guide = ssrGuidePixel(int2(position.xy), size, depth, view);
    return ssrTrace(float2(guide) + 0.5f, color, depth, attributes, mask, previousColor, previousDepth, view, history);
}

fragment float4 frostfield_ssr_resolve(
    float4 position [[position]], texture2d<float, access::read> color [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<uint, access::read> attributes [[texture(2)]],
    texturecube<float> specularEnvironment [[texture(3)]], texture2d<float, access::read> mask [[texture(4)]],
    texture2d<float, access::read> raw [[texture(5)]], constant FrostfieldReflectionView& view [[buffer(0)]]
) {
    uint2 pixel = uint2(position.xy);
    float stored = depth.read(pixel);
    if (!ssrHasSurface(stored, view) || ssrMask(pixel, mask, view) <= 0.0f) { return color.read(pixel); }
    FrostfieldSurfaceAttributes surface = frostfieldUnpackSurfaceAttributes(attributes.read(pixel).xy);
    if (surface.roughness >= view.fade.x || all(surface.specular <= 0.0f)) { return color.read(pixel); }

    int2 size = int2(color.get_width(), color.get_height());
    int2 traceSize = int2(view.trace.xy);
    float2 scale = view.trace.xy / float2(size);
    float radius = surface.roughness * surface.roughness * view.options.z;
    if (all(traceSize == size) && radius < 0.5f) {
        return ssrCompose(position.xy, raw.read(pixel), color, depth, attributes, specularEnvironment, mask, view);
    }
    float3x3 viewFromWorld(view.viewFromWorld[0].xyz, view.viewFromWorld[1].xyz, view.viewFromWorld[2].xyz);
    float distance = ssrViewDepth(stored, view);
    float3 center = ssrViewPosition(position.xy, distance, float2(size), view);
    float3 normal = viewFromWorld * surface.normal;
    float2 location = position.xy * scale - 0.5f;
    int range = radius * max(scale.x, scale.y) >= 2.0f ? 2 : 1;
    float2 stride = max(radius * scale / float(range), 1.0f);
    float4 reflection = float4(0);
    float weights = 0.0f;
    for (int y = -range; y <= range; ++y) {
        for (int x = -range; x <= range; ++x) {
            int2 tap = int2(round(location + float2(x, y) * stride));
            if (any(tap < 0) || any(tap >= traceSize)) { continue; }
            int2 guide = ssrGuidePixel(tap, size, depth, view);
            float sample = depth.read(uint2(guide));
            if (!ssrHasSurface(sample, view) || ssrMask(uint2(guide), mask, view) <= 0.0f) { continue; }
            FrostfieldSurfaceAttributes neighbor = frostfieldUnpackSurfaceAttributes(attributes.read(uint2(guide)).xy);
            float alignment = saturate(dot(surface.normal, neighbor.normal));
            float3 point = ssrViewPosition(float2(guide) + 0.5f, ssrViewDepth(sample, view), float2(size), view);
            float plane = saturate(1.0f - abs(dot(normal, point - center)) / max(0.015f, distance * 0.02f));
            float roughness = saturate(1.0f - 4.0f * abs(surface.roughness - neighbor.roughness));
            float2 offset = (float2(tap) - location) / stride;
            float weight = plane * pow(alignment, 8.0f) * roughness / (1.0f + dot(offset, offset));
            reflection += raw.read(uint2(tap)) * weight;
            weights += weight;
        }
    }
    if (weights <= 0.001f) { return color.read(pixel); }
    return ssrCompose(position.xy, reflection / weights, color, depth, attributes, specularEnvironment, mask, view);
}
