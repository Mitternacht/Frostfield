#include <metal_stdlib>
using namespace metal;

constant float environmentPi = 3.14159265358979323846f;

kernel void frostfield_cube_capture_face(
    texture2d<half, access::read> source [[texture(0)]],
    texturecube<half, access::write> output [[texture(1)]],
    constant uint& face [[buffer(0)]],
    uint2 texel [[thread_position_in_grid]]
) {
    uint size = source.get_width();
    if (any(texel >= size)) {
        return;
    }
    output.write(source.read(uint2(size - 1 - texel.x, texel.y)), texel, face);
}

// Face indices and texel axes follow the Metal cube sampling convention.
static float3 environmentCubeDirection(uint3 texel, uint size) {
    float2 uv = (float2(texel.xy) + 0.5f) * (2.0f / float(size)) - 1.0f;
    switch (texel.z) {
        case 0: return normalize(float3(1, -uv.y, -uv.x));
        case 1: return normalize(float3(-1, -uv.y, uv.x));
        case 2: return normalize(float3(uv.x, 1, uv.y));
        case 3: return normalize(float3(uv.x, -1, -uv.y));
        case 4: return normalize(float3(uv.x, -uv.y, 1));
        default: return normalize(float3(-uv.x, -uv.y, -1));
    }
}

static float3 environmentPanorama(
    texture2d<float, access::read> source,
    float3 direction
) {
    int width = int(source.get_width());
    int height = int(source.get_height());
    float2 uv = float2(
        atan2(direction.z, direction.x) / (2.0f * environmentPi) + 0.5f,
        acos(clamp(direction.y, -1.0f, 1.0f)) / environmentPi
    );
    float2 location = uv * float2(width, height) - 0.5f;
    int2 corner = int2(floor(location));
    float2 f = fract(location);
    int x0 = (corner.x + width) % width;
    int x1 = (x0 + 1) % width;
    int y0 = clamp(corner.y, 0, height - 1);
    int y1 = clamp(corner.y + 1, 0, height - 1);
    float3 north = mix(source.read(uint2(x0, y0)).rgb, source.read(uint2(x1, y0)).rgb, f.x);
    float3 south = mix(source.read(uint2(x0, y1)).rgb, source.read(uint2(x1, y1)).rgb, f.x);
    float3 color = mix(north, south, f.y);
    return clamp(select(float3(0), color, isfinite(color)), 0.0f, 65504.0f);
}

kernel void frostfield_environment_sky(
    texture2d<float, access::read> source [[texture(0)]],
    texturecube<half, access::write> output [[texture(1)]],
    uint3 texel [[thread_position_in_grid]]
) {
    uint size = output.get_width();
    if (any(texel.xy >= size) || texel.z >= 6) {
        return;
    }
    float3 color = environmentPanorama(source, environmentCubeDirection(texel, size));
    output.write(half4(half3(color), 1), texel.xy, texel.z);
}

kernel void frostfield_environment_cube(
    texturecube<float> source [[texture(0)]],
    texturecube<half, access::write> output [[texture(1)]],
    uint3 texel [[thread_position_in_grid]]
) {
    uint size = output.get_width();
    if (any(texel.xy >= size) || texel.z >= 6) {
        return;
    }
    constexpr sampler radiance(filter::linear);
    float3 color = source.sample(radiance, environmentCubeDirection(texel, size), level(0)).rgb;
    color = clamp(select(float3(0), color, isfinite(color)), 0.0f, 65504.0f);
    output.write(half4(half3(color), 1), texel.xy, texel.z);
}

// Inverse of environmentPanorama's lookup, sampled at each panorama texel center.
kernel void frostfield_environment_panorama(
    texturecube<float> sky [[texture(0)]],
    device half4* pixels [[buffer(0)]],
    constant uint2& size [[buffer(1)]],
    uint2 texel [[thread_position_in_grid]]
) {
    if (any(texel >= size)) { return; }
    constexpr sampler radiance(filter::linear);
    float2 uv = (float2(texel) + 0.5f) / float2(size);
    float phi = (uv.x - 0.5f) * (2.0f * environmentPi);
    float theta = uv.y * environmentPi;
    float radius = sin(theta);
    float3 direction(radius * cos(phi), cos(theta), radius * sin(phi));
    float3 color = sky.sample(radiance, direction, level(0)).rgb;
    color = clamp(select(float3(0), color, isfinite(color)), 0.0f, 65504.0f);
    pixels[texel.y * size.x + texel.x] = half4(half3(color), 1.0h);
}

static float2 environmentSequence(uint index, uint count) {
    return float2((float(index) + 0.5f) / float(count), float(reverse_bits(index)) * 2.3283064365386963e-10f);
}

static float3x3 environmentBasis(float3 normal) {
    float3 axis = abs(normal.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(axis, normal));
    return float3x3(tangent, cross(normal, tangent), normal);
}

static float3 environmentHemisphere(float azimuth, float cosine) {
    float sine = sqrt(max(1.0f - cosine * cosine, 0.0f));
    return float3(cos(azimuth) * sine, sin(azimuth) * sine, cosine);
}

static float3 environmentGGXHalfVector(float2 point, float alphaSquared) {
    float cosine = sqrt((1.0f - point.y) / (1.0f + (alphaSquared - 1.0f) * point.y));
    return environmentHemisphere(2.0f * environmentPi * point.x, cosine);
}

static float environmentSampleLOD(float pdf, uint count, uint sourceSize) {
    float texelSolidAngle = 4.0f * environmentPi / (6.0f * float(sourceSize * sourceSize));
    float sampleSolidAngle = 1.0f / max(float(count) * pdf, 0.000001f);
    return max(0.0f, 0.5f * log2(sampleSolidAngle / texelSolidAngle));
}

kernel void frostfield_environment_diffuse(
    texturecube<half> source [[texture(0)]],
    texturecube<half, access::write> output [[texture(1)]],
    constant uint& sampleCount [[buffer(0)]],
    uint3 texel [[thread_position_in_grid]]
) {
    uint size = output.get_width();
    if (any(texel.xy >= size) || texel.z >= 6) {
        return;
    }

    constexpr sampler radiance(filter::linear, mip_filter::linear);
    float3x3 basis = environmentBasis(environmentCubeDirection(texel, size));
    float3 sum = float3(0);
    for (uint i = 0; i < sampleCount; ++i) {
        float2 point = environmentSequence(i, sampleCount);
        float cosine = sqrt(1.0f - point.y);
        float3 direction = basis * environmentHemisphere(2.0f * environmentPi * point.x, cosine);
        float lod = environmentSampleLOD(cosine / environmentPi, sampleCount, source.get_width());
        sum += float3(source.sample(radiance, direction, level(lod)).rgb);
    }
    // With cosine-weighted sampling this average is irradiance divided by pi.
    output.write(half4(half3(sum / float(sampleCount)), 1), texel.xy, texel.z);
}

kernel void frostfield_environment_specular(
    texturecube<half> source [[texture(0)]],
    texturecube<half, access::write> output [[texture(1)]],
    constant uint4& parameters [[buffer(0)]],
    uint3 texel [[thread_position_in_grid]]
) {
    uint size = parameters.x;
    uint mip = parameters.y;
    uint sampleCount = parameters.w;
    if (any(texel.xy >= size) || texel.z >= 6) {
        return;
    }

    float roughness = float(mip) / float(parameters.z - 1);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float3 normal = environmentCubeDirection(texel, size);
    float3x3 basis = environmentBasis(normal);
    constexpr sampler radiance(filter::linear, mip_filter::linear);
    float3 sum = float3(0);
    float weight = 0.0f;
    for (uint i = 0; i < sampleCount; ++i) {
        float3 halfVector = basis * environmentGGXHalfVector(environmentSequence(i, sampleCount), alphaSquared);
        float noH = max(dot(normal, halfVector), 0.0f);
        float3 light = 2.0f * noH * halfVector - normal;
        float noL = dot(normal, light);
        if (noL > 0) {
            // With V=N, the GGX half-vector PDF converts to D/4 for L.
            float term = noH * noH * (alphaSquared - 1.0f) + 1.0f;
            float pdf = alphaSquared / (4.0f * environmentPi * term * term);
            float lod = environmentSampleLOD(pdf, sampleCount, source.get_width());
            sum += float3(source.sample(radiance, light, level(lod)).rgb) * noL;
            weight += noL;
        }
    }
    float3 color = sum / max(weight, 0.000001f);
    output.write(half4(half3(color), 1), texel.xy, texel.z, mip);
}

static float environmentSmithVisibility(float noV, float noL, float alphaSquared) {
    float view = noL * sqrt(noV * noV * (1.0f - alphaSquared) + alphaSquared);
    float light = noV * sqrt(noL * noL * (1.0f - alphaSquared) + alphaSquared);
    return 0.5f / max(view + light, 0.000001f);
}

kernel void frostfield_environment_brdf(
    texture2d<half, access::write> output [[texture(0)]],
    constant uint& sampleCount [[buffer(0)]],
    uint2 texel [[thread_position_in_grid]]
) {
    uint2 size(output.get_width(), output.get_height());
    if (any(texel >= size)) {
        return;
    }
    float2 uv = (float2(texel) + 0.5f) / float2(size);
    float noV = uv.x;
    float alpha = max(uv.y * uv.y, 0.0001f);
    float alphaSquared = alpha * alpha;
    float3 view = float3(sqrt(max(1.0f - noV * noV, 0.0f)), 0, noV);
    float2 integrated = float2(0);
    for (uint i = 0; i < sampleCount; ++i) {
        float3 halfVector = environmentGGXHalfVector(environmentSequence(i, sampleCount), alphaSquared);
        float voH = max(dot(view, halfVector), 0.0f);
        float noL = (2.0f * voH * halfVector - view).z;
        if (noL > 0) {
            float visibility = environmentSmithVisibility(noV, noL, alphaSquared);
            float weight = 4.0f * noL * voH * visibility / max(halfVector.z, 0.000001f);
            float edge = 1.0f - voH;
            float fresnel = edge * edge * edge * edge * edge;
            integrated += weight * float2(1.0f - fresnel, fresnel);
        }
    }
    output.write(half4(half2(integrated / float(sampleCount)), 0, 1), texel);
}
