#ifndef FROSTFIELD_LIGHT_PROBE_H
#define FROSTFIELD_LIGHT_PROBE_H

#include <metal_stdlib>

// Buffer 26. RGB contains scene radiance SH. The first W marks lighting
// state (0 off, 1 probes, 2 ambient only). The next three W lanes hold ambient.
// W lanes 4 and 5 hold the low and high 16 bits of the capture camera layers.
struct FrostfieldLightProbe {
    metal::float4 coefficients[9];
};

static inline metal::float3 frostfieldAmbientIrradiance(constant FrostfieldLightProbe& probe) {
    return metal::float3(probe.coefficients[1].w, probe.coefficients[2].w, probe.coefficients[3].w);
}

static inline uint frostfieldProbeLayers(constant FrostfieldLightProbe& probe) {
    return uint(probe.coefficients[4].w) | (uint(probe.coefficients[5].w) << 16);
}

static inline uint frostfieldProbeGridOffset(metal::uint3 index, metal::uint3 size) {
    return ((index.y * size.z + index.z) * size.x + index.x) * 9;
}

// Fragment buffer 5 starts with the grid count, then three float4 records per
// grid, then signed RGB SH coefficients. Descriptors are ordered by priority.
static inline metal::float3 frostfieldProbeGridIrradiance(
    device const metal::float4* grids, uint layers, metal::float3 position, metal::float3 normal
) {
    using namespace metal;
    uint count = uint(grids[0].x);
    for (uint index = 0; index < count; ++index) {
        device const float4* descriptor = grids + 1 + index * 3;
        float4 extent = descriptor[1], resolution = descriptor[2];
        uint volumeLayers = uint(extent.w) | (uint(resolution.w) << 16);
        if ((volumeLayers & layers) == 0) continue;
        float4 minimum = descriptor[0];
        float3 cell = (position - minimum.xyz) / extent.xyz * resolution.xyz + normal * 0.5f;
        if (!all(cell >= -0.01f) || !all(cell <= resolution.xyz + 0.01f)) continue;

        float3 coordinate = clamp(cell - 0.5f, float3(0.0f), resolution.xyz - 1.0f);
        uint3 size = uint3(resolution.xyz);
        uint3 lower = uint3(coordinate);
        uint3 upper = min(lower + 1u, size - 1u);
        float3 fraction = coordinate - float3(lower);
        uint offsets[8] = {
            frostfieldProbeGridOffset(lower, size),
            frostfieldProbeGridOffset(uint3(upper.x, lower.y, lower.z), size),
            frostfieldProbeGridOffset(uint3(lower.x, lower.y, upper.z), size),
            frostfieldProbeGridOffset(uint3(upper.x, lower.y, upper.z), size),
            frostfieldProbeGridOffset(uint3(lower.x, upper.y, lower.z), size),
            frostfieldProbeGridOffset(uint3(upper.x, upper.y, lower.z), size),
            frostfieldProbeGridOffset(uint3(lower.x, upper.y, upper.z), size),
            frostfieldProbeGridOffset(upper, size)
        };
        float x = normal.x, y = normal.y, z = normal.z;
        float basis[9] = {
            0.886227f, 1.023328f * y, 1.023328f * z, 1.023328f * x,
            0.858086f * x * y, 0.858086f * y * z, 0.743125f * z * z - 0.247708f,
            0.858086f * x * z, 0.429043f * (x * x - y * y)
        };
        device const float4* coefficients = grids + uint(minimum.w);
        float3 irradiance = float3(0);
        for (uint coefficient = 0; coefficient < 9; ++coefficient) {
            device const float4* values = coefficients + coefficient;
            float3 lowerY = mix(mix(values[offsets[0]].xyz, values[offsets[1]].xyz, fraction.x),
                                mix(values[offsets[2]].xyz, values[offsets[3]].xyz, fraction.x), fraction.z);
            float3 upperY = mix(mix(values[offsets[4]].xyz, values[offsets[5]].xyz, fraction.x),
                                mix(values[offsets[6]].xyz, values[offsets[7]].xyz, fraction.x), fraction.z);
            irradiance += mix(lowerY, upperY, fraction.y) * basis[coefficient];
        }
        return max(irradiance, 0.0f);
    }
    return metal::float3(0);
}

#endif
