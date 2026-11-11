#include <metal_stdlib>
using namespace metal;

constant bool scalarVolumeHasDepth [[function_constant(0)]];

struct ScalarVolumeVaryings {
    float4 position [[position]];
    float2 uv;
};

struct ScalarVolumeUniforms {
    float4x4 voxelFromClip;
    float4x4 worldFromVoxel;
    float4x4 voxelFromWorld;
    float4 scalar;
    float4 ray;
    uint4 flags;
};

vertex ScalarVolumeVaryings frostfield_scalar_volume_vertex(uint id [[vertex_id]]) {
    float2 uv(id == 1 ? 2.0f : 0.0f, id == 2 ? 2.0f : 0.0f);
    return { float4(uv * float2(2, -2) + float2(-1, 1), 0, 1), uv };
}

static bool scalarVolumeFinite(float value) {
    return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

static float3 scalarVolumeUnproject(float2 ndc, float depth, float4x4 voxelFromClip) {
    float4 point = voxelFromClip * float4(ndc, depth, 1);
    return point.xyz / point.w;
}

static bool scalarVolumeBounds(
    float3 origin, float3 direction, float3 size, thread float& start, thread float& end
) {
    for (uint axis = 0; axis < 3; ++axis) {
        float low = -0.5f;
        float high = size[axis] - 0.5f;
        float component = direction[axis];
        if (component == 0.0f) {
            if (origin[axis] < low || origin[axis] > high) return false;
        } else {
            float a = (low - origin[axis]) / component;
            float b = (high - origin[axis]) / component;
            start = max(start, min(a, b));
            end = min(end, max(a, b));
        }
    }
    return end > start;
}

static float scalarVolumeSample(
    texture3d<float, access::read> texture, float3 point, float3 size, bool linear
) {
    float3 voxel = clamp(point, float3(0), size - 1);
    if (!linear) return texture.read(uint3(floor(voxel + 0.5f))).x;

    // Manual filtering also works when R32Float linear sampling is unavailable.
    uint3 a = uint3(floor(voxel));
    uint3 b = min(a + 1, uint3(size) - 1);
    float3 f = fract(voxel);
    float x00 = mix(texture.read(a).x, texture.read(uint3(b.x, a.yz)).x, f.x);
    float x10 = mix(texture.read(uint3(a.x, b.y, a.z)).x, texture.read(uint3(b.xy, a.z)).x, f.x);
    float x01 = mix(texture.read(uint3(a.xy, b.z)).x, texture.read(uint3(b.x, a.y, b.z)).x, f.x);
    float x11 = mix(texture.read(uint3(a.x, b.yz)).x, texture.read(b).x, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

static float scalarVolumeMaximum(
    texture3d<float, access::read> texture, float3 origin, float3 direction, float3 size,
    float start, float end, uint steps, bool linear
) {
    float step = (end - start) / float(steps);
    float best = as_type<float>(0x7fc00000u);
    float bestAt = start;
    for (uint index = 0; index < steps; ++index) {
        float at = start + (float(index) + 0.5f) * step;
        float value = scalarVolumeSample(texture, origin + direction * at, size, linear);
        if (scalarVolumeFinite(value) && (!scalarVolumeFinite(best) || value > best)) {
            best = value;
            bestAt = at;
        }
    }
    if (scalarVolumeFinite(best)) {
        // Search either side of the best coarse sample within the clipped ray.
        for (int index = -2; index <= 2; ++index) {
            if (index == 0) continue;
            float at = clamp(bestAt + float(index) * step * 0.25f, start, end);
            float value = scalarVolumeSample(texture, origin + direction * at, size, linear);
            if (scalarVolumeFinite(value)) best = max(best, value);
        }
    }
    return best;
}

static bool scalarVolumeSurface(
    texture3d<float, access::read> texture, float3 origin, float3 direction, float3 size,
    float start, float end, uint steps, bool linear, float threshold, thread float& hit
) {
    float previous = scalarVolumeSample(texture, origin + direction * start, size, linear);
    // A solid reaching the front edge has a surface on the volume's boundary.
    if (start > 0 && scalarVolumeFinite(previous) && previous >= threshold) {
        hit = start;
        return true;
    }
    float previousAt = start;
    for (uint index = 1; index <= steps; ++index) {
        float at = mix(start, end, float(index) / float(steps));
        float value = scalarVolumeSample(texture, origin + direction * at, size, linear);
        bool high = previous >= threshold;
        if (scalarVolumeFinite(previous) && scalarVolumeFinite(value) && high != (value >= threshold)) {
            float a = previousAt;
            float b = at;
            for (uint refinement = 0; refinement < 6; ++refinement) {
                float mid = (a + b) * 0.5f;
                float sample = scalarVolumeSample(texture, origin + direction * mid, size, linear);
                if (!scalarVolumeFinite(sample)) break;
                if ((sample >= threshold) == high) a = mid;
                else b = mid;
            }
            hit = (a + b) * 0.5f;
            return true;
        }
        previous = value;
        previousAt = at;
    }
    return false;
}

static float scalarVolumeLight(
    texture3d<float, access::read> texture, float3 point, float3 direction, float3 size,
    bool linear, constant ScalarVolumeUniforms& u
) {
    float3 gradient;
    for (uint axis = 0; axis < 3; ++axis) {
        float3 offset(0);
        offset[axis] = 0.75f;
        gradient[axis] = scalarVolumeSample(texture, point + offset, size, linear)
            - scalarVolumeSample(texture, point - offset, size, linear);
    }
    float3x3 voxelFromWorld(u.voxelFromWorld[0].xyz, u.voxelFromWorld[1].xyz, u.voxelFromWorld[2].xyz);
    float3 normal = transpose(voxelFromWorld) * gradient;
    float lengthSquared = dot(normal, normal);
    if (!scalarVolumeFinite(lengthSquared) || lengthSquared <= 0) return 1;

    float3x3 worldFromVoxel(u.worldFromVoxel[0].xyz, u.worldFromVoxel[1].xyz, u.worldFromVoxel[2].xyz);
    float3 ray = normalize(worldFromVoxel * direction);
    float diffuse = abs(dot(normal * rsqrt(lengthSquared), ray));
    return scalarVolumeFinite(diffuse) ? 0.25f + 0.75f * saturate(diffuse) : 1;
}

fragment float4 frostfield_scalar_volume_fragment(
    ScalarVolumeVaryings in [[stage_in]],
    texture3d<float, access::read> texture [[texture(0)]],
    texture2d<float, access::read> colors [[texture(1)]],
    depth2d<float, access::read> depth [[texture(2), function_constant(scalarVolumeHasDepth)]],
    constant ScalarVolumeUniforms& u [[buffer(0)]]
) {
    if (u.scalar.w == 0) discard_fragment();
    float2 ndc = in.uv * float2(2, -2) + float2(-1, 1);
    float3 origin = scalarVolumeUnproject(ndc, u.ray.x, u.voxelFromClip);
    float3 delta = scalarVolumeUnproject(ndc, u.ray.y, u.voxelFromClip) - origin;
    float length = metal::length(delta);
    if (!scalarVolumeFinite(length) || length <= 0) discard_fragment();

    float3 direction = delta / length;
    float start = 0;
    float end = u.ray.w != 0 ? INFINITY : length;
    if (scalarVolumeHasDepth) {
        float z = depth.read(uint2(in.position.xy));
        if (scalarVolumeFinite(z) && z >= 0 && z <= 1 && z != 1 - u.ray.x) {
            float3 opaque = scalarVolumeUnproject(ndc, z, u.voxelFromClip);
            float limit = dot(opaque - origin, direction);
            if (scalarVolumeFinite(limit)) end = min(end, limit);
        }
    }

    float3 size(texture.get_width(), texture.get_height(), texture.get_depth());
    if (!scalarVolumeBounds(origin, direction, size, start, end)) discard_fragment();

    float3 extent = abs(direction * (end - start));
    float lengthInVoxels = max(extent.x, max(extent.y, extent.z));
    uint steps = uint(clamp(ceil(lengthInVoxels * u.ray.z), 1.0f, float(u.flags.x)));
    bool linear = u.flags.z != 0;

    float scalar;
    float light = 1;
    if (u.flags.y == 0) {
        scalar = scalarVolumeMaximum(texture, origin, direction, size, start, end, steps, linear);
        if (!scalarVolumeFinite(scalar) || scalar < u.scalar.x) discard_fragment();
    } else {
        float hit;
        scalar = u.scalar.z;
        if (!scalarVolumeSurface(texture, origin, direction, size, start, end, steps, linear, scalar, hit)) {
            discard_fragment();
        }
        light = scalarVolumeLight(texture, origin + direction * hit, direction, size, linear, u);
    }
    float level = (clamp(scalar, u.scalar.x, u.scalar.y) - u.scalar.x) / (u.scalar.y - u.scalar.x);
    uint colorIndex = uint(floor(saturate(level) * float(colors.get_width() - 1) + 0.5f));
    float3 color = colors.read(uint2(colorIndex, 0)).rgb;
    return float4(color * (light * u.scalar.w), u.scalar.w);
}
