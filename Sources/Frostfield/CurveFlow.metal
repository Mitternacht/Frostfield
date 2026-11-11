#include <metal_stdlib>
using namespace metal;

// Layout shared with PackedMeshVertex, CurveFlowMetal.Interval and Placement.
struct FrostfieldFlowVertex {
    float4 position;
    float4 normal;
    float4 uv;
    float4 tangent;
};

struct FrostfieldFlowInterval {
    float4 centerStart;
    float4 deltaEnd;
    float4 tangentSpan;
    float4 nextTangent;
    float4 rotation;
    float4 axisAngle;
};

struct FrostfieldFlowPlacement {
    uint first;
    uint count;
    uint closed;
    float offset;
};

struct FrostfieldFlowGrid {
    uint sourceCount;
    uint outputCount;
};

static float4 frostfield_flow_quaternion_product(float4 a, float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

static float3 frostfield_flow_rotate(float4 q, float3 vector) {
    const float3 first = cross(q.xyz, vector);
    return vector + 2.0f * (q.w * first + cross(q.xyz, first));
}

static float3 frostfield_flow_unit(float3 direction) {
    const float3 magnitudes = abs(direction);
    return normalize(direction / max(magnitudes.x, max(magnitudes.y, magnitudes.z)));
}

kernel void frostfield_curve_flow_write(
    device const FrostfieldFlowVertex *source [[buffer(0)]],
    device const FrostfieldFlowInterval *intervals [[buffer(1)]],
    device const FrostfieldFlowPlacement *placements [[buffer(2)]],
    device FrostfieldFlowVertex *output [[buffer(3)]],
    constant FrostfieldFlowGrid &grid [[buffer(4)]],
    constant float &distanceScale [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= grid.outputCount) return;
    const FrostfieldFlowVertex input = source[index % grid.sourceCount];
    const FrostfieldFlowPlacement placement = placements[index / grid.sourceCount];
    uint low = placement.first;
    uint high = low + placement.count - 1;
    const float length = intervals[high].deltaEnd.w;
    float distance = placement.offset + input.position.x;
    if (placement.closed != 0) {
        distance = fmod(distance, length);
        if (distance < 0.0f) distance += length;
    }

    const float extension = distance - clamp(distance, 0.0f, length);
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (intervals[middle].deltaEnd.w < distance) low = middle + 1;
        else high = middle;
    }

    const FrostfieldFlowInterval frame = intervals[low];
    const float span = frame.tangentSpan.w;
    const float t = clamp((distance - frame.centerStart.w) / (frame.deltaEnd.w - frame.centerStart.w), 0.0f, 1.0f);
    const float t2 = t * t;
    const float3 a = frame.tangentSpan.xyz;
    const float3 b = frame.nextTangent.xyz;
    const float3 delta = frame.deltaEnd.xyz;
    const float3 center = frame.centerStart.xyz + (3.0f * t2 - 2.0f * t2 * t) * delta
        + (t2 * t - 2.0f * t2 + t) * span * a + (t2 * t - t2) * span * b;
    const float3 derivative = (6.0f * t - 6.0f * t2) * (delta / span)
        + (3.0f * t2 - 4.0f * t + 1.0f) * a + (3.0f * t2 - 2.0f * t) * b;
    const float halfTurn = frame.axisAngle.w * t * 0.5f;
    const float4 rotation = frostfield_flow_quaternion_product(
        float4(sin(halfTurn) * frame.axisAngle.xyz, cos(halfTurn)), frame.rotation
    );
    const float3 x = frostfield_flow_rotate(rotation, float3(1, 0, 0));
    const float3 y = frostfield_flow_rotate(rotation, float3(0, 1, 0));
    const float3 z = frostfield_flow_rotate(rotation, float3(0, 0, 1));
    const float3 side = input.position.y * y + input.position.z * z;
    const float3 omega = extension == 0.0f ? frame.axisAngle.xyz * (frame.axisAngle.w / span) : float3(0);
    const float3 along = distanceScale * (derivative + cross(omega, side));
    const float3 normalSource = frostfield_flow_unit(input.normal.xyz);
    const float3 normal = frostfield_flow_unit(
        normalSource.x * x + normalSource.y * cross(z, along) + normalSource.z * cross(along, y)
    );
    float4 tangent = input.tangent;
    if (tangent.w != 0.0f) {
        const float3 direction = frostfield_flow_unit(tangent.x * along + tangent.y * y + tangent.z * z);
        tangent.xyz = frostfield_flow_unit(direction - normal * dot(direction, normal));
    }

    output[index] = {
        float4(center + extension * derivative + side, 1.0f), float4(normal, 0.0f), input.uv, tangent
    };
}
