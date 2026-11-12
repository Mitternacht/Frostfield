#include <metal_stdlib>
using namespace metal;

// Matches MTLDrawPrimitivesIndirectArguments.
struct FrostfieldPrimitiveArguments {
    uint vertexCount;
    uint instanceCount;
    uint vertexStart;
    uint baseInstance;
};
static_assert(sizeof(FrostfieldPrimitiveArguments) == 16, "Metal primitive argument stride");

kernel void frostfield_validate_primitive_indirect(
    device const uchar *input [[buffer(0)]],
    constant ulong *byteOffsets [[buffer(1)]],
    device FrostfieldPrimitiveArguments *output [[buffer(2)]],
    constant uint3 &limits [[buffer(3)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= limits.z) return;
    const FrostfieldPrimitiveArguments args =
        *reinterpret_cast<device const FrostfieldPrimitiveArguments *>(input + byteOffsets[slot]);
    // The prepared primitive owns the topology. A record may draw fewer instances,
    // but a different vertex count, start or base would address vertices this draw did not bind.
    if (args.vertexCount != limits.x || args.vertexStart != 0 || args.baseInstance != 0
        || args.instanceCount == 0 || args.instanceCount > limits.y) {
        output[slot] = {};
        return;
    }
    output[slot] = args;
}
