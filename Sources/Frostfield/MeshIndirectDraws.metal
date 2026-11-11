#include <metal_stdlib>
using namespace metal;

// Matches MTLDrawIndexedPrimitivesIndirectArguments.
struct FrostfieldIndexedArguments {
    uint indexCount;
    uint instanceCount;
    uint indexStart;
    int baseVertex;
    uint baseInstance;
};
static_assert(sizeof(FrostfieldIndexedArguments) == 20, "Metal indexed argument stride");

static uint2 frostfield_merge_index_bounds(uint2 a, uint2 b) {
    return uint2(min(a.x, b.x), max(a.y, b.y));
}

static uint2 frostfield_indirect_index_bounds(
    uint start, uint end, device const uint *indices, device const uint2 *tree, uint2 layout
) {
    const uint firstLeaf = layout.x;
    const uint blockSize = layout.y;
    const uint firstBlock = start / blockSize + uint(start % blockSize != 0);
    const uint lastBlock = end / blockSize;
    uint2 bounds = uint2(0xffffffffu, 0u);

    // Read individual indices only on the two partial edges. The aligned
    // interior needs at most two cached bounds per level of the tree.
    if (firstBlock < lastBlock) {
        uint left = firstLeaf + firstBlock;
        uint right = firstLeaf + lastBlock;
        while (left < right) {
            if ((left & 1u) != 0) bounds = frostfield_merge_index_bounds(bounds, tree[left++]);
            if ((right & 1u) != 0) bounds = frostfield_merge_index_bounds(bounds, tree[--right]);
            left /= 2;
            right /= 2;
        }
        for (uint i = lastBlock * blockSize; i < end; ++i) {
            bounds = frostfield_merge_index_bounds(bounds, uint2(indices[i]));
        }
        end = firstBlock * blockSize;
    }
    for (uint i = start; i < end; ++i) {
        bounds = frostfield_merge_index_bounds(bounds, uint2(indices[i]));
    }
    return bounds;
}

kernel void frostfield_validate_mesh_indirect(
    device const uchar *input [[buffer(0)]],
    constant ulong *byteOffsets [[buffer(1)]],
    device FrostfieldIndexedArguments *output [[buffer(2)]],
    constant uint4 &limits [[buffer(3)]],
    device const uint *indices [[buffer(4)]],
    device const uint2 *tree [[buffer(5)]],
    constant uint2 &layout [[buffer(6)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= limits.z) return;
    const FrostfieldIndexedArguments args =
        *reinterpret_cast<device const FrostfieldIndexedArguments *>(input + byteOffsets[slot]);
    if (args.indexCount == 0 || args.instanceCount == 0
        || args.indexStart > limits.x || args.indexCount > limits.x - args.indexStart
        || args.baseInstance > limits.y || args.instanceCount > limits.y - args.baseInstance) {
        output[slot] = {};
        return;
    }
    // The prepared mesh already validates every raw index for base zero.
    if (args.baseVertex != 0) {
        const uint2 bounds = frostfield_indirect_index_bounds(
            args.indexStart, args.indexStart + args.indexCount, indices, tree, layout
        );
        const long base = long(args.baseVertex);
        if (long(bounds.x) + base < 0 || long(bounds.y) + base >= long(limits.w)) {
            output[slot] = {};
            return;
        }
    }
    output[slot] = args;
}
