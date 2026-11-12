import Foundation
import Metal

/// Sorts indices by finite float keys on the GPU, in the role of Three.js `BitonicSort`.
///
/// The result is nondecreasing in the keys. Equal keys are not kept in their original
/// order. The sort pads to the next power of two internally. The caller commits the
/// command buffer; a retained buffer keeps the working storage alive until it completes.
public struct BitonicSort {
    public static let maximumCount = RadixSort.maximumCount

    public let device: any MTLDevice
    private let pipeline: any MTLComputePipelineState

    public init(device: any MTLDevice) throws {
        self.device = device
        let library = try device.makeLibrary(source: Self.source, options: nil)
        guard let function = library.makeFunction(name: "frostfield_bitonic_step") else {
            throw Failure.pipeline("The bitonic kernel is missing.")
        }
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Writes `0..<keys.count` into `indices`, ordered by `keys`. `indices` holds
    /// at least `keys.count` tightly packed `UInt32` values and belongs to this device.
    /// Encode before the command buffer is committed.
    public func encode(
        keys: [Float], into indices: any MTLBuffer, ascending: Bool = true, commandBuffer: any MTLCommandBuffer
    ) throws {
        let count = keys.count
        guard count <= Self.maximumCount else { throw Failure.count }
        guard keys.allSatisfy(\.isFinite) else { throw Failure.nonfinite }
        guard indices.device === device, commandBuffer.device === device else { throw Failure.device }
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else { throw Failure.commandBuffer }
        guard indices.length >= count * MemoryLayout<UInt32>.stride else { throw Failure.capacity }
        guard count > 1 else {
            if count == 1 {
                var zero = UInt32(0)
                guard let source = device.makeBuffer(bytes: &zero, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
                      let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
                blit.copy(from: source, sourceOffset: 0, to: indices, destinationOffset: 0, size: MemoryLayout<UInt32>.stride)
                blit.endEncoding()
            }
            return
        }
        let padded = paddedCount(count)
        let keyBuffer = try buffer(bytes: paddedKeys(keys, padded: padded, ascending: ascending), label: "Frostfield bitonic keys")
        let indexBuffer = try buffer(bytes: paddedIndices(count: count, padded: padded), label: "Frostfield bitonic indices")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        encoder.label = "Frostfield bitonic sort"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(keyBuffer, offset: 0, index: 0)
        encoder.setBuffer(indexBuffer, offset: 0, index: 1)
        var direction = ascending ? UInt32(1) : 0
        encoder.setBytes(&direction, length: MemoryLayout<UInt32>.stride, index: 4)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let groups = MTLSize(width: (padded + width - 1) / width, height: 1, depth: 1)
        let threads = MTLSize(width: width, height: 1, depth: 1)
        var stage = 2
        while stage <= padded {
            var pass = stage >> 1
            while pass > 0 {
                var stageValue = UInt32(stage)
                var passValue = UInt32(pass)
                encoder.setBytes(&stageValue, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.setBytes(&passValue, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
                pass >>= 1
            }
            stage <<= 1
        }
        encoder.endEncoding()
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
        blit.label = "Frostfield bitonic sort copy"
        blit.copy(
            from: indexBuffer, sourceOffset: 0, to: indices, destinationOffset: 0,
            size: count * MemoryLayout<UInt32>.stride
        )
        blit.endEncoding()
    }

    public enum Failure: LocalizedError, Equatable {
        case count
        case nonfinite
        case device
        case commandBuffer
        case capacity
        case encoder
        case pipeline(String)

        public var errorDescription: String? {
            switch self {
            case .count: "A bitonic sort accepts at most \(BitonicSort.maximumCount) keys."
            case .nonfinite: "A bitonic sort key must be finite."
            case .device: "Sort the keys with buffers from the same Metal device."
            case .commandBuffer: "Encode a bitonic sort before committing the command buffer."
            case .capacity: "The index buffer is shorter than the key list."
            case .encoder: "Metal could not create the bitonic sort encoder."
            case .pipeline(let detail): "Metal could not compile the bitonic sort. \(detail)"
            }
        }
    }

    private func paddedKeys(_ keys: [Float], padded: Int, ascending: Bool) -> [Float] {
        // Padding sorts after every finite key: +infinity when ascending, -infinity when descending.
        var values = [Float](repeating: ascending ? .infinity : -.infinity, count: padded)
        values.replaceSubrange(0..<keys.count, with: keys)
        return values
    }

    private func paddedIndices(count: Int, padded: Int) -> [UInt32] {
        var values = [UInt32](repeating: .max, count: padded)
        for index in 0..<count { values[index] = UInt32(index) }
        return values
    }

    private func paddedCount(_ count: Int) -> Int {
        var padded = 1
        while padded < count { padded <<= 1 }
        return padded
    }

    private func buffer<T>(bytes: [T], label: String) throws -> any MTLBuffer {
        let buffer = bytes.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }
        guard let buffer else { throw Failure.encoder }
        buffer.label = label
        return buffer
    }

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void frostfield_bitonic_step(
        device float* keys [[buffer(0)]],
        device uint* indices [[buffer(1)]],
        constant uint& stage [[buffer(2)]],
        constant uint& pass [[buffer(3)]],
        constant uint& ascending [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        uint partner = id ^ pass;
        if (partner <= id) { return; }
        float left = keys[id];
        float right = keys[partner];
        bool outOfOrder = ascending != 0u ? left > right : left < right;
        bool reverse = ascending != 0u ? left < right : left > right;
        bool swapValues = (id & stage) == 0u ? outOfOrder : reverse;
        if (!swapValues) { return; }
        keys[id] = right;
        keys[partner] = left;
        uint leftIndex = indices[id];
        indices[id] = indices[partner];
        indices[partner] = leftIndex;
    }
    """
}
