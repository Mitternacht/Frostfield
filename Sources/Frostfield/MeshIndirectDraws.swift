import Foundation
import Metal
import simd

/// An ordered list of indexed triangle draw arguments in a Metal buffer.
///
/// Each offset identifies one `MTLDrawIndexedPrimitivesIndirectArguments`
/// record. The records can come from a compute pass and are validated on the
/// GPU before drawing. A record with a zero count or an out-of-range index
/// or instance draws nothing. The renderer checks each signed base vertex
/// against the indices in that record's selected range.
public struct MeshIndirectDraws {
    public static let maximumCount = 512
    /// Maximum number of argument records supplied to one scene graph encode.
    public static let maximumSceneCount = 16_384
    public static let recordByteCount = MemoryLayout<MTLDrawIndexedPrimitivesIndirectArguments>.stride

    public let buffer: any MTLBuffer
    /// Byte offsets in submission order. An offset can be used more than once.
    public let byteOffsets: [Int]

    /// A device capability check for Metal indirect draw arguments.
    public static func isSupported(on device: any MTLDevice) -> Bool {
        device.supportsFamily(.apple3) || device.supportsFamily(.apple7)
    }

    /// Uses a hazard-tracked buffer. End the producer encoder before encoding
    /// the draw on the same command buffer. For work on a different queue,
    /// synchronize the producer with the consumer before using the buffer.
    public init(buffer: any MTLBuffer, byteOffsets: [Int] = [0]) throws {
        guard (1...Self.maximumCount).contains(byteOffsets.count) else { throw Failure.count }
        guard buffer.hazardTrackingMode != .untracked else { throw Failure.untracked }
        for offset in byteOffsets {
            guard offset >= 0, offset.isMultiple(of: MemoryLayout<UInt32>.alignment),
                  offset <= buffer.length - Self.recordByteCount else { throw Failure.offset(offset) }
        }
        self.buffer = buffer
        self.byteOffsets = byteOffsets
    }

    func validate(commandBuffer: any MTLCommandBuffer) throws {
        guard buffer.device === commandBuffer.device else { throw MeshRenderer.Failure.wrongDevice }
        guard Self.isSupported(on: buffer.device) else { throw Failure.unsupported }
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
    }

    struct Validated {
        let buffer: any MTLBuffer
        let count: Int
    }

    struct Validator {
        private static let blockSize = 64

        let pipeline: any MTLComputePipelineState
        let indices: any MTLBuffer
        let indexCount: UInt32
        let vertexCount: UInt32
        let bounds: any MTLBuffer
        let layout: SIMD2<UInt32>

        init(geometry: MeshGeometry, indices: any MTLBuffer) throws {
            guard let indexCount = UInt32(exactly: geometry.indices.count),
                  let vertexCount = UInt32(exactly: geometry.vertices.count) else { throw Failure.indexLimit }
            let device = indices.device
            self.pipeline = try Self.makePipeline(device: device)
            self.indices = indices
            self.indexCount = indexCount
            self.vertexCount = vertexCount

            let blockCount = (geometry.indices.count + Self.blockSize - 1) / Self.blockSize
            var firstLeaf = 1
            while firstLeaf < blockCount { firstLeaf *= 2 }
            let length = firstLeaf * 2 * MemoryLayout<SIMD2<UInt32>>.stride
            guard length <= device.maxBufferLength else { throw Failure.resource }
            var tree = [SIMD2<UInt32>](repeating: SIMD2(.max, 0), count: firstLeaf * 2)
            for block in 0..<blockCount {
                let start = block * Self.blockSize
                let end = min(start + Self.blockSize, geometry.indices.count)
                var lower = UInt32.max
                var upper: UInt32 = 0
                for value in geometry.indices[start..<end] {
                    lower = min(lower, value)
                    upper = max(upper, value)
                }
                tree[firstLeaf + block] = SIMD2(lower, upper)
            }
            for node in (1..<firstLeaf).reversed() {
                let a = tree[node * 2], b = tree[node * 2 + 1]
                tree[node] = SIMD2(min(a.x, b.x), max(a.y, b.y))
            }
            self.bounds = try tree.withUnsafeBytes { bytes in
                guard let buffer = device.makeBuffer(bytes: bytes.baseAddress!, length: length, options: .storageModeShared) else {
                    throw Failure.resource
                }
                buffer.label = "Frostfield mesh indirect index bounds"
                return buffer
            }
            self.layout = SIMD2(UInt32(firstLeaf), UInt32(Self.blockSize))
        }

        private static func makePipeline(device: any MTLDevice) throws -> any MTLComputePipelineState {
            do {
                let library = try device.makeDefaultLibrary(bundle: .module)
                guard let kernel = library.makeFunction(name: "frostfield_validate_mesh_indirect") else {
                    throw Failure.pipeline("The validation kernel is missing.")
                }
                return try device.makeComputePipelineState(function: kernel)
            } catch let failure as Failure {
                throw failure
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
        }
    }

    func encodeValidation(
        into commandBuffer: any MTLCommandBuffer, validator: Validator, instanceCount: UInt32
    ) throws -> Validated {
        let count = byteOffsets.count
        guard let output = buffer.device.makeBuffer(
            length: count * Self.recordByteCount, options: [.storageModePrivate, .hazardTrackingModeTracked]
        ) else { throw Failure.resource }
        output.label = "Frostfield validated mesh indirect arguments"
        guard let compute = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        compute.label = "Frostfield validate mesh indirect arguments"
        compute.setComputePipelineState(validator.pipeline)
        compute.setBuffer(buffer, offset: 0, index: 0)
        let offsets = byteOffsets.map(UInt64.init)
        offsets.withUnsafeBytes { compute.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        compute.setBuffer(output, offset: 0, index: 2)
        var limits = SIMD4(validator.indexCount, instanceCount, UInt32(count), validator.vertexCount)
        compute.setBytes(&limits, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
        compute.setBuffer(validator.indices, offset: 0, index: 4)
        compute.setBuffer(validator.bounds, offset: 0, index: 5)
        var layout = validator.layout
        compute.setBytes(&layout, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 6)
        let width = validator.pipeline.threadExecutionWidth
        compute.dispatchThreadgroups(
            MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        compute.endEncoding()
        return Validated(buffer: output, count: count)
    }

    public enum Failure: LocalizedError, Equatable {
        case count
        case offset(Int)
        case untracked
        case unsupported
        case commandBuffer
        case indexLimit
        case unknownNode(UUID)
        case incompatibleNode(UUID)
        case sceneBudget
        case instancesChanged
        case resource
        case encoder
        case pipeline(String)

        public var errorDescription: String? {
            switch self {
            case .count: "Provide between 1 and \(MeshIndirectDraws.maximumCount) mesh indirect argument offsets."
            case .offset(let offset): "Mesh indirect arguments at byte \(offset) must be four-byte aligned and entirely inside the buffer."
            case .untracked: "Mesh indirect arguments require a hazard-tracked Metal buffer."
            case .unsupported: "This Metal device does not support indexed indirect draw arguments."
            case .commandBuffer: "Use an uncommitted command buffer for mesh indirect draws."
            case .indexLimit: "The mesh vertex or index count exceeds the range of Metal's indirect draw arguments."
            case .unknownNode(let id): "Mesh indirect arguments refer to missing scene node \(id)."
            case .incompatibleNode(let id): "Scene node \(id) needs an ordinary mesh for indirect arguments."
            case .sceneBudget: "The scene exceeds the mesh indirect argument record limit."
            case .instancesChanged: "Finish editing the instance set before encoding its mesh indirect arguments."
            case .resource: "Metal could not allocate mesh indirect validation storage."
            case .encoder: "Metal could not start the mesh indirect validation pass."
            case .pipeline(let reason): "Could not prepare mesh indirect argument validation: \(reason)"
            }
        }
    }
}
