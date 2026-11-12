import Foundation
import Metal
import Synchronization

/// GPU-written draw arguments for one point, line or sprite node.
///
/// Each offset identifies one `MTLDrawPrimitivesIndirectArguments` record.
/// The records can come from a compute pass. Before drawing, a kernel keeps a
/// record only when its vertex count matches the prepared topology, its start
/// and base instance are zero, and its instance count is within the prepared
/// range. Anything else becomes an empty draw. End the producer encoder before
/// the scene encode on the same command buffer.
public struct PrimitiveIndirectDraws {
    public static let maximumCount = 512
    /// Maximum number of argument records supplied to one scene graph encode.
    public static let maximumSceneCount = 16_384
    public static let recordByteCount = MemoryLayout<MTLDrawPrimitivesIndirectArguments>.stride

    public let buffer: any MTLBuffer
    /// Byte offsets in submission order. An offset can be used more than once.
    public let byteOffsets: [Int]

    public static func isSupported(on device: any MTLDevice) -> Bool {
        MeshIndirectDraws.isSupported(on: device)
    }

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

    func encodeValidation(
        into commandBuffer: any MTLCommandBuffer, vertexCount: UInt32, instanceCount: UInt32
    ) throws -> Validated {
        let count = byteOffsets.count
        let device = buffer.device
        guard let output = device.makeBuffer(
            length: count * Self.recordByteCount, options: [.storageModePrivate, .hazardTrackingModeTracked]
        ) else { throw Failure.resource }
        output.label = "Frostfield validated primitive indirect arguments"
        guard let compute = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        compute.label = "Frostfield validate primitive indirect arguments"
        let pipeline = try Self.pipeline(on: device)
        compute.setComputePipelineState(pipeline)
        compute.setBuffer(buffer, offset: 0, index: 0)
        let offsets = byteOffsets.map(UInt64.init)
        offsets.withUnsafeBytes { compute.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        compute.setBuffer(output, offset: 0, index: 2)
        var limits = SIMD3(vertexCount, instanceCount, UInt32(count))
        compute.setBytes(&limits, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
        let width = pipeline.threadExecutionWidth
        compute.dispatchThreadgroups(
            MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        compute.endEncoding()
        return Validated(buffer: output, count: count)
    }

    private static let pipelines = Mutex<[ObjectIdentifier: any MTLComputePipelineState]>([:])

    private static func pipeline(on device: any MTLDevice) throws -> any MTLComputePipelineState {
        let key = ObjectIdentifier(device)
        if let cached = pipelines.withLock({ $0[key] }) { return cached }
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let kernel = library.makeFunction(name: "frostfield_validate_primitive_indirect") else {
                throw Failure.pipeline("The validation kernel is missing.")
            }
            let pipeline = try device.makeComputePipelineState(function: kernel)
            pipelines.withLock { cache in
                if cache[key] == nil { cache[key] = pipeline }
            }
            return pipelines.withLock { $0[key] ?? pipeline }
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
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
        case resource
        case encoder
        case pipeline(String)

        public var errorDescription: String? {
            switch self {
            case .count: "Provide between 1 and \(PrimitiveIndirectDraws.maximumCount) primitive indirect argument offsets."
            case .offset(let offset): "Primitive indirect arguments at byte \(offset) must be four-byte aligned and entirely inside the buffer."
            case .untracked: "Primitive indirect arguments require a hazard-tracked Metal buffer."
            case .unsupported: "This Metal device does not support primitive indirect draw arguments."
            case .commandBuffer: "Use an uncommitted command buffer for primitive indirect draws."
            case .indexLimit: "The primitive count exceeds the range of Metal's indirect draw arguments."
            case .unknownNode(let id): "Primitive indirect arguments refer to missing scene node \(id)."
            case .incompatibleNode(let id): "Scene node \(id) needs points, lines or sprites for indirect arguments."
            case .sceneBudget: "The scene exceeds the primitive indirect argument record limit."
            case .resource: "Metal could not allocate primitive indirect validation storage."
            case .encoder: "Metal could not start the primitive indirect validation pass."
            case .pipeline(let reason): "Could not prepare primitive indirect argument validation: \(reason)"
            }
        }
    }
}
