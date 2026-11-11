import Foundation
internal import FrostfieldDraco

/// Decodes standalone Draco triangle meshes and the triangle payloads used by
/// `KHR_draco_mesh_compression`. Decoding runs locally on the caller's thread.
public enum DracoDecoder {
    /// The order and kinds of named attributes stored in Draco.
    public enum Semantic: UInt32, Sendable {
        case position, normal, color, textureCoordinate, generic
    }

    /// A scalar in the decoded buffer. Multibyte values use little-endian order.
    public enum Component: UInt32, Sendable {
        case int8 = 1, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64, bool

        public var byteWidth: Int {
            switch self {
            case .int8, .uint8, .bool: 1
            case .int16, .uint16: 2
            case .int32, .uint32, .float32: 4
            case .int64, .uint64, .float64: 8
            }
        }
    }

    public struct Attribute: Sendable {
        /// The stream's attribute ID, independent of its position in the list.
        /// Legacy standalone streams can repeat IDs.
        public let id: UInt32
        public let semantic: Semantic
        public let component: Component
        public let components: Int
        public let normalized: Bool
        /// Tightly packed components in decoded point order, before normalizing integer values.
        public let data: Data
    }

    public struct Mesh: Sendable {
        public let pointCount: Int
        /// Triangle list in decoded point order.
        public let indices: [UInt32]
        public let attributes: [Attribute]
    }

    /// Reads a mesh stream using Google's native Draco decoder. Limits input to
    /// 64 MiB, one million points and indices, and 32 attributes. The combined
    /// materialized attribute buffers and indices may occupy up to 64 MiB.
    public static func decode(_ data: Data) throws -> Mesh {
        var decoded: OpaquePointer?
        let status = data.withUnsafeBytes { ff_draco_decode($0.baseAddress, $0.count, &decoded) }
        switch status {
        case FFDracoStatusOK: break
        case FFDracoStatusUnsupportedGeometry: throw Failure.unsupportedGeometry
        case FFDracoStatusSizeLimit: throw Failure.sizeLimit
        default: throw Failure.invalidBitstream
        }
        guard let decoded else { throw Failure.invalidBitstream }
        defer { ff_draco_release(decoded) }

        let pointCount = ff_draco_point_count(decoded)
        let indexCount = ff_draco_index_count(decoded)
        var remainingBytes = 64 * 1_024 * 1_024 - indexCount * MemoryLayout<UInt32>.stride
        var indices = [UInt32](repeating: 0, count: indexCount)
        let copied = indices.withUnsafeMutableBufferPointer { ff_draco_copy_indices(decoded, $0.baseAddress, $0.count) }
        guard copied else { throw Failure.invalidBitstream }

        var attributes: [Attribute] = []
        for index in 0..<ff_draco_attribute_count(decoded) {
            var info = FFDracoAttribute()
            guard ff_draco_attribute(decoded, index, &info),
                  let semantic = Semantic(rawValue: info.semantic),
                  let component = Component(rawValue: info.component),
                  info.width == component.byteWidth else { throw Failure.invalidBitstream }
            let count = pointCount * info.components * info.width
            guard count <= remainingBytes else { throw Failure.sizeLimit }
            remainingBytes -= count
            var storage = Data(count: count)
            let copied = storage.withUnsafeMutableBytes {
                ff_draco_copy_attribute(decoded, index, $0.baseAddress, $0.count)
            }
            guard copied else { throw Failure.invalidBitstream }
            attributes.append(Attribute(
                id: info.id, semantic: semantic, component: component, components: info.components,
                normalized: info.normalized, data: storage
            ))
        }
        guard attributes.contains(where: { $0.semantic == .position }) else { throw Failure.invalidBitstream }
        return Mesh(pointCount: pointCount, indices: indices, attributes: attributes)
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidBitstream
        case unsupportedGeometry
        case sizeLimit

        public var errorDescription: String? {
            switch self {
            case .invalidBitstream: "Draco mesh data is malformed or exceeds a decoder layout limit."
            case .unsupportedGeometry: "The Draco payload must contain a triangle mesh."
            case .sizeLimit: "Draco mesh data exceeds the decoder's byte limit."
            }
        }
    }
}
