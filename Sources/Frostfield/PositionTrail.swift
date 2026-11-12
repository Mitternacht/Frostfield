import Foundation
import simd

/// A ribbon behind a moving point, like drei `<Trail>`.
///
/// The width lies across the path. A nearly vertical step uses world +X as the
/// up axis; every other step uses world +Y. One point has no ribbon yet.
public struct PositionTrail: Sendable {
    public let width: Float
    public let capacity: Int
    public private(set) var positions: [SIMD3<Float>] = []

    public init(width: Float, capacity: Int) throws {
        guard width.isFinite, width > 0 else { throw Failure.invalidWidth }
        guard (2...65_536).contains(capacity) else { throw Failure.invalidCapacity }
        self.width = width
        self.capacity = capacity
    }

    /// Records a finite position. A repeat of the last position is ignored.
    public mutating func append(_ position: SIMD3<Float>) throws {
        guard boundsFinite(position) else { throw Failure.invalidPosition }
        guard positions.count < capacity else { throw Failure.capacity }
        if positions.last == position { return }
        positions.append(position)
    }

    /// The ribbon, or nil until there are two distinct positions.
    public func ribbon() throws -> MeshGeometry? {
        guard positions.count >= 2 else { return nil }
        let half = Double(width) / 2
        var sides: [SIMD3<Double>] = []
        sides.reserveCapacity(positions.count)
        for index in positions.indices {
            let previous = index == 0 ? positions[0] : positions[index - 1]
            let next = index == positions.count - 1 ? positions[index] : positions[index + 1]
            let step = SIMD3<Double>(next - previous)
            let direction = simd_normalize(step)
            let up: SIMD3<Double> = abs(direction.y) > 0.99 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
            sides.append(simd_normalize(simd_cross(direction, up)))
        }
        var vertices: [MeshVertex] = []
        vertices.reserveCapacity(positions.count * 2)
        for (index, position) in positions.enumerated() {
            let previous = index == 0 ? positions[0] : positions[index - 1]
            let next = index == positions.count - 1 ? positions[index] : positions[index + 1]
            let direction = simd_normalize(SIMD3<Double>(next - previous))
            let center = SIMD3<Double>(position)
            let side = sides[index] * half
            let u = Float(index) / Float(positions.count - 1)
            let left = SIMD3<Float>(center - side)
            let right = SIMD3<Float>(center + side)
            let normal = SIMD3<Float>(simd_normalize(simd_cross(direction, sides[index])))
            vertices.append(MeshVertex(position: left, normal: normal, textureCoordinate: SIMD2(u, 0)))
            vertices.append(MeshVertex(position: right, normal: normal, textureCoordinate: SIMD2(u, 1)))
        }
        var indices: [UInt32] = []
        indices.reserveCapacity((positions.count - 1) * 6)
        for index in 0..<(positions.count - 1) {
            let left = UInt32(index * 2)
            let right = left + 1
            let nextLeft = left + 2
            let nextRight = left + 3
            indices += [left, nextLeft, right, right, nextLeft, nextRight]
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidWidth
        case invalidCapacity
        case invalidPosition
        case capacity

        public var errorDescription: String? {
            switch self {
            case .invalidWidth: "A trail width must be finite and positive."
            case .invalidCapacity: "A trail holds from 2 to 65,536 positions."
            case .invalidPosition: "A trail position must be finite."
            case .capacity: "The trail has no room for another position."
            }
        }
    }
}
