import Foundation
import simd

/// A regularly sampled scalar field and indexed marching-cubes extractor.
/// Samples are ordered with X varying fastest, then Y, then Z.
public struct MarchingCubes: Sendable {
    public static let maximumAxisSamples = 1_024
    public static let maximumSamples = 2_097_152
    public static let maximumTriangles = 262_144

    public enum Axis: Int, Sendable { case x, y, z }

    public enum Boundary: Sendable {
        /// Keep any intersection with the grid boundary as an open edge.
        case open
        /// Require the exterior to fit in the grid, including any level-aligned boundary.
        case requireClosed
    }

    /// Number of samples, including both endpoints of each axis.
    public let resolution: SIMD3<Int>
    public let bounds: AxisAlignedBox
    public private(set) var values: [Float]
    let axes: [[Double]]
    private(set) var palette: [SIMD3<Float>]?

    public init(resolution: SIMD3<Int>, bounds: AxisAlignedBox, values: [Float]? = nil) throws {
        self.resolution = resolution
        self.bounds = bounds
        axes = try Self.makeAxes(resolution, bounds)
        let count = resolution.x * resolution.y * resolution.z
        if let values {
            try Self.validate(values, count: count)
            self.values = values
        } else {
            self.values = .init(repeating: 0, count: count)
        }
    }

    /// Samples the supplied function in world coordinates.
    public init(
        resolution: SIMD3<Int>, bounds: AxisAlignedBox, sampling: (SIMD3<Float>) throws -> Float
    ) throws {
        self.resolution = resolution
        self.bounds = bounds
        axes = try Self.makeAxes(resolution, bounds)
        var samples: [Float] = []
        samples.reserveCapacity(resolution.x * resolution.y * resolution.z)
        for z in axes[2] {
            for y in axes[1] {
                for x in axes[0] {
                    let value = try sampling(SIMD3(Float(x), Float(y), Float(z)))
                    guard value.isFinite else { throw Failure.sampleValue }
                    samples.append(value)
                }
            }
        }
        values = samples
    }

    public mutating func replaceValues(_ values: [Float]) throws {
        try Self.validate(values, count: self.values.count)
        self.values = values
        palette = nil
    }

    public mutating func reset() {
        values = .init(repeating: 0, count: values.count)
        palette = nil
    }

    /// Adds a metaball with falloff `max(abs(strength) / (1e-6 + distance²) - subtract, 0)`.
    /// The center and distance are in world units. Negative strength carves the field.
    /// An optional linear RGB color adds a smoothly tapered contribution to the mesh palette.
    public mutating func addMetaball(
        center: SIMD3<Float>, strength: Float, subtract: Float, color: SIMD3<Float>? = nil
    ) throws {
        guard center.isFiniteVector else { throw Failure.potential }
        if let color, !(0...1).contains(color.x) || !(0...1).contains(color.y) || !(0...1).contains(color.z) {
            throw Failure.color
        }
        let center = SIMD3<Double>(center)
        try addPotential(strength: strength, subtract: subtract, epsilon: 0.000001, color: color) {
            simd_length_squared($0 - center)
        }
    }

    /// Adds an axis-aligned plane potential with falloff
    /// `max(abs(strength) / (1e-4 + distance²) - subtract, 0)`, in world units.
    /// A plane at the minimum bound gives the corresponding boundary plane.
    public mutating func addPlane(perpendicularTo axis: Axis, at coordinate: Float, strength: Float, subtract: Float) throws {
        guard coordinate.isFinite else { throw Failure.potential }
        try addPotential(strength: strength, subtract: subtract, epsilon: 0.0001) {
            let distance = $0[axis.rawValue] - Double(coordinate)
            return distance * distance
        }
    }

    /// Extracts the surface bounding samples strictly greater than the level.
    /// Exact-level samples are exterior and intersections on them share a vertex.
    /// Normals point toward lower values. UVs are planar XZ coordinates over the
    /// grid bounds. An empty field returns nil. Budget and singular topology
    /// failures throw before publishing a mesh.
    public func mesh(
        isovalue: Float, boundary: Boundary = .open, maximumTriangles: Int = Self.maximumTriangles
    ) throws -> MeshGeometry? {
        guard isovalue.isFinite else { throw Failure.isovalue }
        guard (1...Self.maximumTriangles).contains(maximumTriangles) else { throw Failure.triangleLimit }
        var extractor = MarchingCubeExtractor(grid: self, level: Double(isovalue), maximumTriangles: maximumTriangles)
        return try extractor.mesh(boundary: boundary)
    }

    private mutating func addPotential(
        strength: Float, subtract: Float, epsilon: Double, color: SIMD3<Float>? = nil,
        squaredDistance: (SIMD3<Double>) -> Double
    ) throws {
        guard strength.isFinite, subtract.isFinite, subtract > 0 else { throw Failure.potential }
        guard strength != 0 else { return }
        let magnitude = abs(Double(strength))
        let sign: Double = strength > 0 ? 1 : -1
        var updated = values
        var updatedPalette = palette
        let inverseRadiusSquared = Double(subtract) / magnitude
        var index = 0
        for z in axes[2] {
            for y in axes[1] {
                for x in axes[0] {
                    let distanceSquared = squaredDistance(SIMD3(x, y, z))
                    let potential = max(0, magnitude / (epsilon + distanceSquared) - Double(subtract))
                    let value = Float(Double(values[index]) + potential * sign)
                    guard value.isFinite else { throw Failure.sampleValue }
                    updated[index] = value
                    if let color, potential > 0 {
                        if updatedPalette == nil { updatedPalette = .init(repeating: .zero, count: values.count) }
                        let existing = updatedPalette?[index] ?? .zero
                        let radiusFraction = (distanceSquared * inverseRadiusSquared).squareRoot()
                        let ratioSquared = radiusFraction * radiusFraction
                        // The r187 palette uses a quintic fade across the ball's support.
                        let fade = 1 - ratioSquared * radiusFraction
                            * (radiusFraction * (6 * radiusFraction - 15) + 10)
                        let color = SIMD3<Float>(SIMD3<Double>(existing) + SIMD3<Double>(color) * max(0, fade))
                        guard color.isFiniteVector else { throw Failure.color }
                        updatedPalette?[index] = color
                    }
                    index += 1
                }
            }
        }
        values = updated
        palette = updatedPalette
    }

    private static func makeAxes(_ resolution: SIMD3<Int>, _ bounds: AxisAlignedBox) throws -> [[Double]] {
        let limits = 2...maximumAxisSamples
        guard limits.contains(resolution.x), limits.contains(resolution.y), limits.contains(resolution.z),
              resolution.x * resolution.y <= maximumSamples / resolution.z else { throw Failure.resolution }
        return try (0..<3).map { axis in
            let low = Double(bounds.minimum[axis]), high = Double(bounds.maximum[axis])
            let count = resolution[axis]
            guard low < high else { throw Failure.bounds }
            let step = (high - low) / Double(count - 1)
            let coordinates = (0..<count).map { $0 == count - 1 ? high : low + Double($0) * step }
            guard zip(coordinates, coordinates.dropFirst()).allSatisfy({ Float($0) < Float($1) }) else {
                throw Failure.bounds
            }
            return coordinates
        }
    }

    private static func validate(_ values: [Float], count: Int) throws {
        guard values.count == count else { throw Failure.sampleCount }
        guard values.allSatisfy(\.isFinite) else { throw Failure.sampleValue }
    }

    public enum Failure: LocalizedError, Equatable {
        case resolution, bounds, sampleCount, sampleValue, potential, isovalue, color
        case triangleLimit, openSurface, singularSurface

        public var errorDescription: String? {
            switch self {
            case .resolution: "Use 2...1024 samples per axis and at most 2097152 samples total."
            case .bounds: "Use positive grid extents with distinct Float sample positions."
            case .sampleCount: "Supply one field value per grid sample."
            case .sampleValue: "The scalar grid needs finite Float samples."
            case .potential: "Use a finite potential and a positive finite subtract value."
            case .color: "Use a linear RGB color in 0...1 and finite accumulated palette values."
            case .isovalue: "Use a finite isosurface level."
            case .triangleLimit: "The triangle budget is invalid or the isosurface exceeds it."
            case .openSurface: "The isosurface extends to the grid boundary."
            case .singularSurface: "This field and level produce a singular surface. Change the level or sample grid."
            }
        }
    }
}
