import Foundation
import simd

/// A point and interpolated attributes in the mesh's local coordinate space.
public struct MeshSurfaceSample: Equatable, Sendable {
    public let position: SIMD3<Float>
    public let normal: SIMD3<Float>
    public let textureCoordinate: SIMD2<Float>

    public init(position: SIMD3<Float>, normal: SIMD3<Float>, textureCoordinate: SIMD2<Float> = .zero) {
        self.position = position
        self.normal = normal
        self.textureCoordinate = textureCoordinate
    }
}

/// Samples indexed triangles with probability proportional to their area.
public struct MeshSurfaceSampler {
    public static let maximumSampleCount = 262_144

    private struct Triangle {
        let indexOffset: Int
        let cumulativeArea: Double
    }

    private let geometry: MeshGeometry
    private let triangles: [Triangle]
    private let totalArea: Double

    public init(geometry: MeshGeometry) throws {
        var triangles: [Triangle] = []
        triangles.reserveCapacity(geometry.indices.count / 3)
        var cumulativeArea: Double = 0
        for offset in stride(from: 0, to: geometry.indices.count, by: 3) {
            let a = Self.double(geometry.vertices[Int(geometry.indices[offset])].position)
            let b = Self.double(geometry.vertices[Int(geometry.indices[offset + 1])].position)
            let c = Self.double(geometry.vertices[Int(geometry.indices[offset + 2])].position)
            let nextArea = cumulativeArea + simd_length(simd_cross(b - a, c - a)) * 0.5
            guard nextArea > cumulativeArea else { continue }
            cumulativeArea = nextArea
            triangles.append(Triangle(indexOffset: offset, cumulativeArea: cumulativeArea))
        }
        guard cumulativeArea > 0 else { throw Failure.emptySurface }
        self.geometry = geometry
        self.triangles = triangles
        totalArea = cumulativeArea
    }

    /// Returns reproducible area-uniform positions. Interpolated normals are
    /// normalized, with the triangle normal used when vertex normals cancel.
    public func samples(count: Int, seed: UInt64) throws -> [MeshSurfaceSample] {
        guard (0...Self.maximumSampleCount).contains(count) else { throw Failure.count(count) }
        var random = SceneSeedGenerator(state: seed)
        var result: [MeshSurfaceSample] = []
        result.reserveCapacity(count)
        for _ in 0..<count {
            let triangle = triangle(at: random.nextDouble() * totalArea)
            let a = geometry.vertices[Int(geometry.indices[triangle.indexOffset])]
            let b = geometry.vertices[Int(geometry.indices[triangle.indexOffset + 1])]
            let c = geometry.vertices[Int(geometry.indices[triangle.indexOffset + 2])]

            let radius = sqrt(random.nextDouble())
            let fraction = random.nextDouble()
            let wa = 1 - radius
            let wb = radius * (1 - fraction)
            let wc = radius * fraction

            let pa = Self.double(a.position)
            let pb = Self.double(b.position)
            let pc = Self.double(c.position)
            let position = wa * pa + wb * pb + wc * pc
            var normal = wa * Self.double(a.normal) + wb * Self.double(b.normal) + wc * Self.double(c.normal)
            if normal == .zero { normal = simd_cross(pb - pa, pc - pa) }
            let uv = wa * SIMD2<Double>(a.textureCoordinate) + wb * SIMD2<Double>(b.textureCoordinate)
                + wc * SIMD2<Double>(c.textureCoordinate)
            result.append(MeshSurfaceSample(
                position: SIMD3<Float>(position),
                normal: SIMD3<Float>(simd_normalize(normal)),
                textureCoordinate: SIMD2<Float>(uv)
            ))
        }
        return result
    }

    private func triangle(at area: Double) -> Triangle {
        var lower = 0
        var upper = triangles.count - 1
        while lower < upper {
            let middle = lower + (upper - lower) / 2
            if triangles[middle].cumulativeArea <= area {
                lower = middle + 1
            } else {
                upper = middle
            }
        }
        return triangles[lower]
    }

    private static func double(_ value: SIMD3<Float>) -> SIMD3<Double> {
        SIMD3(Double(value.x), Double(value.y), Double(value.z))
    }

    public enum Failure: LocalizedError, Equatable {
        case emptySurface
        case count(Int)

        public var errorDescription: String? {
            switch self {
            case .emptySurface: "Surface sampling needs at least one triangle with positive area."
            case .count(let count): "Sample count \(count) is outside 0...\(MeshSurfaceSampler.maximumSampleCount)."
            }
        }
    }
}

// SplitMix64 supplies stable seeds without depending on platform RNG state.
struct SceneSeedGenerator {
    var state: UInt64

    mutating func nextUnit() -> Float {
        Float(next() >> 40) / 16_777_216
    }

    mutating func nextDouble() -> Double {
        Double(next() >> 11) / 9_007_199_254_740_992
    }

    private mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var bits = state
        bits = (bits ^ (bits >> 30)) &* 0xBF58476D1CE4E5B9
        bits = (bits ^ (bits >> 27)) &* 0x94D049BB133111EB
        return bits ^ (bits >> 31)
    }
}
