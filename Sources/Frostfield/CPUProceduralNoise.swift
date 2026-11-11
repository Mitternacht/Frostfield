import Foundation
import simd

public extension ShaderNoise {
    /// Signed three-dimensional improved Perlin noise using the standard
    /// 256-cell permutation. Nonfinite inputs return NaN.
    static func improvedPerlin(_ position: SIMD3<Double>) -> Double {
        guard position.x.isFinite, position.y.isFinite, position.z.isFinite else { return .nan }
        let floorPosition = SIMD3(
            position.x.rounded(.down), position.y.rounded(.down), position.z.rounded(.down)
        )
        let local = position - floorPosition
        let ix = wrappedCell(floorPosition.x)
        let iy = wrappedCell(floorPosition.y)
        let iz = wrappedCell(floorPosition.z)

        func fade(_ x: Double) -> Double { x * x * x * (x * (x * 6 - 15) + 10) }
        func blend(_ first: Double, _ second: Double, _ t: Double) -> Double { (1 - t) * first + t * second }
        func corner(_ dx: Int, _ dy: Int, _ dz: Int) -> Double {
            let permutation = improvedPermutation
            let first = Int(permutation[(ix + dx) & 255])
            let second = Int(permutation[(first + iy + dy) & 255])
            let hash = permutation[(second + iz + dz) & 255] & 15
            let x = local.x - Double(dx), y = local.y - Double(dy), z = local.z - Double(dz)
            let u = hash < 8 ? x : y
            let v = hash < 4 ? y : (hash == 12 || hash == 14 ? x : z)
            return (hash & 1 == 0 ? u : -u) + (hash & 2 == 0 ? v : -v)
        }

        let x = fade(local.x), y = fade(local.y), z = fade(local.z)
        let bottom = blend(blend(corner(0, 0, 0), corner(1, 0, 0), x),
                           blend(corner(0, 1, 0), corner(1, 1, 0), x), y)
        let top = blend(blend(corner(0, 0, 1), corner(1, 0, 1), x),
                        blend(corner(0, 1, 1), corner(1, 1, 1), x), y)
        return blend(bottom, top, z)
    }
}

/// Seeded signed simplex noise in two, three and four dimensions.
/// Use a stable seed for repeatable asset generation. Nonfinite inputs or
/// coordinates whose skewed lattice cannot be represented return NaN.
public struct SimplexNoise: Sendable {
    private let table: [UInt8]

    public init(seed: UInt64 = 0) {
        var random = SceneSeedGenerator(state: seed)
        table = (0..<256).map { _ in UInt8(random.nextUnit() * 256) }
    }

    /// Imports 256 lattice values. Repeated values are accepted.
    public init(table: [UInt8]) throws {
        guard table.count == 256 else { throw Failure.tableSize(table.count) }
        self.table = table
    }

    public func noise(_ position: SIMD2<Double>) -> Double {
        sample(SIMD4(position.x, position.y, 0, 0), dimensions: 2,
               skew: Self.skew2, unskew: Self.unskew2, radius: 0.5, scale: 70)
    }

    public func noise(_ position: SIMD3<Double>) -> Double {
        sample(SIMD4(position.x, position.y, position.z, 0), dimensions: 3,
               skew: 1.0 / 3, unskew: 1.0 / 6, radius: 0.6, scale: 32)
    }

    public func noise(_ position: SIMD4<Double>) -> Double {
        sample(position, dimensions: 4, skew: Self.skew4, unskew: Self.unskew4, radius: 0.6, scale: 27)
    }

    public enum Failure: LocalizedError, Equatable {
        case tableSize(Int)

        public var errorDescription: String? {
            switch self {
                case .tableSize(let count): "Simplex noise needs 256 lattice values. Received \(count)."
            }
        }
    }

    private static let skew2 = (Double(3).squareRoot() - 1) / 2
    private static let unskew2 = (3 - Double(3).squareRoot()) / 6
    private static let skew4 = (Double(5).squareRoot() - 1) / 4
    private static let unskew4 = (5 - Double(5).squareRoot()) / 20
    private static let gradients: [SIMD3<Double>] = [
        SIMD3(1, 1, 0), SIMD3(-1, 1, 0), SIMD3(1, -1, 0), SIMD3(-1, -1, 0),
        SIMD3(1, 0, 1), SIMD3(-1, 0, 1), SIMD3(1, 0, -1), SIMD3(-1, 0, -1),
        SIMD3(0, 1, 1), SIMD3(0, -1, 1), SIMD3(0, 1, -1), SIMD3(0, -1, -1),
    ]

    private func sample(
        _ point: SIMD4<Double>, dimensions: Int, skew: Double, unskew: Double, radius: Double, scale: Double
    ) -> Double {
        let axes = 0..<dimensions
        let shifted = axes.reduce(0.0) { $0 + point[$1] } * skew
        guard shifted.isFinite else { return .nan }
        var cell = SIMD4<Double>.zero
        for axis in axes {
            cell[axis] = (point[axis] + shifted).rounded(.down)
            guard cell[axis].isFinite else { return .nan }
        }
        let unshifted = axes.reduce(0.0) { $0 + cell[$1] } * unskew
        guard unshifted.isFinite else { return .nan }

        var local = SIMD4<Double>.zero
        var indices = SIMD4<Int32>.zero
        for axis in axes {
            local[axis] = point[axis] - (cell[axis] - unshifted)
            indices[axis] = Int32(wrappedCell(cell[axis]))
        }

        // The coordinate rank identifies the simplex's successive corners.
        // In 3D earlier axes win ties, matching the reference's XYZ order.
        var rank = SIMD4<Int32>.zero
        for first in axes {
            for second in (first + 1)..<dimensions {
                let firstWins = dimensions == 3 ? local[first] >= local[second] : local[first] > local[second]
                rank[firstWins ? first : second] += 1
            }
        }

        var result = 0.0
        for corner in 0...dimensions {
            let threshold = Int32(dimensions - corner)
            var displacement = SIMD4<Double>.zero
            var hash = 0
            var weight = radius
            for axis in axes.reversed() {
                let step = rank[axis] >= threshold ? 1 : 0
                hash = Int(table[(Int(indices[axis]) + step + hash) & 255])
                displacement[axis] = local[axis] - Double(step) + Double(corner) * unskew
            }
            for axis in axes { weight -= displacement[axis] * displacement[axis] }
            guard weight > 0 else { continue }
            let gradient = dimensions == 4 ? Self.gradient4(hash, displacement) : Self.gradient3(hash, displacement)
            let squared = weight * weight
            result += squared * squared * gradient
        }
        return scale * result
    }

    private static func gradient3(_ hash: Int, _ position: SIMD4<Double>) -> Double {
        let direction = gradients[hash % gradients.count]
        return direction.x * position.x + direction.y * position.y + direction.z * position.z
    }

    private static func gradient4(_ hash: Int, _ position: SIMD4<Double>) -> Double {
        let zeroAxis = (hash & 31) >> 3
        var signBit = 4
        var projection = 0.0
        for axis in 0..<4 where axis != zeroAxis {
            projection += hash & signBit == 0 ? position[axis] : -position[axis]
            signBit >>= 1
        }
        return projection
    }
}

private func wrappedCell(_ floor: Double) -> Int {
    Int(floor.truncatingRemainder(dividingBy: 256)) & 255
}

// Ken Perlin's fixed improved-noise permutation, also used by Three.js r187.
private let improvedPermutation: [UInt8] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
]
