// The area and search lookup tables of SMAA, generated in code by a line-by-line
// port of Scripts/AreaTex.py and Scripts/SearchTex.py from
// https://github.com/iryoku/smaa. For SMAA 1x the output equals the bytes the
// repository publishes in Textures/AreaTex.h (its first 80 rows) and SearchTex.h.
//
// Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
// Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
// Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
// Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
// Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to
// do so, subject to the following conditions: The above copyright notice and
// this permission notice shall be included in all copies or substantial portions
// of the Software. As clarification, there is no requirement that the copyright
// notice and permission be included in binary distributions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import Foundation

/// SMAA's precomputed lookup tables, computed on the CPU instead of shipped as data.
enum SMAALookupTextures {
    /// The area table holds orthogonal patterns in its left 80 columns and diagonal
    /// patterns in its right 80, for subsample offset 0 (SMAA 1x). It equals the
    /// first 80 rows of the published 160 × 560 AreaTex.
    static let areaWidth = 160
    static let areaHeight = 80
    static let searchWidth = 64
    static let searchHeight = 16

    private static let orthogonalSize = 16
    private static let diagonalSize = 20
    private static let diagonalSamples = 30
    private static let smoothMaxDistance = 32.0

    // Pattern placement in 5 × 5 (orthogonal) and 4 × 4 (diagonal) grids of subtables.
    private static let orthogonalSlots: [(Int, Int)] = [
        (0, 0), (3, 0), (0, 3), (3, 3), (1, 0), (4, 0), (1, 3), (4, 3),
        (0, 1), (3, 1), (0, 4), (3, 4), (1, 1), (4, 1), (1, 4), (4, 4),
    ]
    private static let diagonalSlots: [(Int, Int)] = [
        (0, 0), (1, 0), (0, 2), (1, 2), (2, 0), (3, 0), (2, 2), (3, 2),
        (0, 1), (1, 1), (0, 3), (1, 3), (2, 1), (3, 1), (2, 3), (3, 3),
    ]

    typealias Vector = SIMD2<Double>

    /// Row-major RG8 texels, two bytes each, `areaWidth` × `areaHeight`.
    static func areaBytes() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: areaWidth * areaHeight * 2)
        func put(_ x: Int, _ y: Int, _ value: Vector) {
            let index = (y * areaWidth + x) * 2
            bytes[index] = quantize(value.x)
            bytes[index + 1] = quantize(value.y)
        }
        // Orthogonal distances are stored quadratically: texel (l, r) holds distances (l², r²).
        for pattern in 0..<16 {
            let (slotX, slotY) = orthogonalSlots[pattern]
            for left in 0..<orthogonalSize {
                for right in 0..<orthogonalSize {
                    put(
                        left + orthogonalSize * slotX, right + orthogonalSize * slotY,
                        orthogonalArea(pattern: pattern, left: left * left, right: right * right, offset: 0)
                    )
                }
            }
        }
        for pattern in 0..<16 {
            let (slotX, slotY) = diagonalSlots[pattern]
            for left in 0..<diagonalSize {
                for right in 0..<diagonalSize {
                    put(
                        5 * orthogonalSize + left + diagonalSize * slotX, right + diagonalSize * slotY,
                        diagonalArea(pattern: pattern, left: left, right: right)
                    )
                }
            }
        }
        return bytes
    }

    /// Row-major R8 texels, `searchWidth` × `searchHeight`.
    static func searchBytes() -> [UInt8] {
        func lerp(_ a: Double, _ b: Double, _ p: Double) -> Double { a + (b - a) * p }
        // The value a bilinear fetch at (-0.25, -0.125) returns for four edges.
        func bilinear(_ e: [Double]) -> Double {
            lerp(lerp(e[0], e[1], 1.0 - 0.25), lerp(e[2], e[3], 1.0 - 0.25), 1.0 - 0.125)
        }
        var edges: [Double: [Int]] = [:]
        for bits in 0..<16 {
            let e = [(bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1, bits & 1]
            edges[bilinear(e.map(Double.init))] = e
        }
        func deltaLeft(_ left: [Int], _ top: [Int]) -> Int {
            var d = 0
            if top[3] == 1 { d += 1 }
            if d == 1 && top[2] == 1 && left[1] != 1 && left[3] != 1 { d += 1 }
            return d
        }
        func deltaRight(_ left: [Int], _ top: [Int]) -> Int {
            var d = 0
            if top[3] == 1 && left[1] != 1 && left[3] != 1 { d += 1 }
            if d == 1 && top[2] == 1 && left[0] != 1 && left[2] != 1 { d += 1 }
            return d
        }
        var image = [[UInt8]](repeating: [UInt8](repeating: 0, count: 66), count: 33)
        for x in 0..<33 {
            for y in 0..<33 {
                guard let left = edges[0.03125 * Double(x)], let top = edges[0.03125 * Double(y)] else { continue }
                image[y][x] = UInt8(127 * deltaLeft(left, top))
                image[y][33 + x] = UInt8(127 * deltaRight(left, top))
            }
        }
        // Crop rows 17...32 and columns 0..<64, then flip vertically.
        var bytes: [UInt8] = []
        bytes.reserveCapacity(searchWidth * searchHeight)
        for row in 0..<searchHeight {
            bytes += image[32 - row][0..<searchWidth]
        }
        return bytes
    }

    private static func quantize(_ value: Double) -> UInt8 {
        // Python's int(255.0 * a) truncates.
        UInt8(min(255, max(0, Int(255.0 * value))))
    }

    // MARK: - Orthogonal areas

    /// The area under the line p1 → p2 for the pixel x..x+1, split into its
    /// parts below and above the edge.
    private static func area(_ p1: Vector, _ p2: Vector, _ x: Int) -> Vector {
        let d = p2 - p1
        let x1 = Double(x)
        let x2 = Double(x) + 1.0
        let y1 = p1.y + d.y * (x1 - p1.x) / d.x
        let y2 = p1.y + d.y * (x2 - p1.x) / d.x

        let inside = (x1 >= p1.x && x1 < p2.x) || (x2 > p1.x && x2 <= p2.x)
        guard inside else { return .zero }
        let isTrapezoid = copysign(1.0, y1) == copysign(1.0, y2) || abs(y1) < 1e-4 || abs(y2) < 1e-4
        if isTrapezoid {
            let a = (y1 + y2) / 2.0
            return a < 0.0 ? Vector(abs(a), 0) : Vector(0, abs(a))
        }
        // Two triangles.
        let crossing = -p1.y * d.x / d.y + p1.x
        let fraction = crossing - crossing.rounded(.towardZero)
        let a1 = crossing > p1.x ? y1 * fraction / 2.0 : 0.0
        let a2 = crossing < p2.x ? y2 * (1.0 - fraction) / 2.0 : 0.0
        let a = abs(a1) > abs(a2) ? a1 : -a2
        return a < 0.0 ? Vector(abs(a1), abs(a2)) : Vector(abs(a2), abs(a1))
    }

    /// Smoothing for short U patterns.
    private static func smoothArea(_ d: Int, _ a1: Vector, _ a2: Vector) -> Vector {
        let b1 = (a1 * 2.0).squareRoot() * 0.5
        let b2 = (a2 * 2.0).squareRoot() * 0.5
        let p = min(max(Double(d) / smoothMaxDistance, 0.0), 1.0)
        let c1 = b1 + (a1 - b1) * p
        let c2 = b2 + (a2 - b2) * p
        return c1 + c2
    }

    static func orthogonalArea(pattern: Int, left: Int, right: Int, offset: Double) -> Vector {
        let d = left + right + 1
        let half = Double(d) / 2.0
        let o1 = 0.5 + offset
        let o2 = 0.5 + offset - 1.0
        switch pattern {
        case 1:
            return left <= right ? area(Vector(0, o2), Vector(half, 0), left) : .zero
        case 2:
            return left >= right ? area(Vector(half, 0), Vector(Double(d), o2), left) : .zero
        case 3:
            let a1 = area(Vector(0, o2), Vector(half, 0), left)
            let a2 = area(Vector(half, 0), Vector(Double(d), o2), left)
            return smoothArea(d, a1, a2)
        case 4:
            return left <= right ? area(Vector(0, o1), Vector(half, 0), left) : .zero
        case 6:
            if abs(offset) > 0 {
                let a1 = area(Vector(0, o1), Vector(Double(d), o2), left)
                let a2 = area(Vector(0, o1), Vector(half, 0), left) + area(Vector(half, 0), Vector(Double(d), o2), left)
                return (a1 + a2) / 2.0
            }
            return area(Vector(0, o1), Vector(Double(d), o2), left)
        case 7:
            return area(Vector(0, o1), Vector(Double(d), o2), left)
        case 8:
            return left >= right ? area(Vector(half, 0), Vector(Double(d), o1), left) : .zero
        case 9:
            if abs(offset) > 0 {
                let a1 = area(Vector(0, o2), Vector(Double(d), o1), left)
                let a2 = area(Vector(0, o2), Vector(half, 0), left) + area(Vector(half, 0), Vector(Double(d), o1), left)
                return (a1 + a2) / 2.0
            }
            return area(Vector(0, o2), Vector(Double(d), o1), left)
        case 11:
            return area(Vector(0, o2), Vector(Double(d), o1), left)
        case 12:
            let a1 = area(Vector(0, o1), Vector(half, 0), left)
            let a2 = area(Vector(half, 0), Vector(Double(d), o1), left)
            return smoothArea(d, a1, a2)
        case 13:
            return area(Vector(0, o2), Vector(Double(d), o1), left)
        case 14:
            return area(Vector(0, o1), Vector(Double(d), o2), left)
        default:
            // Patterns 0, 5, 10 and 15 are not filtered.
            return .zero
        }
    }

    // MARK: - Diagonal areas

    /// The fraction of the unit pixel at `p` on the positive side of the line
    /// p1 → p2, by sampling a 30 × 30 grid that includes the pixel's edges.
    private static func pixelCoverage(_ p1: Vector, _ p2: Vector, _ p: Vector) -> Double {
        guard p1 != p2 else { return 1 }
        // The reference tests every sample. Each test below evaluates the same
        // expression in the same order, so the count is bit-exact; because every
        // rounded step is monotonic in y, a binary search over each column finds
        // the same count with far fewer tests, which keeps debug builds fast.
        let middleX = (p1.x + p2.x) / 2.0
        let middleY = (p1.y + p2.y) / 2.0
        let a = p2.y - p1.y
        let b = p1.x - p2.x
        let spacing = Double(diagonalSamples - 1)
        var count = 0
        for x in 0..<diagonalSamples {
            let horizontal = a * (p.x + Double(x) / spacing - middleX)
            func inside(_ y: Int) -> Bool { horizontal + b * (p.y + Double(y) / spacing - middleY) > 0 }
            if b == 0 {
                count += inside(0) ? diagonalSamples : 0
                continue
            }
            // The first y whose test differs from the test at y = 0.
            let first = inside(0)
            var low = 0
            var high = diagonalSamples
            while low < high {
                let middle = (low + high) / 2
                if inside(middle) == first { low = middle + 1 } else { high = middle }
            }
            count += first ? low : diagonalSamples - low
        }
        return Double(count) / Double(diagonalSamples * diagonalSamples)
    }

    /// The areas for the pixel and its opposite under the line p1 → p2 (offset 0).
    private static func diagonalLineArea(_ p1: Vector, _ p2: Vector, _ left: Int) -> Vector {
        let base = Vector(Double(left), Double(left))
        let a1 = pixelCoverage(p1, p2, Vector(1, 0) + base)
        let a2 = pixelCoverage(p1, p2, Vector(1, 1) + base)
        return Vector(1.0 - a1, a2)
    }

    static func diagonalArea(pattern: Int, left: Int, right: Int) -> Vector {
        let d = Double(left + right + 1)
        let dd = Vector(d, d)
        func line(_ p1: Vector, _ p2: Vector) -> Vector { diagonalLineArea(p1, p2 + dd, left) }
        func blend(_ a1: Vector, _ a2: Vector) -> Vector { (a1 + a2) / 2.0 }
        let zero = Vector(0, 0), right0 = Vector(1, 0), one = Vector(1, 1)
        switch pattern {
        case 0: return blend(line(one, one), line(right0, right0))
        case 1: return blend(line(right0, zero), line(right0, right0))
        case 2: return blend(line(zero, right0), line(right0, right0))
        case 3: return line(right0, right0)
        case 4: return blend(line(one, zero), line(one, right0))
        case 5: return blend(line(one, zero), line(right0, right0))
        case 6: return line(one, right0)
        case 7: return blend(line(one, right0), line(right0, right0))
        case 8: return blend(line(zero, one), line(right0, one))
        case 9: return line(right0, one)
        case 10: return blend(line(zero, one), line(right0, right0))
        case 11: return blend(line(right0, one), line(right0, right0))
        case 12: return line(one, one)
        case 13: return blend(line(one, one), line(right0, one))
        case 14: return blend(line(one, one), line(one, right0))
        default: return blend(line(one, one), line(right0, right0))
        }
    }
}
