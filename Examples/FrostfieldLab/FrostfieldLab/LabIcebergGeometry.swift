import Foundation
import Frostfield
import simd

/// A closed, calved glacier berg. Y is up and the mean waterline is Y = 0.
enum LabIcebergGeometry {
    private struct Section {
        let front: SIMD3<Float>
        let back: SIMD3<Float>
        let crest: Float
        let crestDepth: Float

        init(_ front: SIMD3<Float>, _ back: SIMD3<Float>, _ crest: Float, _ crestDepth: Float) {
            self.front = front
            self.back = back
            self.crest = crest
            self.crestDepth = crestDepth
        }
    }

    // The high block is offset toward the back. A lower slab projects at the right.
    private static let sections: [Section] = [
        Section(SIMD3(-4.99, 0.86, 0.84), SIMD3(-4.87, 1.00, -1.16), 1.08, 0.53),
        Section(SIMD3(-4.45, 1.91, 1.48), SIMD3(-4.43, 2.42, -1.76), 2.53, 0.72),
        Section(SIMD3(-3.88, 2.34, 2.12), SIMD3(-3.96, 3.40, -2.27), 3.66, 0.73),
        Section(SIMD3(-3.30, 3.30, 2.23), SIMD3(-3.31, 3.71, -2.51), 4.02, 0.65),
        Section(SIMD3(-2.86, 3.35, 2.70), SIMD3(-2.86, 4.05, -2.62), 4.41, 0.69),
        Section(SIMD3(-2.35, 3.62, 2.80), SIMD3(-2.41, 4.27, -2.74), 4.50, 0.64),
        Section(SIMD3(-1.71, 3.68, 2.51), SIMD3(-1.70, 4.02, -2.76), 4.35, 0.54),
        Section(SIMD3(-1.22, 3.37, 2.29), SIMD3(-1.12, 3.96, -2.60), 4.22, 0.53),
        Section(SIMD3(-0.75, 3.10, 2.35), SIMD3(-0.81, 3.68, -2.72), 3.65, 0.61),
        Section(SIMD3(-0.04, 2.55, 2.14), SIMD3(-0.04, 3.12, -2.65), 3.08, 0.70),
        Section(SIMD3( 0.61, 2.89, 2.41), SIMD3( 0.62, 3.05, -2.52), 3.38, 0.46),
        Section(SIMD3( 1.21, 3.21, 2.69), SIMD3( 1.18, 3.29, -2.68), 3.82, 0.39),
        Section(SIMD3( 1.71, 3.02, 2.72), SIMD3( 1.66, 3.21, -2.65), 3.60, 0.44),
        Section(SIMD3( 2.23, 2.12, 2.56), SIMD3( 2.26, 2.99, -2.43), 3.10, 0.64),
        Section(SIMD3( 2.65, 1.38, 2.65), SIMD3( 2.71, 2.47, -2.18), 2.57, 0.73),
        Section(SIMD3( 3.34, 1.46, 2.54), SIMD3( 3.30, 2.08, -1.91), 2.01, 0.65),
        Section(SIMD3( 4.02, 1.10, 2.17), SIMD3( 4.00, 1.70, -1.72), 1.58, 0.60),
        Section(SIMD3( 4.72, 0.77, 1.77), SIMD3( 4.61, 1.27, -1.13), 1.36, 0.58),
        Section(SIMD3( 4.99, 0.53, 1.09), SIMD3( 4.93, 0.87, -0.39), 0.94, 0.62),
    ]

    private struct Spall {
        let center: SIMD2<Float>
        let extent: SIMD2<Float>
        let slant: Float
        let setback: Float

        init(x: Float, y: Float, width: Float, height: Float, slant: Float, setback: Float) {
            center = SIMD2(x, y)
            extent = SIMD2(width, height)
            self.slant = slant
            self.setback = setback
        }
    }

    // Broad ice plates have different directions of breakage and stop at different heights.
    private static let frontSpalls = [
        Spall(x: -2.90, y: 1.94, width: 1.40, height: 1.55, slant: -0.38, setback: 0.48),
        Spall(x: -0.25, y: 1.36, width: 1.22, height: 1.48, slant:  0.54, setback: 0.64),
        Spall(x:  1.85, y: 0.87, width: 0.98, height: 0.88, slant: -0.76, setback: 0.39),
        Spall(x:  3.58, y: 0.59, width: 1.15, height: 0.62, slant:  0.26, setback: 0.30),
    ]

    static func make() throws -> MeshGeometry {
        let capColumns = 120
        let capRows = 60
        let wallRows = 50
        let stride = capColumns + 1
        var positions: [SIMD3<Float>] = []
        var capUV: [SIMD2<Float>] = []
        var triangles: [SIMD3<Int>] = []

        for row in 0...capRows {
            for column in 0...capColumns {
                let uv = SIMD2(Float(column) / Float(capColumns), Float(row) / Float(capRows))
                positions.append(capPoint(uv))
                capUV.append(uv)
            }
        }

        let capCount = positions.count
        for row in 0..<capRows {
            for column in 0..<capColumns {
                let i = row * stride + column
                triangles.append(SIMD3(i, i + 1, i + stride))
                triangles.append(SIMD3(i + 1, i + stride + 1, i + stride))
            }
        }
        let capTriangleCount = triangles.count

        var rim = Array(0..<capColumns)
        rim += (0..<capRows).map { $0 * stride + capColumns }
        rim += (1...capColumns).reversed().map { capRows * stride + $0 }
        rim += (1...capRows).reversed().map { $0 * stride }
        let rimPoints = rim.map { positions[$0] }
        let count = rim.count
        var rimLengths: [Float] = [0]
        for column in 0..<count {
            let step = simd_distance(rimPoints[column], rimPoints[(column + 1) % count])
            rimLengths.append(rimLengths[column] + step)
        }

        for row in 1...wallRows {
            let down = Float(row) / Float(wallRows)
            for point in rimPoints { positions.append(wallPoint(from: point, down: down)) }
        }

        func ringIndex(_ row: Int, _ column: Int) -> Int {
            row == 0 ? rim[column] : capCount + (row - 1) * count + column
        }

        for row in 0..<wallRows {
            for column in 0..<count {
                let next = (column + 1) % count
                let a = ringIndex(row, column)
                let b = ringIndex(row, next)
                let c = ringIndex(row + 1, column)
                let d = ringIndex(row + 1, next)
                if (column + row).isMultiple(of: 2) {
                    triangles.append(SIMD3(a, c, b))
                    triangles.append(SIMD3(b, c, d))
                } else {
                    triangles.append(SIMD3(a, c, d))
                    triangles.append(SIMD3(a, d, b))
                }
            }
        }

        let floorCenter = positions.count
        positions.append(SIMD3(0.12, -2.61, -0.09))
        for column in 0..<count {
            triangles.append(SIMD3(ringIndex(wallRows, column), floorCenter, ringIndex(wallRows, (column + 1) % count)))
        }

        // Accumulate on the joined surface before duplicating positions for UV charts.
        var normals = Array(repeating: SIMD3<Float>.zero, count: positions.count)
        for triangle in triangles {
            let a = positions[triangle.x]
            let areaNormal = simd_cross(positions[triangle.y] - a, positions[triangle.z] - a)
            normals[triangle.x] += areaNormal
            normals[triangle.y] += areaNormal
            normals[triangle.z] += areaNormal
        }
        normals = normals.map { simd_normalize($0) }

        func vertex(_ index: Int, uv: SIMD2<Float>) -> MeshVertex {
            MeshVertex(position: positions[index], normal: normals[index], textureCoordinate: uv)
        }

        var vertices = capUV.enumerated().map { vertex($0.offset, uv: $0.element) }
        var indices: [UInt32] = []
        indices.reserveCapacity(triangles.count * 3)
        for face in triangles.prefix(capTriangleCount) {
            indices += [UInt32(face.x), UInt32(face.y), UInt32(face.z)]
        }
        let wallBase = vertices.count
        let wallStride = count + 1
        for row in 0...wallRows {
            for column in 0...count {
                let u = rimLengths[column] / rimLengths[count]
                vertices.append(vertex(ringIndex(row, column % count), uv: SIMD2(u, Float(row) / Float(wallRows))))
            }
        }
        for row in 0..<wallRows {
            for column in 0..<count {
                let a = UInt32(wallBase + row * wallStride + column)
                let c = a + UInt32(wallStride)
                if (column + row).isMultiple(of: 2) {
                    indices += [a, c, a + 1, a + 1, c, c + 1]
                } else {
                    indices += [a, c, c + 1, a, c + 1, a + 1]
                }
            }
        }

        let floorBase = UInt32(vertices.count)
        for column in 0..<count {
            let i = ringIndex(wallRows, column)
            let point = positions[i]
            vertices.append(vertex(i, uv: SIMD2(0.5 + point.x / 10.8, 0.5 - point.z / 6.6)))
        }
        let center = UInt32(vertices.count)
        vertices.append(vertex(floorCenter, uv: SIMD2(0.5, 0.5)))
        for column in 0..<count {
            indices += [floorBase + UInt32(column), center, floorBase + UInt32((column + 1) % count)]
        }

        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    private static func capPoint(_ uv: SIMD2<Float>) -> SIMD3<Float> {
        let index = uv.x * Float(sections.count - 1)
        let left = min(Int(index), sections.count - 2)
        let fraction = index - Float(left)
        let a = sections[left]
        let b = sections[left + 1]
        let front = simd_mix(a.front, b.front, SIMD3(repeating: fraction))
        let back = simd_mix(a.back, b.back, SIMD3(repeating: fraction))
        var point = simd_mix(front, back, SIMD3(repeating: uv.y))
        let crest = mix(a.crest, b.crest, fraction)
        let crestDepth = mix(a.crestDepth, b.crestDepth, fraction)
        let lift = crest - mix(front.y, back.y, crestDepth)
        let slope = min(uv.y / crestDepth, (1 - uv.y) / (1 - crestDepth))
        point.y += lift * min(slope / 0.80, 1)

        let horizontal = SIMD3(point.x, 0, point.z)
        let relief = noise(horizontal * 1.06 + SIMD3(3, 5, 2))
        let chips = noise(horizontal * 3.7 + SIMD3(1, 5, 8))
        point.z += 0.10 * relief + 0.062 * chips
        point.y += 0.28 * relief + 0.13 * chips
        point.y += 0.025 * noise(horizontal * 12 + SIMD3(2, 1, 6))

        // The deep joint crosses the two slabs obliquely. The shorter scar dies out on the cap.
        let jointX = -0.77 + 0.55 * point.z + 0.13 * relief
        let jointDistance = abs(point.x - jointX)
        let joint = 0.64 * (1 - smooth(0.048, 0.30 + 0.07 * chips, jointDistance))
        let scarDistance = abs(point.z - (0.48 + 0.41 * point.x + 0.14 * relief))
        let scarSpan = smooth(-4.05, -3.38, point.x) * (1 - smooth(-1.72, -0.84, point.x))
        let scar = 0.31 * scarSpan * (1 - smooth(0.03, 0.22, scarDistance))
        point.y -= max(joint, scar)
        return point
    }

    private static func wallPoint(from top: SIMD3<Float>, down: Float) -> SIMD3<Float> {
        let center = SIMD2<Float>(0.12, -0.09)
        let xz = SIMD2(top.x, top.z) - center
        let outward = simd_normalize(xz)
        let floor = -2.47 + 0.12 * noise(SIMD3(top.x * 0.8, 8, top.z))
        let y = mix(top.y, floor, down)
        let distance = top.y - y
        let weight = smooth(0, 0.38, distance) * (1 - 0.86 * smooth(0.85, 1, down))
        let taper = 1 + 0.064 * smooth(0, 1.45, distance) - 0.19 * smooth(0.64, 1, down)

        let sample = SIMD3(top.x + 0.24 * y, y, top.z - 0.16 * y)
        let face = noise(sample * SIMD3(0.94, 0.82, 1.29) + SIMD3(0, 1, 3))
        let blocks = noise(sample * SIMD3(2.30, 1.48, 2.82) + SIMD3(4, 5, 0))
        let chips = noise(sample * SIMD3(7.8, 5.6, 7.2))
        let front = smooth(0.30, 1.80, top.z)
        let spall = front * spallDepth(x: top.x, y: y, edgeNoise: blocks)

        // A wave-cut notch is interrupted by blocks that project into the water.
        let notchCenter = 0.04 + 0.035 * top.x + 0.10 * face
        let notch = (0.05 + 0.065 * smooth(-0.55, 0.55, blocks))
            * (1 - smooth(0.04, 0.31, abs(y - notchCenter)))
        let relief = 0.23 * face + 0.11 * blocks + 0.036 * chips - spall - notch
        let shear = SIMD2(0.10 * (front - 0.42), -0.055) * min(distance, 2.4)
        let edge = center + xz * taper + weight * (outward * relief + shear)
        return SIMD3(edge.x, y, edge.y)
    }

    private static func spallDepth(x: Float, y: Float, edgeNoise: Float) -> Float {
        frontSpalls.reduce(0) { depth, plate in
            let vertical = y - plate.center.y
            let across = (x - plate.center.x - vertical * plate.slant) / plate.extent.x
            let height = vertical / plate.extent.y
            let outline = max(abs(across), abs(height) + 0.31 * abs(across + 0.47 * height))
            let plateDepth = plate.setback * (1 - smooth(0.42, 1, outline + 0.11 * edgeNoise))
            return max(depth, plateDepth)
        }
    }

    private static func mix(_ a: Float, _ b: Float, _ weight: Float) -> Float {
        a + (b - a) * weight
    }

    private static func smooth(_ a: Float, _ b: Float, _ value: Float) -> Float {
        let t = min(max((value - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }

    private static func noise(_ p: SIMD3<Float>) -> Float {
        let f = SIMD3(floor(p.x), floor(p.y), floor(p.z))
        let cell = SIMD3(Int32(f.x), Int32(f.y), Int32(f.z))
        let t = p - f
        let blend = t * t * (SIMD3(repeating: 3) - 2 * t)

        func hash(_ x: Int32, _ y: Int32, _ z: Int32) -> Float {
            var h = UInt32(bitPattern: cell.x + x) &* 0x8da6b343
            h ^= UInt32(bitPattern: cell.y + y) &* 0xd8163841
            h ^= UInt32(bitPattern: cell.z + z) &* 0xcb1ab31f
            h ^= h >> 16
            h = h &* 0x7feb352d
            h ^= h >> 15
            h = h &* 0x846ca68b
            h ^= h >> 16
            return Float(h & 0x00ffffff) / 8_388_607.5 - 1
        }

        let z0 = mix(mix(hash(0, 0, 0), hash(1, 0, 0), blend.x), mix(hash(0, 1, 0), hash(1, 1, 0), blend.x), blend.y)
        let z1 = mix(mix(hash(0, 0, 1), hash(1, 0, 1), blend.x), mix(hash(0, 1, 1), hash(1, 1, 1), blend.x), blend.y)
        return mix(z0, z1, blend.z)
    }
}
