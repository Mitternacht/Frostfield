import Frostfield
import Foundation
import Metal
import Testing
import simd

/// SplitMix64, so randomized geometry tests are reproducible.
struct BVHTestRandom: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func float(in range: ClosedRange<Float>) -> Float {
        Float.random(in: range, using: &self)
    }

    mutating func vector(in range: ClosedRange<Float>) -> SIMD3<Float> {
        SIMD3(float(in: range), float(in: range), float(in: range))
    }
}

struct MeshBVHTests {
    private static func vase() throws -> MeshGeometry {
        try MeshGeometry.lathe(
            profile: [
                LatheProfilePoint(radius: 0.4, y: -1),
                LatheProfilePoint(radius: 0.9, y: -0.4),
                LatheProfilePoint(radius: 0.5, y: 0.3),
                LatheProfilePoint(radius: 0.75, y: 0.9),
            ],
            segments: 40
        )
    }

    private static let transforms: [simd_float4x4] = [
        matrix_identity_float4x4,
        SceneTransform(
            translation: SIMD3(0.5, -1, 2),
            rotation: simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 0.5))),
            scale: SIMD3(1.5, 0.75, 2)
        ).matrix,
        // A mirrored transform keeps the mesh's own front faces.
        SceneTransform(translation: SIMD3(-3, 0.25, 1), scale: SIMD3(-2, 1, 1)).matrix,
    ]

    /// Rays aimed near the mesh, from outside and inside it, some with short maximum distances.
    private static func rays(count: Int, random: inout BVHTestRandom, around matrix: simd_float4x4) throws -> [SceneRay] {
        let center = SIMD3(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
        return try (0..<count).map { index in
            let inside = index % 5 == 0
            let origin = center + (inside ? random.vector(in: -0.2...0.2) : random.vector(in: -6...6))
            let aim = center + random.vector(in: -2.5...2.5)
            let direction = aim == origin ? SIMD3<Float>(0, 0, -1) : aim - origin
            let limit: Float = index % 7 == 0 ? random.float(in: 0...5) : .infinity
            return try SceneRay(origin: origin, direction: direction, maximumDistance: limit)
        }
    }

    @Test(arguments: ["sphere", "vase"])
    func `BVH nearest hits equal the linear scan on randomized rays`(shape: String) throws {
        let geometry = shape == "sphere" ? try MeshGeometry.uvSphere(radius: 1.2, segments: 48, rings: 24) : try Self.vase()
        let bvh = try MeshBVH(geometry: geometry)
        var random = BVHTestRandom(seed: shape == "sphere" ? 17 : 29)
        var hits = 0
        var misses = 0
        var backHits = 0
        for matrix in Self.transforms {
            for ray in try Self.rays(count: 300, random: &random, around: matrix) {
                for culls in [true, false] {
                    let linear = try geometry.intersection(ray: ray, localToWorld: matrix, cullsBackFaces: culls)
                    let accelerated = try bvh.intersection(ray: ray, localToWorld: matrix, cullsBackFaces: culls)
                    #expect(accelerated?.triangleIndex == linear?.triangleIndex)
                    if let linear, let accelerated {
                        #expect(abs(accelerated.distance - linear.distance) <= 1e-5)
                        #expect(accelerated == linear)
                        hits += 1
                        if !culls,
                           try geometry.intersection(ray: ray, localToWorld: matrix, cullsBackFaces: true)?
                               .triangleIndex != linear.triangleIndex {
                            backHits += 1
                        }
                    } else {
                        misses += 1
                    }
                }
            }
        }
        // The sample must exercise front hits, back-face hits and misses.
        #expect(hits > 400)
        #expect(misses > 400)
        #expect(backHits > 100)
    }

    @Test
    func `Rays grazing shared vertices and edges match the linear scan`() throws {
        let geometry = try Self.vase()
        let bvh = try MeshBVH(geometry: geometry)
        var random = BVHTestRandom(seed: 41)
        var hits = 0
        for triangle in 0..<(geometry.indices.count / 3) {
            let corners = (0..<3).map { geometry.vertices[Int(geometry.indices[triangle * 3 + $0])].position }
            let outward = geometry.vertices[Int(geometry.indices[triangle * 3])].normal
            let matrix = Self.transforms[triangle % Self.transforms.count]
            func world(_ point: SIMD3<Float>) -> SIMD3<Float> {
                let moved = matrix * SIMD4(point, 1)
                return SIMD3(moved.x, moved.y, moved.z)
            }
            for aim in [corners[0], (corners[0] + corners[1]) * 0.5] {
                // Approach from outside the surface at an oblique angle.
                let origin = aim + simd_normalize(outward + random.vector(in: -0.6...0.6)) * random.float(in: 2...6)
                let ray = try SceneRay(origin: world(origin), direction: world(aim) - world(origin))
                for culls in [true, false] {
                    let linear = try geometry.intersections(ray: ray, localToWorld: matrix, cullsBackFaces: culls)
                    #expect(try bvh.intersections(ray: ray, localToWorld: matrix, cullsBackFaces: culls) == linear)
                    #expect(try bvh.intersection(ray: ray, localToWorld: matrix, cullsBackFaces: culls) == linear.first)
                    if !linear.isEmpty { hits += 1 }
                }
            }
        }
        #expect(hits > 1000)
    }

    @Test
    func `Exact ties at shared vertices and edges keep the lowest triangle index`() throws {
        // Integer grid coordinates keep every triangle test exact, so shared points tie exactly.
        let plane = try MeshGeometry.plane(width: 8, height: 8, columns: 8, rows: 8)
        let bvh = try MeshBVH(geometry: plane)
        let shift = SceneTransform(translation: SIMD3(3, -2, 1)).matrix
        var ties = 0
        for x in stride(from: Float(-4), through: 4, by: 0.5) {
            for y in stride(from: Float(-4), through: 4, by: 0.5) {
                for direction in [SIMD3<Float>(0, 0, -1), SIMD3<Float>(1, 1, -2), SIMD3<Float>(-1, 0.5, -1)] {
                    for matrix in [matrix_identity_float4x4, shift] {
                        let point = SIMD3(x + matrix.columns.3.x, y + matrix.columns.3.y, matrix.columns.3.z)
                        let ray = try SceneRay(origin: point - direction * 2, direction: direction)
                        let linear = try plane.intersections(ray: ray, localToWorld: matrix)
                        #expect(try bvh.intersections(ray: ray, localToWorld: matrix) == linear)
                        #expect(try bvh.intersection(ray: ray, localToWorld: matrix) == linear.first)
                        #expect(try plane.intersection(ray: ray, localToWorld: matrix) == linear.first)
                        if linear.count >= 2, linear[0].distance == linear[1].distance {
                            #expect(linear[0].triangleIndex < linear[1].triangleIndex)
                            ties += 1
                        }
                    }
                }
            }
        }
        #expect(ties > 300)
    }

    @Test
    func `BVH lists every crossing in the linear scan's order`() throws {
        let geometry = try Self.vase()
        let bvh = try MeshBVH(geometry: geometry)
        var random = BVHTestRandom(seed: 5)
        var multiple = 0
        for matrix in Self.transforms {
            for ray in try Self.rays(count: 120, random: &random, around: matrix) {
                for culls in [true, false] {
                    let linear = try geometry.intersections(ray: ray, localToWorld: matrix, cullsBackFaces: culls)
                    let accelerated = try bvh.intersections(ray: ray, localToWorld: matrix, cullsBackFaces: culls)
                    #expect(accelerated == linear)
                    #expect(linear.first == (try geometry.intersection(ray: ray, localToWorld: matrix, cullsBackFaces: culls)))
                    #expect(zip(linear, linear.dropFirst()).allSatisfy { $0.distance <= $1.distance })
                    if linear.count >= 2 { multiple += 1 }
                }
            }
        }
        #expect(multiple > 50)
    }

    @Test
    func `Duplicate and degenerate triangles build and keep the lowest index on ties`() throws {
        let corners = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(2, 2, 0), SIMD3<Float>(1, 1, 0),
        ]
        let vertices = corners.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }
        // Points and segments first, then 40 copies of one triangle.
        var indices: [UInt32] = [0, 0, 0, 3, 3, 3, 0, 1, 1, 0, 3, 3]
        for _ in 0..<40 { indices += [0, 1, 2] }
        let geometry = try MeshGeometry(vertices: vertices, indices: indices)
        let bvh = try MeshBVH(geometry: geometry)
        #expect(bvh.triangleCount == 44)
        #expect(bvh.nodeCount <= 2 * bvh.triangleCount - 1)
        #expect(bvh.depth <= MeshBVH.maximumDepth)

        let down = try SceneRay(origin: SIMD3(0.25, 0.25, 1), direction: SIMD3(0, 0, -1))
        let hit = try #require(try bvh.intersection(ray: down, localToWorld: matrix_identity_float4x4))
        #expect(hit.triangleIndex == 4)
        #expect(hit == (try geometry.intersection(ray: down, localToWorld: matrix_identity_float4x4)))
        #expect(try bvh.intersections(ray: down, localToWorld: matrix_identity_float4x4).map(\.triangleIndex) == Array(4..<44))

        // Coincident points, repeated indices and collinear corners on the line y = x.
        let degenerate = try MeshGeometry(vertices: vertices, indices: [0, 0, 0, 1, 1, 1, 0, 4, 3, 3, 4, 0, 1, 1, 4])
        let degenerateBVH = try MeshBVH(geometry: degenerate)
        let across = try SceneRay(origin: SIMD3(1, 1, 1), direction: SIMD3(0, 0, -1))
        #expect(try degenerateBVH.intersection(ray: across, localToWorld: matrix_identity_float4x4, cullsBackFaces: false) == nil)
        #expect(try degenerate.intersection(ray: across, localToWorld: matrix_identity_float4x4, cullsBackFaces: false) == nil)
    }

    @Test
    func `Large meshes stay within the depth bound and query sub-linearly`() throws {
        let geometry = try MeshGeometry.uvSphere(radius: 1, segments: 256, rings: 128)
        let bvh = try MeshBVH(geometry: geometry)
        #expect(bvh.triangleCount == geometry.indices.count / 3)
        #expect(bvh.nodeCount <= 2 * bvh.triangleCount - 1)
        #expect(bvh.depth <= MeshBVH.maximumDepth)
        #expect(bvh.bounds == geometry.bounds)

        var random = BVHTestRandom(seed: 99)
        let rays = try Self.rays(count: 24, random: &random, around: matrix_identity_float4x4)
        let clock = ContinuousClock()
        var linearHits: [MeshRayHit?] = []
        let linearTime = try clock.measure {
            linearHits = try rays.map { try geometry.intersection(ray: $0, localToWorld: matrix_identity_float4x4) }
        }
        var acceleratedHits: [MeshRayHit?] = []
        let acceleratedTime = try clock.measure {
            acceleratedHits = try rays.map { try bvh.intersection(ray: $0, localToWorld: matrix_identity_float4x4) }
        }
        #expect(acceleratedHits == linearHits)
        // 65,024 triangles: a linear scan tests them all, the hierarchy a few dozen per ray.
        #expect(acceleratedTime * 10 < linearTime)
    }

    @Test
    func `Geometry overload accepts its own hierarchy and rejects another mesh's`() throws {
        let geometry = try MeshGeometry.uvSphere(radius: 1, segments: 12, rings: 6)
        let copy = geometry
        let rebuilt = try MeshGeometry.uvSphere(radius: 1, segments: 12, rings: 6)
        let other = try MeshGeometry.uvSphere(radius: 2, segments: 12, rings: 6)
        let bvh = try MeshBVH(geometry: geometry)
        let ray = try SceneRay(origin: SIMD3(0.1, 0.2, 5), direction: SIMD3(0, 0, -1))
        let expected = try #require(try geometry.intersection(ray: ray, localToWorld: matrix_identity_float4x4))

        #expect(try copy.intersection(ray: ray, localToWorld: matrix_identity_float4x4, cullsBackFaces: true, bvh: bvh) == expected)
        #expect(try rebuilt.intersection(ray: ray, localToWorld: matrix_identity_float4x4, cullsBackFaces: true, bvh: bvh) == expected)
        #expect(throws: MeshBVH.Failure.geometryMismatch) {
            try other.intersection(ray: ray, localToWorld: matrix_identity_float4x4, cullsBackFaces: true, bvh: bvh)
        }
        #expect(throws: ScenePickingError.invalidTransform) {
            try bvh.intersection(ray: ray, localToWorld: SceneTransform(scale: SIMD3(1, 0, 1)).matrix)
        }
    }

    @Test
    func `Box queries use the triangle, not only its bounds`() throws {
        // A 4 × 4 grid of unit cells; each cell is split along its bottom-left to top-right diagonal.
        let plane = try MeshGeometry.plane(width: 4, height: 4, columns: 4, rows: 4)
        let bvh = try MeshBVH(geometry: plane)
        let corner = try AxisAlignedBox(minimum: SIMD3(0.9, 0.1, -0.1), maximum: SIMD3(0.95, 0.15, 0.1))
        let found = bvh.triangles(intersecting: corner)
        #expect(found.count == 1)
        let triangle = try #require(found.first)
        let positions = Set((0..<3).map { plane.vertices[Int(plane.indices[triangle * 3 + $0])].position })
        #expect(positions == [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 1, 0)])

        let above = try AxisAlignedBox(minimum: SIMD3(-1, -1, 0.01), maximum: SIMD3(1, 1, 1))
        #expect(bvh.triangles(intersecting: above).isEmpty)
        // Touching counts: 4 inner cells and 8 edge cells contribute both triangles. Corner
        // cells meet the box at one point, which both triangles share only where the
        // diagonal ends: the top-right and bottom-left cells.
        let touching = try AxisAlignedBox(minimum: SIMD3(-1, -1, 0), maximum: SIMD3(1, 1, 1))
        #expect(bvh.triangles(intersecting: touching).count == 4 * 2 + 8 * 2 + 2 * 2 + 2 * 1)
    }

    @Test
    func `Box queries include contained triangles and exclude disjoint bounds`() throws {
        let geometry = try MeshGeometry.uvSphere(radius: 1, segments: 32, rings: 16)
        let bvh = try MeshBVH(geometry: geometry)
        var random = BVHTestRandom(seed: 3)
        for _ in 0..<40 {
            let center = random.vector(in: -1...1)
            let half = random.vector(in: 0.05...0.6)
            let box = try AxisAlignedBox(minimum: center - half, maximum: center + half)
            let found = bvh.triangles(intersecting: box)
            #expect(found == Array(Set(found)).sorted())
            let selected = Set(found)
            for triangle in 0..<(geometry.indices.count / 3) {
                let corners = (0..<3).map { geometry.vertices[Int(geometry.indices[triangle * 3 + $0])].position }
                let bounds = try AxisAlignedBox(enclosing: corners)
                if corners.contains(where: box.contains) { #expect(selected.contains(triangle)) }
                if !bounds.intersects(box) { #expect(!selected.contains(triangle)) }
            }
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Scene picking through the cached hierarchy matches the linear scan`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try Self.vase()
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba8Unorm, geometry: geometry)
        let transform = SceneTransform(
            translation: SIMD3(0.2, -0.1, -4), rotation: simd_quatf(angle: 0.4, axis: SIMD3(1, 0, 0)), scale: SIMD3(1.4, 1.1, 1)
        )
        let node = SceneNode(transform: transform, mesh: renderer)
        let graph = SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20))
        var hits = 0
        for pass in 0..<2 {
            for row in 0...12 {
                for column in 0...12 {
                    let point = SIMD2(Float(column) / 12, Float(row) / 12)
                    let ray = try graph.camera.ray(at: point, aspectRatio: 1.25, cameraTransform: graph.cameraTransform)
                    let expected = try geometry.intersection(ray: ray, localToWorld: transform.matrix)
                    let picked = try graph.pick(at: point, aspectRatio: 1.25)
                    #expect(picked?.intersection == expected)
                    #expect(picked.map { $0.nodeID == node.id } ?? true)
                    if pass == 0, expected != nil { hits += 1 }
                }
            }
        }
        #expect(hits > 20)
    }
}
