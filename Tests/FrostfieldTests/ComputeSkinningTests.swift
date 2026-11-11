import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ComputeSkinningTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    /// A deterministic generator so failures reproduce.
    private struct Random {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24)
        }
        mutating func signed() -> Float { next() * 2 - 1 }
    }

    /// A sphere with tangents of both handednesses and random four-joint influences.
    private func skinnedSphere(
        seed: UInt64, jointCount: Int, segments: Int = 48, rings: Int = 24
    ) throws -> (MeshGeometry, MeshSkin) {
        let sphere = try MeshGeometry.uvSphere(radius: 1.2, segments: segments, rings: rings)
        var random = Random(state: seed)
        let vertices = sphere.vertices.enumerated().map { index, vertex in
            let helper: SIMD3<Float> = abs(vertex.normal.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            let tangent = simd_normalize(simd_cross(helper, vertex.normal))
            return MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                secondaryTextureCoordinate: SIMD2(vertex.textureCoordinate.y, 0.5),
                tangent: SIMD4(tangent, index.isMultiple(of: 2) ? 1 : -1),
                color: SIMD4(1, 0.5, 0.25, index.isMultiple(of: 2) ? 0.5 : 1)
            )
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: sphere.indices)
        let influences = geometry.vertices.map { _ in
            VertexSkinInfluence(
                joints: SIMD4((0..<4).map { _ in UInt16(min(Int(random.next() * Float(jointCount)), jointCount - 1)) }),
                weights: SIMD4(random.next(), random.next(), random.next(), random.next() + 0.01)
            )
        }
        return (geometry, try MeshSkin(jointCount: jointCount, influences: influences))
    }

    /// Rigid and nonuniformly scaled joints, all mirrored when requested. Rotations stay
    /// within π/3 so blended matrices are well conditioned.
    private func palette(seed: UInt64, jointCount: Int, mirrored: Bool = false) throws -> JointPalette {
        var random = Random(state: seed)
        return try JointPalette(matrices: (0..<jointCount).map { joint in
            let axis = simd_normalize(SIMD3(random.signed(), random.signed(), random.signed()) + SIMD3(0, 0, 0.01))
            var scale = joint % 2 == 1 ? SIMD3<Float>(1.5, 0.7, 1.1) : SIMD3<Float>(repeating: 1)
            if mirrored { scale.x = -scale.x }
            return SceneTransform(
                translation: SIMD3(random.signed(), random.signed(), random.signed()) * 2,
                rotation: simd_quatf(angle: random.signed() * .pi / 3, axis: axis), scale: scale
            ).matrix
        })
    }

    private func expectMatches(_ gpu: MeshGeometry, _ cpu: MeshGeometry) {
        #expect(gpu.indices == cpu.indices)
        var worst: Float = 0
        for (a, b) in zip(gpu.vertices, cpu.vertices) {
            worst = max(worst, simd_reduce_max(abs(a.position - b.position)), simd_reduce_max(abs(a.normal - b.normal)))
            #expect(a.color == b.color)
            #expect((a.tangent == nil) == (b.tangent == nil))
            if let x = a.tangent, let y = b.tangent { worst = max(worst, simd_reduce_max(abs(x - y))) }
            #expect(a.textureCoordinate == b.textureCoordinate && a.secondaryTextureCoordinate == b.secondaryTextureCoordinate)
        }
        #expect(worst <= 1e-5, "largest difference \(worst)")
    }

    @Test
    func `GPU skinning equals MeshSkin.deform for many meshes in one pass`() throws {
        let skinning = try ComputeSkinning(device: device)
        let (first, firstSkin) = try skinnedSphere(seed: 7, jointCount: 9)
        let (second, secondSkin) = try skinnedSphere(seed: 21, jointCount: 64)
        let a = try skinning.makeMesh(geometry: first, skin: firstSkin)
        let b = try skinning.makeMesh(geometry: second, skin: secondSkin)
        let firstPose = try palette(seed: 3, jointCount: 9)
        let secondPose = try palette(seed: 5, jointCount: 64, mirrored: true)
        let skinned = try skinning.skin([(a, firstPose), (b, secondPose)], queue: queue)
        expectMatches(skinned[0], try firstSkin.deform(first, using: firstPose))
        expectMatches(skinned[1], try secondSkin.deform(second, using: secondPose))

        // Re-posing rewrites the output.
        let again = try palette(seed: 11, jointCount: 9)
        expectMatches(try skinning.skin([(a, again)], queue: queue)[0], try firstSkin.deform(first, using: again))

        #expect(throws: ComputeSkinning.Failure.duplicateMesh) { try skinning.skin([(a, firstPose), (a, again)], queue: queue) }
        #expect(throws: SkinningError.paletteCount) { try skinning.skin([(a, secondPose)], queue: queue) }
    }

    @Test
    func `Path tracing scenes rebuild skinned nodes from compute skinning like the CPU path`() throws {
        let fixture = try PathTestFixture()
        let strip = try MeshGeometry.plane(width: 2, height: 0.4, columns: 8, rows: 1)
        let influences = strip.vertices.map { vertex in
            VertexSkinInfluence(joints: SIMD4(vertex.position.x > 0.01 ? 1 : 0, 0, 0, 0), weights: SIMD4(1, 0, 0, 0))
        }
        let skinned = try fixture.pbr(strip, emission: SIMD3(1, 0.5, 0.25), skin: try MeshSkin(jointCount: 2, influences: influences))
        func pose(_ angle: Float) throws -> JointPalette {
            try JointPalette(matrices: [
                matrix_identity_float4x4, SceneTransform(rotation: simd_quatf(angle: angle, axis: SIMD3(0, 0, 1))).matrix,
            ])
        }
        let id = UUID()
        var graph = SceneGraph(roots: [
            SceneNode(id: id, transform: .init(translation: SIMD3(0, -0.3, -3)), mesh: skinned, jointPalette: try pose(0.9)),
        ], camera: fixture.camera)
        let gpu = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        gpu.computeSkinning = try ComputeSkinning(device: fixture.device)
        let cpu = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        graph.roots[0].jointPalette = try pose(-0.7)
        try gpu.updateGeometry(for: [id], in: graph)
        try cpu.updateGeometry(for: [id], in: graph)
        #expect(gpu.statistics.bottomLevelBuildCount == 2 && gpu.revision == 1)
        // The rebuild used the compute pass, and the CPU scene never does.
        #expect(gpu.statistics.computeSkinnedBuildCount == 1)
        #expect(cpu.statistics.computeSkinnedBuildCount == 0)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 4)
        let lighting = try PathTracer.Lighting(backgroundVisible: false)
        let a = try fixture.trace(try PathTracer(queue: fixture.queue, scene: gpu), width: 48, height: 32, samples: 4, settings: settings, lighting: lighting)
        let b = try fixture.trace(try PathTracer(queue: fixture.queue, scene: cpu), width: 48, height: 32, samples: 4, settings: settings, lighting: lighting)
        #expect(PathTestFixture.maximumDifference(a, b) < 1e-3)
        #expect(a.map(\.w) == b.map(\.w))
        #expect(a.contains { $0.w > 0.5 })
        // A second update reuses the uploaded bind pose and follows the new pose.
        graph.roots[0].jointPalette = try pose(0.3)
        try gpu.updateGeometry(for: [id], in: graph)
        try cpu.updateGeometry(for: [id], in: graph)
        let c = try fixture.trace(try PathTracer(queue: fixture.queue, scene: gpu), width: 48, height: 32, samples: 4, settings: settings, lighting: lighting)
        let d = try fixture.trace(try PathTracer(queue: fixture.queue, scene: cpu), width: 48, height: 32, samples: 4, settings: settings, lighting: lighting)
        #expect(PathTestFixture.maximumDifference(c, d) < 1e-3)
        #expect(gpu.statistics.computeSkinnedBuildCount == 2)
        #expect(zip(a, c).filter { ($0.w > 0.5) != ($1.w > 0.5) }.count > 20)
    }

    /// Times one compute pass over ten 64-joint meshes against `MeshSkin.deform` on
    /// the CPU. Skipped unless `FROSTFIELD_BENCHMARK=1`; with `FROSTFIELD_BENCHMARK_OUTPUT`
    /// set to a path it also writes the per-iteration samples there as JSON.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["FROSTFIELD_BENCHMARK"] == "1", "Set FROSTFIELD_BENCHMARK=1 to time skinning"))
    func `Benchmark compute skinning against MeshSkin.deform`() throws {
        let skinning = try ComputeSkinning(device: device)
        var meshes: [(mesh: ComputeSkinning.Mesh, geometry: MeshGeometry, skin: MeshSkin, palette: JointPalette)] = []
        for index in 0..<10 {
            let (geometry, skin) = try skinnedSphere(seed: UInt64(100 + index), jointCount: 64, segments: 256, rings: 128)
            meshes.append((try skinning.makeMesh(geometry: geometry, skin: skin), geometry, skin, try palette(seed: UInt64(200 + index), jointCount: 64)))
        }
        let jobs = meshes.map { ($0.mesh, $0.palette) }
        func milliseconds(_ body: () throws -> Void) rethrows -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            try body()
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        }
        func dispatch() throws -> Double {
            let buffer = try #require(queue.makeCommandBuffer())
            try skinning.encode(jobs, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            #expect(buffer.status == .completed)
            return (buffer.gpuEndTime - buffer.gpuStartTime) * 1000
        }
        for _ in 0..<5 {
            _ = try dispatch()
            _ = try skinning.skin(jobs, queue: queue)
        }
        // Each run times 20 back-to-back iterations of one path at a time, so the GPU stays busy
        // between dispatches instead of idling through the CPU timing.
        var samples: [String: [[Double]]] = ["gpu": [], "dispatchAndWait": [], "withReadback": [], "cpuDeform": []]
        for _ in 0..<3 {
            var gpu: [Double] = [], wall: [Double] = [], readback: [Double] = [], cpu: [Double] = []
            for _ in 0..<20 {
                var time = 0.0
                wall.append(try milliseconds { time = try dispatch() })
                gpu.append(time)
            }
            for _ in 0..<20 { readback.append(try milliseconds { _ = try skinning.skin(jobs, queue: queue) }) }
            for _ in 0..<20 {
                cpu.append(try milliseconds { for item in meshes { _ = try item.skin.deform(item.geometry, using: item.palette) } })
            }
            samples["gpu"]!.append(gpu)
            samples["dispatchAndWait"]!.append(wall)
            samples["withReadback"]!.append(readback)
            samples["cpuDeform"]!.append(cpu)
        }
        func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
        func format(_ value: Double) -> String { String(format: "%.3f", value) }
        for (name, runs) in samples.sorted(by: { $0.key < $1.key }) {
            let all = runs.flatMap { $0 }.sorted()
            print(
                "FROSTFIELD_BENCHMARK compute skinning \(name) ms: run medians [\(runs.map(median).map(format).joined(separator: ", "))], "
                    + "p10 \(format(all[all.count / 10])), p90 \(format(all[all.count * 9 / 10])), min \(format(all[0])), max \(format(all[all.count - 1]))"
            )
        }
        print("FROSTFIELD_BENCHMARK compute skinning: \(meshes.count) meshes of \(meshes[0].geometry.vertices.count) vertices, 64 joints, \(device.name)")
        if let path = ProcessInfo.processInfo.environment["FROSTFIELD_BENCHMARK_OUTPUT"] {
            let report: [String: Any] = [
                "device": device.name, "meshes": meshes.count, "verticesPerMesh": meshes[0].geometry.vertices.count,
                "joints": 64, "runs": 3, "iterationsPerRun": 20, "warmUpIterations": 5, "samplesMilliseconds": samples,
            ]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path))
        }
        expectMatches(try skinning.skin(jobs, queue: queue)[0], try meshes[0].skin.deform(meshes[0].geometry, using: meshes[0].palette))
    }
}
