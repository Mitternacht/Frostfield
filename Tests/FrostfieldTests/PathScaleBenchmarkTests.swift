import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Preparation and trace times for large procedural meshes under each acceleration. Runs only
/// when FROSTFIELD_PATH_BENCHMARK is set, preferably in a Release build:
///
///     TEST_RUNNER_FROSTFIELD_PATH_BENCHMARK=1 xcodebuild test -configuration Release \
///       -scheme Frostfield-Package -destination 'platform=macOS,arch=arm64' \
///       -only-testing:FrostfieldTests/PathScaleBenchmarkTests
///
/// Each result prints as one `PATHSCALE` JSON line.
@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["FROSTFIELD_PATH_BENCHMARK"] != nil, "Set FROSTFIELD_PATH_BENCHMARK"),
    .serialized
)
struct PathScaleBenchmarkTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    /// A 20 × 10 m terrain of `columns` × `rows` quads with ridges and bumps, two triangles per quad.
    private static func terrain(columns: Int, rows: Int) throws -> MeshGeometry {
        func height(_ x: Float, _ z: Float) -> Float {
            0.35 * sin(x * 0.9) * cos(z * 1.3) + 0.12 * sin(x * 4.1 + z * 2.7) + 0.03 * sin(x * 23 + 1.3) * sin(z * 19)
        }
        var vertices: [MeshVertex] = []
        vertices.reserveCapacity((columns + 1) * (rows + 1))
        let step = SIMD2(20 / Float(columns), 10 / Float(rows))
        for row in 0...rows {
            for column in 0...columns {
                let x = Float(column) * step.x - 10
                let z = Float(row) * step.y - 5
                let h = height(x, z)
                let dx = (height(x + 0.01, z) - height(x - 0.01, z)) / 0.02
                let dz = (height(x, z + 0.01) - height(x, z - 0.01)) / 0.02
                vertices.append(MeshVertex(
                    position: SIMD3(x, h, z), normal: simd_normalize(SIMD3(-dx, 1, -dz)),
                    textureCoordinate: SIMD2(Float(column) / Float(columns), Float(row) / Float(rows))
                ))
            }
        }
        var indices: [UInt32] = []
        indices.reserveCapacity(columns * rows * 6)
        for row in 0..<rows {
            for column in 0..<columns {
                let top = UInt32(row * (columns + 1) + column)
                let bottom = top + UInt32(columns + 1)
                indices += [top, bottom, top + 1, top + 1, bottom, bottom + 1]
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    @Test("Preparation and 1280 × 720 trace time for 100,000 to 5,000,000 triangles", arguments: [100_000, 1_000_000, 5_000_000])
    func proceduralMesh(triangles: Int) throws {
        let rows = Int((Double(triangles) / 4).squareRoot().rounded())
        let columns = (triangles / 2 + rows - 1) / rows
        let geometry = try Self.terrain(columns: columns, rows: rows)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.6, 0.55, 0.5, 1),
                alphaMode: .opaque
            ))
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: 50 * .pi / 180, near: 0.05, far: 100)
        let view = SceneTransform(
            translation: SIMD3(0, 2.4, 7), rotation: simd_quatf(angle: -0.32, axis: SIMD3(1, 0, 0))
        )
        let graph = SceneGraph(roots: [SceneNode(mesh: mesh)], camera: camera, cameraTransform: view)
        let lighting = try PathTracer.Lighting(
            lights: [try PathTracer.AreaLight(
                center: SIMD3(3, 6, 2), aim: .zero, size: SIMD2(3, 2), radiance: SIMD3(repeating: 8)
            )],
            environment: SIMD3(0.25, 0.3, 0.4)
        )
        let accelerations: [PathTracingScene.Options.Acceleration] = fixture.device.supportsRaytracing ? [.software, .metal] : [.software]
        for acceleration in accelerations {
            // A small warm-up preparation loads the builders' pipelines before timing.
            _ = try PathTracingScene(
                queue: fixture.queue, graph: SceneGraph(roots: [SceneNode(mesh: try MeshRenderer(
                    device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube(), material: .physicallyBased(.init())
                ))], camera: camera), options: .init(acceleration: acceleration)
            )
            var builds: [Double] = []
            var scene: PathTracingScene?
            for _ in 0..<3 {
                scene = nil
                let start = ContinuousClock.now
                scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
                let build = ContinuousClock.now - start
                builds.append(Double(build.components.seconds) + Double(build.components.attoseconds) * 1e-18)
            }
            let prepared = try #require(scene)
            let trace = try traceTimes(prepared, camera: camera, view: view, lighting: lighting, name: "terrain-\(triangles)-\(acceleration)")
            try report([
                "scene": "terrain", "acceleration": "\(acceleration)", "triangles": prepared.triangleCount,
                "instances": prepared.statistics.instanceCount,
                "buildSecondsMedian": builds.sorted()[1], "buildSecondsMinimum": builds.min()!,
                "traceMillisecondsP50": trace.median, "traceMillisecondsP95": trace.p95, "traceMillisecondsMinimum": trace.minimum,
                "frames": trace.count, "bufferBytes": prepared.statistics.bufferByteCount,
                "accelerationStructureBytes": prepared.statistics.accelerationStructureByteCount,
                "softwareNodes": prepared.nodeCount,
            ])
        }
    }

    /// GPU time of 1-sample, 8-bounce frames at 1280 × 720: one warm-up, then 30 restarts with new seeds.
    private func traceTimes(
        _ scene: PathTracingScene, camera: PerspectiveCamera, view: SceneTransform, lighting: PathTracer.Lighting, name: String
    ) throws -> (median: Double, p95: Double, minimum: Double, count: Int) {
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: lighting)
        let accumulation = try tracer.makeAccumulation(width: 1_280, height: 720)
        let target = try fixture.output(width: 1_280, height: 720)
        let timing = try PathTracer.Timing(device: fixture.device)
        var durations: [Double] = []
        for frame in 0..<31 {
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            _ = try tracer.encode(
                camera: camera, cameraTransform: view, destination: target, accumulation: accumulation,
                commandBuffer: buffer, settings: .init(maxBounces: 8, seed: UInt32(frame)), timing: timing
            )
            try fixture.finish(buffer)
            if frame > 0 { durations.append(try #require(timing.duration) * 1_000) }
        }
        try PathFullResolutionTextureTests.save(try fixture.readback(target), width: 1_280, height: 720, name: name)
        durations.sort()
        return (durations[durations.count / 2], durations[durations.count * 95 / 100], durations[0], durations.count)
    }

    private func report(_ values: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: values, options: [.sortedKeys])
        print("PATHSCALE \(String(decoding: data, as: UTF8.self))")
    }
}
