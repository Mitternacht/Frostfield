import Foundation
import FrostfieldShaderTypes
import Metal
import Testing
import simd
@testable import Frostfield

private let systemDevice = MTLCreateSystemDefaultDevice()

/// The vertex layout the mesh pipeline declares must match how `makeMesh`
/// writes `SIMD3<Float>` arrays and the indices the shader names.
struct MeshVertexLayoutTests {
    @Test func `Vertex streams use the stride of SIMD3<Float>`() {
        #expect(MemoryLayout<SIMD3<Float>>.stride == Int(FF_MESH_VERTEX_STREAM_STRIDE))
        #expect(FF_MESH_VERTEX_STREAM_STRIDE == 16)
    }

    @Test func `The vertex descriptor reads positions and normals from their own streams`() {
        let descriptor = MeshRenderer.vertexDescriptor
        let position = descriptor.attributes[Int(FF_MESH_ATTRIBUTE_POSITION)]!
        let normal = descriptor.attributes[Int(FF_MESH_ATTRIBUTE_NORMAL)]!
        #expect(position.format == .float3 && position.offset == 0)
        #expect(normal.format == .float3 && normal.offset == 0)
        #expect(position.bufferIndex == Int(FF_MESH_POSITION_BUFFER_INDEX))
        #expect(normal.bufferIndex == Int(FF_MESH_NORMAL_BUFFER_INDEX))
        #expect(descriptor.layouts[Int(FF_MESH_POSITION_BUFFER_INDEX)]!.stride == 16)
        #expect(descriptor.layouts[Int(FF_MESH_NORMAL_BUFFER_INDEX)]!.stride == 16)
        // Neither stream collides with the transform record's buffer.
        #expect(FF_MESH_POSITION_BUFFER_INDEX != FF_TRANSFORM_UNIFORMS_BUFFER_INDEX)
        #expect(FF_MESH_NORMAL_BUFFER_INDEX != FF_TRANSFORM_UNIFORMS_BUFFER_INDEX)
    }
}

/// A CPU reference for what a camera sees of an axis-aligned box: cast a ray
/// through a pixel and find the face it enters first. It shares no code with
/// the rasterizer, culling, or depth test.
private struct BoxRayCaster {
    let inverseViewProjection: simd_float4x4
    let inverseModel: simd_float4x4
    let half: Float

    init(camera: Camera, transform: Transform, aspectRatio: Float, size: Float) throws {
        inverseViewProjection = try camera.viewProjectionMatrix(aspectRatio: aspectRatio).inverse
        inverseModel = transform.matrix.inverse
        half = size / 2
    }

    enum Hit: Equatable {
        /// The ray enters the face with this model-space normal, at least
        /// `margin` (in model units) away from the face's edges.
        case face(normal: SIMD3<Float>, margin: Float)
        /// The ray misses a box 10 % larger than the real one.
        case clearMiss
        /// Too close to a silhouette to judge.
        case ambiguous
    }

    func hit(ndc: SIMD2<Float>) -> Hit {
        // Unproject the near and far points of the pixel's ray into model space.
        func modelPoint(depth: Float) -> SIMD3<Float> {
            let world = inverseViewProjection * SIMD4(ndc.x, ndc.y, depth, 1)
            let model = inverseModel * (world / world.w)
            return SIMD3(model.x, model.y, model.z) / model.w
        }
        let origin = modelPoint(depth: 0)
        let direction = modelPoint(depth: 1) - origin
        if !intersects(origin: origin, direction: direction, half: half * 1.1) { return .clearMiss }

        var entry: Float = -.infinity, exit: Float = .infinity, axis = 0
        for i in 0..<3 {
            let t1 = (-half - origin[i]) / direction[i]
            let t2 = (half - origin[i]) / direction[i]
            if min(t1, t2) > entry { entry = min(t1, t2); axis = i }
            exit = min(exit, max(t1, t2))
        }
        guard entry <= exit, entry > 0 else { return .ambiguous }
        let point = origin + entry * direction
        let margin = (0..<3).filter { $0 != axis }.map { half - abs(point[$0]) }.min()!
        var normal = SIMD3<Float>.zero
        normal[axis] = direction[axis] > 0 ? -1 : 1
        return .face(normal: normal, margin: margin)
    }

    private func intersects(origin: SIMD3<Float>, direction: SIMD3<Float>, half: Float) -> Bool {
        var entry: Float = -.infinity, exit: Float = .infinity
        for i in 0..<3 {
            let t1 = (-half - origin[i]) / direction[i]
            let t2 = (half - origin[i]) / direction[i]
            entry = max(entry, min(t1, t2))
            exit = min(exit, max(t1, t2))
        }
        return entry <= exit && exit > 0
    }
}

/// A deterministic view of the unit cube for the reference comparison.
struct CubePose: Sendable, CustomTestStringConvertible {
    var name: String
    var transform = Transform.identity
    var camera = CubePose.frontCamera
    var testDescription: String { name }

    static let frontCamera = Camera(
        position: [0, 0, 3],
        projection: try! PerspectiveProjection(verticalFieldOfView: .pi / 3, near: 0.1, far: 10)
    )

    static let all: [CubePose] = [
        CubePose(name: "front face"),
        CubePose(name: "quarter turn about y", transform: Transform(rotation: simd_quatf(angle: .pi / 2, axis: [0, 1, 0]))),
        CubePose(name: "quarter turn about x", transform: Transform(rotation: simd_quatf(angle: .pi / 2, axis: [1, 0, 0]))),
        CubePose(
            name: "corner view",
            transform: Transform(rotation: simd_quatf(angle: 0.6, axis: [1, 0, 0]) * simd_quatf(angle: .pi / 4, axis: [0, 1, 0]))
        ),
        CubePose(
            name: "moved, turned, and stretched",
            transform: Transform(
                translation: [0.3, -0.2, -1],
                rotation: simd_quatf(angle: 0.5, axis: simd_normalize([1, 1, 0])),
                scale: [1.5, 0.7, 1.2]
            )
        ),
        CubePose(
            name: "camera behind the cube",
            camera: Camera(
                position: [0.4, 0.6, -3],
                orientation: simd_quatf(angle: .pi, axis: [0, 1, 0]),
                projection: frontCamera.projection
            )
        ),
    ]
}

/// Renders indexed meshes off screen on the system GPU with depth testing
/// and back-face culling, and reads the color and depth attachments back.
@Suite(.enabled(if: systemDevice != nil))
struct MeshRenderingTests {
    let device: any MTLDevice
    let completions: AsyncStream<FrameCompletion>
    let reportCompletion: AsyncStream<FrameCompletion>.Continuation

    init() throws {
        device = try #require(systemDevice)
        (completions, reportCompletion) = AsyncStream.makeStream(of: FrameCompletion.self)
    }

    func makeRenderer() throws -> MeshRenderer {
        let report = reportCompletion
        return try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm, onFrameCompleted: { report.yield($0) })
    }

    func makeTarget(width: Int = 128, height: Int = 128) throws -> OffscreenTarget {
        try OffscreenTarget(device: device, width: width, height: height, depthPixelFormat: .depth32Float)
    }

    /// The color the diagnostic shader gives a face: its model-space normal
    /// mapped to [0, 255] per channel.
    static func faceColor(_ normal: SIMD3<Float>) -> SIMD3<Int> {
        SIMD3<Int>((normal * 0.5 + 0.5) * 255 + 0.5, rounding: .down)
    }

    /// Whether a pixel shows the face with this normal, within 2 of 255 per
    /// channel for unorm rounding.
    static func pixel(_ pixel: OffscreenTarget.Pixel, showsFace normal: SIMD3<Float>) -> Bool {
        let expected = faceColor(normal)
        return abs(Int(pixel.red) - expected.x) <= 2
            && abs(Int(pixel.green) - expected.y) <= 2
            && abs(Int(pixel.blue) - expected.z) <= 2
    }

    /// Every pixel of a 128 × 128 render is compared with the ray-cast
    /// reference: where a ray enters a face away from its edges, the pixel
    /// shows that face's color; where it clearly misses, the pixel is clear.
    /// A wrong winding convention shows back faces, a broken depth test shows
    /// whichever face was drawn last, and a wrong transform moves the faces.
    @Test(arguments: CubePose.all)
    func `Visible faces match a ray-cast reference`(pose: CubePose) async throws {
        let renderer = try makeRenderer()
        let mesh = try renderer.makeMesh(.cube())
        let target = try makeTarget()
        try await render(renderer, mesh, pose.transform, pose.camera, into: target)

        let reference = try BoxRayCaster(camera: pose.camera, transform: pose.transform, aspectRatio: 1, size: 1)
        var facesSeen = Set<SIMD3<Float>>()
        var compared = 0
        var mismatches: [String] = []
        for y in 0..<128 {
            for x in 0..<128 {
                let pixel = target.pixel(x: x, y: y)
                switch reference.hit(ndc: target.ndc(x: x, y: y)) {
                case .face(let normal, let margin) where margin > 0.03:
                    if !Self.pixel(pixel, showsFace: normal) {
                        mismatches.append("(\(x), \(y)) expected \(Self.faceColor(normal)) for \(normal), got \(pixel)")
                    }
                    facesSeen.insert(normal)
                    compared += 1
                case .clearMiss:
                    if !pixel.isClear { mismatches.append("(\(x), \(y)) expected clear, got \(pixel)") }
                    compared += 1
                case .face, .ambiguous:
                    break
                }
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) of \(compared) pixels differ, first: \(mismatches.prefix(5))")
        #expect(compared > 128 * 128 * 3 / 4, "only \(compared) pixels were unambiguous")
        #expect(!facesSeen.isEmpty)
    }

    /// Reversing every triangle turns the cube inside out: seen from the
    /// front, the near face now winds clockwise and is culled, and the inside
    /// of the far face (wound counter-clockwise from here) shows instead.
    @Test func `Back faces are culled by winding`() async throws {
        let renderer = try makeRenderer()
        let cube = IndexedGeometry.cube()
        var reversed: [UInt32] = []
        for triangle in 0..<cube.triangleCount {
            let i = cube.indices[(triangle * 3)..<(triangle * 3 + 3)]
            reversed += [i[i.startIndex], i[i.startIndex + 2], i[i.startIndex + 1]]
        }
        let insideOut = try IndexedGeometry(positions: cube.positions, normals: cube.normals, indices: reversed)

        let outside = try makeTarget()
        try await render(renderer, try renderer.makeMesh(cube), .identity, CubePose.frontCamera, into: outside)
        let inside = try makeTarget()
        try await render(renderer, try renderer.makeMesh(insideOut), .identity, CubePose.frontCamera, into: inside)

        let centre = SIMD2<Float>(0, 0)
        let front = outside.pixel(at: centre), back = inside.pixel(at: centre)
        #expect(Self.pixel(front, showsFace: [0, 0, 1]), "outside: \(front)")
        #expect(Self.pixel(back, showsFace: [0, 0, -1]), "inside out: \(back)")
    }

    /// Two quads face the camera: a small near one tagged with normal +z and a
    /// larger far one tagged +x. Whichever is drawn first, the near quad wins
    /// where they overlap, and the far quad shows around it.
    @Test(arguments: [false, true])
    func `The nearest surface wins regardless of draw order`(farFirst: Bool) async throws {
        let near: [SIMD3<Float>] = [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        let far: [SIMD3<Float>] = [[-1.5, -1.5, -0.5], [1.5, -1.5, -0.5], [1.5, 1.5, -0.5], [-1.5, 1.5, -0.5]]
        let nearQuad: [UInt32] = [0, 1, 2, 0, 2, 3]
        let farQuad: [UInt32] = [4, 5, 6, 4, 6, 7]
        let geometry = try IndexedGeometry(
            positions: near + far,
            normals: Array(repeating: [0, 0, 1], count: 4) + Array(repeating: [1, 0, 0], count: 4),
            indices: farFirst ? farQuad + nearQuad : nearQuad + farQuad
        )
        let renderer = try makeRenderer()
        let target = try makeTarget()
        try await render(renderer, try renderer.makeMesh(geometry), .identity, CubePose.frontCamera, into: target)

        let centre = target.pixel(at: [0, 0])
        #expect(Self.pixel(centre, showsFace: [0, 0, 1]), "centre: \(centre)")
        // NDC 0.5 is outside the near quad (its edge projects to 0.346) and
        // inside the far quad (1.5 at distance 3.5 projects to 0.742).
        let around = target.pixel(at: [0.5, 0.5])
        #expect(Self.pixel(around, showsFace: [1, 0, 0]), "around: \(around)")
    }

    /// The depth attachment holds Metal depth in [0, 1]: the cube's front face
    /// at distance 2.5 has the depth the projection predicts, below the
    /// cleared value of 1 that remains outside the cube.
    @Test func `Depth is written in Metal's zero-to-one range`() async throws {
        let renderer = try makeRenderer()
        let target = try makeTarget()
        let camera = CubePose.frontCamera
        try await render(renderer, try renderer.makeMesh(.cube()), .identity, camera, into: target)

        let queue = try #require(device.makeCommandQueue())
        let depths = try await target.depthValues(queue: queue)
        let projection = try camera.projection.matrix(aspectRatio: 1)
        let clip = projection * SIMD4<Float>(0, 0, -2.5, 1)
        let expected = clip.z / clip.w
        #expect(expected > 0.96 && expected < 0.98, "projection depth \(expected)")

        let centre = target.pixelPosition(ndc: [0, 0])
        let corner = target.pixelPosition(ndc: [-0.95, 0.95])
        #expect(abs(depths[centre.y * 128 + centre.x] - expected) < 1e-4, "centre depth \(depths[centre.y * 128 + centre.x])")
        #expect(depths[corner.y * 128 + corner.x] == 1, "cleared depth")
        #expect(depths.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    /// The renderer creates its pipeline, depth state, and uniform buffers
    /// once; `makeMesh` adds exactly the mesh's three buffers; 100 frames of
    /// transform and camera changes add nothing.
    @Test func `Transform-only updates reuse the mesh buffers`() async throws {
        let renderer = try makeRenderer()
        let initial = renderer.resourceStatistics
        #expect(initial.renderPipelinesCreated == 1)
        #expect(initial.depthStencilStatesCreated == 1)
        #expect(initial.buffersAllocated == renderer.maxFramesInFlight)

        let cube = IndexedGeometry.cube()
        let mesh = try renderer.makeMesh(cube)
        let afterMesh = renderer.resourceStatistics
        #expect(afterMesh.buffersAllocated == initial.buffersAllocated + 3)
        #expect(afterMesh.bufferBytesAllocated == initial.bufferBytesAllocated + 24 * 16 * 2 + 36 * 4)
        let meshBuffers = [mesh.positionBuffer, mesh.normalBuffer, mesh.indexBuffer].map { ObjectIdentifier($0) }

        let target = try makeTarget(width: 64, height: 64)
        var camera = CubePose.frontCamera
        for step in 0..<100 {
            let angle = Float(step) * 0.07
            camera.position.z = 3 + sin(angle)
            let transform = Transform(
                translation: [sin(angle) * 0.3, 0, 0],
                rotation: simd_quatf(angle: angle, axis: simd_normalize([1, 1, 0.3]))
            )
            try await render(renderer, mesh, transform, camera, into: target)
        }

        #expect(renderer.resourceStatistics == afterMesh)
        #expect([mesh.positionBuffer, mesh.normalBuffer, mesh.indexBuffer].map { ObjectIdentifier($0) } == meshBuffers)
        #expect(renderer.statistics.submitted == 100)
        #expect(renderer.statistics.failed == 0)
    }

    @Test func `A mesh uploads the geometry unchanged`() throws {
        let renderer = try makeRenderer()
        let cube = IndexedGeometry.cube(size: 0.75)
        let mesh = try renderer.makeMesh(cube)
        #expect(mesh.vertexCount == 24 && mesh.indexCount == 36)
        func contents<T>(_ buffer: any MTLBuffer, as type: T.Type, count: Int) -> [T] {
            Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: T.self), count: count))
        }
        #expect(contents(mesh.positionBuffer, as: SIMD3<Float>.self, count: 24) == cube.positions)
        #expect(contents(mesh.normalBuffer, as: SIMD3<Float>.self, count: 24) == cube.normals)
        #expect(contents(mesh.indexBuffer, as: UInt32.self, count: 36) == cube.indices)
    }

    @Test func `A missing device throws noMetalDevice`() {
        #expect(throws: RendererError.noMetalDevice) {
            try MeshRenderer(device: nil, colorPixelFormat: .bgra8Unorm)
        }
    }

    // MARK: Helpers

    private func render(_ renderer: MeshRenderer, _ mesh: Mesh, _ transform: Transform, _ camera: Camera, into target: OffscreenTarget) async throws {
        let outcome = renderer.drawFrame(mesh: mesh, transform: transform, camera: camera) { target.frameTarget }
        guard case .submitted(let frame) = outcome else {
            Issue.record("Frame was not submitted: \(outcome)")
            return
        }
        var iterator = completions.makeAsyncIterator()
        let completion = try #require(await iterator.next())
        #expect(completion == FrameCompletion(frame: frame, succeeded: true))
    }
}

/// Drawing into a pass without the depth attachment the pipeline tests
/// against would be a Metal validation error. The renderer stops first, with
/// a message that says how to fix the target.
@Suite(.enabled(if: systemDevice != nil))
struct MeshRendererPreconditionTests {
    @Test func `A target without a depth attachment stops before encoding`() async {
        let result = await #expect(processExitsWith: .failure, observing: [\.standardErrorContent]) {
            guard let device = MTLCreateSystemDefaultDevice() else { return }
            let renderer = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm)
            let mesh = try renderer.makeMesh(.cube())
            let target = try OffscreenTarget(device: device, width: 16, height: 16)
            let camera = Camera(position: [0, 0, 3], projection: try PerspectiveProjection(verticalFieldOfView: 1, near: 0.1, far: 10))
            _ = renderer.drawFrame(mesh: mesh, transform: .identity, camera: camera) { target.frameTarget }
        }
        let message = String(decoding: result?.standardErrorContent ?? [], as: UTF8.self)
        #expect(message.contains("needs a depth attachment"), "stderr: \(message)")
        #expect(message.contains("depthStencilPixelFormat"))
    }
}
