import Foundation
import FrostfieldShaderTypes
import Metal
import Testing
import simd
@testable import Frostfield

private let systemDevice = MTLCreateSystemDefaultDevice()

/// `FFTransformUniforms` is the first transform record uploaded to the GPU.
/// The host layout is checked field by field here; the round-trip test below
/// checks what the shader compiler reads from the same bytes.
struct TransformUniformsLayoutTests {
    @Test func `Host layout matches the shader contract`() {
        typealias Uniforms = FFTransformUniforms
        #expect(MemoryLayout<Uniforms>.size == 64)
        #expect(MemoryLayout<Uniforms>.stride == 64)
        #expect(MemoryLayout<Uniforms>.alignment == 16)
        #expect(MemoryLayout<Uniforms>.offset(of: \.modelViewProjection) == 0)
        #expect(FF_TRANSFORM_UNIFORMS_BUFFER_INDEX == 0)
    }

    /// Column N of a `simd_float4x4` starts at byte 16 * N, and row R of that
    /// column at 4 * R more: the column-major order the shader indexes.
    @Test func `Matrix columns are contiguous in column-major order`() {
        var uniforms = FFTransformUniforms(modelViewProjection: TransformRoundTrip.distinctiveMatrix)
        let floats = withUnsafeBytes(of: &uniforms) { Array($0.bindMemory(to: Float.self)) }
        #expect(floats == TransformRoundTrip.expectedElements)
    }

    @Test func `Per-slot uniform buffers fit either record`() {
        #expect(TriangleRenderer.uniformBufferLength >= MemoryLayout<FFTransformUniforms>.stride)
        #expect(TriangleRenderer.uniformBufferLength >= MemoryLayout<FFTriangleUniforms>.stride)
    }
}

/// Distinctive values for the GPU round trip: element (column c, row r) holds
/// 10 * c + r + 0.5, so a transposed, shifted, or truncated read is visible.
enum TransformRoundTrip {
    static let distinctiveMatrix = simd_float4x4(columns: (
        SIMD4(0.5, 1.5, 2.5, 3.5),
        SIMD4(10.5, 11.5, 12.5, 13.5),
        SIMD4(20.5, 21.5, 22.5, 23.5),
        SIMD4(30.5, 31.5, 32.5, 33.5)
    ))

    static let expectedElements: [Float] = (0..<16).map { Float(10 * ($0 / 4) + $0 % 4) + 0.5 }

    /// The shared header's source, read from the package checkout so the
    /// kernel below compiles the exact declarations the renderer uses.
    static func headerSource() throws -> String {
        let header = URL(filePath: #filePath)
            .deletingLastPathComponent()  // FrostfieldTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
            .appending(path: "Sources/FrostfieldShaderTypes/include/FrostfieldShaderTypes.h")
        return try String(contentsOf: header, encoding: .utf8)
    }

    static let kernelSource = """
        kernel void ff_test_transform_round_trip(
            constant FFTransformUniforms &record [[buffer(0)]],
            device float *elements [[buffer(1)]],
            device uint *layout [[buffer(2)]],
            uint index [[thread_position_in_grid]])
        {
            elements[index] = record.modelViewProjection[index / 4][index % 4];
            if (index == 0) {
                layout[0] = sizeof(FFTransformUniforms);
                layout[1] = alignof(FFTransformUniforms);
                layout[2] = FF_TRANSFORM_UNIFORMS_BUFFER_INDEX;
            }
        }
        """
}

@Suite(.enabled(if: systemDevice != nil))
struct TransformUniformsRoundTripTests {
    /// Writes the record exactly as the renderer does (`storeBytes` into a
    /// shared buffer), has a kernel compiled from the shared header copy each
    /// matrix element out by MSL's `[column][row]` indexing, and compares.
    @Test func `The GPU reads every matrix element where the host wrote it`() async throws {
        let device = try #require(systemDevice)
        let source = try TransformRoundTrip.headerSource() + "\n" + TransformRoundTrip.kernelSource
        let library = try await device.makeLibrary(source: source, options: nil)
        let function = try #require(library.makeFunction(name: "ff_test_transform_round_trip"))
        let pipeline = try await device.makeComputePipelineState(function: function)

        // The record is followed by a sentinel so an over-long read would
        // show up as a wrong element rather than zeros.
        let recordBuffer = try #require(device.makeBuffer(length: 128, options: .storageModeShared))
        recordBuffer.contents().initializeMemory(as: Float.self, repeating: -999, count: 32)
        recordBuffer.contents().storeBytes(
            of: FFTransformUniforms(modelViewProjection: TransformRoundTrip.distinctiveMatrix),
            as: FFTransformUniforms.self
        )
        let elementBuffer = try #require(device.makeBuffer(length: 16 * MemoryLayout<Float>.stride, options: .storageModeShared))
        let layoutBuffer = try #require(device.makeBuffer(length: 3 * MemoryLayout<UInt32>.stride, options: .storageModeShared))

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(recordBuffer, offset: 0, index: 0)
        encoder.setBuffer(elementBuffer, offset: 0, index: 1)
        encoder.setBuffer(layoutBuffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        encoder.endEncoding()
        let status = await commitAndWait(commandBuffer)
        #expect(status == .completed)

        let elements = Array(UnsafeBufferPointer(start: elementBuffer.contents().assumingMemoryBound(to: Float.self), count: 16))
        #expect(elements == TransformRoundTrip.expectedElements)
        let layout = Array(UnsafeBufferPointer(start: layoutBuffer.contents().assumingMemoryBound(to: UInt32.self), count: 3))
        #expect(layout == [64, 16, 0], "MSL sizeof, alignof, and buffer index")
    }
}

/// Waits for GPU completion without blocking a thread.
func commitAndWait(_ commandBuffer: any MTLCommandBuffer) async -> MTLCommandBufferStatus {
    await withCheckedContinuation { continuation in
        commandBuffer.addCompletedHandler { buffer in continuation.resume(returning: buffer.status) }
        commandBuffer.commit()
    }
}

/// One deterministic pose and the normalized device coordinates where its
/// corners must land, computed by hand from the documented convention (camera
/// at (0, 0, 2) looking along -z, 90° vertical field of view unless stated).
struct ProjectedPose: Sendable, CustomTestStringConvertible {
    var name: String
    var transform = Transform.identity
    var camera = ProjectedPose.standardCamera
    var width = 256
    var height = 256
    /// Expected NDC of the red apex, the green corner, and the blue corner.
    var red: SIMD2<Float>
    var green: SIMD2<Float>
    var blue: SIMD2<Float>

    var testDescription: String { name }

    static let standardCamera = Camera(
        position: [0, 0, 2],
        projection: try! PerspectiveProjection(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    )

    static let all: [ProjectedPose] = [
        // Model corners (0, 0.75), (-0.6495, -0.375), (0.6495, -0.375) at
        // distance 2 with a focal scale of 1: NDC = model / 2.
        ProjectedPose(name: "identity", red: [0, 0.375], green: [-0.32476, -0.1875], blue: [0.32476, -0.1875]),
        ProjectedPose(
            name: "translated right and up",
            transform: Transform(translation: [0.5, 0.25, 0]),
            red: [0.25, 0.5], green: [-0.07476, -0.0625], blue: [0.57476, -0.0625]
        ),
        // Twice as far away: half the size.
        ProjectedPose(
            name: "translated away from the camera",
            transform: Transform(translation: [0, 0, -2]),
            red: [0, 0.1875], green: [-0.16238, -0.09375], blue: [0.16238, -0.09375]
        ),
        // A positive quarter turn about +z: (x, y) becomes (-y, x).
        ProjectedPose(
            name: "rotated a quarter turn about z",
            transform: Transform(rotation: simd_quatf(angle: .pi / 2, axis: [0, 0, 1])),
            red: [-0.375, 0], green: [0.1875, -0.32476], blue: [0.1875, 0.32476]
        ),
        ProjectedPose(
            name: "scaled non-uniformly",
            transform: Transform(scale: [2, 0.5, 1]),
            red: [0, 0.1875], green: [-0.64952, -0.09375], blue: [0.64952, -0.09375]
        ),
        // Scale x by 2, turn a quarter about z, then move right by 0.5.
        ProjectedPose(
            name: "scaled, rotated, then translated",
            transform: Transform(
                translation: [0.5, 0, 0],
                rotation: simd_quatf(angle: .pi / 2, axis: [0, 0, 1]),
                scale: [2, 1, 1]
            ),
            red: [-0.125, 0], green: [0.4375, -0.64952], blue: [0.4375, 0.64952]
        ),
        // A positive turn about +y moves +x away from the camera (right-handed):
        // the blue corner recedes and shrinks toward the centre, the green
        // corner comes closer and spreads out.
        ProjectedPose(
            name: "rotated 60 degrees about y",
            transform: Transform(rotation: simd_quatf(angle: .pi / 3, axis: [0, 1, 0])),
            red: [0, 0.375], green: [-0.22592, -0.26087], blue: [0.12674, -0.14634]
        ),
        // Seen from behind, the corners swap sides; nothing is culled.
        ProjectedPose(
            name: "turned around",
            transform: Transform(rotation: simd_quatf(angle: .pi, axis: [0, 1, 0])),
            red: [0, 0.375], green: [0.32476, -0.1875], blue: [-0.32476, -0.1875]
        ),
        // Moving the camera right moves the image left.
        ProjectedPose(
            name: "camera moved right",
            camera: Camera(position: [0.5, 0, 2], projection: standardCamera.projection),
            red: [-0.25, 0.375], green: [-0.57476, -0.1875], blue: [0.07476, -0.1875]
        ),
        // Rolling the camera a quarter turn counter-clockwise turns the image
        // a quarter turn clockwise.
        ProjectedPose(
            name: "camera rolled a quarter turn",
            camera: Camera(
                position: [0, 0, 2],
                orientation: simd_quatf(angle: .pi / 2, axis: [0, 0, 1]),
                projection: standardCamera.projection
            ),
            red: [0.375, 0], green: [-0.1875, 0.32476], blue: [-0.1875, -0.32476]
        ),
        // Twice as wide as tall: x is compressed by the aspect ratio of 2.
        ProjectedPose(
            name: "wide target",
            width: 512, height: 256,
            red: [0, 0.375], green: [-0.16238, -0.1875], blue: [0.16238, -0.1875]
        ),
        // A 60° field of view magnifies by 1 / tan(30°).
        ProjectedPose(
            name: "narrower field of view",
            camera: Camera(
                position: [0, 0, 2],
                projection: try! PerspectiveProjection(verticalFieldOfView: .pi / 3, near: 0.1, far: 10)
            ),
            red: [0, 0.64952], green: [-0.5625, -0.32476], blue: [0.5625, -0.32476]
        ),
    ]
}

/// Renders the triangle through ``TriangleRenderer/drawFrame(transform:camera:target:)``
/// on the system GPU and reads pixels back at positions derived by hand.
@Suite(.enabled(if: systemDevice != nil))
struct TransformRenderingTests {
    let device: any MTLDevice
    let completions: AsyncStream<FrameCompletion>
    let reportCompletion: AsyncStream<FrameCompletion>.Continuation

    init() throws {
        device = try #require(systemDevice)
        (completions, reportCompletion) = AsyncStream.makeStream(of: FrameCompletion.self)
    }

    func makeRenderer(device: (any MTLDevice)? = nil) throws -> TriangleRenderer {
        let report = reportCompletion
        return try TriangleRenderer(
            device: device ?? self.device,
            colorPixelFormat: .bgra8Unorm,
            onFrameCompleted: { report.yield($0) }
        )
    }

    /// Near each corner (80 % corner, 10 % each other corner) the corner's
    /// color dominates; the centroid is covered; reflecting a corner through
    /// the centroid lands beyond the opposite edge, which must be clear.
    @Test(arguments: ProjectedPose.all)
    func `Corners land at hand-computed screen positions`(pose: ProjectedPose) async throws {
        let renderer = try makeRenderer()
        let target = try OffscreenTarget(device: device, width: pose.width, height: pose.height)
        try await render(renderer, pose.transform, pose.camera, into: target)

        let corners: [(SIMD2<Float>, OffscreenTarget.Pixel.Channel)] = [(pose.red, .red), (pose.green, .green), (pose.blue, .blue)]
        let centroid = (pose.red + pose.green + pose.blue) / 3
        #expect(!target.pixel(at: centroid).isClear, "centroid \(centroid)")
        for (index, (corner, channel)) in corners.enumerated() {
            let others = corners.indices.filter { $0 != index }.map { corners[$0].0 }
            let nearCorner = 0.8 * corner + 0.1 * others[0] + 0.1 * others[1]
            let pixel = target.pixel(at: nearCorner)
            #expect(pixel.dominantChannel == channel, "near the \(channel) corner at \(nearCorner): \(pixel)")

            let beyondOppositeEdge = 2 * centroid - corner
            if simd_reduce_max(simd_abs(beyondOppositeEdge)) < 0.98 {
                let outside = target.pixel(at: beyondOppositeEdge)
                #expect(outside.isClear, "beyond the edge opposite \(channel) at \(beyondOppositeEdge): \(outside)")
            }
        }
    }

    /// With near 1 and far 10, Metal keeps depth in [0, 1]. A projection
    /// built for OpenGL's [-1, 1] range would put the "just beyond near" case
    /// at depth -0.96 and clip it.
    @Test(arguments: [
        (Float(-1.02), Float(1), true),   // just beyond the near plane
        (Float(-0.98), Float(1), false),  // just in front of the near plane
        (Float(-9.9), Float(10), true),   // just inside the far plane
        (Float(-10.2), Float(10), false), // beyond the far plane
    ])
    func `Only geometry between the near and far planes is drawn`(z: Float, scale: Float, visible: Bool) async throws {
        let renderer = try makeRenderer()
        let target = try OffscreenTarget(device: device, width: 64, height: 64)
        let camera = Camera(position: .zero, projection: try PerspectiveProjection(verticalFieldOfView: .pi / 2, near: 1, far: 10))
        let transform = Transform(translation: [0, 0, z], scale: SIMD3(repeating: scale))
        try await render(renderer, transform, camera, into: target)
        #expect(target.pixel(clipX: 0, clipY: 0).isClear == !visible, "z = \(z): \(target.pixel(clipX: 0, clipY: 0))")
    }

    /// Camera and transform changes, including a new aspect ratio and a new
    /// projection, only rewrite the slot's existing uniform buffer.
    ///
    /// The renderer is created on a ``CountingDevice``, which logs every
    /// object-creating message the renderer sends to its device. That log,
    /// not the renderer's own `resourceStatistics`, shows that drawing
    /// creates nothing.
    @Test func `Camera and transform updates create no GPU resources`() async throws {
        let counting = CountingDevice(wrapping: device)
        let renderer = try makeRenderer(device: counting.device)
        let created = counting.creations
        let pipelines = created.filter { $0.hasPrefix("newRenderPipelineState") }
        let buffers = created.filter { $0.hasPrefix("newBuffer") }
        #expect(pipelines.count == 2, "\(created)")
        #expect(buffers.count == renderer.maxFramesInFlight, "\(created)")
        // The renderer's own report agrees with what its device saw.
        #expect(renderer.resourceStatistics.renderPipelinesCreated == pipelines.count)
        #expect(renderer.resourceStatistics.buffersAllocated == buffers.count)

        let targets = [
            try OffscreenTarget(device: device, width: 64, height: 64),
            try OffscreenTarget(device: device, width: 96, height: 48),
        ]
        var camera = ProjectedPose.standardCamera
        var lastExpected = matrix_identity_float4x4
        for step in 0..<60 {
            let angle = Float(step) * 0.1
            let transform = Transform(
                translation: [sin(angle) * 0.5, 0, -Float(step % 5)],
                rotation: simd_quatf(angle: angle, axis: simd_normalize([0.3, 1, 0.2])),
                scale: SIMD3(repeating: 1 + Float(step % 3) * 0.25)
            )
            camera.position.x = cos(angle)
            camera.projection = try PerspectiveProjection(verticalFieldOfView: 0.6 + Float(step % 4) * 0.2, near: 0.1, far: 50)
            let target = targets[step % 2]
            try await render(renderer, transform, camera, into: target)

            let aspectRatio = Float(target.texture.width) / Float(target.texture.height)
            lastExpected = try camera.viewProjectionMatrix(aspectRatio: aspectRatio) * transform.matrix
        }

        let createdWhileDrawing = Array(counting.creations.dropFirst(created.count))
        #expect(createdWhileDrawing.isEmpty, "drawing created \(createdWhileDrawing.count) objects: \(Set(createdWhileDrawing).sorted())")
        #expect(renderer.statistics.submitted == 60)
        // Frame 60 used slot 1 (frames alternate between the two slots while
        // each waits for the previous one), and that buffer holds its matrix.
        let written = renderer.uniformBuffers[1].contents().load(as: FFTransformUniforms.self).modelViewProjection
        #expect(written == lastExpected)
    }

    // MARK: Helpers

    private func render(_ renderer: TriangleRenderer, _ transform: Transform, _ camera: Camera, into target: OffscreenTarget) async throws {
        let outcome = renderer.drawFrame(transform: transform, camera: camera) { target.frameTarget }
        guard case .submitted(let frame) = outcome else {
            Issue.record("Frame was not submitted: \(outcome)")
            return
        }
        var iterator = completions.makeAsyncIterator()
        let completion = try #require(await iterator.next())
        #expect(completion == FrameCompletion(frame: frame, succeeded: true))
    }
}
