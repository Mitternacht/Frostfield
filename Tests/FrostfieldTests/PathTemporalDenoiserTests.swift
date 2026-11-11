import Frostfield
import Metal
import Testing
import simd

#if (os(macOS) || os(iOS)) && !targetEnvironment(simulator)
@Suite(.enabled(
    if: MTLCreateSystemDefaultDevice().map { PathTemporalDenoiser.isSupported(device: $0) } == true,
    "Requires a GPU that supports MetalFX temporal denoising"
))
struct PathTemporalDenoiserTests {
    private let fixture: PathTestFixture
    private let inputWidth = 48
    private let inputHeight = 36
    private let outputWidth = 96
    private let outputHeight = 72

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Native denoising uses camera history, upscales, and resets to the first frame", arguments: [false, true])
    func cameraMoveAndReset(orthographic: Bool) throws {
        let (_, _, tracer) = try tracerFixture()
        let denoiser = try makeDenoiser(tracer)
        #expect(denoiser.outputTexture.width == outputWidth && denoiser.outputTexture.height == outputHeight)
        #expect(denoiser.outputTexture.storageMode == .private)
        #expect(denoiser.outputTexture.usage.contains(.shaderRead))

        let cameraModel = orthographic
            ? PathTracer.CameraModel(projection: .orthographic(height: 2.5), near: 0.1, far: 30)
            : model
        let first = try render(denoiser, model: cameraModel)
        #expect(!first.frame.reusedHistory)
        #expect(first.frame.jitter == TemporalAntiAliasing.jitter(at: 0))
        let initialCenter = try centroid(first.pixels)
        let initialRed = first.pixels[(outputHeight / 2) * outputWidth + outputWidth / 2].x
        #expect(abs(initialRed - 1.5) < 0.12)
        #expect(first.pixels.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite })

        let movedCamera = SceneTransform(translation: SIMD3(0.35, 0, 0))
        let moved = try render(denoiser, model: cameraModel, camera: movedCamera)
        #expect(moved.frame.reusedHistory)
        #expect(moved.frame.jitter == TemporalAntiAliasing.jitter(at: 1))
        #expect(try centroid(moved.pixels) < initialCenter - 3)

        denoiser.reset()
        let reset = try render(denoiser, model: cameraModel)
        #expect(!reset.frame.reusedHistory)
        #expect(reset.frame.jitter == first.frame.jitter)
        #expect(PathTestFixture.maximumDifference(reset.pixels, first.pixels) < 0.003)

        let fresh = try render(makeDenoiser(tracer), model: cameraModel)
        #expect(PathTestFixture.maximumDifference(fresh.pixels, reset.pixels) < 0.003)
    }

    @Test("Native denoising owns its queue and only promotes a submitted successful frame")
    func submissionAndDiscard() throws {
        let (_, _, tracer) = try tracerFixture()
        let denoiser = try makeDenoiser(tracer)
        let foreignQueue = try #require(fixture.device.makeCommandQueue())
        let foreign = try #require(foreignQueue.makeCommandBuffer())
        #expect(throws: PathTemporalDenoiser.Failure.commandBuffer) {
            try denoiser.encode(cameraModel: model, cameraTransform: .init(), commandBuffer: foreign)
        }
        let pending = try #require(fixture.queue.makeCommandBuffer())
        let frame = try denoiser.encode(cameraModel: model, cameraTransform: .init(), commandBuffer: pending)
        #expect(!frame.reusedHistory)
        let next = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTemporalDenoiser.Failure.frameInFlight) {
            try denoiser.encode(cameraModel: model, cameraTransform: .init(), commandBuffer: next)
        }
        #expect(throws: PathTemporalDenoiser.Failure.discard) { try denoiser.discardUnsubmittedFrame(next) }
        try denoiser.discardUnsubmittedFrame(pending)
        let afterDiscard = try render(denoiser)
        #expect(afterDiscard.frame == frame)

        #expect(throws: PathTemporalDenoiser.Failure.dimensions) {
            try PathTemporalDenoiser(tracer: tracer, inputWidth: 48, inputHeight: 36, outputWidth: 96, outputHeight: 96)
        }
        let transparent = try PathTracer.Lighting(environment: .zero, backgroundVisible: false)
        #expect(throws: PathTemporalDenoiser.Failure.lighting) {
            try denoiser.encode(cameraModel: model, cameraTransform: .init(), lighting: transparent, commandBuffer: next)
        }
    }

    @Test("Moving objects keep native history and a geometry rebuild restarts it")
    func movingMeshAndGeometryReset() throws {
        let (scene, graph, tracer) = try tracerFixture()
        let denoiser = try makeDenoiser(tracer)
        var initial = try render(denoiser)
        for _ in 1..<5 { initial = try render(denoiser) }
        let initialCenter = try centroid(initial.pixels)
        let centerPixel = (outputHeight / 2) * outputWidth + outputWidth / 2
        #expect(initial.pixels[centerPixel].x > 1.25)

        var moved = graph
        moved.roots[0].transform.translation.x = 1.15
        #expect(try scene.updateTransforms(moved))
        let shifted = try render(denoiser)
        #expect(shifted.frame.reusedHistory)
        #expect(shifted.frame.jitter == TemporalAntiAliasing.jitter(at: 5))
        #expect(try centroid(shifted.pixels) > initialCenter + 12)
        #expect(shifted.pixels[centerPixel].x < 0.2)

        try scene.updateGeometry(for: [moved.roots[0].id], in: moved)
        let reset = try render(denoiser)
        let fresh = try render(makeDenoiser(tracer))
        #expect(!reset.frame.reusedHistory)
        #expect(reset.frame.jitter == TemporalAntiAliasing.jitter(at: 0))
        #expect(PathTestFixture.maximumDifference(reset.pixels, fresh.pixels) < 0.003)
        #expect(try render(denoiser).frame.reusedHistory)
    }

    @Test("A morph keeps native history and exposes its previous silhouette")
    func morphAndDisocclusion() throws {
        let (scene, graph, tracer) = try tracerFixture(morphs: true)
        let denoiser = try makeDenoiser(tracer)
        var before = try render(denoiser)
        for _ in 1..<5 { before = try render(denoiser) }
        let center = (outputHeight / 2) * outputWidth + outputWidth / 2
        #expect(before.pixels[center].x > 1.25)

        var posed = graph
        let id = posed.roots[0].id
        posed.roots[0].morphWeights = [1]
        try scene.updateGeometry(for: [id], in: posed)
        let after = try render(denoiser)
        #expect(after.frame.reusedHistory)
        #expect(after.frame.jitter == TemporalAntiAliasing.jitter(at: 5))
        #expect(try centroid(after.pixels) > centroid(before.pixels) + 12)
        #expect(after.pixels[center].x < 0.2)

        let source = try #require(posed.roots[0].mesh).geometry
        try scene.updateGeometry(for: [], in: posed, replacements: [id: source])
        let reset = try render(denoiser)
        let fresh = try render(makeDenoiser(tracer))
        #expect(!reset.frame.reusedHistory)
        #expect(reset.frame.jitter == TemporalAntiAliasing.jitter(at: 0))
        #expect(PathTestFixture.maximumDifference(reset.pixels, fresh.pixels) < 0.003)
    }

    @Test("Temporal path denoising lowers error on a static area-lit surface")
    func denoisingQuality() throws {
        let mesh = try fixture.pbr(.plane(width: 8, height: 6), color: SIMD3(0.75, 0.7, 0.65))
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: fixture.camera
        )
        let light = try PathTracer.AreaLight(
            center: SIMD3(0.7, 0.55, -1.7), aim: SIMD3(0, 0, -3), size: SIMD2(1.8, 1.4),
            radiance: SIMD3(repeating: 6), cameraVisible: false
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lights: [light])
        let settings = PathTracer.Settings(maxBounces: 2, seed: 19)
        let reference = try fixture.trace(
            tracer, width: outputWidth, height: outputHeight, samples: 128, settings: settings
        )
        let noisy = try fixture.trace(
            tracer, width: outputWidth, height: outputHeight, samples: 1, settings: settings
        )
        let denoiser = try makeDenoiser(tracer)
        var denoised = try render(denoiser, settings: settings)
        for _ in 1..<12 { denoised = try render(denoiser, settings: settings) }
        #expect(denoised.frame.reusedHistory)

        func center(_ pixels: [SIMD4<Float>]) -> [SIMD4<Float>] {
            (outputHeight / 4..<(outputHeight * 3 / 4)).flatMap { y in
                Array(pixels[(y * outputWidth + outputWidth / 4)..<(y * outputWidth + outputWidth * 3 / 4)])
            }
        }
        let target = center(reference)
        let noisyError = PathTestFixture.rmse(center(noisy), target)
        let denoisedError = PathTestFixture.rmse(center(denoised.pixels), target)
        #expect(noisyError > 0.05)
        #expect(denoisedError < noisyError * 0.9, "Native RMSE \(denoisedError), one sample RMSE \(noisyError)")
    }

    private var model: PathTracer.CameraModel {
        .init(
            projection: .perspective(verticalFieldOfView: fixture.camera.verticalFieldOfView),
            near: fixture.camera.near, far: fixture.camera.far
        )
    }

    private func tracerFixture(morphs: Bool = false) throws -> (PathTracingScene, SceneGraph, PathTracer) {
        let material = try PhysicallyBasedMaterial(
            roughness: 1, baseColorFactor: SIMD4(0, 0, 0, 1),
            emissiveFactor: SIMD3(1, 0.4 / 1.5, 0.18 / 1.5), alphaMode: .opaque, emissiveStrength: 1.5
        )
        let plane = try MeshGeometry.plane(width: 1.6, height: 1.4)
        let targets = morphs ? try MeshMorphTargets(
            vertexCount: plane.vertices.count,
            targets: [.init(positionOffsets: plane.vertices.map { SIMD3(1.15 + 0.22 * $0.position.y, 0, 0) })]
        ) : nil
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: plane,
            material: .physicallyBased(material), morphTargets: targets
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, environment: SIMD3(0.08, 0.08, 0.08))
        return (scene, graph, tracer)
    }

    private func makeDenoiser(_ tracer: PathTracer) throws -> PathTemporalDenoiser {
        try PathTemporalDenoiser(
            tracer: tracer, inputWidth: inputWidth, inputHeight: inputHeight, outputWidth: outputWidth, outputHeight: outputHeight
        )
    }

    private func render(
        _ denoiser: PathTemporalDenoiser, model: PathTracer.CameraModel? = nil, camera: SceneTransform = .init(),
        settings: PathTracer.Settings = .init()
    ) throws -> (frame: PathTemporalDenoiser.Frame, pixels: [SIMD4<Float>]) {
        let command = try #require(fixture.queue.makeCommandBuffer())
        let frame = try denoiser.encode(
            cameraModel: model ?? self.model, cameraTransform: camera, settings: settings, commandBuffer: command
        )
        let pixelBytes = MemoryLayout<SIMD4<Float16>>.stride
        let rowBytes = outputWidth * pixelBytes
        let storage = try #require(fixture.device.makeBuffer(length: rowBytes * outputHeight, options: .storageModeShared))
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(
            from: denoiser.outputTexture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(),
            sourceSize: MTLSize(width: outputWidth, height: outputHeight, depth: 1),
            to: storage, destinationOffset: 0, destinationBytesPerRow: rowBytes, destinationBytesPerImage: 0
        )
        blit.endEncoding()
        try fixture.finish(command)
        let raw = UnsafeBufferPointer(
            start: storage.contents().assumingMemoryBound(to: SIMD4<Float16>.self), count: outputWidth * outputHeight
        )
        return (frame, raw.map(SIMD4<Float>.init))
    }

    private func centroid(_ pixels: [SIMD4<Float>]) throws -> Double {
        var sum = 0.0
        var mass = 0.0
        for (index, pixel) in pixels.enumerated() {
            let red = Double(max(pixel.x - pixel.y, 0))
            sum += Double(index % outputWidth) * red
            mass += red
        }
        try #require(mass > 100)
        return sum / mass
    }
}
#else
struct PathTemporalDenoiserTests {
    @Test("MetalFX temporal path denoising reports unavailable on this platform")
    func unavailable() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(!PathTemporalDenoiser.isSupported(device: device))
    }
}
#endif
