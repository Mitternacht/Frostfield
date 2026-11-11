import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// Progressive path tracing for the Lab's code-built scenes.
@MainActor
final class LabPathTracingFixture {
    static let argument = "--path-traced-camera"
    static let captureArgument = "--capture-path-traced-camera"
    static let icebergArgument = "--iceberg"
    static let icebergCaptureArgument = "--capture-iceberg"
    static let icebergYaw: Float = -0.36
    static let porscheArgument = "--porsche"
    static let porscheCaptureArgument = "--capture-porsche"
    static let porscheYaw: Float = 0.68

    enum Subject {
        case rangefinder, lightRoom, iceberg, porsche
    }

    private struct Targets {
        let radiance: any MTLTexture
        let filtered: any MTLTexture
        let normalDepth: any MTLTexture
        let albedoRoughness: any MTLTexture
        let scratch: any MTLTexture
        let accumulation: PathTracer.Accumulation
    }

    private let queue: any MTLCommandQueue
    private let tracer: PathTracer
    private let camera: PerspectiveCamera
    private let denoiser: PathTraceDenoiser
    private let display: HDRDisplay
    private let detail: Bool
    private let subject: Subject
    private var targets: Targets?
    private var history: PathTracer.History?
    private var lightingOverride: PathTracer.Lighting?
    private var exposureStops: Float = -0.22
    private var revision = 0
    private var yaw: Float = 0
    private var elevation: Float
    private var distance: Float

    private static let previewSampleLimit = 32
    private static let defaultModelYaw: Float = -0.36

    var needsMoreSamples: Bool { (history?.sampleCount ?? 0) < Self.previewSampleLimit }

    init(
        device: any MTLDevice, destinationFormat: MTLPixelFormat, detail: Bool = false,
        subject: Subject = .rangefinder, generatedFinish: Bool = false
    ) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("path tracing queue") }
        self.queue = queue
        self.detail = detail && subject == .rangefinder
        self.subject = subject
        let fieldOfView: Float
        switch subject {
        case .iceberg:
            elevation = 0.78
            distance = 22.2
            yaw = Self.icebergYaw
            exposureStops = 0.08
            fieldOfView = 0.445
        case .porsche:
            elevation = 0.99
            distance = 8.72
            yaw = Self.porscheYaw
            exposureStops = 0.05
            fieldOfView = 0.348
        case .lightRoom:
            elevation = 0.28
            distance = 8.9
            fieldOfView = 0.61
        case .rangefinder:
            elevation = detail ? 1.0 : 1.85
            distance = detail ? 4.8 : 7.25
            fieldOfView = 0.61
        }
        camera = try PerspectiveCamera(verticalFieldOfView: fieldOfView, near: 0.1, far: subject == .iceberg ? 11_000 : 65)
        switch subject {
        case .iceberg:
            tracer = try LabIcebergScene.makeTracer(queue: queue, camera: camera)
        case .porsche:
            tracer = try LabPorscheScene.makeTracer(queue: queue, camera: camera)
        case .lightRoom:
            let room = try Self.makeLightRoom(device: device, camera: camera)
            let scene = try PathTracingScene(queue: queue, graph: room)
            let emitter = try PathTracer.AreaLight(
                center: SIMD3(0, 4.47, -1.57), halfU: SIMD3(0.92, 0, 0), halfV: SIMD3(0, 0, 0.65),
                radiance: SIMD3(12.5, 11.8, 10.9)
            )
            tracer = try PathTracer(queue: queue, scene: scene, lights: [emitter])
        case .rangefinder:
            tracer = try Self.makeStudio(queue: queue, camera: camera, generatedFinish: generatedFinish)
        }
        try tracer.prepareGuides()
        denoiser = try PathTraceDenoiser(device: device)
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)
    }

    private static func makeStudio(
        queue: any MTLCommandQueue, camera: PerspectiveCamera, generatedFinish: Bool
    ) throws -> PathTracer {
        let device = queue.device
        let model = LabProceduralCamera(
            device: device, light: SIMD3(0, 1, 1), intensity: 0, environment: nil, generatedFinish: generatedFinish
        )
        let asset = try LabStudioAsset(reference: model.graph(camera: camera))
        var graph = try asset.graph(camera: camera, cameraTransform: .init(), angle: Self.defaultModelYaw)
        let stageMaterial = try PhysicallyBasedMaterial(
            metallic: 0.07, roughness: 0.43, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0.11, 0.145, 0.177, 1), alphaMode: .opaque
        )
        let sweep = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: LabStudioFixture.sweepGeometry(includesBackdrop: true, frontExtent: 36),
            material: .physicallyBased(stageMaterial)
        )
        graph.roots.append(SceneNode(mesh: sweep))
        let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 512)
        let lighting = try LabPathLightSettings().makeLighting()
        return try PathTracer(queue: queue, scene: scene, lights: lighting.lights, environment: lighting.environment)
    }

    private static func makeLightRoom(device: any MTLDevice, camera: PerspectiveCamera) throws -> SceneGraph {
        func object(
            _ geometry: MeshGeometry, color: SIMD3<Float>, at position: SIMD3<Float>,
            rotation: simd_quatf = simd_quatf(real: 1, imag: .zero), metal: Float = 0, roughness: Float = 0.75
        ) throws -> SceneNode {
            let material = try PhysicallyBasedMaterial(
                metallic: metal, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(color, 1), alphaMode: .opaque
            )
            let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                                        material: .physicallyBased(material))
            return SceneNode(transform: .init(translation: position, rotation: rotation), mesh: mesh)
        }
        let white = SIMD3<Float>(0.73, 0.72, 0.69)
        let horizontal = try MeshGeometry.plane(width: 4.6, height: 5.6)
        let sides = try MeshGeometry.plane(width: 5.6, height: 4.5)
        let floorFacing = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let ceilingFacing = simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))
        let leftFacing = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
        let rightFacing = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))
        let crate = try MeshGeometry.bevelBox(size: SIMD3(1.35, 1.49, 1.35), bevel: 0.065)
        let sphere = try MeshGeometry.uvSphere(radius: 0.84, segments: 80, rings: 48)
        let nodes = try [
            object(horizontal, color: white, at: SIMD3(0, 0, -1.7), rotation: floorFacing),
            object(horizontal, color: white, at: SIMD3(0, 4.5, -1.7), rotation: ceilingFacing),
            object(.plane(width: 4.6, height: 4.5), color: white, at: SIMD3(0, 2.25, -4.5)),
            object(sides, color: SIMD3(0.71, 0.054, 0.035), at: SIMD3(-2.3, 2.25, -1.7), rotation: leftFacing),
            object(sides, color: SIMD3(0.035, 0.49, 0.27), at: SIMD3(2.3, 2.25, -1.7), rotation: rightFacing),
            object(crate, color: white, at: SIMD3(-1.02, 0.745, -0.91),
                   rotation: simd_quatf(angle: 0.39, axis: SIMD3(0, 1, 0))),
            object(sphere, color: SIMD3(0.89, 0.85, 0.78), at: SIMD3(0.94, 0.84, -1.57), metal: 1, roughness: 0.065),
        ]
        return SceneGraph(roots: nodes, camera: camera)
    }

    func setLighting(_ settings: LabPathLightSettings) throws {
        guard subject == .rangefinder else { return }
        let lighting = try settings.makeLighting()
        if lighting != lightingOverride {
            lightingOverride = lighting
            reset()
        }
        exposureStops = Float(settings.exposure)
    }

    func orbit(horizontal: Float, vertical: Float) {
        guard horizontal.isFinite, vertical.isFinite, horizontal != 0 || vertical != 0 else { return }
        yaw = (yaw - horizontal * 0.006).truncatingRemainder(dividingBy: 2 * .pi)
        elevation = min(max(elevation + vertical * 0.010, 0.22), subject == .iceberg ? 14 : 5.0)
        reset()
    }

    func zoom(by scale: Float) {
        guard scale.isFinite, scale > 0, scale != 1 else { return }
        let limits: ClosedRange<Float>
        switch subject {
        case .iceberg: limits = 11...43
        case .porsche: limits = 4.7...17
        default: limits = (detail ? 3.0 : 4.6)...13
        }
        distance = min(max(distance / scale, limits.lowerBound), limits.upperBound)
        reset()
    }

    func draw(drawable: any CAMetalDrawable, completed: @escaping @Sendable (String?) -> Void) throws {
        let output = drawable.texture
        let targets = try renderTargets(width: output.width, height: output.height)
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("path tracing preview") }
        let version = revision
        let next: PathTracer.History?
        if needsMoreSamples {
            next = try trace(
                targets: targets, previous: history, samples: 2, bounces: 5, commandBuffer: buffer
            )
            try filter(targets, passes: 2, commandBuffer: buffer)
        } else {
            next = nil
        }
        try present(targets.filtered, into: output, commandBuffer: buffer)
        buffer.addCompletedHandler { [weak self] command in
            let message = command.status == .completed ? nil : command.error?.localizedDescription ?? "Path tracing failed."
            Task { @MainActor in
                if let self, self.revision == version, message == nil, let next {
                    self.history = next
                }
                completed(message)
            }
        }
        buffer.present(drawable)
        buffer.commit()
    }

    #if os(macOS)
    func capture(to url: URL, samples: Int, yaw: Float, bounces: Int, denoise: Bool, preview: Bool = false) throws {
        self.yaw = yaw
        let size: (width: Int, height: Int)
        switch subject {
        case .porsche: size = preview ? (1280, 768) : (2560, 1536)
        case .iceberg: size = (1920, 1152)
        default: size = (1440, 960)
        }
        let output = try LabImageCapture.target(device: queue.device, width: size.width, height: size.height)
        let targets = try renderTargets(width: output.width, height: output.height)
        var previous: PathTracer.History?
        var remaining = samples
        while remaining > 0 {
            let count = min(8, remaining)
            guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("path tracing capture") }
            let next = try trace(
                targets: targets, previous: previous, samples: count, bounces: bounces, commandBuffer: buffer
            )
            remaining -= count
            if remaining == 0 {
                if denoise { try filter(targets, passes: subject == .iceberg || subject == .porsche ? 2 : 3, commandBuffer: buffer) }
                try present(denoise ? targets.filtered : targets.radiance, into: output, commandBuffer: buffer)
            }
            buffer.commit()
            buffer.waitUntilCompleted()
            guard buffer.status == .completed else {
                throw buffer.error ?? MeshRenderer.Failure.resource("path tracing capture completion")
            }
            previous = next
            if next.sampleCount.isMultiple(of: 32) || remaining == 0 {
                FileHandle.standardError.write(Data("Frostfield path trace: \(next.sampleCount)/\(samples) samples per pixel\n".utf8))
            }
        }
        try LabImageCapture.write(output, to: url)
    }
    #endif

    private func trace(
        targets: Targets, previous: PathTracer.History?, samples: Int, bounces: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PathTracer.History {
        let aspect = Float(targets.radiance.width) / Float(targets.radiance.height)
        let center: SIMD3<Float>
        switch subject {
        case .iceberg: center = SIMD3(0, 1.39, 0)
        case .porsche: center = SIMD3(0, 0.62, 0.28)
        case .lightRoom: center = SIMD3(0, 2.13, -0.65)
        case .rangefinder: center = detail ? SIMD3(-0.16, 1.37, 0) : SIMD3(0.27, 1.42, 0)
        }
        let orbitDistance = distance / min(1, max(aspect, 0.28) / 1.35)
        let levelOrbit = subject == .iceberg || subject == .porsche
        let offset: Float = levelOrbit ? 0 : 0.20
        let eye = center + SIMD3(sin(yaw) * orbitDistance + offset, elevation, cos(yaw) * orbitDistance)
        let forward = simd_normalize(center - eye)
        let rotation: simd_quatf
        if levelOrbit {
            let azimuth = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
            let pitch = simd_quatf(angle: -atan2(elevation, orbitDistance), axis: SIMD3(1, 0, 0))
            rotation = azimuth * pitch
        } else {
            rotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: forward)
        }
        let pose = SceneTransform(translation: eye, rotation: rotation)
        let focusPoint = center + (subject == .porsche ? SIMD3<Float>(0.15, -0.14, 0.92)
                                                     : SIMD3<Float>(-0.24, -0.04, detail ? 0.77 : 0.35))
        let focusDistance = max(0.5, simd_dot(focusPoint - eye, forward))
        let aperture: Float
        switch subject {
        case .porsche: aperture = 0.006
        case .rangefinder: aperture = detail ? 0.036 : 0.019
        default: aperture = 0
        }
        return try tracer.encode(
            camera: camera, cameraTransform: pose, destination: targets.radiance, accumulation: targets.accumulation,
            previous: previous, guides: .init(normalDepth: targets.normalDepth, albedoRoughness: targets.albedoRoughness),
            commandBuffer: commandBuffer, samplesPerPixel: samples,
            settings: .init(maxBounces: bounces, seed: 35, apertureRadius: aperture,
                            focusDistance: focusDistance), lighting: lightingOverride
        )
    }

    private func filter(_ targets: Targets, passes: Int, commandBuffer: any MTLCommandBuffer) throws {
        try denoiser.encode(
            source: targets.radiance, normalDepth: targets.normalDepth, albedoRoughness: targets.albedoRoughness,
            destination: targets.filtered, scratch: targets.scratch, passes: passes, commandBuffer: commandBuffer
        )
    }

    private func present(
        _ source: any MTLTexture, into destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try display.encode(
            source: source, destination: destination, settings: .init(exposureStops: exposureStops), commandBuffer: commandBuffer
        )
    }

    private func reset() {
        revision &+= 1
        history = nil
    }

    private func renderTargets(width: Int, height: Int) throws -> Targets {
        if let targets, targets.radiance.width == width, targets.radiance.height == height { return targets }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.shaderRead, .shaderWrite]
        func texture(_ label: String) throws -> any MTLTexture {
            guard let result = queue.device.makeTexture(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource(label)
            }
            result.label = label
            return result
        }
        let fresh = try Targets(
            radiance: texture("Path-traced HDR radiance"), filtered: texture("Denoised HDR radiance"),
            normalDepth: texture("Path trace first-hit normal and distance"),
            albedoRoughness: texture("Path trace first-hit albedo and roughness"),
            scratch: texture("Path trace denoise scratch"), accumulation: tracer.makeAccumulation(width: width, height: height)
        )
        targets = fresh
        reset()
        return fresh
    }
}
