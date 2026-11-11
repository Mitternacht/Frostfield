import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// A studio set lit by one `LightRig`: a softbox, a spotlight, a bulb and a sun,
/// plus an HDR sky. Every control edits the rig, which drives raster lights and
/// shadows and the path tracer's lighting. Light edits restart path accumulation;
/// exposure only changes the display.
@MainActor
final class LabLightLab: LabSceneFixture {
    private struct LightSpec {
        let id: String
        let title: String
        let power: Double
        let kelvin: Double
        let azimuth: Double
        let elevation: Double
        let distance: Float
        let size: Double
        let sizeRange: ClosedRange<Double>
        let sizeTitle: String
        let powerRange: ClosedRange<Double>
    }

    private static let target = SIMD3<Float>(0, 0.6, 0)
    private static let specs: [LightSpec] = [
        .init(id: "key", title: "Softbox", power: 120, kelvin: 5_200, azimuth: -42, elevation: 38, distance: 3.2,
              size: 1, sizeRange: 0.3...3, sizeTitle: "Size", powerRange: 0...600),
        .init(id: "spot", title: "Spot", power: 40, kelvin: 3_200, azimuth: 48, elevation: 52, distance: 3.8,
              size: 32, sizeRange: 8...70, sizeTitle: "Cone", powerRange: 0...200),
        .init(id: "bulb", title: "Bulb", power: 25, kelvin: 2_700, azimuth: 128, elevation: 14, distance: 1.7,
              size: 0, sizeRange: 0...0, sizeTitle: "", powerRange: 0...150),
        .init(id: "sun", title: "Sun", power: 1.4, kelvin: 5_800, azimuth: -125, elevation: 34, distance: 10,
              size: 1.5, sizeRange: 0...8, sizeTitle: "Diameter", powerRange: 0...6),
    ]

    private let queue: any MTLCommandQueue
    private let controls: LabSceneControls
    private let camera: PerspectiveCamera
    private let objects: [SceneNode]
    private let sunShadow: DirectionalShadowMap
    private let localShadows: PunctualShadowMaps
    private let background: EnvironmentBackground
    private let display: HDRDisplay
    private let pathEnvironment: PathTracer.EnvironmentMap
    private let pathScene: PathTracingScene
    private var tracer: PathTracer?
    private var accumulation: PathTracer.Accumulation?
    private var history: PathTracer.History?
    private var tracedPose: SceneTransform?
    private var tracedLighting: PathTracer.Lighting?
    private var traced: (any MTLTexture)?
    private var targets: (scene: any MTLTexture, depth: any MTLTexture)?
    private var rig: LightRig
    private var angle: Float = 0.18
    private var elevation: Float = 1.4
    private var distance: Float = 6.6

    init(device: any MTLDevice, destinationFormat: MTLPixelFormat, controls: LabSceneControls) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("light lab queue") }
        self.queue = queue
        self.controls = controls
        camera = try PerspectiveCamera(verticalFieldOfView: 0.6, near: 0.1, far: 60)
        let lighting = try LabProceduralLighting.make(queue: queue)
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: 1, yaw: 0)
        func material(_ color: SIMD3<Float>, metallic: Float = 0, roughness: Float) throws -> MeshMaterial {
            .physicallyBased(try PhysicallyBasedMaterial(
                metallic: metallic, roughness: roughness, ambientIntensity: 0, receivesShadows: true,
                baseColorFactor: SIMD4(color, 1), alphaMode: .opaque, environment: environment
            ))
        }
        func mesh(_ geometry: MeshGeometry, _ material: MeshMaterial) throws -> MeshRenderer {
            try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material)
        }
        let vase = try MeshGeometry.lathe(profile: [
            (0, 0), (0.3, 0), (0.36, 0.25), (0.22, 0.7), (0.3, 1.05), (0.18, 1.3), (0.2, 1.42), (0, 1.42),
        ].map { LatheProfilePoint(radius: $0.0, y: $0.1) }, segments: 96)
        objects = [
            SceneNode(mesh: try mesh(LabStudioFixture.sweepGeometry(includesBackdrop: true, frontExtent: 12),
                                     material(SIMD3(0.42, 0.43, 0.45), roughness: 0.82)), castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(-1.25, 0, 0.1)),
                      mesh: try mesh(vase, material(SIMD3(0.86, 0.84, 0.8), roughness: 0.9))),
            SceneNode(transform: .init(translation: SIMD3(0, 0.5, 0.45)),
                      mesh: try mesh(.uvSphere(radius: 0.5, segments: 96, rings: 48),
                                     material(SIMD3(0.75, 0.06, 0.05), roughness: 0.18))),
            SceneNode(transform: .init(translation: SIMD3(1.3, 0.4, -0.15), rotation: simd_quatf(angle: 0.6, axis: SIMD3(0, 1, 0))),
                      mesh: try mesh(.bevelBox(size: SIMD3(repeating: 0.8), bevel: 0.06, segments: 4),
                                     material(SIMD3(0.92, 0.9, 0.86), metallic: 1, roughness: 0.32))),
            SceneNode(transform: .init(translation: SIMD3(0.55, 0.9, -1.1)),
                      mesh: try mesh(.bevelBox(size: SIMD3(0.18, 1.8, 0.18), bevel: 0.02, segments: 2),
                                     material(SIMD3(0.2, 0.22, 0.25), roughness: 0.5))),
        ]
        sunShadow = try DirectionalShadowMap(
            queue: queue, resolution: 2048, directionToLight: SIMD3(0, 1, 0), center: Self.target, halfExtent: 4, intensity: 0
        )
        localShadows = try PunctualShadowMaps(queue: queue, resolution: 1024, spotCapacity: 2, pointCapacity: 2)
        background = try EnvironmentBackground(lighting: lighting, destinationPixelFormat: .rgba16Float)
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)
        pathEnvironment = try LabProceduralLighting.pathEnvironment(device: device)
        pathScene = try PathTracingScene(queue: queue, graph: SceneGraph(roots: objects, camera: camera), textureSize: 64)
        rig = try LightRig(lights: [])

        controls.register(Self.specs.map { spec in
            var items = [
                LabControl(id: "\(spec.id).on", title: "On", kind: .toggle, initial: 1),
                LabControl(id: "\(spec.id).power", title: spec.id == "sun" ? "Lux" : "Power",
                           kind: .slider(spec.powerRange, step: spec.powerRange.upperBound / 200), initial: spec.power),
                LabControl(id: "\(spec.id).kelvin", title: "Kelvin", kind: .slider(1_800...12_000, step: 50), initial: spec.kelvin),
                LabControl(id: "\(spec.id).azimuth", title: "Azimuth", kind: .slider(-180...180, step: 1), initial: spec.azimuth),
                LabControl(id: "\(spec.id).elevation", title: "Elevation", kind: .slider(3...88, step: 1), initial: spec.elevation),
                LabControl(id: "\(spec.id).shadows", title: "Shadows", kind: .toggle, initial: 1),
            ]
            if spec.sizeRange.upperBound > 0 {
                items.insert(LabControl(id: "\(spec.id).size", title: spec.sizeTitle,
                                        kind: .slider(spec.sizeRange, step: spec.sizeRange.upperBound / 100), initial: spec.size), at: 2)
            }
            return LabControlGroup(id: spec.id, title: spec.title, controls: items)
        } + [LabControlGroup(id: "scene", title: "Sky & View", controls: [
            LabControl(id: "scene.path", title: "Path traced", kind: .toggle, initial: 0),
            LabControl(id: "sky.intensity", title: "Sky", kind: .slider(0...2, step: 0.01), initial: 0.35),
            LabControl(id: "sky.rotation", title: "Sky turn", kind: .slider(-180...180, step: 1), initial: 0),
            LabControl(id: "scene.exposure", title: "Exposure", kind: .slider(-4...4, step: 0.05), initial: 0),
            LabControl(id: "scene.samples", title: "Samples", kind: .slider(16...2048, step: 16), initial: 256),
        ])])
        try applyControls()
    }

    var animates: Bool { false }
    var needsMoreSamples: Bool { controls.isOn("scene.path") && (history?.sampleCount ?? 0) < sampleTarget }
    var previewLimit: CGFloat? {
        guard controls.isOn("scene.path") else { return nil }
        #if os(macOS)
        return 960
        #else
        return 640
        #endif
    }

    private var sampleTarget: Int { max(1, Int(controls.value("scene.samples"))) }

    func applyControls() throws {
        let values = controls.values
        rig = try LightRig(lights: Self.specs.map { spec in
            func value(_ name: String) -> Double { values["\(spec.id).\(name)"] ?? 0 }
            let azimuth = Float(value("azimuth")) * .pi / 180
            let elevation = Float(value("elevation")) * .pi / 180
            let offset = SIMD3(cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth)) * spec.distance
            let kind: LightRig.Light.Kind
            let emission: LightRig.Light.Emission
            switch spec.id {
            case "key":
                let scale = Float(value("size"))
                kind = .rectangle(width: 1.6 * scale, height: 1.0 * scale)
                emission = .power(Float(value("power")))
            case "spot":
                let outer = Float(value("size")) * .pi / 180
                kind = .spot(innerAngle: outer * 0.45, outerAngle: outer)
                emission = .power(Float(value("power")))
            case "bulb":
                kind = .point()
                emission = .power(Float(value("power")))
            default:
                kind = .directional(angularDiameter: Float(value("size")) * .pi / 180)
                emission = .intensity(Float(value("power")))
            }
            return LightRig.Light(
                id: spec.id, kind: kind, position: Self.target + offset, target: Self.target,
                color: try LightRig.Light.color(temperature: Float(value("kelvin"))), emission: emission,
                castsShadow: value("shadows") > 0.5, isEnabled: value("on") > 0.5
            )
        })
        let sun = rig.directionalShadow
        try sunShadow.update(
            directionToLight: sun?.directionToLight ?? SIMD3(0, 1, 0), center: Self.target, halfExtent: 4,
            intensity: sun?.intensity ?? 0, bias: 0.0008, lightAngularRadius: min(sun?.angularRadius ?? 0, 0.25)
        )
        let mode = controls.isOn("scene.path") ? "Path traced" : "Raster"
        let samples = history.map { " · \($0.sampleCount) samples per pixel" } ?? ""
        let powers = try rig.lights.filter(\.isEnabled).compactMap { light -> String? in
            guard let power = try light.power() else { return nil }
            return "\(light.id) \(Int(power.rounded()))"
        }.joined(separator: ", ")
        controls.status = "\(mode)\(samples). Power: \(powers). Light edits restart sampling; exposure does not."
    }

    func orbit(horizontal: Float, vertical: Float) {
        guard horizontal.isFinite, vertical.isFinite else { return }
        angle = (angle + horizontal * 0.006).truncatingRemainder(dividingBy: 2 * .pi)
        elevation = min(max(elevation + vertical * 0.009, 0.15), 3.5)
    }

    func zoom(by scale: Float) {
        guard scale.isFinite, scale > 0 else { return }
        distance = min(max(distance / scale, 3), 12)
    }

    func draw(drawable: any CAMetalDrawable, completed: @escaping @Sendable (String?) -> Void) throws {
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("light lab frame") }
        try encode(destination: drawable.texture, buffer: buffer, pathSamples: 2)
        buffer.addCompletedHandler { command in
            completed(command.status == .completed ? nil : command.error?.localizedDescription ?? "The light lab frame did not finish.")
        }
        buffer.present(drawable)
        buffer.commit()
    }

    #if os(macOS)
    var commandQueue: any MTLCommandQueue { queue }

    func encodeOffscreen(into destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
        try encode(destination: destination, buffer: commandBuffer, pathSamples: 1)
    }

    func capture(to url: URL, width: Int, height: Int) throws {
        let target = try LabImageCapture.target(device: queue.device, width: width, height: height)
        let passes = controls.isOn("scene.path") ? max(1, sampleTarget / 8) : 1
        for _ in 0..<passes {
            guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("light lab capture") }
            try encode(destination: target, buffer: buffer, pathSamples: 8)
            buffer.commit()
            buffer.waitUntilCompleted()
            guard buffer.status == .completed else { throw buffer.error ?? MeshRenderer.Failure.resource("light lab capture") }
        }
        try LabImageCapture.write(target, to: url)
    }
    #endif

    private func cameraPose(aspect: Float) -> SceneTransform {
        let framing = min(1, aspect / 1.4)
        let eye = Self.target + SIMD3(sin(angle), 0, cos(angle)) * (distance / framing) + SIMD3(0, elevation, 0)
        return SceneTransform(translation: eye, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: simd_normalize(Self.target - eye)))
    }

    private func encode(destination: any MTLTexture, buffer: any MTLCommandBuffer, pathSamples: Int) throws {
        let targets = try renderTargets(width: destination.width, height: destination.height)
        let pose = cameraPose(aspect: Float(destination.width) / Float(destination.height))
        let skyIntensity = Float(controls.value("sky.intensity"))
        let skyYaw = Float(controls.value("sky.rotation")) * .pi / 180
        let settings = HDRDisplay.Settings(exposureStops: Float(controls.value("scene.exposure")), antiAliasing: !controls.isOn("scene.path"))
        if controls.isOn("scene.path") {
            let image = try trace(pose: pose, width: destination.width, height: destination.height, samples: pathSamples,
                                  sky: .map(pathEnvironment, intensity: skyIntensity, yaw: skyYaw), buffer: buffer)
            try display.encode(source: image, destination: destination, settings: settings, commandBuffer: buffer)
            try applyControls()
            return
        }
        releasePathTracing()
        var graph = SceneGraph(roots: objects + (try rig.sceneNodes(omittingDirectionalShadow: true)), camera: camera, cameraTransform: pose)
        graph.environmentIntensity = skyIntensity
        graph.environmentRotation = skyYaw
        try background.encode(destination: targets.scene, camera: camera, cameraTransform: pose, yaw: skyYaw,
                              intensity: skyIntensity, commandBuffer: buffer)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.scene
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: sunShadow, punctualShadowMaps: localShadows)
        try display.encode(source: targets.scene, destination: destination, settings: settings, commandBuffer: buffer)
    }

    private func trace(
        pose: SceneTransform, width: Int, height: Int, samples: Int, sky: PathTracer.Environment, buffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        let lighting = try rig.pathLighting(sky: sky)
        if tracer == nil {
            tracer = try PathTracer(queue: queue, scene: pathScene, lighting: lighting)
        }
        guard let tracer else { throw MeshRenderer.Failure.resource("light lab tracer") }
        if accumulation?.width != width || accumulation?.height != height {
            accumulation = try tracer.makeAccumulation(width: width, height: height)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite]
            traced = queue.device.makeTexture(descriptor: descriptor)
            history = nil
        }
        guard let accumulation, let traced else { throw MeshRenderer.Failure.resource("light lab path targets") }
        if tracedPose.map({ $0.translation != pose.translation || $0.rotation != pose.rotation }) ?? true {
            history = nil
            tracedPose = pose
        }
        // Rig and sky edits restart sampling here, before a converged history would
        // return early and skip the tracer's own lighting check.
        if lighting != tracedLighting {
            history = nil
            tracedLighting = lighting
        }
        if let previous = history, previous.sampleCount >= sampleTarget { return traced }
        let settings = PathTracer.Settings(maxBounces: 6, seed: 11)
        do {
            history = try tracer.encode(camera: camera, cameraTransform: pose, destination: traced, accumulation: accumulation,
                                        previous: history, commandBuffer: buffer, samplesPerPixel: samples,
                                        settings: settings, lighting: lighting)
        } catch PathTracer.Failure.lightingChanged {
            // A light edit: same prepared scene and tracer, new accumulation.
            history = try tracer.encode(camera: camera, cameraTransform: pose, destination: traced, accumulation: accumulation,
                                        previous: nil, commandBuffer: buffer, samplesPerPixel: samples,
                                        settings: settings, lighting: lighting)
        }
        return traced
    }

    /// Releases every piece of path state together. An accumulation belongs to the
    /// tracer that made it, so a later tracer must not inherit it.
    private func releasePathTracing() {
        tracer = nil
        accumulation = nil
        history = nil
        traced = nil
        tracedPose = nil
        tracedLighting = nil
    }

    private func renderTargets(width: Int, height: Int) throws -> (scene: any MTLTexture, depth: any MTLTexture) {
        if let targets, targets.scene.width == width, targets.scene.height == height { return targets }
        let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        color.storageMode = .private
        color.usage = [.renderTarget, .shaderRead]
        let depth = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: MeshRenderer.depthPixelFormat, width: width, height: height, mipmapped: false)
        depth.storageMode = .private
        depth.usage = .renderTarget
        guard let scene = queue.device.makeTexture(descriptor: color), let depthTexture = queue.device.makeTexture(descriptor: depth) else {
            throw MeshRenderer.Failure.resource("light lab targets")
        }
        targets = (scene, depthTexture)
        return (scene, depthTexture)
    }
}
