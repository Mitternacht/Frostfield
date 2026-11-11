import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// The imported dolphin's swim cycle mixed with clips built in Swift: a cruise
/// path, a leap that cross-fades in and out, and an additive roll layered on top.
/// A damped orbit controller frames the scene.
@MainActor
final class LabAnimationLab: LabSceneFixture {
    private let queue: any MTLCommandQueue
    private let controls: LabSceneControls
    private let camera: PerspectiveCamera
    private let prepared: PreparedGLTFScene
    private let floor: MeshRenderer
    private let shadow: DirectionalShadowMap
    private let background: EnvironmentBackground
    private let display: HDRDisplay
    private let framing: SceneTransform
    private let framingID = UUID()
    private let motionID: UUID
    private let floorID = UUID()
    private var mixer: AnimationMixer
    private let swim: AnimationAction.ID
    private let cruise: AnimationAction.ID
    private let leap: AnimationAction.ID
    private let roll: AnimationAction.ID
    private var leaping = false
    private var orbit: OrbitCameraController
    private var lastTime: CFTimeInterval?
    private var targets: (scene: any MTLTexture, depth: any MTLTexture)?
    private var lastEvent = "Ready"

    init(device: any MTLDevice, destinationFormat: MTLPixelFormat, controls: LabSceneControls) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("animation lab queue") }
        self.queue = queue
        self.controls = controls
        camera = try PerspectiveCamera(verticalFieldOfView: 0.6, near: 0.1, far: 80)
        guard let url = Bundle.main.url(forResource: "Dolphin", withExtension: "glb") else {
            throw MeshRenderer.Failure.resource("bundled Dolphin.glb")
        }
        let source = try GLTFMeshLoader.loadScene(
            contentsOf: url, includeMaterials: true, includeAnimations: true, includeSkins: true
        )
        guard let swimClip = source.animations.first else { throw MeshRenderer.Failure.resource("dolphin swim clip") }
        let lighting = try LabProceduralLighting.make(queue: queue)
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: 0.45, yaw: 0)
        let light = SIMD3<Float>(-0.4, 1.3, 0.7)
        let skin = try PhysicallyBasedMaterial(
            metallic: 0.3, roughness: 0.32, directionToLight: light, lightIntensity: 2.4, ambientIntensity: 0,
            receivesShadows: true, baseColorFactor: SIMD4(0.3, 0.55, 0.78, 1), alphaMode: .opaque, environment: environment
        )
        prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: .rgba16Float, scene: source, defaultMaterial: .physicallyBased(skin)
        )
        // Fit the bind mesh to about 3 meters, nose toward +X.
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = -lower
        for node in try prepared.graph(camera: camera).resolve() {
            guard let mesh = node.mesh else { continue }
            for vertex in mesh.geometry.vertices {
                let point = node.worldTransform * SIMD4(vertex.position, 1)
                lower = simd_min(lower, SIMD3(point.x, point.y, point.z))
                upper = simd_max(upper, SIMD3(point.x, point.y, point.z))
            }
        }
        let rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
        let scale = 3 / max(simd_reduce_max(upper - lower), 1e-3)
        let center = rotation.act((lower + upper) * 0.5) * scale
        framing = SceneTransform(translation: SIMD3(-center.x, 1.1 - center.y, -center.z), rotation: rotation, scale: SIMD3(repeating: scale))

        floor = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: LabStudioFixture.sweepGeometry(includesBackdrop: false, frontExtent: 30),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 0.85, directionToLight: light, lightIntensity: 2.4, ambientIntensity: 0, receivesShadows: true,
                baseColorFactor: SIMD4(0.2, 0.23, 0.27, 1), alphaMode: .opaque,
                environment: try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: 0.2, yaw: 0)
            ))
        )
        shadow = try DirectionalShadowMap(
            queue: queue, resolution: 2048, directionToLight: light, center: SIMD3(0, 1, 0), halfExtent: 4,
            intensity: 2.4, bias: 0.0008, lightAngularRadius: 0.05
        )
        background = try EnvironmentBackground(lighting: lighting, destinationPixelFormat: .rgba16Float)
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)

        // Clips authored in Swift target the motion node above the imported hierarchy.
        let motion = UUID()
        motionID = motion
        func clip(_ name: String, duration: Float, steps: Int, pose: (Float) -> SceneTransform) throws -> AnimationClip {
            let keys = (0...steps).map { step -> TransformKeyframe in
                let time = duration * Float(step) / Float(steps)
                return TransformKeyframe(time: time, transform: pose(time / duration))
            }
            return AnimationClip(try SceneAnimation(tracks: [SceneAnimationTrack(nodeID: motion, keyframes: keys)]), name: name)
        }
        let cruiseClip = try clip("Cruise", duration: 4, steps: 32) { phase in
            let angle = phase * 2 * .pi
            return SceneTransform(translation: SIMD3(0, 0.15 * sin(angle * 2), 0.5 * sin(angle)),
                                  rotation: simd_quatf(angle: -0.25 * cos(angle), axis: SIMD3(0, 1, 0)))
        }
        let leapClip = try clip("Leap", duration: 2.2, steps: 44) { phase in
            let height = 1.4 * sin(phase * .pi)
            return SceneTransform(translation: SIMD3(0, height, 0),
                                  rotation: simd_quatf(angle: 0.55 * cos(phase * .pi), axis: SIMD3(0, 0, 1)))
        }
        let rollClip = try clip("Roll", duration: 1.6, steps: 32) { phase in
            SceneTransform(rotation: simd_quatf(angle: 0.6 * sin(phase * 2 * .pi), axis: SIMD3(1, 0, 0)))
        }.makingAdditive(referenceTime: 0)
        var mixer = AnimationMixer()
        swim = try mixer.addAction(for: AnimationClip(swimClip, nodeIDs: prepared.nodeIDs))
        cruise = try mixer.addAction(for: cruiseClip)
        leap = try mixer.addAction(for: leapClip, weight: 1)
        roll = try mixer.addAction(for: rollClip, weight: 0)
        try mixer.play(swim)
        try mixer.play(cruise)
        try mixer.play(roll)
        self.mixer = mixer

        orbit = try OrbitCameraController(
            target: SIMD3(0, 1.1, 0), distance: 7.5, yaw: 0.35, pitch: 0.22, projection: .perspective(verticalFieldOfView: 0.6),
            limits: .init(distance: 2.5...16, pitch: -0.05...1.25), dampingRate: 9
        )
        controls.register([LabControlGroup(id: "motion", title: "Motion", controls: [
            LabControl(id: "motion.leap", title: "Leap", kind: .toggle, initial: 0),
            LabControl(id: "motion.blend", title: "Blend time", kind: .slider(0.05...2, step: 0.05), initial: 0.6),
            LabControl(id: "motion.swim", title: "Swim speed", kind: .slider(0...3, step: 0.05), initial: 1),
            LabControl(id: "motion.roll", title: "Roll layer", kind: .slider(0...1, step: 0.01), initial: 0.4),
            LabControl(id: "motion.paused", title: "Paused", kind: .toggle, initial: 0),
            LabControl(id: "scene.exposure", title: "Exposure", kind: .slider(-3...3, step: 0.05), initial: 0),
        ])])
        try applyControls()
    }

    var animates: Bool { true }
    var needsMoreSamples: Bool { false }
    var previewLimit: CGFloat? { nil }

    func applyControls() throws {
        let blend = controls.value("motion.blend")
        let wantsLeap = controls.isOn("motion.leap")
        if wantsLeap != leaping {
            let (outgoing, incoming) = wantsLeap ? (cruise, leap) : (leap, cruise)
            // A stopped incoming clip starts from its first frame; crossFade then plays
            // it from zero weight. One still fading out keeps its time and weight and
            // fades back in from there, so neither pose nor weight jumps.
            if try !mixer.action(incoming).isPlaying { try mixer.reset(incoming) }
            if try mixer.action(outgoing).isPlaying {
                try mixer.crossFade(from: outgoing, to: incoming, duration: blend)
            } else {
                try mixer.fadeIn(incoming, duration: blend)
            }
            leaping = wantsLeap
        }
        try mixer.setTimeScale(Float(controls.value("motion.swim")), for: swim)
        try mixer.setWeight(Float(controls.value("motion.roll")), for: roll)
        let weights = try [cruise, leap].map { try mixer.action($0).effectiveWeight }
        controls.status = String(format: "Cruise %.2f · Leap %.2f · %@", weights[0], weights[1], lastEvent)
    }

    func orbit(horizontal: Float, vertical: Float) {
        guard horizontal.isFinite, vertical.isFinite else { return }
        try? orbit.rotate(by: SIMD2(-horizontal * 0.006, vertical * 0.006))
    }

    func zoom(by scale: Float) {
        guard scale.isFinite, scale > 0 else { return }
        try? orbit.dolly(by: scale)
    }

    func draw(drawable: any CAMetalDrawable, completed: @escaping @Sendable (String?) -> Void) throws {
        let now = CACurrentMediaTime()
        let delta = min(max(now - (lastTime ?? now), 0), 0.1)
        lastTime = now
        try advance(by: delta)
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("animation lab frame") }
        try encode(destination: drawable.texture, buffer: buffer)
        buffer.addCompletedHandler { command in
            completed(command.status == .completed ? nil : command.error?.localizedDescription ?? "The animation lab frame did not finish.")
        }
        buffer.present(drawable)
        buffer.commit()
    }

    #if os(macOS)
    var commandQueue: any MTLCommandQueue { queue }

    func encodeOffscreen(into destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
        try advance(by: 1.0 / 60)
        try encode(destination: destination, buffer: commandBuffer)
    }

    /// Advances 1.5 seconds at 60 Hz so the capture shows a blended pose.
    func capture(to url: URL, width: Int, height: Int) throws {
        for _ in 0..<90 { try advance(by: 1.0 / 60) }
        let target = try LabImageCapture.target(device: queue.device, width: width, height: height)
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("animation capture") }
        try encode(destination: target, buffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw buffer.error ?? MeshRenderer.Failure.resource("animation capture") }
        try LabImageCapture.write(target, to: url)
    }
    #endif

    private func advance(by delta: Double) throws {
        try orbit.update(deltaTime: Float(delta))
        guard !controls.isOn("motion.paused") else { return }
        for event in try mixer.update(deltaTime: delta) {
            switch event {
            case .looped: lastEvent = "looped"
            case .finished: lastEvent = "finished"
            }
        }
        try applyControls()
    }

    private func encode(destination: any MTLTexture, buffer: any MTLCommandBuffer) throws {
        let targets = try renderTargets(width: destination.width, height: destination.height)
        let pose = orbit.cameraTransform
        var graph = try prepared.graph(camera: camera, cameraTransform: pose)
        graph.roots = [
            SceneNode(id: framingID, transform: framing, children: [SceneNode(id: motionID, children: graph.roots)]),
            SceneNode(id: floorID, mesh: floor, castsShadow: false),
        ]
        graph = try prepared.posing(mixer.apply(to: graph))
        try background.encode(destination: targets.scene, camera: camera, cameraTransform: pose, yaw: 0,
                              intensity: 0.3, commandBuffer: buffer)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.scene
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        try display.encode(
            source: targets.scene, destination: destination,
            settings: .init(exposureStops: Float(controls.value("scene.exposure")), antiAliasing: true), commandBuffer: buffer
        )
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
            throw MeshRenderer.Failure.resource("animation lab targets")
        }
        targets = (scene, depthTexture)
        return (scene, depthTexture)
    }
}
