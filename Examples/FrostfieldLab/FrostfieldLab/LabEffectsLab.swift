import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// A plasma core shaded by an animated surface graph, a 65,536-particle vortex
/// simulated by a compute graph, and bloom plus graph post-processing.
@MainActor
final class LabEffectsLab: LabSceneFixture {
    private static let particleCount = 65_536

    private let queue: any MTLCommandQueue
    private let controls: LabSceneControls
    private let camera: PerspectiveCamera
    private let core: MeshRenderer
    private let coreUniforms: ShaderUniforms
    private let floor: MeshRenderer
    private let particles: ShaderParticleSystem
    private let particleUniforms: ShaderUniforms
    private let post: PostProcessChain
    private let display: HDRDisplay
    private let coreID = UUID()
    private let floorID = UUID()
    private var targets: (scene: any MTLTexture, depth: any MTLTexture, post: any MTLTexture)?
    private var orbit: OrbitCameraController
    private var lastTime: CFTimeInterval?
    private var time: Float = 0

    init(device: any MTLDevice, destinationFormat: MTLPixelFormat, controls: LabSceneControls) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("effects lab queue") }
        self.queue = queue
        self.controls = controls
        camera = try PerspectiveCamera(verticalFieldOfView: 0.7, near: 0.05, far: 60)

        let plasma = try Self.plasma()
        coreUniforms = try ShaderUniforms(graph: plasma)
        core = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .uvSphere(radius: 0.45, segments: 96, rings: 48),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 0.3, lightIntensity: 0, ambientIntensity: 0.02, alphaMode: .opaque,
                surfaceGraph: plasma, surfaceUniforms: coreUniforms
            ))
        )
        floor = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 30, height: 30),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                metallic: 0.2, roughness: 0.25, lightIntensity: 0, ambientIntensity: 0.01,
                baseColorFactor: SIMD4(0.05, 0.055, 0.07, 1), alphaMode: .opaque
            ))
        )

        let simulation = try Self.vortex()
        particleUniforms = try ShaderUniforms(simulation.uniforms)
        var initial: [SIMD4<Float>] = []
        initial.reserveCapacity(Self.particleCount * 3)
        var random = SystemRandomNumberGenerator()
        for _ in 0..<Self.particleCount {
            // Every particle starts hidden far below the floor, beyond the camera's far
            // plane, at age zero with a random remaining lifetime, so the first respawns
            // spread over four seconds and emission is continuous from the first frame.
            // The point is off the vortex axis so the swirl direction stays defined.
            let remaining = Float.random(in: 0...4, using: &random)
            initial += [SIMD4(0.9, -100, 0, 0), SIMD4(0, 0, 0, remaining), SIMD4(1, 1, 1, 0.03)]
        }
        particles = try ShaderParticleSystem(
            device: device, capacity: Self.particleCount, simulation: simulation, initial: initial, uniforms: particleUniforms
        )
        post = try PostProcessChain(queue: queue, effects: [
            .screenEffects(ScreenEffects.Settings(bloomThreshold: 1, bloomStrength: 0.9, bloomRadius: 10)),
            .graph(try LabShaderLab.splitTone()),
            .graph(try .vignette(strength: 0.45)),
        ])
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)
        orbit = try OrbitCameraController(
            target: SIMD3(0, 0.9, 0), distance: 5.6, yaw: 0.3, pitch: 0.32, projection: .perspective(verticalFieldOfView: 0.7),
            limits: .init(distance: 1.5...14, pitch: -0.05...1.4), dampingRate: 8
        )
        controls.register([
            LabControlGroup(id: "vortex", title: "Vortex", controls: [
                LabControl(id: "vortex.swirl", title: "Swirl", kind: .slider(0...6, step: 0.05), initial: 2.4),
                LabControl(id: "vortex.lift", title: "Lift", kind: .slider(-1...2, step: 0.02), initial: 0.3),
                LabControl(id: "vortex.turbulence", title: "Turbulence", kind: .slider(0...4, step: 0.05), initial: 0.8),
                LabControl(id: "vortex.hue", title: "Hue", kind: .slider(0...1, step: 0.01), initial: 0.55),
                LabControl(id: "vortex.glow", title: "Glow", kind: .slider(0...20, step: 0.1), initial: 1.6),
            ]),
            LabControlGroup(id: "core", title: "Core", controls: [
                LabControl(id: "core.energy", title: "Energy", kind: .slider(0...40, step: 0.5), initial: 14),
                LabControl(id: "core.scale", title: "Scale", kind: .slider(0.5...8, step: 0.1), initial: 3),
            ]),
            LabControlGroup(id: "post", title: "Post", controls: [
                LabControl(id: "post.bloom", title: "Bloom", kind: .slider(0...4, step: 0.05), initial: 0.9),
                LabControl(id: "post.tone", title: "Split tone", kind: .slider(0...1, step: 0.01), initial: 0.4),
                LabControl(id: "scene.exposure", title: "Exposure", kind: .slider(-3...3, step: 0.05), initial: -0.4),
                LabControl(id: "scene.paused", title: "Paused", kind: .toggle, initial: 0),
            ]),
        ])
        controls.sources["vortex"] = try simulation.metalFunction(named: "lab_vortex_step")
        controls.sources["core"] = try plasma.metalFunction(named: "lab_plasma_surface")
        try applyControls()
    }

    var animates: Bool { true }
    var needsMoreSamples: Bool { false }
    var previewLimit: CGFloat? { nil }

    func applyControls() throws {
        try particleUniforms.set("Swirl", to: Float(controls.value("vortex.swirl")))
        try particleUniforms.set("Lift", to: Float(controls.value("vortex.lift")))
        try particleUniforms.set("Turbulence", to: Float(controls.value("vortex.turbulence")))
        try particleUniforms.set("Hue", to: Float(controls.value("vortex.hue")))
        try particleUniforms.set("Glow", to: Float(controls.value("vortex.glow")))
        try coreUniforms.set("Energy", to: Float(controls.value("core.energy")))
        try coreUniforms.set("Scale", to: Float(controls.value("core.scale")))
        try post.replaceEffect(at: 0, with: .screenEffects(ScreenEffects.Settings(
            bloomThreshold: 1, bloomStrength: Float(controls.value("post.bloom")), bloomRadius: 10
        )))
        try post.uniforms(at: 1)?.set("Amount", to: Float(controls.value("post.tone")))
        controls.status = "\(Self.particleCount) particles step in a compute graph every frame; the core is a surface graph."
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
        let delta = Float(min(max(now - (lastTime ?? now), 0), 1.0 / 20))
        lastTime = now
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("effects lab frame") }
        try encode(destination: drawable.texture, buffer: buffer, delta: delta)
        buffer.addCompletedHandler { command in
            completed(command.status == .completed ? nil : command.error?.localizedDescription ?? "The effects lab frame did not finish.")
        }
        buffer.present(drawable)
        buffer.commit()
    }

    #if os(macOS)
    var commandQueue: any MTLCommandQueue { queue }

    func encodeOffscreen(into destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
        try encode(destination: destination, buffer: commandBuffer, delta: 1.0 / 60)
    }

    /// Simulates four seconds at 60 Hz so the vortex is fully formed.
    func capture(to url: URL, width: Int, height: Int) throws {
        let target = try LabImageCapture.target(device: queue.device, width: width, height: height)
        for frame in 0..<240 {
            guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("effects capture") }
            try encode(destination: target, buffer: buffer, delta: 1.0 / 60)
            buffer.commit()
            if frame == 239 { buffer.waitUntilCompleted() }
        }
        try LabImageCapture.write(target, to: url)
    }
    #endif

    private func encode(destination: any MTLTexture, buffer: any MTLCommandBuffer, delta: Float) throws {
        try orbit.update(deltaTime: delta)
        let step = controls.isOn("scene.paused") ? 0 : delta
        time += step
        try particleUniforms.set("DeltaTime", to: step)
        try particleUniforms.set("Time", to: time)
        try coreUniforms.set("Time", to: time)
        if step > 0 { try particles.encodeUpdate(commandBuffer: buffer) }

        let targets = try renderTargets(width: destination.width, height: destination.height)
        let pose = orbit.cameraTransform
        let graph = SceneGraph(roots: [
            SceneNode(id: coreID, transform: .init(translation: SIMD3(0, 0.6, 0)), mesh: core),
            SceneNode(id: floorID, transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: floor),
            SceneNode(transform: .init(translation: SIMD3(0, 0.6, 0)), light: try PunctualLight(
                kind: .point(range: 12), color: SIMD3(1, 0.62, 0.35), intensity: Float(controls.value("core.energy")) * 0.25
            )),
        ], camera: camera, cameraTransform: pose)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.scene
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.004, green: 0.005, blue: 0.009, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1
        try graph.encode(pass: pass, commandBuffer: buffer)

        let overlay = MTLRenderPassDescriptor()
        overlay.colorAttachments[0].texture = targets.scene
        overlay.colorAttachments[0].loadAction = .load
        overlay.colorAttachments[0].storeAction = .store
        overlay.depthAttachment.texture = targets.depth
        overlay.depthAttachment.loadAction = .load
        overlay.depthAttachment.storeAction = .dontCare
        try particles.encodeDraw(pass: overlay, camera: camera, cameraTransform: pose, commandBuffer: buffer)
        try post.encode(source: targets.scene, destination: targets.post, commandBuffer: buffer)
        try display.encode(
            source: targets.post, destination: destination,
            settings: .init(exposureStops: Float(controls.value("scene.exposure")), antiAliasing: true), commandBuffer: buffer
        )
    }

    private func renderTargets(width: Int, height: Int) throws -> (scene: any MTLTexture, depth: any MTLTexture, post: any MTLTexture) {
        if let targets, targets.scene.width == width, targets.scene.height == height { return targets }
        let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        color.storageMode = .private
        color.usage = [.renderTarget, .shaderRead]
        let depth = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: MeshRenderer.depthPixelFormat, width: width, height: height, mipmapped: false)
        depth.storageMode = .private
        depth.usage = .renderTarget
        guard let scene = queue.device.makeTexture(descriptor: color), let post = queue.device.makeTexture(descriptor: color),
              let depthTexture = queue.device.makeTexture(descriptor: depth) else {
            throw MeshRenderer.Failure.resource("effects lab targets")
        }
        targets = (scene, depthTexture, post)
        return (scene, depthTexture, post)
    }

    // MARK: - Graphs

    /// Glowing plasma: fractal noise drifting through time, raised to bright filaments.
    static func plasma() throws -> SurfaceShaderGraph {
        try SurfaceShaderGraph { s in
            let time = s.uniform("Time", Float(0))
            let energy = s.uniform("Energy", Float(14))
            let scale = s.uniform("Scale", Float(3))
            let p = s.worldPosition * scale + s.vector(time * 0.3, time * 0.5, time * -0.2)
            let field = s.fractalNoise(p, kind: .gradient, octaves: 5).named("plasma")
            let filaments = (field.remap(0.35, 0.75, 0, 1).saturated()).pow(3).named("filaments")
            let hot = s.constant(SIMD3<Float>(1, 0.55, 0.2)).mix(s.constant(SIMD3<Float>(1, 0.95, 0.85)), by: filaments)
            return .init(baseColor: s.constant(SIMD3<Float>(0.05, 0.03, 0.02)), emission: hot * filaments * energy)
        }
    }

    /// One step of a swirling vortex. Expired particles respawn on a ring around
    /// the core with hashed directions; living ones swirl, lift, and drift through noise.
    static func vortex() throws -> BufferShaderGraph {
        try BufferShaderGraph(fieldCount: 3) { s in
            let dt = s.uniform("DeltaTime", Float(0))
            let time = s.uniform("Time", Float(0))
            let swirl = s.uniform("Swirl", Float(2.4))
            let lift = s.uniform("Lift", Float(0.55))
            let turbulence = s.uniform("Turbulence", Float(1.2))
            let hue = s.uniform("Hue", Float(0.55))
            let glow = s.uniform("Glow", Float(9))
            let state = s.elementField(0)
            let motion = s.elementField(1)
            func hash(_ salt: Float) -> ShaderValue<Float> {
                ((s.elementIndex * (12.9898 + salt) + time * (78.233 + salt)).sin() * 43_758.547).fract()
            }
            // Respawn.
            let angle = hash(1) * (2 * .pi)
            let radius = hash(2) * 0.3 + 0.55
            let spawnPosition = s.vector(angle.cos() * radius, hash(3) * 0.1 + 0.55, angle.sin() * radius)
            let tangent = s.vector(-angle.sin(), s.constant(Float(0)), angle.cos())
            let spawnVelocity = tangent * swirl + s.vector(s.constant(Float(0)), lift, s.constant(Float(0)))
            let spawnLifetime = hash(4) * 2.5 + 1.5
            // Advance.
            let position = state.xyz
            // Constant tangential speed around the axis, pulled toward a ring of radius 0.9.
            let flat = s.vector(position.x, s.constant(Float(0)), position.z)
            let radial = flat.normalized()
            let around = s.vector(-radial.z, s.constant(Float(0)), radial.x)
            let inward = radial * ((flat.length() - 0.9) * -2.5)
            let noisePoint = position * 1.7 + s.vector(time * 0.4, time * 0.3, time * 0.5)
            let drift = s.vector(
                s.noise(noisePoint, kind: .gradient), s.noise(noisePoint + 17.3, kind: .gradient),
                s.noise(noisePoint + 41.7, kind: .gradient)
            ) - 0.5
            let target = around * swirl + inward + s.vector(s.constant(Float(0)), lift, s.constant(Float(0))) + drift * turbulence * 2
            let velocity = motion.xyz.mix(target, by: dt * 3)
            let advanced = position + velocity * dt
            let age = state.w + dt
            let expired = age.greaterThan(motion.w)
            let newState = s.select(expired, s.vector(spawnPosition, s.constant(Float(0))), s.vector(advanced, age))
            let newMotion = s.select(expired, s.vector(spawnVelocity, spawnLifetime), s.vector(velocity, motion.w))
            let progress = (age / motion.w).saturated()
            let color = s.vector(hue + progress * 0.25, s.constant(Float(0.75)), s.constant(Float(1))).hsvToRGB() * glow
            let size = progress.oneMinus() * 0.03 + 0.012
            return [0: newState, 1: newMotion, 2: s.vector(color, size)]
        }
    }
}
