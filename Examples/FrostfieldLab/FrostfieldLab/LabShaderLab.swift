import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// Five generated objects whose finishes and motion come from Swift-built shader
/// graphs. Every slider edits a live uniform, so neither renderer recompiles.
/// The same graphs shade the raster view and the progressive path-traced view.
@MainActor
final class LabShaderLab: LabSceneFixture {
    private struct Exhibit {
        let id: String
        let title: String
        let graph: SurfaceShaderGraph
        let uniforms: ShaderUniforms?
        let renderer: MeshRenderer
        let transform: SceneTransform
        let controls: [LabControl]
        /// Maps control values to uniform writes.
        let apply: (LabSceneControls, ShaderUniforms) throws -> Void
    }

    private let queue: any MTLCommandQueue
    private let controls: LabSceneControls
    private let camera: PerspectiveCamera
    private let exhibits: [Exhibit]
    private let floor: MeshRenderer
    private let shadow: DirectionalShadowMap
    private let background: EnvironmentBackground
    private let display: HDRDisplay
    private let post: PostProcessChain
    private let nodeIDs: [String: UUID]
    private let floorID = UUID()
    private var targets: (scene: any MTLTexture, depth: any MTLTexture, post: any MTLTexture)?
    private var angle: Float = 0.32
    private var elevation: Float = 1.75
    private var distance: Float = 7.4
    /// Wave time in seconds of animation, advanced by each frame's elapsed time
    /// times the Speed control, so speed changes and pauses resume without a jump.
    private var frozenTime: Float = 1.2
    private var lastFrameTime: CFTimeInterval?

    // Progressive path tracing of the same scene.
    private var tracer: PathTracer?
    private var tracedWaveRevision: UInt64?
    private var accumulation: PathTracer.Accumulation?
    private var history: PathTracer.History?
    private var tracedCamera: SceneTransform?
    private var tracedUniformRevisions: [UInt64?]?
    private var traced: (any MTLTexture)?
    private var pathSampleTarget: Int { max(1, Int(controls.value("scene.samples"))) }

    static let environmentYaw: Float = 0

    init(device: any MTLDevice, destinationFormat: MTLPixelFormat, controls: LabSceneControls) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("shader lab queue") }
        self.queue = queue
        self.controls = controls
        camera = try PerspectiveCamera(verticalFieldOfView: 0.62, near: 0.1, far: 60)
        let lighting = try LabProceduralLighting.make(queue: queue)
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: 0.42, yaw: Self.environmentYaw)
        let light = SIMD3<Float>(-0.52, 1.45, 0.58)
        let intensity: Float = 2.4
        func material(_ graph: SurfaceShaderGraph, _ uniforms: ShaderUniforms?, metallic: Float = 0,
                      textures: [String: any MTLTexture] = [:]) throws -> MeshMaterial {
            .physicallyBased(try PhysicallyBasedMaterial(
                metallic: metallic, roughness: 0.5, directionToLight: light, lightIntensity: intensity,
                ambientIntensity: 0.2, receivesShadows: true, alphaMode: .opaque, environment: environment,
                surfaceGraph: graph, surfaceUniforms: uniforms, surfaceTextures: textures
            ))
        }

        var exhibits: [Exhibit] = []
        func add(
            _ id: String, _ title: String, _ graph: SurfaceShaderGraph, geometry: MeshGeometry, at transform: SceneTransform,
            metallic: Float = 0, textures: [String: any MTLTexture] = [:], controls: [LabControl],
            apply: @escaping (LabSceneControls, ShaderUniforms) throws -> Void
        ) throws {
            let uniforms = graph.uniforms.isEmpty ? nil : try ShaderUniforms(graph: graph)
            let renderer = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: material(graph, uniforms, metallic: metallic, textures: textures)
            )
            exhibits.append(Exhibit(
                id: id, title: title, graph: graph, uniforms: uniforms, renderer: renderer, transform: transform,
                controls: controls, apply: apply
            ))
        }

        try add("marble", "Marble", Self.marble(), geometry: .uvSphere(radius: 0.55, segments: 96, rings: 48),
                at: .init(translation: SIMD3(-2.3, 0.55, -0.4)), controls: [
                    .init(id: "marble.scale", title: "Scale", kind: .slider(0.5...6, step: 0.05), initial: 3.2),
                    .init(id: "marble.warp", title: "Warp", kind: .slider(0...3, step: 0.05), initial: 1.4),
                    .init(id: "marble.veins", title: "Veins", kind: .slider(1...40, step: 0.5), initial: 14),
                ]) { controls, uniforms in
                    try uniforms.set("Scale", to: Float(controls.value("marble.scale")))
                    try uniforms.set("Warp", to: Float(controls.value("marble.warp")))
                    try uniforms.set("Sharpness", to: Float(controls.value("marble.veins")))
                }
        try add("copper", "Hammered", Self.hammered(), geometry: Self.vase(), at: .init(translation: SIMD3(-0.8, 0, -0.6)),
                metallic: 1, controls: [
                    .init(id: "copper.dimples", title: "Dimples", kind: .slider(4...60, step: 1), initial: 22),
                    .init(id: "copper.depth", title: "Depth", kind: .slider(0...2, step: 0.02), initial: 0.9),
                    .init(id: "copper.roughness", title: "Roughness", kind: .slider(0.05...0.8, step: 0.01), initial: 0.28),
                ]) { controls, uniforms in
                    try uniforms.set("Dimples", to: Float(controls.value("copper.dimples")))
                    try uniforms.set("Depth", to: Float(controls.value("copper.depth")))
                    try uniforms.set("Roughness", to: Float(controls.value("copper.roughness")))
                }
        try add("film", "Thin Film", Self.thinFilm(), geometry: .uvSphere(radius: 0.5, segments: 96, rings: 48),
                at: .init(translation: SIMD3(0.75, 0.5, -0.5)), metallic: 1, controls: [
                    .init(id: "film.thickness", title: "Thickness", kind: .slider(0...3, step: 0.02), initial: 1.1),
                    .init(id: "film.hue", title: "Hue", kind: .slider(0...1, step: 0.01), initial: 0.55),
                ]) { controls, uniforms in
                    try uniforms.set("Thickness", to: Float(controls.value("film.thickness")))
                    try uniforms.set("Hue", to: Float(controls.value("film.hue")))
                }
        // A compute graph generates the wood texture on the GPU; the tile's surface graph samples it.
        let wood = try ShaderTextureGenerator(graph: Self.woodGrain(), device: device)
            .makeTexture(width: 1024, height: 1024, format: .rgba8Unorm_srgb, mipmapped: true, queue: queue)
        try add("tile", "Texture", Self.transformedTexture(),
                geometry: .bevelBox(size: SIMD3(1.2, 1.2, 0.12), bevel: 0.035, segments: 3),
                at: .init(translation: SIMD3(2.25, 0.62, -0.7), rotation: simd_quatf(angle: -0.35, axis: SIMD3(0, 1, 0))),
                textures: ["Grain": wood], controls: [
                    .init(id: "tile.tiling", title: "Tiling", kind: .slider(0.25...6, step: 0.05), initial: 1.5),
                    .init(id: "tile.rotation", title: "Rotation", kind: .slider(-180...180, step: 1), initial: 25),
                    .init(id: "tile.warmth", title: "Warmth", kind: .slider(-1...1, step: 0.01), initial: 0.2),
                ]) { controls, uniforms in
                    try uniforms.set("Tiling", to: Float(controls.value("tile.tiling")))
                    try uniforms.set("Rotation", to: Float(controls.value("tile.rotation")) * .pi / 180)
                    let warmth = Float(controls.value("tile.warmth"))
                    try uniforms.set("Tint", to: SIMD3(1 + 0.25 * warmth, 1, 1 - 0.3 * warmth))
                }
        try add("wave", "Waves", Self.waves(), geometry: .plane(width: 6.4, height: 2.2, columns: 160, rows: 56),
                at: .init(translation: SIMD3(0, 0.22, 1.5), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
                controls: [
                    .init(id: "wave.amplitude", title: "Amplitude", kind: .slider(0...0.2, step: 0.005), initial: 0.05),
                    .init(id: "wave.frequency", title: "Frequency", kind: .slider(1...14, step: 0.1), initial: 5.5),
                    .init(id: "wave.speed", title: "Speed", kind: .slider(0...4, step: 0.05), initial: 1.4),
                    .init(id: "wave.foam", title: "Foam", kind: .slider(0...1, step: 0.01), initial: 0.8),
                ]) { controls, uniforms in
                    try uniforms.set("Amplitude", to: Float(controls.value("wave.amplitude")))
                    try uniforms.set("Frequency", to: Float(controls.value("wave.frequency")))
                    try uniforms.set("Foam", to: Float(controls.value("wave.foam")))
                }
        self.exhibits = exhibits
        nodeIDs = Dictionary(uniqueKeysWithValues: exhibits.map { ($0.id, UUID()) })

        floor = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: LabStudioFixture.sweepGeometry(includesBackdrop: false, frontExtent: 30),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 0.78, directionToLight: light, lightIntensity: intensity, ambientIntensity: 0.2,
                receivesShadows: true, baseColorFactor: SIMD4(0.11, 0.12, 0.14, 1), alphaMode: .opaque,
                environment: try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: 0.14, yaw: Self.environmentYaw)
            ))
        )
        shadow = try DirectionalShadowMap(
            queue: queue, resolution: 2048, directionToLight: light, center: SIMD3(0, 0.6, 0),
            halfExtent: 4.2, intensity: intensity, bias: 0.0008, lightAngularRadius: 0.08
        )
        background = try EnvironmentBackground(lighting: lighting, destinationPixelFormat: .rgba16Float)
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)
        post = try PostProcessChain(queue: queue, effects: [
            .graph(try Self.splitTone()),
            .graph(try .chromaticAberration(amount: 1.2)),
            .graph(try .vignette(strength: 0.35)),
        ])

        let scene = [
            LabControl(id: "scene.path", title: "Path traced", kind: .toggle, initial: 0),
            LabControl(id: "scene.animate", title: "Animate", kind: .toggle, initial: 1),
            LabControl(id: "scene.exposure", title: "Exposure", kind: .slider(-3...3, step: 0.05), initial: 0),
            LabControl(id: "scene.samples", title: "Samples", kind: .slider(16...1024, step: 16), initial: 128),
        ]
        let postControls = [
            LabControl(id: "post.tone", title: "Split tone", kind: .slider(0...1, step: 0.01), initial: 0.35),
            LabControl(id: "post.fringe", title: "Fringing", kind: .slider(0...6, step: 0.1), initial: 1.2),
            LabControl(id: "post.vignette", title: "Vignette", kind: .slider(0...1, step: 0.01), initial: 0.35),
        ]
        controls.register(exhibits.map { LabControlGroup(id: $0.id, title: $0.title, controls: $0.controls) }
            + [LabControlGroup(id: "post", title: "Post", controls: postControls),
               LabControlGroup(id: "scene", title: "Scene", controls: scene)])
        controls.sources["post"] = try Self.splitTone().metalFunction(named: "lab_split_tone")
        for exhibit in exhibits {
            var source = try exhibit.graph.metalFunction(named: "lab_\(exhibit.id)_surface")
            if let vertex = try exhibit.graph.metalVertexFunction(named: "lab_\(exhibit.id)_vertex") {
                source += "\n\n" + vertex
            }
            controls.sources[exhibit.id] = source
        }
        try applyControls()
    }

    var animates: Bool { true }
    var needsMoreSamples: Bool { controls.isOn("scene.path") && (history?.sampleCount ?? 0) < pathSampleTarget }
    var previewLimit: CGFloat? {
        guard controls.isOn("scene.path") else { return nil }
        #if os(macOS)
        return 960
        #else
        return 640
        #endif
    }

    func applyControls() throws {
        for exhibit in exhibits {
            guard let uniforms = exhibit.uniforms else { continue }
            try exhibit.apply(controls, uniforms)
        }
        try post.uniforms(at: 0)?.set("Amount", to: Float(controls.value("post.tone")))
        try post.uniforms(at: 1)?.set("Amount", to: Float(controls.value("post.fringe")))
        try post.uniforms(at: 2)?.set("Strength", to: Float(controls.value("post.vignette")))
        let mode = controls.isOn("scene.path") ? "Path traced" : "Raster"
        let samples = history.map { " · \($0.sampleCount) samples per pixel" } ?? ""
        controls.status = "\(mode)\(samples). Sliders edit graph uniforms; pipelines stay compiled."
    }

    func orbit(horizontal: Float, vertical: Float) {
        guard horizontal.isFinite, vertical.isFinite else { return }
        angle = (angle + horizontal * 0.006).truncatingRemainder(dividingBy: 2 * .pi)
        elevation = min(max(elevation + vertical * 0.009, 0.2), 4.5)
    }

    func zoom(by scale: Float) {
        guard scale.isFinite, scale > 0 else { return }
        distance = min(max(distance / scale, 3), 14)
    }

    func draw(drawable: any CAMetalDrawable, completed: @escaping @Sendable (String?) -> Void) throws {
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("shader lab frame") }
        try encode(destination: drawable.texture, buffer: buffer, pathSamples: 2)
        buffer.addCompletedHandler { command in
            completed(command.status == .completed ? nil : command.error?.localizedDescription ?? "The shader lab frame did not finish.")
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
        let passes = controls.isOn("scene.path") ? pathSampleTarget / 8 : 1
        for pass in 0..<passes {
            guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("shader lab capture") }
            try encode(destination: target, buffer: buffer, pathSamples: 8, time: frozenTime)
            buffer.commit()
            buffer.waitUntilCompleted()
            guard buffer.status == .completed else { throw buffer.error ?? MeshRenderer.Failure.resource("capture \(pass)") }
        }
        try LabImageCapture.write(target, to: url)
    }
    #endif

    // MARK: - Frame

    private func cameraPose(aspect: Float) -> SceneTransform {
        let target = SIMD3<Float>(0, 0.55, 0)
        let framing = min(1, aspect / 1.4)
        let eye = target + SIMD3(sin(angle), 0, cos(angle)) * (distance / framing) + SIMD3(0, elevation, 0)
        return SceneTransform(translation: eye, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: simd_normalize(target - eye)))
    }

    private func sceneGraph(pose: SceneTransform) -> SceneGraph {
        var roots = exhibits.map { exhibit in
            SceneNode(id: nodeIDs[exhibit.id]!, transform: exhibit.transform, mesh: exhibit.renderer)
        }
        roots.append(SceneNode(id: floorID, mesh: floor, castsShadow: false))
        return SceneGraph(roots: roots, camera: camera, cameraTransform: pose)
    }

    private func encode(
        destination: any MTLTexture, buffer: any MTLCommandBuffer, pathSamples: Int, time: Float? = nil
    ) throws {
        let targets = try renderTargets(width: destination.width, height: destination.height)
        let pose = cameraPose(aspect: Float(destination.width) / Float(destination.height))
        let pathTraced = controls.isOn("scene.path")
        if controls.isOn("scene.animate"), !pathTraced, time == nil {
            let now = CACurrentMediaTime()
            // Cap a long gap, such as a hidden window, so the waves resume rather than leap.
            let elapsed = min(max(now - (lastFrameTime ?? now), 0), 0.25)
            frozenTime += Float(elapsed) * Float(controls.value("wave.speed"))
            lastFrameTime = now
        } else {
            lastFrameTime = nil
            if let time { frozenTime = time }
        }
        if let waves = exhibits.first(where: { $0.id == "wave" })?.uniforms {
            try waves.set("Time", to: frozenTime)
        }
        let graph = sceneGraph(pose: pose)
        let settings = HDRDisplay.Settings(exposureStops: Float(controls.value("scene.exposure")) - 0.2, antiAliasing: !pathTraced)

        if pathTraced {
            let image = try trace(graph, pose: pose, width: destination.width, height: destination.height,
                                  samples: pathSamples, buffer: buffer)
            // Post and exposure run on the accumulated image without restarting it.
            try post.encode(source: image, destination: targets.post, commandBuffer: buffer)
            try display.encode(source: targets.post, destination: destination, settings: settings, commandBuffer: buffer)
            try applyControls()
            return
        }
        releasePathTracing()
        try background.encode(
            destination: targets.scene, camera: camera, cameraTransform: pose, yaw: Self.environmentYaw,
            intensity: 0.32, commandBuffer: buffer
        )
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.scene
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        try post.encode(source: targets.scene, destination: targets.post, commandBuffer: buffer)
        try display.encode(source: targets.post, destination: destination, settings: settings, commandBuffer: buffer)
    }

    /// Edits to shader values restart accumulation. Wave edits change geometry and
    /// re-snapshot the scene; compiled libraries are reused. Exposure is display-only.
    private func trace(
        _ graph: SceneGraph, pose: SceneTransform, width: Int, height: Int, samples: Int, buffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        let waveRevision = exhibits.first(where: { $0.id == "wave" })?.uniforms?.revision
        if tracer == nil || tracedWaveRevision != waveRevision {
            let scene = try PathTracingScene(queue: queue, graph: graph, textureSize: 512)
            tracer = try PathTracer(queue: queue, scene: scene, lights: Self.pathLights(), environment: SIMD3(0.05, 0.06, 0.075))
            tracedWaveRevision = waveRevision
            history = nil
            accumulation = nil
        }
        guard let tracer else { throw MeshRenderer.Failure.resource("shader lab tracer") }
        if accumulation?.width != width || accumulation?.height != height {
            accumulation = try tracer.makeAccumulation(width: width, height: height)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite]
            traced = queue.device.makeTexture(descriptor: descriptor)
            history = nil
        }
        guard let accumulation, let traced else { throw MeshRenderer.Failure.resource("shader lab path targets") }
        if tracedCamera.map({ $0.translation != pose.translation || $0.rotation != pose.rotation }) ?? true {
            history = nil
            tracedCamera = pose
        }
        // Surface uniform edits restart sampling here, before a converged history
        // would return early and skip the tracer's own uniform check.
        let revisions = exhibits.map { $0.uniforms?.revision }
        if revisions != tracedUniformRevisions {
            history = nil
            tracedUniformRevisions = revisions
        }
        if let previous = history, previous.sampleCount >= pathSampleTarget { return traced }
        let settings = PathTracer.Settings(maxBounces: 6, seed: 7)
        do {
            history = try tracer.encode(
                camera: camera, cameraTransform: pose, destination: traced, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: samples, settings: settings
            )
        } catch PathTracer.Failure.surfaceUniformsChanged {
            history = try tracer.encode(
                camera: camera, cameraTransform: pose, destination: traced, accumulation: accumulation,
                previous: nil, commandBuffer: buffer, samplesPerPixel: samples, settings: settings
            )
        }
        return traced
    }

    private func releasePathTracing() {
        guard tracer != nil else { return }
        tracer = nil
        accumulation = nil
        history = nil
        traced = nil
        tracedWaveRevision = nil
        tracedCamera = nil
        tracedUniformRevisions = nil
    }

    private static func pathLights() throws -> [PathTracer.AreaLight] {
        try [
            .init(center: SIMD3(-3.6, 5.2, 4.1), aim: SIMD3(0, 0.6, 0), size: SIMD2(4.6, 2.2),
                  radiantPower: SIMD3(330, 290, 240), cameraVisible: false),
            .init(center: SIMD3(4.4, 2.8, 1.6), aim: SIMD3(0, 0.7, 0), size: SIMD2(1.4, 3.6),
                  radiantPower: SIMD3(80, 105, 150), cameraVisible: false),
            .init(center: SIMD3(1.0, 4.2, -4.4), aim: SIMD3(0, 0.8, 0), size: SIMD2(4.2, 1.5),
                  radiantPower: SIMD3(170, 190, 230), cameraVisible: false),
        ]
    }

    private func renderTargets(
        width: Int, height: Int
    ) throws -> (scene: any MTLTexture, depth: any MTLTexture, post: any MTLTexture) {
        if let targets, targets.scene.width == width, targets.scene.height == height { return targets }
        let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        color.storageMode = .private
        color.usage = [.renderTarget, .shaderRead]
        let depth = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: width, height: height, mipmapped: false
        )
        depth.storageMode = .private
        depth.usage = .renderTarget
        guard let scene = queue.device.makeTexture(descriptor: color), let depthTexture = queue.device.makeTexture(descriptor: depth),
              let post = queue.device.makeTexture(descriptor: color) else {
            throw MeshRenderer.Failure.resource("shader lab targets")
        }
        let targets = (scene: scene, depth: depthTexture, post: post)
        self.targets = targets
        return targets
    }

    // MARK: - Graphs

    /// A GPU-generated wood texture: fractal noise warps growth rings, with fine fibers.
    static func woodGrain() throws -> TextureShaderGraph {
        try TextureShaderGraph { t in
            let uv = t.gridUV
            let warp = t.fractalNoise(t.vector(uv * 3, t.constant(Float(0.5))), octaves: 5)
            let rings = ((uv.y * 22 + warp * 7).sin() * 0.5 + 0.5).pow(1.6)
            let fibers = t.noise(t.vector(uv.x * 4, uv.y * 260, t.constant(Float(1.5))), kind: .value)
            let light = t.constant(SIMD3<Float>(0.62, 0.38, 0.19))
            return light.mix(t.constant(SIMD3<Float>(0.3, 0.15, 0.065)), by: rings * 0.75 + fibers * 0.25)
        }
    }

    /// Cools shadows and warms highlights by luminance, as a reusable post-process graph.
    static func splitTone() throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            let amount = p.uniform("Amount", Float(0.35))
            let luminance = color.rgb.luminance()
            let cool = p.constant(SIMD3<Float>(0.88, 1.0, 1.16))
            let warm = p.constant(SIMD3<Float>(1.12, 1.0, 0.85))
            let tint = cool.mix(warm, by: luminance.smoothstep(0.02, 0.9))
            let graded = color.rgb * p.constant(SIMD3<Float>(1, 1, 1)).mix(tint, by: amount)
            return p.vector(graded, color.a)
        }
    }

    /// Veined stone: fractal noise warps sine bands in world space.
    static func marble() throws -> SurfaceShaderGraph {
        try SurfaceShaderGraph { s in
            let scale = s.uniform("Scale", Float(2.2))
            let warp = s.uniform("Warp", Float(1.4))
            let sharpness = s.uniform("Sharpness", Float(7))
            let p = s.worldPosition * scale
            let turbulence = s.fractalNoise(p, octaves: 5).named("turbulence")
            // Thin dark veins follow the zero crossings of warped sine bands.
            let bands = (p.x + p.y * 0.6 + turbulence * warp * 6).sin().abs().named("bands")
            let veins = bands.oneMinus().pow(sharpness).named("veins")
            let stone = s.constant(SIMD3<Float>(0.86, 0.85, 0.82)).mix(s.constant(SIMD3<Float>(0.62, 0.64, 0.66)), by: turbulence)
            let vein = s.constant(SIMD3<Float>(0.12, 0.13, 0.15))
            return .init(baseColor: stone.mix(vein, by: veins), roughness: veins.remap(0, 1, 0.12, 0.42))
        }
    }

    /// Hammered metal: a tangent-space normal from finite differences of cellular dimples.
    static func hammered() throws -> SurfaceShaderGraph {
        try SurfaceShaderGraph { s in
            let density = s.uniform("Dimples", Float(22))
            let depth = s.uniform("Depth", Float(0.9))
            let roughness = s.uniform("Roughness", Float(0.28))
            func height(_ uv: ShaderValue<SIMD2<Float>>) -> ShaderValue<Float> {
                let scaled = uv * density
                let cell = s.cellularNoise(s.vector(scaled.x * 2, scaled.y, s.constant(Float(0.5))))
                return cell.x * cell.x
            }
            let step: Float = 0.0015
            let center = height(s.uv0)
            let du = (height(s.uv0 + s.constant(SIMD2<Float>(step, 0))) - center) / step
            let dv = (height(s.uv0 + s.constant(SIMD2<Float>(0, step))) - center) / step
            let slope = depth / density
            let normal = s.vector(-du * slope, -dv * slope, s.constant(Float(1))).named("dimple normal")
            let copper = s.constant(SIMD3<Float>(0.95, 0.62, 0.48))
            return .init(baseColor: copper.mix(s.constant(SIMD3<Float>(0.7, 0.38, 0.27)), by: center * 2),
                         roughness: roughness + center * 0.3, normal: normal)
        }
    }

    /// An interference-style hue that shifts with viewing angle.
    static func thinFilm() throws -> SurfaceShaderGraph {
        try SurfaceShaderGraph { s in
            let thickness = s.uniform("Thickness", Float(1.1))
            let offset = s.uniform("Hue", Float(0.55))
            let facing = s.normal.dot(s.viewDirection).saturated()
            let hue = (facing.oneMinus() * thickness + offset).fract().named("film hue")
            let film = s.vector(hue, s.constant(Float(0.55)), s.constant(Float(0.95))).hsvToRGB()
            return .init(baseColor: film, roughness: s.constant(Float(0.12)))
        }
    }

    /// A sampled texture with a graph-authored UV transform: tiling, rotation, and tint.
    static func transformedTexture() throws -> SurfaceShaderGraph {
        try SurfaceShaderGraph { s in
            let tiling = s.uniform("Tiling", Float(1.5))
            let rotation = s.uniform("Rotation", Float(0.44))
            let tint = s.uniform("Tint", SIMD3<Float>(1.05, 1, 0.94))
            let grain = s.texture("Grain", content: .color, addressing: .mirroredRepeat)
            let centered = s.uv0 - 0.5
            let c = rotation.cos()
            let n = rotation.sin()
            let turned = s.vector(centered.x * c - centered.y * n, centered.x * n + centered.y * c)
            let wood = grain.sample(turned * tiling + 0.5).rgb
            let shade = wood.luminance()
            return .init(baseColor: wood * tint, roughness: shade.remap(0, 0.5, 0.62, 0.32).saturated())
        }
    }

    /// Two travelling waves displace the surface; crests turn to foam.
    static func waves() throws -> SurfaceShaderGraph {
        try SurfaceShaderGraph { s in
            let time = s.uniform("Time", Float(0))
            let amplitude = s.uniform("Amplitude", Float(0.07))
            let frequency = s.uniform("Frequency", Float(5.5))
            let foam = s.uniform("Foam", Float(0.55))
            let x = s.objectPosition.x
            let y = s.objectPosition.y
            let primary = (x * frequency + time).sin()
            let secondary = (y * frequency * 1.7 + x * frequency * 0.4 - time * 1.3).sin() * 0.45
            let height = ((primary + secondary) * amplitude).named("wave height")
            // The fragment stage sees the deformed world height above the pool.
            let surface = s.uniform("Level", Float(0.22))
            let crest = ((s.worldPosition.y - surface) / (amplitude * 1.45 + 0.001)).smoothstep(foam, foam + 0.25)
            let water = s.constant(SIMD3<Float>(0.015, 0.09, 0.13))
            return .init(
                baseColor: water.mix(s.constant(SIMD3<Float>(0.82, 0.88, 0.9)), by: crest),
                roughness: crest.remap(0, 1, 0.06, 0.55),
                positionOffset: s.displacement(height)
            )
        }
    }

    private static func vase() throws -> MeshGeometry {
        let profile: [(Float, Float)] = [
            (0, 0), (0.32, 0), (0.42, 0.18), (0.48, 0.45), (0.42, 0.78), (0.26, 1.02), (0.22, 1.18), (0.3, 1.32), (0.28, 1.36),
        ]
        return try .lathe(profile: profile.map { LatheProfilePoint(radius: $0.0, y: $0.1) }, segments: 96, capTop: false)
    }
}
