import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// A generated field of rocks, crystals and trees beside a row of hero objects.
///
/// The field is a 16 × 16 grid of chunks. Each chunk holds one scene node per
/// object kind, and each node draws its kind with one `InstanceSet` and switches
/// between three prepared meshes with `LevelOfDetail`. The camera frustum culls
/// whole chunk nodes. The hero row shows the KHR material extensions on
/// `PhysicallyBasedMaterial`. A `LightRig` sun and a generated sky light both
/// renderers. In path-traced mode, motion refits the traced scene's top level
/// with `PathTracingScene.updateTransforms(_:)`; a new instance count, level
/// selection or material prepares a new scene.
@MainActor
final class LabScaleLab: LabSceneFixture {
    typealias Kind = LabScaleGeometry.Kind

    static let maximumInstances = 100_000
    static let grid = 16
    static let extent: Float = 200
    static let lodDistances: [Float] = [0, 24, 60]
    private static let plinthTop: Float = 0.3

    /// One field object: chunk-local placement, tint and optional hovering spin.
    private struct Placement {
        var bucket: Int
        var translation: SIMD3<Float>
        var yaw: Float
        var scale: SIMD3<Float>
        var tint: SIMD4<Float>
        /// Radians per second; zero keeps the object still.
        var spin: Float
        var phase: Float

        func matrix(time: Float) -> simd_float4x4 {
            var angle = yaw
            var offset = translation
            if spin != 0 {
                angle += spin * time
                offset.y += 0.3 + 0.18 * sin(time * 1.3 + phase)
            }
            return SceneTransform(translation: offset, rotation: simd_quatf(angle: angle, axis: SIMD3(0, 1, 0)), scale: scale).matrix
        }
    }

    /// One chunk node's instances.
    private struct Chunk {
        let kind: Kind
        let set: InstanceSet
        var placements: [Placement] = []
        var tints: [SIMD4<Float>] = []
        var animated = false
    }

    private struct Hero {
        let id = UUID()
        let geometry: MeshGeometry
        var renderer: MeshRenderer?
        var key: [Double] = []
    }

    /// Counts from the latest frame, for the status line and benchmark reports.
    private struct FrameStats {
        var draws = SceneDrawStatistics()
        var lodDraws = [0, 0, 0]
        var lodInstances = [0, 0, 0]
        var fieldTriangles = 0
        var totalTriangles = 0
        var animationSeconds: Double = 0
        var animatedFrames = 0
        var refitSeconds: Double = 0
        var refitFrames = 0
    }

    private let queue: any MTLCommandQueue
    private let controls: LabSceneControls
    private let camera: PerspectiveCamera
    private let environment: PhysicallyBasedMaterial.Environment
    private let background: EnvironmentBackground
    private let display: HDRDisplay
    private let bloom: MipmapBloom
    private let sunShadow: DirectionalShadowMap
    private let pathSky: PathTracer.EnvironmentMap
    private let plaid: any MTLTexture
    private let checker: any MTLTexture
    private let film: any MTLTexture
    private let brush: any MTLTexture
    private let fieldMeshes: [[MeshRenderer]]
    private let fieldTriangleCounts: [[Int]]
    private let allPlacements: [Placement]
    private let terrain: SceneNode
    private let plinth: SceneNode
    private var chunks: [Chunk]
    private var fieldNodes: [SceneNode]
    /// Chunk nodes with at least one instance, in graph order.
    private var activeChunks: [Int] = []
    private var fieldCount = -1
    private var appliedLevels: SIMD2<Float>?
    private var appliedView = -1
    private var heroes: [Hero]
    private var rig: LightRig
    private var orbit: OrbitCameraController
    private var time: Float = 2
    private var lastTime: CFTimeInterval?
    private var targets: (scene: any MTLTexture, depth: any MTLTexture, backdrop: any MTLTexture, post: any MTLTexture)?
    private var stats = FrameStats()
    private var statusTime: CFTimeInterval = 0

    private var pathScene: PathTracingScene?
    private var tracer: PathTracer?
    private var accumulation: PathTracer.Accumulation?
    private var traced: (any MTLTexture)?
    private var history: PathTracer.History?
    private var tracedPose: SceneTransform?
    private var pathBuilds = 0
    private var pathRefits = 0
    private var lastBuildSeconds: Double = 0
    private let timing: PathTracer.Timing?
    private var timedBuffer: (any MTLCommandBuffer)?

    init(device: any MTLDevice, destinationFormat: MTLPixelFormat, controls: LabSceneControls) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("scale lab queue") }
        self.queue = queue
        self.controls = controls
        camera = try PerspectiveCamera(verticalFieldOfView: 0.75, near: 0.1, far: 2_500)
        let lighting = try LabScaleGeometry.skyLighting(queue: queue)
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting)
        self.environment = environment
        background = try EnvironmentBackground(lighting: lighting, destinationPixelFormat: .rgba16Float)
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)
        bloom = try MipmapBloom(
            device: device, settings: .init(threshold: 1.5, knee: 0, blending: .additive)
        )
        try bloom.prepare(pixelFormat: .rgba16Float)
        sunShadow = try DirectionalShadowMap(
            queue: queue, resolution: 4096, directionToLight: SIMD3(0, 1, 0), center: .zero, halfExtent: 30, intensity: 0
        )
        pathSky = try LabScaleGeometry.pathSky(device: device)
        plaid = try LabScaleGeometry.plaid(queue: queue)
        checker = try LabScaleGeometry.checker(queue: queue)
        film = try LabScaleGeometry.filmThickness(queue: queue)
        brush = try LabScaleGeometry.brushDirections(queue: queue)
        timing = try? PathTracer.Timing(device: device)

        func lit(
            _ color: SIMD3<Float>, metallic: Float = 0, roughness: Float, texture: (any MTLTexture)? = nil,
            transform: PhysicallyBasedMaterial.TextureTransform = .identity
        ) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                baseColorTexture: texture, metallic: metallic, roughness: roughness, ambientIntensity: 0, receivesShadows: true,
                baseColorFactor: SIMD4(color, 1), alphaMode: .opaque, environment: environment, baseColorTextureTransform: transform
            )
        }

        // Field meshes: three kinds at three levels, all prepared for instancing.
        let atlas = try LabScaleGeometry.treeAtlas(queue: queue)
        var meshes: [[MeshRenderer]] = []
        var triangles: [[Int]] = []
        for kind in Kind.allCases {
            let material: PhysicallyBasedMaterial = switch kind {
            case .rock: try lit(SIMD3(0.62, 0.6, 0.56), roughness: 0.85)
            case .crystal: try lit(SIMD3(0.85, 0.9, 1), roughness: 0.1)
            case .tree: try lit(SIMD3(repeating: 1), roughness: 0.8, texture: atlas)
            }
            var row: [MeshRenderer] = []
            var counts: [Int] = []
            for level in 0..<3 {
                let geometry = try LabScaleGeometry.field(kind, level: level)
                row.append(try MeshRenderer(
                    device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                    material: .physicallyBased(material), supportsInstancing: true
                ))
                counts.append(geometry.indices.count / 3)
            }
            meshes.append(row)
            triangles.append(counts)
        }
        fieldMeshes = meshes
        fieldTriangleCounts = triangles

        // Every placement for the largest field; a smaller field uses a prefix.
        allPlacements = (0..<Self.maximumInstances).map(Self.placement)
        var capacities = Array(repeating: 0, count: Self.grid * Self.grid * Kind.allCases.count)
        for placement in allPlacements { capacities[placement.bucket] += 1 }
        let chunkSize = Self.extent / Float(Self.grid)
        let margin: Float = 1.6
        let bounds = try AxisAlignedBox(
            minimum: SIMD3(-chunkSize / 2 - margin, -0.6, -chunkSize / 2 - margin),
            maximum: SIMD3(chunkSize / 2 + margin, 5.2, chunkSize / 2 + margin)
        )
        var chunks: [Chunk] = []
        var nodes: [SceneNode] = []
        for bucket in capacities.indices {
            let kind = Kind(rawValue: bucket % Kind.allCases.count)!
            let cell = bucket / Kind.allCases.count
            chunks.append(Chunk(kind: kind, set: try InstanceSet(device: device, transforms: [], capacity: max(1, capacities[bucket]))))
            nodes.append(SceneNode(
                transform: SceneTransform(translation: Self.chunkCenter(cell)), instances: chunks[bucket].set,
                cullingBounds: bounds, levelsOfDetail: try LevelOfDetail(levels: (0..<3).map {
                    LevelOfDetail.Level(mesh: meshes[kind.rawValue][$0], distance: Self.lodDistances[$0])
                })
            ))
        }
        self.chunks = chunks
        fieldNodes = nodes

        let ground = try LabScaleGeometry.ground(queue: queue)
        terrain = SceneNode(
            transform: SceneTransform(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
            mesh: try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: try .plane(width: 3_000, height: 3_000),
                material: .physicallyBased(try lit(SIMD3(repeating: 1), roughness: 0.95, texture: ground,
                                                   transform: try .init(scale: SIMD2(repeating: 750))))
            ),
            castsShadow: false
        )
        plinth = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, Self.plinthTop / 2, 0)),
            mesh: try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float,
                geometry: try .bevelBox(size: SIMD3(12.2, Self.plinthTop, 1.8), bevel: 0.04, segments: 2),
                material: .physicallyBased(try lit(SIMD3(0.5, 0.5, 0.52), roughness: 0.7))
            )
        )
        heroes = [
            Hero(geometry: try .bevelBox(size: SIMD3(1.1, 0.5, 1.1), bevel: 0.22, segments: 6)),
            Hero(geometry: try .uvSphere(radius: 0.55, segments: 96, rings: 48)),
            Hero(geometry: try LabScaleGeometry.disc(radius: 0.6, thickness: 0.06, segments: 96)),
            Hero(geometry: try .uvSphere(radius: 0.55, segments: 96, rings: 48)),
            Hero(geometry: try .bevelBox(size: SIMD3(repeating: 0.85), bevel: 0.28, segments: 8)),
            Hero(geometry: try LabScaleGeometry.prism(side: 0.9, height: 1.1)),
            Hero(geometry: try .uvSphere(radius: 0.32, segments: 64, rings: 32)),
            Hero(geometry: .cube()),
        ]
        rig = try LightRig(lights: [])
        orbit = try Self.orbit(view: 0)

        controls.register([
            LabControlGroup(id: "field", title: "Field", controls: [
                LabControl(id: "field.count", title: "Instances", kind: .slider(1_000...Double(Self.maximumInstances), step: 1_000), initial: 20_000),
                LabControl(id: "field.culling", title: "Frustum culling", kind: .toggle, initial: 1),
                LabControl(id: "field.lod", title: "LOD distance", kind: .slider(0.25...4, step: 0.05), initial: 1),
                LabControl(id: "field.hysteresis", title: "Hysteresis", kind: .slider(0...0.5, step: 0.01), initial: 0.1),
                LabControl(id: "field.animate", title: "Animate", kind: .toggle, initial: 1),
                LabControl(id: "scene.view", title: "View", kind: .choice(["Hero row", "Field"]), initial: 0),
            ]),
            LabControlGroup(id: "materials", title: "Materials", controls: [
                LabControl(id: "mat.sheen", title: "Sheen rough", kind: .slider(0...1, step: 0.01), initial: 0.45),
                LabControl(id: "mat.film", title: "Film nm", kind: .slider(100...1_200, step: 10), initial: 680),
                LabControl(id: "mat.aniso", title: "Anisotropy", kind: .slider(0...1, step: 0.01), initial: 0.85),
                LabControl(id: "mat.anisoTurn", title: "Brush turn", kind: .slider(-90...90, step: 1), initial: 0),
                LabControl(id: "mat.dispersion", title: "Dispersion", kind: .slider(0...10, step: 0.1), initial: 3),
                LabControl(id: "mat.glow", title: "Emissive", kind: .slider(0...40, step: 0.5), initial: 14),
                LabControl(id: "tex.tiling", title: "Tiling", kind: .slider(0.5...8, step: 0.1), initial: 2),
                LabControl(id: "tex.turn", title: "Texture turn", kind: .slider(-180...180, step: 1), initial: 20),
            ]),
            LabControlGroup(id: "light", title: "Sun & Sky", controls: [
                LabControl(id: "sun.lux", title: "Sun lux", kind: .slider(0...12, step: 0.05), initial: 5),
                LabControl(id: "sun.azimuth", title: "Azimuth", kind: .slider(-180...180, step: 1), initial: -35),
                LabControl(id: "sun.elevation", title: "Elevation", kind: .slider(3...88, step: 1), initial: 38),
                LabControl(id: "sky.intensity", title: "Sky", kind: .slider(0...3, step: 0.01), initial: 1),
                LabControl(id: "sky.rotation", title: "Sky turn", kind: .slider(-180...180, step: 1), initial: 0),
            ]),
            LabControlGroup(id: "render", title: "Render", controls: [
                LabControl(id: "scene.path", title: "Path traced", kind: .toggle, initial: 0),
                LabControl(id: "scene.samples", title: "Samples", kind: .slider(16...2_048, step: 16), initial: 256),
                LabControl(id: "scene.exposure", title: "Exposure", kind: .slider(-4...4, step: 0.05), initial: 0),
                LabControl(id: "post.bloom", title: "Bloom", kind: .slider(0...2, step: 0.05), initial: 0.35),
                LabControl(id: "post.radius", title: "Bloom spread", kind: .slider(1...32, step: 1), initial: 4),
            ]),
        ])
        try applyControls()
    }

    var animates: Bool { true }
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
        try populate(count: Int(controls.value("field.count")))

        let levels = SIMD2(Float(controls.value("field.lod")), Float(controls.value("field.hysteresis")))
        if levels != appliedLevels {
            for index in fieldNodes.indices {
                let kind = chunks[index].kind
                fieldNodes[index].levelsOfDetail = try LevelOfDetail(levels: (0..<3).map {
                    LevelOfDetail.Level(mesh: fieldMeshes[kind.rawValue][$0], distance: Self.lodDistances[$0] * levels.x,
                                        hysteresis: $0 == 0 ? 0 : levels.y)
                }, currentLevel: fieldNodes[index].levelsOfDetail?.currentLevel ?? 0)
            }
            appliedLevels = levels
        }
        let culling = controls.isOn("field.culling")
        for index in fieldNodes.indices { fieldNodes[index].frustumCulled = culling }

        let view = Int(controls.value("scene.view"))
        if view != appliedView {
            orbit = try Self.orbit(view: view)
            appliedView = view
        }

        for index in heroes.indices {
            let (key, material) = try heroMaterial(index)
            guard key != heroes[index].key || heroes[index].renderer == nil else { continue }
            heroes[index].renderer = try MeshRenderer(
                device: queue.device, colorPixelFormat: .rgba16Float, geometry: heroes[index].geometry,
                material: .physicallyBased(material)
            )
            heroes[index].key = key
        }

        let azimuth = Float(controls.value("sun.azimuth")) * .pi / 180
        let elevation = Float(controls.value("sun.elevation")) * .pi / 180
        let toSun = SIMD3(cos(elevation) * sin(azimuth), sin(elevation), cos(elevation) * cos(azimuth))
        rig = try LightRig(lights: [LightRig.Light(
            id: "sun", kind: .directional(angularDiameter: 0.6 * .pi / 180), position: toSun * 60, target: .zero,
            color: try LightRig.Light.color(temperature: 5_600), emission: .intensity(Float(controls.value("sun.lux")))
        )])
        bloom.settings.strength = Float(controls.value("post.bloom"))
        bloom.settings.radius = Float(controls.value("post.radius")) / 32
        try updateShadow()
        updateStatus(force: true)
    }

    /// Fits the sun's shadow map to the ground the camera looks at: centered ahead
    /// of the orbit target and wider for distant views. Casters outside it are skipped.
    private func updateShadow() throws {
        let pose = orbit.pose
        let forward = SIMD3(-sin(pose.yaw), 0, -cos(pose.yaw))
        let halfExtent = max(18, pose.distance * 0.9)
        let sun = rig.directionalShadow
        try sunShadow.update(
            directionToLight: sun?.directionToLight ?? SIMD3(0, 1, 0),
            center: SIMD3(pose.target.x, 0, pose.target.z) + forward * (halfExtent * 0.5), halfExtent: halfExtent,
            intensity: sun?.intensity ?? 0, bias: 0.0015, lightAngularRadius: min(sun?.angularRadius ?? 0, 0.25)
        )
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
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("scale lab frame") }
        try encode(destination: drawable.texture, buffer: buffer, delta: delta, pathSamples: 2)
        buffer.addCompletedHandler { command in
            completed(command.status == .completed ? nil : command.error?.localizedDescription ?? "The scale lab frame did not finish.")
        }
        buffer.present(drawable)
        buffer.commit()
    }

    #if os(macOS)
    var commandQueue: any MTLCommandQueue { queue }

    func encodeOffscreen(into destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
        try encode(destination: destination, buffer: commandBuffer, delta: 1.0 / 60, pathSamples: 1)
    }

    /// Renders the current state without advancing time, so path tracing accumulates.
    func capture(to url: URL, width: Int, height: Int) throws {
        let target = try LabImageCapture.target(device: queue.device, width: width, height: height)
        let passes = controls.isOn("scene.path") ? max(1, sampleTarget / 8) : 2
        for _ in 0..<passes {
            guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("scale lab capture") }
            try encode(destination: target, buffer: buffer, delta: 0, pathSamples: 8)
            buffer.commit()
            buffer.waitUntilCompleted()
            guard buffer.status == .completed else { throw buffer.error ?? MeshRenderer.Failure.resource("scale lab capture") }
        }
        try LabImageCapture.write(target, to: url)
        updateStatus(force: true)
        FileHandle.standardError.write(Data("\(controls.status ?? "")\n".utf8))
    }

    var benchmarkDetails: [String: Any] {
        var details: [String: Any] = [
            "fieldInstances": fieldCount,
            "chunkNodes": activeChunks.count,
            "animationCPUMeanMs": stats.animatedFrames > 0 ? stats.animationSeconds / Double(stats.animatedFrames) * 1000 : 0,
            "levelTriangles": Dictionary(uniqueKeysWithValues: Kind.allCases.map { ("\($0)", fieldTriangleCounts[$0.rawValue]) }),
        ]
        if controls.isOn("scene.path"), let statistics = pathScene?.statistics {
            details["path"] = [
                "samplesPerPixel": history?.sampleCount ?? 0,
                "instanceCount": statistics.instanceCount,
                "bottomLevelCount": statistics.bottomLevelCount,
                "storedTriangles": statistics.triangleCount,
                "tracedTriangles": statistics.instancedTriangleCount,
                "bufferBytes": statistics.bufferByteCount,
                "acceleration": statistics.acceleration == .metal ? "metal" : "software",
                "accelerationBytes": statistics.accelerationStructureByteCount,
                "textureBytes": statistics.textureByteCount,
                "boundTextures": statistics.boundTextureCount,
                "bakedTextures": statistics.bakedTextureCount,
                "referencedTextureBytes": statistics.referencedTextureByteCount,
                "topLevelBuilds": statistics.topLevelBuildCount,
                "topLevelRefits": statistics.topLevelRefitCount,
                "sceneBuilds": pathBuilds,
                "lastSceneBuildMs": lastBuildSeconds * 1000,
                "updateTransformsMeanMs": stats.refitFrames > 0 ? stats.refitSeconds / Double(stats.refitFrames) * 1000 : 0,
                "lastTraceGPUMs": timing?.duration.map { $0 * 1000 } ?? -1,
            ] as [String: Any]
        } else {
            details["raster"] = [
                "drawnMeshCount": stats.draws.drawnMeshCount,
                "drawnInstanceCount": stats.draws.drawnInstanceCount,
                "culledMeshCount": stats.draws.culledMeshCount,
                "shadowDrawCount": stats.draws.shadowDrawCount,
                "culledShadowCount": stats.draws.culledShadowCount,
                "fieldDrawsPerLevel": stats.lodDraws,
                "fieldInstancesPerLevel": stats.lodInstances,
                "fieldTriangles": stats.fieldTriangles,
                "drawnTriangles": stats.totalTriangles,
            ] as [String: Any]
        }
        return details
    }
    #endif

    // MARK: - Frame

    private func encode(destination: any MTLTexture, buffer: any MTLCommandBuffer, delta: Float, pathSamples: Int) throws {
        try orbit.update(deltaTime: delta)
        try updateShadow()
        if controls.isOn("field.animate"), delta > 0 {
            let start = CACurrentMediaTime()
            time += delta
            try animateField()
            stats.animationSeconds += CACurrentMediaTime() - start
            stats.animatedFrames += 1
        }
        let pose = orbit.cameraTransform
        let graph = try sceneGraph(pose: pose)
        let targets = try renderTargets(width: destination.width, height: destination.height)
        let path = controls.isOn("scene.path")
        let skyIntensity = Float(controls.value("sky.intensity"))
        let skyYaw = Float(controls.value("sky.rotation")) * .pi / 180
        let source: any MTLTexture
        if path {
            source = try trace(graph: graph, pose: pose, width: destination.width, height: destination.height,
                               samples: pathSamples, sky: .map(pathSky, intensity: skyIntensity, yaw: skyYaw), buffer: buffer)
        } else {
            history = nil
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
            let drawn = try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: sunShadow, transmissionBackdrop: targets.backdrop)
            try measure(drawn, graph: graph, pose: pose, aspect: Float(destination.width) / Float(destination.height))
            source = targets.scene
        }
        let displaySource: any MTLTexture
        if bloom.settings.strength > 0, source.width > 1, source.height > 1 {
            try bloom.encode(source: source, destination: targets.post, commandBuffer: buffer)
            displaySource = targets.post
        } else {
            displaySource = source
        }
        try display.encode(
            source: displaySource, destination: destination,
            settings: .init(exposureStops: Float(controls.value("scene.exposure")), antiAliasing: !path), commandBuffer: buffer
        )
        updateStatus(force: false)
    }

    /// The hero row, ground and non-empty chunk nodes, with each chunk's level of
    /// detail stored for this camera so hysteresis carries to the next frame.
    private func sceneGraph(pose: SceneTransform) throws -> SceneGraph {
        let culling = controls.isOn("field.culling")
        var roots = [terrain, plinth]
        for (index, hero) in heroes.enumerated() {
            roots.append(SceneNode(id: hero.id, transform: heroTransform(index), mesh: hero.renderer, frustumCulled: culling))
        }
        let fixed = roots.count
        roots += activeChunks.map { fieldNodes[$0] }
        var graph = SceneGraph(roots: roots, camera: camera, cameraTransform: pose)
        graph.environmentIntensity = Float(controls.value("sky.intensity"))
        graph.environmentRotation = Float(controls.value("sky.rotation")) * .pi / 180
        try graph.updateLevelsOfDetail(cameraPosition: pose.translation)
        for (offset, index) in activeChunks.enumerated() {
            fieldNodes[index] = graph.roots[fixed + offset]
        }
        return graph
    }

    /// Spins and lifts the hovering crystals: one `replaceAll` per animated chunk.
    private func animateField() throws {
        for index in chunks.indices where chunks[index].animated {
            try chunks[index].set.replaceAll(
                transforms: chunks[index].placements.map { $0.matrix(time: time) }, tints: chunks[index].tints
            )
        }
    }

    /// Fills each chunk's instance set with its share of the first `count` placements.
    private func populate(count: Int) throws {
        guard count != fieldCount else { return }
        var buckets = Array(repeating: [Placement](), count: chunks.count)
        for placement in allPlacements.prefix(count) { buckets[placement.bucket].append(placement) }
        for index in chunks.indices {
            let placements = buckets[index]
            let tints = placements.map(\.tint)
            try chunks[index].set.replaceAll(transforms: placements.map { $0.matrix(time: time) }, tints: tints)
            chunks[index].placements = placements
            chunks[index].tints = tints
            chunks[index].animated = placements.contains { $0.spin != 0 }
        }
        activeChunks = chunks.indices.filter { chunks[$0].set.count > 0 }
        fieldCount = count
    }

    /// Counts what the color pass drew. Totals come from `SceneDrawStatistics`; the
    /// per-level split repeats the frustum test with `ViewFrustum` on each chunk's bounds.
    private func measure(_ drawn: SceneDrawStatistics, graph: SceneGraph, pose: SceneTransform, aspect: Float) throws {
        stats.draws = drawn
        let frustum = try ViewFrustum(clipFromWorld: try camera.projection(aspectRatio: aspect) * simd_inverse(pose.matrix))
        let culling = controls.isOn("field.culling")
        var draws = [0, 0, 0], instances = [0, 0, 0], fieldTriangles = 0
        for index in activeChunks {
            let node = fieldNodes[index]
            guard let level = node.levelsOfDetail?.currentLevel, let bounds = node.cullingBounds else { continue }
            let box = try bounds.transformed(by: node.transform.matrix)
            if culling, frustum.classify(box: box) == .outside { continue }
            let count = chunks[index].set.count
            draws[level] += 1
            instances[level] += count
            fieldTriangles += count * fieldTriangleCounts[chunks[index].kind.rawValue][level]
        }
        var total = fieldTriangles
        for node in graph.roots.prefix(2 + heroes.count) {
            guard let mesh = node.mesh else { continue }
            let box = try mesh.bounds.transformed(by: node.transform.matrix)
            if culling, frustum.classify(box: box) == .outside { continue }
            total += mesh.geometry.indices.count / 3
        }
        stats.lodDraws = draws
        stats.lodInstances = instances
        stats.fieldTriangles = fieldTriangles
        stats.totalTriangles = total
    }

    // MARK: - Path tracing

    private func trace(
        graph: SceneGraph, pose: SceneTransform, width: Int, height: Int, samples: Int,
        sky: PathTracer.Environment, buffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        let lighting = try rig.pathLighting(sky: sky)
        if let scene = pathScene {
            let start = CACurrentMediaTime()
            do {
                if try scene.updateTransforms(graph) {
                    pathRefits += 1
                    stats.refitSeconds += CACurrentMediaTime() - start
                    stats.refitFrames += 1
                }
            } catch let failure as PathTracingScene.Failure {
                // New instance counts, level selections and hero materials change the traced structure.
                guard case .structureChanged = failure else { throw failure }
                pathScene = nil
            }
        }
        if pathScene == nil {
            let start = CACurrentMediaTime()
            let scene = try PathTracingScene(queue: queue, graph: graph)
            tracer = try PathTracer(queue: queue, scene: scene, lighting: lighting)
            pathScene = scene
            accumulation = nil
            history = nil
            pathBuilds += 1
            lastBuildSeconds = CACurrentMediaTime() - start
        }
        guard let tracer else { throw MeshRenderer.Failure.resource("scale lab tracer") }
        if accumulation == nil || accumulation?.width != width || accumulation?.height != height {
            accumulation = try tracer.makeAccumulation(width: width, height: height)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .shaderWrite]
            traced = queue.device.makeTexture(descriptor: descriptor)
            history = nil
        }
        guard let accumulation, let traced else { throw MeshRenderer.Failure.resource("scale lab path targets") }
        if tracedPose.map({ $0.translation != pose.translation || $0.rotation != pose.rotation }) ?? true {
            history = nil
            tracedPose = pose
        }
        if let previous = history, previous.sampleCount >= sampleTarget { return traced }
        // One timing measures one pending command buffer; skip it while the last one runs.
        let timed = timedBuffer.map { $0.status == .completed || $0.status == .error } ?? true ? timing : nil
        let settings = PathTracer.Settings(maxBounces: 6, seed: 7, maximumSampleRadiance: 40)
        func run(_ previous: PathTracer.History?) throws -> PathTracer.History {
            try tracer.encode(camera: camera, cameraTransform: pose, destination: traced, accumulation: accumulation,
                              previous: previous, commandBuffer: buffer, samplesPerPixel: samples,
                              settings: settings, lighting: lighting, timing: timed)
        }
        do {
            history = try run(history)
        } catch PathTracer.Failure.lightingChanged {
            history = try run(nil)
        } catch PathTracer.Failure.sceneChanged {
            // Moved instances: the refit scene restarts accumulation.
            history = try run(nil)
        }
        if timed != nil { timedBuffer = buffer }
        return traced
    }

    // MARK: - Status

    private func updateStatus(force: Bool) {
        let now = CACurrentMediaTime()
        guard force || now - statusTime > 0.25 else { return }
        statusTime = now
        if controls.isOn("scene.path") {
            guard let statistics = pathScene?.statistics else {
                controls.status = "Path traced · preparing \(fieldCount.formatted()) instances."
                return
            }
            let megabytes = Double(statistics.bufferByteCount + statistics.accelerationStructureByteCount) / 1_048_576
            let trace = timing?.duration.map { String(format: " · trace %.1f ms", $0 * 1000) } ?? ""
            controls.status = "Path traced · \(history?.sampleCount ?? 0) spp · \(fieldCount.formatted()) field instances · "
                + "\(statistics.instanceCount.formatted()) traced instances, \(statistics.bottomLevelCount) bottom levels, "
                + "\(statistics.triangleCount.formatted()) stored / \(statistics.instancedTriangleCount.formatted()) traced triangles, "
                + String(format: "%.1f MB", megabytes)
                + " · \(pathBuilds) builds (last \(Int((lastBuildSeconds * 1000).rounded())) ms), \(pathRefits) refits\(trace)"
        } else {
            let draws = stats.draws
            let millions = Double(stats.totalTriangles) / 1_000_000
            controls.status = "Raster · \(fieldCount.formatted()) field instances: \(draws.drawnInstanceCount.formatted()) drawn in "
                + "\(draws.drawnMeshCount) draws, \(draws.culledMeshCount) draws culled · "
                + String(format: "%.2f M triangles", millions)
                + " · LOD 0/1/2 draws \(stats.lodDraws.map(String.init).joined(separator: "/")), instances "
                + "\(stats.lodInstances.map { $0.formatted() }.joined(separator: "/"))"
                + " · shadow draws \(draws.shadowDrawCount), \(draws.culledShadowCount) culled"
        }
    }

    // MARK: - Hero row

    private func heroTransform(_ index: Int) -> SceneTransform {
        let x = -4.9 + 1.4 * Float(index)
        func turn(_ angle: Float) -> simd_quatf { simd_quatf(angle: angle, axis: SIMD3(0, 1, 0)) }
        let top = Self.plinthTop
        switch index {
        case 0: return SceneTransform(translation: SIMD3(x, top + 0.25, 0), rotation: turn(0.45))
        case 1: return SceneTransform(translation: SIMD3(x, top + 0.55, 0), rotation: turn(time * 0.35))
        case 2: return SceneTransform(translation: SIMD3(x, top + 0.6, 0), rotation: turn(-0.45 + 0.3 * sin(time * 0.7)))
        case 3: return SceneTransform(translation: SIMD3(x, top + 0.55, 0))
        case 4: return SceneTransform(translation: SIMD3(x, top + 0.425, 0), rotation: turn(0.6))
        case 5: return SceneTransform(translation: SIMD3(x, top + 0.55, 0), rotation: turn(0.4 + time * 0.25))
        case 6: return SceneTransform(translation: SIMD3(x, top + 0.75 + 0.1 * sin(time * 1.4), 0))
        default: return SceneTransform(translation: SIMD3(x, top + 0.4, 0), rotation: turn(0.5 + time * 0.4), scale: SIMD3(repeating: 0.8))
        }
    }

    /// The material for a hero and the control values it depends on.
    private func heroMaterial(_ index: Int) throws -> (key: [Double], material: PhysicallyBasedMaterial) {
        func value(_ id: String) -> Float { Float(controls.value(id)) }
        let tiling = value("tex.tiling"), turn = value("tex.turn")
        let transform = try PhysicallyBasedMaterial.TextureTransform(rotation: turn * .pi / 180, scale: SIMD2(repeating: tiling))
        func lit(
            _ color: SIMD3<Float>, metallic: Float = 0, roughness: Float, texture: (any MTLTexture)? = nil,
            emissive: SIMD3<Float> = .zero, emissiveStrength: Float = 1,
            clearcoat: PhysicallyBasedMaterial.Clearcoat? = nil, transmission: PhysicallyBasedMaterial.Transmission? = nil,
            specular: PhysicallyBasedMaterial.Specular? = nil, sheen: PhysicallyBasedMaterial.Sheen? = nil,
            anisotropy: PhysicallyBasedMaterial.Anisotropy? = nil, iridescence: PhysicallyBasedMaterial.Iridescence? = nil
        ) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                baseColorTexture: texture, metallic: metallic, roughness: roughness, ambientIntensity: 0, receivesShadows: true,
                baseColorFactor: SIMD4(color, 1), emissiveFactor: emissive, alphaMode: .opaque, environment: environment,
                clearcoat: clearcoat, transmission: transmission, baseColorTextureTransform: transform,
                emissiveStrength: emissiveStrength, specular: specular, sheen: sheen, anisotropy: anisotropy, iridescence: iridescence
            )
        }
        switch index {
        case 0:
            // Velvet: a pink sheen over a tiled plaid.
            let roughness = value("mat.sheen")
            return ([Double(roughness), Double(tiling), Double(turn)], try lit(
                SIMD3(repeating: 1), roughness: 0.85, texture: plaid,
                sheen: try .init(colorFactor: SIMD3(1, 0.62, 0.78), roughness: roughness)
            ))
        case 1:
            // An oil film on a black sphere; the map varies thickness from 120 nm up to the control.
            let thickness = value("mat.film")
            return ([Double(thickness)], try lit(
                SIMD3(0.02, 0.02, 0.025), roughness: 0.12,
                iridescence: try .init(factor: 1, indexOfRefraction: 1.8, minimumThickness: 120, maximumThickness: thickness,
                                       thicknessTexture: .init(texture: film))
            ))
        case 2:
            // Brushed metal: circular grain from a direction map, turned by the rotation.
            let strength = value("mat.aniso"), rotation = value("mat.anisoTurn")
            return ([Double(strength), Double(rotation)], try lit(
                SIMD3(0.93, 0.93, 0.95), metallic: 1, roughness: 0.32,
                anisotropy: try .init(strength: strength, rotation: rotation * .pi / 180, texture: .init(texture: brush))
            ))
        case 3:
            // A navy dielectric whose specular color tints its reflections gold.
            return ([], try lit(SIMD3(0.04, 0.07, 0.2), roughness: 0.18,
                                specular: try .init(factor: 1, colorFactor: SIMD3(4.5, 2.4, 0.6))))
        case 4:
            // Car paint: a rough metallic red under a smooth clear coat.
            return ([], try lit(SIMD3(0.55, 0.02, 0.04), metallic: 0.45, roughness: 0.5,
                                clearcoat: try .init(factor: 1, roughness: 0.04)))
        case 5:
            // A glass prism volume; dispersion splits it by wavelength in the path tracer.
            let dispersion = value("mat.dispersion")
            return ([Double(dispersion)], try lit(
                SIMD3(repeating: 1), roughness: 0,
                transmission: try .init(factor: 1, indexOfRefraction: 1.52, thickness: 0.45, attenuationDistance: 1.5,
                                        attenuationColor: SIMD3(0.8, 0.92, 1), dispersion: dispersion)
            ))
        case 6:
            let strength = value("mat.glow")
            return ([Double(strength)], try lit(SIMD3(0.05, 0.03, 0.02), roughness: 0.4,
                                                emissive: SIMD3(1, 0.42, 0.12), emissiveStrength: strength))
        default:
            return ([Double(tiling), Double(turn)], try PhysicallyBasedMaterial(
                baseColorTexture: checker, alphaMode: .opaque, baseColorTextureTransform: transform, shadingModel: .unlit
            ))
        }
    }

    // MARK: - Field layout

    private static func chunkCenter(_ cell: Int) -> SIMD3<Float> {
        let size = extent / Float(grid)
        return SIMD3((Float(cell % grid) + 0.5) * size - extent / 2, 0, (Float(cell / grid) + 0.5) * size - extent / 2)
    }

    /// The deterministic placement of field object `index`. Objects avoid the hero row
    /// and the space in front of it; a prefix of the sequence is a smaller field.
    private static func placement(_ index: Int) -> Placement {
        func random(_ salt: UInt64) -> Float { LabScaleGeometry.random(index, salt) }
        let pick = random(40)
        let kind: Kind = pick < 0.42 ? .rock : (pick < 0.75 ? .tree : .crystal)
        // Rocks and crystals may come close behind the row; trees stand farther back.
        let clear: (x: Float, back: Float) = kind == .tree ? (16, -16) : (9, -3)
        var x: Float = 0, z: Float = 0
        var salt: UInt64 = 1
        repeat {
            x = (random(salt) - 0.5) * extent
            z = (random(salt + 1) - 0.5) * extent
            salt += 2
        } while abs(x) < clear.x && z > clear.back && z < 20
        let column = min(grid - 1, Int((x / extent + 0.5) * Float(grid)))
        let row = min(grid - 1, Int((z / extent + 0.5) * Float(grid)))
        let cell = row * grid + column
        let center = chunkCenter(cell)
        let size = random(41)
        let palette: [SIMD3<Float>]
        var scale: SIMD3<Float>
        var spin: Float = 0
        var y: Float = 0
        switch kind {
        case .rock:
            palette = [SIMD3(0.95, 0.93, 0.9), SIMD3(0.8, 0.76, 0.7), SIMD3(1, 0.95, 0.85),
                       SIMD3(0.7, 0.7, 0.72), SIMD3(0.88, 0.8, 0.68), SIMD3(0.6, 0.58, 0.55)]
            let base = 0.4 + 1.2 * size * size
            scale = SIMD3(base * (0.8 + 0.4 * random(42)), base * (0.7 + 0.5 * random(43)), base * (0.8 + 0.4 * random(44)))
            y = -0.08 * scale.y
        case .crystal:
            palette = [SIMD3(0.55, 0.85, 1), SIMD3(0.75, 0.6, 1), SIMD3(1, 0.55, 0.8),
                       SIMD3(0.5, 1, 0.85), SIMD3(1, 0.85, 0.5), SIMD3(0.9, 0.95, 1)]
            scale = SIMD3(repeating: 0.6 + 0.8 * size)
            if random(45) < 0.5 { spin = (0.4 + 0.8 * random(46)) * (random(47) < 0.5 ? -1 : 1) }
        case .tree:
            palette = [SIMD3(1, 1, 1), SIMD3(0.85, 0.95, 0.8), SIMD3(0.75, 0.85, 0.7),
                       SIMD3(1, 0.95, 0.75), SIMD3(0.9, 1, 0.9), SIMD3(0.7, 0.8, 0.75)]
            let base = 0.65 + 0.8 * size
            scale = SIMD3(base * (0.9 + 0.2 * random(48)), base, base * (0.9 + 0.2 * random(49)))
        }
        let tint = palette[min(palette.count - 1, Int(random(50) * Float(palette.count)))]
        return Placement(
            bucket: cell * Kind.allCases.count + kind.rawValue, translation: SIMD3(x, y, z) - center,
            yaw: random(51) * 2 * .pi, scale: scale, tint: SIMD4(tint, 1), spin: spin, phase: random(52) * 2 * .pi
        )
    }

    private static func orbit(view: Int) throws -> OrbitCameraController {
        let limits = OrbitCameraController.Limits(distance: 3...160, pitch: 0.02...1.4)
        if view == 1 {
            return try OrbitCameraController(
                target: SIMD3(0, 0, -28), distance: 56, yaw: 0.35, pitch: 0.3, projection: .perspective(verticalFieldOfView: 0.75),
                limits: limits, dampingRate: 8
            )
        }
        return try OrbitCameraController(
            target: SIMD3(0, 1.15, -1), distance: 9.6, yaw: 0, pitch: 0.1, projection: .perspective(verticalFieldOfView: 0.75),
            limits: limits, dampingRate: 8
        )
    }

    private func renderTargets(width: Int, height: Int) throws -> (
        scene: any MTLTexture, depth: any MTLTexture, backdrop: any MTLTexture, post: any MTLTexture
    ) {
        if let targets, targets.scene.width == width, targets.scene.height == height { return targets }
        func make(_ format: MTLPixelFormat, _ usage: MTLTextureUsage, mipmapped: Bool = false) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: mipmapped)
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = usage
            guard let texture = queue.device.makeTexture(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("scale lab targets")
            }
            return texture
        }
        let made = (
            scene: try make(.rgba16Float, [.renderTarget, .shaderRead]),
            depth: try make(MeshRenderer.depthPixelFormat, .renderTarget),
            backdrop: try make(.rgba16Float, .shaderRead, mipmapped: true),
            post: try make(.rgba16Float, [.renderTarget, .shaderRead])
        )
        targets = made
        return made
    }
}
