import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Renders scene graphs straight down onto the XZ plane with an orthographic view.
struct PunctualLightingFixture {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let side: Int
    let halfExtent: Float
    private let color: any MTLTexture
    private let depth: any MTLTexture

    init(side: Int = 64, halfExtent: Float = 4) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        self.side = side
        self.halfExtent = halfExtent
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        depth = try #require(device.makeTexture(descriptor: depthDescriptor))
    }

    /// A rough dielectric with no specular reflectance (IOR 1) and no ambient or material light.
    static func diffuse(baseColor: SIMD4<Float> = SIMD4(repeating: 1)) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: baseColor,
            transmission: PhysicallyBasedMaterial.Transmission(factor: 0, indexOfRefraction: 1)
        )
    }

    /// A square in the XZ plane facing +Y.
    func floor(halfExtent: Float, material: PhysicallyBasedMaterial? = nil) throws -> MeshRenderer {
        let corners: [SIMD3<Float>] = [
            SIMD3(-halfExtent, 0, halfExtent), SIMD3(halfExtent, 0, halfExtent),
            SIMD3(halfExtent, 0, -halfExtent), SIMD3(-halfExtent, 0, -halfExtent),
        ]
        let geometry = try MeshGeometry(
            vertices: corners.map { MeshVertex(position: $0, normal: SIMD3(0, 1, 0)) }, indices: [0, 1, 2, 0, 2, 3]
        )
        return try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(material ?? Self.diffuse())
        )
    }

    func box(material: MeshMaterial = .unlit) throws -> MeshRenderer {
        try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: .cube(), material: material)
    }

    func render(
        _ roots: [SceneNode], maps: PunctualShadowMaps?, on queue: (any MTLCommandQueue)? = nil,
        onStatistics: ((SceneDrawStatistics) -> Void)? = nil
    ) throws -> [SIMD4<Float>] {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        let graph = SceneGraph(roots: roots, camera: camera)
        let buffer = try #require((queue ?? self.queue).makeCommandBuffer())
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        // World x maps to clip x, world -z to clip y, and height 10...-10 to depth 0...1.
        let scale = 1 / halfExtent
        let clipFromWorld = simd_float4x4(columns: (
            SIMD4(scale, 0, 0, 0), SIMD4(0, 0, -1.0 / 20, 0), SIMD4(0, -scale, 0, 0), SIMD4(0, 0, 0.5, 1)
        ))
        let statistics = try graph.encode(
            pass: pass, commandBuffer: buffer, clipFromWorld: clipFromWorld,
            cameraPosition: SIMD3(0, 10, 0), directionToCamera: SIMD3(0, 1, 0), punctualShadowMaps: maps
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "The lighting draw failed.")")
        onStatistics?(statistics)
        var pixels = [SIMD4<Float16>](repeating: .zero, count: side * side)
        pixels.withUnsafeMutableBytes { bytes in
            color.getBytes(
                bytes.baseAddress!, bytesPerRow: side * MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0
            )
        }
        return pixels.map(SIMD4<Float>.init)
    }

    /// The index of the pixel that sees a point on the XZ plane.
    func pixel(_ x: Float, _ z: Float) -> Int {
        let column = Int(((x / halfExtent) * 0.5 + 0.5) * Float(side))
        let row = Int(((z / halfExtent) * 0.5 + 0.5) * Float(side))
        return row * side + column
    }

    /// Rotates local -Z, a light's emission direction, from `position` toward `target`.
    static func aim(_ position: SIMD3<Float>, at target: SIMD3<Float>) -> SceneTransform {
        let rotation = simd_quatf(from: SIMD3(0, 0, -1), to: simd_normalize(target - position))
        return SceneTransform(translation: position, rotation: rotation)
    }
}

struct PunctualShadowValidationTests {
    @Test("Shadow settings round trip and older shadow data keeps filtered sampling and full strength")
    func coding() throws {
        let decoder = JSONDecoder()
        let legacy = Data(#"{"depthBias":0,"normalBias":1,"filterRadius":1.5,"nearPlane":0.05}"#.utf8)
        #expect(try decoder.decode(PunctualLight.Shadow.self, from: legacy) == PunctualLight.Shadow())
        let basic = try PunctualLight.Shadow(depthBias: 0.01, normalBias: 0, sampling: .basic, shadowStrength: 0.35)
        #expect(try decoder.decode(PunctualLight.Shadow.self, from: JSONEncoder().encode(basic)) == basic)
    }

    @Test("Shadow settings, kinds, cones, and ranges are validated")
    func lightValidation() throws {
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(depthBias: -0.1) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(normalBias: 17) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(filterRadius: 8.5) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(filterRadius: .nan) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(nearPlane: 0) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(shadowStrength: -0.1) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(shadowStrength: 1.1) }
        #expect(throws: PunctualLight.Failure.shadow) { try PunctualLight.Shadow(shadowStrength: .nan) }
        let shadow = try PunctualLight.Shadow(nearPlane: 0.5)
        #expect(throws: PunctualLight.Failure.shadowKind) { try PunctualLight(kind: .directional, shadow: shadow) }
        #expect(throws: PunctualLight.Failure.shadowCone) {
            try PunctualLight(kind: .spot(outerConeAngle: 1.4), shadow: shadow)
        }
        #expect(throws: PunctualLight.Failure.shadowRange) { try PunctualLight(kind: .point(range: 0.5), shadow: shadow) }
        // The original initializer and unshadowed wide cones remain valid.
        let unshadowed = try PunctualLight(kind: .spot(outerConeAngle: 1.4))
        #expect(unshadowed.shadow == nil)
        let shadowed = try PunctualLight(kind: .spot(range: 2, outerConeAngle: 1.3), shadow: shadow)
        #expect(shadowed.shadow == shadow)
        #expect(shadowed != (try PunctualLight(kind: .spot(range: 2, outerConeAngle: 1.3))))
    }

    @Test("Shadow views use reversed infinite depth with a guard band around the covered angle")
    func projection() throws {
        let origin = SIMD3<Float>(1, 2, 3)
        let forward = SIMD3<Float>(0, -1, 0)
        let view = try PunctualShadowMaps.projection(origin: origin, forward: forward, tangent: 0.5, near: 0.25)
        let ahead = view * SIMD4(origin + forward * 4, 1)
        #expect(simd_distance(ahead, SIMD4(0, 0, 0.25, 4)) < 1e-5)
        // A point at the half-angle tangent lands on the clip edge in one axis.
        let edge = view * SIMD4(origin + forward * 4 + SIMD3(2, 0, 0), 1)
        #expect(abs(abs(edge.x / edge.w) + abs(edge.y / edge.w) - 1) < 1e-5)
        #expect(abs(edge.z / edge.w - 0.0625) < 1e-6)
        #expect(PunctualShadowMaps.faceDirections.count == 6)
    }

    @Test("The packed light record matches the 128-byte Metal layout")
    func layout() {
        typealias Record = MeshRenderer.PackedPunctualLight
        #expect(MemoryLayout<Record>.size == 128)
        #expect(MemoryLayout<Record>.stride == 128)
        #expect(MemoryLayout<Record>.offset(of: \.positionKind) == 0)
        #expect(MemoryLayout<Record>.offset(of: \.directionRange) == 16)
        #expect(MemoryLayout<Record>.offset(of: \.colorIntensity) == 32)
        #expect(MemoryLayout<Record>.offset(of: \.cone) == 48)
        #expect(MemoryLayout<Record>.offset(of: \.axisU) == 64)
        #expect(MemoryLayout<Record>.offset(of: \.axisV) == 80)
        #expect(MemoryLayout<Record>.offset(of: \.shadow) == 96)
        #expect(MemoryLayout<Record>.offset(of: \.shadowReceiver) == 112)
        #expect(Record().shadow.x == -1 && Record().shadowReceiver.x == 1)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PunctualShadowTests {
    @Test("Shadow map capacity, resolution, and queue ownership are validated before encoding")
    func resourceValidation() throws {
        let fixture = try PunctualLightingFixture(side: 16)
        #expect(throws: PunctualShadowMaps.Failure.resolution) {
            try PunctualShadowMaps(queue: fixture.queue, resolution: 8)
        }
        #expect(throws: PunctualShadowMaps.Failure.resolution) {
            try PunctualShadowMaps(queue: fixture.queue, resolution: 8192)
        }
        #expect(throws: PunctualShadowMaps.Failure.capacity) {
            try PunctualShadowMaps(queue: fixture.queue, spotCapacity: 0, pointCapacity: 0)
        }
        #expect(throws: PunctualShadowMaps.Failure.capacity) {
            try PunctualShadowMaps(queue: fixture.queue, spotCapacity: 9, pointCapacity: 0)
        }
        #expect(throws: PunctualShadowMaps.Failure.capacity) {
            try PunctualShadowMaps(queue: fixture.queue, spotCapacity: 0, pointCapacity: 5)
        }
        let maps = try PunctualShadowMaps(queue: fixture.queue, resolution: 64, spotCapacity: 1, pointCapacity: 1)
        #expect(maps.resolution == 64 && maps.spotCapacity == 1 && maps.pointCapacity == 1)

        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let light = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 2, 0)),
            light: try PunctualLight(kind: .point(), intensity: 4, shadow: PunctualLight.Shadow())
        )
        let otherQueue = try #require(fixture.device.makeCommandQueue())
        #expect(throws: PunctualShadowMaps.Failure.wrongQueue) {
            try fixture.render([floor, light], maps: maps, on: otherQueue)
        }
        // Without a shadowed light, no depth pass runs, so the maps' queue is not involved.
        let unshadowed = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 2, 0)),
            light: try PunctualLight(kind: .point(), intensity: 4)
        )
        _ = try fixture.render([floor, unshadowed], maps: maps, on: otherQueue)
    }

    @Test("A spot shadow darkens the receiver behind an occluder and leaves the rest of the cone lit")
    func spotShadow() throws {
        let fixture = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let spot = SceneNode(
            transform: PunctualLightingFixture.aim(SIMD3(0, 3, 0), at: .zero),
            light: try PunctualLight(
                kind: .spot(innerConeAngle: 0.7, outerConeAngle: 0.87), intensity: 20, shadow: PunctualLight.Shadow()
            )
        )
        // The occluder's shadow falls around x = 2, outside its own footprint from above.
        let occluder = SceneNode(
            transform: SceneTransform(translation: SIMD3(1, 1.5, 0), scale: SIMD3(repeating: 0.6)), mesh: try fixture.box()
        )
        let maps = try PunctualShadowMaps(queue: fixture.queue, resolution: 512, spotCapacity: 1, pointCapacity: 0)
        let shadowed = try fixture.render([floor, spot, occluder], maps: maps)
        let unshadowed = try fixture.render([floor, spot, occluder], maps: nil)

        let behind = fixture.pixel(2.2, 0)
        let opposite = fixture.pixel(-2.2, 0)
        #expect(unshadowed[behind].x > 0.2)
        #expect(abs(unshadowed[behind].x - unshadowed[opposite].x) < 0.01 * unshadowed[opposite].x)
        #expect(shadowed[behind].x < 0.02 * unshadowed[behind].x)
        #expect(abs(shadowed[opposite].x - unshadowed[opposite].x) <= 0.005 * unshadowed[opposite].x)
        // Outside the cone the floor stays black either way.
        #expect(shadowed[fixture.pixel(3.8, 3.8)] == SIMD4(0, 0, 0, 1))
    }

    @Test("Point and spot shadow strengths mix independently while retaining the seven cached projections",
          arguments: [ShadowMapSampling.filtered, .variance])
    func shadowStrength(spotSampling: ShadowMapSampling) async throws {
        let fixture = try PunctualLightingFixture()
        let maps = try PunctualShadowMaps(
            queue: fixture.queue, resolution: 128, spotCapacity: 1, pointCapacity: 1, updateMode: .onDemand
        )
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4), castsShadow: false)
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(1, 1.5, 0), scale: SIMD3(repeating: 0.6)), mesh: try fixture.box()
        )
        let pose = PunctualLightingFixture.aim(SIMD3(0, 3, 0), at: .zero)
        let pointKind = PunctualLight.Kind.point(range: 7)
        let spotKind = PunctualLight.Kind.spot(range: 7, innerConeAngle: 0.7, outerConeAngle: 0.87)
        var point = SceneNode(transform: pose)
        var spot = SceneNode(transform: pose)
        let behind = fixture.pixel(2.2, 0), opposite = fixture.pixel(-2.2, 0)

        func draw(_ pointStrength: Float, _ spotStrength: Float, usingMaps: Bool = true, passes: Int) throws
            -> [SIMD4<Float>] {
            point.light = try PunctualLight(
                kind: pointKind, color: SIMD3(1, 0, 0), intensity: 20,
                shadow: .init(sampling: .basic, shadowStrength: pointStrength)
            )
            spot.light = try PunctualLight(
                kind: spotKind, color: SIMD3(0, 1, 0), intensity: 20,
                shadow: .init(sampling: spotSampling, varianceBlurRadius: 1, shadowStrength: spotStrength)
            )
            return try fixture.render([floor, point, spot, blocker], maps: usingMaps ? maps : nil) { statistics in
                #expect(statistics.shadowPassCount == passes)
            }
        }

        func depths() async throws -> [Data] {
            var slices: [Data] = []
            for slice in 0..<maps.sliceCount {
                slices.append(try await TextureTransfer.read(from: maps.texture, slice: slice, commandQueue: fixture.queue).bytes)
            }
            return slices
        }

        let full = try draw(1, 1, passes: 7)
        let initialDepths = try await depths()
        let unshadowed = try draw(1, 1, usingMaps: false, passes: 0)
        #expect(unshadowed[behind].x > 0.2 && unshadowed[behind].y > 0.2)
        #expect(full[behind].x < 0.02 * unshadowed[behind].x && full[behind].y < 0.02 * unshadowed[behind].y)
        #expect(full[opposite] == unshadowed[opposite])

        let mixed = try draw(0, 0.35, passes: 0)
        #expect(mixed[behind].x == unshadowed[behind].x)
        #expect(abs(mixed[behind].y - (0.35 * full[behind].y + 0.65 * unshadowed[behind].y)) < 0.002)
        #expect(mixed[opposite] == unshadowed[opposite])

        let alternate = try draw(0.6, 0, passes: 0)
        #expect(alternate[behind].y == unshadowed[behind].y)
        #expect(abs(alternate[behind].x - (0.6 * full[behind].x + 0.4 * unshadowed[behind].x)) < 0.002)
        #expect(alternate[opposite] == unshadowed[opposite])

        #expect(try draw(0, 0, passes: 0) == unshadowed)
        #expect(try draw(1, 1, passes: 0) == full)
        #expect(try await depths() == initialDepths)
    }

    @Test("Only nondefault punctual shadow strength needs archive v26")
    func strengthArchive() throws {
        let fixture = try PunctualLightingFixture(side: 16)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)

        func archive(_ strength: Float) throws -> (SceneArchive, PunctualLight) {
            let light = try PunctualLight(kind: .spot(), intensity: 8, shadow: .init(shadowStrength: strength))
            let graph = SceneGraph(roots: [SceneNode(light: light)], camera: camera)
            return (try SceneArchive(capturing: graph), light)
        }

        let (legacy, legacyLight) = try archive(1)
        let legacyJSON = String(decoding: legacy.manifest, as: UTF8.self)
        #expect(legacyJSON.contains("\"version\" : 1") && !legacyJSON.contains("shadowStrength"))
        #expect(try legacy.load(device: fixture.device).graph.roots.first?.light == legacyLight)

        let (authored, authoredLight) = try archive(0.35)
        let authoredJSON = String(decoding: authored.manifest, as: UTF8.self)
        #expect(authoredJSON.contains("\"version\" : 26"))
        #expect(try authored.load(device: fixture.device).graph.roots.first?.light == authoredLight)
        let mislabeled = Data(authoredJSON.replacingOccurrences(of: "\"version\" : 26", with: "\"version\" : 25").utf8)
        #expect(throws: SceneArchive.Failure.decoding("punctual shadow strength requires archive version 26")) {
            try SceneArchive(manifest: mislabeled, binary: authored.binary)
        }
    }

    @Test("Punctual shadow casters follow the directional rules for skinning, masks, blending, and opt-out")
    func casterRules() throws {
        let fixture = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let spot = SceneNode(
            transform: PunctualLightingFixture.aim(SIMD3(0, 3, 0), at: .zero),
            light: try PunctualLight(
                kind: .spot(innerConeAngle: 0.7, outerConeAngle: 0.87), intensity: 20, shadow: PunctualLight.Shadow()
            )
        )
        // Occluders sit 1 m from the axis at five azimuths; each shadow centers 2 m out.
        let azimuths = (0..<5).map { Float($0) * 2 * .pi / 5 }
        func place(_ index: Int) -> SceneTransform {
            SceneTransform(
                translation: SIMD3(cos(azimuths[index]), 1.5, sin(azimuths[index])), scale: SIMD3(repeating: 0.6)
            )
        }
        func cube(_ material: PhysicallyBasedMaterial) throws -> MeshRenderer {
            try fixture.box(material: .physicallyBased(material))
        }
        func masked(alpha: Float) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(baseColorFactor: SIMD4(1, 1, 1, alpha), alphaMode: .mask(cutoff: 0.5))
        }
        let geometry = MeshGeometry.cube()
        let influence = VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0))
        let skin = try MeshSkin(jointCount: 1, influences: Array(repeating: influence, count: geometry.vertices.count))
        let skinned = try MeshRenderer(device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, skin: skin)
        let occluders = [
            // The joint palette, not the node, moves this cube off the floor's center.
            SceneNode(mesh: skinned, jointPalette: try JointPalette(matrices: [place(0).matrix])),
            SceneNode(transform: place(1), mesh: try cube(masked(alpha: 1))),
            SceneNode(transform: place(2), mesh: try cube(masked(alpha: 0.25))),
            SceneNode(transform: place(3), mesh: try cube(PhysicallyBasedMaterial(alphaMode: .blend))),
            SceneNode(transform: place(4), mesh: try fixture.box(), castsShadow: false),
        ]
        let maps = try PunctualShadowMaps(queue: fixture.queue, resolution: 512, spotCapacity: 1, pointCapacity: 0)
        let shadowed = try fixture.render([floor, spot] + occluders, maps: maps)
        let unshadowed = try fixture.render([floor, spot] + occluders, maps: nil)

        for (index, casts) in [true, true, false, false, false].enumerated() {
            let pixel = fixture.pixel(2.2 * cos(azimuths[index]), 2.2 * sin(azimuths[index]))
            #expect(unshadowed[pixel].x > 0.2, "Occluder \(index)")
            if casts {
                #expect(shadowed[pixel].x < 0.02 * unshadowed[pixel].x, "Occluder \(index) should cast a shadow.")
            } else {
                #expect(abs(shadowed[pixel].x - unshadowed[pixel].x) <= 0.005 * unshadowed[pixel].x, "Occluder \(index)")
            }
        }
    }

    @Test("A point shadow blocks occluders that fall on different cube faces")
    func pointShadow() throws {
        let fixture = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let light = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 1, 0)),
            light: try PunctualLight(kind: .point(), intensity: 20, shadow: PunctualLight.Shadow())
        )
        func occluder(_ position: SIMD3<Float>, size: Float) throws -> SceneNode {
            SceneNode(transform: SceneTransform(translation: position, scale: SIMD3(repeating: size)), mesh: try fixture.box())
        }
        // Each occluder and the floor point behind it share a face: -X, +Z, and -Y.
        let occluders = [
            try occluder(SIMD3(-1.2, 0.6, 0), size: 0.3),
            try occluder(SIMD3(0, 0.6, 1.2), size: 0.3),
            try occluder(SIMD3(0.4, 0.5, -0.4), size: 0.2),
        ]
        let maps = try PunctualShadowMaps(queue: fixture.queue, resolution: 512, spotCapacity: 0, pointCapacity: 1)
        let shadowed = try fixture.render([floor, light] + occluders, maps: maps)
        let unshadowed = try fixture.render([floor, light] + occluders, maps: nil)

        for (x, z) in [(Float(-2.6), Float(0)), (0, 2.6), (0.8, -0.8)] {
            let index = fixture.pixel(x, z)
            #expect(unshadowed[index].x > 0.05, "(\(x), \(z)) should be lit without shadows")
            #expect(shadowed[index].x < 0.05 * unshadowed[index].x, "(\(x), \(z)) should be shadowed")
        }
        for (x, z) in [(Float(2.6), Float(0)), (0, -2.6), (-0.8, 0.8)] {
            let index = fixture.pixel(x, z)
            #expect(unshadowed[index].x > 0.05)
            #expect(abs(shadowed[index].x - unshadowed[index].x) <= 0.005 * unshadowed[index].x, "(\(x), \(z))")
        }
    }

    @Test("A self-shadowing receiver at a grazing angle has no acne", arguments: [false, true])
    func grazingBias(point: Bool) throws {
        let fixture = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let shadow = try PunctualLight.Shadow()
        // The spot meets the floor about 83 degrees from its normal. The low point light
        // spans five cube faces and grazes the floor everywhere beyond a meter.
        let light = point
            ? SceneNode(
                transform: SceneTransform(translation: SIMD3(0.3, 0.15, -0.2)),
                light: try PunctualLight(kind: .point(), intensity: 2, shadow: shadow)
            )
            : SceneNode(
                transform: PunctualLightingFixture.aim(SIMD3(0, 0.5, 4), at: SIMD3(0, 0, -1)),
                light: try PunctualLight(kind: .spot(innerConeAngle: 0.2, outerConeAngle: 0.5), intensity: 40, shadow: shadow)
            )
        let maps = try PunctualShadowMaps(queue: fixture.queue, resolution: 256, spotCapacity: 1, pointCapacity: 1)
        let shadowed = try fixture.render([floor, light], maps: maps)
        let unshadowed = try fixture.render([floor, light], maps: nil)

        let brightest = unshadowed.map(\.x).max() ?? 0
        #expect(brightest > 0.05)
        #expect(unshadowed.filter { $0.x > 0.005 }.count > fixture.side * fixture.side / 8)
        var worst: Float = 0
        for (a, b) in zip(shadowed, unshadowed) {
            worst = max(worst, abs(a.x - b.x) / max(b.x, 0.002))
        }
        #expect(worst < 0.01, "The largest relative difference was \(worst).")
    }

    @Test(
        "Shadowed lights beyond capacity render unshadowed in depth-first order",
        arguments: [false, true]
    )
    func capacity(points: Bool) throws {
        let fixture = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let shadow = try PunctualLight.Shadow()
        func light(_ x: Float) throws -> SceneNode {
            let position = SIMD3<Float>(x, 2, 0)
            return points
                ? SceneNode(
                    transform: SceneTransform(translation: position),
                    light: try PunctualLight(kind: .point(), intensity: 8, shadow: shadow)
                )
                : SceneNode(
                    transform: PunctualLightingFixture.aim(position, at: SIMD3(x, 0, 0)),
                    light: try PunctualLight(
                        kind: .spot(innerConeAngle: 0.44, outerConeAngle: 0.52), intensity: 8, shadow: shadow
                    )
                )
        }
        func occluder(_ x: Float) throws -> SceneNode {
            SceneNode(transform: SceneTransform(translation: SIMD3(x, 1, 0), scale: SIMD3(repeating: 0.3)), mesh: try fixture.box())
        }
        let left = try light(-2)
        let right = try light(2)
        let blockers = [try occluder(-1.6), try occluder(1.6)]
        func maps(_ count: Int) throws -> PunctualShadowMaps {
            try PunctualShadowMaps(
                queue: fixture.queue, resolution: 256, spotCapacity: points ? 0 : count, pointCapacity: points ? count : 0
            )
        }
        let one = try maps(1)
        let none = try fixture.render([floor, left, right] + blockers, maps: nil)
        let both = try fixture.render([floor, left, right] + blockers, maps: maps(2))
        let leftFirst = try fixture.render([floor, left, right] + blockers, maps: one)
        // Moving the right light first gives it the only slot.
        let rightFirst = try fixture.render([floor, SceneNode(children: [right]), left] + blockers, maps: one)

        let leftShadow = fixture.pixel(-1.2, 0)
        let rightShadow = fixture.pixel(1.2, 0)
        for index in [leftShadow, rightShadow] {
            #expect(both[index].x < 0.75 * none[index].x)
        }
        func same(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Bool { abs(a.x - b.x) <= 0.005 * max(b.x, 0.01) }
        #expect(same(leftFirst[leftShadow], both[leftShadow]))
        #expect(same(leftFirst[rightShadow], none[rightShadow]))
        #expect(same(rightFirst[rightShadow], both[rightShadow]))
        #expect(same(rightFirst[leftShadow], none[leftShadow]))
    }
}
