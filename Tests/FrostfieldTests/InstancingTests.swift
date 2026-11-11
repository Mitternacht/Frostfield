import Foundation
import Frostfield
import Metal
import Testing
import simd

struct InstanceSetValidationTests {
    @Test("The packed instance record matches the 64-byte Metal layout")
    func layout() {
        typealias Record = InstanceSet.PackedInstance
        #expect(MemoryLayout<Record>.size == 64)
        #expect(MemoryLayout<Record>.stride == 64)
        #expect(MemoryLayout<Record>.offset(of: \.row0) == 0)
        #expect(MemoryLayout<Record>.offset(of: \.row1) == 16)
        #expect(MemoryLayout<Record>.offset(of: \.row2) == 32)
        #expect(MemoryLayout<Record>.offset(of: \.tint) == 48)
        // Rows hold the affine transform's first three rows: the translation is in w.
        let record = Record(transform: SceneTransform(translation: SIMD3(4, 5, 6), scale: SIMD3(2, 3, 7)).matrix,
                            tint: SIMD4(0.1, 0.2, 0.3, 0.4))
        #expect(record.row0 == SIMD4(2, 0, 0, 4))
        #expect(record.row1 == SIMD4(0, 3, 0, 5))
        #expect(record.row2 == SIMD4(0, 0, 7, 6))
        #expect(record.tint == SIMD4(0.1, 0.2, 0.3, 0.4))
        #expect(Record.identity.row0 == SIMD4(1, 0, 0, 0) && Record.identity.tint == SIMD4(repeating: 1))
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct InstancingTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("One instanced draw renders every instance with its transform and tint, matching separate nodes")
    func instancedDraw() throws {
        let geometry = try GeometryRenderFixture.square(0.25)
        let mesh = try fixture.unlit(geometry, supportsInstancing: true)
        // The middle instance turns 60° about Y with a stretched width, so its normal turns too.
        let placements = [
            SceneTransform(translation: SIMD3(-1, 0, 0)),
            SceneTransform(rotation: simd_quatf(angle: .pi / 3, axis: SIMD3(0, 1, 0)), scale: SIMD3(1.5, 1, 1)),
            SceneTransform(translation: SIMD3(1, 0, 0)),
        ]
        let tints: [SIMD4<Float>] = [SIMD4(1, 0, 0, 1), SIMD4(0.5, 1, 0.25, 1), SIMD4(0, 0, 1, 1)]
        let set = try InstanceSet(device: fixture.device, transforms: placements.map(\.matrix), tints: tints)
        let node = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh,
            materialParameters: SIMD4(1, 1, 0.5, 1), instances: set
        )
        let (pixels, statistics) = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera))
        #expect(statistics.drawnMeshCount == 1)
        #expect(statistics.drawnInstanceCount == 3)

        let parameters = SIMD4<Float>(1, 1, 0.5, 1)
        #expect(isNearly(fixture.pixel(pixels, x: 16, y: 32), GeometryRenderFixture.unlitFacingCamera(tints[0] * parameters)))
        #expect(isNearly(fixture.pixel(pixels, x: 48, y: 32), GeometryRenderFixture.unlitFacingCamera(tints[2] * parameters)))
        // The turned normal is (sin 60°, 0, cos 60°), so the unlit color is (0.22 + 0.78|n|) × tint.
        let turned = SIMD3<Float>(0.22, 0.22, 0.22) + 0.78 * SIMD3(sin(Float.pi / 3), 0, cos(Float.pi / 3))
        let middle = turned * SIMD3(0.5, 1, 0.125)
        #expect(isNearly(fixture.pixel(pixels, x: 32, y: 32), SIMD4(middle, 1)))
        // Between instances the target stays clear.
        #expect(fixture.pixel(pixels, x: 24, y: 32) == .zero)
        #expect(fixture.pixel(pixels, x: 32, y: 20) == .zero)

        // Separate nodes with composed transforms and tinted parameters draw the same image.
        let plain = try fixture.unlit(geometry)
        let separate = zip(placements, tints).map { placement, tint in
            var world = placement
            world.translation += SIMD3(0, 0, -2)
            return SceneNode(transform: world, mesh: plain, materialParameters: parameters * tint)
        }
        let reference = try fixture.render(SceneGraph(roots: separate, camera: fixture.camera))
        #expect(reference.statistics.drawnMeshCount == 3)
        #expect(zip(pixels, reference.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.002) })
    }

    @Test("Edits reach the next encode while earlier command buffers keep the records they encoded")
    func editsAndBuffering() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25), supportsInstancing: true)
        let set = try InstanceSet(device: fixture.device, transforms: [translationMatrix(-1.5)], capacity: 4)
        #expect(set.capacity == 4 && set.count == 1 && set.tints == [SIMD4(repeating: 1)])
        let graph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh, instances: set)],
            camera: fixture.camera
        )
        // Five encodes before any commit: three take the persistent buffers and two
        // use transient copies, yet each frame shows only the position it encoded.
        let positions: [Float] = [-1.5, -0.75, 0, 0.75, 1.5]
        var frames: [(color: any MTLTexture, buffer: any MTLCommandBuffer)] = []
        for x in positions {
            try set.setTransform(translationMatrix(x), at: 0)
            let target = try fixture.makeTarget()
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            try graph.encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer)
            frames.append((target.color, buffer))
        }
        // An edit after encoding must not leak into those frames.
        try set.setTransform(translationMatrix(-1.5), at: 0)
        for frame in frames { frame.buffer.commit() }
        for frame in frames {
            frame.buffer.waitUntilCompleted()
            try #require(frame.buffer.status == .completed, "\(frame.buffer.error?.localizedDescription ?? "Incomplete")")
        }
        for (x, frame) in zip(positions, frames) {
            let pixels = try fixture.pixels(of: frame.color)
            for other in positions {
                let covered = fixture.pixel(pixels, x: fixture.column(x: other, depth: 2), y: 32).w == 1
                #expect(covered == (other == x), "frame at \(x), probe at \(other)")
            }
        }

        // Growing within the capacity draws the new count; tints update in place.
        try set.replaceAll(transforms: [translationMatrix(-1), translationMatrix(1)])
        try set.setTint(SIMD4(0, 1, 0, 1), at: 1)
        let (pixels, statistics) = try fixture.render(graph)
        #expect(statistics.drawnInstanceCount == 2)
        #expect(isNearly(fixture.pixel(pixels, x: 48, y: 32), GeometryRenderFixture.unlitFacingCamera(SIMD4(0, 1, 0, 1))))
        #expect(isNearly(fixture.pixel(pixels, x: 16, y: 32), GeometryRenderFixture.unlitFacingCamera(SIMD4(repeating: 1))))

        // An empty set draws nothing.
        try set.replaceAll(transforms: [])
        let empty = try fixture.render(graph)
        #expect(empty.statistics.drawnMeshCount == 0 && empty.pixels.allSatisfy { $0 == .zero })
    }

    @Test("Mirrored instance sets flip the draw's winding so front faces stay visible")
    func mirroredInstances() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25), supportsInstancing: true)
        let mirror = SceneTransform(translation: SIMD3(0.5, 0, 0), scale: SIMD3(-1, 1, 1)).matrix
        let set = try InstanceSet(device: fixture.device, transforms: [mirror, translationMatrix(-0.5) * mirror])
        #expect(set.isMirrored)
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh, instances: set)
        let (pixels, _) = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera))
        #expect(fixture.pixel(pixels, x: fixture.column(x: 0.5, depth: 2), y: 32).w == 1)
        #expect(fixture.pixel(pixels, x: fixture.column(x: 0, depth: 2), y: 32).w == 1)
        // A set cannot mix handedness, since one draw has one winding.
        #expect(throws: InstanceSet.Failure.mixedHandedness) {
            try set.setTransform(translationMatrix(0), at: 0)
        }
        #expect(throws: InstanceSet.Failure.mixedHandedness) {
            try InstanceSet(device: fixture.device, transforms: [mirror, translationMatrix(0)])
        }
    }

    @Test("Instanced casters shadow directional and punctual receivers")
    func instancedShadows() throws {
        // Directional: blockers at z = −2 shadow a receiver at z = −3 along (0.6, 0, 0.8).
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let blockers = try InstanceSet(
            device: fixture.device, transforms: [translationMatrix(0.75), translationMatrix(-0.55, 0.6)]
        )
        let blockerMesh = try fixture.unlit(try GeometryRenderFixture.square(0.25), supportsInstancing: true)
        let blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: blockerMesh, instances: blockers
        )
        var unshadowedBlocker = blocker
        unshadowedBlocker.castsShadow = false
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let shadowed = try fixture.render(SceneGraph(roots: [receiver, blocker], camera: fixture.camera), shadowMap: shadowMap)
        let lit = try fixture.render(SceneGraph(roots: [receiver, unshadowedBlocker], camera: fixture.camera), shadowMap: shadowMap)
        #expect(shadowed.statistics.shadowDrawCount == 1)
        #expect(lit.statistics.shadowDrawCount == 0)
        // The receiver's lit value is 0.16 + 0.84 × 0.8; in shadow only the 0.16 ambient remains.
        let center = fixture.pixel(lit.pixels, x: 32, y: 32)
        #expect(abs(center.x - 0.832) < 0.01)
        #expect(abs(fixture.pixel(shadowed.pixels, x: 32, y: 32).x - 0.16) < 0.01)
        // The second instance's shadow falls around (−1.3, 0.6), beside the blocker's own image.
        let secondShadow = (x: fixture.column(x: -1.3, depth: 3), y: fixture.row(y: 0.6, depth: 3))
        #expect(abs(fixture.pixel(shadowed.pixels, x: secondShadow.x, y: secondShadow.y).x - 0.16) < 0.01)
        #expect(abs(fixture.pixel(lit.pixels, x: secondShadow.x, y: secondShadow.y).x - 0.832) < 0.01)

        // Punctual: one spot light, two instanced cubes, two shadows.
        let lighting = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try lighting.floor(halfExtent: 4))
        let spot = SceneNode(
            transform: PunctualLightingFixture.aim(SIMD3(0, 3, 0), at: .zero),
            light: try PunctualLight(
                kind: .spot(innerConeAngle: 0.7, outerConeAngle: 0.87), intensity: 20, shadow: PunctualLight.Shadow()
            )
        )
        let cube = try MeshRenderer(
            device: lighting.device, colorPixelFormat: .rgba16Float, geometry: .cube(), supportsInstancing: true
        )
        let cubes = try InstanceSet(device: lighting.device, transforms: [
            SceneTransform(translation: SIMD3(1, 1.5, 0), scale: SIMD3(repeating: 0.6)).matrix,
            SceneTransform(translation: SIMD3(-1, 1.5, 0), scale: SIMD3(repeating: 0.6)).matrix,
        ])
        let occluders = SceneNode(mesh: cube, instances: cubes)
        let maps = try PunctualShadowMaps(queue: lighting.queue, resolution: 512, spotCapacity: 1, pointCapacity: 0)
        let withShadows = try lighting.render([floor, spot, occluders], maps: maps)
        let without = try lighting.render([floor, spot, occluders], maps: nil)
        for x: Float in [2.2, -2.2] {
            let behind = lighting.pixel(x, 0)
            #expect(without[behind].x > 0.2)
            #expect(withShadows[behind].x < 0.02 * without[behind].x)
        }
        let side = lighting.pixel(0, 2.2)
        #expect(abs(withShadows[side].x - without[side].x) <= 0.005 * without[side].x)
    }

    @Test("Picking reports the nearest instance and keeps the lower index on ties")
    func picking() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25), supportsInstancing: true)
        // Instance 3 sits in front of instance 1; instance 4 duplicates instance 2.
        let set = try InstanceSet(device: fixture.device, transforms: [
            translationMatrix(-1), translationMatrix(0), translationMatrix(1), translationMatrix(0, 0, 0.5),
            translationMatrix(1),
        ])
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh, instances: set)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let left = try #require(try graph.pick(at: SIMD2(0.25, 0.5), aspectRatio: 1))
        #expect(left.nodeID == node.id && left.instanceIndex == 0)
        #expect(abs(left.intersection.position.x - -1) < 1e-5 && abs(left.intersection.position.z - -2) < 1e-5)
        let center = try #require(try graph.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1))
        #expect(center.instanceIndex == 3)
        #expect(abs(center.intersection.position.z - -1.5) < 1e-5)
        #expect(try graph.pick(at: SIMD2(0.75, 0.5), aspectRatio: 1)?.instanceIndex == 2)
        #expect(try graph.pick(at: SIMD2(0.375, 0.5), aspectRatio: 1) == nil)

        // Non-instanced nodes report no instance.
        let plain = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let single = try #require(try SceneGraph(roots: [plain], camera: fixture.camera).pick(at: SIMD2(0.5, 0.5), aspectRatio: 1))
        #expect(single.instanceIndex == nil)
    }

    @Test("Path-traced instances cover the raster pixels with each instance's tinted albedo")
    func pathTracedInstances() throws {
        let side = 32
        let fixture = try GeometryRenderFixture(side: side)
        let material = try PhysicallyBasedMaterial(roughness: 0.5, baseColorFactor: SIMD4(1, 1, 1, 1), alphaMode: .opaque)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(0.3),
            material: .physicallyBased(material), supportsInstancing: true
        )
        let tints: [SIMD4<Float>] = [SIMD4(1, 0.2, 0.2, 1), SIMD4(0.2, 1, 0.2, 1), SIMD4(0.2, 0.2, 1, 1)]
        let set = try InstanceSet(device: fixture.device, transforms: [
            translationMatrix(-1),
            SceneTransform(rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1))).matrix,
            SceneTransform(translation: SIMD3(1, 0.4, 0), scale: SIMD3(1, 2, 1)).matrix,
        ], tints: tints)
        let graph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh, instances: set)],
            camera: fixture.camera
        )
        let raster = try fixture.render(graph).pixels

        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        // Three top-level instances share one two-triangle bottom level and trace six triangles.
        #expect(scene.triangleCount == 2)
        #expect(scene.statistics.instanceCount == 3)
        #expect(scene.statistics.bottomLevelCount == 1)
        #expect(scene.statistics.instancedTriangleCount == 6)
        #expect(scene.materialCount == 3)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        func output() throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = [.shaderWrite, .shaderRead]
            return try #require(fixture.device.makeTexture(descriptor: descriptor))
        }
        let destination = try output()
        let guides = PathTracer.Guides(normalDepth: try output(), albedoRoughness: try output())
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: fixture.camera, cameraTransform: SceneTransform(), destination: destination,
            accumulation: accumulation, guides: guides, commandBuffer: buffer, settings: .init(maxBounces: 1, seed: 3)
        )
        try fixture.finish(buffer)
        let depth = try fixture.pixels(of: guides.normalDepth)
        let albedo = try fixture.pixels(of: guides.albedoRoughness)

        // Away from silhouettes, the traced first hits cover exactly the rasterized pixels.
        func rasterCovered(_ x: Int, _ y: Int) -> Bool { raster[y * side + x].w > 0.5 }
        var interior = 0
        for y in 1..<(side - 1) {
            for x in 1..<(side - 1) {
                let neighbors = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].map(rasterCovered)
                guard neighbors.allSatisfy({ $0 == neighbors[0] }) else { continue }
                interior += 1
                #expect((depth[y * side + x].w > 0) == neighbors[0], "pixel \(x), \(y)")
            }
        }
        #expect(interior > side * side * 3 / 4)
        // Instance tints reach the traced base color.
        for (tint, column) in zip(tints, [fixture.column(x: -1, depth: 2), side / 2, fixture.column(x: 1, depth: 2)]) {
            let row = column == fixture.column(x: 1, depth: 2) ? fixture.row(y: 0.4, depth: 2) : side / 2
            let sample = albedo[row * side + column]
            #expect(simd_reduce_max(abs(SIMD3(sample.x, sample.y, sample.z) - SIMD3(tint.x, tint.y, tint.z))) < 0.01)
        }
    }

    @Test("Instance sets and instanced draws are validated before encoding")
    func validation() throws {
        let device = fixture.device
        #expect(throws: InstanceSet.Failure.capacity) { try InstanceSet(device: device, transforms: []) }
        #expect(throws: InstanceSet.Failure.capacity) {
            try InstanceSet(device: device, transforms: [], capacity: InstanceSet.maximumCapacity + 1)
        }
        #expect(throws: InstanceSet.Failure.count) {
            try InstanceSet(device: device, transforms: [matrix_identity_float4x4, matrix_identity_float4x4], capacity: 1)
        }
        var projective = matrix_identity_float4x4
        projective.columns.0.w = 0.5
        for invalid in [SceneTransform(scale: SIMD3(1, 0, 1)).matrix, translationMatrix(.nan), projective] {
            #expect(throws: InstanceSet.Failure.transform(1)) {
                try InstanceSet(device: device, transforms: [matrix_identity_float4x4, invalid])
            }
        }
        #expect(throws: InstanceSet.Failure.tintCount) {
            try InstanceSet(device: device, transforms: [matrix_identity_float4x4], tints: [])
        }
        #expect(throws: InstanceSet.Failure.tint(0)) {
            try InstanceSet(device: device, transforms: [matrix_identity_float4x4], tints: [SIMD4(.infinity, 0, 0, 1)])
        }
        let set = try InstanceSet(device: device, transforms: [matrix_identity_float4x4])
        #expect(throws: InstanceSet.Failure.index(1)) { try set.setTransform(matrix_identity_float4x4, at: 1) }
        #expect(throws: InstanceSet.Failure.index(-1)) { try set.setTint(SIMD4(repeating: 1), at: -1) }
        let revision = set.revision
        #expect(throws: InstanceSet.Failure.transform(0)) { try set.setTransform(translationMatrix(.nan), at: 0) }
        #expect(set.revision == revision && set.transforms == [matrix_identity_float4x4])

        // Drawing instances needs a mesh prepared for them, and no transmission.
        let plain = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        let graph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: plain, instances: set)],
            camera: fixture.camera
        )
        #expect(throws: MeshRenderer.Failure.instancingNotPrepared) { try fixture.render(graph) }
        #expect(throws: MeshRenderer.Failure.instancingNotPrepared) {
            try PathTracingScene(queue: fixture.queue, graph: SceneGraph(
                roots: [SceneNode(mesh: try MeshRenderer(
                    device: device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(0.25),
                    material: .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque))
                ), instances: set)],
                camera: fixture.camera
            ))
        }
        let glass = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(0.25),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                alphaMode: .opaque, transmission: .init(factor: 1, indexOfRefraction: 1.5)
            )),
            supportsInstancing: true
        )
        let transmitting = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: glass, instances: set)],
            camera: fixture.camera
        )
        #expect(throws: MeshRenderer.Failure.instancedTransmission) { try fixture.render(transmitting) }
    }
}
