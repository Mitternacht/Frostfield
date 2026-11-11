import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PlanarReflectionTests {
    private let gpu: GeometryRenderFixture
    private let center = SIMD3<Float>(0, 0, -4)
    private let localNormal = simd_normalize(SIMD3<Float>(0.2, 0.16, 1))
    private let frame = SceneTransform(
        translation: SIMD3(4.5, -3, 8), rotation: simd_quatf(angle: 0.55, axis: simd_normalize(SIMD3(1, 2, 0.25)))
    )

    init() throws { gpu = try GeometryRenderFixture(side: 128) }

    @Test("A tilted transformed mirror captures the correct half-space and aligns on the main surface",
          arguments: [MeshDepthMode.forward, .reversed], [true, false])
    func tiltedMirror(mode: MeshDepthMode, perspective: Bool) throws {
        let target = try texture(.rgba16Float, usage: [.renderTarget, .shaderRead])
        let targetDepth = try texture(.depth32Float, usage: [.renderTarget, .shaderRead])
        let mirror = try mirrorNode(sampling: target, mode: mode)
        let reflection = PlanarReflection(plane: try worldPlane(), reflector: mirror.id)
        let camera = try camera(mode: mode, perspective: perspective)
        if !perspective {
            let virtual = try #require(try reflection.reflectedView(camera, depthMode: mode))
            let vertical = try #require(PrimitiveView.verticalScale(
                of: virtual.camera.projectionMatrix * virtual.camera.viewMatrix,
                orthographicBack: virtual.directionToCamera
            ))
            #expect(abs(vertical - 0.53) < 1e-5)
        }
        let first = planePoint(-0.8, 0.22)
        let second = planePoint(0.84, -0.37)
        let firstDirection = reflectedDirection(at: first, perspective: perspective)
        let secondDirection = reflectedDirection(at: second, perspective: perspective)
        let red = SIMD3<Float>(0.85, 0.1, 0.05)
        let green = SIMD3<Float>(0.06, 0.8, 0.25)
        let blue = SIMD3<Float>(0.05, 0.1, 0.95)
        let nodes: [SceneNode] = [
            try marker(at: first + 0.9 * firstDirection, toward: -firstDirection, color: red, mode: mode),
            try marker(at: second + 1.1 * secondDirection, toward: -secondDirection, color: green, mode: mode),
            // A closer draw covers the red marker when the clip plane is omitted.
            try marker(at: first - 0.55 * firstDirection, toward: -firstDirection, color: blue, mode: mode),
        ]
        var reflector = mirror
        let excluded = try marker(at: first + 0.25 * firstDirection, toward: -firstDirection, color: blue, mode: mode)
        reflector.children = [SceneNode(
            transform: .init(translation: worldToLocalPosition(excluded.transform.translation, of: mirror.transform),
                             rotation: simd_inverse(mirror.transform.rotation) * excluded.transform.rotation),
            mesh: excluded.mesh, frustumCulled: false
        )]
        let scene = SceneGraph(roots: [SceneNode(transform: frame, children: nodes + [reflector])], camera: gpu.camera)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let counts = try #require(try reflection.encode(
            scene, camera: camera, color: target, depth: targetDepth, depthMode: mode, commandBuffer: command
        ))
        // The wrong-side marker disables CPU culling so the GPU must clip it.
        #expect(counts.drawnMeshCount == 3)
        let output = try gpu.makeTarget()
        let mainPass = gpu.pass(color: output.color, depth: output.depth, depthMode: mode)
        let mirrorScene = SceneGraph(roots: [SceneNode(transform: frame, children: [mirror])], camera: gpu.camera)
        try mirrorScene.encode(
            pass: mainPass, commandBuffer: command, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
            cameraPosition: frame.translation,
            directionToCamera: perspective ? nil : frame.rotation.act(SIMD3(0, 0, 1)),
            depthMode: mode, graphCamera: camera
        )
        try gpu.finish(command)

        let capture = try gpu.pixels(of: target)
        let composed = try gpu.pixels(of: output.color)
        for (point, expected) in [(first, red), (second, green)] {
            let (x, y) = pixel(point, projection: camera.projectionMatrix)
            #expect(isNearly(gpu.pixel(capture, x: gpu.side - 1 - x, y: y), SIMD4(expected, 1), tolerance: 0.006))
            #expect(isNearly(gpu.pixel(composed, x: x, y: y), SIMD4(expected, 1), tolerance: 0.006))
        }
        #expect(!capture.contains(where: { $0.z > 0.9 }))
        #expect(gpu.pixel(composed, x: 4, y: 4) == .zero)
    }

    @Test("A directional sky follows tilted off-center reflected rays without writing depth",
          arguments: [MeshDepthMode.forward, .reversed], [true, false])
    func reflectedSky(mode: MeshDepthMode, perspective: Bool) async throws {
        let color = try texture(.rgba16Float, usage: [.renderTarget, .shaderRead])
        let depth = try texture(.depth32Float, usage: [.renderTarget, .shaderRead])
        let background = try directionalBackground()
        let skyRotation = simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0))
        let camera = try camera(mode: mode, perspective: perspective)
        let reflector = PlanarReflection(plane: try worldPlane(), reflector: UUID())
        let command = try #require(gpu.queue.makeCommandBuffer())
        let graph = SceneGraph(roots: [], camera: gpu.camera)
        let stats = try #require(try reflector.encode(
            graph, camera: camera, color: color, depth: depth, depthMode: mode,
            background: background, backgroundOptions: .init(rotation: skyRotation, intensity: 0.7), commandBuffer: command
        ))
        #expect(stats.drawnMeshCount == 0)
        try gpu.finish(command)

        let pixels = try gpu.pixels(of: color)
        let projection = camera.projectionMatrix
        for (x, y) in [(19, 32), (63, 71), (98, 90)] {
            let ndcX = 2 * (Float(x) + 0.5) / Float(gpu.side) - 1
            let ndcY = 1 - 2 * (Float(y) + 0.5) / Float(gpu.side)
            let ray = perspective ? SIMD3(
                (ndcX + projection[2].x) / projection[0].x, (ndcY + projection[2].y) / projection[1].y, -1
            ) : SIMD3<Float>(0, 0, -1)
            let direction = simd_normalize(frame.rotation.act(ray))
            let normal = frame.rotation.act(localNormal)
            let reflected = direction - 2 * simd_dot(direction, normal) * normal
            let sky = skyRotation.inverse.act(reflected)
            let expected = SIMD4((sky + 1) * 0.35, 1)
            #expect(isNearly(gpu.pixel(pixels, x: gpu.side - 1 - x, y: y), expected, tolerance: 0.002))
        }
        let stored = try await TextureTransfer.read(
            from: depth, region: MTLRegionMake2D(64, 64, 1, 1), commandQueue: gpu.queue
        )
        #expect(stored.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) } == (mode == .reversed ? 0 : 1))
    }

    @Test("A multisampled reflection resolves sky and geometry coverage, nearest depth, and color mips",
          .enabled(if: MTLCreateSystemDefaultDevice().map {
              PlanarReflection.MultisampleTargets.isSupported(sampleCount: 4, on: $0)
          } == true, "Requires 4× raster sampling and native depth resolve"),
          arguments: [MeshDepthMode.forward, .reversed])
    func multisampledCapture(mode: MeshDepthMode) async throws {
        let color = try texture(.rgba16Float, usage: [.renderTarget, .shaderRead], mipmapped: true)
        let depth = try texture(.depth32Float, usage: [.renderTarget, .shaderRead])
        let msaa = try PlanarReflection.MultisampleTargets(device: gpu.device, width: gpu.side, height: gpu.side, sampleCount: 4)
        let background = try EnvironmentBackground(device: gpu.device, source: .color(SIMD3(0, 0, 1)), sampleCount: 4)
        let reflector = PlanarReflection(plane: try Plane3D(normal: SIMD3(0, 0, 1), through: SIMD3(0, 0, -4)), reflector: UUID())
        let projection = try gpu.camera.projection(aspectRatio: 1, depthMode: mode)
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: gpu.camera.near, far: gpu.camera.far
        )
        let positions: [SIMD3<Float>] = [SIMD3(-1, -0.7, 0), SIMD3(0.3, 1, 0), SIMD3(1, -0.7, 0)]
        let geometry = try MeshGeometry(
            vertices: positions.map { MeshVertex(position: $0, normal: SIMD3(0, 0, -1)) }, indices: [0, 1, 2]
        )
        let material = try ClassicMaterial(model: .basic, color: SIMD3(1, 0, 0))
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .classic(material),
            depthMode: mode, sampleCount: 4
        )
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh),
        ], camera: gpu.camera)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let stats = try #require(try reflector.encode(
            graph, camera: camera, color: color, depth: depth, depthMode: mode, generateMipmaps: true,
            background: background, multisample: msaa, commandBuffer: command
        ))
        #expect(stats.drawnMeshCount == 1 && msaa.sampleCount == 4)
        try gpu.finish(command)

        let pixels = try gpu.pixels(of: color)
        let mixed = pixels.indices.filter { pixels[$0].x > 0 && pixels[$0].x < 1 }
        try #require(mixed.count > 12)
        #expect(mixed.allSatisfy { pixels[$0].x + pixels[$0].z == 1 && pixels[$0].w == 1 })
        #expect(gpu.pixel(pixels, x: 64, y: 64) == SIMD4(1, 0, 0, 1))
        #expect(pixels[0] == SIMD4(0, 0, 1, 1))

        let virtual = try #require(try reflector.reflectedView(camera, depthMode: mode))
        let clip = virtual.camera.projectionMatrix * virtual.camera.viewMatrix * SIMD4<Float>(0, 0, -3, 1)
        let expected = clip.z / clip.w
        let readback = try await TextureTransfer.read(from: depth, commandQueue: gpu.queue)
        readback.bytes.withUnsafeBytes { bytes in
            #expect(bytes.loadUnaligned(as: Float.self) == (mode == .reversed ? 0 : 1))
            for index in mixed {
                let resolved = bytes.loadUnaligned(fromByteOffset: index * 4, as: Float.self)
                #expect(abs(resolved - expected) < 0.00001)
            }
        }

        let mip = try await TextureTransfer.read(from: color, mipLevel: color.mipmapLevelCount - 1, commandQueue: gpu.queue)
        let average = pixels.reduce(SIMD4<Float>.zero, +) / Float(pixels.count)
        #expect(isNearly(mip.bytes.withUnsafeBytes { SIMD4<Float>($0.loadUnaligned(as: SIMD4<Float16>.self)) },
                         average, tolerance: 0.001))

        // An inactive mirror still resolves clear values and builds their mips.
        let inactive = PlanarReflection(plane: try Plane3D(normal: SIMD3(0, 0, -1), through: SIMD3(0, 0, -4)), reflector: UUID())
        let next = try #require(gpu.queue.makeCommandBuffer())
        #expect(try inactive.encode(
            graph, camera: camera, color: color, depth: depth, depthMode: mode,
            clearColor: MTLClearColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1), generateMipmaps: true,
            background: background, multisample: msaa, commandBuffer: next
        ) == nil)
        try gpu.finish(next)
        #expect(try gpu.pixels(of: color).allSatisfy { $0 == SIMD4(0.25, 0.5, 0.75, 1) })
        let clear = try await TextureTransfer.read(
            from: depth, region: MTLRegionMake2D(64, 64, 1, 1), commandQueue: gpu.queue
        )
        #expect(clear.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) } == (mode == .reversed ? 0 : 1))
    }

    @Test("The back of the mirror clears every requested mip and stored depth")
    func backFaceClears() async throws {
        let color = try texture(.rgba16Float, usage: [.renderTarget, .shaderRead], mipmapped: true)
        let depth = try texture(.depth32Float, usage: [.renderTarget, .shaderRead])
        let mode = MeshDepthMode.reversed
        let plane = try Plane3D(normal: -worldPlane().normal, through: worldPoint(center))
        let reflection = PlanarReflection(plane: plane, reflector: UUID())
        let graph = SceneGraph(roots: [], camera: gpu.camera)
        let camera = try camera(mode: mode, perspective: true)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let clear = MTLClearColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1)
        #expect(try reflection.encode(
            graph, camera: camera, color: color, depth: depth, depthMode: mode,
            clearColor: clear, generateMipmaps: true, commandBuffer: command
        ) == nil)
        try gpu.finish(command)
        #expect(try gpu.pixels(of: color).allSatisfy { $0 == SIMD4(0.25, 0.5, 0.75, 1) })
        let mip = try await TextureTransfer.read(from: color, mipLevel: color.mipmapLevelCount - 1, commandQueue: gpu.queue)
        let mipRGBA = mip.bytes.withUnsafeBytes { bytes in
            SIMD4<Float>(bytes.loadUnaligned(as: SIMD4<Float16>.self))
        }
        #expect(mipRGBA == SIMD4(0.25, 0.5, 0.75, 1))
        let clearDepth = try await TextureTransfer.read(
            from: depth, region: MTLRegionMake2D(0, 0, 1, 1), commandQueue: gpu.queue
        )
        #expect(clearDepth.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) } == 0)
    }

    @Test("Planar capture rejects texture feedback and incompatible camera and targets")
    func validation() throws {
        let color = try texture(.rgba16Float, usage: [.renderTarget, .shaderRead])
        let depth = try texture(.depth32Float, usage: .renderTarget)
        let view = try #require(color.makeTextureView(pixelFormat: .rgba16Float))
        let mirror = try mirrorNode(sampling: view, mode: .forward)
        let graph = SceneGraph(roots: [SceneNode(transform: frame, children: [mirror])], camera: gpu.camera)
        let camera = try camera(mode: .forward, perspective: true)
        let reflection = PlanarReflection(plane: try worldPlane(), reflector: UUID())
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PlanarReflection.Failure.feedback) {
            try reflection.encode(graph, camera: camera, color: color, depth: depth, commandBuffer: command)
        }
        let invalidColor = try texture(.rgba16Float, usage: .renderTarget)
        #expect(throws: PlanarReflection.Failure.target) {
            try reflection.encode(graph, camera: camera, color: invalidColor, depth: depth, commandBuffer: command)
        }
        #expect(throws: PlanarReflection.Failure.projection) {
            try reflection.reflectedView(camera, depthMode: .reversed)
        }
        let scaled = try ShaderGraphCamera(
            viewMatrix: SceneTransform(scale: SIMD3(2, 1, 1)).matrix * camera.viewMatrix,
            projectionMatrix: camera.projectionMatrix, near: camera.near, far: camera.far
        )
        #expect(throws: PlanarReflection.Failure.camera) {
            try reflection.reflectedView(scaled, depthMode: .forward)
        }
        let backgroundAlias = try EnvironmentBackground(device: gpu.device, source: .flat(color))
        #expect(throws: PlanarReflection.Failure.feedback) {
            try reflection.encode(
                SceneGraph(roots: [], camera: gpu.camera), camera: camera, color: color, depth: depth,
                background: backgroundAlias, commandBuffer: command
            )
        }
        #expect(throws: PlanarReflection.Failure.sampleCount) {
            try PlanarReflection.MultisampleTargets(device: gpu.device, width: gpu.side, height: gpu.side, sampleCount: 1)
        }
        let excluded = try #require(try reflection.encode(
            graph, camera: camera, color: color, depth: depth, excluding: [mirror.id], commandBuffer: command
        ))
        #expect(excluded.drawnMeshCount == 0)
        try gpu.finish(command)
    }

    private func directionalBackground() throws -> EnvironmentBackground {
        let width = 128
        let height = 64
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(gpu.device.makeTexture(descriptor: descriptor))
        var pixels: [SIMD4<Float16>] = []
        for row in 0..<height {
            let theta = (Float(row) + 0.5) * .pi / Float(height)
            for column in 0..<width {
                let phi = ((Float(column) + 0.5) / Float(width) - 0.5) * 2 * .pi
                let direction = SIMD3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi))
                pixels.append(SIMD4(SIMD3<Float16>((direction + 1) * 0.5), 1))
            }
        }
        pixels.withUnsafeBytes {
            panorama.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: $0.baseAddress!, bytesPerRow: width * 8
            )
        }
        return try EnvironmentBackground(device: gpu.device, source: .equirectangular(panorama))
    }

    private func texture(_ format: MTLPixelFormat, usage: MTLTextureUsage, mipmapped: Bool = false) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: gpu.side, height: gpu.side, mipmapped: mipmapped
        )
        descriptor.storageMode = format == .depth32Float ? .private : .shared
        descriptor.usage = usage
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func camera(mode: MeshDepthMode, perspective: Bool) throws -> ShaderGraphCamera {
        var projection: simd_float4x4
        if perspective {
            projection = try gpu.camera.projection(aspectRatio: 1, depthMode: mode)
            projection[2].x = 0.13
            projection[2].y = -0.08
        } else {
            let sign: Float = mode == .forward ? -1 : 1
            projection = simd_float4x4(columns: (
                SIMD4(0.45, 0, 0, 0), SIMD4(0, 0.53, 0, 0), SIMD4(0, 0, sign / 50, 0),
                SIMD4(0.09, -0.06, mode == .forward ? 0 : 1, 1)
            ))
        }
        return try ShaderGraphCamera(
            viewMatrix: simd_inverse(frame.matrix), projectionMatrix: projection,
            near: perspective ? gpu.camera.near : 0, far: 50
        )
    }

    private func mirrorNode(sampling texture: any MTLTexture, mode: MeshDepthMode) throws -> SceneNode {
        let graph = try SurfaceShaderGraph { surface in
            let map = surface.texture("PlanarReflection", addressing: .clampToEdge)
            return .init(baseColor: map.samplePlanarReflection().rgb)
        }
        let material = try PhysicallyBasedMaterial(
            surfaceGraph: graph, surfaceTextures: ["PlanarReflection": texture], shadingModel: .unlit
        )
        let mirror = try gpu.unlit(
            MeshGeometry.plane(width: 3.3, height: 2.5), material: .physicallyBased(material), depthMode: mode
        )
        return SceneNode(transform: .init(translation: center, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: localNormal)), mesh: mirror)
    }

    private func marker(at position: SIMD3<Float>, toward direction: SIMD3<Float>, color: SIMD3<Float>, mode: MeshDepthMode) throws -> SceneNode {
        let mesh = try gpu.unlit(
            GeometryRenderFixture.square(0.16), material: .classic(ClassicMaterial(model: .basic, color: color)), depthMode: mode
        )
        return SceneNode(
            transform: .init(translation: position, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: direction)),
            mesh: mesh, frustumCulled: false
        )
    }

    private func planePoint(_ x: Float, _ y: Float) -> SIMD3<Float> {
        SIMD3(x, y, center.z - (localNormal.x * x + localNormal.y * y) / localNormal.z)
    }

    private func reflectedDirection(at point: SIMD3<Float>, perspective: Bool) -> SIMD3<Float> {
        if perspective {
            let reflectedOrigin = 2 * simd_dot(localNormal, center) * localNormal
            return simd_normalize(point - reflectedOrigin)
        }
        let forward = SIMD3<Float>(0, 0, -1)
        return simd_normalize(forward - 2 * simd_dot(localNormal, forward) * localNormal)
    }

    private func worldPoint(_ local: SIMD3<Float>) -> SIMD3<Float> {
        frame.translation + frame.rotation.act(local)
    }

    private func worldPlane() throws -> Plane3D {
        try Plane3D(normal: frame.rotation.act(localNormal) * 12, through: worldPoint(center))
    }

    private func worldToLocalPosition(_ position: SIMD3<Float>, of transform: SceneTransform) -> SIMD3<Float> {
        simd_inverse(transform.rotation).act(position - transform.translation)
    }

    private func pixel(_ position: SIMD3<Float>, projection: simd_float4x4) -> (Int, Int) {
        let clip = projection * SIMD4(position, 1)
        let normalized = SIMD2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5)
        return (Int(normalized.x * Float(gpu.side)), Int(normalized.y * Float(gpu.side)))
    }
}
