import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PlanarReflectionMaterialTests {
    private let gpu: GeometryRenderFixture
    private let mirrorCenter = SIMD3<Float>(0, 0, -4)
    private let mirrorNormal = simd_normalize(SIMD3<Float>(0.24, 0.11, 1))
    private let nearEdge = 48
    private let farEdge = 80

    init() throws { gpu = try GeometryRenderFixture(side: 128) }

    @Test("Tilted planar capture contracts near blur, widens far blur, and supports material roughness and distortion",
          arguments: [MeshDepthMode.forward, .reversed])
    func tiltedReflection(mode: MeshDepthMode) throws {
        let sharp = try texture(.rgba16Float)
        let depth = try texture(.depth32Float)
        let blurred = try texture(.rgba16Float)
        let proximity = try texture(.r16Float)
        let reflection = PlanarReflection(plane: try Plane3D(normal: mirrorNormal, through: mirrorCenter), reflector: UUID())
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: gpu.camera.projection(aspectRatio: 1, depthMode: mode),
            near: gpu.camera.near, far: gpu.camera.far
        )
        let reflected = try #require(try reflection.reflectedView(camera, depthMode: mode))
        let worldFromClip = simd_inverse(reflected.camera.projectionMatrix * reflected.camera.viewMatrix)
        func world(x: Int, y: Int, forwardDepth: Float) -> SIMD3<Float> {
            let px = 2 * Float(x) / Float(gpu.side) - 1
            let py = 1 - 2 * Float(y) / Float(gpu.side)
            let p = worldFromClip * SIMD4(px, py, mode == .reversed ? 1 - forwardDepth : forwardDepth, 1)
            return SIMD3(p.x, p.y, p.z) / p.w
        }
        let row = gpu.side / 2
        let nearPoint = reflected.camera.viewMatrix * SIMD4(world(x: nearEdge, y: row, forwardDepth: 0.10), 1)
        let farPoint = reflected.camera.viewMatrix * SIMD4(world(x: farEdge, y: row, forwardDepth: 0.65), 1)
        #expect(-farPoint.z > -nearPoint.z + 4)

        // Place real geometry on opposite sides of the reflected viewport. Each
        // patch has a black/white seam and a different oblique capture depth.
        func patch(_ x: ClosedRange<Int>, _ depth: Float, white: Bool) throws -> SceneNode {
            let points = [
                world(x: x.lowerBound, y: 32, forwardDepth: depth), world(x: x.upperBound, y: 32, forwardDepth: depth),
                world(x: x.upperBound, y: 96, forwardDepth: depth), world(x: x.lowerBound, y: 96, forwardDepth: depth),
            ]
            let mesh = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float,
                geometry: MeshGeometry(vertices: points.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) },
                                       indices: [0, 1, 2, 0, 2, 3]),
                material: .classic(ClassicMaterial(model: .basic, color: SIMD3(repeating: white ? 1 : 0))),
                depthMode: mode, renderState: .init(side: .double)
            )
            return SceneNode(mesh: mesh)
        }
        let markers = try [
            patch(29...nearEdge, 0.10, white: false), patch(nearEdge...63, 0.10, white: true),
            patch(66...farEdge, 0.65, white: false), patch(farEdge...99, 0.65, white: true),
        ]
        let command = try #require(gpu.queue.makeCommandBuffer())
        try reflection.encode(
            SceneGraph(roots: markers, camera: gpu.camera), camera: camera, color: sharp, depth: depth,
            depthMode: mode, commandBuffer: command
        )
        let blur = try PlanarReflectionBlur(
            device: gpu.device, settings: .init(radius: SIMD2(10, 2), depthThresholds: SIMD2(0.2, 0.8), depthScale: 1)
        )
        try blur.encode(
            source: sharp, depth: depth, destination: blurred, depthFactor: proximity, depthMode: mode, commandBuffer: command
        )
        let distortion = try constantDistortion()
        let sharpMirror = try mirror(sharp, mix: .init(mirror: 1), roughness: 0, camera: camera, mode: mode, command: command)
        let smoothMirror = try mirror(sharp, blurred: blurred, mix: .init(blurMix: 1, mirror: 1), roughness: 0,
                                      camera: camera, mode: mode, command: command)
        let roughMirror = try mirror(sharp, blurred: blurred, mix: .init(blurMix: 1, mirror: 1), roughness: 1,
                                     camera: camera, mode: mode, command: command)
        let depthMirror = try mirror(sharp, blurred: blurred, depthFactor: proximity, mix: .init(blurMix: 1, mirror: 1),
                                     roughness: 1, camera: camera, mode: mode, command: command)
        let distortedMirror = try mirror(sharp, distortionMap: distortion, mix: .init(strength: 0.5, mirror: 1, distortion: 0.22),
                                         roughness: 0, camera: camera, mode: mode, command: command)
        let normalMirror = try automaticNormalMirror(markers, camera: camera, mode: mode, command: command)
        try gpu.finish(command)

        let raw = try gpu.pixels(of: sharp)
        let filtered = try gpu.pixels(of: blurred)
        func red(_ pixels: [SIMD4<Float>], _ x: Int) -> Float { gpu.pixel(pixels, x: x, y: row).x }
        let nearBlack = nearEdge - 3
        let farBlack = farEdge - 3
        #expect(red(raw, nearBlack) == 0 && red(raw, nearEdge + 3) == 1)
        #expect(red(raw, farBlack) == 0 && red(raw, farEdge + 3) == 1)
        #expect(red(filtered, nearBlack) < 0.005 && red(filtered, nearEdge + 3) > 0.995)
        let spread = red(filtered, farBlack)
        #expect(spread > 0.15 && spread < 0.5)
        #expect(abs(factor(proximity, x: nearBlack, y: row) - 1) < 0.001)
        let farFactor = factor(proximity, x: farBlack, y: row)
        #expect(abs(farFactor - 0.15625) < 0.002)

        let nearMirrorX = gpu.side - 1 - nearBlack
        let farMirrorX = gpu.side - 1 - farBlack
        #expect(try red(gpu.pixels(of: sharpMirror), nearMirrorX) < 0.005)
        #expect(try red(gpu.pixels(of: smoothMirror), farMirrorX) < 0.005)
        #expect(try abs(red(gpu.pixels(of: roughMirror), farMirrorX) - spread) < 0.005)
        #expect(try abs(red(gpu.pixels(of: depthMirror), farMirrorX) - spread * (farFactor + 0.25)) < 0.006)
        #expect(try abs(red(gpu.pixels(of: distortedMirror), nearMirrorX) - 0.5) < 0.005)
        #expect(try abs(red(gpu.pixels(of: normalMirror), nearMirrorX) - 0.5) < 0.005)
    }

    @Test("Planar blur validates missing depth, output formats and texture feedback before encoding")
    func validation() throws {
        let color = try texture(.rgba16Float)
        let output = try texture(.rgba16Float)
        let depth = try texture(.depth32Float)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let blur = try PlanarReflectionBlur(device: gpu.device, settings: .init(radius: SIMD2(repeating: 2), depthScale: 1))
        #expect(throws: PostProcessEffectError.missingDepth) {
            try blur.encode(source: color, destination: output, commandBuffer: command)
        }
        #expect(throws: PlanarReflectionBlur.Failure.depthFactor) {
            try blur.encode(source: color, depth: depth, destination: output, depthFactor: output, commandBuffer: command)
        }
        let alias = try #require(color.makeTextureView(pixelFormat: .rgba16Float))
        #expect(throws: PlanarReflectionBlur.Failure.feedback) {
            try blur.encode(source: color, depth: depth, destination: alias, commandBuffer: command)
        }
        blur.settings.depthThresholds = SIMD2(0.7, 0.6)
        #expect(throws: PlanarReflectionBlur.Failure.settings) {
            try blur.encode(source: color, depth: depth, destination: output, commandBuffer: command)
        }
        #expect(throws: PlanarReflectionMix.Failure.settings) { try PlanarReflectionMix(strength: .infinity) }
    }

    private func automaticNormalMirror(
        _ markers: [SceneNode], camera: ShaderGraphCamera, mode: MeshDepthMode, command: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let normal = try #require(gpu.device.makeTexture(descriptor: descriptor))
        var sample = SIMD4<UInt8>(255, 128, 255, 255)
        normal.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &sample, bytesPerRow: 4)
        let pbr = try PhysicallyBasedMaterial(
            roughness: 0, normalTexture: .init(texture: normal), shadingModel: .unlit
        )
        let mesh = try gpu.unlit(MeshGeometry.plane(width: 4.6, height: 4.4), material: .physicallyBased(pbr), depthMode: mode)
        let node = SceneNode(transform: .init(
            translation: mirrorCenter, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: mirrorNormal)
        ), mesh: mesh)
        let appearance = try PlanarReflectorSet.Appearance(
            blur: .init(depthThresholds: SIMD2(0.2, 0.8), depthScale: 1), mix: .init(strength: 0.5, mirror: 1)
        )
        let owner = try PlanarReflectorSet(queue: gpu.queue, width: gpu.side, height: gpu.side, mirrors: [
            .init(nodeID: node.id, localPlane: try Plane3D(normal: SIMD3(0, 0, 1), constant: 0), appearance: appearance),
        ])
        let frame = try owner.encode(
            SceneGraph(roots: markers + [node], camera: gpu.camera), camera: camera, depthMode: mode, commandBuffer: command
        )
        let mirror = try #require(frame.graph.node(withID: node.id))
        let output = try gpu.makeTarget()
        try SceneGraph(roots: [mirror], camera: gpu.camera).encode(
            pass: gpu.pass(color: output.color, depth: output.depth, depthMode: mode), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix * camera.viewMatrix, cameraPosition: .zero, depthMode: mode, graphCamera: camera
        )
        return output.color
    }

    private func mirror(
        _ sharp: any MTLTexture, blurred: (any MTLTexture)? = nil, depthFactor: (any MTLTexture)? = nil,
        distortionMap: (any MTLTexture)? = nil, mix: PlanarReflectionMix, roughness: Float,
        camera: ShaderGraphCamera, mode: MeshDepthMode, command: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        let surface = try SurfaceShaderGraph { s in
            let color = s.texture("sharp", addressing: .clampToEdge)
            let soft = blurred.map { _ in s.texture("blurred", addressing: .clampToEdge) }
            let factor = depthFactor.map { _ in s.texture("depth", content: .data, addressing: .clampToEdge, filtering: .nearest) }
            let distortion = distortionMap.map { _ in s.texture("distortion", content: .data, addressing: .clampToEdge) }
            return .init(baseColor: s.baseColor * mix.multiplier(
                sharp: color, blurred: soft, depthFactor: factor, distortionMap: distortion
            ))
        }
        var textures = ["sharp": sharp]
        textures["blurred"] = blurred
        textures["depth"] = depthFactor
        textures["distortion"] = distortionMap
        let material = try PhysicallyBasedMaterial(
            roughness: roughness, surfaceGraph: surface, surfaceTextures: textures, shadingModel: .unlit
        )
        let mesh = try gpu.unlit(MeshGeometry.plane(width: 4.6, height: 4.4), material: .physicallyBased(material), depthMode: mode)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(
            translation: mirrorCenter, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: mirrorNormal)
        ), mesh: mesh)], camera: gpu.camera)
        let output = try gpu.makeTarget()
        try graph.encode(
            pass: gpu.pass(color: output.color, depth: output.depth, depthMode: mode), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix * camera.viewMatrix, cameraPosition: .zero,
            depthMode: mode, graphCamera: camera
        )
        return output.color
    }

    private func texture(_ format: MTLPixelFormat) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = format == .depth32Float ? .private : .shared
        descriptor.usage = [.shaderRead, .renderTarget]
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func factor(_ texture: any MTLTexture, x: Int, y: Int) -> Float {
        var value: Float16 = 0
        texture.getBytes(&value, bytesPerRow: 2, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        return Float(value)
    }

    private func constantDistortion() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let map = try #require(gpu.device.makeTexture(descriptor: descriptor))
        var red: UInt8 = 255
        map.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &red, bytesPerRow: 1)
        return map
    }
}
