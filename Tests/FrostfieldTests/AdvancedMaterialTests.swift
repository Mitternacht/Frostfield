import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct AdvancedMaterialTests {
    private static let side = 128
    private static let colorFormat: MTLPixelFormat = .rgba8Unorm
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    }

    @Test("Textured material preserves top-left UVs and premultiplies tinted alpha")
    func textureOrientationAndTint() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.colorFormat, width: 1, height: 4, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        // Duplicate each row so linear filtering gives a solid sample away from the seam.
        let texels: [UInt8] = [
            255, 0, 0, 128, 255, 0, 0, 128,
            0, 0, 255, 255, 0, 0, 255, 255,
        ]
        try texels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, 1, 4), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: 4
            )
        }
        let renderer = try makeRenderer(halfExtent: 1, material: .textured(texture))
        let scene = MeshScene(
            camera: camera,
            transform: SceneTransform(translation: SIMD3<Float>(0, 0, -2)),
            materialParameters: SIMD4<Float>(0.5, 1, 0.75, 0.5)
        )

        let pixels = try render { pass, buffer in
            try renderer.encode(scene, pass: pass, commandBuffer: buffer)
        }

        expectPixel(pixels, x: 64, y: 48, rgba: SIMD4<Int>(32, 0, 0, 64))
        expectPixel(pixels, x: 64, y: 80, rgba: SIMD4<Int>(0, 0, 96, 128))
        expectPixel(pixels, x: 16, y: 64, rgba: .zero)
    }

    @Test("A directional shadow darkens a receiver without covering it in the camera view", arguments: [false, true])
    func shadowCasterFlag(physicallyBased: Bool) throws {
        let material: MeshMaterial = physicallyBased
            ? .physicallyBased(try PhysicallyBasedMaterial(roughness: 0.6, receivesShadows: true))
            : .shadowedLambert
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3<Float>(0, 0, -3)),
            mesh: try makeRenderer(halfExtent: 1.6, material: material),
            materialParameters: SIMD4<Float>(0.75, 0.5, 0.25, 1),
            castsShadow: false
        )
        // The light ray from the receiver center reaches x = 0.75 at z = -2.
        // The blocker is at that point but projects to the camera's right.
        var blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3<Float>(0.75, 0, -2)),
            mesh: try makeRenderer(halfExtent: 0.25, material: .unlit),
            castsShadow: false
        )
        var graph = SceneGraph(roots: [receiver, blocker], camera: camera)
        let shadow = try DirectionalShadowMap(
            queue: queue, resolution: 256, directionToLight: SIMD3<Float>(0.6, 0, 0.8),
            center: SIMD3<Float>(0, 0, -3), halfExtent: 3
        )

        let lit = try render { pass, buffer in
            try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        }
        blocker.castsShadow = true
        graph.roots[1] = blocker
        let shaded = try render { pass, buffer in
            try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        }

        if physicallyBased {
            let center = (64 * Self.side + 64) * 4
            let clear = (64 * Self.side + 45) * 4
            #expect(Int(lit[center]) > Int(shaded[center]) + 10)
            #expect(shaded[center] > 0)
            #expect(shaded[center + 3] == 255)
            #expect(lit[clear..<clear + 4] == shaded[clear..<clear + 4])
        } else {
            expectPixel(lit, x: 64, y: 64, rgba: SIMD4<Int>(159, 106, 53, 255))
            expectPixel(shaded, x: 64, y: 64, rgba: SIMD4<Int>(31, 20, 10, 255))
            expectPixel(shaded, x: 45, y: 64, rgba: SIMD4<Int>(159, 106, 53, 255))
        }

        let otherQueue = try #require(device.makeCommandQueue())
        #expect(throws: DirectionalShadowMap.Failure.wrongQueue) {
            try render(on: otherQueue) { pass, buffer in
                try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
            }
        }
    }

    @Test("Environment radiance lights both diffuse and metallic PBR without a direct light", arguments: [Float(0), 1])
    func imageBasedLighting(metallic: Float) throws {
        let width = 32
        let height = 16
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(device.makeTexture(descriptor: descriptor))
        let radiance = SIMD4<Float16>(0.6, 0.24, 0.08, 1)
        let texels = Array(repeating: radiance, count: width * height)
        try texels.withUnsafeBytes { bytes in
            panorama.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * MemoryLayout<SIMD4<Float16>>.stride
            )
        }
        let lighting = try EnvironmentLighting(
            queue: queue, equirectangular: panorama,
            settings: .init(cubeSize: 16, diffuseSize: 8, brdfSize: 16, sampleCount: 32)
        )
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting, yaw: .pi / 4)
        let material = try PhysicallyBasedMaterial(
            metallic: metallic, roughness: 0.35, lightIntensity: 0, ambientIntensity: 0, environment: environment
        )
        let mesh = try makeRenderer(halfExtent: 1.2, material: .physicallyBased(material))
        let scene = MeshScene(camera: camera, transform: SceneTransform(translation: SIMD3(0, 0, -2)))
        let pixels = try render { pass, buffer in
            try mesh.encode(scene, pass: pass, commandBuffer: buffer)
        }
        let center = (64 * Self.side + 64) * 4
        #expect(pixels[center] > 105)
        #expect(pixels[center + 1] > 38)
        #expect(pixels[center + 2] > 10)
        #expect(pixels[center] > pixels[center + 1])
        #expect(pixels[center + 1] > pixels[center + 2])
        #expect(pixels[center + 3] == 255)
    }

    @Test("A finite angular light produces a wider shadow transition")
    func contactHardeningShadow() throws {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try makeRenderer(halfExtent: 1.6, material: .shadowedLambert), castsShadow: false
        )
        let blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(0.75, 0, -2)),
            mesh: try makeRenderer(halfExtent: 0.25, material: .unlit)
        )
        let graph = SceneGraph(roots: [receiver, blocker], camera: camera)
        func transition(angularRadius: Float) throws -> Int {
            let map = try DirectionalShadowMap(
                queue: queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
                center: SIMD3(0, 0, -3), halfExtent: 3, lightAngularRadius: angularRadius
            )
            let pixels = try render { pass, buffer in
                try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: map)
            }
            return (48..<76).filter { x in
                (55..<196).contains(pixels[(64 * Self.side + x) * 4])
            }.count
        }
        let sharp = try transition(angularRadius: 0)
        let soft = try transition(angularRadius: 0.23)
        #expect(soft > sharp)
    }

    @Test("An external reversed-depth view keeps the nearer mesh and respects its viewport")
    func externalReversedDepth() throws {
        let mesh = try makeRenderer(halfExtent: 1, material: .unlit, depthMode: .reversed)
        let graph = SceneGraph(roots: [
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, -2)),
                mesh: mesh, materialParameters: SIMD4(1, 0, 0, 1)
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, -3)),
                mesh: mesh, materialParameters: SIMD4(0, 0, 1, 1)
            ),
        ], camera: camera)
        var reverseDepth = matrix_identity_float4x4
        reverseDepth.columns.2.z = -1
        reverseDepth.columns.3.z = 1
        let clip = reverseDepth * (try camera.projection(aspectRatio: 1))
        let viewport = MTLViewport(originX: 0, originY: 0, width: 64, height: 64, znear: 0, zfar: 1)

        let pixels = try render { pass, buffer in
            pass.depthAttachment.clearDepth = 0
            #expect(throws: MeshRenderer.Failure.depthMode) {
                try graph.encode(pass: pass, commandBuffer: buffer)
            }
            try graph.encode(
                pass: pass, commandBuffer: buffer, clipFromWorld: clip, cameraPosition: .zero,
                viewport: viewport, depthMode: .reversed
            )
        }

        expectPixel(pixels, x: 32, y: 32, rgba: SIMD4(56, 0, 0, 255))
        expectPixel(pixels, x: 80, y: 80, rgba: .zero)
    }

    @Test("A grazing shadow receiver stays lit when its own surface casts")
    func grazingReceiver() throws {
        let mesh = try makeRenderer(halfExtent: 1.6, material: .shadowedLambert)
        var plane = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: mesh, castsShadow: false
        )
        var graph = SceneGraph(roots: [plane], camera: camera)
        let shadow = try DirectionalShadowMap(
            queue: queue, resolution: 128, directionToLight: SIMD3(1, 0, 0.4),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let lit = try render { pass, buffer in
            try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        }
        plane.castsShadow = true
        graph.roots = [plane]
        let selfCasting = try render { pass, buffer in
            try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadow)
        }

        #expect(lit == selfCasting)
    }

    @Test("PBR opacity modes apply cutout and ordered premultiplied blending")
    func materialCoverage() throws {
        func material(_ color: SIMD3<Float>, _ mode: PhysicallyBasedMaterial.AlphaMode) throws -> MeshMaterial {
            .physicallyBased(try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, emissiveFactor: color, alphaMode: mode
            ))
        }
        func node(_ mesh: MeshRenderer, depth: Float, opacity: Float) -> SceneNode {
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, depth)),
                mesh: mesh, materialParameters: SIMD4(1, 1, 1, opacity)
            )
        }
        let opaque = try makeRenderer(halfExtent: 1, material: material(SIMD3(0, 0, 1), .opaque))
        let cutout = try makeRenderer(halfExtent: 1, material: material(SIMD3(0, 1, 0), .mask(cutoff: 0.5)))
        let red = try makeRenderer(halfExtent: 1, material: material(SIMD3(1, 0, 0), .blend))
        let green = try makeRenderer(halfExtent: 1, material: material(SIMD3(0, 1, 0), .blend))
        let graph = SceneGraph(roots: [
            node(red, depth: -2, opacity: 0.5),
            node(cutout, depth: -1.5, opacity: 0.2),
            node(green, depth: -2.5, opacity: 0.5),
            node(opaque, depth: -3, opacity: 0),
        ], camera: camera)
        let pixels = try render { pass, buffer in
            try graph.encode(pass: pass, commandBuffer: buffer)
        }
        expectPixel(pixels, x: 64, y: 64, rgba: SIMD4(128, 64, 64, 255))
    }

    private func makeRenderer(
        halfExtent: Float, material: MeshMaterial, depthMode: MeshDepthMode = .forward
    ) throws -> MeshRenderer {
        let normal = SIMD3<Float>(0, 0, 1)
        let h = halfExtent
        let geometry = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3<Float>(-h, -h, 0), normal: normal, textureCoordinate: SIMD2<Float>(0, 1)),
            MeshVertex(position: SIMD3<Float>(h, -h, 0), normal: normal, textureCoordinate: SIMD2<Float>(1, 1)),
            MeshVertex(position: SIMD3<Float>(h, h, 0), normal: normal, textureCoordinate: SIMD2<Float>(1, 0)),
            MeshVertex(position: SIMD3<Float>(-h, h, 0), normal: normal, textureCoordinate: SIMD2<Float>(0, 0)),
        ], indices: [0, 1, 2, 0, 2, 3])
        return try MeshRenderer(
            device: device, colorPixelFormat: Self.colorFormat, geometry: geometry, material: material, depthMode: depthMode
        )
    }

    private func render(
        on alternateQueue: (any MTLCommandQueue)? = nil,
        _ encode: (MTLRenderPassDescriptor, any MTLCommandBuffer) throws -> Void
    ) throws -> [UInt8] {
        let side = Self.side
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.colorFormat, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1

        let buffer = try #require((alternateQueue ?? queue).makeCommandBuffer())
        try encode(pass, buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        let failure = buffer.error?.localizedDescription ?? "Render did not complete."
        try #require(buffer.status == .completed, "\(failure)")

        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        try pixels.withUnsafeMutableBytes { bytes in
            color.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: side * 4,
                from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0
            )
        }
        return pixels
    }

    private func expectPixel(_ pixels: [UInt8], x: Int, y: Int, rgba: SIMD4<Int>) {
        let offset = (y * Self.side + x) * 4
        for channel in 0..<4 {
            #expect(abs(Int(pixels[offset + channel]) - rgba[channel]) <= 2)
        }
    }
}
