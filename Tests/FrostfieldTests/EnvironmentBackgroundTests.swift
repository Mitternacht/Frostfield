import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct EnvironmentBackgroundTests {
    @Test("HDR sky yaw and camera rotation agree, and a scene loads the stored background")
    func composedBackground() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        var panorama = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 32 +X 64\n".utf8)
        for y in 0..<32 {
            for x in 0..<64 {
                let hot = (26...37).contains(x) && (12...19).contains(y)
                let pixel: [UInt8] = hot ? [128, 96, 64, 134] : [128, 128, 128, 125]
                panorama.append(contentsOf: pixel)
            }
        }
        let lighting = try EnvironmentLighting(
            queue: queue, radianceHDR: panorama,
            settings: .init(cubeSize: 32, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        )
        let background = try EnvironmentBackground(lighting: lighting)
        #expect(throws: EnvironmentBackground.Failure.destinationFormat) {
            try EnvironmentBackground(lighting: lighting, destinationPixelFormat: .rgba8Unorm)
        }

        let width = 32
        let height = 24
        let color = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        color.storageMode = .shared
        color.usage = .renderTarget
        let target = try #require(device.makeTexture(descriptor: color))
        let rotated = try #require(device.makeTexture(descriptor: color))
        let blurred = try #require(device.makeTexture(descriptor: color))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let buffer = try #require(queue.makeCommandBuffer())
        try background.encode(destination: target, camera: camera, commandBuffer: buffer)
        let badViewport = MTLViewport(originX: -1, originY: 6, width: 16, height: 12, znear: 0, zfar: 1)
        #expect(throws: EnvironmentBackground.Failure.viewport) {
            try background.encode(destination: target, camera: camera, viewport: badViewport, commandBuffer: buffer)
        }
        let viewport = MTLViewport(originX: 8, originY: 6, width: 16, height: 12, znear: 0, zfar: 1)
        try background.encode(
            destination: target, camera: camera, yaw: .pi / 2, viewport: viewport, commandBuffer: buffer
        )
        let pose = SceneTransform(
            translation: SIMD3(12, 34, 56), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))
        )
        try background.encode(destination: rotated, camera: camera, cameraTransform: pose, commandBuffer: buffer)
        try background.encode(
            destination: blurred, camera: camera,
            options: .init(rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)), blurriness: 1),
            commandBuffer: buffer
        )

        let triangle = try MeshGeometry(
            vertices: [SIMD3<Float>(-0.08, -0.08, -1), SIMD3(0.08, -0.08, -1), SIMD3(0, 0.08, -1)].map {
                MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
            },
            indices: [0, 1, 2]
        )
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: triangle)
        let scene = SceneGraph(
            roots: [SceneNode(mesh: renderer, materialParameters: SIMD4(0, 1, 0, 1))], camera: camera
        )
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        try scene.encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "Sky draw failed.")")

        let center = pixel(target, x: 16, y: 12)
        let visibleSky = pixel(target, x: 19, y: 12)
        let outside = pixel(target, x: 25, y: 12)
        let cameraSky = pixel(rotated, x: 16, y: 12)
        #expect(center.x == 0 && abs(center.y - 0.22) < 0.01 && center.w == 1)
        #expect(visibleSky.x > 20 && visibleSky.w == 1)
        #expect(outside.x < 0.1 && outside.w == 1)
        #expect(simd_distance(visibleSky, cameraSky) < 0.25)
        #expect(pixel(blurred, x: 16, y: 12).x < cameraSky.x * 0.5)
    }

    @Test("A flat sRGB image fits its viewport, applies a UV transform, and stores premultiplied color")
    func flatBackground() throws {
        let fixture = try PathTestFixture()
        let device = fixture.device
        let source = try texture(device, width: 2, height: 2, format: .rgba8Unorm_srgb)
        let colors: [SIMD4<UInt8>] = [
            SIMD4(255, 128, 0, 128), SIMD4(0, 255, 0, 255),
            SIMD4(0, 0, 255, 255), SIMD4(255, 255, 255, 0),
        ]
        write(source, colors, width: 2, height: 2)
        let target = try texture(device, width: 8, height: 6)
        let premultipliedCopy = try texture(device, width: 8, height: 6)
        let display = try texture(device, width: 8, height: 6, format: .bgra8Unorm_srgb)
        let nearest = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest, minification: .nearest,
            mipFilter: .notMipmapped
        )
        let color = try EnvironmentBackground(device: device, source: .color(SIMD3(0.125, 0.25, 0.5)))
        let flat = try EnvironmentBackground(device: device, source: .flat(source, sampling: nearest))
        let mirror = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(1, 0), scale: SIMD2(-1, 1))
        let mirrored = try EnvironmentBackground(
            device: device, source: .flat(source, transform: mirror, sampling: nearest),
            destinationPixelFormat: .bgra8Unorm_srgb
        )
        let feedback = try EnvironmentBackground(device: device, source: .flat(target))
        let premultiplied = try EnvironmentBackground(
            device: device, source: .flat(target, sampling: nearest, premultipliedAlpha: true)
        )
        let command = try #require(fixture.queue.makeCommandBuffer())
        let viewport = MTLViewport(originX: 2, originY: 1, width: 4, height: 4, znear: 0, zfar: 1)
        let unrelatedPose = SceneTransform(
            translation: SIMD3(10, 20, 30), rotation: simd_quatf(angle: 2, axis: SIMD3(0, 0, 1))
        )
        try color.encode(destination: target, camera: fixture.camera, options: .init(intensity: 0), commandBuffer: command)
        try flat.encode(
            destination: target, camera: fixture.camera, cameraTransform: unrelatedPose,
            options: .init(intensity: 2, blurriness: 1), viewport: viewport, commandBuffer: command
        )
        try mirrored.encode(destination: display, camera: fixture.camera, commandBuffer: command)
        try premultiplied.encode(destination: premultipliedCopy, camera: fixture.camera, commandBuffer: command)
        #expect(throws: EnvironmentBackground.Failure.feedback) {
            try feedback.encode(destination: target, camera: fixture.camera, commandBuffer: command)
        }
        #expect(throws: EnvironmentBackground.Failure.source) {
            try EnvironmentBackground(device: device, source: .equirectangular(source))
        }
        try fixture.finish(command)

        let topLeft = pixel(target, x: 2, y: 1)
        let alpha: Float = 128 / 255
        #expect(abs(topLeft.x - 2 * alpha) < 0.002 && abs(topLeft.y - 0.2167) < 0.003)
        #expect(topLeft.z == 0 && abs(topLeft.w - alpha) < 0.001)
        #expect(pixel(premultipliedCopy, x: 2, y: 1) == topLeft)
        #expect(pixel(target, x: 5, y: 1) == SIMD4(0, 2, 0, 1))
        #expect(pixel(target, x: 2, y: 4) == SIMD4(0, 0, 2, 1))
        #expect(pixel(target, x: 5, y: 4) == .zero)
        #expect(pixel(target, x: 1, y: 1) == SIMD4(0.125, 0.25, 0.5, 1))
        #expect(pixel(target, x: 5, y: 5) == SIMD4(0.125, 0.25, 0.5, 1))
        #expect(read(display, x: 0, y: 0, as: SIMD4<UInt8>.self) == SIMD4(0, 255, 0, 255))
        let mirroredAlpha = read(display, x: 7, y: 0, as: SIMD4<UInt8>.self)
        #expect(abs(Int(mirroredAlpha.z) - 188) <= 1 && mirroredAlpha.w == 128)
    }

    @Test("Raw cube and panorama backgrounds support full orientation and directional mips")
    func directionalSources() throws {
        let fixture = try PathTestFixture()
        let device = fixture.device
        let cubeDescriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: 4, mipmapped: true)
        cubeDescriptor.storageMode = .shared
        cubeDescriptor.usage = .shaderRead
        let cube = try #require(device.makeTexture(descriptor: cubeDescriptor))
        for face in 0..<6 {
            for mip in 0..<3 {
                let side = 4 >> mip
                var color = SIMD4<Float16>.zero
                color[mip] = Float16(face + 1)
                write(cube, Array(repeating: color, count: side * side), width: side, height: side, mip: mip, slice: face)
            }
        }
        let panorama = try texture(device, width: 8, height: 4, mipmapped: true)
        let panoramaWithoutMips = try texture(device, width: 8, height: 4)
        let texels: [SIMD4<Float16>] = (0..<32).map { index in
            SIMD4(Float16(index / 8 + 1), Float16(index % 8), 0, 1)
        }
        write(panorama, texels, width: 8, height: 4)
        for mip in 1..<panorama.mipmapLevelCount {
            let width = max(1, 8 >> mip)
            let height = max(1, 4 >> mip)
            let color = SIMD4<Float16>(0, 0, Float16(4 * mip - 2), 1)
            write(panorama, Array(repeating: color, count: width * height), width: width, height: height, mip: mip)
        }
        let cubeBackground = try EnvironmentBackground(device: device, source: .cube(cube))
        let panoramaBackground = try EnvironmentBackground(device: device, source: .equirectangular(panorama))
        let sharpOnly = try EnvironmentBackground(device: device, source: .equirectangular(panoramaWithoutMips))
        let targets = try (0..<6).map { _ in try texture(device, width: 9, height: 9) }
        let command = try #require(fixture.queue.makeCommandBuffer())
        let pitch = simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))
        let roll = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
        try cubeBackground.encode(destination: targets[0], camera: fixture.camera, commandBuffer: command)
        try cubeBackground.encode(
            destination: targets[1], camera: fixture.camera,
            options: .init(rotation: pitch, intensity: 2, blurriness: 0.75), commandBuffer: command
        )
        try panoramaBackground.encode(destination: targets[2], camera: fixture.camera, commandBuffer: command)
        try panoramaBackground.encode(
            destination: targets[3], camera: fixture.camera, yaw: -.pi / 2, commandBuffer: command
        )
        try panoramaBackground.encode(
            destination: targets[4], camera: fixture.camera,
            cameraTransform: .init(translation: SIMD3(1, 2, 3)), options: .init(rotation: roll), commandBuffer: command
        )
        try panoramaBackground.encode(
            destination: targets[5], camera: fixture.camera, options: .init(blurriness: 0.5), commandBuffer: command
        )
        #expect(throws: EnvironmentBackground.Failure.mipmaps) {
            try sharpOnly.encode(
                destination: targets[2], camera: fixture.camera, options: .init(blurriness: 1), commandBuffer: command
            )
        }
        #expect(throws: EnvironmentBackground.Failure.environment) {
            try cubeBackground.encode(
                destination: targets[0], camera: fixture.camera,
                options: .init(rotation: simd_quatf(real: 0, imag: .zero)), commandBuffer: command
            )
        }
        try fixture.finish(command)
        #expect(pixel(targets[0], x: 4, y: 4) == SIMD4(6, 0, 0, 1))
        #expect(pixel(targets[1], x: 4, y: 4) == SIMD4(0, 4, 4, 1))
        #expect(simd_distance(pixel(targets[2], x: 4, y: 4), SIMD4(2.5, 1.5, 0, 1)) < 0.01)
        #expect(simd_distance(pixel(targets[3], x: 4, y: 4), SIMD4(2.5, 3.5, 0, 1)) < 0.01)
        #expect(pixel(targets[2], x: 4, y: 0).x < 2.1)
        #expect(abs(pixel(targets[4], x: 4, y: 0).x - 2.5) < 0.01)
        #expect(pixel(targets[5], x: 4, y: 4) == SIMD4(0, 0, 4, 1))
    }

    @Test("Explicit sky views use asymmetric frusta and orthographic parallel rays")
    func explicitViews() throws {
        let fixture = try PathTestFixture()
        let device = fixture.device
        let panorama = try texture(device, width: 8, height: 4)
        let texels: [SIMD4<Float16>] = (0..<32).map { index in
            SIMD4(Float16(index / 8 + 1), Float16(index % 8), 0, 1)
        }
        write(panorama, texels, width: 8, height: 4)
        let sky = try EnvironmentBackground(device: device, source: .equirectangular(panorama))
        let fill = try EnvironmentBackground(device: device, source: .color(SIMD3(0.25, 0.5, 0.75)))
        let camera = fixture.camera
        let legacy = try texture(device, width: 9, height: 5)
        let explicit = try texture(device, width: 15, height: 11)
        let shifted = try texture(device, width: 15, height: 11)
        let orthographic = try texture(device, width: 9, height: 5)
        let viewport = MTLViewport(originX: 3, originY: 3, width: 9, height: 5, znear: 0, zfar: 1)
        let projection = try camera.projection(aspectRatio: 9.0 / 5.0)
        let symmetric = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: camera.near, far: camera.far
        )
        var offCenter = projection
        offCenter.columns.2 = SIMD4(projection.columns.0.x, projection.columns.1.y * 0.5, 0, -1)
        offCenter.columns.3.z = camera.near
        let reversedInfinite = try ShaderGraphCamera(
            viewMatrix: SceneTransform(translation: SIMD3(21, -7, 4)).matrix.inverse,
            projectionMatrix: offCenter, near: camera.near, far: .infinity
        )

        let orthoProjection = simd_float4x4(columns: (
            SIMD4(0.5, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 0.05, 0), SIMD4(0.25, -0.5, 1, 1)
        ))
        let pose = SceneTransform(
            translation: SIMD3(-3, 9, 14), rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))
        )
        let parallel = try ShaderGraphCamera(viewMatrix: pose.matrix.inverse, projectionMatrix: orthoProjection, near: 0, far: 20)

        let command = try #require(fixture.queue.makeCommandBuffer())
        try fill.encode(destination: shifted, view: symmetric, commandBuffer: command)
        try fill.encode(destination: explicit, view: symmetric, commandBuffer: command)
        try sky.encode(destination: legacy, camera: camera, commandBuffer: command)
        try sky.encode(destination: explicit, view: symmetric, viewport: viewport, commandBuffer: command)
        try sky.encode(destination: shifted, view: reversedInfinite, viewport: viewport, commandBuffer: command)
        try sky.encode(destination: orthographic, view: parallel, commandBuffer: command)
        var skewed = projection
        skewed.columns.1.x = 0.3
        let unsupported = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: skewed, near: camera.near, far: camera.far
        )
        #expect(throws: EnvironmentBackground.Failure.projection) {
            try sky.encode(destination: legacy, view: unsupported, commandBuffer: command)
        }
        try fixture.finish(command)

        for y in 0..<5 {
            for x in 0..<9 {
                #expect(simd_distance(pixel(legacy, x: x, y: y), pixel(explicit, x: x + 3, y: y + 3)) < 0.005)
                #expect(simd_distance(pixel(orthographic, x: x, y: y), SIMD4(2.5, 3.5, 0, 1)) < 0.01)
            }
        }
        let center = pixel(shifted, x: 7, y: 5)
        let expectedRed = 0.5 + 4 * acos(Float(1) / 3) / .pi
        #expect(simd_distance(center, SIMD4(expectedRed, 2.5, 0, 1)) < 0.01)
        #expect(pixel(shifted, x: 2, y: 5) == SIMD4(0.25, 0.5, 0.75, 1))
        #expect(pixel(shifted, x: 7, y: 8) == SIMD4(0.25, 0.5, 0.75, 1))
    }

    private func texture(
        _ device: any MTLDevice, width: Int, height: Int, format: MTLPixelFormat = .rgba16Float, mipmapped: Bool = false
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func write<T>(
        _ texture: any MTLTexture, _ pixels: [T], width: Int, height: Int, mip: Int = 0, slice: Int = 0
    ) {
        pixels.withUnsafeBytes {
            let rowBytes = width * MemoryLayout<T>.stride
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: mip, slice: slice,
                withBytes: $0.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: height * rowBytes
            )
        }
    }

    private func pixel(_ texture: any MTLTexture, x: Int, y: Int) -> SIMD4<Float> {
        SIMD4<Float>(read(texture, x: x, y: y, as: SIMD4<Float16>.self))
    }

    private func read<T: SIMD>(_ texture: any MTLTexture, x: Int, y: Int, as: T.Type) -> T where T.Scalar: AdditiveArithmetic {
        var result = T(repeating: .zero)
        withUnsafeMutablePointer(to: &result) {
            texture.getBytes(
                $0, bytesPerRow: MemoryLayout<T>.stride,
                from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0
            )
        }
        return result
    }
}
