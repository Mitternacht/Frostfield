import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScalarVolumeRendererTests {
    private let fixture: PathTestFixture
    private let side = 9

    init() throws { fixture = try PathTestFixture() }

    @Test("Volume MIP keeps physical bounds and composites source-over into the loaded target")
    func mipAndComposition() throws {
        let renderer = try slabs()
        let camera = try orthographic()
        let destination = try target()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let background = MTLClearColor(red: 0.2, green: 0.1, blue: 0.4, alpha: 1)
        try renderer.encode(
            camera: camera, settings: .init(window: 0...10, opacity: 0.5, sampling: .nearest),
            destination: destination, clearColor: background, commandBuffer: command
        )
        // The second pass loads the first result and uses the same 0.9 maximum.
        try renderer.encode(
            camera: camera, settings: .init(window: 0...10, opacity: 0.5, sampling: .nearest),
            destination: destination, commandBuffer: command
        )
        try fixture.finish(command)
        let pixels = try fixture.readback(destination)
        #expect(isNearly(pixels[4 * side + 4], SIMD4(0.725, 0.7, 0.775, 1)))
        #expect(isNearly(pixels[0], SIMD4(0.2, 0.1, 0.4, 1)))
        #expect(pixels.filter { $0.x > 0.5 }.count == 35)

        let turned = SceneTransform(rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1)))
        let rotation = try render(renderer, camera: camera, settings: .init(window: 0...10), placement: turned)
        #expect(rotation[4 * side + 1].w == 0)
        #expect(rotation[1 * side + 4].w == 1)
        #expect(rotation.filter { $0.w == 1 }.count == 35)

        // A shear makes these two corners outside the physical slab, even
        // though its enclosing world-axis box covers them.
        let shear = try slabs(directions: "(2,0,0) (2,2,0) (0,0,-1)", origin: "(-3,-1,-3)")
        let sheared = try render(shear, camera: camera, settings: .init(window: 0...10))
        #expect(sheared[2 * side + 1].w == 0 && sheared[6 * side + 7].w == 0)
        #expect(sheared[2 * side + 7].w == 1 && sheared[6 * side + 1].w == 1)
    }

    @Test("Opaque depth clips a ray before its brightest voxel in both depth modes",
          arguments: [MeshDepthMode.forward, .reversed])
    func opaqueDepth(_ depthMode: MeshDepthMode) throws {
        let renderer = try slabs()
        let camera = try orthographic(depthMode: depthMode)
        let settings = ScalarVolumeRenderer.Settings(window: 0...10, sampling: .nearest)
        let visible = try render(renderer, camera: camera, depthMode: depthMode, settings: settings)
        #expect(isNearly(visible[4 * side + 4], SIMD4(0.9, 0.9, 0.9, 1)))

        let destination = try target()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let depth = try depthBuffer(at: 4.4, mode: depthMode, command: command)
        try renderer.encode(
            camera: camera, depthMode: depthMode, settings: settings, destination: destination,
            opaqueDepth: depth, clearColor: Self.transparent, commandBuffer: command
        )
        try fixture.finish(command)
        let occluded = try fixture.readback(destination)
        #expect(isNearly(occluded[4 * side + 4], SIMD4(0.4, 0.4, 0.4, 1)))
        #expect(occluded[0] == .zero)
    }

    @Test("Perspective clip planes and nearest or trilinear reconstruction change measured voxels")
    func cameraAndInterpolation() throws {
        let renderer = try volume(SIMD3(2, 1, 1), values: [0, 1], origin: "(-0.51,0,-3)")
        let camera = try perspective(far: 10)
        let linear = try render(renderer, camera: camera, settings: .init(sampling: .linear))
        let nearest = try render(renderer, camera: camera, settings: .init(sampling: .nearest))
        #expect(isNearly(linear[4 * side + 4], SIMD4(0.5, 0.5, 0.5, 1)))
        #expect(nearest[4 * side + 4] == SIMD4(1, 1, 1, 1))
        #expect(linear[0] == .zero)
        let clipped = try render(renderer, camera: perspective(far: 2), settings: .init())
        #expect(clipped.allSatisfy { $0 == .zero })
        let infinite = try render(renderer, camera: perspective(far: .infinity), settings: .init())
        #expect(isNearly(infinite[4 * side + 4], SIMD4(0.5, 0.5, 0.5, 1)))

        let low = try volume(SIMD3(1, 1, 1), values: [-0.5], origin: "(0,0,-3)")
        let hidden = try render(low, camera: camera, settings: .init(window: 0...1))
        #expect(hidden.allSatisfy { $0 == .zero })
        let negativeWindow = try render(low, camera: camera, settings: .init(window: -1...0))
        #expect(isNearly(negativeWindow[4 * side + 4], SIMD4(0.5, 0.5, 0.5, 1)))
        let invalid = try volume(SIMD3(1, 1, 1), values: [.nan], origin: "(0,0,-3)")
        #expect(try render(invalid, camera: camera, settings: .init(sampling: .nearest)).allSatisfy { $0 == .zero })
    }

    @Test("Isosurfaces locate the first crossing and respect opaque depth and an inside camera")
    func isosurface() throws {
        let layers: [Float] = [0, 0, 1, 1, 0]
        let values = layers.flatMap { Array(repeating: $0, count: 9) }
        let renderer = try volume(SIMD3(3, 3, 5), values: values, origin: "(-1,-1,-3)")
        let camera = try orthographic()
        let settings = ScalarVolumeRenderer.Settings(mode: .isosurface(threshold: 0.5))
        let visible = try render(renderer, camera: camera, settings: settings)
        #expect(isNearly(visible[4 * side + 4], SIMD4(0.5, 0.5, 0.5, 1)))
        #expect(visible[0] == .zero)
        let absent = try render(renderer, camera: camera, settings: .init(mode: .isosurface(threshold: 1.1)))
        #expect(absent.allSatisfy { $0 == .zero })
        let inside = try render(renderer, camera: orthographic(near: 5.2), settings: settings)
        #expect(isNearly(inside[4 * side + 4], SIMD4(0.5, 0.5, 0.5, 1)))

        let destination = try target()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let depth = try depthBuffer(at: 4.1, mode: .forward, command: command)
        try renderer.encode(
            camera: camera, settings: settings, destination: destination, opaqueDepth: depth,
            clearColor: Self.transparent, commandBuffer: command
        )
        try fixture.finish(command)
        #expect(try fixture.readback(destination).allSatisfy { $0 == .zero })
    }

    @Test("Isosurface lighting uses physical gradients and the configured color lookup table")
    func physicalGradient() throws {
        let values = (0..<5).flatMap { z in
            (0..<5).flatMap { _ in (0..<5).map { x in Float(x + z) * 0.25 } }
        }
        let palette = try ScalarColorLookupTable(stops: [
            .init(0, linearRGB: SIMD3(0, 0, 1)), .init(1, linearRGB: SIMD3(1, 0.5, 0)),
        ], intervalCount: 10)
        let renderer = try volume(
            SIMD3(5, 5, 5), values: values, directions: "(2,0,0) (0,1,0) (0,0,-1)",
            origin: "(-4,-2,-3)", palette: palette
        )
        let pixels = try render(renderer, camera: orthographic(), settings: .init(mode: .isosurface(threshold: 1)))
        let light: Float = 0.25 + 0.75 * 2 / sqrt(5)
        #expect(isNearly(pixels[4 * side + 4], SIMD4(light, light * 0.5, 0, 1), tolerance: 0.006))
    }

    @Test("Volume encoding validates settings, scene transforms, and target roles")
    func validation() throws {
        let renderer = try slabs()
        let camera = try orthographic()
        let destination = try target()
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: ScalarVolumeRenderer.Failure.settings) {
            try renderer.encode(
                camera: camera, settings: .init(window: 1...1), destination: destination, commandBuffer: command
            )
        }
        #expect(throws: ScalarVolumeRenderer.Failure.transform) {
            try renderer.encode(
                camera: camera, placement: .init(scale: .zero), destination: destination, commandBuffer: command
            )
        }
        #expect(throws: ScalarVolumeRenderer.Failure.depth) {
            try renderer.encode(camera: camera, destination: destination, opaqueDepth: destination, commandBuffer: command)
        }
        #expect(throws: ScalarVolumeRenderer.Failure.texture) { try ScalarVolumeRenderer(texture: destination) }
    }

    private static let transparent = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

    private func target() throws -> any MTLTexture {
        try fixture.output(width: side, height: side, usage: [.renderTarget, .shaderRead])
    }

    private func render(
        _ renderer: ScalarVolumeRenderer, camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward,
        settings: ScalarVolumeRenderer.Settings, placement: SceneTransform = .init()
    ) throws -> [SIMD4<Float>] {
        let destination = try target()
        let command = try #require(fixture.queue.makeCommandBuffer())
        try renderer.encode(
            camera: camera, depthMode: depthMode, placement: placement, settings: settings, destination: destination,
            clearColor: Self.transparent, commandBuffer: command
        )
        try fixture.finish(command)
        return try fixture.readback(destination)
    }

    private func slabs(
        directions: String = "(2,0,0) (0,2,0) (0,0,-1)", origin: String = "(-2,-1,-3)"
    ) throws -> ScalarVolumeRenderer {
        let layers: [Float] = [2, 4, 9, 6]
        let values = layers.flatMap { Array(repeating: $0, count: 6) }
        return try volume(SIMD3(3, 2, 4), values: values, directions: directions, origin: origin)
    }

    private func volume(
        _ size: SIMD3<Int>, values: [Float], directions: String = "(1,0,0) (0,1,0) (0,0,-1)",
        origin: String, palette: ScalarColorLookupTable? = nil
    ) throws -> ScalarVolumeRenderer {
        let header = [
            "NRRD0005", "type: float", "dimension: 3", "sizes: \(size.x) \(size.y) \(size.z)",
            "encoding: raw", "endian: little", "space directions: \(directions)", "space origin: \(origin)", "", "",
        ].joined(separator: "\n")
        let data = Data(header.utf8) + values.withUnsafeBytes { Data($0) }
        let source = try NRRDLoader.load(data)
        let colors = try palette ?? ScalarColorLookupTable(stops: [
            .init(0, linearRGB: .zero), .init(1, linearRGB: SIMD3(repeating: 1)),
        ], intervalCount: 10)
        return try ScalarVolumeRenderer(device: fixture.device, volume: source, colorMap: colors)
    }

    private func orthographic(near: Float = 1, depthMode: MeshDepthMode = .forward) throws -> ShaderGraphCamera {
        let far: Float = 10
        let zScale: Float = depthMode == .forward ? -1 / (far - near) : 1 / (far - near)
        let zOffset: Float = depthMode == .forward ? -near / (far - near) : far / (far - near)
        let projection = simd_float4x4(columns: (
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 0.25, 0, 0), SIMD4(0, 0, zScale, 0), SIMD4(0, 0, zOffset, 1)
        ))
        return try ShaderGraphCamera(viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: near, far: far)
    }

    private func perspective(far: Float) throws -> ShaderGraphCamera {
        if far.isInfinite {
            let projection = simd_float4x4(columns: (
                SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, -1, -1), SIMD4(0, 0, -1, 0)
            ))
            return try ShaderGraphCamera(
                viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: 1, far: far
            )
        }
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: far)
        return try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: camera.projection(aspectRatio: 1), near: 1, far: far
        )
    }

    private func depthBuffer(at distance: Double, mode: MeshDepthMode, command: any MTLCommandBuffer) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let pass = MTLRenderPassDescriptor()
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        let normalized = (distance - 1) / 9
        pass.depthAttachment.clearDepth = mode == .forward ? normalized : 1 - normalized
        let encoder = try #require(command.makeRenderCommandEncoder(descriptor: pass))
        encoder.endEncoding()
        return depth
    }
}
