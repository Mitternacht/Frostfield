import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct AODenoiseTests {
    private let gpu: PostProcessFixture
    private let width = 64
    private let height = 28

    init() throws { gpu = try PostProcessFixture() }

    @Test("Denoise reduces AO noise and preserves depth, clear and alpha boundaries",
          arguments: [MeshDepthMode.forward, .reversed], [true, false])
    func depthGuided(mode: MeshDepthMode, perspective: Bool) throws {
        let camera = try camera(mode: mode, perspective: perspective)
        let clear: Float = mode == .reversed ? 0 : 1
        let nearDepth = depth(distance: 2, camera: camera)
        let farDepth = depth(distance: 6, camera: camera)
        let depthTexture = try gpu.depth(width: width, height: height) { x, y in
            if y < 4 { return clear }
            return x < width / 2 ? nearDepth : farDepth
        }
        let source = try gpu.color(width: width, height: height) { x, y in
            if y < 4 { return SIMD4(0.1, 0.7, 0.9, 1) }
            if x == 16 && y == 12 { return .zero }
            let alpha: Float = x % 3 == 0 ? 0.5 : 1
            let noise: Float = (x + y).isMultiple(of: 2) ? 0.13 : -0.13
            let ambientOcclusion: Float = x < width / 2 ? 0.25 : 0.75
            return SIMD4(SIMD3(repeating: (ambientOcclusion + noise) * alpha), alpha)
        }
        let original = try gpu.read(source)
        let output = try gpu.color(width: width, height: height)
        let graph = try PostProcessGraph.denoise(camera: camera, depthMode: mode)
        let restored = try JSONDecoder().decode(PostProcessGraph.self, from: JSONEncoder().encode(graph))
        #expect(restored == graph && restored.readsDepth)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(restored)])
        let uniforms = try #require(try chain.uniforms(at: 0))
        try uniforms.set("DepthPhi", to: 0.5)
        try gpu.run { try chain.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }
        let denoised = try gpu.read(output)
        #expect(denoised.prefix(width * 4).elementsEqual(original.prefix(width * 4)))
        #expect(denoised[12 * width + 16] == .zero)
        #expect(zip(denoised, original).allSatisfy { $0.w == $1.w })

        var before: Float = 0
        var after: Float = 0
        for y in 8..<height - 4 {
            for x in 4..<width - 4 {
                let index = y * width + x
                let pixel = denoised[index]
                guard pixel.w > 0 else { continue }
                let target: Float = x < width / 2 ? 0.25 : 0.75
                before += abs(original[index].x / original[index].w - target)
                after += abs(pixel.x / pixel.w - target)
                if x == width / 2 - 1 || x == width / 2 {
                    #expect(abs(pixel.x / pixel.w - target) < 0.12)
                }
            }
        }
        #expect(after < before * 0.45)

        try uniforms.set("SampleIndex", to: Float(2))
        try gpu.run { try chain.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }
        #expect(try gpu.read(output) != denoised)
        try uniforms.set("Radius", to: Float(0))
        try gpu.run { try chain.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }
        #expect(try gpu.read(output) == original)
    }

    @Test("Signed normal textures and luminance guidance stop taps across different material values")
    func normalsAndLuminance() throws {
        let mode = MeshDepthMode.forward
        let camera = try camera(mode: mode, perspective: true)
        let flatDepth = depth(distance: 2, camera: camera)
        let depthTexture = try gpu.depth(width: width, height: height) { _, _ in flatDepth }
        let source = try gpu.color(width: width, height: height) { x, y in
            let value: Float = x < width / 2 ? 0.15 : 0.85
            if x == 12 && y == 12 { return SIMD4(0.45, 0.45, 0.45, 1) }
            return SIMD4(SIMD3(repeating: value), 1)
        }
        let output = try gpu.color(width: width, height: height)
        let normals = try gpu.color(width: width, height: height) { x, y in
            if x == 12 && y == 12 { return .zero }
            return x < width / 2 ? SIMD4(0, 0, 1, 1) : SIMD4(-1, 0, 0, 1)
        }
        let graph = try PostProcessGraph.denoise(camera: camera, normalTextureName: "ViewNormals")
        let pass = try PostProcessPass(graph: graph, device: gpu.device, textures: ["ViewNormals": normals])
        let original = try gpu.read(source)
        try gpu.run { try pass.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }
        let protected = try gpu.read(output)
        #expect(protected == original)

        let plain = try PostProcessPass(graph: .denoise(camera: camera), device: gpu.device)
        try gpu.run { try plain.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }
        let blurred = try gpu.read(output)
        let x = width / 2 - 1
        #expect((6..<height - 6).contains { blurred[$0 * width + x].x > 0.25 })
        try plain.uniforms.set("LumaPhi", to: 0.05)
        try gpu.run { try plain.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }
        #expect(try gpu.read(output) == original)
    }

    @Test("One valid neighbor matches r187 luminance, depth and normal weights")
    func singleTap() throws {
        let camera = try camera(mode: .reversed, perspective: false)
        let centerDepth = depth(distance: 2, camera: camera)
        let neighborDepth = depth(distance: 2.25, camera: camera)
        let depthTexture = try gpu.depth(width: width, height: height) { x, y in
            guard y == 0 else { return 0 }
            if x == 0 { return centerDepth }
            return x == 1 ? neighborDepth : 0
        }
        let source = try gpu.color(width: width, height: height) { x, _ in
            x == 0 ? SIMD4(0.125, 0.125, 0.125, 0.5) : SIMD4(0.75, 0.75, 0.75, 1)
        }
        let normals = try gpu.color(width: width, height: height) { x, _ in
            x == 0 ? SIMD4(0, 0, 1, 0) : SIMD4(0, 0.75, 1, 0)
        }
        let graph = try PostProcessGraph.denoise(camera: camera, depthMode: .reversed, normalTextureName: "Normals")
        let pass = try PostProcessPass(graph: graph, device: gpu.device, textures: ["Normals": normals])
        let output = try gpu.color(width: width, height: height)
        try gpu.run { try pass.encode(source: source, depth: depthTexture, destination: output, commandBuffer: $0) }

        // Phase zero at pixel (0, 0) points the first tap to (1, 0).
        let lumaWeight: Float = 1 - 0.5 / 5
        let depthWeight: Float = 1 - 0.25 / 5
        let normalWeight = pow(Float(0.8), 5)
        let weight = lumaWeight * depthWeight * normalWeight
        let expected = (0.25 + 0.75 * weight) / (1 + weight) * 0.5
        let pixel = try gpu.read(output)[0]
        #expect(PostProcessFixture.close(pixel, SIMD4(SIMD3(repeating: expected), 0.5), tolerance: 0.0005))
    }

    @Test("AO denoise validates stage, bindings, aliased destination and image sizes")
    func validation() throws {
        let camera = try camera(mode: .forward, perspective: true)
        let graph = try PostProcessGraph.denoise(camera: camera, normalTextureName: "Normals")
        let small = try gpu.color(width: 8, height: 8)
        let source = try gpu.color(width: width, height: height)
        let depthTexture = try gpu.depth(width: width, height: height) { _, _ in 0.5 }
        let output = try gpu.color(width: width, height: height)
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessPass.Failure.missingTexture("Normals")) {
            try PostProcessPass(graph: graph, device: gpu.device)
        }
        let pass = try PostProcessPass(graph: graph, device: gpu.device, textures: ["Normals": small])
        #expect(throws: PostProcessPass.Failure.sizeMismatch) {
            try pass.encode(source: source, depth: depthTexture, destination: output, commandBuffer: command)
        }
        let plain = try PostProcessPass(graph: .denoise(camera: camera), device: gpu.device)
        #expect(throws: PostProcessPass.Failure.destination) {
            try plain.encode(source: source, depth: depthTexture, destination: source, commandBuffer: command)
        }
        #expect(throws: PostProcessPass.Failure.missingDepth) {
            try plain.encode(source: source, destination: output, commandBuffer: command)
        }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(graph, textures: ["Normals": output])])
        #expect(throws: PostProcessChain.Failure.scaledDepth(stage: 0)) {
            try chain.setResolutionScale(0.5, at: 0)
        }
        let sourceAlias = try #require(output.makeTextureView(pixelFormat: output.pixelFormat))
        #expect(throws: PostProcessPass.Failure.destination) {
            try plain.encode(source: output, depth: depthTexture, destination: sourceAlias, commandBuffer: command)
        }
        #expect(throws: MaterialGraph.Failure.self) {
            try SurfaceShaderGraph { p in
                let denoised = try p.denoise(
                    at: p.uv0, inverseProjection: p.inverseProjection(of: camera)
                )
                return .init(baseColor: denoised.rgb)
            }
        }

        let foreign = ShaderGraphBuilder()
        let normals = foreign.texture("Foreign", content: .data)
        #expect(throws: MaterialGraph.Failure.invalidDeclaration("Denoise normals belong to a different graph builder.")) {
            try PostProcessGraph { p in
                try p.denoise(at: p.screenUV, inverseProjection: p.inverseProjection(of: camera), normalTexture: normals)
            }
        }
    }

    private func camera(mode: MeshDepthMode, perspective: Bool) throws -> ShaderGraphCamera {
        let source = GLTFScene.Camera(sourceIndex: 0, name: nil, projection: perspective
            ? .perspective(verticalFieldOfView: .pi / 3, aspectRatio: nil, near: 0.1, far: 12)
            : .orthographic(halfWidth: 2.5, halfHeight: 1.5, near: 0.1, far: 12)
        )
        return try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4,
            projectionMatrix: source.metalProjection(viewportAspectRatio: Float(width) / Float(height), depthMode: mode),
            near: 0.1, far: 12
        )
    }

    private func depth(distance: Float, camera: ShaderGraphCamera) -> Float {
        let clip = camera.projectionMatrix * SIMD4<Float>(0, 0, -distance, 1)
        return clip.z / clip.w
    }
}
