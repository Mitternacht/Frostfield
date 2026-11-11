import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessMaskTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("A rendered silhouette preserves the prior frame outside each effect, with inverse and clear masks",
          arguments: [false, true])
    func sceneMask(inverse: Bool) throws {
        let width = 8, height = 4
        let source = try gpu.color(width: width, height: height) { x, y in
            SIMD4(Float(x + 2) / 4, Float(y + 1) / 8, 0.375, x.isMultiple(of: 2) ? 0.5 : 0.75)
        }
        let firstSave = try gpu.color(width: width, height: height)
        let secondSave = try gpu.color(width: width, height: height)
        let destination = try gpu.color(width: width, height: height)
        let gain = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(color.rgb * 2, color.a)
        }
        let shift = try PostProcessGraph { p in
            p.sceneColor(at: p.screenUV - p.texelSize * SIMD2<Float>(1, 0))
        }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [
            .graph(gain), .save(to: firstSave), .graph(shift), .save(to: secondSave),
        ])

        let white = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float,
            geometry: .plane(width: 1, height: 2), material: .textured(white)
        )
        let near: Float = 0.1, far: Float = 10
        var projection = matrix_identity_float4x4
        projection.columns.2.z = -1 / (far - near)
        projection.columns.3.z = -near / (far - near)
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: near, far: far
        )
        let scene = SceneGraph(roots: [SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, materialParameters: SIMD4(repeating: 1)
        )], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: near, far: far))
        let scenePass = PostProcessScenePass(queue: gpu.queue)

        try gpu.run { buffer in
            let frame = try scenePass.encode(scene, width: width, height: height, commandBuffer: buffer, camera: camera)
            #expect(frame.statistics.submittedTriangleCount == 2)
            try chain.setMask(.init(texture: frame.color, channel: .alpha, inverse: inverse), stages: 0..<4)
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }

        let original = try gpu.read(source)
        let doubled = original.map { $0 * SIMD4(2, 2, 2, 1) }
        func selected(_ index: Int) -> Bool { (2..<6).contains(index % width) != inverse }
        func previousX(_ index: Int) -> Int { index % width == 0 ? index : index - 1 }
        let first = original.indices.map { selected($0) ? doubled[$0] : original[$0] }
        let second = original.indices.map { selected($0) ? first[previousX($0)] : first[$0] }
        #expect(try gpu.read(firstSave) == first)
        #expect(try gpu.read(secondSave) == second)
        #expect(try gpu.read(destination) == second)

        try chain.setMask(nil, stages: 0..<4)
        try chain.setEnabled(false, at: 1)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        let unmasked = original.indices.map { doubled[previousX($0)] }
        #expect(try gpu.read(firstSave) == first)
        #expect(try gpu.read(secondSave) == unmasked)
        #expect(try gpu.read(destination) == unmasked)
    }

    @Test("A scaled effect is composited at full resolution before a save")
    func scaledMask() throws {
        let input = SIMD4<Float>(0.75, 0.5, 1.25, 0.5)
        let gain = try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            return p.vector(color.rgb * 2, color.a)
        }
        let source = try gpu.color(width: 4, height: 2) { _, _ in input }
        let mask = try gpu.color(width: 4, height: 2) { x, _ in SIMD4(x.isMultiple(of: 2) ? 0.5 : 0.25, 0, 0, 0) }
        let saved = try gpu.color(width: 4, height: 2)
        let destination = try gpu.color(width: 4, height: 2)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(gain), .save(to: saved)])
        try chain.setResolutionScale(0.5, at: 0)
        try chain.setMask(.init(texture: mask), stages: 0..<1)
        try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
        let expected = (0..<8).map { $0.isMultiple(of: 2) ? input * SIMD4(2, 2, 2, 1) : input }
        #expect(try gpu.read(saved) == expected)
        #expect(try gpu.read(destination) == expected)
        #expect(chain.intermediateAllocationCount == 4)
    }

    @Test("Save-only chains preserve linear HDR and present sRGB without an intermediate")
    func saveOnly() throws {
        let input = SIMD4<Float>(2, 0.25, 0, 1)
        let source = try gpu.color(width: 1, height: 1) { _, _ in input }
        let saved = try gpu.color(width: 1, height: 1)
        let display = try gpu.color(width: 1, height: 1, pixelFormat: .rgba8Unorm_srgb)
        let chain = try PostProcessChain(
            queue: gpu.queue, destinationPixelFormat: .rgba8Unorm_srgb, effects: [.save(to: saved)]
        )
        try gpu.run { try chain.encode(source: source, destination: display, commandBuffer: $0) }
        #expect(try gpu.read(saved) == [input])
        var rgba = SIMD4<UInt8>.zero
        withUnsafeMutableBytes(of: &rgba) { bytes in
            display.getBytes(bytes.baseAddress!, bytesPerRow: 4, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0)
        }
        #expect(rgba.x == 255 && abs(Int(rgba.y) - 137) <= 1 && rgba.z == 0 && rgba.w == 255)
        #expect(chain.intermediateAllocationCount == 0)
    }

    @Test("Mask and save validation preserve the previous binding after invalid edits")
    func validation() throws {
        let source = try gpu.color(width: 4, height: 4)
        let destination = try gpu.color(width: 4, height: 4)
        let mismatch = try gpu.color(width: 2, height: 2)
        let saved = try gpu.color(width: 4, height: 4)
        let graph = try PostProcessGraph { $0.sceneColor(at: $0.screenUV) }
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.graph(graph), .save(to: saved)])
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessChain.Failure.stageIndex(2)) {
            try chain.setMask(.init(texture: mismatch), stages: 0..<3)
        }
        try chain.setMask(.init(texture: mismatch), stages: 0..<1)
        #expect(throws: PostProcessChain.Failure.mask) {
            try chain.setMask(.init(texture: source, threshold: .nan), stages: 0..<1)
        }
        #expect(throws: PostProcessChain.Failure.scaledSave(stage: 1)) {
            try chain.setResolutionScale(0.5, at: 1)
        }
        #expect(try chain.resolutionScale(at: 1) == 1)
        #expect(throws: PostProcessChain.Failure.sizeMismatch) {
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }
        try chain.setMask(.init(texture: destination), stages: 0..<1)
        #expect(throws: PostProcessChain.Failure.mask) {
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }
        try chain.setMask(nil, stages: 0..<1)
        try chain.replaceEffect(at: 1, with: .save(to: source))
        #expect(throws: PostProcessChain.Failure.saveDestination) {
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }
        try chain.replaceEffect(at: 1, with: .save(to: mismatch))
        #expect(throws: PostProcessChain.Failure.sizeMismatch) {
            try chain.encode(source: source, destination: destination, commandBuffer: buffer)
        }
    }
}
