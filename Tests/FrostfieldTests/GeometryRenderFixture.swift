import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Renders scene graphs into a square RGBA16Float target and reads every pixel back.
/// The default camera sits at the origin with a 90° field of view, so at depth d
/// the view spans ±d world units on both axes.
struct GeometryRenderFixture {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let side: Int
    let camera: PerspectiveCamera

    init(side: Int = 64) throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        self.side = side
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50)
    }

    func makeTarget() throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        return (
            try #require(device.makeTexture(descriptor: colorDescriptor)),
            try #require(device.makeTexture(descriptor: depthDescriptor))
        )
    }

    func pass(
        color: any MTLTexture, depth: any MTLTexture, depthMode: MeshDepthMode = .forward
    ) -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1
        return pass
    }

    /// Encodes and waits for one graph, returning its pixels and draw counts.
    func render(
        _ graph: SceneGraph, shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        depthMode: MeshDepthMode = .forward
    ) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        let target = try makeTarget()
        let buffer = try #require(queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: pass(color: target.color, depth: target.depth, depthMode: depthMode), commandBuffer: buffer,
            shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps, depthMode: depthMode
        )
        try finish(buffer)
        return (try pixels(of: target.color), statistics)
    }

    func finish(_ buffer: any MTLCommandBuffer) throws {
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "The render did not complete.")")
    }

    func pixels(of texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var pixels = [SIMD4<Float16>](repeating: .zero, count: side * side)
        try pixels.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: side * MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0
            )
        }
        return pixels.map(SIMD4<Float>.init)
    }

    func pixel(_ pixels: [SIMD4<Float>], x: Int, y: Int) -> SIMD4<Float> {
        pixels[y * side + x]
    }

    /// The pixel column or row that sees a camera-space offset at a depth, for the default camera.
    func column(x: Float, depth: Float) -> Int {
        Int((x / depth * 0.5 + 0.5) * Float(side))
    }

    func row(y: Float, depth: Float) -> Int {
        Int((0.5 - y / depth * 0.5) * Float(side))
    }

    /// A square in the XY plane facing +Z with a +X tangent.
    static func square(_ halfExtent: Float) throws -> MeshGeometry {
        let h = halfExtent
        let corners = [SIMD3(-h, -h, 0), SIMD3(h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, h, 0)]
        return try MeshGeometry(
            vertices: corners.map {
                MeshVertex(
                    position: $0, normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2($0.x, $0.y),
                    tangent: SIMD4(1, 0, 0, 1)
                )
            },
            indices: [0, 1, 2, 0, 2, 3]
        )
    }

    func unlit(
        _ geometry: MeshGeometry, skin: MeshSkin? = nil, morphTargets: MeshMorphTargets? = nil,
        supportsInstancing: Bool = false, material: MeshMaterial = .unlit, depthMode: MeshDepthMode = .forward
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material,
            depthMode: depthMode, skin: skin, morphTargets: morphTargets, supportsInstancing: supportsInstancing
        )
    }

    /// The unlit material's color for a +Z normal: `(0.22 + 0.78 |n|) × tint`.
    static func unlitFacingCamera(_ tint: SIMD4<Float>) -> SIMD4<Float> {
        let rgb = SIMD3<Float>(0.22, 0.22, 1) * SIMD3(tint.x, tint.y, tint.z) * min(max(tint.w, 0), 1)
        return SIMD4(rgb, min(max(tint.w, 0), 1))
    }
}

func translationMatrix(_ x: Float, _ y: Float = 0, _ z: Float = 0) -> simd_float4x4 {
    SceneTransform(translation: SIMD3(x, y, z)).matrix
}

func isNearly(_ a: SIMD4<Float>, _ b: SIMD4<Float>, tolerance: Float = 0.004) -> Bool {
    simd_reduce_max(abs(a - b)) <= tolerance
}
