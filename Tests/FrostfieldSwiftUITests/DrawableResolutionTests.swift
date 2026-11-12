import Foundation
import Frostfield
@testable import FrostfieldSwiftUI
import Metal
import QuartzCore
import simd
import Testing

@MainActor
struct DrawableResolutionTests {
    typealias Resolution = FrostfieldSceneView.DrawableResolution

    @Test("Fixed and display resolution apply to one Metal layer without resizing it")
    func drawableResolution() throws {
        let layer = CAMetalLayer()
        let logical = CGSize(width: 80.5, height: 40.25)
        layer.frame = CGRect(origin: .zero, size: logical)
        let fixed = try Resolution(pixelsPerPoint: 0.75)
        let nativePixels = CGSize(width: 161, height: 80.5)

        let scaled = try fixed.apply(
            to: layer, logicalSize: logical, backingSize: nativePixels, displayScale: 2
        ).get()
        #expect(scaled == CGSize(width: 61, height: 31))
        #expect(layer.drawableSize == scaled)
        #expect(layer.bounds.size == logical)
        #expect(layer.contentsScale == 2)

        let sameScale = try fixed.apply(
            to: layer, logicalSize: logical,
            backingSize: CGSize(width: 241.5, height: 120.75), displayScale: 3
        ).get()
        #expect(sameScale == scaled)
        #expect(layer.drawableSize == scaled)
        #expect(layer.contentsScale == 3)

        let native = try Resolution.display.apply(
            to: layer, logicalSize: logical, backingSize: nativePixels, displayScale: 2
        ).get()
        #expect(native == CGSize(width: 161, height: 81))
        #expect(layer.drawableSize == native)
        #expect(layer.bounds.size == logical)
    }

    @Test("Invalid ratios and oversized targets fail before changing the layer")
    func invalidAndEmptySizes() throws {
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: 0) }
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: -1) }
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: .infinity) }
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: .nan) }

        let layer = CAMetalLayer()
        layer.contentsScale = 2
        layer.drawableSize = CGSize(width: 64, height: 32)
        let large = try Resolution(pixelsPerPoint: 8)
        #expect(large.apply(
            to: layer, logicalSize: CGSize(width: 2_049, height: 300), backingSize: .zero, displayScale: 3
        ) == .failure(.size))
        #expect(try large.apply(
            to: layer, logicalSize: .zero, backingSize: .zero, displayScale: 3
        ).get() == nil)
        #expect(layer.drawableSize == CGSize(width: 64, height: 32))
        #expect(layer.contentsScale == 2)
    }

    @Test("An acquired Metal texture has the requested pixel size",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func acquiredTextureDimensions() throws {
        let layer = CAMetalLayer()
        layer.device = try #require(MTLCreateSystemDefaultDevice())
        let logical = CGSize(width: 96, height: 48)
        layer.frame = CGRect(origin: .zero, size: logical)
        let resolution = try Resolution(pixelsPerPoint: 0.5)
        _ = try resolution.apply(
            to: layer, logicalSize: logical, backingSize: CGSize(width: 192, height: 96), displayScale: 2
        ).get()
        let texture = try #require(layer.nextDrawable()?.texture)
        #expect(texture.width == 48)
        #expect(texture.height == 24)
    }

    @Test("A post chain scales the scene color the view presents")
    func postChain() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(1, 0, 0), alphaMode: .opaque
        )
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 4),
            material: .physicallyBased(material)
        )
        let scene = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: camera
        )
        let destination = try texture(device: device, format: .rgba16Float)
        var sceneColor: (any MTLTexture)?
        var depth: (any MTLTexture)?
        try encode(queue: queue, scene: scene, chain: nil, destination: destination, sceneColor: &sceneColor, depth: &depth)
        let direct = try read(destination)
        #expect(direct.x > 0.9 && direct.y < 0.05 && direct.z < 0.05, "\(direct)")

        let effect = try PostProcessGraph { p in p.sceneColor(at: p.screenUV) * 0.5 }
        let chain = try PostProcessChain(queue: queue, destinationPixelFormat: .rgba16Float, effects: [.graph(effect)])
        try encode(queue: queue, scene: scene, chain: chain, destination: destination, sceneColor: &sceneColor, depth: &depth)
        let graded = try read(destination)
        #expect(abs(graded.x - direct.x * 0.5) < 0.02 && graded.y < 0.05 && graded.z < 0.05, "\(graded)")

        let displayChain = try PostProcessChain(queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.graph(effect)])
        #expect(throws: SceneFrameEncoder.Failure.chainDestinationFormat) {
            try encode(queue: queue, scene: scene, chain: displayChain, destination: destination, sceneColor: &sceneColor, depth: &depth)
        }
        let prepared = try MeshRenderer(
            device: device, colorPixelFormat: .bgra8Unorm, geometry: .plane(width: 2, height: 4),
            material: .physicallyBased(material)
        )
        let eightBit = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: prepared)], camera: camera
        )
        #expect(throws: SceneFrameEncoder.Failure.sceneColorFormat) {
            try SceneFrameEncoder.encode(
                queue: queue, graph: eightBit, colorPixelFormat: .bgra8Unorm, chain: chain,
                clearColor: SIMD4(0, 0, 0, 1), destination: destination, sceneColor: &sceneColor, depth: &depth,
                commandBuffer: try #require(queue.makeCommandBuffer())
            )
        }
    }

    private func encode(
        queue: any MTLCommandQueue, scene: SceneGraph, chain: PostProcessChain?, destination: any MTLTexture,
        sceneColor: inout (any MTLTexture)?, depth: inout (any MTLTexture)?
    ) throws {
        let buffer = try #require(queue.makeCommandBuffer())
        try SceneFrameEncoder.encode(
            queue: queue, graph: scene, colorPixelFormat: .rgba16Float, chain: chain,
            clearColor: SIMD4(0, 0, 0, 1), destination: destination, sceneColor: &sceneColor, depth: &depth,
            commandBuffer: buffer
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
    }

    private func texture(device: any MTLDevice, format: MTLPixelFormat) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: 16, height: 16, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func read(_ texture: any MTLTexture) throws -> SIMD4<Float> {
        var pixel = SIMD4<Float16>.zero
        try withUnsafeMutableBytes(of: &pixel) { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: 8,
                from: MTLRegionMake2D(8, 8, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(pixel)
    }
}
