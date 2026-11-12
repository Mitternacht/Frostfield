import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MaterialDitheringTests {
    @Test("A material dithers an 8-bit target and leaves a float target unchanged")
    func eightBitTarget() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let side = 8
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        func graph(dithering: Bool, format: MTLPixelFormat) throws -> SceneGraph {
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: format, geometry: try GeometryRenderFixture.square(3),
                material: .classic(try ClassicMaterial(model: .basic, color: SIMD3(repeating: 0.5), dithering: dithering))
            )
            return SceneGraph(
                roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: camera
            )
        }
        func target(_ format: MTLPixelFormat) throws -> (color: any MTLTexture, depth: any MTLTexture) {
            let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: side, height: side, mipmapped: false
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
        func draw(_ graph: SceneGraph, format: MTLPixelFormat) throws -> [UInt8] {
            let attachments = try target(format)
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = attachments.color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.depthAttachment.texture = attachments.depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .dontCare
            pass.depthAttachment.clearDepth = 1
            let buffer = try #require(queue.makeCommandBuffer())
            _ = try graph.encode(pass: pass, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            var bytes = [UInt8](repeating: 0, count: side * side * 4)
            bytes.withUnsafeMutableBytes { raw in
                attachments.color.getBytes(
                    raw.baseAddress!, bytesPerRow: side * 4, from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0
                )
            }
            return bytes
        }
        let steady = try draw(graph(dithering: false, format: .rgba8Unorm), format: .rgba8Unorm)
        let dithered = try draw(graph(dithering: true, format: .rgba8Unorm), format: .rgba8Unorm)
        #expect(steady[0] == steady[1] && steady[1] == steady[2] && steady[3] == 255)
        #expect(stride(from: 0, to: steady.count, by: 4).allSatisfy { steady[$0] == steady[0] && steady[$0 + 3] == 255 })
        #expect(dithered.contains { $0 != steady[0] })
        for index in stride(from: 0, to: dithered.count, by: 4) {
            let pixel = index / 4
            let column = pixel % side
            let row = pixel / side
            let shift = R187DitheringReference.mirrorShift(column: column, row: row, height: side)
            for channel in 0..<3 {
                let expected = Int(((0.5 + shift[channel]) * 255).rounded())
                #expect(abs(Int(dithered[index + channel]) - expected) <= 1, "\(column), \(row) \(channel)")
            }
        }
        let wide = try draw(graph(dithering: true, format: .rgba16Float), format: .rgba16Float)
        let plainWide = try draw(graph(dithering: false, format: .rgba16Float), format: .rgba16Float)
        #expect(wide == plainWide)

        func physicallyBased(dithering: Bool) throws -> SceneGraph {
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: .rgba8Unorm, geometry: try GeometryRenderFixture.square(3),
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.5, 0.5, 0.5, 1),
                    alphaMode: .opaque, shadingModel: .unlit, dithering: dithering
                ))
            )
            return SceneGraph(
                roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: camera
            )
        }
        let pbrSteady = try draw(physicallyBased(dithering: false), format: .rgba8Unorm)
        let pbrDithered = try draw(physicallyBased(dithering: true), format: .rgba8Unorm)
        #expect(pbrSteady[0] == steady[0])
        #expect(pbrDithered == dithered)
    }
}
