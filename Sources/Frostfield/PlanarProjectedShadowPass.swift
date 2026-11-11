import Foundation
import Metal
import simd

/// Blends a group of projected caster meshes once per covered pixel. Encode
/// after the receiver and opaque scene, using their stored color and depth.
/// The caller supplies a separate stencil scratch attachment, cleared by the
/// pass. Use the same world-to-clip transform and depth mode as the scene.
public final class PlanarProjectedShadowPass {
    public let device: any MTLDevice
    public let colorPixelFormat: MTLPixelFormat
    public let depthMode: MeshDepthMode
    private let pipeline: any MTLRenderPipelineState
    private let depthState: any MTLDepthStencilState

    public init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat = .rgba16Float, depthMode: MeshDepthMode = .forward
    ) throws {
        guard depthMode == .forward || depthMode == .reversed else { throw Failure.depthMode }
        guard [.rgba16Float, .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb].contains(colorPixelFormat) else {
            throw Failure.colorFormat
        }
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.depthMode = depthMode

        let state = MeshRenderState(blending: .normal, transparent: true, depthTest: false, depthWrite: false, side: .double)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield projected planar shadow"
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        descriptor.stencilAttachmentPixelFormat = .stencil8
        descriptor.inputPrimitiveTopology = .triangle
        state.configure(descriptor)
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_planar_shadow_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_planar_shadow_fragment") else {
                throw Failure.resource
            }
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        let depth = state.depthDescriptor(depthMode: depthMode)
        let stencil = MTLStencilDescriptor()
        stencil.stencilCompareFunction = .equal
        stencil.depthStencilPassOperation = .incrementClamp
        depth.frontFaceStencil = stencil
        depth.backFaceStencil = stencil
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else { throw Failure.resource }
        self.depthState = depthState
    }

    /// Encodes projected world-space meshes, for example the output of
    /// `PlanarShadowProjection.project(_:sourceToWorld:)`. Color, Depth32Float
    /// and Stencil8 must be separate, matching, single-sample 2D textures.
    /// Depth needs shader-read usage. The tint is unassociated linear RGBA.
    /// Depth bias is a tolerance in normalized stored depth units.
    public func encode(
        projectedMeshes: [MeshGeometry], clipFromWorld: simd_float4x4, color: any MTLTexture,
        depth: any MTLTexture, stencil: any MTLTexture, tint: SIMD4<Float> = SIMD4(0, 0, 0, 0.6),
        depthBias: Float = 1e-6, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        let attachments = [color, depth, stencil]
        guard commandBuffer.device === device, attachments.allSatisfy({ $0.device === device }) else {
            throw Failure.wrongDevice
        }
        guard color.pixelFormat == colorPixelFormat, stencil.pixelFormat == .stencil8,
              color.storageMode != .memoryless, PostProcessPass.isReadable(depth, format: .depth32Float),
              depth.width == color.width, depth.height == color.height,
              [color, stencil].allSatisfy({ texture in
                  texture.textureType == .type2D && texture.sampleCount == 1
                      && texture.width == color.width && texture.height == color.height
                      && texture.usage.contains(.renderTarget) && texture.hazardTrackingMode != .untracked
              }),
              !TextureTransfer.sharesStorage(color, depth), !TextureTransfer.sharesStorage(color, stencil),
              !TextureTransfer.sharesStorage(depth, stencil) else { throw Failure.attachments }
        guard planarShadowFinite(clipFromWorld), (0...1).contains(depthBias),
              (0..<3).allSatisfy({ (0...65_504).contains(tint[$0]) }), (0...1).contains(tint.w) else {
            throw Failure.parameters
        }
        guard !projectedMeshes.isEmpty, tint.w > 0 else { return }

        let draws = try projectedMeshes.map { mesh in
            (positions: try buffer(mesh.vertices.map(\.position)), indices: try buffer(mesh.indices), count: mesh.indices.count)
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.stencilAttachment.texture = stencil
        pass.stencilAttachment.loadAction = .clear
        pass.stencilAttachment.clearStencil = 0
        pass.stencilAttachment.storeAction = .dontCare
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.resource }
        encoder.label = "Frostfield projected planar shadow"
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)
        encoder.setStencilReferenceValue(0)
        var transform = clipFromWorld
        var colorValue = SIMD4(tint.x * tint.w, tint.y * tint.w, tint.z * tint.w, tint.w)
        var depthControl = SIMD2<Float>(depthBias, depthMode == .reversed ? 1 : 0)
        encoder.setVertexBytes(&transform, length: MemoryLayout<simd_float4x4>.stride, index: 1)
        encoder.setFragmentBytes(&colorValue, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.setFragmentBytes(&depthControl, length: MemoryLayout<SIMD2<Float>>.stride, index: 1)
        encoder.setFragmentTexture(depth, index: 0)
        for draw in draws {
            encoder.setVertexBuffer(draw.positions, offset: 0, index: 0)
            encoder.drawIndexedPrimitives(
                type: .triangle, indexCount: draw.count, indexType: .uint32, indexBuffer: draw.indices, indexBufferOffset: 0
            )
        }
        encoder.endEncoding()
    }

    private func buffer<Element>(_ array: [Element]) throws -> any MTLBuffer {
        try array.withUnsafeBytes { bytes in
            guard let start = bytes.baseAddress, bytes.count <= device.maxBufferLength,
                  let buffer = device.makeBuffer(bytes: start, length: bytes.count, options: .storageModeShared) else {
                throw Failure.resource
            }
            return buffer
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case depthMode, colorFormat, commandBuffer, wrongDevice, attachments, parameters, resource
        case pipeline(String)

        public var errorDescription: String? {
            switch self {
            case .depthMode: "Use forward or reversed projection depth for planar shadows."
            case .colorFormat: "Use a blendable RGBA16Float or RGBA/BGRA8 color attachment."
            case .commandBuffer: "Use an uncommitted retaining command buffer for planar shadows."
            case .wrongDevice: "The planar shadow attachments and command buffer need the pass's Metal device."
            case .attachments: "Use separate tracked single-sample color, readable Depth32Float and Stencil8 textures with matching dimensions."
            case .parameters: "Use a finite clip transform, linear RGBA tint and depth bias in 0...1."
            case .resource: "A planar shadow Metal resource or encoder could not be created."
            case .pipeline(let message): "The planar shadow pipeline could not be created. \(message)"
            }
        }
    }
}
