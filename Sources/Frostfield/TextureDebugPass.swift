import Foundation
import Metal

/// Shows a texture subresource as an opaque image, optionally over a rendered frame.
/// The pass uses nearest texels in top-left image order and leaves tone mapping to the caller.
public final class TextureDebugPass {
    public enum Channel: UInt32, Sendable {
        case rgb, red, green, blue, alpha
    }

    public enum Source {
        /// A 2D image, 2D array layer, 3D Z slice, or cube face at the selected mip.
        /// For cube maps the face order is +X, −X, +Y, −Y, +Z, −Z.
        case color(any MTLTexture, slice: Int = 0, mipLevel: Int = 0, channel: Channel = .rgb)
        /// A Depth32Float image or 2D array layer. Near depth is white.
        case depth(any MTLTexture, slice: Int = 0, mipLevel: Int = 0, reversed: Bool = false)
        /// A cascade after the graph has encoded the directional depth pass.
        case directionalShadow(DirectionalShadowMap, cascade: Int = 0)
        /// A rendered slice of the punctual depth array. Spot slots come first.
        /// Each point or rectangle slot then uses six slices in Metal cube face order.
        /// Select a slice that was rendered for the frame being inspected.
        case punctualShadow(PunctualShadowMaps, slice: Int)
    }

    public let device: any MTLDevice
    public let pixelFormat: MTLPixelFormat
    private let pipelines: [Kind: any MTLRenderPipelineState]

    enum Kind: String, CaseIterable {
        case color2D = "color_2d"
        case colorArray = "color_array"
        case color3D = "color_3d"
        case colorCube = "color_cube"
        case depth2D = "depth_2d"
        case depthArray = "depth_array"
        case label
    }

    private struct View {
        let texture: any MTLTexture
        let kind: Kind
        let slice: UInt32
        let mip: UInt32
        let channel: Channel
        let reversedDepth: Bool
    }

    /// Prepares the pipelines for linear RGBA16Float or an 8-bit sRGB destination.
    public init(device: any MTLDevice, pixelFormat: MTLPixelFormat = .rgba16Float) throws {
        guard PostProcessPass.supportedPixelFormats.contains(pixelFormat) else { throw Failure.pixelFormat }
        self.device = device
        self.pixelFormat = pixelFormat
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_texture_debug_vertex") else {
                throw Failure.missingShader
            }
            var pipelines: [Kind: any MTLRenderPipelineState] = [:]
            for kind in Kind.allCases {
                guard let fragment = library.makeFunction(name: "frostfield_texture_debug_" + kind.rawValue) else {
                    throw Failure.missingShader
                }
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.label = "Frostfield texture viewer (\(kind.rawValue))"
                descriptor.vertexFunction = vertex
                descriptor.fragmentFunction = fragment
                descriptor.colorAttachments[0].pixelFormat = pixelFormat
                if kind == .label { MeshRenderState.transparent.configure(descriptor) }
                pipelines[kind] = try device.makeRenderPipelineState(descriptor: descriptor)
            }
            self.pipelines = pipelines
        } catch let error as Failure {
            throw error
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Draws on the caller's uncommitted command buffer, which must retain its references.
    /// Sources and destination must be separate, single-sample, hazard-tracked textures.
    /// End the source's writing pass before encoding this draw. For native shadow maps,
    /// encode on their owning queue and after the selected slice has been rendered.
    /// On-demand shadow maps require a valid slice and use the map's command-buffer lease.
    ///
    /// A nil region fills the destination. An explicit top-left pixel rectangle
    /// loads the existing target and keeps the pixels outside the rectangle. The
    /// source stretches to that rectangle. sRGB textures decode and encode in Metal.
    public func encode(
        source: Source, destination: any MTLTexture, region: MTLRegion? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(commandBuffer: commandBuffer)
        guard destination.device === device else { throw Failure.wrongDevice }
        let view = try view(for: source, commandBuffer: commandBuffer)
        guard !Self.aliases(view.texture, destination) else { throw Failure.destination }
        try validate(destination: destination, region: region)
        guard let pipeline = pipelines[view.kind] else { throw Failure.missingShader }

        let shadowUse: ShadowMapUpdateCache.Use?
        switch source {
        case let .directionalShadow(map, cascade):
            shadowUse = try map.beginRead(slice: cascade, commandBuffer: commandBuffer)
        case let .punctualShadow(maps, slice):
            shadowUse = try maps.beginRead(slice: slice, commandBuffer: commandBuffer)
        default:
            shadowUse = nil
        }
        let encoder: any MTLRenderCommandEncoder
        do {
            encoder = try makeEncoder(destination: destination, region: region, commandBuffer: commandBuffer)
        } catch {
            shadowUse?.cancelBeforeAccess()
            throw error
        }
        encoder.label = "Frostfield texture viewer"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(view.texture, index: 0)
        var options = SIMD4(view.slice, view.mip, view.channel.rawValue, view.reversedDepth ? 1 : 0)
        encoder.setFragmentBytes(&options, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        shadowUse?.didEncode()
    }

    /// Composites a prepared label over existing target pixels at a top-left
    /// pixel origin. Supply straight linear RGBA, with alpha in 0...1.
    public func encode(
        label: Label, destination: any MTLTexture, origin: SIMD2<Int>,
        color: SIMD4<Float> = SIMD4(1, 0, 0, 1), commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(commandBuffer: commandBuffer)
        guard destination.device === device, label.mask.device === device else { throw Failure.wrongDevice }
        let limit = Float(Float16.greatestFiniteMagnitude)
        guard (0...limit).contains(color.x), (0...limit).contains(color.y), (0...limit).contains(color.z),
              (0...1).contains(color.w) else { throw Failure.labelColor }
        let region = MTLRegionMake2D(origin.x, origin.y, label.width, label.height)
        try validate(destination: destination, region: region)
        guard let pipeline = pipelines[.label] else { throw Failure.missingShader }

        let encoder = try makeEncoder(destination: destination, region: region, commandBuffer: commandBuffer)
        encoder.label = "Frostfield texture viewer label"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(label.mask, index: 0)
        var tint = color
        encoder.setFragmentBytes(&tint, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func validate(commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device else { throw Failure.wrongDevice }
    }

    private func validate(destination: any MTLTexture, region: MTLRegion?) throws {
        guard destination.pixelFormat == pixelFormat, destination.textureType == .type2D,
              destination.sampleCount == 1, destination.usage.contains(.renderTarget),
              destination.storageMode != .memoryless, destination.hazardTrackingMode != .untracked else {
            throw Failure.destination
        }
        if let region {
            guard region.origin.x >= 0, region.origin.y >= 0, region.origin.z == 0,
                  region.size.width > 0, region.size.height > 0, region.size.depth == 1,
                  region.size.width <= destination.width - region.origin.x,
                  region.size.height <= destination.height - region.origin.y else { throw Failure.region }
        }
    }

    private func makeEncoder(
        destination: any MTLTexture, region: MTLRegion?, commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = region == nil ? .dontCare : .load
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        if let region {
            encoder.setViewport(MTLViewport(
                originX: Double(region.origin.x), originY: Double(region.origin.y),
                width: Double(region.size.width), height: Double(region.size.height), znear: 0, zfar: 1
            ))
            encoder.setScissorRect(MTLScissorRect(
                x: region.origin.x, y: region.origin.y, width: region.size.width, height: region.size.height
            ))
        }
        return encoder
    }

    private func view(for source: Source, commandBuffer: any MTLCommandBuffer) throws -> View {
        switch source {
        case let .color(texture, slice, mip, channel):
            return try view(texture, slice: slice, mip: mip, channel: channel, reversedDepth: nil)
        case let .depth(texture, slice, mip, reversed):
            return try view(texture, slice: slice, mip: mip, channel: .rgb, reversedDepth: reversed)
        case let .directionalShadow(map, cascade):
            try map.validateFitted()
            try map.validate(commandBuffer: commandBuffer)
            return try view(map.texture, slice: cascade, mip: 0, channel: .rgb, reversedDepth: false)
        case let .punctualShadow(maps, slice):
            try maps.validate(commandBuffer: commandBuffer)
            return try view(maps.texture, slice: slice, mip: 0, channel: .rgb, reversedDepth: true)
        }
    }

    private func view(
        _ texture: any MTLTexture, slice: Int, mip: Int, channel: Channel, reversedDepth: Bool?
    ) throws -> View {
        let kind: Kind
        let sliceCount: Int
        if reversedDepth != nil {
            try Self.validateSource(texture, device: device, mip: mip)
            guard texture.pixelFormat == .depth32Float else { throw Failure.source }
            switch texture.textureType {
            case .type2D: (kind, sliceCount) = (.depth2D, 1)
            case .type2DArray: (kind, sliceCount) = (.depthArray, texture.arrayLength)
            default: throw Failure.source
            }
        } else {
            (kind, sliceCount) = try Self.describeColor(texture, device: device, mip: mip)
        }
        guard (0..<sliceCount).contains(slice) else { throw Failure.subresource }
        return View(texture: texture, kind: kind, slice: UInt32(slice), mip: UInt32(mip), channel: channel,
                    reversedDepth: reversedDepth ?? false)
    }

    static func describeColor(
        _ texture: any MTLTexture, device: any MTLDevice, mip: Int
    ) throws -> (kind: Kind, sliceCount: Int) {
        try validateSource(texture, device: device, mip: mip)
        guard PostProcessPass.isFloatSampleable(texture.pixelFormat, linear: false, device: device) else {
            throw Failure.source
        }
        switch texture.textureType {
        case .type2D: return (.color2D, 1)
        case .type2DArray: return (.colorArray, texture.arrayLength)
        case .type3D: return (.color3D, max(texture.depth >> mip, 1))
        case .typeCube: return (.colorCube, 6)
        default: throw Failure.source
        }
    }

    private static func validateSource(_ texture: any MTLTexture, device: any MTLDevice, mip: Int) throws {
        guard texture.device === device else { throw Failure.wrongDevice }
        guard texture.sampleCount == 1, texture.usage.contains(.shaderRead), !texture.isFramebufferOnly,
              texture.storageMode != .memoryless, texture.hazardTrackingMode != .untracked else { throw Failure.source }
        guard (0..<texture.mipmapLevelCount).contains(mip) else { throw Failure.subresource }
    }

    static func aliases(_ first: any MTLTexture, _ second: any MTLTexture) -> Bool {
        let firstRoot = root(of: first), secondRoot = root(of: second)
        return firstRoot === secondRoot || firstRoot.buffer.map { $0 === secondRoot.buffer } == true
    }

    private static func root(of texture: any MTLTexture) -> any MTLTexture {
        var result = texture
        while let parent = result.parent { result = parent }
        return result
    }

    public enum Failure: LocalizedError, Equatable {
        case pixelFormat, missingShader, pipeline(String), commandBuffer, wrongDevice
        case source, subresource, destination, region, encoder
        case labelText, labelResource, labelColor

        public var errorDescription: String? {
            switch self {
            case .pixelFormat: "Use RGBA16Float or an 8-bit sRGB format for the texture viewer."
            case .missingShader: "The Frostfield texture viewer shaders are unavailable."
            case .pipeline(let detail): "Could not prepare the Frostfield texture viewer. \(detail)"
            case .commandBuffer: "Use an uncommitted command buffer that retains its references."
            case .wrongDevice: "Use the texture viewer's Metal device for all resources."
            case .source: "Use a shader-readable, tracked, single-sample color or Depth32Float source."
            case .subresource: "The selected texture mip or slice does not exist."
            case .destination: "Use a separate tracked 2D render target in the viewer's pixel format."
            case .region: "Use a nonempty 2D viewer region inside the destination."
            case .encoder: "Could not start the Frostfield texture viewer draw."
            case .labelText: "Use one visible text line up to 1024 UTF-16 units and 4096 by 1024 pixels."
            case .labelResource: "Could not allocate the texture viewer label."
            case .labelColor: "Use finite linear label RGB in 0...65504 and alpha in 0...1."
            }
        }
    }
}
