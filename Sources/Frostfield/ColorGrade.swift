import Foundation
import Metal

/// A prepared, single-pass warm color grade for linear sRGB, premultiplied SDR color.
public final class ColorGrade {
    private let device: any MTLDevice
    private let destinationPixelFormat: MTLPixelFormat
    private let pipeline: any MTLRenderPipelineState

    public init(device: any MTLDevice, destinationPixelFormat: MTLPixelFormat) throws {
        self.device = device
        self.destinationPixelFormat = destinationPixelFormat
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield color grade"
        descriptor.colorAttachments[0].pixelFormat = destinationPixelFormat
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_grade_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_grade_fragment") else {
                throw Failure.missingShader
            }
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let error as Failure {
            throw error
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Reads one source texel per destination pixel. Use a stand-alone linear
    /// RGBA16Float texture and an equal-sized destination with the prepared format.
    /// Strength zero passes through the linear channels before the destination's
    /// normal format conversion. Submit a command buffer that retains references,
    /// or keep this grade and its textures alive until the GPU has completed.
    public func encode(
        source: any MTLTexture,
        destination: any MTLTexture,
        strength: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, destination: destination, strength: strength, commandBuffer: commandBuffer)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        var strength = strength
        encoder.label = "Frostfield color grade"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentBytes(&strength, length: MemoryLayout<Float>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Checks everything `encode` requires without encoding.
    func validate(
        source: any MTLTexture, destination: any MTLTexture, strength: Float, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard strength.isFinite, (0...1).contains(strength) else { throw Failure.invalidStrength }
        guard commandBuffer.device === device, source.device === device, destination.device === device else {
            throw Failure.wrongDevice
        }
        guard source !== destination, source.textureType == .type2D, destination.textureType == .type2D,
              source.width == destination.width, source.height == destination.height,
              source.pixelFormat == .rgba16Float, destination.pixelFormat == destinationPixelFormat,
              source.usage.contains(.shaderRead), destination.usage.contains(.renderTarget),
              source.parent == nil, destination.parent == nil,
              source.buffer == nil, source.iosurface == nil, source.heap == nil else {
            throw Failure.incompatibleTextures
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case invalidStrength
        case wrongDevice
        case incompatibleTextures
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield's color grade shaders are missing."
            case .pipeline(let reason): "Could not prepare Frostfield's color grade: \(reason)"
            case .invalidStrength: "Color grade strength must be finite and between 0 and 1."
            case .wrongDevice: "Color grade textures and command buffer must use its Metal device."
            case .incompatibleTextures: "Color grade needs separate, equal-sized textures with compatible formats and usage."
            case .encoder: "Metal could not create the color grade encoder."
            }
        }
    }
}
