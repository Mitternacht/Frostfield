import Foundation
import Metal

/// Physically based texture maps generated on the GPU from one typed graph of material channels.
///
/// The maps follow `ProceduralSurface`'s conventions, so they plug straight into
/// `PhysicallyBasedMaterial`:
///
/// - `baseColorTexture`: `.rgba8Unorm_srgb` with opaque alpha; sampling decodes linear RGB.
/// - `metallicRoughnessTexture`: `.rgba8Unorm` with occlusion, roughness and metallic in
///   red, green and blue, bound as both the metallic-roughness and occlusion texture.
/// - `normalTexture`: `.rgba8Unorm` tangent-space normals derived on the GPU from the
///   height channel by wrapping central differences.
///
/// Each texture is shared-storage, mipmapped, and complete when the initializer returns.
public struct ShaderMaterialMaps {
    /// Material channels evaluated at each texel center from `gridUV` and `texelSize`.
    /// Unset channels use `ProceduralSurface.Sample`'s defaults: roughness 0.5, metallic 0, occlusion 1.
    /// Values are clamped to 0...1 when stored.
    public struct Channels {
        /// Linear RGB.
        public var baseColor: ShaderValue<SIMD3<Float>>
        /// Perceptual roughness.
        public var roughness: ShaderValue<Float>?
        public var metallic: ShaderValue<Float>?
        /// Ambient occlusion, where 1 is unoccluded.
        public var occlusion: ShaderValue<Float>?
        /// A unitless height for the normal map. Without one, no normal map is generated.
        public var height: ShaderValue<Float>?

        public init(
            baseColor: ShaderValue<SIMD3<Float>>, roughness: ShaderValue<Float>? = nil,
            metallic: ShaderValue<Float>? = nil, occlusion: ShaderValue<Float>? = nil,
            height: ShaderValue<Float>? = nil
        ) {
            self.baseColor = baseColor
            self.roughness = roughness
            self.metallic = metallic
            self.occlusion = occlusion
            self.height = height
        }
    }

    /// sRGB color. Samples decode to linear RGB in Metal.
    public let baseColorTexture: any MTLTexture
    /// Linear occlusion, roughness, and metallic in the red, green, and blue channels.
    public let metallicRoughnessTexture: any MTLTexture
    /// Linear tangent-space normals, or nil when the channels have no height.
    public let normalTexture: (any MTLTexture)?

    /// Builds the channels once and generates every map in one command buffer on
    /// `queue`, waiting for completion.
    ///
    /// Declare app-controlled values with `uniform(_:_:)`; the maps use the declared
    /// values. Uniform values never change the generated Metal source, so building
    /// again with new values reuses the cached pipelines.
    ///
    /// - Parameters:
    ///   - width, height: 1–8192 texels per side and at most 16,777,216 texels.
    ///   - heightScale: Finite, in 0...1. A unit height difference represents
    ///     `heightScale` times the texture width, with square texels; zero produces flat normals.
    ///   - textures: One texture per name the channels declare; see `ShaderTextureGenerator`.
    public init(
        queue: any MTLCommandQueue, width: Int, height: Int, heightScale: Float = 0.01,
        textures: [String: any MTLTexture] = [:],
        building body: (ShaderGraphBuilder) throws -> Channels
    ) throws {
        guard heightScale.isFinite, (0...1).contains(heightScale) else { throw Failure.heightScale }
        guard (1...ShaderTextureGenerator.maximumDimension).contains(width),
              (1...ShaderTextureGenerator.maximumDimension).contains(height),
              width <= ShaderTextureGenerator.maximumTexelCount / height else {
            throw ShaderComputeFailure.dimensions(width: width, height: height)
        }
        let builder = ShaderGraphBuilder()
        let channels = try body(builder)
        let owned: [(any ShaderBuilderOwned)?] = [
            channels.baseColor, channels.roughness, channels.metallic, channels.occlusion, channels.height,
        ]
        guard owned.allSatisfy({ $0.map { $0.owner === builder } ?? true }) else {
            throw MaterialGraph.Failure.invalidDeclaration("A graph output belongs to a different builder.")
        }
        try builder.checkOperands()
        let surfaceData = builder.vector(
            channels.occlusion ?? builder.constant(Float(1)), channels.roughness ?? builder.constant(Float(0.5)),
            channels.metallic ?? builder.constant(Float(0)), builder.constant(Float(1))
        )
        let declarations = builder.declarations
        func generator(_ output: MaterialGraph.NodeID) throws -> ShaderTextureGenerator {
            let graph = try TextureShaderGraph(
                nodes: builder.nodes, uniforms: declarations.uniforms, textures: declarations.textures,
                functions: declarations.functions, labels: declarations.labels, output: output
            )
            return try ShaderTextureGenerator(graph: graph, device: queue.device)
        }
        let color = try generator(channels.baseColor.id)
        let data = try generator(surfaceData.id)
        let relief = try channels.height.map { try generator($0.id) }

        let device = queue.device
        func texture(
            _ format: MTLPixelFormat, mipmapped: Bool, storage: MTLStorageMode, label: String
        ) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: mipmapped
            )
            descriptor.storageMode = storage
            descriptor.usage = format == .rgba8Unorm_srgb ? [.shaderRead, .shaderWrite, .pixelFormatView] : [.shaderRead, .shaderWrite]
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw ShaderComputeFailure.resource("a \(width) × \(height) \(label) texture")
            }
            texture.label = label
            return texture
        }
        let base = try texture(.rgba8Unorm_srgb, mipmapped: true, storage: .shared, label: "Frostfield graph color")
        let packed = try texture(.rgba8Unorm, mipmapped: true, storage: .shared, label: "Frostfield graph surface data")
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ShaderComputeFailure.resource("a command buffer")
        }
        commandBuffer.label = "Frostfield graph material maps"
        try color.encode(into: base, commandBuffer: commandBuffer, textures: textures)
        try data.encode(into: packed, commandBuffer: commandBuffer, textures: textures)

        var normal: (any MTLTexture)?
        if let relief {
            let heights = try texture(.r32Float, mipmapped: false, storage: .private, label: "Frostfield graph heights")
            let normals = try texture(.rgba8Unorm, mipmapped: true, storage: .shared, label: "Frostfield graph normal")
            try relief.encode(into: heights, commandBuffer: commandBuffer, textures: textures)
            let pipeline = try Self.normalPipeline(device: device)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ShaderComputeFailure.encoder("compute")
            }
            encoder.label = "Frostfield height to normal"
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(heights, index: 0)
            encoder.setTexture(normals, index: 1)
            var slopeScale = heightScale * Float(width) * 0.5
            encoder.setBytes(&slopeScale, length: MemoryLayout<Float>.size, index: 0)
            let groupWidth = pipeline.threadExecutionWidth
            let groupHeight = max(1, pipeline.maxTotalThreadsPerThreadgroup / groupWidth)
            encoder.dispatchThreadgroups(
                MTLSize(
                    width: (width + groupWidth - 1) / groupWidth, height: (height + groupHeight - 1) / groupHeight,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: groupWidth, height: groupHeight, depth: 1)
            )
            encoder.endEncoding()
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw ShaderComputeFailure.encoder("blit") }
            blit.label = "Frostfield graph normal mipmaps"
            blit.generateMipmaps(for: normals)
            blit.endEncoding()
            normal = normals
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw ShaderComputeFailure.execution(
                commandBuffer.error?.localizedDescription ?? "The command buffer did not complete."
            )
        }
        baseColorTexture = base
        metallicRoughnessTexture = packed
        normalTexture = normal
    }

    /// Uses the generated maps in a physically based material with repeat sampling,
    /// like `ProceduralSurface.material(...)`.
    public func material(
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1),
        lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12,
        receivesShadows: Bool = false,
        environment: PhysicallyBasedMaterial.Environment? = nil,
        normalScale: Float = 1
    ) throws -> PhysicallyBasedMaterial {
        let surfaceData = PhysicallyBasedMaterial.TextureBinding(texture: metallicRoughnessTexture)
        return try PhysicallyBasedMaterial(
            baseColorTexture: baseColorTexture, metallic: 1, roughness: 1,
            directionToLight: directionToLight, lightIntensity: lightIntensity,
            ambientIntensity: ambientIntensity, receivesShadows: receivesShadows,
            metallicRoughnessTexture: surfaceData,
            normalTexture: normalTexture.map { .init(texture: $0) }, normalScale: normalScale,
            occlusionTexture: surfaceData, alphaMode: .opaque, environment: environment
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case heightScale

        public var errorDescription: String? {
            switch self {
            case .heightScale: "Graph material height scale must be finite and between zero and one."
            }
        }
    }

    private static func normalPipeline(device: any MTLDevice) throws -> any MTLComputePipelineState {
        let name = "frostfield_height_to_normal"
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let function = library.makeFunction(name: name) else {
                throw ShaderComputeFailure.pipeline("Frostfield's shader library does not contain \(name).")
            }
            return try device.makeComputePipelineState(function: function)
        } catch let failure as ShaderComputeFailure {
            throw failure
        } catch {
            throw ShaderComputeFailure.pipeline(error.localizedDescription)
        }
    }
}
