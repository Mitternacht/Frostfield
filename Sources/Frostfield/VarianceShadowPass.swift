import Metal

/// Reusable separable filter for the first and second moments of shadow depth.
struct VarianceShadowPass {
    let moments: any MTLTexture
    private let scratch: any MTLTexture
    private let vertical: any MTLComputePipelineState
    private let horizontal: any MTLComputePipelineState

    init(device: any MTLDevice, resolution: Int, slices: Int) throws {
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw DirectionalShadowMap.Failure.library(error.localizedDescription)
        }
        guard let vertical = library.makeFunction(name: "frostfield_variance_shadow_vertical"),
              let horizontal = library.makeFunction(name: "frostfield_variance_shadow_horizontal") else {
            throw DirectionalShadowMap.Failure.missingShader
        }
        do {
            self.vertical = try device.makeComputePipelineState(function: vertical)
            self.horizontal = try device.makeComputePipelineState(function: horizontal)
        } catch {
            throw DirectionalShadowMap.Failure.pipeline(error.localizedDescription)
        }
        let descriptor = Self.descriptor(format: .rg32Float, resolution: resolution, slices: slices, usage: [.shaderRead, .shaderWrite])
        guard let scratch = device.makeTexture(descriptor: descriptor),
              let moments = device.makeTexture(descriptor: descriptor) else {
            throw DirectionalShadowMap.Failure.resource("variance shadow moments")
        }
        scratch.label = "Frostfield variance shadow intermediate"
        moments.label = "Frostfield filtered shadow moments"
        self.scratch = scratch
        self.moments = moments
    }

    /// A valid binding for the unused moments slot in basic and PCF pipelines.
    static func fallback(device: any MTLDevice) throws -> any MTLTexture {
        let descriptor = descriptor(format: .rg16Float, resolution: 1, slices: 1, usage: .shaderRead)
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DirectionalShadowMap.Failure.resource("shadow moment fallback")
        }
        return texture
    }

    private static func descriptor(
        format: MTLPixelFormat, resolution: Int, slices: Int, usage: MTLTextureUsage
    ) -> MTLTextureDescriptor {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: resolution, height: resolution, mipmapped: false
        )
        descriptor.textureType = .type2DArray
        descriptor.arrayLength = slices
        descriptor.storageMode = .private
        descriptor.usage = usage
        descriptor.hazardTrackingMode = .tracked
        return descriptor
    }

    func encode(depth: any MTLTexture, slices: [SIMD2<UInt32>], commandBuffer: any MTLCommandBuffer) throws {
        guard !slices.isEmpty else { return }
        try encode(vertical, from: depth, to: scratch, slices: slices,
                   label: "Frostfield vertical shadow moments", commandBuffer: commandBuffer)
        try encode(horizontal, from: scratch, to: moments, slices: slices,
                   label: "Frostfield horizontal shadow moments", commandBuffer: commandBuffer)
    }

    private func encode(
        _ pipeline: any MTLComputePipelineState, from source: any MTLTexture, to target: any MTLTexture,
        slices: [SIMD2<UInt32>], label: String, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw DirectionalShadowMap.Failure.momentEncoder }
        encoder.label = label
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(target, index: 1)
        slices.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        let width = pipeline.threadExecutionWidth
        let height = min(pipeline.maxTotalThreadsPerThreadgroup / width, 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (target.width + width - 1) / width,
                    height: (target.height + height - 1) / height, depth: slices.count),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
    }
}
