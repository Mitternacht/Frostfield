import Metal
import simd

/// The nearest depth of opaque and blended meshes, for a fog pass.
///
/// A blended material that does not write the color-pass depth would otherwise fog
/// at the opaque surface behind it. This capture draws every visible mesh with depth
/// writes forced, including blended ones, using the graph camera. Pass the texture to
/// `PostProcessPass.encode` as the fog depth. The color image is still one composite,
/// so the pass fogs that pixel at the front surface rather than each layer before blending.
public struct FogDepthCapture {
    private let pipelines: MeshDepthCapturePipelines
    private let forwardDepth: any MTLDepthStencilState
    private let reversedDepth: any MTLDepthStencilState

    public init(device: any MTLDevice) throws {
        pipelines = try MeshDepthCapturePipelines(device: device, label: "Frostfield fog depth")
        func depth(_ compare: MTLCompareFunction) throws -> any MTLDepthStencilState {
            let descriptor = MTLDepthStencilDescriptor()
            descriptor.depthCompareFunction = compare
            descriptor.isDepthWriteEnabled = true
            guard let state = device.makeDepthStencilState(descriptor: descriptor) else {
                throw PostProcessEffectError.pipeline("Fog depth state could not be created.")
            }
            return state
        }
        forwardDepth = try depth(.less)
        reversedDepth = try depth(.greater)
    }

    /// Clears `depth` and draws the graph. The texture is Depth32Float, a render target,
    /// and the same size as the fog pass. Returns the number of draws.
    @discardableResult
    public func encode(
        graph: SceneGraph, into depth: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        depthMode: MeshDepthMode = .forward
    ) throws -> Int {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PostProcessEffectError.commandBuffer
        }
        guard commandBuffer.device === depth.device else { throw PostProcessEffectError.wrongDevice }
        guard depth.pixelFormat == .depth32Float, depth.textureType == .type2D, depth.sampleCount == 1,
              depth.width > 0, depth.height > 0, depth.usage.contains(.renderTarget),
              depth.storageMode != .memoryless else {
            throw PostProcessEffectError.unsupportedInput("Fog depth needs a Depth32Float render target.")
        }
        try depthMode.requireProjectionDepth()
        let aspect = Float(depth.width) / Float(depth.height)
        let clip = try SelectionOutline.projection(graph, aspectRatio: aspect, depthMode: depthMode)
            * simd_inverse(graph.cameraTransform.matrix)
        let depthState = depthMode == .forward ? forwardDepth : reversedDepth
        let pipelines = pipelines
        return try MeshRenderer.encodeDepthOnly(
            graph, clipFromWorld: clip, commandBuffer: commandBuffer, includes: { _ in true },
            makeEncoder: {
                let pass = MTLRenderPassDescriptor()
                pass.depthAttachment.texture = depth
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.storeAction = .store
                pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
                guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
                    throw PostProcessEffectError.encoder
                }
                encoder.label = "Frostfield fog depth"
                encoder.setRenderPipelineState(pipelines.plain)
                encoder.setDepthStencilState(depthState)
                return encoder
            },
            setPipeline: pipelines.setPipeline
        )
    }
}
