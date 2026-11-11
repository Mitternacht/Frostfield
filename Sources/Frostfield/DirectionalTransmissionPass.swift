import Metal

/// Depth peels retain the RGB transmittance and exact forward depth of each
/// crossing. Two private depth textures are reused across all cascade slices.
final class DirectionalTransmissionPass {
    enum Caster {
        case uniform(SIMD3<Float>)
        case textured(TexturedCaster)
        case graph(GraphCaster)
    }

    // Only a textured draw owns this larger material record. The common draw
    // representation stays small for disabled maps and uniform casters.
    final class TexturedCaster {
        let material: PhysicallyBasedMaterial.Prepared
        let inputs: PhysicallyBasedMaterial.Prepared.TransmissionShadow

        init(material: PhysicallyBasedMaterial.Prepared, inputs: PhysicallyBasedMaterial.Prepared.TransmissionShadow) {
            self.material = material
            self.inputs = inputs
        }
    }

    final class GraphCaster {
        let pipeline: any MTLRenderPipelineState
        let material: PhysicallyBasedMaterial

        init(pipeline: any MTLRenderPipelineState, material: PhysicallyBasedMaterial) {
            self.pipeline = pipeline
            self.material = material
        }
    }

    private struct Pipelines {
        let plain: any MTLRenderPipelineState
        let skinned: any MTLRenderPipelineState
    }

    let layers: any MTLTexture
    let layerCount: Int
    private let depth: [any MTLTexture]
    private let uniform: Pipelines
    private let textured: Pipelines
    private let depthState: any MTLDepthStencilState

    init(device: any MTLDevice, resolution: Int, slices: Int, layerCount: Int) throws {
        // At most 128 MiB including the scratch depth surfaces.
        let bytesPerTexel = 16 * slices * layerCount + 8
        guard (1...DirectionalShadowMap.maximumTransmissionLayers).contains(layerCount),
              resolution * resolution * bytesPerTexel <= 128 * 1024 * 1024 else {
            throw DirectionalShadowMap.Failure.transmissionLayers
        }
        self.layerCount = layerCount

        let color = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: resolution, height: resolution, mipmapped: false
        )
        color.textureType = .type2DArray
        color.arrayLength = slices * layerCount
        color.storageMode = .private
        color.hazardTrackingMode = .tracked
        color.usage = [.renderTarget, .shaderRead]
        guard let layers = device.makeTexture(descriptor: color) else {
            throw DirectionalShadowMap.Failure.resource("directional transmission layers")
        }
        layers.label = "Frostfield directional shadow transmission layers"
        self.layers = layers

        let depth = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: resolution, height: resolution, mipmapped: false
        )
        depth.storageMode = .private
        depth.hazardTrackingMode = .tracked
        depth.usage = [.renderTarget, .shaderRead]
        self.depth = try (0..<2).map { index in
            guard let texture = device.makeTexture(descriptor: depth) else {
                throw DirectionalShadowMap.Failure.resource("directional transmission peel depth")
            }
            texture.label = "Frostfield directional transmission peel depth \(index)"
            return texture
        }

        let descriptor = MTLDepthStencilDescriptor()
        descriptor.depthCompareFunction = .less
        descriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: descriptor) else {
            throw DirectionalShadowMap.Failure.resource("directional transmission depth state")
        }
        self.depthState = depthState

        let library: any MTLLibrary
        do { library = try device.makeDefaultLibrary(bundle: .module) }
        catch { throw DirectionalShadowMap.Failure.library(error.localizedDescription) }
        guard let vertex = library.makeFunction(name: "frostfield_shadow_vertex"),
              let skinnedVertex = library.makeFunction(name: "frostfield_skinned_shadow_vertex"),
              let uniformFragment = library.makeFunction(name: "frostfield_transmitted_shadow_fragment"),
              let texturedFragment = library.makeFunction(name: "frostfield_textured_transmitted_shadow_fragment") else {
            throw DirectionalShadowMap.Failure.missingShader
        }
        let render = MTLRenderPipelineDescriptor()
        render.label = "Frostfield directional shadow transmission"
        render.colorAttachments[0].pixelFormat = .rgba32Float
        render.depthAttachmentPixelFormat = .depth32Float
        render.inputPrimitiveTopology = .triangle
        func prepare(_ fragment: any MTLFunction) throws -> Pipelines {
            render.fragmentFunction = fragment
            render.vertexFunction = vertex
            let plain = try device.makeRenderPipelineState(descriptor: render)
            render.vertexFunction = skinnedVertex
            let skinned = try device.makeRenderPipelineState(descriptor: render)
            return Pipelines(plain: plain, skinned: skinned)
        }
        do {
            uniform = try prepare(uniformFragment)
            textured = try prepare(texturedFragment)
        } catch {
            throw DirectionalShadowMap.Failure.pipeline(error.localizedDescription)
        }
    }

    func makeEncoder(
        slice: Int, layer: Int, commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = layers
        pass.colorAttachments[0].slice = slice * layerCount + layer
        pass.colorAttachments[0].clearColor = .init(red: 1, green: 1, blue: 1, alpha: 1)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth[layer % 2]
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = layer + 1 < layerCount ? .store : .dontCare
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw DirectionalShadowMap.Failure.encoder
        }
        encoder.label = "Frostfield directional transmission cascade \(slice) layer \(layer)"
        encoder.setDepthStencilState(depthState)
        encoder.setFragmentTexture(depth[(layer + 1) % 2], index: 1)
        var followsPeel: UInt32 = layer == 0 ? 0 : 1
        encoder.setFragmentBytes(&followsPeel, length: MemoryLayout<UInt32>.size, index: 1)
        return encoder
    }

    func setPipeline(on encoder: any MTLRenderCommandEncoder, skinned: Bool, textured: Bool) {
        let pipelines = textured ? self.textured : uniform
        encoder.setRenderPipelineState(skinned ? pipelines.skinned : pipelines.plain)
    }
}
