import Metal
import simd

/// Adds authored lens flare sprites over linear scene color using one depth-tested light.
public final class SpriteLensFlare: PostProcessEffect {
    public static let maximumElements = 16

    public struct Element {
        /// A top-left texture with straight alpha and linear or sRGB color storage.
        public let texture: any MTLTexture
        /// Full width and height of the square sprite in pixels.
        public let size: Float
        /// Zero places the sprite on the light, one half at frame center, and one opposite the light.
        public let distance: Float
        /// Linear RGB multiplier independent of the texture's stored color space.
        public let color: SIMD3<Float>

        public init(texture: any MTLTexture, size: Float = 1, distance: Float = 0, color: SIMD3<Float> = .one) {
            self.texture = texture
            self.size = size
            self.distance = distance
            self.color = color
        }
    }

    public let device: any MTLDevice
    public let elements: [Element]
    /// Assign a freshly projected light when the camera or light moves. Nil bypasses the flare.
    public var light: LensFlare.Light?
    /// Must match the scene depth and the projected light's stored depth.
    public var depthMode: MeshDepthMode
    public let occlusionRadius: Float
    public let occlusionSamples: Int
    public let depthBias: Float
    public var readsDepth: Bool { true }

    private let shaders: PostProcessShaders

    /// Elements are composited in array order. Treat their textures as read-only
    /// until the command buffer completes. Use a distinct color destination.
    public init(
        device: any MTLDevice, elements: [Element], light: LensFlare.Light? = nil, depthMode: MeshDepthMode = .forward,
        occlusionRadius: Float = 8, occlusionSamples: Int = 9, depthBias: Float = 1e-4
    ) throws {
        guard elements.count <= Self.maximumElements, (0...64).contains(occlusionRadius),
              (1...256).contains(occlusionSamples), (0...1).contains(depthBias) else {
            throw PostProcessEffectError.invalidSettings("Use up to 16 sprites, a radius of 0...64 pixels, 1...256 depth taps, and bias in 0...1.")
        }
        for element in elements {
            guard element.size.isNormal, (0...16_384).contains(element.size), (0...1).contains(element.distance),
                  element.color.isFiniteVector, element.color.min() >= 0, element.color.max() <= 65_504 else {
                throw PostProcessEffectError.invalidSettings("Use a positive sprite size up to 16384, a distance in 0...1, and color in 0...65504.")
            }
            guard element.texture.device === device else { throw PostProcessEffectError.wrongDevice }
            do {
                try EnvironmentBackground.validate(element.texture, device: device, type: .type2D)
            } catch {
                throw PostProcessEffectError.unsupportedInput("Sprite lens flares need filterable, tracked 2D RGBA color textures.")
            }
        }
        self.device = device
        self.elements = elements
        self.light = light
        self.depthMode = depthMode
        self.occlusionRadius = occlusionRadius
        self.occlusionSamples = occlusionSamples
        self.depthBias = depthBias
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_copy_fragment", formats: [pixelFormat])
        _ = try shaders.render("frostfield_flare_sprites", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_flare_visibility")
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try depthMode.requireProjectionDepth()
        if let light, !light.isValid || light.intensity > 65_504 || light.color.max() > 65_504 {
            throw PostProcessEffectError.invalidSettings("Use a finite projected light with color and intensity in 0...65504.")
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer, requiresDepth: true
        )
        guard !TextureTransfer.sharesStorage(source, destination),
              depth.map({ !TextureTransfer.sharesStorage($0, destination) }) ?? true,
              !elements.contains(where: { TextureTransfer.sharesStorage($0.texture, destination) }) else {
            throw PostProcessEffectError.destination
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        guard let depth else { throw PostProcessEffectError.missingDepth }
        guard let light, light.intensity > 0, !elements.isEmpty else {
            let copy = try shaders.render("frostfield_copy_fragment", formats: [destination.pixelFormat])
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "Frostfield bypass sprite flare", commandBuffer: commandBuffer, pipeline: copy
            ) { $0.setFragmentTexture(source, index: 0) }
            return
        }
        let composite = try shaders.render("frostfield_flare_sprites", formats: [destination.pixelFormat])
        let occlusion = try shaders.compute("frostfield_flare_visibility")
        guard let visibility = device.makeBuffer(
            length: MemoryLayout<Float>.stride, options: [.storageModePrivate, .hazardTrackingModeTracked]
        ) else { throw PostProcessEffectError.allocation }
        try FlareDepthVisibility.encode(
            lights: [.init(light)], radius: occlusionRadius, samples: occlusionSamples, bias: depthBias,
            depthMode: depthMode, depth: depth, output: visibility, pipeline: occlusion, commandBuffer: commandBuffer
        )

        let packed = elements.map { element in
            let center = light.position + (SIMD2(repeating: 0.5) - light.position) * (2 * element.distance)
            return PackedElement(placement: SIMD4(center.x, center.y, element.size, 0),
                                 tint: SIMD4(element.color * light.color * light.intensity, 0))
        }
        var textures: [(any MTLTexture)?] = elements.map(\.texture)
        // The shader declares a fixed texture array. Unused slots are never sampled.
        textures += Array(repeating: source, count: Self.maximumElements - textures.count)
        try packed.withUnsafeBytes { bytes in
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "Frostfield sprite lens flare", commandBuffer: commandBuffer, pipeline: composite
            ) { encoder in
                var count = UInt32(packed.count)
                encoder.setFragmentTexture(source, index: 0)
                encoder.setFragmentTextures(textures, range: 1..<(1 + Self.maximumElements))
                encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 0)
                encoder.setFragmentBytes(&count, length: MemoryLayout<UInt32>.stride, index: 1)
                encoder.setFragmentBuffer(visibility, offset: 0, index: 2)
            }
        }
    }

    private struct PackedElement {
        let placement: SIMD4<Float>
        let tint: SIMD4<Float>
    }
}

extension SpriteLensFlare: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
