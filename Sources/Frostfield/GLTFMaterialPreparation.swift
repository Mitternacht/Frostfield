import Metal
import simd

public extension GLTFScene.Material {
    enum BasisTextureStorage: Equatable, Sendable {
        /// Decodes LDR Basis images to RGBA8.
        case rgba8
        /// Keeps LDR Basis images compressed when the device and sampler allow it.
        /// Falls back to RGBA8 when a compressed upload cannot meet those constraints.
        case preferDeviceCompressed
    }

    /// Prepares glTF maps and samplers on one device. Do this during asset preparation.
    /// Texture transforms, emissive strength, specular, sheen, anisotropy, iridescence,
    /// dispersion, diffuse transmission, diffuse roughness, volume scattering and unlit
    /// shading carry over to the material. Imported MTL height maps are prepared
    /// when active. Unlit materials keep only their base color and alpha.
    /// A volume without KHR_materials_transmission (one that diffuse transmission enters)
    /// becomes a `Transmission` with a zero factor that carries the thickness and attenuation.
    ///
    /// A KTX2 image that fails to decode or uses a block format the device cannot
    /// sample uses the texture's PNG or JPEG fallback when the asset provides one.
    func prepare(
        device: any MTLDevice,
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1),
        lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12,
        receivesShadows: Bool = false,
        environment: PhysicallyBasedMaterial.Environment? = nil
    ) throws -> PhysicallyBasedMaterial {
        try prepare(device: device, directionToLight: directionToLight, lightIntensity: lightIntensity,
                    ambientIntensity: ambientIntensity, receivesShadows: receivesShadows, environment: environment,
                    basisTextureStorage: .rgba8)
    }

    /// Prepares the material with the selected Basis texture storage policy.
    func prepare(
        device: any MTLDevice,
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1),
        lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12,
        receivesShadows: Bool = false,
        environment: PhysicallyBasedMaterial.Environment? = nil,
        basisTextureStorage: BasisTextureStorage
    ) throws -> PhysicallyBasedMaterial {
        let loader = TextureLoader(device: device)
        var textures: [TextureKey: any MTLTexture] = [:]
        var samplers: [PhysicallyBasedMaterial.TextureSampling: any MTLSamplerState] = [:]
        func loadPrimary(
            _ image: GLTFScene.Image, interpretation: TextureLoader.Interpretation, mipmapped: Bool
        ) throws -> any MTLTexture {
            if basisTextureStorage == .preferDeviceCompressed, image.mimeType == "image/ktx2" {
                do {
                    return try loader.loadKTX2Data(image.data, interpretation: interpretation, mipmapped: mipmapped,
                                                   allowingSpatialTextures: false, basisStorage: .deviceCompressed)
                } catch let failure as TextureLoader.Failure {
                    switch failure {
                    case .unsupportedTextureFormat, .mipmapGeneration, .textureAllocation, .invalidKTX2: break
                    default: throw failure
                    }
                }
            }
            return try loader.load(data: image.data, interpretation: interpretation, mipmapped: mipmapped)
        }
        func prepareTexture(_ source: GLTFScene.Texture?, linear: Bool) throws -> PhysicallyBasedMaterial.TextureBinding? {
            guard let source else { return nil }
            let sampling = source.sampling.textureSampling
            let key = TextureKey(
                image: source.image.sourceIndex, linear: linear, mipmapped: sampling.mipFilter != .notMipmapped
            )
            let texture: any MTLTexture
            if let cached = textures[key] {
                texture = cached
            } else {
                let interpretation: TextureLoader.Interpretation = linear ? .linearData : .encodedSRGB
                do {
                    texture = try loadPrimary(source.image, interpretation: interpretation, mipmapped: key.mipmapped)
                } catch let failure as TextureLoader.Failure {
                    switch failure {
                    case .unsupportedTextureFormat, .invalidKTX2:
                        guard let fallback = source.fallbackImage else { throw failure }
                        texture = try loader.load(data: fallback.data, interpretation: interpretation, mipmapped: key.mipmapped)
                    default: throw failure
                    }
                }
                textures[key] = texture
            }
            // Equal sampling settings share one sampler state. The description lets the path
            // tracer sample as the raster renderer does.
            let sampler: any MTLSamplerState
            if let cached = samplers[sampling] {
                sampler = cached
            } else {
                sampler = try sampling.makeSamplerState(device: device)
                TextureSampling.record(sampler, descriptor: sampling.descriptor)
                samplers[sampling] = sampler
            }
            return PhysicallyBasedMaterial.TextureBinding(
                texture: texture, sampler: sampler, textureCoordinateSet: source.textureCoordinateSet,
                transform: source.transform, sampling: sampling
            )
        }

        let coverage: PhysicallyBasedMaterial.AlphaMode
        switch alphaMode {
        case .opaque: coverage = .opaque
        case .mask(let cutoff): coverage = .mask(cutoff: cutoff)
        case .blend: coverage = .blend
        }
        let color = try prepareTexture(baseColorTexture, linear: false)
        if isUnlit {
            return try PhysicallyBasedMaterial(
                baseColorTexture: color?.texture, directionToLight: directionToLight, lightIntensity: lightIntensity,
                ambientIntensity: ambientIntensity, receivesShadows: receivesShadows, baseColorFactor: baseColorFactor,
                baseColorSampler: color?.sampler, alphaMode: coverage, isDoubleSided: isDoubleSided,
                baseColorTextureCoordinateSet: color?.textureCoordinateSet ?? .primary,
                baseColorTextureTransform: color?.transform ?? .identity, shadingModel: .unlit,
                baseColorSampling: color?.sampling
            )
        }
        let coat = try clearcoat.map { source in
            try PhysicallyBasedMaterial.Clearcoat(
                factor: source.factor, roughness: source.roughness,
                factorTexture: prepareTexture(source.factorTexture, linear: true),
                roughnessTexture: prepareTexture(source.roughnessTexture, linear: true),
                normalTexture: prepareTexture(source.normalTexture, linear: true),
                normalScale: source.normalScale
            )
        }
        let opticalTransmission: PhysicallyBasedMaterial.Transmission?
        if transmission != nil || volume != nil || indexOfRefraction != 1.5 {
            opticalTransmission = try PhysicallyBasedMaterial.Transmission(
                factor: transmission?.factor ?? 0,
                indexOfRefraction: indexOfRefraction, thickness: volume?.thickness ?? 0,
                attenuationDistance: volume?.attenuationDistance,
                attenuationColor: volume?.attenuationColor ?? SIMD3(repeating: 1),
                factorTexture: prepareTexture(transmission?.factorTexture, linear: true),
                thicknessTexture: prepareTexture(volume?.thicknessTexture, linear: true),
                dispersion: dispersion
            )
        } else {
            opticalTransmission = nil
        }
        let preparedBump: PhysicallyBasedMaterial.BumpMap?
        if hasActiveBumpMap, let bumpMap, let binding = try prepareTexture(bumpMap.texture, linear: true) {
            preparedBump = try .init(texture: binding, scale: bumpMap.scale)
        } else {
            preparedBump = nil
        }
        return try PhysicallyBasedMaterial(
            baseColorTexture: color?.texture,
            metallic: metallicFactor, roughness: roughnessFactor,
            directionToLight: directionToLight, lightIntensity: lightIntensity, ambientIntensity: ambientIntensity,
            receivesShadows: receivesShadows, baseColorFactor: baseColorFactor, baseColorSampler: color?.sampler,
            metallicRoughnessTexture: prepareTexture(metallicRoughnessTexture, linear: true),
            normalTexture: prepareTexture(normalTexture, linear: true), normalScale: normalScale,
            occlusionTexture: prepareTexture(occlusionTexture, linear: true), occlusionStrength: occlusionStrength,
            emissiveTexture: prepareTexture(emissiveTexture, linear: false), emissiveFactor: emissiveFactor,
            alphaMode: coverage, isDoubleSided: isDoubleSided, environment: environment,
            baseColorTextureCoordinateSet: color?.textureCoordinateSet ?? .primary, clearcoat: coat,
            transmission: opticalTransmission, baseColorTextureTransform: color?.transform ?? .identity,
            emissiveStrength: emissiveStrength,
            specular: specular.map { source in
                try PhysicallyBasedMaterial.Specular(
                    factor: source.factor, colorFactor: source.colorFactor,
                    factorTexture: prepareTexture(source.factorTexture, linear: true),
                    colorTexture: prepareTexture(source.colorTexture, linear: false)
                )
            },
            sheen: sheen.map { source in
                try PhysicallyBasedMaterial.Sheen(
                    colorFactor: source.colorFactor, roughness: source.roughness,
                    colorTexture: prepareTexture(source.colorTexture, linear: false),
                    roughnessTexture: prepareTexture(source.roughnessTexture, linear: true)
                )
            },
            anisotropy: anisotropy.map { source in
                try PhysicallyBasedMaterial.Anisotropy(
                    strength: source.strength, rotation: source.rotation,
                    texture: prepareTexture(source.texture, linear: true)
                )
            },
            iridescence: iridescence.map { source in
                try PhysicallyBasedMaterial.Iridescence(
                    factor: source.factor, indexOfRefraction: source.indexOfRefraction,
                    minimumThickness: source.minimumThickness, maximumThickness: source.maximumThickness,
                    factorTexture: prepareTexture(source.factorTexture, linear: true),
                    thicknessTexture: prepareTexture(source.thicknessTexture, linear: true)
                )
            },
            baseColorSampling: color?.sampling,
            diffuseTransmission: try diffuseTransmission.map { source in
                try PhysicallyBasedMaterial.DiffuseTransmission(
                    factor: source.factor, colorFactor: source.colorFactor,
                    factorTexture: prepareTexture(source.factorTexture, linear: true),
                    colorTexture: prepareTexture(source.colorTexture, linear: false)
                )
            },
            diffuseRoughness: try diffuseRoughness.map { source in
                try PhysicallyBasedMaterial.DiffuseRoughness(
                    factor: source.factor, texture: prepareTexture(source.texture, linear: true)
                )
            },
            // A thin-walled volume (zero thickness), or one that no light enters, has no medium to scatter in.
            volumeScatter: try volumeScatter.flatMap { source in
                guard (volume?.thickness ?? 0) > 0, indexOfRefraction != 0,
                      (transmission?.factor ?? 0) > 0 || (diffuseTransmission?.factor ?? 0) > 0 else { return nil }
                return try PhysicallyBasedMaterial.VolumeScatter(
                    multiscatterColor: source.multiscatterColorFactor, anisotropy: source.scatterAnisotropy,
                    multiscatterColorTexture: prepareTexture(source.multiscatterColorTexture, linear: false)
                )
            },
            bumpMap: preparedBump
        )
    }
}

private struct TextureKey: Hashable {
    let image: Int
    let linear: Bool
    let mipmapped: Bool
}

private extension GLTFScene.Sampling {
    /// The same settings as a Metal-independent description.
    var textureSampling: PhysicallyBasedMaterial.TextureSampling {
        func address(_ wrap: Wrap) -> PhysicallyBasedMaterial.TextureSampling.Addressing {
            switch wrap {
            case .clampToEdge: .clampToEdge
            case .mirroredRepeat: .mirroredRepeat
            case .repeat: .repeat
            }
        }
        typealias Description = PhysicallyBasedMaterial.TextureSampling
        let (minFilter, mipFilter): (Description.Filtering, Description.MipFiltering) = switch minification {
        case .nearest: (.nearest, .notMipmapped)
        case .linear: (.linear, .notMipmapped)
        case .nearestMipmapNearest: (.nearest, .nearest)
        case .linearMipmapNearest: (.linear, .nearest)
        case .nearestMipmapLinear: (.nearest, .linear)
        case .linearMipmapLinear: (.linear, .linear)
        }
        return PhysicallyBasedMaterial.TextureSampling(
            addressU: address(wrapS), addressV: address(wrapT),
            magnification: magnification == .nearest ? .nearest : .linear,
            minification: minFilter, mipFilter: mipFilter
        )
    }
}
