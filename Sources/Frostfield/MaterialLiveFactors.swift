import Foundation
import Metal
import simd

extension PhysicallyBasedMaterial {
    /// The values of a material that can change after preparation without a new
    /// pipeline, like the mutable `MeshStandardMaterial` and `MeshPhysicalMaterial`
    /// factors in Three.js. Optional fields are nil when the material lacks the
    /// map, mode or layer they belong to, and stay nil.
    public struct Factors: Equatable, Sendable {
        public var baseColorFactor: SIMD4<Float>
        public var metallic: Float
        public var roughness: Float
        public var emissiveFactor: SIMD3<Float>
        public var emissiveStrength: Float
        /// Present when the material has a normal map.
        public var normalScale: Float?
        /// Present when the material has an occlusion map.
        public var occlusionStrength: Float?
        /// Present when the alpha mode is `mask`.
        public var alphaCutoff: Float?
        /// One transform per bound core map.
        public var textureTransforms: [AnimationProperty.MaterialMap: TextureTransform]
        /// Present when the clearcoat layer is active.
        public var clearcoatFactor: Float?
        public var clearcoatRoughness: Float?
        /// Present when transmission is active.
        public var transmissionFactor: Float?
        /// Present when the material has a specular extension.
        public var specularFactor: Float?
        public var specularColorFactor: SIMD3<Float>?
        /// Present when sheen is active.
        public var sheenColorFactor: SIMD3<Float>?
        public var sheenRoughness: Float?
        /// Present when iridescence is active.
        public var iridescenceFactor: Float?
        /// Present when anisotropy is active.
        public var anisotropyStrength: Float?
        public var anisotropyRotation: Float?

        /// The material's authored factors.
        public init(_ material: PhysicallyBasedMaterial) {
            let maps = material.coreMaps
            baseColorFactor = material.baseColorFactor
            metallic = material.settings.y
            roughness = material.settings.x
            emissiveFactor = material.emissiveFactor
            emissiveStrength = material.emissiveStrength
            normalScale = maps.bindings[2] == nil ? nil : maps.normalScale
            occlusionStrength = maps.bindings[3] == nil ? nil : maps.occlusionStrength
            if case .mask(let cutoff) = material.alphaMode { alphaCutoff = cutoff }
            var transforms: [AnimationProperty.MaterialMap: TextureTransform] = [:]
            for (index, map) in AnimationProperty.MaterialMap.allCases.enumerated() {
                if let binding = maps.bindings[index] { transforms[map] = binding.transform }
            }
            textureTransforms = transforms
            if material.hasClearcoat, let coat = material.clearcoat {
                clearcoatFactor = coat.factor
                clearcoatRoughness = coat.roughness
            }
            if material.hasTransmission { transmissionFactor = material.transmission?.factor }
            if material.shadingModel == .physicallyBased, let specular = material.specular {
                specularFactor = specular.factor
                specularColorFactor = specular.colorFactor
            }
            if material.shadingModel == .physicallyBased, material.hasSheen, let sheen = material.sheen {
                sheenColorFactor = sheen.colorFactor
                sheenRoughness = sheen.roughness
            }
            if material.shadingModel == .physicallyBased, material.hasIridescence {
                iridescenceFactor = material.iridescence?.factor
            }
            if material.shadingModel == .physicallyBased, material.hasAnisotropy, let anisotropy = material.anisotropy {
                anisotropyStrength = anisotropy.strength
                anisotropyRotation = anisotropy.rotation
            }
        }

        /// Whether the same optional fields and maps are present in both.
        func hasSameShape(as other: Factors) -> Bool {
            (normalScale == nil) == (other.normalScale == nil)
                && (occlusionStrength == nil) == (other.occlusionStrength == nil)
                && (alphaCutoff == nil) == (other.alphaCutoff == nil)
                && Set(textureTransforms.keys) == Set(other.textureTransforms.keys)
                && (clearcoatFactor == nil) == (other.clearcoatFactor == nil)
                && (clearcoatRoughness == nil) == (other.clearcoatRoughness == nil)
                && (transmissionFactor == nil) == (other.transmissionFactor == nil)
                && (specularFactor == nil) == (other.specularFactor == nil)
                && (specularColorFactor == nil) == (other.specularColorFactor == nil)
                && (sheenColorFactor == nil) == (other.sheenColorFactor == nil)
                && (sheenRoughness == nil) == (other.sheenRoughness == nil)
                && (iridescenceFactor == nil) == (other.iridescenceFactor == nil)
                && (anisotropyStrength == nil) == (other.anisotropyStrength == nil)
                && (anisotropyRotation == nil) == (other.anisotropyRotation == nil)
        }

        /// The value of one animatable factor, or nil when the material lacks it.
        func value(of property: AnimationProperty.Material) -> AnimationValue? {
            switch property {
            case .baseColorFactor: .vector4(baseColorFactor)
            case .baseColorRGB: .color(SIMD3(baseColorFactor.x, baseColorFactor.y, baseColorFactor.z))
            case .metallic: .number(metallic)
            case .roughness: .number(roughness)
            case .emissiveFactor: .color(emissiveFactor)
            case .emissiveStrength: .number(emissiveStrength)
            case .normalScale: normalScale.map { .number($0) }
            case .occlusionStrength: occlusionStrength.map { .number($0) }
            case .alphaCutoff: alphaCutoff.map { .number($0) }
            case .textureOffset(let map): textureTransforms[map].map { .vector2($0.offset) }
            case .textureRotation(let map): textureTransforms[map].map { .number($0.rotation) }
            case .textureScale(let map): textureTransforms[map].map { .vector2($0.scale) }
            case .clearcoatFactor: clearcoatFactor.map { .number($0) }
            case .clearcoatRoughness: clearcoatRoughness.map { .number($0) }
            case .transmissionFactor: transmissionFactor.map { .number($0) }
            case .specularFactor: specularFactor.map { .number($0) }
            case .specularColor: specularColorFactor.map { .color($0) }
            case .sheenColor: sheenColorFactor.map { .color($0) }
            case .sheenRoughness: sheenRoughness.map { .number($0) }
            case .iridescenceFactor: iridescenceFactor.map { .number($0) }
            case .anisotropyStrength: anisotropyStrength.map { .number($0) }
            case .anisotropyRotation: anisotropyRotation.map { .number($0) }
            }
        }

        /// Sets one factor from a value of its kind. Throws when the material lacks the
        /// factor, the value has the wrong kind, or a texture transform becomes invalid.
        mutating func set(_ property: AnimationProperty.Material, to value: AnimationValue) throws {
            guard self.value(of: property) != nil else { throw LiveFactors.Failure.missingFactor(property) }
            func number() throws -> Float {
                guard let number = value.floatValue else { throw LiveFactors.Failure.valueKind(property) }
                return number
            }
            func color() throws -> SIMD3<Float> {
                guard let color = value.vector3Value else { throw LiveFactors.Failure.valueKind(property) }
                return color
            }
            func vector2() throws -> SIMD2<Float> {
                guard let vector = value.vector2Value else { throw LiveFactors.Failure.valueKind(property) }
                return vector
            }
            func transform(_ map: AnimationProperty.MaterialMap, _ change: (inout (SIMD2<Float>, Float, SIMD2<Float>)) throws -> Void) throws {
                let old = textureTransforms[map]!
                var parts = (old.offset, old.rotation, old.scale)
                try change(&parts)
                textureTransforms[map] = try TextureTransform(offset: parts.0, rotation: parts.1, scale: parts.2)
            }
            switch property {
            case .baseColorFactor:
                guard let vector = value.vector4Value else { throw LiveFactors.Failure.valueKind(property) }
                baseColorFactor = vector
            case .baseColorRGB:
                baseColorFactor = SIMD4(try color(), baseColorFactor.w)
            case .metallic: metallic = try number()
            case .roughness: roughness = try number()
            case .emissiveFactor: emissiveFactor = try color()
            case .emissiveStrength: emissiveStrength = try number()
            case .normalScale: normalScale = try number()
            case .occlusionStrength: occlusionStrength = try number()
            case .alphaCutoff: alphaCutoff = try number()
            case .textureOffset(let map): let offset = try vector2(); try transform(map) { $0.0 = offset }
            case .textureRotation(let map): let rotation = try number(); try transform(map) { $0.1 = rotation }
            case .textureScale(let map): let scale = try vector2(); try transform(map) { $0.2 = scale }
            case .clearcoatFactor: clearcoatFactor = try number()
            case .clearcoatRoughness: clearcoatRoughness = try number()
            case .transmissionFactor: transmissionFactor = try number()
            case .specularFactor: specularFactor = try number()
            case .specularColor: specularColorFactor = try color()
            case .sheenColor: sheenColorFactor = try color()
            case .sheenRoughness: sheenRoughness = try number()
            case .iridescenceFactor: iridescenceFactor = try number()
            case .anisotropyStrength: anisotropyStrength = try number()
            case .anisotropyRotation: anisotropyRotation = try number()
            }
        }
    }

    /// Live factors for one material, like a Three.js material whose `roughness` or
    /// `color` changes between frames. Every renderer prepared with the material
    /// copies the current values into each draw, and `PathTracingScene.updateMaterials()`
    /// repacks them for path tracing, so no pipeline or texture is prepared again.
    ///
    /// Each change is validated by building the material it describes with the same
    /// initializer, so the ranges and errors are those of `PhysicallyBasedMaterial`.
    /// Factors can change a layer's strength, including to zero, but cannot activate
    /// a clearcoat, transmission, sheen, anisotropy or iridescence layer that was
    /// inactive when the material was made. Thread-safe.
    public final class LiveFactors: @unchecked Sendable {
        /// The authored factors, which `reset()` restores and animation blends from.
        public let rest: Factors
        /// Immutable authored material, including the layers and maps of its rest factors.
        let base: PhysicallyBasedMaterial
        private let lock = NSLock()
        // Guarded by `lock`.
        private var currentFactors: Factors
        private var currentMaterial: PhysicallyBasedMaterial
        private var currentRevision: UInt64 = 0

        init(material: PhysicallyBasedMaterial) {
            var base = material
            base.liveFactors = nil
            self.base = base
            rest = Factors(base)
            currentFactors = rest
            currentMaterial = base
        }

        /// The factors renderers use now.
        public var current: Factors { lock.withLock { currentFactors } }

        /// Increases after every change. Equal revisions from one instance mean equal factors.
        public var revision: UInt64 { lock.withLock { currentRevision } }

        /// The material that the current factors describe.
        var material: PhysicallyBasedMaterial { lock.withLock { currentMaterial } }

        typealias Snapshot = (material: PhysicallyBasedMaterial, revision: UInt64)

        /// Both values must describe the same change when a scene packs its records.
        var snapshot: Snapshot { lock.withLock { (currentMaterial, currentRevision) } }

        /// Replaces every factor. The factors must have the same optional fields and
        /// texture maps as `rest`, stay within the material's documented ranges, and
        /// leave inactive layers inactive. Throws without changing anything otherwise.
        public func set(_ factors: Factors) throws {
            guard factors.hasSameShape(as: rest) else { throw Failure.shapeChanged }
            let resolved = try base.applying(factors)
            for (name, wasActive, isActive) in [
                ("clearcoat", base.hasClearcoat, resolved.hasClearcoat),
                ("transmission", base.hasTransmission, resolved.hasTransmission),
                ("sheen", base.hasSheen, resolved.hasSheen),
                ("anisotropy", base.hasAnisotropy, resolved.hasAnisotropy),
                ("iridescence", base.hasIridescence, resolved.hasIridescence),
            ] where isActive && !wasActive {
                throw Failure.inactiveLayer(name)
            }
            lock.withLock {
                guard currentFactors != factors else { return }
                currentFactors = factors
                currentMaterial = resolved
                currentRevision &+= 1
            }
        }

        /// Changes some factors with `set(_:)` semantics.
        public func update(_ change: (inout Factors) throws -> Void) throws {
            var factors = current
            try change(&factors)
            try set(factors)
        }

        /// Restores the authored factors.
        public func reset() {
            lock.withLock {
                guard currentFactors != rest else { return }
                currentFactors = rest
                currentMaterial = base
                currentRevision &+= 1
            }
        }

        public enum Failure: LocalizedError, Equatable {
            case shapeChanged
            case inactiveLayer(String)
            case missingFactor(AnimationProperty.Material)
            case valueKind(AnimationProperty.Material)

            public var errorDescription: String? {
                switch self {
                case .shapeChanged:
                    "Live material factors must keep the material's maps, alpha mode and layers; only their values can change."
                case .inactiveLayer(let name):
                    "The \(name) layer was inactive when the material was made, so live factors cannot activate it."
                case .missingFactor(let property): "The material has no \(property) factor to change."
                case .valueKind(let property): "The value does not have the kind of the material's \(property) factor."
                }
            }
        }
    }

    /// Returns a copy whose factors can change after renderers are prepared with it,
    /// through `liveFactors`. Every renderer prepared with the copy shares them.
    public func withLiveFactors() -> PhysicallyBasedMaterial {
        var copy = self
        copy.liveFactors = LiveFactors(material: self)
        return copy
    }

    /// The material these factors describe, built with the public initializer.
    func applying(
        _ factors: Factors,
        surfaceOverride: (graph: SurfaceShaderGraph, uniforms: ShaderUniforms?, textures: [String: any MTLTexture])? = nil
    ) throws -> PhysicallyBasedMaterial {
        let maps = coreMaps.bindings
        func rebind(_ index: Int) -> TextureBinding? {
            guard let binding = maps[index] else { return nil }
            let map = AnimationProperty.MaterialMap.allCases[index]
            return TextureBinding(
                texture: binding.texture, sampler: binding.sampler, textureCoordinateSet: binding.textureCoordinateSet,
                transform: factors.textureTransforms[map] ?? binding.transform, sampling: binding.sampling,
                normalEncoding: binding.normalEncoding, normalSpace: binding.normalSpace
            )
        }
        var mode = alphaMode
        if case .mask = alphaMode, let cutoff = factors.alphaCutoff { mode = .mask(cutoff: cutoff) }
        let coat = try clearcoat.map { coat in
            try Clearcoat(
                factor: factors.clearcoatFactor ?? coat.factor, roughness: factors.clearcoatRoughness ?? coat.roughness,
                factorTexture: coat.factorTexture, roughnessTexture: coat.roughnessTexture,
                normalTexture: coat.normalTexture, normalScale: coat.normalScale
            )
        }
        let optical = try transmission.map { source in
            try Transmission(
                factor: factors.transmissionFactor ?? source.factor, indexOfRefraction: source.indexOfRefraction,
                thickness: source.thickness, attenuationDistance: source.attenuationDistance,
                attenuationColor: source.attenuationColor, factorTexture: source.factorTexture,
                thicknessTexture: source.thicknessTexture, dispersion: source.dispersion
            )
        }
        let specularLayer = try specular.map { source in
            try Specular(
                factor: factors.specularFactor ?? source.factor, colorFactor: factors.specularColorFactor ?? source.colorFactor,
                factorTexture: source.factorTexture, colorTexture: source.colorTexture
            )
        }
        let sheenLayer = try sheen.map { source in
            try Sheen(
                colorFactor: factors.sheenColorFactor ?? source.colorFactor, roughness: factors.sheenRoughness ?? source.roughness,
                colorTexture: source.colorTexture, roughnessTexture: source.roughnessTexture
            )
        }
        let anisotropyLayer = try anisotropy.map { source in
            try Anisotropy(
                strength: factors.anisotropyStrength ?? source.strength, rotation: factors.anisotropyRotation ?? source.rotation,
                texture: source.texture
            )
        }
        let iridescenceLayer = try iridescence.map { source in
            try Iridescence(
                factor: factors.iridescenceFactor ?? source.factor, indexOfRefraction: source.indexOfRefraction,
                minimumThickness: source.minimumThickness, maximumThickness: source.maximumThickness,
                factorTexture: source.factorTexture, thicknessTexture: source.thicknessTexture
            )
        }
        var textures: [String: any MTLTexture] = [:]
        for (declaration, texture) in zip(surfaceGraph?.textures ?? [], surfaceTextures) { textures[declaration.name] = texture }
        let baseBinding = maps[0]
        return try PhysicallyBasedMaterial(
            baseColorTexture: baseColorTexture,
            metallic: factors.metallic,
            roughness: factors.roughness,
            directionToLight: SIMD3(lightParameters.x, lightParameters.y, lightParameters.z),
            lightIntensity: lightParameters.w,
            ambientIntensity: settings.z,
            receivesShadows: receivesShadows,
            baseColorFactor: factors.baseColorFactor,
            baseColorSampler: baseBinding?.sampler,
            metallicRoughnessTexture: rebind(1),
            normalTexture: rebind(2),
            normalScale: factors.normalScale ?? coreMaps.normalScale,
            occlusionTexture: rebind(3),
            occlusionStrength: factors.occlusionStrength ?? (maps[3] == nil ? 1 : coreMaps.occlusionStrength),
            emissiveTexture: rebind(4),
            emissiveFactor: factors.emissiveFactor,
            alphaMode: mode,
            isDoubleSided: isDoubleSided,
            environment: environment,
            baseColorTextureCoordinateSet: baseBinding?.textureCoordinateSet ?? .primary,
            clearcoat: coat,
            transmission: optical,
            surfaceGraph: surfaceOverride?.graph ?? surfaceGraph,
            surfaceUniforms: surfaceOverride == nil ? surfaceUniforms : surfaceOverride?.uniforms,
            surfaceTextures: surfaceOverride?.textures ?? textures,
            baseColorTextureTransform: factors.textureTransforms[.baseColor] ?? baseColorTextureTransform,
            emissiveStrength: factors.emissiveStrength,
            specular: specularLayer,
            sheen: sheenLayer,
            anisotropy: anisotropyLayer,
            iridescence: iridescenceLayer,
            shadingModel: shadingModel,
            baseColorSampling: baseBinding?.sampling,
            alphaTexture: alphaTexture,
            diffuseTransmission: diffuseTransmission,
            diffuseRoughness: diffuseRoughness,
            volumeScatter: volumeScatter,
            subsurface: subsurface,
            bumpMap: bumpMap,
            lightMap: lightMap,
            lightMapIntensity: lightMapIntensity,
            retroreflectivity: retroreflectivity,
            unlitUsesGraphEmission: unlitUsesGraphEmission
        )
    }
}
