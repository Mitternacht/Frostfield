import CoreGraphics
import Foundation
import ImageIO
import ModelIO
import simd

/// Converts ModelIO and MTL materials into `GLTFScene.Material` records with encoded images.
struct ModelIOMaterialImporter {
    typealias Warning = ModelIOSceneLoader.Warning

    struct MapReference: Hashable {
        /// A resolved file path, `package|entry` for a USDZ entry, or `memory|id` for a ModelIO texture.
        let location: String
        let transform: PhysicallyBasedMaterial.TextureTransform
        let clamps: Bool
    }

    /// A material's factors and maps before channel packing.
    struct Inputs: Hashable {
        var name: String
        var baseColor = SIMD3<Float>(repeating: 1)
        var baseColorMap: MapReference?
        var opacity: Float = 1
        /// Red channel, or the base map's alpha when both name the same file.
        var opacityMap: MapReference?
        var metallic: Float = 0
        var metallicMap: MapReference?
        var roughness: Float = 1
        var roughnessMap: MapReference?
        var normalMap: MapReference?
        var normalScale: Float = 1
        var bumpMap: MapReference?
        var bumpScale: Float = 1
        var occlusionMap: MapReference?
        var emissive = SIMD3<Float>.zero
        var emissiveMap: MapReference?
        var clearcoat: Float = 0
        var clearcoatRoughness: Float = 0
        var indexOfRefraction: Float = 1.5
        /// A file shared by the metallic and roughness inputs uses glTF's packed layout:
        /// green roughness and blue metallic. Otherwise each map's red channel is used.
        var sharedMapIsPacked = false
        /// USD authors `doubleSided` on the mesh, so one ModelIO material can yield both variants.
        var isDoubleSided = false
    }

    let assetURL: URL
    let isUSD: Bool
    private let directory: String
    private(set) var materials: [GLTFScene.Material] = []
    private(set) var warnings: [Warning] = []
    private var library: [String: (material: WavefrontMaterialLibrary.Material, directory: URL)] = [:]
    private var byObject: [MaterialUse: Int] = [:]
    /// Each ModelIO material's inputs, read once so its warnings are reported once.
    private var inputsByObject: [ObjectIdentifier: Inputs] = [:]
    private var byInputs: [Inputs: Int] = [:]
    private var images: [String: GLTFScene.Image] = [:]
    private var failedImages: Set<String> = []
    private var memoryTextures: [String: Data] = [:]
    /// The memory-mapped USDZ package and its directory, read on first use.
    private var package: (archive: Data, records: [String: USDZArchive.Record])?
    private var imageCount = 0
    private var textureCount = 0
    private static let maximumImageBytes = 64 * 1_024 * 1_024

    init(assetURL: URL, isUSD: Bool) {
        self.assetURL = assetURL
        self.isUSD = isUSD
        directory = assetURL.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL.path
    }

    mutating func importWavefront(_ library: WavefrontMaterialLibrary) {
        let base = assetURL.deletingLastPathComponent()
        for material in library.materials.values.sorted(by: { $0.name < $1.name }) where !material.name.isEmpty {
            let converted = inputs(fromMTL: material, directory: base)
            materials.append(build(converted))
        }
    }

    /// Parses the MTL libraries an OBJ names. Missing libraries are reported as warnings.
    /// OBJ and MTL text that is not valid UTF-8 is read as ISO Latin-1, which accepts every byte.
    mutating func loadMaterialLibraries(forOBJ url: URL) {
        guard let text = Self.text(at: url) else {
            warnings.append(.materialInputIgnored(material: url.lastPathComponent, input: "mtllib"))
            return
        }
        for name in WavefrontMaterialLibrary.libraries(inOBJ: text) {
            guard let file = resolveFile(name, relativeTo: url.deletingLastPathComponent()),
                  let contents = Self.text(at: URL(fileURLWithPath: file)) else {
                warnings.append(.materialInputIgnored(material: name, input: "mtllib"))
                continue
            }
            let parsed = WavefrontMaterialLibrary(contents)
            for (key, material) in parsed.materials {
                library[key] = (material, URL(fileURLWithPath: file).deletingLastPathComponent())
            }
        }
    }

    private struct MaterialUse: Hashable {
        let material: ObjectIdentifier
        let isDoubleSided: Bool
    }

    /// Returns the material index for a ModelIO material on a single- or double-sided mesh,
    /// importing it on first use.
    mutating func index(for material: MDLMaterial, doubleSided: Bool = false) -> Int {
        let use = MaterialUse(material: ObjectIdentifier(material), isDoubleSided: doubleSided)
        if let existing = byObject[use] { return existing }
        var inputs: Inputs
        if let existing = inputsByObject[use.material] {
            inputs = existing
        } else if let entry = library[material.name] {
            inputs = self.inputs(fromMTL: entry.material, directory: entry.directory)
        } else {
            inputs = self.inputs(from: material)
        }
        inputsByObject[use.material] = inputs
        inputs.isDoubleSided = doubleSided
        let index: Int
        if let existing = byInputs[inputs] {
            index = existing
        } else {
            index = materials.count
            materials.append(build(inputs))
            byInputs[inputs] = index
        }
        byObject[use] = index
        return index
    }

    private static func text(at url: URL) -> String? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        return String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1)
    }

    // MARK: Input extraction

    private mutating func inputs(from material: MDLMaterial) -> Inputs {
        var inputs = Inputs(name: material.name)
        inputs.sharedMapIsPacked = isUSD
        func property(_ names: [String], _ semantic: MDLMaterialSemantic?) -> MDLMaterialProperty? {
            for name in names {
                if let property = material.propertyNamed(name) { return property }
            }
            return semantic.flatMap { material.property(with: $0) }
        }
        let name = material.name

        if let base = property(["diffuseColor"], .baseColor) {
            if let map = mapReference(base, material: name) {
                inputs.baseColorMap = map
            } else if let color = Self.color(base) {
                inputs.baseColor = color
            }
        }
        if let opacity = property(["opacity"], .opacity) {
            if let map = mapReference(opacity, material: name) { inputs.opacityMap = map } else if let value = Self.scalar(opacity) { inputs.opacity = value }
        }
        if let metallic = property(["metallic"], .metallic) {
            if let map = mapReference(metallic, material: name) { inputs.metallicMap = map } else if let value = Self.scalar(metallic) { inputs.metallic = value }
        }
        if let roughness = property(["roughness"], .roughness) {
            if let map = mapReference(roughness, material: name) { inputs.roughnessMap = map } else if let value = Self.scalar(roughness) { inputs.roughness = value }
        }
        if let normal = property(["normal"], .tangentSpaceNormal) { inputs.normalMap = mapReference(normal, material: name) }
        if isUSD, let occlusion = property(["occlusion"], nil) { inputs.occlusionMap = mapReference(occlusion, material: name) }
        if let emission = property(["emissiveColor"], .emission) {
            if let map = mapReference(emission, material: name) {
                inputs.emissiveMap = map
                inputs.emissive = SIMD3(repeating: 1)
            } else if let color = Self.color(emission) {
                inputs.emissive = color
            }
        }
        if let coat = property(["clearcoat"], .clearcoat), let value = Self.scalar(coat), value > 0 {
            inputs.clearcoat = value
            if let roughness = material.propertyNamed("clearcoatRoughness").flatMap(Self.scalar) {
                inputs.clearcoatRoughness = roughness
            } else if let gloss = material.property(with: .clearcoatGloss).flatMap(Self.scalar) {
                inputs.clearcoatRoughness = 1 - gloss
            }
        }
        if isUSD, let ior = material.propertyNamed("ior").flatMap(Self.scalar), ior >= 1, ior.isFinite {
            inputs.indexOfRefraction = ior
        }
        if isUSD, let workflow = material.propertyNamed("useSpecularWorkflow").flatMap(Self.scalar), workflow != 0 {
            warnings.append(.materialInputIgnored(material: name, input: "useSpecularWorkflow"))
        }
        return inputs
    }

    private mutating func inputs(fromMTL source: WavefrontMaterialLibrary.Material, directory: URL) -> Inputs {
        var inputs = Inputs(name: source.name)
        func map(_ reference: WavefrontMaterialLibrary.TextureReference?) -> MapReference? {
            guard let reference else { return nil }
            guard let file = resolveFile(reference.path, relativeTo: directory) else {
                warnings.append(.textureNotLoaded(material: source.name, path: reference.path))
                return nil
            }
            return MapReference(location: file, transform: reference.transform, clamps: reference.clamps)
        }
        inputs.baseColor = source.diffuse ?? SIMD3(repeating: 1)
        inputs.baseColorMap = map(source.diffuseMap)
        inputs.opacity = source.dissolve ?? 1
        inputs.opacityMap = map(source.dissolveMap)
        if let roughness = source.roughness {
            inputs.roughness = roughness
        } else if let exponent = source.specularExponent, exponent >= 0 {
            // Blinn-Phong exponent to Beckmann alpha (Walter et al. 2007), whose square root is glTF roughness.
            inputs.roughness = pow(2 / (exponent + 2), 0.25)
        }
        inputs.roughnessMap = map(source.roughnessMap)
        inputs.metallic = source.metallic ?? 0
        inputs.metallicMap = map(source.metallicMap)
        if let normal = source.normalMap {
            inputs.normalMap = map(normal)
            inputs.normalScale = normal.bumpMultiplier
        }
        if let bump = source.bumpMap {
            inputs.bumpMap = map(bump)
            inputs.bumpScale = bump.bumpMultiplier
        }
        inputs.emissiveMap = map(source.emissiveMap)
        inputs.emissive = source.emissive ?? (inputs.emissiveMap == nil ? .zero : SIMD3(repeating: 1))
        inputs.clearcoat = source.clearcoat ?? 0
        inputs.clearcoatRoughness = source.clearcoatRoughness ?? 0
        // Ni is a refraction density; values at or below one would remove dielectric reflection.
        if let ior = source.indexOfRefraction, ior > 1 { inputs.indexOfRefraction = ior }
        for statement in Set(source.ignored).sorted() {
            warnings.append(.materialInputIgnored(material: source.name, input: statement))
        }
        return inputs
    }

    private static func scalar(_ property: MDLMaterialProperty) -> Float? {
        let value: Float
        switch property.type {
        case .float: value = property.floatValue
        case .float2: value = property.float2Value.x
        case .float3: value = property.float3Value.x
        case .float4: value = property.float4Value.x
        default: return nil
        }
        return value.isFinite ? value : nil
    }

    private static func color(_ property: MDLMaterialProperty) -> SIMD3<Float>? {
        let value: SIMD3<Float>
        switch property.type {
        case .float: value = SIMD3(repeating: property.floatValue)
        case .float3: value = property.float3Value
        case .float4: value = SIMD3(property.float4Value.x, property.float4Value.y, property.float4Value.z)
        case .color:
            guard let color = property.color, let components = color.components else { return nil }
            let floats = components.map(Float.init)
            value = floats.count >= 3 ? SIMD3(floats[0], floats[1], floats[2]) : SIMD3(repeating: floats.first ?? 1)
        default: return nil
        }
        return value.x.isFinite && value.y.isFinite && value.z.isFinite ? value : nil
    }

    /// A texture input's location: a path string, a URL, or a ModelIO texture.
    private mutating func mapReference(_ property: MDLMaterialProperty, material: String) -> MapReference? {
        var reference: String?
        switch property.type {
        case .string: reference = property.stringValue
        case .URL: reference = property.urlValue?.path
        case .texture:
            if let texture = property.textureSamplerValue?.texture {
                if let url = (texture as? MDLURLTexture)?.url {
                    reference = url.path
                } else if let image = texture.imageFromTexture()?.takeUnretainedValue(), let data = Self.png(image) {
                    let key = "memory|\(memoryTextures.count)"
                    memoryTextures[key] = data
                    return MapReference(location: key, transform: .identity, clamps: false)
                }
            }
        default: return nil
        }
        guard let reference, !reference.isEmpty else { return nil }
        let location: String?
        if let entry = packageEntry(reference) {
            // An entry of a package nested inside this one is not read.
            location = entry.contains("[") ? nil : "package|\(entry)"
        } else {
            location = resolveFile(reference, relativeTo: assetURL.deletingLastPathComponent())
        }
        guard let location else {
            warnings.append(.textureNotLoaded(material: material, path: reference))
            return nil
        }
        return MapReference(location: location, transform: .identity, clamps: false)
    }

    /// The entry of a `package.usdz[entry]` reference into this asset's own package. The package
    /// path may itself contain brackets, so each `[` is tried as the separator.
    private func packageEntry(_ reference: String) -> String? {
        guard isUSD, reference.hasSuffix("]") else { return nil }
        let ownPackage = assetURL.resolvingSymlinksInPath().standardizedFileURL.path
        for open in reference.indices where reference[open] == "[" {
            let packagePath = URL(fileURLWithPath: String(reference[..<open])).resolvingSymlinksInPath().standardizedFileURL.path
            if packagePath == ownPackage {
                return String(reference[reference.index(after: open)..<reference.index(before: reference.endIndex)])
            }
        }
        return nil
    }

    /// A regular file inside the asset's folder, after resolving symbolic links.
    private func resolveFile(_ path: String, relativeTo base: URL) -> String? {
        let url = path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path)
        let resolved = url.resolvingSymlinksInPath().standardizedFileURL
        guard resolved.path.hasPrefix(directory + "/"),
              (try? resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return nil }
        return resolved.path
    }

    // MARK: Material assembly

    private mutating func build(_ inputs: Inputs) -> GLTFScene.Material {
        let name = inputs.name
        var baseTexture: GLTFScene.Texture?
        if let map = inputs.baseColorMap {
            if let opacity = inputs.opacityMap, opacity.location != map.location {
                baseTexture = packed([.channel(map, 0), .channel(map, 1), .channel(map, 2), .channel(opacity, 0)], material: name)
            } else {
                baseTexture = texture(map, material: name)
            }
        } else if let opacity = inputs.opacityMap {
            baseTexture = packed([.constant(255), .constant(255), .constant(255), .channel(opacity, 0)], material: name)
        }

        var metallicFactor = inputs.metallic, roughnessFactor = inputs.roughness
        var surfaceTexture: GLTFScene.Texture?
        if let metallic = inputs.metallicMap, let roughness = inputs.roughnessMap,
           metallic.location == roughness.location, inputs.sharedMapIsPacked {
            surfaceTexture = texture(metallic, material: name)
            metallicFactor = 1
            roughnessFactor = 1
        } else if inputs.metallicMap != nil || inputs.roughnessMap != nil {
            let green: PackedChannel = inputs.roughnessMap.map { .channel($0, 0) } ?? .constant(255)
            let blue: PackedChannel = inputs.metallicMap.map { .channel($0, 0) } ?? .constant(255)
            surfaceTexture = packed([.constant(255), green, blue, .constant(255)], material: name)
            if inputs.roughnessMap != nil { roughnessFactor = 1 }
            if inputs.metallicMap != nil { metallicFactor = 1 }
        }

        let emissive = simd_max(inputs.emissive, .zero)
        let strength = max(1, emissive.max())
        let usesAlpha = inputs.opacity < 1 || inputs.opacityMap != nil
        func unit(_ value: Float) -> Float { min(max(value, 0), 1) }
        return GLTFScene.Material(
            sourceIndex: materials.count, name: name,
            baseColorFactor: SIMD4(simd_clamp(inputs.baseColor, .zero, SIMD3(repeating: 1)), unit(inputs.opacity)),
            metallicFactor: unit(metallicFactor), roughnessFactor: unit(roughnessFactor),
            baseColorTexture: baseTexture, metallicRoughnessTexture: surfaceTexture,
            normalTexture: inputs.normalMap.flatMap { texture($0, material: name) }, normalScale: inputs.normalScale.isFinite ? inputs.normalScale : 1,
            occlusionTexture: inputs.occlusionMap.flatMap { texture($0, material: name) }, occlusionStrength: 1,
            emissiveTexture: inputs.emissiveMap.flatMap { texture($0, material: name) }, emissiveFactor: emissive / strength,
            alphaMode: usesAlpha ? .blend : .opaque, isDoubleSided: inputs.isDoubleSided,
            clearcoat: inputs.clearcoat > 0 ? GLTFScene.Material.Clearcoat(
                factor: unit(inputs.clearcoat), roughness: unit(inputs.clearcoatRoughness), factorTexture: nil,
                roughnessTexture: nil, normalTexture: nil, normalScale: 1
            ) : nil,
            transmission: nil, volume: nil, indexOfRefraction: inputs.indexOfRefraction, emissiveStrength: strength,
            specular: nil, sheen: nil, anisotropy: nil, iridescence: nil, dispersion: 0, isUnlit: false,
            diffuseTransmission: nil, diffuseRoughness: nil, volumeScatter: nil,
            bumpMap: inputs.bumpMap.flatMap { texture($0, material: name) }.map {
                .init(texture: $0, scale: inputs.bumpScale)
            }
        )
    }

    private enum PackedChannel {
        case constant(UInt8)
        case channel(MapReference, Int)

        var map: MapReference? {
            if case .channel(let map, _) = self { return map }
            return nil
        }
    }

    private mutating func texture(_ map: MapReference, material: String) -> GLTFScene.Texture? {
        guard let image = image(map.location, material: material) else { return nil }
        return makeTexture(image, placement: map)
    }

    private mutating func makeTexture(_ image: GLTFScene.Image, placement: MapReference) -> GLTFScene.Texture {
        let wrap: GLTFScene.Sampling.Wrap = placement.clamps ? .clampToEdge : .repeat
        defer { textureCount += 1 }
        return GLTFScene.Texture(
            sourceIndex: textureCount, image: image,
            sampling: GLTFScene.Sampling(magnification: .linear, minification: .linearMipmapLinear, wrapS: wrap, wrapT: wrap),
            textureCoordinateSet: .primary, transform: placement.transform, fallbackImage: nil
        )
    }

    /// Builds an RGBA8 PNG from constant and image channels at the largest source size.
    private mutating func packed(_ channels: [PackedChannel], material: String) -> GLTFScene.Texture? {
        let maps = channels.compactMap(\.map)
        guard let placement = maps.first else { return nil }
        if maps.contains(where: { $0.transform != placement.transform || $0.clamps != placement.clamps }) {
            warnings.append(.packedMapPlacementMerged(material: material))
        }
        let key = "packed|" + channels.map { channel -> String in
            switch channel {
            case .constant(let value): "c\(value)"
            case .channel(let map, let index): "\(map.location)#\(index)"
            }
        }.joined(separator: "|")
        if let existing = images[key] { return makeTexture(existing, placement: placement) }
        guard !failedImages.contains(key) else { return nil }

        var decoded: [String: (width: Int, height: Int, data: Data)] = [:]
        for map in maps where decoded[map.location] == nil {
            guard let source = encodedImage(map.location, material: material),
                  let size = ImageChannels.size(of: source) else {
                failedImages.insert(key)
                return nil
            }
            decoded[map.location] = (size.width, size.height, source)
        }
        let width = decoded.values.map(\.width).max() ?? 1, height = decoded.values.map(\.height).max() ?? 1
        var planes: [String: [UInt8]] = [:]
        for (location, source) in decoded {
            guard let pixels = ImageChannels.rgba8(source.data, width: width, height: height) else {
                warnings.append(.textureNotLoaded(material: material, path: location))
                failedImages.insert(key)
                return nil
            }
            planes[location] = pixels
        }
        var output = [UInt8](repeating: 0, count: width * height * 4)
        for (lane, channel) in channels.enumerated() {
            switch channel {
            case .constant(let value):
                for pixel in 0..<(width * height) { output[pixel * 4 + lane] = value }
            case .channel(let map, let index):
                let plane = planes[map.location]!
                for pixel in 0..<(width * height) { output[pixel * 4 + lane] = plane[pixel * 4 + index] }
            }
        }
        guard let png = ImageChannels.png(output, width: width, height: height) else {
            failedImages.insert(key)
            return nil
        }
        let image = GLTFScene.Image(sourceIndex: imageCount, name: "packed", mimeType: "image/png", data: png)
        imageCount += 1
        images[key] = image
        return makeTexture(image, placement: placement)
    }

    /// The image at a location, kept in a format `TextureLoader` decodes.
    private mutating func image(_ location: String, material: String) -> GLTFScene.Image? {
        if let existing = images[location] { return existing }
        guard !failedImages.contains(location), var data = encodedImage(location, material: material) else { return nil }
        var mimeType = ImageSignature(data)?.mimeType
        if mimeType == nil || ImageSignature(data)?.isDecodable == false {
            guard let png = ImageChannels.transcode(data) else {
                warnings.append(.textureNotLoaded(material: material, path: location))
                failedImages.insert(location)
                return nil
            }
            warnings.append(.imageTranscoded(path: location))
            data = png
            mimeType = "image/png"
        }
        let name = location.split(separator: "/").last.map(String.init)
        let image = GLTFScene.Image(sourceIndex: imageCount, name: name, mimeType: mimeType!, data: data)
        imageCount += 1
        images[location] = image
        return image
    }

    private mutating func encodedImage(_ location: String, material: String) -> Data? {
        let data: Data?
        if location.hasPrefix("memory|") {
            data = memoryTextures[location]
        } else if location.hasPrefix("package|") {
            if package == nil, let archive = try? Data(contentsOf: assetURL, options: .mappedIfSafe),
               let records = try? USDZArchive.records(in: archive) {
                package = (archive, Dictionary(records.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first }))
            }
            let entry = String(location.dropFirst("package|".count))
            data = package.flatMap { package in
                package.records[entry].flatMap { try? USDZArchive.file($0, in: package.archive) }
            }
        } else {
            let size = (try? URL(fileURLWithPath: location).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            data = size <= Self.maximumImageBytes ? try? Data(contentsOf: URL(fileURLWithPath: location)) : nil
        }
        guard let data, !data.isEmpty, data.count <= Self.maximumImageBytes else {
            if !failedImages.contains(location) {
                warnings.append(.textureNotLoaded(material: material, path: location))
                failedImages.insert(location)
            }
            return nil
        }
        return data
    }

    private static func png(_ image: CGImage) -> Data? {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? output as Data : nil
    }
}

/// Decodes images to 8-bit channels without color conversion and encodes RGBA8 PNG.
enum ImageChannels {
    static func size(of data: Data) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
              let width = properties[kCGImagePropertyPixelWidth as String] as? Int,
              let height = properties[kCGImagePropertyPixelHeight as String] as? Int,
              (1...8_192).contains(width), (1...8_192).contains(height) else { return nil }
        return (width, height)
    }

    /// Straight-alpha RGBA8 at the requested size. Gray and RGB images are drawn in their own
    /// color space, so stored code values pass through unconverted; resizing filters them.
    static func rgba8(_ data: Data, width: Int, height: Int) -> [UInt8]? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        if image.colorSpace?.model == .monochrome, let space = image.colorSpace,
           let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                   space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue) {
            context.interpolationQuality = .high
            context.setBlendMode(.copy)
            context.draw(image, in: rect)
            guard let gray = context.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
            var output = [UInt8](repeating: 255, count: width * height * 4)
            for row in 0..<height {
                for column in 0..<width {
                    let value = gray[row * context.bytesPerRow + column]
                    let index = (row * width + column) * 4
                    output[index] = value
                    output[index + 1] = value
                    output[index + 2] = value
                }
            }
            return output
        }
        let space = image.colorSpace?.model == .rgb ? image.colorSpace! : CGColorSpace(name: CGColorSpace.sRGB)!
        var output = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = output.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: space, bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.setBlendMode(.copy)
            context.draw(image, in: rect)
            return true
        }
        guard drawn else { return nil }
        for index in stride(from: 0, to: output.count, by: 4) {
            let alpha = Int(output[index + 3])
            guard alpha < 255 else { continue }
            for channel in index..<(index + 3) {
                output[channel] = alpha == 0 ? 0 : UInt8(min(255, (Int(output[channel]) * 255 + alpha / 2) / alpha))
            }
        }
        return output
    }

    static func png(_ pixels: [UInt8], width: Int, height: Int) -> Data? {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else { return nil }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? output as Data : nil
    }

    /// Re-encodes the first image of any ImageIO format as PNG.
    static func transcode(_ data: Data) -> Data? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? output as Data : nil
    }
}
