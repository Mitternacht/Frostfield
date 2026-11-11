import CoreGraphics
import Foundation
import Metal
import simd
import Synchronization

/// A color image that a spot light projects, also called a gobo or cookie.
///
/// Each texel is a linear RGB transmittance from zero to one that filters the light
/// leaving in its direction. The image spans the square that circumscribes the
/// spot's outer cone at unit distance, `[-tan(outer), tan(outer)]²` in the light's
/// local X and Y, as three.js `SpotLight.map` does through the spot's shadow camera.
/// Its top row lies toward local +Y and its left column toward local −X, so it reads
/// unmirrored on a wall when seen from behind the light. The spot cone still shapes
/// the beam; texels outside the cone receive no light.
///
/// Both renderers filter the image bilinearly with clamped edges and no mipmaps.
/// Lights compare cookies by contents.
public struct LightCookie: Equatable, Sendable {
    /// The largest width or height.
    public static let maximumDimension = 2_048

    /// Cookie width in texels.
    public var width: Int { storage.width }
    /// Cookie height in texels.
    public var height: Int { storage.height }
    /// Row-major linear RGB transmittance, top row first.
    public var transmittance: [SIMD3<Float>] { storage.pixels }

    let storage: Storage

    /// Uses row-major linear RGB transmittance from the top-left texel, with every
    /// component from 0 to 1, and dimensions from 1 to 2,048.
    public init(width: Int, height: Int, transmittance: [SIMD3<Float>]) throws {
        guard (1...Self.maximumDimension).contains(width), (1...Self.maximumDimension).contains(height),
              transmittance.count == width * height else { throw Failure.dimensions }
        guard transmittance.allSatisfy({
            (0...1).contains($0.x) && (0...1).contains($0.y) && (0...1).contains($0.z)
        }) else {
            throw Failure.values
        }
        storage = Storage(width: width, height: height, pixels: transmittance)
    }

    /// Converts an image to linear sRGB transmittance. Alpha multiplies the
    /// linear color, so transparent texels block light.
    public init(cgImage image: CGImage) throws {
        let width = image.width
        let height = image.height
        guard (1...Self.maximumDimension).contains(width), (1...Self.maximumDimension).contains(height) else {
            throw Failure.dimensions
        }
        let bitmapInfo = CGBitmapInfo.byteOrder16Little.rawValue | CGBitmapInfo.floatComponents.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let space = CGColorSpace(name: CGColorSpace.linearSRGB),
              let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 16, bytesPerRow: 0, space: space,
                bitmapInfo: bitmapInfo
              ) else { throw Failure.image }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { throw Failure.image }
        let rowStride = context.bytesPerRow / MemoryLayout<SIMD4<Float16>>.stride
        let source = data.bindMemory(to: SIMD4<Float16>.self, capacity: rowStride * height)
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(width * height)
        // CGContext's memory starts at the top row and can pad each row.
        for y in 0..<height {
            for x in 0..<width {
                let texel = source[y * rowStride + x]
                let alpha = simd_clamp(Float(texel.w), 0, 1)
                let color = SIMD3(Float(texel.x), Float(texel.y), Float(texel.z))
                pixels.append(simd_clamp(color, .zero, SIMD3(repeating: alpha)))
            }
        }
        storage = Storage(width: width, height: height, pixels: pixels)
    }

    /// Bilinear transmittance at normalized coordinates with clamped edges.
    /// A NaN coordinate returns zero transmittance.
    public func transmittance(u: Float, v: Float) -> SIMD3<Float> {
        Self.interpolate(u: u, v: v, width: width, height: height) { x, y in
            storage.pixels[y * width + x]
        }
    }

    /// One renderer array texel, resampled to the array size and stored in linear RGBA16Float.
    func packedTexel(x: Int, y: Int, width: Int, height: Int) -> SIMD4<Float16> {
        let value: SIMD3<Float>
        if self.width == width, self.height == height {
            value = storage.pixels[y * width + x]
        } else {
            value = transmittance(u: (Float(x) + 0.5) / Float(width), v: (Float(y) + 0.5) / Float(height))
        }
        return SIMD4(Float16(value.x), Float16(value.y), Float16(value.z), 1)
    }

    /// Reads the same four packed texels the renderer filters, without creating an array texture.
    func packedTransmittance(u: Float, v: Float, width: Int, height: Int) -> SIMD3<Float> {
        Self.interpolate(u: u, v: v, width: width, height: height) { x, y in
            let value = packedTexel(x: x, y: y, width: width, height: height)
            return SIMD3(Float(value.x), Float(value.y), Float(value.z))
        }
    }

    private static func interpolate(
        u: Float, v: Float, width: Int, height: Int, texel: (Int, Int) -> SIMD3<Float>
    ) -> SIMD3<Float> {
        guard !u.isNaN, !v.isNaN else { return .zero }
        let x = min(max(u * Float(width) - 0.5, 0), Float(width - 1))
        let y = min(max(v * Float(height) - 0.5, 0), Float(height - 1))
        let x0 = Int(x)
        let y0 = Int(y)
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = x - Float(x0)
        let ty = y - Float(y0)
        let top = simd_mix(texel(x0, y0), texel(x1, y0), SIMD3(repeating: tx))
        let bottom = simd_mix(texel(x0, y1), texel(x1, y1), SIMD3(repeating: tx))
        return simd_mix(top, bottom, SIMD3(repeating: ty))
    }

    public static func == (left: LightCookie, right: LightCookie) -> Bool {
        left.storage === right.storage
            || (left.width == right.width && left.height == right.height && left.storage.pixels == right.storage.pixels)
    }

    public enum Failure: LocalizedError, Equatable {
        case dimensions
        case values
        case image

        public var errorDescription: String? {
            switch self {
            case .dimensions: "A light cookie needs 1 to 2,048 texels per side and one value for each."
            case .values: "Light cookie transmittance components must be from 0 to 1."
            case .image: "Could not draw the image into a linear sRGB light cookie."
            }
        }
    }

    // Immutable after initialization; shared between copies and used as a cache identity.
    final class Storage: Sendable {
        let width: Int
        let height: Int
        let pixels: [SIMD3<Float>]

        init(width: Int, height: Int, pixels: [SIMD3<Float>]) {
            self.width = width
            self.height = height
            self.pixels = pixels
        }

    }
}

/// The GPU tables of the profiles and cookies that one light list uses, shared by
/// raster PBR and the path tracer. Profiles fill an R16Float array of
/// `IESProfile.tableWidth × tableHeight` layers; cookies fill an RGBA16Float array
/// at the largest cookie size, with smaller cookies resampled bilinearly. Equal
/// profiles and cookies share a layer. A renderer caches its most recent set.
final class LightTextureSet: @unchecked Sendable {
    // Immutable after creation. Command encoders only read the textures.
    let profiles: any MTLTexture
    let cookies: any MTLTexture
    /// One plus each input's profile layer, or zero without one.
    let profileLayers: [Float]
    /// One plus each input's cookie layer, or zero without one.
    let cookieLayers: [Float]

    private init(profiles: any MTLTexture, cookies: any MTLTexture, profileLayers: [Float], cookieLayers: [Float]) {
        self.profiles = profiles
        self.cookies = cookies
        self.profileLayers = profileLayers
        self.cookieLayers = cookieLayers
    }

    private struct Entry {
        let device: ObjectIdentifier
        // Holding the storages keeps their identifiers from being reused while cached.
        let profiles: [IESProfile.Storage?]
        let cookies: [LightCookie.Storage?]
        let set: LightTextureSet
    }

    /// One immutable set per owner. Lighting edits replace the previous set
    /// without leaving it resident in a process-wide cache.
    final class Cache {
        private let entry = Mutex<Entry?>(nil)

        func make(device: any MTLDevice, profiles: [IESProfile?], cookies: [LightCookie?]) throws -> LightTextureSet {
            let profileStorages = profiles.map { $0?.storage }
            let cookieStorages = cookies.map { $0?.storage }
            let deviceID = ObjectIdentifier(device)
            return try entry.withLock { current in
                if let existing = current, existing.device == deviceID,
                   existing.profiles.count == profileStorages.count, existing.cookies.count == cookieStorages.count,
                   zip(existing.profiles, profileStorages).allSatisfy({ $0 === $1 }),
                   zip(existing.cookies, cookieStorages).allSatisfy({ $0 === $1 }) {
                    return existing.set
                }
                let set = try LightTextureSet.build(device: device, profiles: profiles, cookies: cookies)
                current = Entry(device: deviceID, profiles: profileStorages, cookies: cookieStorages, set: set)
                return set
            }
        }
    }

    func retainIfNeeded(on commandBuffer: any MTLCommandBuffer) {
        guard !commandBuffer.retainedReferences else { return }
        commandBuffer.addCompletedHandler { [self] _ in withExtendedLifetime(self) {} }
    }

    private static func build(device: any MTLDevice, profiles: [IESProfile?], cookies: [LightCookie?]) throws -> LightTextureSet {
        var uniqueProfiles: [IESProfile] = []
        let profileLayers = profiles.map { profile -> Float in
            guard let profile else { return 0 }
            if let index = uniqueProfiles.firstIndex(of: profile) { return Float(index + 1) }
            uniqueProfiles.append(profile)
            return Float(uniqueProfiles.count)
        }
        var uniqueCookies: [LightCookie] = []
        let cookieLayers = cookies.map { cookie -> Float in
            guard let cookie else { return 0 }
            if let index = uniqueCookies.firstIndex(of: cookie) { return Float(index + 1) }
            uniqueCookies.append(cookie)
            return Float(uniqueCookies.count)
        }

        let profileTexture: any MTLTexture
        if uniqueProfiles.isEmpty {
            profileTexture = try placeholder(device: device, format: .r16Float, bytes: [Float16(0)])
        } else {
            let width = IESProfile.tableWidth
            let height = IESProfile.tableHeight
            let texture = try makeArray(device: device, format: .r16Float, width: width, height: height, layers: uniqueProfiles.count)
            for (layer, profile) in uniqueProfiles.enumerated() {
                let halves = profile.storage.table.map(Float16.init)
                halves.withUnsafeBytes { bytes in
                    texture.replace(
                        region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, slice: layer,
                        withBytes: bytes.baseAddress!, bytesPerRow: width * 2, bytesPerImage: width * height * 2
                    )
                }
            }
            texture.label = "Frostfield light profiles"
            profileTexture = texture
        }

        let cookieTexture: any MTLTexture
        if uniqueCookies.isEmpty {
            cookieTexture = try placeholder(device: device, format: .rgba16Float, bytes: [Float16](repeating: 1, count: 4))
        } else {
            let width = uniqueCookies.map(\.width).max()!
            let height = uniqueCookies.map(\.height).max()!
            let texture = try makeArray(device: device, format: .rgba16Float, width: width, height: height, layers: uniqueCookies.count)
            var halves = [SIMD4<Float16>](repeating: .zero, count: width * height)
            for (layer, cookie) in uniqueCookies.enumerated() {
                for y in 0..<height {
                    for x in 0..<width {
                        halves[y * width + x] = cookie.packedTexel(x: x, y: y, width: width, height: height)
                    }
                }
                halves.withUnsafeBytes { bytes in
                    texture.replace(
                        region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, slice: layer,
                        withBytes: bytes.baseAddress!, bytesPerRow: width * 8, bytesPerImage: width * height * 8
                    )
                }
            }
            texture.label = "Frostfield light cookies"
            cookieTexture = texture
        }
        return LightTextureSet(
            profiles: profileTexture, cookies: cookieTexture, profileLayers: profileLayers, cookieLayers: cookieLayers
        )
    }

    static func makeArray(
        device: any MTLDevice, format: MTLPixelFormat, width: Int, height: Int, layers: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type2DArray
        descriptor.pixelFormat = format
        descriptor.width = width
        descriptor.height = height
        descriptor.arrayLength = layers
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw LightTextureFailure.allocation }
        return texture
    }

    private static func placeholder(device: any MTLDevice, format: MTLPixelFormat, bytes: [Float16]) throws -> any MTLTexture {
        let texture = try makeArray(device: device, format: format, width: 1, height: 1, layers: 1)
        bytes.withUnsafeBytes { raw in
            texture.replace(
                region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: 0,
                withBytes: raw.baseAddress!, bytesPerRow: raw.count, bytesPerImage: raw.count
            )
        }
        texture.label = "Frostfield light table placeholder"
        return texture
    }
}

/// Errors from preparing light profile and cookie textures.
public enum LightTextureFailure: LocalizedError, Equatable {
    case allocation

    public var errorDescription: String? {
        switch self {
        case .allocation: "Metal could not allocate the light profile or cookie textures."
        }
    }
}

extension MeshRenderer.PackedPunctualLight {
    /// Writes each point or spot light's table layers into its record and returns
    /// the tables to bind at fragment textures 25 and 26. For those kinds `axisU`
    /// holds the light's local +X and one plus its profile layer, and `axisV` holds
    /// one plus its cookie layer and the tangent of its outer cone; zero means none.
    static func applyTables(
        to lights: inout [Self], from resolved: [ResolvedPunctualLight],
        device: any MTLDevice, cache: LightTextureSet.Cache
    ) throws -> LightTextureSet {
        guard resolved.contains(where: { $0.light.iesProfile != nil || $0.light.cookie != nil }) else {
            return try cache.make(device: device, profiles: [], cookies: [])
        }
        let tables = try cache.make(
            device: device, profiles: resolved.map(\.light.iesProfile), cookies: resolved.map(\.light.cookie)
        )
        for index in lights.indices where resolved[index].light.iesProfile != nil || resolved[index].light.cookie != nil {
            let light = resolved[index]
            var scale: Float = 0
            if case .spot(_, _, let outer) = light.light.kind { scale = tan(outer) }
            lights[index].axisU = SIMD4(light.right, tables.profileLayers[index])
            lights[index].axisV = SIMD4(tables.cookieLayers[index], scale, 0, 0)
        }
        return tables
    }
}
