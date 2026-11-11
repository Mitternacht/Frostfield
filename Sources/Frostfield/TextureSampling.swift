import Foundation
import Metal
import Synchronization

/// A serializable description of a Metal sampler, matching `MTLSamplerDescriptor`.
///
/// Metal sampler states do not expose their settings, so `SceneArchive` can save a
/// material's sampler only when Frostfield knows how it was made. Samplers created
/// with `makeSamplerState(device:)`, or by glTF material preparation, record their
/// description; `recorded(for:)` returns it while the sampler is alive.
public struct TextureSampling: Hashable, Sendable, Codable {
    public enum Filter: String, Hashable, Sendable, Codable {
        case nearest, linear
    }

    public enum MipFilter: String, Hashable, Sendable, Codable {
        case notMipmapped, nearest, linear
    }

    public enum AddressMode: String, Hashable, Sendable, Codable {
        case clampToEdge, mirrorClampToEdge, `repeat`, mirrorRepeat, clampToZero, clampToBorderColor
    }

    public enum BorderColor: String, Hashable, Sendable, Codable {
        case transparentBlack, opaqueBlack, opaqueWhite
    }

    public enum CompareFunction: String, Hashable, Sendable, Codable {
        case never, less, equal, lessEqual, greater, notEqual, greaterEqual, always
    }

    public var minification: Filter
    public var magnification: Filter
    public var mipFilter: MipFilter
    public var addressU: AddressMode
    public var addressV: AddressMode
    public var addressW: AddressMode
    /// From 1 through 16.
    public var maxAnisotropy: Int
    public var lodMinimumClamp: Float
    public var lodMaximumClamp: Float
    /// Nil leaves comparison off, as `MTLCompareFunction.never` does in a descriptor.
    public var compareFunction: CompareFunction?
    public var borderColor: BorderColor
    public var normalizedCoordinates: Bool
    public var supportsArgumentBuffers: Bool

    /// The defaults match a new `MTLSamplerDescriptor`, except that filters default to linear with mipmaps
    /// and addressing to repeat, as Frostfield's PBR materials sample without a sampler.
    public init(
        minification: Filter = .linear, magnification: Filter = .linear, mipFilter: MipFilter = .linear,
        addressU: AddressMode = .repeat, addressV: AddressMode = .repeat, addressW: AddressMode = .repeat,
        maxAnisotropy: Int = 1, lodMinimumClamp: Float = 0, lodMaximumClamp: Float = .greatestFiniteMagnitude,
        compareFunction: CompareFunction? = nil, borderColor: BorderColor = .transparentBlack,
        normalizedCoordinates: Bool = true, supportsArgumentBuffers: Bool = false
    ) {
        self.minification = minification
        self.magnification = magnification
        self.mipFilter = mipFilter
        self.addressU = addressU
        self.addressV = addressV
        self.addressW = addressW
        self.maxAnisotropy = maxAnisotropy
        self.lodMinimumClamp = lodMinimumClamp
        self.lodMaximumClamp = lodMaximumClamp
        self.compareFunction = compareFunction
        self.borderColor = borderColor
        self.normalizedCoordinates = normalizedCoordinates
        self.supportsArgumentBuffers = supportsArgumentBuffers
    }

    /// Reads every setting this type records from a descriptor.
    public init(descriptor: MTLSamplerDescriptor) {
        func filter(_ value: MTLSamplerMinMagFilter) -> Filter { value == .nearest ? .nearest : .linear }
        func address(_ value: MTLSamplerAddressMode) -> AddressMode {
            switch value {
            case .clampToEdge: .clampToEdge
            case .mirrorClampToEdge: .mirrorClampToEdge
            case .repeat: .repeat
            case .mirrorRepeat: .mirrorRepeat
            case .clampToZero: .clampToZero
            case .clampToBorderColor: .clampToBorderColor
            @unknown default: .clampToEdge
            }
        }
        let mip: MipFilter = switch descriptor.mipFilter {
        case .notMipmapped: .notMipmapped
        case .nearest: .nearest
        default: .linear
        }
        let compare: CompareFunction? = switch descriptor.compareFunction {
        case .never: nil
        case .less: .less
        case .equal: .equal
        case .lessEqual: .lessEqual
        case .greater: .greater
        case .notEqual: .notEqual
        case .greaterEqual: .greaterEqual
        case .always: .always
        @unknown default: nil
        }
        let border: BorderColor = switch descriptor.borderColor {
        case .opaqueBlack: .opaqueBlack
        case .opaqueWhite: .opaqueWhite
        default: .transparentBlack
        }
        self.init(
            minification: filter(descriptor.minFilter), magnification: filter(descriptor.magFilter), mipFilter: mip,
            addressU: address(descriptor.sAddressMode), addressV: address(descriptor.tAddressMode),
            addressW: address(descriptor.rAddressMode), maxAnisotropy: descriptor.maxAnisotropy,
            lodMinimumClamp: descriptor.lodMinClamp, lodMaximumClamp: descriptor.lodMaxClamp,
            compareFunction: compare, borderColor: border, normalizedCoordinates: descriptor.normalizedCoordinates,
            supportsArgumentBuffers: descriptor.supportArgumentBuffers
        )
    }

    /// A new descriptor with these settings.
    public var descriptor: MTLSamplerDescriptor {
        let result = MTLSamplerDescriptor()
        func filter(_ value: Filter) -> MTLSamplerMinMagFilter { value == .nearest ? .nearest : .linear }
        func address(_ value: AddressMode) -> MTLSamplerAddressMode {
            switch value {
            case .clampToEdge: .clampToEdge
            case .mirrorClampToEdge: .mirrorClampToEdge
            case .repeat: .repeat
            case .mirrorRepeat: .mirrorRepeat
            case .clampToZero: .clampToZero
            case .clampToBorderColor: .clampToBorderColor
            }
        }
        result.minFilter = filter(minification)
        result.magFilter = filter(magnification)
        result.mipFilter = switch mipFilter {
        case .notMipmapped: .notMipmapped
        case .nearest: .nearest
        case .linear: .linear
        }
        result.sAddressMode = address(addressU)
        result.tAddressMode = address(addressV)
        result.rAddressMode = address(addressW)
        result.maxAnisotropy = maxAnisotropy
        result.lodMinClamp = lodMinimumClamp
        result.lodMaxClamp = lodMaximumClamp
        result.compareFunction = switch compareFunction {
        case nil: .never
        case .never?: .never
        case .less?: .less
        case .equal?: .equal
        case .lessEqual?: .lessEqual
        case .greater?: .greater
        case .notEqual?: .notEqual
        case .greaterEqual?: .greaterEqual
        case .always?: .always
        }
        result.borderColor = switch borderColor {
        case .transparentBlack: .transparentBlack
        case .opaqueBlack: .opaqueBlack
        case .opaqueWhite: .opaqueWhite
        }
        result.normalizedCoordinates = normalizedCoordinates
        result.supportArgumentBuffers = supportsArgumentBuffers
        return result
    }

    /// Creates a sampler state and records this description for `SceneArchive`.
    public func makeSamplerState(device: any MTLDevice) throws -> any MTLSamplerState {
        guard (1...16).contains(maxAnisotropy), lodMinimumClamp.isFinite, lodMinimumClamp >= 0,
              lodMaximumClamp >= lodMinimumClamp else {
            throw SceneArchive.Failure.invalidSampler
        }
        let descriptor = descriptor
        guard let sampler = device.makeSamplerState(descriptor: descriptor) else {
            throw SceneArchive.Failure.resource("sampler state")
        }
        Self.record(sampler, sampling: self)
        return sampler
    }

    /// The description recorded when the sampler was created, or nil when Frostfield did not create it.
    public static func recorded(for sampler: any MTLSamplerState) -> TextureSampling? {
        registry.withLock { entries in
            guard let entry = entries[ObjectIdentifier(sampler)], entry.isAlive(sampler) else { return nil }
            return entry.sampling
        }
    }

    /// Records a sampler made from a descriptor elsewhere in Frostfield.
    package static func record(_ sampler: any MTLSamplerState, descriptor: MTLSamplerDescriptor) {
        record(sampler, sampling: TextureSampling(descriptor: descriptor))
    }

    private static func record(_ sampler: any MTLSamplerState, sampling: TextureSampling) {
        let entry = RegistryEntry(sampler: sampler, sampling: sampling)
        registry.withLock { entries in
            // A dead entry's identifier can be reused by a new sampler, so prune them on insert.
            if entries.count > 64 { entries = entries.filter { $0.value.isOccupied } }
            entries[ObjectIdentifier(sampler)] = entry
        }
    }

    private static let registry = Mutex<[ObjectIdentifier: RegistryEntry]>([:])
}

/// A weak reference to a sampler state and its description.
private final class RegistryEntry: @unchecked Sendable {
    // Written once in init; the weak reference is cleared by the runtime.
    private weak var sampler: AnyObject?
    let sampling: TextureSampling

    init(sampler: any MTLSamplerState, sampling: TextureSampling) {
        self.sampler = sampler
        self.sampling = sampling
    }

    var isOccupied: Bool { sampler != nil }

    func isAlive(_ candidate: any MTLSamplerState) -> Bool {
        sampler === candidate
    }
}
