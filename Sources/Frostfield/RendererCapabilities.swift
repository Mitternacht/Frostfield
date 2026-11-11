import Metal

/// The optional render paths and hardware limits Frostfield can query on a Metal device.
public struct RendererCapabilities: Sendable, Equatable {
    public enum Feature: CaseIterable, Hashable, Sendable {
        /// The Metal acceleration structure path, alongside the software BVH.
        case hardwareRayTracing
        /// The device can bind tier 2 argument buffers, used by direct path texture sampling.
        case tier2ArgumentBuffers
        case temporalPathDenoising
        case batchCommandBundles
        case indexedIndirectArguments
        case stageTimestamps
        case bcTextureCompression
        /// Native ETC2, EAC and LDR ASTC texture sampling on Apple GPU families.
        case appleBlockTextureCompression
        case float32TextureFiltering
    }

    public let deviceName: String
    public let features: Set<Feature>
    /// Sample counts accepted by Frostfield's mesh and primitive renderers on this device.
    public let rasterSampleCounts: [Int]
    /// Device limit in bytes for a single Metal buffer.
    public let maximumBufferLength: Int
    /// Device-wide threadgroup limits by axis. A prepared compute pipeline can impose a lower limit.
    public let maximumThreadsPerThreadgroup: SIMD3<Int>

    /// Reads the current device's capabilities, including a one-command buffer
    /// probe for batching. Use the snapshot when choosing render paths during setup.
    public init(device: any MTLDevice) {
        deviceName = device.name
        features = Set(Feature.allCases.filter { Self.supports($0, on: device) })
        rasterSampleCounts = Self.allowedRasterSampleCounts.filter(device.supportsTextureSampleCount)
        maximumBufferLength = device.maxBufferLength
        let threads = device.maxThreadsPerThreadgroup
        maximumThreadsPerThreadgroup = SIMD3(threads.width, threads.height, threads.depth)
    }

    /// A supported path can still reject an incompatible texture, format, or pass configuration.
    public func hasFeature(_ feature: Feature) -> Bool { features.contains(feature) }

    private static let allowedRasterSampleCounts = [1, 2, 4, 8]

    static func supportsRasterSampleCount(_ count: Int, on device: any MTLDevice) -> Bool {
        allowedRasterSampleCounts.contains(count) && device.supportsTextureSampleCount(count)
    }

    private static func supports(_ feature: Feature, on device: any MTLDevice) -> Bool {
        switch feature {
        case .hardwareRayTracing: device.supportsRaytracing
        case .tier2ArgumentBuffers: device.argumentBuffersSupport == .tier2
        case .temporalPathDenoising: PathTemporalDenoiser.isSupported(device: device)
        case .batchCommandBundles: BatchedMesh.supportsIndirectSubmission(on: device)
        case .indexedIndirectArguments: MeshIndirectDraws.isSupported(on: device)
        case .stageTimestamps: SceneGPUTimingFrame.isSupported(on: device)
        case .bcTextureCompression: device.supportsBCTextureCompression
        case .appleBlockTextureCompression: device.supportsFamily(.apple2)
        case .float32TextureFiltering: device.supports32BitFloatFiltering
        }
    }
}
