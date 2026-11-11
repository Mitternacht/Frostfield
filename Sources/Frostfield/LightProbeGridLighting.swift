import Foundation
import Metal
import simd

/// Prepared spatial diffuse lighting for a `SceneGraph` on one Metal device.
/// The volume list and its GPU data stay unchanged while frames use it.
public struct LightProbeGridLighting {
    public struct Volume: Sendable {
        public let grid: LightProbeGrid
        public let priority: Int
        public let layers: SceneLayers

        public init(_ grid: LightProbeGrid, priority: Int = 0, layers: SceneLayers = .default) {
            self.grid = grid
            self.priority = priority
            self.layers = layers
        }
    }

    public static let maximumVolumeCount = 32
    public static let maximumProbeCount = 65_536
    public let volumes: [Volume]
    public var device: any MTLDevice { buffer.device }

    let buffer: any MTLBuffer
    private let selectionOrder: [Int]

    /// Higher priority wins where volumes overlap. Later entries win equal priorities.
    /// Bounds need finite, normal Float extents for raster sampling.
    public init(device: any MTLDevice, volumes: [Volume]) throws {
        guard (1...Self.maximumVolumeCount).contains(volumes.count) else { throw Failure.volumeCount }
        let order = volumes.indices.sorted {
            let left = volumes[$0].priority, right = volumes[$1].priority
            return left == right ? $0 > $1 : left > right
        }
        var count = 0
        for volume in volumes {
            let extent = volume.grid.bounds.size
            guard extent.x.isNormal, extent.y.isNormal, extent.z.isNormal else { throw Failure.bounds }
            count += volume.grid.probes.count
            guard count <= Self.maximumProbeCount else { throw Failure.probeCount }
        }
        let coefficientStart = 1 + 3 * volumes.count
        var packed = Array(repeating: SIMD4<Float>.zero, count: coefficientStart)
        packed[0].x = Float(volumes.count)
        packed.reserveCapacity(coefficientStart + count * LightProbe.coefficientCount)
        for (slot, index) in order.enumerated() {
            let volume = volumes[index], grid = volume.grid
            let start = 1 + 3 * slot
            let mask = volume.layers.rawValue
            // Offsets and both 16-bit halves of the layer mask fit exactly in Float.
            packed[start] = SIMD4(grid.bounds.minimum, Float(packed.count))
            packed[start + 1] = SIMD4(grid.bounds.size, Float(mask & 0xffff))
            packed[start + 2] = SIMD4(SIMD3<Float>(grid.resolution), Float(mask >> 16))
            for probe in grid.probes {
                let gain = Double(grid.intensity) * Double(probe.intensity)
                packed.append(contentsOf: probe.coefficients.map { SIMD4(SIMD3<Float>(SIMD3<Double>($0) * gain), 0) })
            }
        }
        guard let buffer = packed.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else { throw Failure.allocation }
        buffer.label = "Frostfield light probe volumes"
        self.volumes = volumes
        self.selectionOrder = order
        self.buffer = buffer
    }

    /// Samples the winning volume using the same layer, priority and normal bias
    /// rules as the raster renderer. Returns zero outside all eligible volumes.
    public func irradiance(
        at position: SIMD3<Float>, normal: SIMD3<Float>, cameraLayers: SceneLayers = .default
    ) throws -> SIMD3<Float> {
        let (point, unitNormal) = try LightProbeGrid.sampleInputs(position: position, normal: normal)
        for index in selectionOrder {
            let volume = volumes[index]
            guard volume.layers.intersects(cameraLayers) else { continue }
            let cell = volume.grid.biasedCell(at: point, normal: unitNormal)
            guard all(cell .>= SIMD3(repeating: -0.01)),
                  all(cell .<= SIMD3<Double>(volume.grid.resolution) + 0.01) else { continue }
            return volume.grid.irradiance(cell: cell, normal: unitNormal)
        }
        return .zero
    }

    public enum Failure: LocalizedError, Equatable {
        case volumeCount, probeCount, bounds, allocation

        public var errorDescription: String? {
            switch self {
            case .volumeCount: "Supply 1...\(LightProbeGridLighting.maximumVolumeCount) light probe volumes."
            case .probeCount: "The raster light probe volumes exceed \(LightProbeGridLighting.maximumProbeCount) probes."
            case .bounds: "Use finite, normal Float extents for raster light probe volumes."
            case .allocation: "Metal could not allocate the light probe volume buffer."
            }
        }
    }
}
