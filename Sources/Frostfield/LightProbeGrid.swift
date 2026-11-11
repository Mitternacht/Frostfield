import Foundation
import Metal
import simd

/// An axis-aligned world-space volume of nine-coefficient diffuse light probes.
/// The grid holds one probe at the center of each cell.
public struct LightProbeGrid: Sendable {
    public static let maximumProbeCount = 65_536
    public static let maximumCaptureProbeCount = 4_096
    /// Sum of base-level texels across all six faces and captured positions.
    public static let maximumCaptureTexelCount = 16_777_216
    /// Additional complete capture passes after the initial pass.
    public static let maximumBounces = 8

    public var bounds: AxisAlignedBox { layout.bounds }
    public var resolution: SIMD3<Int> { layout.resolution }
    /// Probes advance along X, then Z, then Y.
    public let probes: [LightProbe]
    public let intensity: Float
    private let layout: Layout

    public init(
        bounds: AxisAlignedBox, resolution: SIMD3<Int>, probes: [LightProbe], intensity: Float = 1
    ) throws {
        try self.init(layout: Layout(bounds: bounds, resolution: resolution), probes: probes, intensity: intensity)
    }

    private init(layout: Layout, probes: [LightProbe], intensity: Float) throws {
        guard probes.count == layout.count else { throw Failure.probeCount(expected: layout.count) }
        try Self.validateIntensity(intensity)
        if intensity > 1 {
            for probe in probes {
                let gain = Double(intensity) * Double(probe.intensity)
                guard probe.coefficients.allSatisfy({
                    Double(simd_reduce_max(simd_abs($0))) * gain <= LightProbe.coefficientLimit
                }) else { throw LightProbe.Failure.coefficientRange }
            }
        }
        self.layout = layout
        self.probes = probes
        self.intensity = intensity
    }

    /// The center of the cell at the given X, Y and Z indices.
    public func probePosition(at index: SIMD3<Int>) throws -> SIMD3<Float> {
        guard index.x >= 0, index.x < resolution.x, index.y >= 0, index.y < resolution.y,
              index.z >= 0, index.z < resolution.z else { throw Failure.probeIndex }
        return layout.position(at: index)
    }

    /// Evaluates cosine-convolved linear irradiance before the Lambertian BRDF.
    /// Sampling offsets half a cell along the unit normal and clamps to the
    /// outer probe centers. Signed SH reconstruction is clamped after blending.
    public func irradiance(at position: SIMD3<Float>, normal: SIMD3<Float>) throws -> SIMD3<Float> {
        let (point, unitNormal) = try Self.sampleInputs(position: position, normal: normal)
        return irradiance(cell: biasedCell(at: point, normal: unitNormal), normal: unitNormal)
    }

    static func sampleInputs(position: SIMD3<Float>, normal: SIMD3<Float>) throws -> (SIMD3<Double>, SIMD3<Double>) {
        guard boundsFinite(position) else { throw Failure.position }
        guard boundsFinite(normal), normal != .zero else { throw LightProbe.Failure.direction }
        return (boundsDouble(position), simd_normalize(boundsDouble(normal)))
    }

    func biasedCell(at position: SIMD3<Double>, normal: SIMD3<Double>) -> SIMD3<Double> {
        (position - layout.minimum) / layout.spacing + 0.5 * normal
    }

    func irradiance(cell: SIMD3<Double>, normal: SIMD3<Double>) -> SIMD3<Float> {
        let last = resolution &- SIMD3(repeating: 1)
        let coordinate = simd_clamp(cell - 0.5, .zero, SIMD3<Double>(last))
        let lower = SIMD3(Int(coordinate.x), Int(coordinate.y), Int(coordinate.z))
        let fraction = coordinate - SIMD3<Double>(lower)
        let upper = SIMD3(min(lower.x + 1, last.x), min(lower.y + 1, last.y), min(lower.z + 1, last.z))
        let basis = LightProbe.Basis(normal, diffuse: true)

        // SH convolution is linear. Blend signed evaluations in Double
        // before clamping to get the same result as blending coefficients.
        var result = SIMD3<Double>.zero
        for y in 0...1 {
            for z in 0...1 {
                for x in 0...1 {
                    let weight = (x == 0 ? 1 - fraction.x : fraction.x)
                        * (y == 0 ? 1 - fraction.y : fraction.y)
                        * (z == 0 ? 1 - fraction.z : fraction.z)
                    guard weight > 0 else { continue }
                    let index = SIMD3(x == 0 ? lower.x : upper.x, y == 0 ? lower.y : upper.y, z == 0 ? lower.z : upper.z)
                    result += weight * probes[layout.offset(of: index)].evaluate(basis)
                }
            }
        }
        return SIMD3<Float>(simd_max(result, .zero) * Double(intensity))
    }

    /// Captures one HDR cube at every cell center and projects it to SH.
    /// The same tracked target is reused after each readback completes.
    /// The scene, sky and shadow resources follow CubeCamera's capture rules.
    /// Keep the scene's meshes and textures unchanged until the capture completes.
    public static nonisolated(nonsending) func capture(
        _ graph: SceneGraph, bounds: AxisAlignedBox, resolution: SIMD3<Int>, commandQueue: any MTLCommandQueue,
        faceSize: Int = 8, near: Float = 0.1, far: Float = 100, layers: SceneLayers = .default,
        excluding: Set<UUID> = [], background: EnvironmentBackground? = nil,
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        intensity: Float = 1
    ) async throws -> LightProbeGrid {
        try await bake(
            graph, bounds: bounds, resolution: resolution, commandQueue: commandQueue, bounces: 0,
            faceSize: faceSize, near: near, far: far, layers: layers, excluding: excluding, background: background,
            clearColor: clearColor, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps, intensity: intensity
        )
    }

    /// Captures an initial grid, then recaptures all probes for each additional bounce.
    /// Each bounce adds the preceding immutable grid to the original capture scene
    /// at `bouncePriority` on the capture layers. The source grid uses `intensity`
    /// in each pass. Supply the base scene with any other volumes that should stay
    /// active throughout the bake. The capture budget applies across all passes.
    public static nonisolated(nonsending) func bake(
        _ graph: SceneGraph, bounds: AxisAlignedBox, resolution: SIMD3<Int>, commandQueue: any MTLCommandQueue,
        bounces: Int, bouncePriority: Int = 0,
        faceSize: Int = 8, near: Float = 0.1, far: Float = 100, layers: SceneLayers = .default,
        excluding: Set<UUID> = [], background: EnvironmentBackground? = nil,
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        intensity: Float = 1
    ) async throws -> LightProbeGrid {
        try Task.checkCancellation()
        let layout = try Layout(bounds: bounds, resolution: resolution)
        try validateIntensity(intensity)
        guard (0...maximumBounces).contains(bounces) else { throw Failure.bounceCount }
        try validateCapture(count: layout.count * (bounces + 1), faceSize: faceSize)
        let session = try CaptureSession(
            commandQueue: commandQueue, faceSize: faceSize, near: near, far: far, layers: layers,
            excluding: excluding, background: background, clearColor: clearColor,
            shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps
        )
        let allProbes = 0..<layout.count
        let initial = try await session.probes(in: allProbes, graph: graph, layout: layout)
        var result = try LightProbeGrid(layout: layout, probes: initial, intensity: intensity)
        for _ in 0..<bounces {
            let source = LightProbeGridLighting.Volume(result, priority: bouncePriority, layers: layers)
            let bounceGraph = try addingBounce(source, to: graph, device: commandQueue.device)
            let probes = try await session.probes(in: allProbes, graph: bounceGraph, layout: layout)
            result = try LightProbeGrid(layout: layout, probes: probes, intensity: intensity)
        }
        return result
    }

    /// Returns a new grid with the specified flat X, Z, Y range recaptured.
    /// A bounce source must have the same bounds and resolution. Reuse the same
    /// completed bounce source for every range in a pass to keep its input fixed.
    /// An empty range returns this grid without submitting any capture commands.
    public nonisolated(nonsending) func updating(
        probes range: Range<Int>, from graph: SceneGraph, commandQueue: any MTLCommandQueue,
        bounceSource: LightProbeGridLighting.Volume? = nil,
        faceSize: Int = 8, near: Float = 0.1, far: Float = 100, layers: SceneLayers = .default,
        excluding: Set<UUID> = [], background: EnvironmentBackground? = nil,
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil
    ) async throws -> LightProbeGrid {
        try Task.checkCancellation()
        guard range.lowerBound >= 0, range.upperBound <= layout.count else { throw Failure.probeRange }
        guard !range.isEmpty else { return self }
        try Self.validateCapture(count: range.count, faceSize: faceSize)
        let captureGraph: SceneGraph
        if let bounceSource {
            guard bounceSource.grid.bounds == bounds, bounceSource.grid.resolution == resolution else { throw Failure.bounceLayout }
            captureGraph = try Self.addingBounce(bounceSource, to: graph, device: commandQueue.device)
        } else {
            captureGraph = graph
        }
        let session = try CaptureSession(
            commandQueue: commandQueue, faceSize: faceSize, near: near, far: far, layers: layers,
            excluding: excluding, background: background, clearColor: clearColor,
            shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps
        )
        let replacement = try await session.probes(in: range, graph: captureGraph, layout: layout)
        var updated = probes
        updated.replaceSubrange(range, with: replacement)
        return try LightProbeGrid(layout: layout, probes: updated, intensity: intensity)
    }

    private static func validateIntensity(_ intensity: Float) throws {
        guard intensity.isFinite, intensity >= 0 else { throw LightProbe.Failure.intensity }
    }

    private static func validateCapture(count: Int, faceSize: Int) throws {
        guard (1...LightProbe.maximumFaceSize).contains(faceSize) else { throw LightProbe.Failure.faceSize }
        guard count <= maximumCaptureProbeCount,
              count <= maximumCaptureTexelCount / (6 * faceSize * faceSize) else { throw Failure.captureBudget }
    }

    private static func addingBounce(
        _ source: LightProbeGridLighting.Volume, to graph: SceneGraph, device: any MTLDevice
    ) throws -> SceneGraph {
        var capture = graph
        var volumes = graph.lightProbeGridLighting?.volumes ?? []
        volumes.append(source)
        capture.lightProbeGridLighting = try LightProbeGridLighting(device: device, volumes: volumes)
        return capture
    }

    private struct CaptureSession {
        let commandQueue: any MTLCommandQueue
        let camera: CubeCamera
        let excluding: Set<UUID>
        let background: EnvironmentBackground?
        let clearColor: MTLClearColor
        let shadowMap: DirectionalShadowMap?
        let punctualShadowMaps: PunctualShadowMaps?

        init(
            commandQueue: any MTLCommandQueue, faceSize: Int, near: Float, far: Float, layers: SceneLayers,
            excluding: Set<UUID>, background: EnvironmentBackground?, clearColor: MTLClearColor,
            shadowMap: DirectionalShadowMap?, punctualShadowMaps: PunctualShadowMaps?
        ) throws {
            let target = try CubeCamera.makeTarget(device: commandQueue.device, size: faceSize, mipmapped: false)
            camera = try CubeCamera(queue: commandQueue, target: target, near: near, far: far, layers: layers)
            self.commandQueue = commandQueue
            self.excluding = excluding
            self.background = background
            self.clearColor = clearColor
            self.shadowMap = shadowMap
            self.punctualShadowMaps = punctualShadowMaps
        }

        nonisolated(nonsending) func probes(in range: Range<Int>, graph: SceneGraph, layout: Layout) async throws -> [LightProbe] {
            var probes: [LightProbe] = []
            probes.reserveCapacity(range.count)
            for offset in range {
                try Task.checkCancellation()
                guard let command = commandQueue.makeCommandBuffer() else { throw Failure.commandBuffer }
                command.label = "Frostfield light probe grid capture"
                try camera.encode(
                    graph, at: layout.position(at: layout.index(at: offset)), excluding: excluding, background: background,
                    clearColor: clearColor, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps, commandBuffer: command
                )
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                    command.addCompletedHandler { completed in
                        if completed.status == .completed {
                            continuation.resume()
                        } else {
                            continuation.resume(throwing: Failure.captureFailed(completed.error?.localizedDescription))
                        }
                    }
                    command.commit()
                }
                probes.append(try await LightProbe(projectingCube: camera.target, commandQueue: commandQueue))
            }
            try Task.checkCancellation()
            return probes
        }
    }

    private struct Layout: Sendable {
        let bounds: AxisAlignedBox
        let resolution: SIMD3<Int>
        let minimum: SIMD3<Double>
        let spacing: SIMD3<Double>
        let count: Int

        init(bounds: AxisAlignedBox, resolution: SIMD3<Int>) throws {
            let minimum = boundsDouble(bounds.minimum)
            let extent = boundsDouble(bounds.maximum) - minimum
            guard all(extent .> .zero) else { throw Failure.bounds }
            let (x, y, z) = (resolution.x, resolution.y, resolution.z)
            guard x > 0, y > 0, z > 0, x <= maximumProbeCount / y,
                  x * y <= maximumProbeCount / z else { throw Failure.resolution }
            self.bounds = bounds
            self.resolution = resolution
            self.minimum = minimum
            spacing = extent / SIMD3<Double>(resolution)
            count = x * y * z
        }

        func position(at index: SIMD3<Int>) -> SIMD3<Float> {
            SIMD3<Float>(minimum + (SIMD3<Double>(index) + 0.5) * spacing)
        }

        func offset(of index: SIMD3<Int>) -> Int {
            (index.y * resolution.z + index.z) * resolution.x + index.x
        }

        func index(at offset: Int) -> SIMD3<Int> {
            let row = offset / resolution.x
            return SIMD3(offset % resolution.x, row / resolution.z, row % resolution.z)
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case bounds
        case resolution
        case probeCount(expected: Int)
        case probeIndex
        case probeRange
        case position
        case captureBudget
        case bounceCount
        case bounceLayout
        case commandBuffer
        case captureFailed(String?)

        public var errorDescription: String? {
            switch self {
            case .bounds: "Use world bounds with positive extent along every axis."
            case .resolution: "Use a positive grid resolution with at most \(LightProbeGrid.maximumProbeCount) probes."
            case .probeCount(let expected): "Supply exactly \(expected) light probes."
            case .probeIndex: "The requested probe is outside the grid resolution."
            case .probeRange: "The requested capture range is outside the grid."
            case .position: "Use a finite world-space sample position."
            case .captureBudget: "The grid exceeds the probe capture count or texel budget."
            case .bounceCount: "Use 0...\(LightProbeGrid.maximumBounces) additional capture passes."
            case .bounceLayout: "Use a completed bounce grid with the same bounds and resolution."
            case .commandBuffer: "Metal could not create a probe capture command buffer."
            case .captureFailed(let message): message ?? "Metal did not finish capturing the light probe."
            }
        }
    }
}
