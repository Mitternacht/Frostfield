import Foundation
import Metal
import simd

/// A CPU diffuse transport bake for an authored or automatically packed UV atlas.
///
/// Prepare a fixed receiver atlas, then add bounded batches and publish
/// lightmaps as they converge. Use each baker from one thread at a time.
/// Previously published textures remain valid and do not change.
public final class ProgressiveLightmapBaker {
    /// An immutable CPU copy of a texture binding's base-level linear RGBA.
    /// Share the snapshot between surfaces that use the same map and UV settings.
    public struct TextureSnapshot: Sendable {
        fileprivate let sample: LightmapTextureSample

        /// Reads level zero and waits for Metal to finish. Supply the producing
        /// queue to order this read after tracked writes. Finish writes from
        /// other queues and untracked writes before calling. Metal decodes sRGB
        /// and supported compressed formats.
        /// The snapshot uses the binding's UV transform, magnification filter
        /// and addressing modes. A custom sampler needs its matching description.
        public init(
            _ binding: PhysicallyBasedMaterial.TextureBinding, commandQueue: (any MTLCommandQueue)? = nil
        ) throws {
            sample = try LightmapTextureReader.snapshot(binding, queue: commandQueue)
        }
    }

    /// Static alpha-test coverage. An unmasked surface uses full coverage.
    public struct AlphaMask: Sendable {
        public let cutoff: Float
        public let opacity: Float
        public let texture: TextureSnapshot?

        /// Multiplies opacity by interpolated vertex alpha and the optional map's
        /// alpha. A texel or hit survives when the product is at least cutoff.
        /// Vertex alpha defaults to one, and the map uses its own UV binding.
        public init(cutoff: Float = 0.5, opacity: Float = 1, texture: TextureSnapshot? = nil) throws {
            guard (0...1).contains(cutoff), (0...1).contains(opacity),
                  texture?.sample.pixels.hasValidAlpha != false else { throw Failure.alphaMask }
            self.cutoff = cutoff
            self.opacity = opacity
            self.texture = texture
        }

        fileprivate func includes(_ triangle: LightmapTriangle, weights: SIMD3<Double>) -> Bool {
            let vertex = min(max(triangle.vertexAlpha(weights: weights) * Double(opacity), 0), 1)
            let alpha = texture.map { triangle.texel($0.sample, weights: weights).w } ?? 1
            return vertex * alpha >= Double(cutoff)
        }
    }

    /// Geometry with linear Lambertian reflectance and emission.
    /// Front faces emit and reflect. Covered back faces still occlude rays.
    public struct Surface: Sendable {
        public let geometry: MeshGeometry
        public let worldFromObject: simd_float4x4
        public let diffuseAlbedo: SIMD3<Float>
        public let emissiveRadiance: SIMD3<Float>
        public let diffuseTexture: TextureSnapshot?
        public let emissiveTexture: TextureSnapshot?
        public let alphaMask: AlphaMask?
        /// Interpolated vertex RGB multiplies the diffuse reflectance at a hit.
        /// Vertices without a color use white. Alpha is controlled by `alphaMask`.
        public let usesVertexColors: Bool

        /// Albedo is in 0...1 and emission in 0...65,504 per channel.
        /// The default black surface absorbs any light that reaches it.
        public init(
            geometry: MeshGeometry, worldFromObject: simd_float4x4 = matrix_identity_float4x4,
            diffuseAlbedo: SIMD3<Float> = .zero, emissiveRadiance: SIMD3<Float> = .zero,
            usesVertexColors: Bool = false
        ) throws {
            try self.init(
                geometry: geometry, worldFromObject: worldFromObject, diffuseAlbedo: diffuseAlbedo,
                emissiveRadiance: emissiveRadiance, diffuseTexture: nil, emissiveTexture: nil,
                usesVertexColors: usesVertexColors
            )
        }

        /// Each map multiplies its linear factor using its own UV set and
        /// transform. Diffuse reflectance clamps to one after multiplication.
        /// Pass nil for a channel with a constant factor.
        public init(
            geometry: MeshGeometry, worldFromObject: simd_float4x4 = matrix_identity_float4x4,
            diffuseAlbedo: SIMD3<Float> = .zero, emissiveRadiance: SIMD3<Float> = .zero,
            diffuseTexture: TextureSnapshot?, emissiveTexture: TextureSnapshot? = nil,
            usesVertexColors: Bool = false
        ) throws {
            try self.init(
                geometry: geometry, worldFromObject: worldFromObject, diffuseAlbedo: diffuseAlbedo,
                emissiveRadiance: emissiveRadiance, diffuseTexture: diffuseTexture, emissiveTexture: emissiveTexture,
                alphaMask: nil, usesVertexColors: usesVertexColors
            )
        }

        /// An alpha-tested surface cuts out direct shadows, bounce hits and
        /// receiver coverage. Passing nil retains opaque transport.
        public init(
            geometry: MeshGeometry, worldFromObject: simd_float4x4 = matrix_identity_float4x4,
            diffuseAlbedo: SIMD3<Float> = .zero, emissiveRadiance: SIMD3<Float> = .zero,
            diffuseTexture: TextureSnapshot? = nil, emissiveTexture: TextureSnapshot? = nil,
            alphaMask: AlphaMask?, usesVertexColors: Bool = false
        ) throws {
            guard (0..<3).allSatisfy({ (0...1).contains(diffuseAlbedo[$0]) }),
                  ProgressiveLightmapBaker.isRadiance(emissiveRadiance) else { throw Failure.surface }
            if (diffuseTexture?.sample.set == .secondary || emissiveTexture?.sample.set == .secondary
                || alphaMask?.texture?.sample.set == .secondary),
               geometry.vertices.contains(where: { $0.secondaryTextureCoordinate == nil }) {
                throw Failure.textureCoordinates
            }
            self.geometry = geometry
            self.worldFromObject = worldFromObject
            self.diffuseAlbedo = diffuseAlbedo
            self.emissiveRadiance = emissiveRadiance
            self.diffuseTexture = diffuseTexture
            self.emissiveTexture = emissiveTexture
            self.alphaMask = alphaMask
            self.usesVertexColors = usesVertexColors
        }

        fileprivate var part: MeshGeometry.MergePart { .init(geometry, transform: worldFromObject) }
    }

    /// The geometry and placement to use with the baker's shared lightmap.
    public struct ReceiverBinding: Sendable {
        /// Local-space mesh, with packed UV1 for an automatically packed bake.
        /// Triangle order, winding, material groups, UV0 and other attributes
        /// remain those of the input mesh.
        public let geometry: MeshGeometry
        /// The transform supplied when placing this receiver into the bake.
        public let worldFromObject: simd_float4x4
        /// The original vertex for each packed vertex, including chart splits.
        /// Nil for an authored atlas, whose vertex indices stay the same.
        public let sourceVertexIndices: [Int]?
    }

    /// A covered receiver point and the absolute progressive sample index.
    public struct IrradianceSample: Sendable {
        public let worldPosition: SIMD3<Double>
        public let worldNormal: SIMD3<Double>
        public let index: Int
    }

    public enum IrradianceSampling: Sendable {
        /// Uses the same fixed texel centers as the diffuse transport bake.
        case texelCenter
        /// Offsets each UV rasterization by one Halton sample within a texel.
        case halton
    }

    public static let maximumTexelCount = 262_144
    public static let maximumSurfaceCount = 1_024
    public static let maximumTriangleCount = 262_144
    public static let maximumVertexCount = 524_288
    public static let maximumBounces = 8
    public static let maximumSamplesPerBatch = 256
    public static let maximumPathsPerBatch = 4_194_304
    public static let maximumSamplesPerTexel = 65_536
    /// Total base-level texels across distinct transport textures.
    public static let maximumTextureTexelCount = 8_388_608

    public let textureCoordinateSet: MeshTextureCoordinateSet
    /// The receivers in input order. Each uses the same snapshot and UV set.
    public private(set) var receiverBindings: [ReceiverBinding]
    public private(set) var samplesPerTexel = 0

    private let device: any MTLDevice
    private let atlas: UVBakeAtlas
    private let coverage: any MTLTexture
    /// Each output texel selects an interior sample. Negative means uncovered.
    private let outputToSample: [Int32]
    private var receivers: [DiffuseLightmapTransport.Receiver]
    private var receiverRaster: ReceiverRaster
    private var receiverSurfaces: [Surface]
    private var receiverTransportGeometry: MeshGeometry?
    private var extraSurfaces: [Surface]
    private var direct: [SIMD3<Double>]
    private var transport: DiffuseLightmapTransport
    private var lighting: DiffuseLightmapTransport.Lighting
    private let seed: UInt64
    private var sums: [SIMD3<Double>]
    private var jitteredSamples: [JitteredSample]?
    private var preview: [SIMD3<Double>]?
    private var previewSample: UInt64 = 0

    /// The receiver's reflectance is only used if an indirect path returns to
    /// its geometry. The published irradiance is independent of the receiver's
    /// color at the baked point. Disable `includesReceiverInTransport` to exclude
    /// the receiver from both shadow and bounce rays.
    ///
    /// `maximumBounces` counts diffuse reflections after leaving the receiver.
    /// Zero still samples visible sky and emissive geometry. Direct lights
    /// use explicit visibility tests at the receiver and every reflected hit.
    /// Rectangle lights use a fixed 8×8 grid at each such point.
    /// Atlas samples are at texel centers. Geometric normals limit both initial
    /// visibility and reflection, and rays start `shadowBias` world units above
    /// the surface.
    public convenience init(
        device: any MTLDevice, receiver: Surface, width: Int, height: Int,
        textureCoordinateSet: MeshTextureCoordinateSet = .secondary,
        lights: [MeshLightmapBaker.DirectionalLight] = [], punctualLights: [PositionedLight] = [], surfaces: [Surface] = [],
        skyRadiance: SIMD3<Float> = .zero, includesReceiverInTransport: Bool = true,
        maximumBounces: Int = 2, shadowBias: Float = 0.0001, seed: UInt64 = 0,
        overlap: MeshTextureBaker.Overlap = .reject, padding: Int = 0
    ) throws {
        try self.init(
            device: device, receiverSurfaces: [receiver], width: width, height: height, layout: .authored(textureCoordinateSet),
            lights: lights, punctualLights: punctualLights, surfaces: surfaces, skyRadiance: skyRadiance,
            includesReceiversInTransport: includesReceiverInTransport,
            maximumBounces: maximumBounces, shadowBias: shadowBias, seed: seed, overlap: overlap, padding: padding
        )
    }

    /// Unwraps all receivers into one square UV1 atlas, with a gutter of
    /// `padding` texels. The resolution is 16 through 512. Each receiver casts
    /// shadows, reflects and emits light in the common world-space scene. Extra
    /// surfaces participate in transport without receiving atlas space.
    /// Rectangle lights use a fixed 8×8 grid at each receiver and reflected hit.
    ///
    /// Use the geometry in `receiverBindings` with the shared snapshot's
    /// `lightMapBinding()`. Transport still samples the input materials on
    /// their original coordinates, including any original UV1. If a display
    /// material already used UV1, its original coordinate needs a separate
    /// channel or remapped texture when applying the new lightmap.
    public convenience init(
        device: any MTLDevice, receivers: [Surface], resolution: Int = 256,
        lights: [MeshLightmapBaker.DirectionalLight] = [], punctualLights: [PositionedLight] = [], surfaces: [Surface] = [],
        skyRadiance: SIMD3<Float> = .zero, maximumBounces: Int = 2, shadowBias: Float = 0.0001,
        seed: UInt64 = 0, padding: Int = 4
    ) throws {
        guard resolution >= 16 else { throw MeshGeometry.UVAtlasError.resolution }
        try self.init(
            device: device, receiverSurfaces: receivers, width: resolution, height: resolution, layout: .packed,
            lights: lights, punctualLights: punctualLights, surfaces: surfaces, skyRadiance: skyRadiance,
            includesReceiversInTransport: true,
            maximumBounces: maximumBounces, shadowBias: shadowBias, seed: seed, overlap: .reject, padding: padding
        )
    }

    private enum AtlasLayout {
        case authored(MeshTextureCoordinateSet)
        case packed
    }

    private init(
        device: any MTLDevice, receiverSurfaces: [Surface], width: Int, height: Int, layout: AtlasLayout,
        lights: [MeshLightmapBaker.DirectionalLight], punctualLights: [PositionedLight], surfaces: [Surface], skyRadiance: SIMD3<Float>,
        includesReceiversInTransport: Bool, maximumBounces: Int, shadowBias: Float, seed: UInt64,
        overlap: MeshTextureBaker.Overlap, padding: Int
    ) throws {
        guard width > 0, height > 0, width <= Self.maximumTexelCount / height else { throw Failure.texelLimit }
        guard !receiverSurfaces.isEmpty else { throw Failure.receiverCount }
        try Self.validateGeometry(receivers: receiverSurfaces, surfaces: surfaces)
        try Self.validateLightCount(lights.count, punctual: punctualLights.count)
        guard shadowBias.isFinite, shadowBias >= 0 else { throw MeshLightmapBaker.Failure.shadowBias }
        guard Self.isRadiance(skyRadiance), (0...Self.maximumBounces).contains(maximumBounces) else {
            throw Failure.transport
        }
        var atlas = try UVBakeAtlas(width: width, height: height, padding: padding)
        let worldGeometry: MeshGeometry
        let geometry: MeshGeometry
        let textureCoordinateSet: MeshTextureCoordinateSet
        let bindings: [ReceiverBinding]
        switch layout {
        case .authored(let set):
            worldGeometry = try MeshGeometry.merged(receiverSurfaces.map(\.part)).geometry
            geometry = worldGeometry
            textureCoordinateSet = set
            bindings = receiverSurfaces.map {
                ReceiverBinding(geometry: $0.geometry, worldFromObject: $0.worldFromObject, sourceVertexIndices: nil)
            }
        case .packed:
            let merged = try MeshGeometry.merged(receiverSurfaces.map(\.part), useGroups: true)
            worldGeometry = merged.geometry
            let packed = try worldGeometry.unwrappedUVAtlas(resolution: width, padding: padding)
            guard packed.geometry.vertices.count <= Self.maximumVertexCount else { throw Failure.geometryLimit }
            geometry = packed.geometry
            textureCoordinateSet = .secondary
            bindings = try Self.packedBindings(packed, ranges: merged.indexRanges, sources: receiverSurfaces)
        }
        let coordinates = try atlas.coordinates(of: geometry, set: textureCoordinateSet)
        let bias = Double(shadowBias)
        let receiverMasks = receiverSurfaces.contains { $0.alphaMask != nil } ? receiverSurfaces.flatMap {
            Array(repeating: $0.alphaMask, count: $0.geometry.indices.count / 3)
        } : nil
        let receiverRaster = ReceiverRaster(
            geometry: geometry, coordinates: coordinates, authoredGeometry: worldGeometry, masks: receiverMasks,
            includesReceiversInTransport: includesReceiversInTransport, bias: bias, overlap: overlap, padding: padding
        )
        let receiverByTexel = try receiverRaster.samples(in: &atlas, offset: .zero)

        var receivers: [DiffuseLightmapTransport.Receiver] = []
        var outputToSample = [Int32](repeating: -1, count: width * height)
        for (index, point) in receiverByTexel.enumerated() {
            guard let point else { continue }
            outputToSample[index] = Int32(receivers.count)
            receivers.append(point)
        }
        try atlas.pad(by: padding) { source, destination in outputToSample[destination] = outputToSample[source] }

        let receiverTransportGeometry = includesReceiversInTransport ? worldGeometry : nil
        let transportSurfaces = includesReceiversInTransport ? receiverSurfaces + surfaces : surfaces
        let transportGeometry = try Self.transportGeometry(receivers: receiverTransportGeometry, surfaces: surfaces)
        let transport = try DiffuseLightmapTransport(
            geometry: transportGeometry, surfaces: transportSurfaces, maximumBounces: maximumBounces, bias: bias
        )
        let lighting = DiffuseLightmapTransport.Lighting(lights, punctualLights, skyRadiance: skyRadiance)
        self.device = device
        self.atlas = atlas
        self.coverage = try atlas.coverageTexture(device: device)
        self.textureCoordinateSet = textureCoordinateSet
        self.receiverBindings = bindings
        self.outputToSample = outputToSample
        self.receivers = receivers
        self.receiverRaster = receiverRaster
        self.receiverSurfaces = receiverSurfaces
        self.receiverTransportGeometry = receiverTransportGeometry
        self.extraSurfaces = surfaces
        self.transport = transport
        self.lighting = lighting
        self.seed = seed
        self.direct = receivers.map { transport.directIrradiance(at: $0, lighting: lighting) }
        self.sums = .init(repeating: .zero, count: receivers.count)
    }

    /// Adds cosine-weighted paths at every covered texel. The absolute sample
    /// number and texel index determine the random sequence. Splitting a sample
    /// count across calls produces the same texture as one batch.
    public func accumulate(samples: Int) throws {
        guard jitteredSamples == nil else { throw Failure.samplingMode }
        let end = try cumulativeSampleEnd(samples)
        let indices = UInt64(samplesPerTexel)..<UInt64(end)
        for (index, receiver) in receivers.enumerated() where receiver.geometricNormal != .zero {
            sums[index] = sampledSum(at: receiver, indices: indices, startingWith: sums[index])
        }
        samplesPerTexel = end
    }

    /// Accumulates a caller-defined indirect irradiance source at covered
    /// receiver positions. Return finite, nonnegative linear RGB without
    /// receiver albedo or exposure. Direct lights are evaluated at those same
    /// positions. Halton sampling normalizes each texel by its own coverage.
    /// Use a function of the sample and stable scene data. Call `reset()` when
    /// changing the source. A failed batch preserves all previous samples.
    public func accumulate(
        samples: Int, sampling: IrradianceSampling = .texelCenter,
        indirectIrradiance: (IrradianceSample) throws -> SIMD3<Float>
    ) throws {
        let end = try cumulativeSampleEnd(samples)
        try accumulateCustom(until: end, sampling: sampling) { points, sample in
            try points.map { point in
                try Self.validateCustomIrradiance(indirectIrradiance(.init(
                    worldPosition: point.position, worldNormal: point.normal, index: sample
                )))
            }
        }
    }

    /// Evaluates a prepared Metal buffer graph for every covered receiver sample.
    /// Use a three-field graph. Input field 0 contains world position in XYZ
    /// and the absolute sample index in W. Field 1 contains the unit world
    /// normal in XYZ and zero in W. Write finite nonnegative linear RGB
    /// irradiance to output field 2. Use a finite value in its unused W.
    ///
    /// Halton sampling is the default. This synchronous method submits one
    /// compute batch per sample and waits before committing the CPU bake.
    /// Supply the same queue that produces any tracked input textures.
    public func accumulate(
        samples: Int, sampling: IrradianceSampling = .halton, irradianceKernel: ShaderBufferKernel,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:], commandQueue: any MTLCommandQueue
    ) throws {
        guard irradianceKernel.device === device, commandQueue.device === device else { throw ShaderComputeFailure.wrongDevice }
        guard irradianceKernel.graph.fieldCount == 3, irradianceKernel.graph.outputs[2] != nil else {
            throw Failure.irradianceKernelLayout
        }
        let end = try cumulativeSampleEnd(samples)
        let length = outputToSample.count * irradianceKernel.graph.elementStride
        guard let input = device.makeBuffer(length: length, options: .storageModeShared),
              let output = device.makeBuffer(length: length, options: .storageModeShared) else { throw Failure.irradianceKernelBuffers }
        input.label = "Frostfield lightmap irradiance input"
        output.label = "Frostfield lightmap irradiance output"
        let inputFields = input.contents().bindMemory(to: SIMD4<Float>.self, capacity: outputToSample.count * 3)
        let outputFields = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: outputToSample.count * 3)

        try accumulateCustom(until: end, sampling: sampling) { points, sample in
            guard !points.isEmpty else { return [] }
            for (index, point) in points.enumerated() {
                inputFields[index * 3] = SIMD4(SIMD3<Float>(point.position), Float(sample))
                inputFields[index * 3 + 1] = SIMD4(SIMD3<Float>(point.normal), 0)
                // Buffer graphs retain the old value of a nonfinite output.
                // A separate negative field makes that fallback fail validation.
                inputFields[index * 3 + 2] = SIMD4(repeating: -1)
            }
            guard let command = commandQueue.makeCommandBuffer() else { throw Failure.irradianceKernelCommand(nil) }
            try irradianceKernel.encode(
                elementCount: points.count, from: input, to: output,
                uniforms: uniforms, textures: textures, commandBuffer: command
            )
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw Failure.irradianceKernelCommand(command.error?.localizedDescription) }
            return try points.indices.map { index in
                let value = outputFields[index * 3 + 2]
                return try Self.validateCustomIrradiance(SIMD3(value.x, value.y, value.z))
            }
        }
    }

    private func accumulateCustom(
        until end: Int, sampling: IrradianceSampling,
        evaluate: ([DiffuseLightmapTransport.Receiver], Int) throws -> [SIMD3<Double>]
    ) throws {
        if case .halton = sampling {
            guard samplesPerTexel == 0 || jitteredSamples != nil else { throw Failure.samplingMode }
            try accumulateJittered(until: end, evaluate: evaluate)
            return
        }
        guard jitteredSamples == nil else { throw Failure.samplingMode }
        let active = receivers.indices.filter { receivers[$0].geometricNormal != .zero }
        let points = active.map { receivers[$0] }
        let scale = 1 / Double(end)
        var updated = sums
        for sample in samplesPerTexel..<end {
            for (index, value) in zip(active, try evaluate(points, sample)) {
                updated[index] += value
            }
        }
        for index in active {
            try Self.validateIrradiance(direct[index] + updated[index] * scale)
        }
        sums = updated
        samplesPerTexel = end
    }

    /// Discards the cumulative bake history and restarts the original sequence.
    public func reset() {
        sums = .init(repeating: .zero, count: receivers.count)
        jitteredSamples = nil
        samplesPerTexel = 0
    }

    /// Replaces the directional, point, spot and rectangle lights and constant sky, then starts
    /// accumulation again at sample zero with the baker's original seed.
    ///
    /// Omitted inputs use empty lists and a black sky. The prepared geometry,
    /// texture snapshots, atlas bindings, and coverage are reused. Previously
    /// published irradiance textures stay valid. If validation fails or the
    /// new direct term exceeds the irradiance format, the current bake is
    /// preserved, including its accumulated samples.
    public func replaceLighting(
        lights: [MeshLightmapBaker.DirectionalLight] = [], punctualLights: [PositionedLight] = [],
        skyRadiance: SIMD3<Float> = .zero
    ) throws {
        try Self.validateLightCount(lights.count, punctual: punctualLights.count)
        guard Self.isRadiance(skyRadiance) else { throw Failure.transport }
        let lighting = DiffuseLightmapTransport.Lighting(lights, punctualLights, skyRadiance: skyRadiance)
        let direct = try Self.validatedDirect(at: receivers, transport: transport, lighting: lighting)
        self.lighting = lighting
        self.direct = direct
        reset()
    }

    /// Replaces every non-receiving transport surface. Supply new placements,
    /// geometry or materials in the same order for a stable-topology BVH refit.
    /// Different topology builds a new hierarchy. Pass an empty array to remove
    /// all extra surfaces. The receiver atlas, bindings, and lighting stay fixed.
    /// A successful edit starts again at sample zero with the original seed.
    /// A rejected edit preserves the current bake and previously published textures.
    public func replaceSurfaces(_ surfaces: [Surface]) throws {
        try Self.validateGeometry(receivers: receiverSurfaces, surfaces: surfaces)
        let geometry = try Self.transportGeometry(receivers: receiverTransportGeometry, surfaces: surfaces)
        let materials = receiverTransportGeometry == nil ? surfaces : receiverSurfaces + surfaces
        let transport = try transport.replacing(geometry: geometry, surfaces: materials)
        let direct = try Self.validatedDirect(at: receivers, transport: transport, lighting: lighting)
        self.transport = transport
        self.extraSurfaces = surfaces
        self.direct = direct
        reset()
    }

    /// Moves the existing receivers in input order. Local geometry, UV charts
    /// and the prepared coverage are reused, including packed UV1 charts.
    /// Supply one affine placement per receiver. A successful edit restarts the
    /// cumulative bake and the rolling preview can blend through the edit.
    /// If an edit fails, the previous placements and bake remain available.
    public func replaceReceiverPlacements(_ placements: [simd_float4x4]) throws {
        guard placements.count == receiverSurfaces.count else { throw Failure.receiverPlacementCount }
        let surfaces = try zip(receiverSurfaces, placements).map { surface, placement in
            try Surface(
                geometry: surface.geometry, worldFromObject: placement, diffuseAlbedo: surface.diffuseAlbedo,
                emissiveRadiance: surface.emissiveRadiance, diffuseTexture: surface.diffuseTexture,
                emissiveTexture: surface.emissiveTexture, alphaMask: surface.alphaMask, usesVertexColors: surface.usesVertexColors
            )
        }
        let bindings = zip(receiverBindings, placements).map { binding, placement in
            ReceiverBinding(geometry: binding.geometry, worldFromObject: placement, sourceVertexIndices: binding.sourceVertexIndices)
        }
        let world = try MeshGeometry.merged(surfaces.map(\.part)).geometry
        let geometry = if bindings[0].sourceVertexIndices != nil {
            try MeshGeometry.merged(bindings.map { .init($0.geometry, transform: $0.worldFromObject) }).geometry
        } else {
            world
        }
        var frame = try UVBakeAtlas(width: atlas.width, height: atlas.height, padding: 0)
        let raster = ReceiverRaster(
            geometry: geometry, coordinates: try frame.coordinates(of: geometry, set: textureCoordinateSet),
            authoredGeometry: world, masks: receiverRaster.masks,
            includesReceiversInTransport: receiverRaster.includesReceiversInTransport, bias: receiverRaster.bias,
            overlap: receiverRaster.overlap, padding: receiverRaster.padding
        )
        let receivers = try raster.samples(in: &frame, offset: .zero).compactMap { $0 }
        guard receivers.count == self.receivers.count, zip(receivers, self.receivers).allSatisfy({ $0.texel == $1.texel }) else {
            throw Failure.receiverCoverage
        }
        let receiverTransportGeometry = raster.includesReceiversInTransport ? world : nil
        let transportGeometry = try Self.transportGeometry(receivers: receiverTransportGeometry, surfaces: extraSurfaces)
        let materials = raster.includesReceiversInTransport ? surfaces + extraSurfaces : extraSurfaces
        let transport = try transport.replacing(geometry: transportGeometry, surfaces: materials)
        let direct = try Self.validatedDirect(at: receivers, transport: transport, lighting: lighting)

        receiverBindings = bindings
        receiverRaster = raster
        self.receivers = receivers
        receiverSurfaces = surfaces
        self.receiverTransportGeometry = receiverTransportGeometry
        self.transport = transport
        self.direct = direct
        reset()
    }

    /// Publishes a rolling preview of the current direct light and a fresh
    /// batch of diffuse paths. A window of 1 uses the current estimate. Larger
    /// windows mix 1/window of the estimate into the preceding preview, up to
    /// 65,536. The first preview uses the current estimate at full weight.
    /// Light and geometry edits preserve preview history and reset the cumulative
    /// bake. Call `resetPreview()` to discard it after an abrupt change.
    public func updatePreview(samples: Int = 1, blendWindow: Int = 100) throws -> MeshLightmapBaker {
        try validateBatch(samples)
        guard (1...Self.maximumSamplesPerTexel).contains(blendWindow) else { throw Failure.previewWindow }
        let (end, overflow) = previewSample.addingReportingOverflow(UInt64(samples))
        guard !overflow else { throw Failure.sampleBudget }
        let indices = previewSample..<end
        let scale = 1 / Double(samples)
        let weight = 1 / Double(blendWindow)
        let next = receivers.enumerated().map { index, receiver in
            let bounce = receiver.geometricNormal == .zero ? .zero : sampledSum(at: receiver, indices: indices)
            let current = direct[index] + bounce * scale
            guard let preview, blendWindow > 1 else { return current }
            return preview[index] + (current - preview[index]) * weight
        }
        let result = try publish { next[$0] }
        preview = next
        previewSample = end
        return result
    }

    /// Discards rolling preview history and restarts its sample sequence.
    public func resetPreview() {
        preview = nil
        previewSample = 0
    }

    /// Publishes a fresh, ready-to-sample irradiance texture. A snapshot before
    /// the first batch contains direct lighting only. Padding follows
    /// the atlas coverage, and the shared coverage texture is read-only.
    public func snapshot() throws -> MeshLightmapBaker {
        if let jitteredSamples { return try snapshot(jitteredSamples) }
        let scale = samplesPerTexel == 0 ? 0 : 1 / Double(samplesPerTexel)
        return try publish { direct[$0] + sums[$0] * scale }
    }

    private struct JitteredSample {
        var sum = SIMD3<Double>.zero
        var count = 0

        var average: SIMD3<Double> { sum / Double(count) }
    }

    private func accumulateJittered(
        until end: Int, evaluate: ([DiffuseLightmapTransport.Receiver], Int) throws -> [SIMD3<Double>]
    ) throws {
        let work = max(outputToSample.count, receiverRaster.geometry.indices.count / 3)
        guard end - samplesPerTexel <= Self.maximumPathsPerBatch / work else { throw Failure.sampleBudget }
        var next = jitteredSamples ?? Array(repeating: JitteredSample(), count: outputToSample.count)
        for sample in samplesPerTexel..<end {
            var frame = try UVBakeAtlas(width: atlas.width, height: atlas.height, padding: 0)
            // LightMapper offsets the projected geometry. Sample at the inverse
            // offset, with the vertical sign following the top-left atlas.
            let offset = SIMD2(0.5 - Self.halton(sample + 1, base: 2), Self.halton(sample + 1, base: 3) - 0.5)
            let points = try receiverRaster.samples(in: &frame, offset: offset).compactMap { point in
                point.flatMap { $0.geometricNormal == .zero ? nil : $0 }
            }
            for (point, indirect) in zip(points, try evaluate(points, sample)) {
                let direct = transport.directIrradiance(at: point, lighting: lighting)
                next[point.texel].sum += direct + indirect
                next[point.texel].count += 1
            }
        }
        for sample in next where sample.count > 0 { try Self.validateIrradiance(sample.average) }
        jitteredSamples = next
        samplesPerTexel = end
    }

    private func snapshot(_ samples: [JitteredSample]) throws -> MeshLightmapBaker {
        var atlas = try UVBakeAtlas(width: atlas.width, height: atlas.height, padding: receiverRaster.padding)
        var pixels = [SIMD4<Float16>](repeating: .zero, count: samples.count)
        var hasCoverage = false
        for (index, sample) in samples.enumerated() where sample.count > 0 {
            pixels[index] = SIMD4<Float16>(SIMD4(SIMD3<Float>(sample.average), 1))
            atlas.markInterior(at: index)
            hasCoverage = true
        }
        if hasCoverage {
            try atlas.pad(by: receiverRaster.padding) { source, destination in pixels[destination] = pixels[source] }
        }
        let texture = try atlas.upload(pixels, device: device, format: .rgba16Float, label: "Frostfield jittered irradiance snapshot")
        return MeshLightmapBaker(
            textureCoordinateSet: textureCoordinateSet, irradianceTexture: texture,
            coverageTexture: try atlas.coverageTexture(device: device)
        )
    }

    private static func validateCustomIrradiance(_ value: SIMD3<Float>) throws -> SIMD3<Double> {
        guard (0..<3).allSatisfy({ value[$0].isFinite && value[$0] >= 0 }) else { throw Failure.customIrradiance }
        return SIMD3<Double>(value)
    }

    private static func halton(_ index: Int, base: Int) -> Double {
        var result = 0.0
        var fraction = 1.0
        var digit = index
        while digit > 0 {
            fraction /= Double(base)
            result += fraction * Double(digit % base)
            digit /= base
        }
        return result
    }

    private func publish(_ irradiance: (Int) -> SIMD3<Double>) throws -> MeshLightmapBaker {
        var pixels = [SIMD4<Float16>](repeating: .zero, count: outputToSample.count)
        for (index, source) in outputToSample.enumerated() where source >= 0 {
            let value = irradiance(Int(source))
            try Self.validateIrradiance(value)
            pixels[index] = SIMD4<Float16>(SIMD4(SIMD3<Float>(value), 1))
        }
        let texture = try atlas.upload(
            pixels, device: device, format: .rgba16Float, label: "Frostfield progressive irradiance snapshot"
        )
        return MeshLightmapBaker(textureCoordinateSet: textureCoordinateSet, irradianceTexture: texture, coverageTexture: coverage)
    }

    private func validateBatch(_ samples: Int) throws {
        guard (1...Self.maximumSamplesPerBatch).contains(samples),
              samples <= Self.maximumPathsPerBatch / receivers.count else { throw Failure.sampleBudget }
    }

    private func cumulativeSampleEnd(_ samples: Int) throws -> Int {
        try validateBatch(samples)
        guard samples <= Self.maximumSamplesPerTexel - samplesPerTexel else { throw Failure.sampleBudget }
        return samplesPerTexel + samples
    }

    private func sampledSum(
        at receiver: DiffuseLightmapTransport.Receiver, indices: Range<UInt64>, startingWith initial: SIMD3<Double> = .zero
    ) -> SIMD3<Double> {
        let frame = DiffuseLightmapTransport.Hemisphere(normal: receiver.normal)
        let texelSeed = seed ^ (UInt64(receiver.texel) &* 0xD6E8_FEB8_6659_FD93)
        var sum = initial
        for sample in indices {
            var random = SceneSeedGenerator(state: texelSeed ^ (sample &* 0xA076_1D64_78BD_642F))
            sum += transport.sample(receiver, frame: frame, lighting: lighting, random: &random)
        }
        return sum
    }

    private static func isRadiance(_ value: SIMD3<Float>) -> Bool {
        (0..<3).allSatisfy { (0...Float(Float16.greatestFiniteMagnitude)).contains(value[$0]) }
    }

    private static func validateLightCount(_ directional: Int, punctual: Int) throws {
        let limit = MeshLightmapBaker.maximumLightCount
        guard directional <= limit else { throw MeshLightmapBaker.Failure.lightCount }
        guard punctual <= limit - directional else { throw Failure.lightCount }
    }

    private static func validateGeometry(receivers: [Surface], surfaces: [Surface]) throws {
        guard receivers.count <= maximumSurfaceCount,
              surfaces.count <= maximumSurfaceCount - receivers.count else { throw Failure.geometryLimit }
        var triangles = 0
        var vertices = 0
        for surface in receivers + surfaces {
            let count = surface.geometry.indices.count / 3
            guard count <= maximumTriangleCount - triangles,
                  surface.geometry.vertices.count <= maximumVertexCount - vertices else { throw Failure.geometryLimit }
            triangles += count
            vertices += surface.geometry.vertices.count
        }
    }

    private static func transportGeometry(receivers: MeshGeometry?, surfaces: [Surface]) throws -> MeshGeometry? {
        guard !surfaces.isEmpty else { return receivers }
        let parts = (receivers.map { [MeshGeometry.MergePart($0)] } ?? []) + surfaces.map(\.part)
        return try MeshGeometry.merged(parts).geometry
    }

    private static func validatedDirect(
        at receivers: [DiffuseLightmapTransport.Receiver], transport: DiffuseLightmapTransport,
        lighting: DiffuseLightmapTransport.Lighting
    ) throws -> [SIMD3<Double>] {
        try receivers.map { receiver in
            let value = transport.directIrradiance(at: receiver, lighting: lighting)
            try Self.validateIrradiance(value)
            return value
        }
    }

    private static func validateIrradiance(_ value: SIMD3<Double>) throws {
        guard simd_reduce_max(value) <= Double(Float16.greatestFiniteMagnitude) else {
            throw MeshLightmapBaker.Failure.irradianceRange
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case receiverCount
        case receiverPlacementCount
        case receiverCoverage
        case lightCount
        case punctualPose
        case unsupportedPunctualLight
        case surface
        case transport
        case texelLimit
        case geometryLimit
        case sampleBudget
        case previewWindow
        case customIrradiance
        case irradianceKernelLayout
        case irradianceKernelBuffers
        case irradianceKernelCommand(String?)
        case samplingMode
        case textureSampler
        case textureCoordinates
        case textureDevice
        case unsupportedTexture
        case textureLimit
        case textureReadback(String)
        case textureValues
        case alphaMask

        public var errorDescription: String? {
            switch self {
            case .receiverCount: "A progressive lightmap needs at least one receiver."
            case .receiverPlacementCount: "Supply one transform per existing lightmap receiver."
            case .receiverCoverage: "The moved receivers must keep the prepared atlas coverage."
            case .lightCount: "A progressive lightmap supports up to \(MeshLightmapBaker.maximumLightCount) combined direct lights."
            case .punctualPose: "A baked punctual light needs a finite position and finite, nonzero direction."
            case .unsupportedPunctualLight: "Use a point, spot or rectangle for a positioned bake light."
            case .surface: "Use linear surface albedo in 0...1 and emission in 0...65504."
            case .transport: "Use linear sky radiance in 0...65504 and at most eight diffuse reflections."
            case .texelLimit: "A progressive lightmap supports at most \(ProgressiveLightmapBaker.maximumTexelCount) texels."
            case .geometryLimit: "The receiver and transport surfaces exceed the progressive geometry budget."
            case .sampleBudget: "The sample batch is empty or exceeds the per-batch, path-count or total sample budget."
            case .previewWindow: "The preview blend window must be from 1 to \(ProgressiveLightmapBaker.maximumSamplesPerTexel)."
            case .customIrradiance: "Use finite, nonnegative linear RGB for custom indirect irradiance."
            case .irradianceKernelLayout: "Use a three-field buffer graph that writes irradiance to field two."
            case .irradianceKernelBuffers: "Metal could not allocate the lightmap irradiance buffers."
            case .irradianceKernelCommand(let detail): detail ?? "Metal could not complete the lightmap irradiance command."
            case .samplingMode: "Reset the cumulative bake before switching receiver sampling modes."
            case .textureSampler: "Provide a TextureSampling description matching the transport map's Metal sampler."
            case .textureCoordinates: "A transport map using UV1 needs secondary texture coordinates on its geometry."
            case .textureDevice: "The readback queue must use the transport texture's Metal device."
            case .unsupportedTexture: "Use a shader-readable, single-sample 2D color texture for transport."
            case .textureLimit: "Transport maps exceed the baker's total base-level texel budget."
            case .textureReadback(let detail): "Could not read a transport texture. \(detail)"
            case .textureValues: "Transport texture RGB values must be finite and nonnegative, and mapped emission cannot exceed 65504."
            case .alphaMask: "Alpha mask opacity, cutoff and map alpha must be finite and within 0...1."
            }
        }
    }
}

private struct ReceiverRaster {
    let geometry: MeshGeometry
    let coordinates: [SIMD2<Int64>]
    let authoredGeometry: MeshGeometry
    let masks: [ProgressiveLightmapBaker.AlphaMask?]?
    let includesReceiversInTransport: Bool
    let bias: Double
    let overlap: MeshTextureBaker.Overlap
    let padding: Int

    func samples(in atlas: inout UVBakeAtlas, offset: SIMD2<Double>) throws -> [DiffuseLightmapTransport.Receiver?] {
        var receivers = [DiffuseLightmapTransport.Receiver?](repeating: nil, count: atlas.width * atlas.height)
        for start in stride(from: 0, to: geometry.indices.count, by: 3) {
            let ia = Int(geometry.indices[start]), ib = Int(geometry.indices[start + 1]), ic = Int(geometry.indices[start + 2])
            let a = geometry.vertices[ia], b = geometry.vertices[ib], c = geometry.vertices[ic]
            let pa = SIMD3<Double>(a.position), pb = SIMD3<Double>(b.position), pc = SIMD3<Double>(c.position)
            let minimum = simd_min(pa, simd_min(pb, pc))
            let maximum = simd_max(pa, simd_max(pb, pc))
            let normal = DiffuseLightmapTransport.normal(pa, pb, pc)
            let na = simd_normalize(SIMD3<Double>(a.normal))
            let nb = simd_normalize(SIMD3<Double>(b.normal))
            let nc = simd_normalize(SIMD3<Double>(c.normal))
            // Packed receiver triangles keep their original transport order.
            let index = start / 3
            let ignored = includesReceiversInTransport ? index : nil
            let acceptsSample: ((SIMD3<Double>) -> Bool)?
            if let mask = masks?[index] {
                let triangle = LightmapTriangle(authoredGeometry, index: index)
                acceptsSample = { mask.includes(triangle, weights: $0) }
            } else {
                acceptsSample = nil
            }
            try atlas.rasterize(
                coordinates[ia], coordinates[ib], coordinates[ic], triangle: index,
                overlap: overlap, sampleOffset: offset, acceptsSample: acceptsSample
            ) { texel, weights in
                let shading = UVBakeAtlas.interpolatedNormal(na, nb, nc, weights: weights)
                // Clamp round-off at a chart edge to the triangle's bounds.
                let position = simd_clamp(pa * weights.x + pb * weights.y + pc * weights.z, minimum, maximum)
                receivers[texel] = .init(
                    position: position, origin: position + normal * bias,
                    normal: simd_dot(shading, normal) > 0 ? shading : normal, geometricNormal: normal,
                    ignoringTriangle: ignored, texel: texel
                )
            }
        }
        return receivers
    }
}

fileprivate struct LightmapTriangle {
    let a: MeshVertex
    let b: MeshVertex
    let c: MeshVertex

    init(_ geometry: MeshGeometry, index: Int) {
        let start = index * 3
        a = geometry.vertices[Int(geometry.indices[start])]
        b = geometry.vertices[Int(geometry.indices[start + 1])]
        c = geometry.vertices[Int(geometry.indices[start + 2])]
    }

    func weights(_ query: PickingQuery) -> SIMD3<Double> {
        query.barycentricWeights(a.position, b.position, c.position)
    }

    func vertexAlpha(weights: SIMD3<Double>) -> Double {
        let opacity = SIMD3(Double(a.color?.w ?? 1), Double(b.color?.w ?? 1), Double(c.color?.w ?? 1))
        return simd_dot(weights, opacity) / (weights.x + weights.y + weights.z)
    }

    func vertexRGB(weights: SIMD3<Double>) -> SIMD3<Double> {
        func rgb(_ vertex: MeshVertex) -> SIMD3<Double> {
            guard let color = vertex.color else { return SIMD3(repeating: 1) }
            return SIMD3(Double(color.x), Double(color.y), Double(color.z))
        }
        return (rgb(a) * weights.x + rgb(b) * weights.y + rgb(c) * weights.z) / (weights.x + weights.y + weights.z)
    }

    func texel(_ map: LightmapTextureSample, weights: SIMD3<Double>) -> SIMD4<Double> {
        let uv = (weights.x * map.coordinates(a) + weights.y * map.coordinates(b) + weights.z * map.coordinates(c))
            / (weights.x + weights.y + weights.z)
        return map.sample(uv)
    }
}

private struct DiffuseLightmapTransport {
    struct Lighting {
        let directional: LightmapDirectionalLighting
        let punctual: LightmapPunctualLighting
        let skyIrradiance: SIMD3<Double>

        init(
            _ lights: [MeshLightmapBaker.DirectionalLight], _ punctualLights: [ProgressiveLightmapBaker.PositionedLight],
            skyRadiance: SIMD3<Float>
        ) {
            directional = LightmapDirectionalLighting(lights)
            punctual = LightmapPunctualLighting(punctualLights)
            skyIrradiance = .pi * SIMD3<Double>(skyRadiance)
        }
    }

    struct Receiver {
        let position: SIMD3<Double>
        let origin: SIMD3<Double>
        let normal: SIMD3<Double>
        let geometricNormal: SIMD3<Double>
        let ignoringTriangle: Int?
        let texel: Int
    }

    private struct Material {
        let albedo: SIMD3<Double>
        let emission: SIMD3<Double>
        let diffuseTexture: LightmapTextureSample?
        let emissiveTexture: LightmapTextureSample?
        let alphaMask: ProgressiveLightmapBaker.AlphaMask?
        let usesVertexColors: Bool

        func sample(
            hit: Int, in geometry: MeshGeometry, origin: SIMD3<Double>, direction: SIMD3<Double>
        ) -> (albedo: SIMD3<Double>, emission: SIMD3<Double>) {
            guard diffuseTexture != nil || emissiveTexture != nil || usesVertexColors else { return (albedo, emission) }
            let triangle = LightmapTriangle(geometry, index: hit)
            let weights = triangle.weights(PickingQuery(origin: origin, unitDirection: direction))
            func texel(_ map: LightmapTextureSample) -> SIMD3<Double> {
                let value = triangle.texel(map, weights: weights)
                return SIMD3(value.x, value.y, value.z)
            }
            let vertexAlbedo = usesVertexColors ? albedo * triangle.vertexRGB(weights: weights) : albedo
            let diffuse = simd_clamp(diffuseTexture.map { vertexAlbedo * texel($0) } ?? vertexAlbedo, .zero, SIMD3(repeating: 1))
            let emitted = emissiveTexture.map { emission * texel($0) } ?? emission
            return (diffuse, emitted)
        }
    }

    private struct Triangle {
        let normal: SIMD3<Double>
        let material: Int
    }

    private let visibility: MeshBVH?
    private let triangles: [Triangle]
    private let materials: [Material]
    private let hasAlphaMasks: Bool
    private let maximumBounces: Int
    private let bias: Double

    init(
        geometry: MeshGeometry?, surfaces: [ProgressiveLightmapBaker.Surface], maximumBounces: Int, bias: Double,
        previousVisibility: MeshBVH? = nil
    ) throws {
        visibility = try geometry.map { geometry in
            if let previousVisibility, previousVisibility.geometry.vertices.count == geometry.vertices.count,
               previousVisibility.geometry.indices == geometry.indices {
                return try previousVisibility.refitted(to: geometry)
            }
            return try MeshBVH(geometry: geometry)
        }
        var triangles: [Triangle] = []
        if let geometry {
            triangles.reserveCapacity(geometry.indices.count / 3)
            for (index, surface) in surfaces.enumerated() {
                for _ in 0..<(surface.geometry.indices.count / 3) {
                    let (a, b, c) = geometry.triangleCorners(triangles.count)
                    let normal = Self.normal(SIMD3<Double>(a), SIMD3<Double>(b), SIMD3<Double>(c))
                    triangles.append(.init(normal: normal, material: index))
                }
            }
        }
        self.triangles = triangles
        var textureIDs = Set<ObjectIdentifier>()
        var textureTexels = 0
        materials = try surfaces.map { surface in
            let diffuse = surface.diffuseTexture?.sample
            let emissive = surface.emissiveTexture?.sample
            let alpha = surface.alphaMask?.texture?.sample
            for map in [diffuse, emissive, alpha].compactMap({ $0 }) where textureIDs.insert(ObjectIdentifier(map.pixels)).inserted {
                guard map.pixels.texels.count <= ProgressiveLightmapBaker.maximumTextureTexelCount - textureTexels else {
                    throw ProgressiveLightmapBaker.Failure.textureLimit
                }
                textureTexels += map.pixels.texels.count
            }
            let emission = SIMD3<Double>(surface.emissiveRadiance)
            if let emissive, simd_reduce_max(emission * SIMD3<Double>(emissive.maximum)) > Double(Float16.greatestFiniteMagnitude) {
                throw ProgressiveLightmapBaker.Failure.textureValues
            }
            return Material(
                albedo: SIMD3<Double>(surface.diffuseAlbedo), emission: emission,
                diffuseTexture: diffuse, emissiveTexture: emissive, alphaMask: surface.alphaMask,
                usesVertexColors: surface.usesVertexColors && surface.geometry.vertices.contains { $0.color != nil }
            )
        }
        hasAlphaMasks = materials.contains { $0.alphaMask != nil }
        self.maximumBounces = maximumBounces
        self.bias = bias
    }

    func replacing(geometry: MeshGeometry?, surfaces: [ProgressiveLightmapBaker.Surface]) throws -> Self {
        try Self(geometry: geometry, surfaces: surfaces, maximumBounces: maximumBounces, bias: bias,
                 previousVisibility: visibility)
    }

    static func normal(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>) -> SIMD3<Double> {
        let cross = simd_cross(b - a, c - a)
        return cross == .zero ? .zero : simd_normalize(cross)
    }

    func directIrradiance(at receiver: Receiver, lighting: Lighting) -> SIMD3<Double> {
        irradiance(
            at: receiver.position, origin: receiver.origin, normal: receiver.normal,
            geometricNormal: receiver.geometricNormal, ignoringTriangle: receiver.ignoringTriangle, lighting: lighting
        )
    }

    private var hitFilter: MeshBVH.HitFilter? {
        guard hasAlphaMasks else { return nil }
        return { query, index in self.acceptsHit(query, index: index) }
    }

    private func acceptsHit(_ query: PickingQuery, index: Int) -> Bool {
        guard let mask = materials[triangles[index].material].alphaMask, let visibility else { return true }
        let triangle = LightmapTriangle(visibility.geometry, index: index)
        return mask.includes(triangle, weights: triangle.weights(query))
    }

    private func irradiance(
        at position: SIMD3<Double>, origin: SIMD3<Double>, normal: SIMD3<Double>,
        geometricNormal: SIMD3<Double>? = nil, ignoringTriangle: Int?, lighting: Lighting
    ) -> SIMD3<Double> {
        let filter = hitFilter
        let directional = lighting.directional.irradiance(
            at: origin, normal: normal, geometricNormal: geometricNormal,
            visibility: visibility, ignoringTriangle: ignoringTriangle, acceptsHit: filter
        )
        guard !lighting.punctual.isEmpty else { return directional }
        return directional + lighting.punctual.irradiance(
            at: position, shadowOrigin: origin, normal: normal, geometricNormal: geometricNormal ?? normal,
            visibility: visibility, ignoringTriangle: ignoringTriangle, acceptsHit: filter
        )
    }

    func sample(_ receiver: Receiver, frame: Hemisphere, lighting: Lighting, random: inout SceneSeedGenerator) -> SIMD3<Double> {
        var direction = frame.direction(random: &random)
        guard simd_dot(receiver.geometricNormal, direction) > 0 else { return .zero }
        var origin = receiver.origin
        var ignoring = receiver.ignoringTriangle
        var throughput = SIMD3<Double>(repeating: 1)
        var total = SIMD3<Double>.zero
        let filter = hitFilter
        for bounce in 0...maximumBounces {
            guard let visibility,
                  let hit = visibility.nearestTriangle(
                    origin: origin, unitDirection: direction, ignoringTriangle: ignoring, acceptsHit: filter
                  ) else {
                return total + throughput * lighting.skyIrradiance
            }
            let triangle = triangles[hit.triangle]
            guard simd_dot(triangle.normal, direction) < 0 else { return total }
            let material = materials[triangle.material].sample(
                hit: hit.triangle, in: visibility.geometry, origin: origin, direction: direction
            )
            // The receiver's cosine-weighted integral is π times incoming radiance.
            total += throughput * (.pi * material.emission)
            guard bounce < maximumBounces, material.albedo != .zero else { return total }
            let position = origin + hit.distance * direction
            origin += hit.distance * direction + bias * triangle.normal
            ignoring = hit.triangle
            // The cosine PDF cancels the Lambertian cosine and 1/π factors.
            throughput *= material.albedo
            total += throughput * irradiance(
                at: position, origin: origin, normal: triangle.normal, ignoringTriangle: ignoring, lighting: lighting
            )
            direction = Hemisphere(normal: triangle.normal).direction(random: &random)
        }
        return total
    }

    struct Hemisphere {
        let normal: SIMD3<Double>
        let tangent: SIMD3<Double>
        let bitangent: SIMD3<Double>

        init(normal: SIMD3<Double>) {
            self.normal = normal
            let axis = abs(normal.z) < 0.999 ? SIMD3<Double>(0, 0, 1) : SIMD3<Double>(0, 1, 0)
            tangent = simd_normalize(simd_cross(axis, normal))
            bitangent = simd_cross(normal, tangent)
        }

        func direction(random: inout SceneSeedGenerator) -> SIMD3<Double> {
            let radiusSquared = random.nextDouble()
            let radius = radiusSquared.squareRoot()
            let angle = 2 * Double.pi * random.nextDouble()
            return tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle)) + normal * (1 - radiusSquared).squareRoot()
        }
    }
}
