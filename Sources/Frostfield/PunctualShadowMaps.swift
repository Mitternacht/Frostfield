import Foundation
import Metal
import simd

/// Caller-owned depth maps for shadow-casting point, spot, and rectangle lights on one Metal queue.
///
/// Every shadow view renders into one slice of a single `depth32Float` 2D array texture.
/// A spot light uses one perspective slice that covers its outer cone. A point light,
/// or a rectangle light cast from its center, uses six slices aligned with the world
/// axes, each slightly wider than 90 degrees so the filter footprint stays on one face.
/// This layout needs no cube-array support, so every Metal device can prepare it.
///
/// Scene graph encoding assigns slots in depth-first node order to lights that carry
/// a `PunctualLight.Shadow`, have nonzero color and intensity, and fit the capacity
/// for their kind. On-demand maps preserve slots for still-eligible nodes before
/// assigning newly eligible lights. Other lights render unshadowed. Depth passes run in the caller's
/// command buffer before the color pass. Encode with one map set at a time in queue order.
public final class PunctualShadowMaps {
    public static let minimumResolution = 16
    public static let maximumResolution = 4096
    public static let maximumSpotCapacity = 8
    public static let maximumPointCapacity = 4

    /// The width and height of every shadow slice, in texels.
    public let resolution: Int
    /// The number of shadowed spot lights per encode.
    public let spotCapacity: Int
    /// The number of shadowed point and rectangle lights per encode. Each uses six slices.
    public let pointCapacity: Int
    /// Default policy for lights without an override.
    public let updateMode: ShadowMapUpdateMode
    package let texture: any MTLTexture
    package var shadowMoments: (any MTLTexture)? { variancePass?.moments }
    package var sliceCount: Int { spotCapacity + 6 * pointCapacity }
    var usesCachedUpdates: Bool { updateCache != nil }

    private let queue: any MTLCommandQueue
    private let pipeline: any MTLRenderPipelineState
    private let skinnedPipeline: any MTLRenderPipelineState
    private let maskedPipeline: any MTLRenderPipelineState
    private let skinnedMaskedPipeline: any MTLRenderPipelineState
    private let depthState: any MTLDepthStencilState
    private var variancePass: VarianceShadowPass?
    private let updateCache: ShadowMapUpdateCache?

    /// Resolution must be 16...4096 texels. Capacities are 0...8 spot and 0...4 point
    /// lights, with at least one slot in total. Memory is
    /// `resolution² × 4 bytes × (spotCapacity + 6 × pointCapacity)`.
    /// Supply per-light modes keyed by scene node ID, or an empty dictionary to
    /// enable later overrides on an automatic set. Sets with on-demand support
    /// hold one command buffer at a time on their creation queue. A new light,
    /// slice assignment or projection refreshes that light's views.
    public init(
        queue: any MTLCommandQueue, resolution: Int = 512, spotCapacity: Int = 4, pointCapacity: Int = 2,
        updateMode: ShadowMapUpdateMode = .automatic, lightUpdateModes: [UUID: ShadowMapUpdateMode]? = nil
    ) throws {
        guard (Self.minimumResolution...Self.maximumResolution).contains(resolution) else {
            throw Failure.resolution
        }
        guard (0...Self.maximumSpotCapacity).contains(spotCapacity),
              (0...Self.maximumPointCapacity).contains(pointCapacity),
              spotCapacity + pointCapacity > 0 else {
            throw Failure.capacity
        }

        let device = queue.device
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        guard let vertex = library.makeFunction(name: "frostfield_shadow_vertex"),
              let skinnedVertex = library.makeFunction(name: "frostfield_skinned_shadow_vertex"),
              let mask = library.makeFunction(name: "frostfield_shadow_mask") else {
            throw Failure.missingShader
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield punctual shadow depth"
        descriptor.vertexFunction = vertex
        descriptor.depthAttachmentPixelFormat = .depth32Float
        descriptor.inputPrimitiveTopology = .triangle
        do {
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.label = "Frostfield punctual skinned shadow depth"
            descriptor.vertexFunction = skinnedVertex
            skinnedPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.label = "Frostfield punctual masked skinned shadow depth"
            descriptor.fragmentFunction = mask
            skinnedMaskedPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.label = "Frostfield punctual masked shadow depth"
            descriptor.vertexFunction = vertex
            maskedPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }

        // Reversed, infinite-far depth: nearer casters store larger values.
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .greater
        depth.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else {
            throw Failure.resource("punctual shadow depth state")
        }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: resolution, height: resolution, mipmapped: false
        )
        textureDescriptor.textureType = .type2DArray
        textureDescriptor.arrayLength = spotCapacity + 6 * pointCapacity
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        textureDescriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw Failure.resource("punctual shadow depth array")
        }
        texture.label = "Frostfield punctual shadow maps"

        self.queue = queue
        self.resolution = resolution
        self.spotCapacity = spotCapacity
        self.pointCapacity = pointCapacity
        self.updateMode = updateMode
        updateCache = if updateMode == .onDemand || lightUpdateModes != nil {
            ShadowMapUpdateCache(defaultMode: updateMode, lightModes: lightUpdateModes ?? [:])
        } else {
            nil
        }
        self.texture = texture
        self.depthState = depthState
    }

    /// Returns a light's override or the set's default update policy.
    public func updateMode(for lightID: UUID) -> ShadowMapUpdateMode {
        updateCache?.updateMode(for: lightID) ?? updateMode
    }

    /// Changes one light's policy for subsequent command buffers. Pass nil to
    /// restore the set default. The node need not have an assigned slot yet.
    public func setUpdateMode(_ mode: ShadowMapUpdateMode?, for lightID: UUID) throws {
        guard let updateCache else { throw ShadowMapUpdateError.lightUpdateModesUnavailable }
        updateCache.setUpdateMode(mode, for: lightID)
    }

    /// Requests fresh depth for all assigned slices in a set with on-demand support.
    public func requestUpdate() {
        updateCache?.requestUpdate()
    }

    /// Requests fresh depth for the supplied light's scene node ID. Point and
    /// rectangle lights refresh all six faces together. Returns false for an
    /// set without on-demand support or a node without a completed or pending slot assignment.
    @discardableResult
    public func requestUpdate(for lightID: UUID) -> Bool {
        updateCache?.requestUpdate(for: lightID) ?? false
    }

    func existingSlots(commandBuffer: any MTLCommandBuffer) -> [UUID: Int] {
        updateCache?.existingSlots(commandBuffer: commandBuffer) ?? [:]
    }

    func beginUse(
        commandBuffer: any MTLCommandBuffer, layout: @autoclosure () -> ShadowMapUpdateCache.Layout
    ) throws -> ShadowMapUpdateCache.Use {
        try validate(commandBuffer: commandBuffer)
        guard let updateCache else { return .automatic }
        return try updateCache.begin(layout(), commandBuffer: commandBuffer)
    }

    func beginRead(slice: Int, commandBuffer: any MTLCommandBuffer) throws -> ShadowMapUpdateCache.Use? {
        try updateCache?.beginRead(slice: slice, commandBuffer: commandBuffer)
    }

    /// Checks that depth passes can be encoded into this command buffer.
    package func validate(commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.commandQueue === queue else { throw Failure.wrongQueue }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
    }

    /// Starts the depth pass for one slice. Call `validate(commandBuffer:)` first.
    package func makeEncoder(
        slice: Int, commandBuffer: any MTLCommandBuffer,
        configure: ((MTLRenderPassDescriptor) throws -> Void)? = nil
    ) throws -> any MTLRenderCommandEncoder {
        let pass = MTLRenderPassDescriptor()
        pass.depthAttachment.texture = texture
        pass.depthAttachment.slice = slice
        pass.depthAttachment.clearDepth = 0
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        try configure?(pass)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        encoder.label = "Frostfield punctual shadow pass \(slice)"
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        return encoder
    }

    /// Reserve bounded moments storage only for maps with a variance spot assignment.
    func prepareVarianceMoments() throws -> any MTLTexture {
        if let variancePass { return variancePass.moments }
        guard spotCapacity > 0, resolution <= 2048, resolution * resolution * spotCapacity <= 2048 * 2048 * 4 else {
            throw Failure.varianceCapacity
        }
        do {
            let pass = try VarianceShadowPass(device: queue.device, resolution: resolution, slices: spotCapacity)
            variancePass = pass
            return pass.moments
        } catch {
            throw Failure.variance(error.localizedDescription)
        }
    }

    func encodeMoments(slices: [SIMD2<UInt32>], commandBuffer: any MTLCommandBuffer) throws {
        guard !slices.isEmpty else { return }
        guard let variancePass else { throw Failure.resource("spot shadow moments") }
        do {
            try variancePass.encode(depth: texture, slices: slices, commandBuffer: commandBuffer)
        } catch {
            throw Failure.variance(error.localizedDescription)
        }
    }

    package func setPipeline(on encoder: any MTLRenderCommandEncoder, skinned: Bool, masked: Bool) {
        if masked {
            encoder.setRenderPipelineState(skinned ? skinnedMaskedPipeline : maskedPipeline)
        } else {
            encoder.setRenderPipelineState(skinned ? skinnedPipeline : pipeline)
        }
    }

    /// The tangent of a view's half field of view, widened by a guard band of
    /// texels that holds the normal offset and the filter footprint at the edge.
    package func projectionTangent(covering tangent: Double, shadow: PunctualLight.Shadow) -> Double {
        let size = Double(resolution)
        let radius: Double = switch shadow.sampling {
        case .basic: 0
        case .filtered: Double(shadow.filterRadius)
        case .variance: Double(shadow.varianceBlurRadius)
        }
        let guardBand = min(ceil(Double(shadow.normalBias) + 2 * radius + 3), size / 4)
        return tangent * size / (size - 2 * guardBand)
    }

    /// A reversed, infinite-far perspective view from `origin` along `forward`.
    /// Clip space is `(x / t, y / t, near, depth)`, so normalized depth is `near / depth`.
    package static func projection(
        origin: SIMD3<Float>, forward: SIMD3<Float>, tangent: Double, near: Float
    ) throws -> simd_float4x4 {
        let matrix = HighPrecisionModelView.packed(projectionMatrix(
            origin: SIMD3<Double>(origin), forward: forward, tangent: tangent, near: near
        ))
        let (a, b, c, d) = matrix.columns
        let scale = 1 / tangent
        guard [a, b, c, d].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }),
              scale.isFinite, scale > 0 else {
            throw Failure.projection
        }
        return matrix
    }

    /// Light-relative form used for both the caster and the receiver.
    static func projectionMatrix(
        origin: SIMD3<Double> = .zero, forward: SIMD3<Float>, tangent: Double, near: Float
    ) -> simd_double4x4 {
        let f = simd_normalize(SIMD3<Double>(forward))
        let reference = abs(f.y) < 0.99 ? SIMD3<Double>(0, 1, 0) : SIMD3<Double>(1, 0, 0)
        let right = simd_normalize(simd_cross(reference, f))
        let up = simd_cross(f, right)
        let scale = 1 / tangent
        return simd_double4x4(
            SIMD4(right.x * scale, up.x * scale, 0, f.x),
            SIMD4(right.y * scale, up.y * scale, 0, f.y),
            SIMD4(right.z * scale, up.z * scale, 0, f.z),
            SIMD4(-simd_dot(right, origin) * scale, -simd_dot(up, origin) * scale,
                  Double(near), -simd_dot(f, origin))
        )
    }

    /// The six face directions of a point-light view, indexed as the shader selects them:
    /// +X, -X, +Y, -Y, +Z, -Z.
    package static let faceDirections: [SIMD3<Float>] = [
        SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
    ]

    public enum Failure: LocalizedError, Equatable {
        case resolution
        case capacity
        case library(String)
        case missingShader
        case pipeline(String)
        case resource(String)
        case projection
        case wrongQueue
        case commandBuffer
        case encoder
        case varianceCapacity
        case variance(String)

        public var errorDescription: String? {
            switch self {
            case .resolution: "Punctual shadow maps need a resolution of 16 to 4096 texels."
            case .capacity: "Use 0–8 spot and 0–4 point shadow slots, with at least one slot in total."
            case .library(let reason): "Could not load Frostfield's shader library: \(reason)"
            case .missingShader: "The shadow depth shaders are missing."
            case .pipeline(let reason): "Could not prepare the punctual shadow depth pipeline: \(reason)"
            case .resource(let name): "Metal could not allocate \(name)."
            case .projection: "A shadowed light's position and direction must give a finite shadow projection."
            case .wrongQueue: "Use a command buffer from the punctual shadow maps' queue."
            case .commandBuffer: "Use an uncommitted command buffer with retained references."
            case .encoder: "Metal could not create a punctual shadow render encoder."
            case .varianceCapacity:
                "Spot variance needs a resolution up to 2048 and at most 256 MiB of moments storage across the spot slots."
            case .variance(let reason): "Could not prepare or encode spot shadow moments. \(reason)"
            }
        }
    }
}
