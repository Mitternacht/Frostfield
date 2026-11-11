import Foundation
import Metal
import simd

/// A light-independent shadow layer for a finite receiver plane. It projects the
/// closest opaque caster above the plane, fades with height and optionally blurs.
/// Encode captures and draw their receivers in submission order on one queue.
public final class ContactShadows {
    public struct Settings: Sendable, Hashable {
        /// Multiplies the linear height fade before it is clamped to opacity.
        public var darkness: Float
        /// Gaussian standard deviation in mask pixels. Zero keeps the caster silhouette sharp.
        public var blurSigma: Float

        public init(darkness: Float = 1, blurSigma: Float = 3.5) {
            self.darkness = darkness
            self.blurSigma = blurSigma
        }
    }

    public let queue: any MTLCommandQueue
    public let plane: SceneTransform
    public let size: SIMD2<Float>
    public let maximumHeight: Float
    public let resolution: Int
    /// Tracked RGBA16Float, with coverage replicated in all channels. Encode a
    /// capture before sampling the texture or drawing its receiver for the first time.
    public let texture: any MTLTexture

    private let depth: any MTLTexture
    private let view: ShaderGraphCamera
    private let depthPipelines: MeshDepthCapturePipelines
    private let depthState: any MTLDepthStencilState
    private let maskPipeline: any MTLRenderPipelineState
    private let blur: GaussianBlur
    private var rawMask: (any MTLTexture)?

    /// The rigid pose places the receiver in local XY, with local +Z pointing
    /// toward casters. Size and maximum height are in world units.
    public init(
        queue: any MTLCommandQueue, plane: SceneTransform = SceneTransform(),
        size: SIMD2<Float> = SIMD2(repeating: 2.5), maximumHeight: Float = 0.3, resolution: Int = 512
    ) throws {
        let rotationLength = simd_length(plane.rotation.vector)
        guard plane.scale == SIMD3(repeating: 1), rotationLength.isNormal,
              plane.translation.x.isFinite, plane.translation.y.isFinite, plane.translation.z.isFinite,
              size.x.isNormal, size.y.isNormal, size.x > 0, size.y > 0,
              (2 / size.x).isFinite, (2 / size.y).isFinite,
              maximumHeight.isNormal, maximumHeight > 0, (1 / maximumHeight).isFinite,
              (8...2048).contains(resolution) else { throw Failure.plane }
        var pose = plane
        pose.rotation = simd_quatf(vector: plane.rotation.vector / rotationLength)
        // Look up from the plane. Flipping the camera's X and Z axes preserves
        // winding, and the receiver mirrors its U coordinate to match the capture.
        let turn = simd_float4x4(diagonal: SIMD4(-1, 1, -1, 1))
        let projection = simd_float4x4(diagonal: SIMD4(2 / size.x, 2 / size.y, -1 / maximumHeight, 1))
        view = try ShaderGraphCamera(
            viewMatrix: turn * simd_inverse(pose.matrix), projectionMatrix: projection, near: 0, far: maximumHeight
        )
        self.queue = queue
        self.plane = pose
        self.size = size
        self.maximumHeight = maximumHeight
        self.resolution = resolution
        let device = queue.device
        depthPipelines = try MeshDepthCapturePipelines(device: device, label: "Frostfield contact shadow depth")
        let state = MTLDepthStencilDescriptor()
        state.depthCompareFunction = .less
        state.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: state) else { throw Failure.resource }
        self.depthState = depthState
        depth = try PostProcessShaders.scratch(
            device: device, width: resolution, height: resolution, pixelFormat: .depth32Float,
            usage: [.renderTarget, .shaderRead], label: "Frostfield contact shadow depth"
        )
        texture = try Self.maskTarget(device: device, resolution: resolution, label: "Frostfield contact shadow mask")
        let shaders = try PostProcessShaders(device: device)
        maskPipeline = try shaders.render("frostfield_contact_shadow_mask", formats: [.rgba16Float])
        blur = try GaussianBlur(device: device)
    }

    /// Captures visible indexed casters in the graph's camera layers, honoring
    /// each node's castsShadow flag, PBR alpha masks and supported graph
    /// cutouts. Exclusions apply to the named nodes and preserve eligible descendants.
    /// The capture uses ordinary Float world transforms.
    @discardableResult
    public func encode(
        graph: SceneGraph, settings: Settings = Settings(), excluding: Set<UUID> = [],
        indirectDraws: [UUID: MeshIndirectDraws] = [:], commandBuffer: any MTLCommandBuffer
    ) throws -> Int {
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard (0...16).contains(settings.darkness), (0...GaussianBlur.maximumSigma).contains(settings.blurSigma) else {
            throw Failure.settings
        }
        let mask: any MTLTexture
        if settings.blurSigma > 0 {
            let raw = try rawMask ?? Self.maskTarget(device: queue.device, resolution: resolution, label: "Frostfield raw contact shadow mask")
            rawMask = raw
            mask = raw
            blur.settings.sigma = settings.blurSigma
            try blur.validate(source: mask, depth: nil, destination: texture, commandBuffer: commandBuffer)
        } else {
            mask = texture
        }

        let extent = SIMD2<Float>(repeating: Float(resolution))
        let context = GraphViewContext(
            cameraPosition: plane.translation, viewport: .init(origin: .zero, size: extent), targetSize: extent,
            camera: view, windowDepth: SIMD2(0, 1), depthMode: .forward
        )
        let drawCount = try MeshRenderer.encodeDepthOnly(
            graph, clipFromWorld: view.projectionMatrix * view.viewMatrix, commandBuffer: commandBuffer,
            includes: { !excluding.contains($0) }, indirectDraws: indirectDraws, shadowView: context,
            makeEncoder: {
                let pass = MTLRenderPassDescriptor()
                pass.depthAttachment.texture = depth
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.clearDepth = 1
                pass.depthAttachment.storeAction = .store
                guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.resource }
                encoder.label = "Frostfield contact shadow depth"
                encoder.setRenderPipelineState(depthPipelines.plain)
                encoder.setDepthStencilState(depthState)
                return encoder
            }, setPipeline: depthPipelines.setPipeline
        )
        try PostProcessShaders.drawFullScreen(
            into: mask, label: "Frostfield contact shadow mask", commandBuffer: commandBuffer, pipeline: maskPipeline
        ) { encoder in
            var darkness = settings.darkness
            encoder.setFragmentBytes(&darkness, length: MemoryLayout<Float>.stride, index: 0)
            encoder.setFragmentTexture(depth, index: 0)
        }
        if settings.blurSigma > 0 {
            try blur.encode(source: mask, destination: texture, commandBuffer: commandBuffer)
        }
        return drawCount
    }

    /// Prepares a tinted overlay for the receiver plane. Add this node to the
    /// visible scene after preparing the capture. It tests scene depth, blends
    /// over opaque geometry, and neither writes depth nor contributes to the mask.
    /// A small nonnegative offset can lift the overlay above a coincident floor.
    public func makeReceiver(
        colorPixelFormat: MTLPixelFormat, depthMode: MeshDepthMode = .forward,
        tint: SIMD3<Float> = .zero, opacity: Float = 1, offset: Float = 0, renderOrder: Int = 1
    ) throws -> SceneNode {
        guard (0..<3).allSatisfy({ (0...1).contains(tint[$0]) }), (0...1).contains(opacity),
              offset.isFinite, offset >= 0 else { throw Failure.settings }
        var pose = plane
        pose.translation += pose.rotation.act(SIMD3(0, 0, offset))
        guard (0..<3).allSatisfy({ pose.translation[$0].isFinite }) else { throw Failure.plane }
        let original = try MeshGeometry.plane(width: size.x, height: size.y)
        let geometry = try MeshGeometry(vertices: original.vertices.map { vertex in
            MeshVertex(position: vertex.position, normal: vertex.normal,
                       textureCoordinate: SIMD2(1 - vertex.textureCoordinate.x, vertex.textureCoordinate.y))
        }, indices: original.indices)
        let sampler = MTLSamplerDescriptor()
        sampler.minFilter = .linear
        sampler.magFilter = .linear
        sampler.sAddressMode = .clampToEdge
        sampler.tAddressMode = .clampToEdge
        guard let samplerState = queue.device.makeSamplerState(descriptor: sampler) else { throw Failure.resource }
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(tint, opacity), alphaMode: .blend, shadingModel: .unlit,
            alphaTexture: .init(texture: texture, sampler: samplerState)
        )
        let mesh = try MeshRenderer(
            device: queue.device, colorPixelFormat: colorPixelFormat, geometry: geometry,
            material: .physicallyBased(material), depthMode: depthMode,
            renderState: .init(blending: .normal, transparent: true, depthWrite: false, depthCompare: .lessEqual)
        )
        return SceneNode(transform: pose, mesh: mesh, castsShadow: false, renderOrder: renderOrder)
    }

    private static func maskTarget(device: any MTLDevice, resolution: Int, label: String) throws -> any MTLTexture {
        try PostProcessShaders.scratch(
            device: device, width: resolution, height: resolution, usage: [.renderTarget, .shaderRead], label: label
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case plane, commandBuffer, settings, resource

        public var errorDescription: String? {
            switch self {
            case .plane: "Contact shadows need a finite rigid receiver pose, positive extent and height, and a resolution from 8 to 2048."
            case .commandBuffer: "Use an uncommitted retaining command buffer on the contact shadow queue."
            case .settings: "Use a darkness from 0 to 16, Gaussian sigma from 0 to 64 pixels, normalized tint and opacity, and a nonnegative offset."
            case .resource: "Metal could not create a contact shadow resource or encoder."
            }
        }
    }
}
