import Foundation
import Metal
import simd

/// Captures a scene's linear color and stored depth for full-screen effects.
/// Each pass owns its attachments and can select its own layers and overrides.
/// Serialize calls on one queue. Consume the returned textures before the next
/// capture at the same size, which reuses their storage.
public final class PostProcessScenePass {
    public struct Frame {
        public let color: any MTLTexture
        public let depth: any MTLTexture
        /// The camera used to write depth, including the scene's projection jitter.
        public let camera: ShaderGraphCamera
        public let depthMode: MeshDepthMode
        public let statistics: SceneDrawStatistics
    }

    public let queue: any MTLCommandQueue
    public private(set) var resolutionScale: Float = 1
    /// Nil keeps the layers selected by the scene's camera.
    public var layers: SceneLayers?
    /// Nil keeps the scene's existing mesh override.
    public var overrideMaterial: SceneMaterialOverride?
    /// Nil keeps the scene's existing point, line and sprite overrides.
    public var primitiveMaterialOverride: ScenePrimitiveMaterialOverride?

    private var targets: Targets?

    public init(queue: any MTLCommandQueue) {
        self.queue = queue
    }

    /// Captures each side at `round(outputSize * scale)`, with a minimum of one
    /// texel. The scene camera retains the unscaled output's aspect ratio.
    public func setResolutionScale(_ scale: Float) throws {
        guard scale.isFinite, (0.125...1).contains(scale) else { throw Failure.resolutionScale }
        resolutionScale = scale
    }

    /// Encodes to reusable tracked RGBA16Float and Depth32Float targets.
    /// Pass a custom camera for orthographic or off-axis views. Its projection
    /// must use the chosen depth convention. Clear color is premultiplied linear.
    /// A transmission backdrop, if needed, must match the scaled targets.
    public func encode(
        _ scene: SceneGraph, width: Int, height: Int, commandBuffer: any MTLCommandBuffer,
        camera: ShaderGraphCamera? = nil, depthMode: MeshDepthMode = .forward,
        clearColor: MTLClearColor = MTLClearColor(), shadowMap: DirectionalShadowMap? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, transmissionBackdrop: (any MTLTexture)? = nil
    ) throws -> Frame {
        let targets = try prepare(width: width, height: height, commandBuffer: commandBuffer)
        return try encode(
            scene, width: width, height: height, targets: targets, commandBuffer: commandBuffer,
            camera: camera, depthMode: depthMode, clearColor: clearColor, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps, transmissionBackdrop: transmissionBackdrop
        )
    }

    struct Targets {
        let color: any MTLTexture
        let depth: any MTLTexture
    }

    func prepare(
        width: Int, height: Int, commandBuffer: any MTLCommandBuffer, captureSize: SIMD2<Int>? = nil
    ) throws -> Targets {
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard (1...16_384).contains(width), (1...16_384).contains(height) else { throw Failure.size }
        let w = captureSize?.x ?? max(1, Int((Float(width) * resolutionScale).rounded()))
        let h = captureSize?.y ?? max(1, Int((Float(height) * resolutionScale).rounded()))
        guard (1...width).contains(w), (1...height).contains(h) else { throw Failure.size }
        if let targets, targets.color.width == w, targets.color.height == h { return targets }

        func makeTexture(_ format: MTLPixelFormat, label: String) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = [.renderTarget, .shaderRead]
            guard let texture = queue.device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
            texture.label = label
            return texture
        }
        let new = try Targets(
            color: makeTexture(.rgba16Float, label: "Frostfield scene pass color"),
            depth: makeTexture(.depth32Float, label: "Frostfield scene pass depth")
        )
        targets = new
        return new
    }

    func encode(
        _ scene: SceneGraph, width: Int, height: Int, targets: Targets, commandBuffer: any MTLCommandBuffer,
        camera customCamera: ShaderGraphCamera?, depthMode: MeshDepthMode, clearColor: MTLClearColor,
        shadowMap: DirectionalShadowMap?, punctualShadowMaps: PunctualShadowMaps?,
        transmissionBackdrop: (any MTLTexture)?
    ) throws -> Frame {
        var graph = scene
        graph.cameraLayers = layers ?? graph.cameraLayers
        graph.overrideMaterial = overrideMaterial ?? graph.overrideMaterial
        graph.primitiveMaterialOverride = primitiveMaterialOverride ?? graph.primitiveMaterialOverride

        let camera = try customCamera ?? Self.camera(for: graph, width: width, height: height, depthMode: depthMode)
        let worldFromView = customCamera.map { simd_inverse($0.viewMatrix) } ?? graph.cameraTransform.matrix
        let origin = worldFromView.columns.3
        let z = worldFromView.columns.2
        let directionToCamera = camera.projectionMatrix.columns.2.w == 0 ? SIMD3(z.x, z.y, z.z) : nil
        let frameCamera: ShaderGraphCamera
        if graph.projectionJitter == .zero {
            frameCamera = camera
        } else {
            guard TemporalAntiAliasing.isValidJitter(graph.projectionJitter) else {
                throw MeshRenderer.Failure.projectionJitter
            }
            let projection = TemporalAntiAliasing.jitteredClipFromWorld(
                camera.projectionMatrix, jitter: graph.projectionJitter,
                width: Double(targets.color.width), height: Double(targets.color.height)
            )
            frameCamera = try ShaderGraphCamera(
                viewMatrix: camera.viewMatrix, projectionMatrix: projection, near: camera.near, far: camera.far
            )
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clearColor
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1
        let statistics = try graph.encode(
            pass: pass, commandBuffer: commandBuffer, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
            cameraPosition: SIMD3(origin.x, origin.y, origin.z), directionToCamera: directionToCamera,
            depthMode: depthMode, shadowMap: shadowMap, transmissionBackdrop: transmissionBackdrop,
            punctualShadowMaps: punctualShadowMaps, graphCamera: camera
        )
        return Frame(color: targets.color, depth: targets.depth, camera: frameCamera, depthMode: depthMode, statistics: statistics)
    }

    static func camera(for scene: SceneGraph, width: Int, height: Int, depthMode: MeshDepthMode) throws -> ShaderGraphCamera {
        try ShaderGraphCamera(
            viewMatrix: simd_inverse(scene.cameraTransform.matrix),
            projectionMatrix: scene.camera.projection(aspectRatio: Float(width) / Float(height), depthMode: depthMode),
            near: scene.camera.near, far: scene.camera.far
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case commandBuffer, resolutionScale, size, allocation

        public var errorDescription: String? {
            switch self {
            case .commandBuffer: "Use an uncommitted command buffer from the scene pass's queue that retains its references."
            case .resolutionScale: "The scene pass resolution scale must be from 0.125 to 1."
            case .size: "A scene pass needs an output size from 1 to 16384 texels per side."
            case .allocation: "Metal could not allocate the scene pass attachments."
            }
        }
    }
}
