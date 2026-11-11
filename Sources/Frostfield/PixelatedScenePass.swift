import CoreGraphics
import Metal
import simd

/// Renders a scene into a reduced grid and expands its texels as integer pixel
/// blocks. The pass owns one reusable capture for the current grid size.
public final class PixelatedScenePass {
    public let queue: any MTLCommandQueue
    public var settings: PixelatedEdgesPass.Settings

    private let capture: PostProcessScenePass
    private let edges: PixelatedEdgesPass
    private let shaders: PostProcessShaders
    private var surface: PostProcessScenePass.Targets?
    private var resolved: (any MTLTexture)?

    public init(queue: any MTLCommandQueue, settings: PixelatedEdgesPass.Settings = .init()) throws {
        self.queue = queue
        self.settings = settings
        capture = PostProcessScenePass(queue: queue)
        edges = try PixelatedEdgesPass(device: queue.device, settings: settings)
        shaders = try PostProcessShaders(device: queue.device)
    }

    /// Encodes reduced scene color, depth and optional normals, then writes
    /// the nearest blocks into a top-left normalized `viewport` within the
    /// destination. Viewport edges round to pixels and partial blocks clip.
    ///
    /// The scene camera keeps the output viewport's aspect ratio. An explicit
    /// camera can supply an orthographic or off-axis view in the chosen depth mode.
    /// Provide a premultiplied linear clear color and a matching reduced-size
    /// transmission backdrop when needed. The returned frame contains the reduced
    /// scene textures before edge shading. Consume them before the next encode.
    ///
    /// Encode on retained buffers from this pass's queue. Serialize calls and
    /// discard the command buffer if a scene draw or encoder fails.
    @discardableResult
    public func encode(
        _ scene: SceneGraph, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        viewport: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1),
        camera customCamera: ShaderGraphCamera? = nil, clearColor: MTLClearColor = MTLClearColor(),
        shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        transmissionBackdrop: (any MTLTexture)? = nil
    ) throws -> PostProcessScenePass.Frame {
        try settings.check()
        let device = queue.device
        guard destination.device === device else { throw PostProcessEffectError.wrongDevice }
        guard destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              PostProcessPass.supportedPixelFormats.contains(destination.pixelFormat) else {
            throw PostProcessEffectError.destination
        }
        let rect = try rasterViewport(viewport, width: destination.width, height: destination.height)
        let block = settings.pixelSize
        let grid = SIMD2((rect.width + block - 1) / block, (rect.height + block - 1) / block)
        let targets = try capture.prepare(
            width: rect.width, height: rect.height, commandBuffer: commandBuffer, captureSize: grid
        )
        guard !TextureTransfer.sharesStorage(targets.color, destination) else { throw PostProcessEffectError.destination }
        let camera = try customCamera ?? PostProcessScenePass.camera(
            for: scene, width: rect.width, height: rect.height, depthMode: settings.depthMode
        )
        let hasNormals = settings.normalEdgeStrength > 0
        let hasEdges = hasNormals || settings.depthEdgeStrength > 0
        let outputPipeline = try shaders.render("frostfield_pixelated_scene_expand", formats: [destination.pixelFormat])

        var normalTargets: PostProcessScenePass.Targets?
        let worldFromView = customCamera.map { simd_inverse($0.viewMatrix) } ?? scene.cameraTransform.matrix
        if hasNormals {
            guard let viewFromWorld = ScreenSpaceProjection.viewFromWorld(worldFromView) else {
                throw PostProcessEffectError.invalidSettings("Pixelation needs a rigid camera view for normal edges.")
            }
            edges.cameraRotation = simd_quatf(viewFromWorld.transpose)
            normalTargets = try normalCapture(size: grid)
        }
        let styled: (any MTLTexture)?
        if hasEdges {
            try edges.prepare(pixelFormat: .rgba16Float)
            styled = try resolvedCapture(size: grid)
        } else {
            styled = nil
        }

        let frame = try capture.encode(
            scene, width: rect.width, height: rect.height, targets: targets, commandBuffer: commandBuffer,
            camera: customCamera, depthMode: settings.depthMode, clearColor: clearColor, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps, transmissionBackdrop: transmissionBackdrop
        )
        if let normalTargets {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = normalTargets.color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor()
            pass.depthAttachment.texture = normalTargets.depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = settings.depthMode == .reversed ? 0 : 1
            let origin = worldFromView.columns.3
            let z = worldFromView.columns.2
            let direction = camera.projectionMatrix.columns.2.w == 0 ? SIMD3(z.x, z.y, z.z) : nil
            try scene.encodeSurfaceAttributes(
                pass: pass, commandBuffer: commandBuffer, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
                cameraPosition: SIMD3(origin.x, origin.y, origin.z), directionToCamera: direction,
                depthMode: settings.depthMode, graphCamera: camera
            )
        }
        if let styled {
            edges.settings = settings
            edges.settings.pixelSize = 1
            edges.surfaceAttributes = normalTargets?.color
            try edges.encode(
                source: frame.color, depth: frame.depth, normalDepth: normalTargets?.depth,
                destination: styled, commandBuffer: commandBuffer
            )
        }

        let partial = rect.width != destination.width || rect.height != destination.height
        var parameters = SIMD4<UInt32>(UInt32(rect.x), UInt32(rect.y), UInt32(block), 0)
        try PostProcessShaders.drawFullScreen(
            into: destination, load: partial, label: "Frostfield pixelated scene", commandBuffer: commandBuffer,
            pipeline: outputPipeline
        ) { encoder in
            encoder.setViewport(MTLViewport(
                originX: Double(rect.x), originY: Double(rect.y), width: Double(rect.width), height: Double(rect.height),
                znear: 0, zfar: 1
            ))
            encoder.setScissorRect(rect)
            encoder.setFragmentTexture(styled ?? frame.color, index: 0)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
        }
        return frame
    }

    private func rasterViewport(_ rect: CGRect, width: Int, height: Int) throws -> MTLScissorRect {
        let x = rect.origin.x, y = rect.origin.y
        let w = rect.size.width, h = rect.size.height
        guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, x >= 0, y >= 0,
              w > 0, h > 0, x + w <= 1, y + h <= 1 else {
            throw PostProcessEffectError.invalidSettings("Pixelation needs a finite positive viewport within the unit rectangle.")
        }
        let minX = Int((x * CGFloat(width)).rounded()), minY = Int((y * CGFloat(height)).rounded())
        let maxX = Int(((x + w) * CGFloat(width)).rounded()), maxY = Int(((y + h) * CGFloat(height)).rounded())
        guard minX < maxX, minY < maxY else {
            throw PostProcessEffectError.invalidSettings("The pixelation viewport must cover at least one output pixel.")
        }
        return MTLScissorRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private func normalCapture(size: SIMD2<Int>) throws -> PostProcessScenePass.Targets {
        if let surface, surface.color.width == size.x, surface.color.height == size.y { return surface }
        let targets = try PostProcessScenePass.Targets(
            color: attachment(format: SceneGraph.surfaceAttributesPixelFormat, size: size, label: "Frostfield pixelation normals"),
            depth: attachment(format: .depth32Float, size: size, label: "Frostfield pixelation normal depth")
        )
        surface = targets
        return targets
    }

    private func resolvedCapture(size: SIMD2<Int>) throws -> any MTLTexture {
        if let resolved, resolved.width == size.x, resolved.height == size.y { return resolved }
        let texture = try attachment(format: .rgba16Float, size: size, label: "Frostfield pixelation styled capture")
        resolved = texture
        return texture
    }

    private func attachment(format: MTLPixelFormat, size: SIMD2<Int>, label: String) throws -> any MTLTexture {
        try PostProcessShaders.scratch(
            device: queue.device, width: size.x, height: size.y, pixelFormat: format,
            usage: [.renderTarget, .shaderRead], label: label
        )
    }
}
