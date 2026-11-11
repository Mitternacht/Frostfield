import Foundation
import Metal
import simd

/// A single reflected scene view on the positive side of an oriented world plane.
public struct PlanarReflection: Sendable {
    public let plane: Plane3D
    /// This node and its descendants are excluded from the capture.
    public let reflector: UUID

    public init(plane: Plane3D, reflector: UUID) {
        self.plane = plane
        self.reflector = reflector
    }

    /// Reusable tracked color and depth attachments for a multisampled capture.
    /// Reuse them on one ordered queue or after their earlier work has completed.
    public struct MultisampleTargets {
        fileprivate let color: any MTLTexture
        fileprivate let depth: any MTLTexture
        public var sampleCount: Int { color.sampleCount }

        public init(device: any MTLDevice, width: Int, height: Int, sampleCount: Int) throws {
            guard Self.isSupported(sampleCount: sampleCount, on: device) else { throw Failure.sampleCount }
            guard (1...16_384).contains(width), (1...16_384).contains(height) else { throw Failure.target }
            let descriptor = MTLTextureDescriptor()
            descriptor.textureType = .type2DMultisample
            descriptor.width = width
            descriptor.height = height
            descriptor.sampleCount = sampleCount
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = .renderTarget
            descriptor.pixelFormat = .rgba16Float
            guard let color = device.makeTexture(descriptor: descriptor) else { throw Failure.multisampleAllocation }
            descriptor.pixelFormat = .depth32Float
            guard let depth = device.makeTexture(descriptor: descriptor) else { throw Failure.multisampleAllocation }
            color.label = "Frostfield planar reflection multisample color"
            depth.label = "Frostfield planar reflection multisample depth"
            self.color = color
            self.depth = depth
        }

        /// Includes both the raster sample-count limit and native depth resolve.
        public static func isSupported(sampleCount: Int, on device: any MTLDevice) -> Bool {
            sampleCount > 1 && RendererCapabilities.supportsRasterSampleCount(sampleCount, on: device)
                && device.supportsFamily(.apple3)
        }
    }

    /// Renders the graph's visible meshes and primitives into the caller's
    /// single-sample, shader-readable RGBA16Float color and matching Depth32Float
    /// targets. Size the textures for the desired reflection resolution and use
    /// the source view's projection, including any crop. Draw the mirror later
    /// with `ShaderTexture.samplePlanarReflection(at:)` in the same viewport space.
    ///
    /// Exclude any additional reflective subtrees with `excluding`. Capture
    /// rendering does not invoke other reflection passes. A camera on the back
    /// side or a plane outside the forward view clears the targets and returns nil.
    /// An optional background uses the reflected camera before oblique clipping.
    /// Match the background and reflected scene pipelines to the sample count.
    /// Multisampling resolves color and the nearest depth into the caller's
    /// single-sample textures. Color mipmaps can be regenerated after capture.
    /// The command buffer retains resources and belongs to the targets' device.
    /// Discard it if encoding throws.
    @discardableResult
    public func encode(
        _ graph: SceneGraph, camera: ShaderGraphCamera, color: any MTLTexture, depth: any MTLTexture,
        depthMode: MeshDepthMode = .forward, excluding: Set<UUID> = [],
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        generateMipmaps: Bool = false, shadowMap: DirectionalShadowMap? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, transmissionBackdrop: (any MTLTexture)? = nil,
        background: EnvironmentBackground? = nil, backgroundOptions: EnvironmentBackground.Options = .init(),
        multisample: MultisampleTargets? = nil, commandBuffer: any MTLCommandBuffer
    ) throws -> SceneDrawStatistics? {
        try Self.encodeCapture(
            graph, view: { try reflectedView(camera, depthMode: depthMode) }, label: "reflection",
            color: color, depth: depth, depthMode: depthMode, excluding: excluding.union([reflector]), clearColor: clearColor,
            generateMipmaps: generateMipmaps, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps,
            transmissionBackdrop: transmissionBackdrop, background: background, backgroundOptions: backgroundOptions,
            multisample: multisample, commandBuffer: commandBuffer
        )
    }

    static func encodeCapture(
        _ graph: SceneGraph, view: () throws -> View?, label: String,
        color: any MTLTexture, depth: any MTLTexture, depthMode: MeshDepthMode, excluding: Set<UUID>,
        clearColor: MTLClearColor, generateMipmaps: Bool, shadowMap: DirectionalShadowMap?,
        punctualShadowMaps: PunctualShadowMaps?, transmissionBackdrop: (any MTLTexture)?,
        background: EnvironmentBackground?, backgroundOptions: EnvironmentBackground.Options,
        multisample: MultisampleTargets?, commandBuffer: any MTLCommandBuffer
    ) throws -> SceneDrawStatistics? {
        try validateCommandBuffer(commandBuffer)
        let targets = try captureTargets(
            color: color, depth: depth, multisample: multisample, clearColor: clearColor, device: commandBuffer.device
        )
        guard multisample == nil || transmissionBackdrop == nil else { throw Failure.multisampleTransmission }
        let otherTargets = [shadowMap?.texture, punctualShadowMaps?.texture, transmissionBackdrop].compactMap { $0 }
        guard !otherTargets.contains(where: { input in targets.contains { TextureTransfer.sharesStorage(input, $0) } })
        else { throw Failure.feedback }

        return try encodeView(
            view: view(), label: label, color: color, depth: depth, depthMode: depthMode,
            excluding: excluding, clearColor: clearColor, generateMipmaps: generateMipmaps, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps, transmissionBackdrop: transmissionBackdrop, background: background,
            backgroundOptions: backgroundOptions, multisample: multisample, forbiddenTextures: targets,
            commandBuffer: commandBuffer, scene: { graph }
        )
    }

    /// One mirror capture from a particular source view. A child sees the
    /// parent's reflected camera, and each occurrence needs its own attachments.
    public struct Capture {
        /// Optional lighting passes and opaque-color scratch for this view.
        /// Give each capture separate resources. Shadow maps use the capture's
        /// command queue. An active view fits cascades using the map's configured
        /// splits and its unclipped reflected camera.
        public struct Attachments {
            public let shadowMap: DirectionalShadowMap?
            public let punctualShadowMaps: PunctualShadowMaps?
            /// A tracked shader-readable RGBA16Float 2D texture with a complete
            /// mip chain and the capture color's dimensions. The opaque pass
            /// writes it before transmitting geometry draws. Use single sampling.
            public let transmissionBackdrop: (any MTLTexture)?

            public init(
                shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
                transmissionBackdrop: (any MTLTexture)? = nil
            ) {
                self.shadowMap = shadowMap
                self.punctualShadowMaps = punctualShadowMaps
                self.transmissionBackdrop = transmissionBackdrop
            }

            fileprivate func writeTargets(
                color: any MTLTexture, multisample: MultisampleTargets?, commandBuffer: any MTLCommandBuffer
            ) throws -> [any MTLTexture] {
                try shadowMap?.validate(commandBuffer: commandBuffer)
                try punctualShadowMaps?.validate(commandBuffer: commandBuffer)
                if let backdrop = transmissionBackdrop {
                    guard multisample == nil else { throw Failure.multisampleTransmission }
                    guard color.iosurface == nil else { throw MeshRenderer.Failure.transmissionAttachments }
                    let largestDimension = max(color.width, color.height)
                    let mipCount = Int.bitWidth - largestDimension.leadingZeroBitCount
                    guard backdrop.iosurface == nil, backdrop.device === commandBuffer.device,
                          backdrop.pixelFormat == .rgba16Float, PlanarReflection.isTarget(backdrop, usage: .shaderRead),
                          backdrop.width == color.width, backdrop.height == color.height, backdrop.mipmapLevelCount == mipCount
                    else { throw MeshRenderer.Failure.invalidTransmissionBackdrop }
                }
                return [
                    shadowMap?.texture, shadowMap?.sampling == .variance ? shadowMap?.shadowMoments : nil,
                    punctualShadowMaps?.texture, punctualShadowMaps?.shadowMoments, transmissionBackdrop,
                ].compactMap { $0 }
            }
        }

        struct Filter {
            let pass: PlanarReflectionBlur
            let settings: PlanarReflectionBlur.Settings
            let color: any MTLTexture
            let depthFactor: (any MTLTexture)?

            var outputs: [any MTLTexture] { [color] + [depthFactor].compactMap { $0 } }
        }

        var filter: Filter?

        public let reflection: PlanarReflection
        public let color: any MTLTexture
        public let depth: any MTLTexture
        public let children: [Capture]
        public let clearColor: MTLClearColor
        public let generateMipmaps: Bool
        public let multisample: MultisampleTargets?
        public let attachments: Attachments

        public init(
            _ reflection: PlanarReflection, color: any MTLTexture, depth: any MTLTexture, children: [Capture] = [],
            clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
            generateMipmaps: Bool = false, multisample: MultisampleTargets? = nil
        ) {
            self.init(
                reflection, color: color, depth: depth, children: children, clearColor: clearColor,
                generateMipmaps: generateMipmaps, multisample: multisample, attachments: .init()
            )
        }

        public init(
            _ reflection: PlanarReflection, color: any MTLTexture, depth: any MTLTexture, children: [Capture] = [],
            clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
            generateMipmaps: Bool = false, multisample: MultisampleTargets? = nil, attachments: Attachments
        ) {
            self.reflection = reflection
            self.color = color
            self.depth = depth
            self.children = children
            self.clearColor = clearColor
            self.generateMipmaps = generateMipmaps
            self.multisample = multisample
            self.attachments = attachments
        }
    }

    /// Encodes up to four reflection levels and thirty-two total views, children
    /// first. The provider receives the active view's reflector path and the
    /// encoded color captures of its direct children. Supply a graph whose
    /// visible mirrors bind those textures, with an authored fallback at leaves.
    /// Each capture excludes the entire reflector subtree at every node on its
    /// path. An inactive view clears its own attachments and skips its children.
    /// The result gives the root captures to bind when drawing the source view.
    /// Optional shadow and transmission attachments are private to their pass.
    /// Reuse the attachments on one ordered queue or after earlier reads finish.
    /// Discard the retained-reference command buffer if encoding throws.
    public static func encodeBounces(
        _ captures: [Capture], camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward,
        excluding: Set<UUID> = [], background: EnvironmentBackground? = nil,
        backgroundOptions: EnvironmentBackground.Options = .init(), commandBuffer: any MTLCommandBuffer,
        sceneForView: (_ path: [UUID], _ childColors: [UUID: any MTLTexture]) throws -> SceneGraph
    ) throws -> [UUID: any MTLTexture] {
        try validateCommandBuffer(commandBuffer)
        var targets: [any MTLTexture] = []
        var count = 0

        func prepare(_ captures: [Capture], camera: ShaderGraphCamera?, ancestors: Set<UUID>) throws -> [PreparedCapture] {
            var siblings = Set<UUID>()
            return try captures.map { capture in
                guard ancestors.count < 4, count < 32 else { throw Failure.captureLimit }
                count += 1
                let reflector = capture.reflection.reflector
                guard !ancestors.contains(reflector), siblings.insert(reflector).inserted else { throw Failure.duplicateReflector }
                let attachments = try captureTargets(
                    color: capture.color, depth: capture.depth, multisample: capture.multisample,
                    clearColor: capture.clearColor, device: commandBuffer.device
                )
                let writes = try capture.attachments.writeTargets(
                    color: capture.color, multisample: capture.multisample, commandBuffer: commandBuffer
                )
                for attachment in attachments + writes + (capture.filter?.outputs ?? []) {
                    guard !targets.contains(where: { TextureTransfer.sharesStorage(attachment, $0) }) else { throw Failure.feedback }
                    targets.append(attachment)
                }
                let view = try camera.flatMap { try capture.reflection.reflectedView($0, depthMode: depthMode) }
                let children = try prepare(capture.children, camera: view?.unclippedCamera(), ancestors: ancestors.union([reflector]))
                return PreparedCapture(capture: capture, view: view, children: children)
            }
        }

        let roots = try prepare(captures, camera: camera, ancestors: [])
        if let background, background.samplesStorage(of: targets, options: backgroundOptions) { throw Failure.feedback }

        func encode(_ views: [PreparedCapture], path: [UUID]) throws -> [UUID: any MTLTexture] {
            var colors: [UUID: any MTLTexture] = [:]
            for prepared in views {
                let capture = prepared.capture
                let reflector = capture.reflection.reflector
                let reflectorPath = path + [reflector]
                let childColors = prepared.view == nil ? [:] : try encode(prepared.children, path: reflectorPath)
                let shadowMap = capture.attachments.shadowMap
                if let view = prepared.view, let shadowMap, shadowMap.cascades != nil {
                    try shadowMap.fitCascades(view: view.unclippedCamera())
                }
                var permitted = Set(childColors.values.map(ObjectIdentifier.init))
                if prepared.view != nil {
                    for child in prepared.children {
                        permitted.formUnion((child.capture.filter?.outputs ?? []).map(ObjectIdentifier.init))
                    }
                }
                let forbidden = targets.filter { !permitted.contains(ObjectIdentifier($0)) }
                _ = try encodeView(
                    view: prepared.view, label: "reflection", color: capture.color, depth: capture.depth, depthMode: depthMode,
                    excluding: excluding.union(reflectorPath), clearColor: capture.clearColor, generateMipmaps: capture.generateMipmaps,
                    shadowMap: shadowMap, punctualShadowMaps: capture.attachments.punctualShadowMaps,
                    transmissionBackdrop: capture.attachments.transmissionBackdrop,
                    background: background, backgroundOptions: backgroundOptions, multisample: capture.multisample,
                    forbiddenTextures: forbidden, commandBuffer: commandBuffer,
                    scene: { try sceneForView(reflectorPath, childColors) }
                )
                if let filter = capture.filter {
                    filter.pass.settings = filter.settings
                    try filter.pass.encode(
                        source: capture.color, depth: filter.settings.depthScale > 0 ? capture.depth : nil,
                        destination: filter.color,
                        depthFactor: filter.depthFactor, depthMode: depthMode, commandBuffer: commandBuffer
                    )
                }
                colors[reflector] = capture.color
            }
            return colors
        }
        return try encode(roots, path: [])
    }

    private struct PreparedCapture {
        let capture: Capture
        let view: View?
        let children: [PreparedCapture]
    }

    private static func validateCommandBuffer(_ commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else { throw Failure.commandBuffer }
    }

    private static func captureTargets(
        color: any MTLTexture, depth: any MTLTexture, multisample: MultisampleTargets?,
        clearColor: MTLClearColor, device: any MTLDevice
    ) throws -> [any MTLTexture] {
        guard color.device === device, depth.device === device,
              color.pixelFormat == .rgba16Float, depth.pixelFormat == .depth32Float,
              Self.isTarget(color, usage: [.renderTarget, .shaderRead]), Self.isTarget(depth, usage: .renderTarget),
              color.width == depth.width, color.height == depth.height else { throw Failure.target }
        if let multisample {
            guard multisample.color.device === device,
                  multisample.color.width == color.width, multisample.color.height == color.height else { throw Failure.target }
        }
        guard !TextureTransfer.sharesStorage(color, depth) else { throw Failure.feedback }
        guard [clearColor.red, clearColor.green, clearColor.blue].allSatisfy({ (0...65_504).contains($0) }),
              (0...1).contains(clearColor.alpha) else { throw Failure.clearColor }
        return [color, depth] + [multisample?.color, multisample?.depth].compactMap { $0 }
    }

    private static func encodeView(
        view: View?, label: String, color: any MTLTexture, depth: any MTLTexture, depthMode: MeshDepthMode, excluding: Set<UUID>,
        clearColor: MTLClearColor, generateMipmaps: Bool, shadowMap: DirectionalShadowMap? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, transmissionBackdrop: (any MTLTexture)? = nil,
        background: EnvironmentBackground?, backgroundOptions: EnvironmentBackground.Options,
        multisample: MultisampleTargets?, forbiddenTextures: [any MTLTexture], commandBuffer: any MTLCommandBuffer,
        scene: () throws -> SceneGraph
    ) throws -> SceneDrawStatistics? {
        let pass = MTLRenderPassDescriptor()
        let colorAttachment = pass.colorAttachments[0]!
        colorAttachment.texture = multisample?.color ?? color
        colorAttachment.loadAction = .clear
        colorAttachment.storeAction = multisample == nil ? .store : .multisampleResolve
        colorAttachment.clearColor = clearColor
        let depthAttachment = pass.depthAttachment!
        depthAttachment.texture = multisample?.depth ?? depth
        depthAttachment.loadAction = .clear
        depthAttachment.storeAction = multisample == nil ? .store : .multisampleResolve
        depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1
        if multisample != nil {
            colorAttachment.resolveTexture = color
            depthAttachment.resolveTexture = depth
            depthAttachment.depthResolveFilter = depthMode == .reversed ? .max : .min
        }

        let statistics: SceneDrawStatistics?
        if let view {
            var capture = try scene()
            if let background {
                guard !background.samplesStorage(of: forbiddenTextures, options: backgroundOptions) else { throw Failure.feedback }
                try background.encode(
                    destination: multisample?.color ?? color, view: view.unclippedCamera(), options: backgroundOptions,
                    commandBuffer: commandBuffer
                )
                colorAttachment.loadAction = .load
            }
            // The caller's explicit projection already defines its crop and jitter.
            capture.projectionJitter = .zero
            statistics = try MeshRenderer.encode(
                capture, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap,
                punctualShadowMaps: punctualShadowMaps,
                view: (view.camera.projectionMatrix * view.camera.viewMatrix, view.position, view.directionToCamera),
                depthMode: depthMode, transmissionBackdrop: transmissionBackdrop, graphCamera: view.camera,
                excluding: excluding, reflectionTargets: forbiddenTextures
            )
        } else {
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
            encoder.label = "Frostfield inactive planar \(label)"
            encoder.endEncoding()
            statistics = nil
        }
        if generateMipmaps, color.mipmapLevelCount > 1 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
            blit.label = "Frostfield planar \(label) mipmaps"
            blit.generateMipmaps(for: color)
            blit.endEncoding()
        }
        return statistics
    }

    private static func isTarget(_ texture: any MTLTexture, usage: MTLTextureUsage) -> Bool {
        texture.textureType == .type2D && texture.sampleCount == 1 && texture.usage.contains(usage)
            && !texture.isFramebufferOnly && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
            && !texture.isAliasable() && texture.parent == nil && texture.buffer == nil && texture.heap == nil
            && texture.isSparse != true
    }

    public struct View {
        /// The split view and oblique Metal projection used to write capture depth.
        public let camera: ShaderGraphCamera
        public let position: SIMD3<Float>
        /// Parallel direction toward the reflected camera for an orthographic view.
        public let directionToCamera: SIMD3<Float>?
        fileprivate let skyProjection: simd_float4x4

        func unclippedCamera() throws -> ShaderGraphCamera {
            try ShaderGraphCamera(
                viewMatrix: camera.viewMatrix, projectionMatrix: skyProjection, near: camera.near, far: camera.far
            )
        }
    }

    /// Computes the same virtual camera used by `encode`. Nil means the camera
    /// is on the back of the mirror or its positive half-space misses the view.
    public func reflectedView(_ camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward) throws -> View? {
        try Self.captureView(camera, plane: plane, operation: .reflection, depthMode: depthMode)
    }

    enum CaptureOperation {
        case reflection
        case refraction(clipBias: Float)
    }

    static func captureView(
        _ camera: ShaderGraphCamera, plane: Plane3D, operation: CaptureOperation, depthMode: MeshDepthMode
    ) throws -> View? {
        guard depthMode == .forward || depthMode == .reversed else { throw Failure.projection }
        func widened(_ matrix: simd_float4x4) -> simd_double4x4 {
            simd_double4x4(SIMD4<Double>(matrix[0]), SIMD4<Double>(matrix[1]), SIMD4<Double>(matrix[2]), SIMD4<Double>(matrix[3]))
        }
        func xyz(_ vector: SIMD4<Double>) -> SIMD3<Double> { SIMD3(vector.x, vector.y, vector.z) }

        let sourceView = widened(camera.viewMatrix)
        var view = sourceView
        let basis = simd_double3x3(xyz(view[0]), xyz(view[1]), xyz(view[2]))
        let gram = simd_transpose(basis) * basis
        guard view[0].w == 0, view[1].w == 0, view[2].w == 0, abs(view[3].w - 1) <= 1e-6,
              simd_determinant(basis) > 0,
              (0..<3).allSatisfy({ simd_reduce_max(abs(gram[$0] - matrix_identity_double3x3[$0])) <= 1e-4 })
        else { throw Failure.camera }
        view[3].w = 1

        let projection = widened(camera.projectionMatrix)
        let a = projection[2].z, b = projection[3].z, c = projection[2].w, d = projection[3].w
        let orthographic = c == 0 && d > 0
        let derivative = a * d - b * c
        guard projection[0].z == 0, projection[1].z == 0, projection[0].w == 0, projection[1].w == 0,
              orthographic || (c < 0 && d == 0),
              depthMode == .reversed ? derivative > 0 : derivative < 0 else { throw Failure.projection }

        let worldFromCamera = simd_inverse(view)
        let n = plane.unitNormal
        let distance = simd_dot(n, xyz(worldFromCamera[3])) + plane.constant
        guard distance.isFinite else { throw Failure.projection }
        guard distance > 0 else { return nil }

        let captureView: simd_double4x4
        let worldFromCapture: simd_double4x4
        let skyProjection: simd_double4x4
        let capturePosition: SIMD3<Double>
        let worldPlane: SIMD4<Double>
        switch operation {
        case .reflection:
            var reflection = matrix_identity_double4x4
            for axis in 0..<3 { reflection[axis] -= SIMD4(n * (2 * n[axis]), 0) }
            reflection[3] -= SIMD4(n * (2 * plane.constant), 0)
            var flipX = matrix_identity_double4x4
            flipX[0].x = -1
            // Reflection and camera-space X flip together preserve front-face winding.
            captureView = flipX * view * reflection
            worldFromCapture = reflection * worldFromCamera * flipX
            capturePosition = xyz(worldFromCapture[3])
            // Conjugating the projection mirrors its authored center offset as well.
            skyProjection = flipX * projection * flipX
            worldPlane = SIMD4(n, plane.constant)
        case .refraction(let clipBias):
            // Preserve the caller's homogeneous matrix, including round-off in w.
            captureView = sourceView
            worldFromCapture = simd_inverse(sourceView)
            capturePosition = xyz(worldFromCamera[3])
            skyProjection = projection
            worldPlane = SIMD4(-n, -plane.constant - Double(clipBias))
        }
        var clipped = skyProjection
        let clipPlane = simd_transpose(worldFromCapture) * worldPlane
        let inverse = simd_inverse(clipped)

        // The kept half-space starts at the new Metal near plane. Use the far
        // corner with the largest positive plane value to place the opposite clip plane.
        let far: Double = depthMode == .reversed ? 0 : 1
        let corners: [SIMD4<Double>] = [
            SIMD4(-1, -1, far, 1), SIMD4(1, -1, far, 1),
            SIMD4(-1, 1, far, 1), SIMD4(1, 1, far, 1),
        ]
        let scale = corners.map { simd_dot(clipPlane, inverse * $0) }.max() ?? 0
        guard scale.isFinite else { throw Failure.projection }
        guard scale > 0 else { return nil }
        let nearRow = clipPlane / scale
        for column in 0..<4 {
            clipped[column].z = depthMode == .reversed ? clipped[column].w - nearRow[column] : nearRow[column]
        }

        let position = SIMD3<Float>(capturePosition)
        guard [position.x, position.y, position.z].allSatisfy(\.isFinite) else { throw Failure.projection }
        let clippedCamera: ShaderGraphCamera
        do {
            clippedCamera = try ShaderGraphCamera(
                viewMatrix: HighPrecisionModelView.packed(captureView),
                projectionMatrix: HighPrecisionModelView.packed(clipped), near: camera.near, far: camera.far
            )
        } catch { throw Failure.projection }
        return View(
            camera: clippedCamera, position: position,
            directionToCamera: orthographic ? SIMD3<Float>(xyz(worldFromCapture[2])) : nil,
            skyProjection: HighPrecisionModelView.packed(skyProjection)
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case commandBuffer, target, feedback, camera, projection, clearColor, encoder
        case sampleCount, multisampleAllocation, multisampleTransmission
        case captureLimit, duplicateReflector

        public var errorDescription: String? {
            switch self {
            case .commandBuffer: "Use an uncommitted command buffer with retained references."
            case .target: "Use separate tracked single-sample RGBA16Float and Depth32Float 2D targets of the same size."
            case .feedback: "A planar reflection target shares storage with a resource used by the reflected pass."
            case .camera: "Planar reflection requires a proper rigid camera view matrix."
            case .projection: "Use an invertible perspective or orthographic Metal projection with the matching depth mapping."
            case .clearColor: "Use finite clear color values with RGB in 0...65504 and alpha in 0...1."
            case .encoder: "Metal could not create a planar reflection encoder."
            case .sampleCount: "Use a supported multisample count on a GPU with native depth resolve."
            case .multisampleAllocation: "Metal could not allocate the planar multisample attachments."
            case .multisampleTransmission: "Planar transmission requires a single-sample capture."
            case .captureLimit: "Use up to four reflection levels and thirty-two captures."
            case .duplicateReflector: "A reflector can appear only once among siblings and along each capture path."
            }
        }
    }
}

public extension ShaderTexture {
    /// Samples a `PlanarReflection` capture at the main surface's viewport
    /// coordinates. A capture uses the main view's projection window and flips
    /// X once so the virtual camera can keep right-handed geometry winding.
    func samplePlanarReflection(at coordinates: ShaderValue<SIMD2<Float>>? = nil) -> ShaderValue<SIMD4<Float>> {
        let uv = coordinates ?? builder.screenUV
        return sample(builder.vector(1 - uv.x, uv.y))
    }
}
