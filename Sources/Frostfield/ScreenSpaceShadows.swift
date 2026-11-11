import Foundation
import Metal
import simd

/// Short, light-directed contact shadows from a raster view's stored depth.
/// The pass can complement a directional shadow map with visible local detail.
/// Serialize encodes and order producers and consumers on the same Metal queue.
public final class ScreenSpaceShadows {
    public struct Settings: Sendable, Hashable {
        /// Ray length in world units.
        public var maximumDistance: Float
        /// Maximum separation behind the visible surface, along the camera axis.
        public var thickness: Float
        /// Reject separations this small or smaller, in the same units as thickness.
        public var depthBias: Float
        /// Fraction of directional light removed at a hit, from 0 to 1.
        public var intensity: Float
        /// Samples per pixel of projected ray length, from 0 to 1.
        public var quality: Float
        /// Upper bound on samples per ray, from 1 to 512.
        public var maximumSteps: Int
        /// Varies sample positions across frames. Nil keeps a fixed spatial pattern.
        public var frameIndex: UInt32?

        public init(
            maximumDistance: Float = 0.1, thickness: Float = 0.01, depthBias: Float = 0.001,
            intensity: Float = 1, quality: Float = 0.5, maximumSteps: Int = 256, frameIndex: UInt32? = nil
        ) {
            self.maximumDistance = maximumDistance
            self.thickness = thickness
            self.depthBias = depthBias
            self.intensity = intensity
            self.quality = quality
            self.maximumSteps = maximumSteps
            self.frameIndex = frameIndex
        }

        fileprivate var isValid: Bool {
            maximumDistance.isFinite && maximumDistance > 0 && thickness.isFinite && thickness > 0
                && depthBias.isFinite && depthBias >= 0 && depthBias < thickness
                && (0...1).contains(intensity) && (0...1).contains(quality) && (1...512).contains(maximumSteps)
        }
    }

    /// Registers a visibility mask with the world-space light direction used to
    /// trace it. A scene encode applies it to matching opaque and masked PBR
    /// directional lighting. Use the depth and projection of that same view.
    public struct DirectionalVisibility {
        public let texture: any MTLTexture
        public let directionToLight: SIMD3<Float>

        public init(texture: any MTLTexture, directionToLight: SIMD3<Float>) throws {
            guard let unit = ScreenSpaceShadows.unitDirection(directionToLight) else { throw Failure.parameters }
            guard ScreenSpaceShadows.visibilityPixelFormats.contains(texture.pixelFormat),
                  ScreenSpaceShadows.isReadable(texture, format: texture.pixelFormat) else { throw Failure.attachments }
            self.texture = texture
            self.directionToLight = unit
        }

        func applies(to light: SIMD4<Float>) -> Bool {
            light.w > 0 && simd_length_squared(SIMD3(light.x, light.y, light.z) - directionToLight) <= 1e-8
        }

        func prepare(
            pass: MTLRenderPassDescriptor, size: SIMD2<Int>, viewport: MTLViewport?, commandBuffer: any MTLCommandBuffer
        ) throws -> PreparedDirectionalVisibility {
            guard texture.device === commandBuffer.device else { throw Failure.device }
            guard commandBuffer.retainedReferences else { throw Failure.commandBuffer }
            guard !texture.isAliasable(), texture.width <= size.x, texture.height <= size.y,
                  pass.rasterizationRateMap == nil, pass.renderTargetArrayLength <= 1,
                  viewport.map({
                      $0.originX == 0 && $0.originY == 0 && $0.width == Double(size.x) && $0.height == Double(size.y)
                          && $0.znear == 0 && $0.zfar == 1
                  }) ?? true else { throw Failure.attachments }
            var attachments: [MTLRenderPassAttachmentDescriptor] = (0..<8).map { pass.colorAttachments[$0]! }
            attachments.append(pass.depthAttachment)
            attachments.append(pass.stencilAttachment)
            guard !attachments.contains(where: { attachment in
                [attachment.texture, attachment.resolveTexture].contains { candidate in
                    candidate.map { TextureTransfer.sharesStorage(texture, $0) } ?? false
                }
            }) else { throw Failure.attachments }
            return PreparedDirectionalVisibility(value: self, extent: SIMD2<Float>(size))
        }
    }

    struct PreparedDirectionalVisibility {
        let value: DirectionalVisibility
        let extent: SIMD2<Float>
    }

    public static let visibilityPixelFormats: [MTLPixelFormat] = [.r8Unorm, .r16Float, .rgba16Float]
    public let device: any MTLDevice
    private let shaders: PostProcessShaders

    public init(device: any MTLDevice) throws {
        self.device = device
        shaders = try PostProcessShaders(device: device)
    }

    /// Writes a mask whose red channel is one for fully visible directional light.
    /// Supply the camera and stored depth of the same raster view. A reduced-size
    /// destination reduces the march grid. RGBA16Float also supports GaussianBlur.
    /// Direction points from the scene toward the light. The view can be off-center
    /// perspective or axis-aligned orthographic. Logarithmic perspective uses the
    /// authored finite clip planes and forward clip matrix from the rasterizer.
    public func encodeVisibility(
        depth: any MTLTexture, destination: any MTLTexture, camera: ShaderGraphCamera,
        depthMode: MeshDepthMode = .forward, directionToLight: SIMD3<Float>,
        settings: Settings = Settings(), commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(commandBuffer)
        guard depth.device === device, destination.device === device else { throw Failure.device }
        guard Self.visibilityPixelFormats.contains(destination.pixelFormat),
              Self.isReadable(depth, format: .depth32Float), Self.isTarget(destination),
              destination.width <= depth.width, destination.height <= depth.height,
              !TextureTransfer.sharesStorage(depth, destination) else { throw Failure.attachments }
        guard settings.isValid, let direction = Self.unitDirection(directionToLight) else { throw Failure.parameters }
        let view = camera.viewMatrix
        guard view[0].w == 0, view[1].w == 0, view[2].w == 0, abs(view[3].w - 1) < 1e-5,
              let rotation = ScreenSpaceProjection.viewFromWorld(simd_inverse(view)),
              let projection = Projection(camera: camera, mode: depthMode) else { throw Failure.camera }
        let ray = rotation * direction
        let pipeline = try shaders.render("frostfield_screen_shadow_visibility", formats: [destination.pixelFormat])
        var uniforms = Uniforms(
            projection: projection.screen, depth: projection.decode, ray: SIMD4(ray, settings.maximumDistance),
            settings: SIMD4(settings.thickness, settings.depthBias, settings.intensity, settings.quality),
            extent: SIMD4(Float(destination.width), Float(destination.height), projection.near, projection.far),
            sampling: SIMD4(UInt32(settings.maximumSteps), settings.frameIndex ?? 0, 0, 0)
        )
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield screen-space directional visibility", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(depth, index: 0)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        }
    }

    /// Convenience for a graph-owned perspective view without additional jitter.
    public func encodeVisibility(
        depth: any MTLTexture, destination: any MTLTexture, camera: PerspectiveCamera,
        cameraTransform: SceneTransform, depthMode: MeshDepthMode = .forward, directionToLight: SIMD3<Float>,
        settings: Settings = Settings(), commandBuffer: any MTLCommandBuffer
    ) throws {
        let projection = try camera.projection(aspectRatio: Float(depth.width) / Float(depth.height), depthMode: depthMode)
        let view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix), projectionMatrix: projection, near: camera.near, far: camera.far
        )
        try encodeVisibility(
            depth: depth, destination: destination, camera: view, depthMode: depthMode,
            directionToLight: directionToLight, settings: settings, commandBuffer: commandBuffer
        )
    }

    /// Attenuates one directional contribution already included in the beauty.
    /// Beauty and directional radiance are equal-sized, premultiplied linear RGBA16Float.
    /// Radiance must include any shadow-map visibility already applied to that light.
    /// Pass beauty for both inputs to attenuate its whole RGB. Beauty alpha is preserved.
    /// A smaller visibility mask is sampled bilinearly in the matching view.
    public func encodeComposite(
        beauty: any MTLTexture, directionalRadiance: any MTLTexture, visibility: any MTLTexture,
        destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(commandBuffer)
        guard [beauty, directionalRadiance, visibility, destination].allSatisfy({ $0.device === device }) else {
            throw Failure.device
        }
        guard Self.isReadable(beauty, format: .rgba16Float), Self.isReadable(directionalRadiance, format: .rgba16Float),
              Self.visibilityPixelFormats.contains(visibility.pixelFormat),
              Self.isReadable(visibility, format: visibility.pixelFormat), Self.isTarget(destination),
              PostProcessPass.supportedPixelFormats.contains(destination.pixelFormat),
              [directionalRadiance, destination].allSatisfy({ $0.width == beauty.width && $0.height == beauty.height }),
              visibility.width <= beauty.width, visibility.height <= beauty.height,
              [beauty, directionalRadiance, visibility].allSatisfy({ !TextureTransfer.sharesStorage($0, destination) })
        else { throw Failure.attachments }
        let pipeline = try shaders.render("frostfield_screen_shadow_composite", formats: [destination.pixelFormat])
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield screen-space directional shadow composite", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(beauty, index: 0)
            encoder.setFragmentTexture(directionalRadiance, index: 1)
            encoder.setFragmentTexture(visibility, index: 2)
        }
    }

    private func validate(_ command: any MTLCommandBuffer) throws {
        guard command.retainedReferences, command.status == .notEnqueued || command.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard command.device === device else { throw Failure.device }
    }

    private static func unitDirection(_ value: SIMD3<Float>) -> SIMD3<Float>? {
        let direction = SIMD3<Double>(value)
        let length = simd_length(direction)
        guard length.isFinite, length > 0 else { return nil }
        return SIMD3<Float>(direction / length)
    }

    private static func isReadable(_ texture: any MTLTexture, format: MTLPixelFormat) -> Bool {
        PostProcessPass.isReadable(texture, format: format) && !texture.isAliasable()
    }

    private static func isTarget(_ texture: any MTLTexture) -> Bool {
        texture.textureType == .type2D && texture.sampleCount == 1 && texture.usage.contains(.renderTarget)
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked && !texture.isAliasable()
    }

    private struct Uniforms {
        let projection: SIMD4<Float>
        let depth: SIMD4<Float>
        let ray: SIMD4<Float>
        let settings: SIMD4<Float>
        let extent: SIMD4<Float>
        let sampling: SIMD4<UInt32>
    }

    private struct Projection {
        let screen: SIMD4<Float>
        let decode: SIMD4<Float>
        let near: Float
        let far: Float

        init?(camera: ShaderGraphCamera, mode: MeshDepthMode) {
            let matrix = camera.projectionMatrix
            let reversed: Float = mode == .reversed ? 1 : 0
            let logRange: LogarithmicRasterDepth?
            if mode == .logarithmic {
                guard let range = try? LogarithmicRasterDepth(projection: matrix, near: camera.near, far: camera.far)
                else { return nil }
                logRange = range
            } else {
                logRange = nil
            }
            if let perspective = ScreenSpaceProjection(clipFromView: matrix, depthMode: mode) {
                let scale = 1 / perspective.scale
                guard scale.x.isFinite, scale.y.isFinite else { return nil }
                screen = SIMD4(lowHalf: scale, highHalf: perspective.center)
                if let logRange {
                    near = camera.near
                    far = camera.far
                    decode = SIMD4(logRange.range.z, logRange.range.y, 2, 0)
                } else {
                    near = perspective.near
                    far = perspective.farInverseDistance > 0 ? 1 / perspective.farInverseDistance : .infinity
                    decode = SIMD4(perspective.farInverseDistance, perspective.inverseDistanceRange, 0, reversed)
                }
                return
            }

            let (x, y, z, w) = matrix.columns
            guard x.x > 0, y.y > 0, w.w > 0, x.y == 0, x.z == 0, x.w == 0, y.x == 0, y.z == 0, y.w == 0,
                  z.x == 0, z.y == 0, z.w == 0, mode == .reversed ? z.z > 0 : z.z < 0 else { return nil }
            let span = abs(Double(w.w) / Double(z.z))
            let first = (Double(w.z) - Double(reversed) * Double(w.w)) / Double(z.z)
            let projection = SIMD4(w.w / x.x, w.w / y.y, -w.x / w.w, -w.y / w.w)
            guard first >= 0, Float(span).isNormal, Float(first + span).isFinite,
                  (0..<4).allSatisfy({ projection[$0].isFinite }) else { return nil }
            screen = projection
            near = Float(first)
            far = Float(first + span)
            decode = SIMD4(near, Float(span), 1, reversed)
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case commandBuffer, device, attachments, parameters, camera

        public var errorDescription: String? {
            switch self {
            case .commandBuffer: "Use an uncommitted Metal command buffer that retains resources."
            case .device: "Screen-space shadow resources and commands must share one Metal device."
            case .attachments: "Use separate tracked single-sample attachments in the documented formats and view sizes."
            case .parameters: "Use a finite nonzero light direction and screen-space shadow settings within their documented ranges."
            case .camera: "Use a rigid view and matching perspective or axis-aligned orthographic depth projection."
            }
        }
    }
}

/// Scene-level screen textures share a PBR argument buffer so the two effects
/// can coexist even on devices limited to 31 direct texture bindings.
struct PBRScreenInputs {
    struct Binding {
        let arguments: any MTLBuffer
        let textures: [any MTLTexture]

        static func argumentEncoder(on device: any MTLDevice) throws -> any MTLArgumentEncoder {
            let textures = (0..<2).map { index in
                let descriptor = MTLArgumentDescriptor()
                descriptor.index = index
                descriptor.dataType = .texture
                descriptor.textureType = .type2D
                descriptor.access = .readOnly
                return descriptor
            }
            let info = MTLArgumentDescriptor()
            info.index = 2
            info.dataType = .float4
            guard let encoder = device.makeArgumentEncoder(arguments: textures + [info]) else {
                throw MeshRenderer.Failure.resource("screen lighting argument encoder")
            }
            return encoder
        }

        init(
            device: any MTLDevice, encoder: any MTLArgumentEncoder, ambient: (any MTLTexture)? = nil,
            directional: ScreenSpaceShadows.PreparedDirectionalVisibility? = nil
        ) throws {
            guard let buffer = device.makeBuffer(length: encoder.encodedLength, options: .storageModeShared) else {
                throw MeshRenderer.Failure.resource("screen lighting argument buffer")
            }
            buffer.label = "Frostfield PBR screen lighting"
            let shadow = directional?.value.texture
            encoder.setArgumentBuffer(buffer, offset: 0)
            encoder.setTexture(ambient, index: 0)
            encoder.setTexture(shadow, index: 1)
            let extent = directional?.extent ?? .zero
            encoder.constantData(at: 2).storeBytes(
                of: SIMD4<Float>(ambient == nil ? 0 : 1, shadow == nil ? 0 : 1, extent.x, extent.y),
                as: SIMD4<Float>.self
            )
            arguments = buffer
            textures = [ambient, shadow].compactMap { $0 }
        }

        func bind(into encoder: any MTLRenderCommandEncoder) {
            for texture in textures { encoder.useResource(texture, usage: .read, stages: .fragment) }
            encoder.setFragmentBuffer(arguments, offset: 0, index: 25)
        }
    }

    private let ambient: Binding?
    private let directional: Binding?
    private let combined: Binding?
    private let selection: ScreenSpaceShadows.DirectionalVisibility?

    init?(
        device: any MTLDevice, ambient texture: (any MTLTexture)?,
        directional shadow: ScreenSpaceShadows.PreparedDirectionalVisibility?
    ) throws {
        guard texture != nil || shadow != nil else { return nil }
        let encoder = try Binding.argumentEncoder(on: device)
        ambient = try texture.map { try Binding(device: device, encoder: encoder, ambient: $0) }
        directional = try shadow.map { try Binding(device: device, encoder: encoder, directional: $0) }
        combined = try texture.flatMap { ambient in
            try shadow.map { try Binding(device: device, encoder: encoder, ambient: ambient, directional: $0) }
        }
        selection = shadow?.value
    }

    func binding(for light: SIMD4<Float>) -> Binding? {
        selection?.applies(to: light) == true ? combined ?? directional : ambient
    }
}
