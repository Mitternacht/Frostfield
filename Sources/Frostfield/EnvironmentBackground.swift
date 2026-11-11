import Foundation
import Metal
import simd

/// Draws a raster background before a scene's color pass.
public final class EnvironmentBackground {
    public enum Source {
        /// Opaque scene-linear RGB. Intensity and orientation affect textures only.
        case color(SIMD3<Float>)
        /// Stretches a 2D image over the viewport with top-left UVs. Color textures
        /// from TextureLoader use straight alpha. Set premultipliedAlpha for scene renders.
        case flat(
            any MTLTexture, transform: PhysicallyBasedMaterial.TextureTransform = .identity,
            sampling: PhysicallyBasedMaterial.TextureSampling = .init(addressU: .clampToEdge, addressV: .clampToEdge),
            premultipliedAlpha: Bool = false
        )
        /// An opaque directional image in Metal cube face order.
        case cube(any MTLTexture)
        /// An opaque 2:1 panorama. The top faces +Y and the center column faces +X.
        case equirectangular(any MTLTexture)
        /// Uses the sharp sky at zero blur and the prepared GGX cube for nonzero blur.
        case filtered(EnvironmentLighting)
    }

    public struct Options: Sendable {
        /// Orientation of a directional image in world space. Supply a unit quaternion.
        public var rotation: simd_quatf
        /// Multiplies texture RGB in linear light. Finite and between zero and 65504.
        public var intensity: Float
        /// From zero for sharp directional images to one for the roughest filtered
        /// level or last source mip. Has no effect on flat images or solid colors.
        public var blurriness: Float

        public init(
            rotation: simd_quatf = simd_quatf(real: 1, imag: .zero),
            intensity: Float = 1, blurriness: Float = 0
        ) {
            self.rotation = rotation
            self.intensity = intensity
            self.blurriness = blurriness
        }
    }

    private let device: any MTLDevice
    private let source: Source
    private let pipeline: any MTLRenderPipelineState
    private let sampler: (any MTLSamplerState)?
    private let destinationFormat: MTLPixelFormat
    /// Sample count of the color attachment this background prepares.
    public let sampleCount: Int

    /// Prepares the original HDR background entry point.
    public convenience init(
        lighting: EnvironmentLighting, destinationPixelFormat: MTLPixelFormat = .rgba16Float, sampleCount: Int = 1
    ) throws {
        guard destinationPixelFormat == .rgba16Float else { throw Failure.destinationFormat }
        try self.init(
            device: lighting.device, source: .filtered(lighting), destinationPixelFormat: destinationPixelFormat,
            sampleCount: sampleCount
        )
    }

    /// Prepares a background for linear HDR or linear/sRGB RGBA8 or BGRA8.
    /// Textures must be tracked and shader-readable. Use RGBA8/BGRA8 (linear
    /// or sRGB), RGBA16Float, or RGBA32Float on devices with float filtering.
    /// The source's Metal format controls sRGB decoding. The pass writes linear
    /// premultiplied color and leaves display exposure and tone mapping to the caller.
    /// A multisample pipeline stores the per-sample color for the following scene pass.
    public init(
        device: any MTLDevice, source: Source, destinationPixelFormat: MTLPixelFormat = .rgba16Float,
        sampleCount: Int = 1
    ) throws {
        guard RendererCapabilities.supportsRasterSampleCount(sampleCount, on: device) else { throw Failure.sampleCount }
        switch destinationPixelFormat {
        case .rgba16Float, .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb: break
        default: throw Failure.destinationFormat
        }

        let fragmentName: String
        var sampler: (any MTLSamplerState)?
        switch source {
        case .color(let rgb):
            guard (0..<3).allSatisfy({ (0...65_504).contains(rgb[$0]) }) else { throw Failure.environment }
            fragmentName = "color"
        case .flat(let texture, _, let sampling, _):
            try Self.validate(texture, device: device, type: .type2D)
            guard let state = device.makeSamplerState(descriptor: sampling.descriptor) else { throw Failure.sampler }
            sampler = state
            fragmentName = "flat"
        case .cube(let texture):
            try Self.validate(texture, device: device, type: .typeCube)
            fragmentName = "cube"
        case .equirectangular(let texture):
            try Self.validate(texture, device: device, type: .type2D)
            guard texture.width == 2 * texture.height else { throw Failure.source }
            fragmentName = "equirectangular"
        case .filtered(let lighting):
            guard lighting.device === device else { throw Failure.wrongDevice }
            fragmentName = "cube"
        }

        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        guard let vertex = library.makeFunction(name: "frostfield_environment_background_vertex"),
              let fragment = library.makeFunction(name: "frostfield_environment_background_\(fragmentName)") else {
            throw Failure.shader
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield \(fragmentName) background"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = destinationPixelFormat
        descriptor.rasterSampleCount = sampleCount
        do {
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        self.device = device
        self.source = source
        self.sampler = sampler
        destinationFormat = destinationPixelFormat
        self.sampleCount = sampleCount
    }

    /// Stores the sharp background without depth. Match yaw to the material
    /// environment's world-space rotation in radians to align both images.
    public func encode(
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform = .init(),
        yaw: Float = 0,
        intensity: Float = 1,
        viewport: MTLViewport? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard yaw.isFinite else { throw Failure.environment }
        try encode(
            destination: destination, camera: camera, cameraTransform: cameraTransform,
            options: .init(rotation: simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)), intensity: intensity),
            viewport: viewport, commandBuffer: commandBuffer
        )
    }

    /// Stores the background without depth. Use a subsequent scene pass with
    /// color load action `.load`. A whole-pixel viewport sets the background's
    /// aspect ratio and preserves pixels outside it. Directional textures
    /// ignore camera translation. Flat images and solid colors ignore the camera.
    /// For raw directional images with nonzero blur, supply a populated mip chain.
    public func encode(
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform = .init(),
        options: Options,
        viewport: MTLViewport? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(destination: destination, options: options, viewport: viewport, commandBuffer: commandBuffer) { aspectRatio in
            try Self.rayMatrix(camera: camera, transform: cameraTransform, rotation: options.rotation, aspectRatio: aspectRatio)
        }
    }

    /// Draws behind a raster view that supplies its own projection, including
    /// `GLTFScene.Camera.View.graphCamera`. Use the same view and viewport for
    /// the scene pass. Off-center perspective uses the supplied frustum, and
    /// orthographic uses the camera's parallel forward rays. Camera translation
    /// and the choice of forward or reversed depth do not affect the sky.
    public func encode(
        destination: any MTLTexture,
        view: ShaderGraphCamera,
        options: Options = .init(),
        viewport: MTLViewport? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(destination: destination, options: options, viewport: viewport, commandBuffer: commandBuffer) { _ in
            try Self.rayMatrix(view: view, rotation: options.rotation)
        }
    }

    private func encode(
        destination: any MTLTexture,
        options: Options,
        viewport: MTLViewport?,
        commandBuffer: any MTLCommandBuffer,
        rayMatrix: (Float) throws -> simd_float3x3
    ) throws {
        guard commandBuffer.device === device, destination.device === device else { throw Failure.wrongDevice }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        let destinationType: MTLTextureType = sampleCount == 1 ? .type2D : .type2DMultisample
        guard destination.textureType == destinationType, destination.pixelFormat == destinationFormat,
              destination.sampleCount == sampleCount, destination.usage.contains(.renderTarget),
              destination.storageMode != .memoryless, destination.hazardTrackingMode != .untracked else {
            throw Failure.destination
        }
        guard (0...65_504).contains(options.intensity), (0...1).contains(options.blurriness) else {
            throw Failure.environment
        }

        let aspectRatio: Float
        if let viewport {
            guard viewport.originX >= 0, viewport.originY >= 0,
                  viewport.width > 0, viewport.height > 0,
                  viewport.originX.rounded() == viewport.originX, viewport.originY.rounded() == viewport.originY,
                  viewport.width.rounded() == viewport.width, viewport.height.rounded() == viewport.height,
                  viewport.originX + viewport.width <= Double(destination.width),
                  viewport.originY + viewport.height <= Double(destination.height),
                  viewport.znear >= 0, viewport.zfar <= 1, viewport.znear < viewport.zfar else {
                throw Failure.viewport
            }
            aspectRatio = Float(viewport.width / viewport.height)
        } else {
            aspectRatio = Float(destination.width) / Float(destination.height)
        }

        let texture = sourceTexture(options)
        if let texture, TextureTransfer.sharesStorage(texture, destination) { throw Failure.feedback }

        var rayFromNDC = matrix_identity_float3x3
        var values = SIMD4<Float>.zero
        var rows = FlatTransform()
        switch source {
        case .color(let rgb):
            values = SIMD4(rgb, 1)
        case .flat(_, let transform, _, let premultiplied):
            let (first, second) = transform.rows
            rows = FlatTransform(first: first, second: second)
            values = SIMD4(options.intensity, premultiplied ? 1 : 0, 0, 0)
        case .cube, .equirectangular, .filtered:
            guard let texture else { throw Failure.source }
            let maxLOD = Float(texture.mipmapLevelCount - 1)
            guard options.blurriness == 0 || maxLOD > 0 else { throw Failure.mipmaps }
            guard abs(simd_length_squared(options.rotation.vector) - 1) <= 0.0001 else { throw Failure.environment }
            rayFromNDC = try rayMatrix(aspectRatio)
            values = SIMD4(options.intensity, options.blurriness * maxLOD, 0, 0)
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = viewport == nil ? .dontCare : .load
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield environment background"
        encoder.setRenderPipelineState(pipeline)
        if let viewport {
            encoder.setViewport(viewport)
            encoder.setScissorRect(MTLScissorRect(
                x: Int(viewport.originX), y: Int(viewport.originY),
                width: Int(viewport.width), height: Int(viewport.height)
            ))
        }
        if let texture { encoder.setFragmentTexture(texture, index: 0) }
        encoder.setFragmentBytes(&values, length: MemoryLayout<SIMD4<Float>>.size, index: 1)
        switch source {
        case .color: break
        case .flat:
            encoder.setFragmentSamplerState(sampler, index: 0)
            encoder.setFragmentBytes(&rows, length: MemoryLayout<FlatTransform>.size, index: 0)
        case .cube, .equirectangular, .filtered:
            encoder.setFragmentBytes(&rayFromNDC, length: MemoryLayout<simd_float3x3>.size, index: 0)
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private struct FlatTransform {
        var first = SIMD4<Float>.zero
        var second = SIMD4<Float>.zero
    }

    private static func rayMatrix(
        camera: PerspectiveCamera, transform: SceneTransform, rotation: simd_quatf, aspectRatio: Float
    ) throws -> simd_float3x3 {
        guard abs(simd_length_squared(transform.rotation.vector) - 1) <= 0.0001 else { throw Failure.cameraTransform }
        let projection = try camera.projection(aspectRatio: aspectRatio)
        let scale = transform.scale
        let ray = SIMD3(
            Double(scale.x) / Double(projection.columns.0.x),
            Double(scale.y) / Double(projection.columns.1.y),
            -Double(scale.z)
        )
        let extent = max(abs(ray.x), max(abs(ray.y), abs(ray.z)))
        guard extent.isFinite, extent > 0 else { throw Failure.cameraTransform }
        let rayScale = SIMD3<Float>(ray / extent)
        guard rayScale.x.isNormal, rayScale.y.isNormal, rayScale.z.isNormal else { throw Failure.cameraTransform }

        let basis = simd_float3x3(rotation.inverse) * simd_float3x3(transform.rotation)
        return simd_float3x3(columns: (
            basis.columns.0 * rayScale.x, basis.columns.1 * rayScale.y, basis.columns.2 * rayScale.z
        ))
    }

    private static func rayMatrix(view: ShaderGraphCamera, rotation: simd_quatf) throws -> simd_float3x3 {
        let pose = view.viewMatrix
        // This validator transposes the rigid input basis. Here the input is
        // view-from-world, so the result is the world-from-view rotation.
        guard pose.columns.0.w == 0, pose.columns.1.w == 0, pose.columns.2.w == 0, abs(pose.columns.3.w - 1) <= 1e-4,
              let worldFromView = ScreenSpaceProjection.viewFromWorld(pose) else { throw Failure.cameraTransform }
        let environmentFromView = simd_float3x3(rotation.inverse) * worldFromView
        let p = view.projectionMatrix
        guard p.columns.0.x > 0, p.columns.1.y > 0,
              p.columns.0.y == 0, p.columns.1.x == 0, p.columns.0.w == 0, p.columns.1.w == 0 else {
            throw Failure.projection
        }
        if p.columns.2.w == 0, p.columns.3.w > 0, p.columns.2.x == 0, p.columns.2.y == 0 {
            return simd_float3x3(columns: (.zero, .zero, -environmentFromView.columns.2))
        }
        guard p.columns.2.w < 0, p.columns.3.w == 0, p.columns.3.x == 0, p.columns.3.y == 0 else {
            throw Failure.projection
        }

        // At view Z = -1, clip W is constant. Inverting NDC on that plane
        // reconstructs the ray and the off-center term independently of depth.
        let w = -Double(p.columns.2.w)
        let x = Double(p.columns.0.x)
        let y = Double(p.columns.1.y)
        let scale = SIMD2(w / x, w / y)
        let center = SIMD3(Double(p.columns.2.x) / x, Double(p.columns.2.y) / y, -1)
        let extent = max(scale.x, scale.y, simd_reduce_max(abs(center)))
        let unitScale = SIMD2<Float>(scale / extent)
        let unitCenter = SIMD3<Float>(center / extent)
        guard unitScale.x.isNormal, unitScale.y.isNormal, unitCenter.z.isNormal else { throw Failure.projection }
        let viewFromNDC = simd_float3x3(columns: (SIMD3(unitScale.x, 0, 0), SIMD3(0, unitScale.y, 0), unitCenter))
        return environmentFromView * viewFromNDC
    }

    static func validate(_ texture: any MTLTexture, device: any MTLDevice, type: MTLTextureType) throws {
        guard texture.device === device else { throw Failure.wrongDevice }
        guard texture.textureType == type, texture.sampleCount == 1, !texture.isFramebufferOnly,
              texture.storageMode != .memoryless, texture.hazardTrackingMode != .untracked,
              texture.usage.isEmpty || texture.usage.contains(.shaderRead) else { throw Failure.source }
        switch texture.pixelFormat {
        case .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb, .rgba16Float: break
        case .rgba32Float where device.supports32BitFloatFiltering: break
        default: throw Failure.source
        }
    }

    private func sourceTexture(_ options: Options) -> (any MTLTexture)? {
        switch source {
        case .color: nil
        case .flat(let image, _, _, _), .cube(let image), .equirectangular(let image): image
        case .filtered(let lighting): options.blurriness == 0 ? lighting.skyCube : lighting.specularCube
        }
    }

    func samplesStorage(of targets: [any MTLTexture], options: Options) -> Bool {
        guard let source = sourceTexture(options) else { return false }
        return targets.contains { TextureTransfer.sharesStorage(source, $0) }
    }

    public enum Failure: LocalizedError, Equatable {
        case destinationFormat
        case sampleCount
        case wrongDevice
        case commandBuffer
        case destination
        case source
        case feedback
        case mipmaps
        case sampler
        case environment
        case viewport
        case cameraTransform
        case projection
        case pipeline(String)
        case shader
        case encoder

        public var errorDescription: String? {
            switch self {
            case .destinationFormat: "Use the destination pixel format accepted by the background initializer."
            case .sampleCount: "Use a supported raster sample count for the background pipeline."
            case .wrongDevice: "Use the background's Metal device."
            case .commandBuffer: "Use an uncommitted command buffer with retained references."
            case .destination: "Use a tracked, storable render target with the prepared pixel format and sample count."
            case .source: "Use a tracked, filterable color texture with the requested projection and shader-read access."
            case .feedback: "Use separate textures for the background source and render target."
            case .mipmaps: "Directional blur needs a populated mip chain or a filtered environment."
            case .sampler: "Metal could not create the flat background's sampler."
            case .environment: "Use a unit rotation, intensity and color from 0 to 65504, and blur from 0 to 1."
            case .viewport: "Use a whole-pixel background viewport within the target with normalized increasing depths."
            case .cameraTransform: "Use a rigid explicit view, or a unit camera rotation and finite, nonzero camera scales."
            case .projection: "Use a perspective or orthographic Metal projection without skew."
            case .pipeline(let detail): "Could not prepare the environment background pipeline: \(detail)"
            case .shader: "The environment background shaders are missing."
            case .encoder: "Metal could not create the environment background render encoder."
            }
        }
    }
}
