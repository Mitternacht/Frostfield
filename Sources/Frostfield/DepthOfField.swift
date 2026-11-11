import Foundation
import Metal

/// A depth-aware, thin-lens approximation for premultiplied linear HDR scene color.
public final class DepthOfField {
    public struct Settings: Sendable {
        /// Distance along the camera's view axis to the plane in focus, in world units.
        public var focusDistance: Float
        /// Lens focal length divided by aperture diameter. Smaller values blur more.
        public var fNumber: Float
        /// Sensor height in world units. 0.024 means 24 mm in a meter-scale scene.
        public var sensorHeight: Float
        /// Maximum circle-of-confusion radius in pixels, from 0 to 24.
        /// Radii at or below half a pixel preserve the original color.
        public var maximumRadius: Float

        public init(
            focusDistance: Float, fNumber: Float = 2.8, sensorHeight: Float = 0.024, maximumRadius: Float = 12
        ) {
            self.focusDistance = focusDistance
            self.fNumber = fNumber
            self.sensorHeight = sensorHeight
            self.maximumRadius = maximumRadius
        }
    }

    private let device: any MTLDevice
    private let pipeline: any MTLRenderPipelineState

    public init(device: any MTLDevice) throws {
        self.device = device
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_dof_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_dof_fragment") else {
                throw Failure.missingShader
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield depth of field"
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = .rgba16Float
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Encodes on the caller's retained command buffer after producing scene color
    /// and storing perspective depth. Use single-sample, equally sized textures and
    /// matching near/far planes and depth mode. Use standalone premultiplied
    /// linear RGBA16Float color and Depth32Float depth, both shader-readable and
    /// hazard-tracked. Cleared depth is treated as the far plane. Use a separate,
    /// non-aliased RGBA16Float target and synchronize any untracked destination.
    /// The fixed gather rejects unrelated depth layers and preserves focus, so
    /// defocused foreground objects do not spread across their silhouettes.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        depthMode: MeshDepthMode = .forward,
        settings: Settings,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, depth.device === device,
              destination.device === device else { throw Failure.wrongDevice }
        guard source !== destination, Self.isReadable(source, format: .rgba16Float),
              Self.isReadable(depth, format: .depth32Float), destination.pixelFormat == .rgba16Float,
              destination.textureType == .type2D, destination.sampleCount == 1,
              source.width > 0, source.height > 0,
              source.width == depth.width, source.height == depth.height,
              source.width == destination.width, source.height == destination.height,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              source.parent == nil, destination.parent == nil,
              source.buffer == nil, source.iosurface == nil, source.heap == nil else {
            throw Failure.incompatibleTextures
        }
        var lens = try Self.lensParameters(camera: camera, settings: settings, height: source.height)
        var reversed: UInt32 = switch depthMode {
        case .forward: 0
        case .reversed: 1
        case .logarithmic: throw MeshRenderer.Failure.logarithmicDepthConsumer
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        encoder.label = "Frostfield depth of field"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(depth, index: 1)
        encoder.setFragmentBytes(&lens, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        encoder.setFragmentBytes(&reversed, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private static func isReadable(_ texture: any MTLTexture, format: MTLPixelFormat) -> Bool {
        texture.pixelFormat == format && texture.textureType == .type2D && texture.sampleCount == 1
            && texture.usage.contains(.shaderRead) && !texture.isFramebufferOnly
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
    }

    private static func lensParameters(
        camera: PerspectiveCamera, settings: Settings, height: Int
    ) throws -> SIMD4<Float> {
        guard (camera.near...camera.far).contains(settings.focusDistance),
              settings.fNumber.isFinite, settings.fNumber > 0,
              settings.sensorHeight.isFinite, settings.sensorHeight > 0,
              (0...24).contains(settings.maximumRadius) else { throw Failure.invalidLens }
        let focus = Double(settings.focusDistance)
        let sensor = Double(settings.sensorHeight)
        let focalLength = sensor / (2 * tan(Double(camera.verticalFieldOfView) / 2))
        guard focalLength > 0, focalLength < focus else { throw Failure.invalidLens }

        // Rearrange the thin-lens circle-of-confusion diameter into pixel radius
        // times abs(1 - focus / objectDistance). Depth is affine in 1 / distance.
        // https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
        let radiusScale = focalLength * focalLength * Double(height)
            / (2 * Double(settings.fNumber) * sensor * (focus - focalLength))
        let nearRatio = focus / Double(camera.near)
        let farRatio = focus / Double(camera.far)
        guard Float(nearRatio).isFinite, Float(radiusScale * nearRatio).isFinite else { throw Failure.invalidLens }
        return SIMD4(Float(farRatio), Float(nearRatio), Float(radiusScale), settings.maximumRadius)
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case invalidLens
        case commandBuffer
        case wrongDevice
        case incompatibleTextures
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield’s depth-of-field shaders are missing."
            case .pipeline(let detail): "Could not prepare Frostfield’s depth of field. \(detail)"
            case .invalidLens:
                "Focus must be within the clip planes and beyond the focal length. Use positive lens values and a blur radius from 0 to 24."
            case .commandBuffer: "Depth of field needs an uncommitted command buffer that retains resources."
            case .wrongDevice: "Depth-of-field textures and command buffer must use its Metal device."
            case .incompatibleTextures:
                "Use equal-sized, shader-readable RGBA16Float color and Depth32Float depth with a separate RGBA16Float render target."
            case .encoder: "Metal could not create the depth-of-field encoder."
            }
        }
    }
}
