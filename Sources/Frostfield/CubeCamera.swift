import Foundation
import Metal
import simd

/// Six 90° perspective views that capture a scene into a Metal HDR cube texture.
public final class CubeCamera {
    public let target: any MTLTexture
    public let near: Float
    public let far: Float
    /// All six views use this mask independently of the scene's camera.
    public var layers: SceneLayers

    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let color: any MTLTexture
    private let depth: any MTLTexture
    private let transmissionBackdrop: any MTLTexture
    private let copyFace: any MTLComputePipelineState

    /// Allocates a tracked, private, shader-readable and shader-writable RGBA16Float cube.
    /// Pass this cube to the initializer, or supply a compatible texture of your own.
    public static func makeTarget(
        device: any MTLDevice, size: Int, mipmapped: Bool = true
    ) throws -> any MTLTexture {
        guard (1...4096).contains(size) else { throw Failure.target }
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba16Float, size: size, mipmapped: mipmapped
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
        texture.label = "Frostfield HDR capture cube"
        return texture
    }

    public init(
        queue: any MTLCommandQueue, target: any MTLTexture, near: Float, far: Float,
        layers: SceneLayers = .default
    ) throws {
        guard target.device === queue.device else { throw Failure.wrongDevice }
        guard target.textureType == .typeCube, target.pixelFormat == .rgba16Float,
              (1...4096).contains(target.width), target.width == target.height, target.sampleCount == 1,
              !target.isFramebufferOnly, target.storageMode != .memoryless,
              target.hazardTrackingMode != .untracked, target.parent == nil,
              target.usage.contains([.shaderRead, .shaderWrite]) else {
            throw Failure.target
        }
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: near, far: far)
        self.queue = queue
        self.target = target
        self.near = near
        self.far = far
        self.layers = layers

        func makeScratch(
            _ format: MTLPixelFormat, usage: MTLTextureUsage, mipmapped: Bool, label: String
        ) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: target.width, height: target.height, mipmapped: mipmapped
            )
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = usage
            guard let texture = queue.device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
            texture.label = "Frostfield cube capture \(label)"
            return texture
        }
        color = try makeScratch(.rgba16Float, usage: [.renderTarget, .shaderRead], mipmapped: false, label: "color")
        depth = try makeScratch(.depth32Float, usage: .renderTarget, mipmapped: false, label: "depth")
        transmissionBackdrop = try makeScratch(.rgba16Float, usage: .shaderRead, mipmapped: true, label: "backdrop")

        do {
            let library = try queue.device.makeDefaultLibrary(bundle: .module)
            guard let function = library.makeFunction(name: "frostfield_cube_capture_face") else {
                throw Failure.pipeline("The cube face copy shader is missing.")
            }
            copyFace = try queue.device.makeComputePipelineState(function: function)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Captures the graph from a world-space point into mip zero of all six faces,
    /// then regenerates the target's mip chain. Submit the command buffer on the
    /// preparation queue. The graph uses this camera's layers and ignores its main
    /// camera and temporal jitter. Excluding a node also excludes its descendants.
    ///
    /// Prepare drawable meshes for single-sample RGBA16Float and forward depth.
    /// Use the graph's normal shadow resources for shadow-receiving materials.
    /// Materials and backgrounds that sample this target must stay out of the
    /// capture. A supplied sky uses the graph's environment rotation and intensity.
    /// The clear color is premultiplied linear RGBA. Discard the command buffer if
    /// encoding throws.
    public func encode(
        _ graph: SceneGraph, at position: SIMD3<Float>, excluding: Set<UUID> = [],
        background: EnvironmentBackground? = nil,
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard position.x.isFinite, position.y.isFinite, position.z.isFinite else { throw Failure.position }
        guard (0...65504).contains(clearColor.red), (0...65504).contains(clearColor.green),
              (0...65504).contains(clearColor.blue), (0...1).contains(clearColor.alpha) else {
            throw Failure.clearColor
        }

        var capture = graph
        capture.camera = camera
        capture.cameraLayers = layers
        capture.projectionJitter = .zero
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = background == nil ? .clear : .load
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clearColor
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1

        for (face, rotation) in Self.faceRotations.enumerated() {
            let transform = SceneTransform(translation: position, rotation: rotation)
            capture.cameraTransform = transform
            try background?.encode(
                destination: color, camera: camera, cameraTransform: transform,
                yaw: graph.environmentRotation, intensity: graph.environmentIntensity,
                commandBuffer: commandBuffer
            )
            _ = try MeshRenderer.encode(
                capture, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap,
                punctualShadowMaps: punctualShadowMaps, transmissionBackdrop: transmissionBackdrop, excluding: excluding
            )
            try encodeFace(UInt32(face), commandBuffer: commandBuffer)
        }
        if target.mipmapLevelCount > 1 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
            blit.label = "Frostfield capture cube mipmaps"
            blit.generateMipmaps(for: target)
            blit.endEncoding()
        }
    }

    /// The up vectors match the Metal cube faces. The perspective render's right
    /// vector is opposite the sampler's texel X axis, so the copy mirrors X.
    private static let faceRotations: [simd_quatf] = {
        let y = SIMD3<Float>(0, 1, 0)
        let z = SIMD3<Float>(0, 0, 1)
        let directions: [SIMD3<Float>] = [
            SIMD3(1, 0, 0), SIMD3(-1, 0, 0), y, -y, z, -z,
        ]
        return zip(directions, [y, y, -z, z, y, y]).map { forward, up in
            simd_quatf(simd_float3x3(columns: (simd_cross(forward, up), up, -forward)))
        }
    }()

    private func encodeFace(_ face: UInt32, commandBuffer: any MTLCommandBuffer) throws {
        guard let copy = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        copy.label = "Frostfield cube capture face \(face)"
        copy.setComputePipelineState(copyFace)
        copy.setTexture(color, index: 0)
        copy.setTexture(target, index: 1)
        var face = face
        copy.setBytes(&face, length: MemoryLayout<UInt32>.size, index: 0)
        let threads = min(copyFace.maxTotalThreadsPerThreadgroup, 64)
        let width = min(threads, 8)
        let height = threads / width
        let size = target.width
        copy.dispatchThreadgroups(
            MTLSize(width: (size + width - 1) / width, height: (size + height - 1) / height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        copy.endEncoding()
    }

    public enum Failure: LocalizedError, Equatable {
        case wrongDevice
        case target
        case allocation
        case pipeline(String)
        case commandBuffer
        case position
        case clearColor
        case encoder

        public var errorDescription: String? {
            switch self {
            case .wrongDevice: "Use a capture target from the queue's Metal device."
            case .target: "Use a tracked, shader-readable and shader-writable RGBA16Float cube sized 1...4096."
            case .allocation: "Metal could not allocate the cube capture textures."
            case .pipeline(let detail): "Could not prepare the cube capture pipeline: \(detail)"
            case .commandBuffer: "Use an uncommitted command buffer with retained references from the capture queue."
            case .position: "Use a finite world-space capture position."
            case .clearColor: "Use finite clear components with RGB in 0...65504 and alpha in 0...1."
            case .encoder: "Metal could not create a cube capture encoder."
            }
        }
    }
}
