import Foundation
import Metal
import simd

/// Raymarches a scalar 3D texture into a color target.
public final class ScalarVolumeRenderer {
    public enum Mode: Equatable, Sendable {
        case maximumIntensity
        /// The first crossing of a scalar threshold, shaded by a camera headlight.
        case isosurface(threshold: Float)
    }

    public enum Sampling: UInt32, Sendable {
        case nearest, linear
    }

    public struct Settings: Equatable, Sendable {
        public var mode: Mode
        /// The scalar domain of the color lookup table. MIP rays below this window are transparent.
        public var window: ClosedRange<Float>
        public var opacity: Float
        public var sampling: Sampling
        /// Steps per voxel along the dominant index-space ray axis, from 0.25 through 4.
        public var samplesPerVoxel: Float
        /// Upper bound on coarse ray steps, from 1 through 4096.
        public var maximumSteps: Int

        public init(
            mode: Mode = .maximumIntensity, window: ClosedRange<Float> = 0...1,
            opacity: Float = 1, sampling: Sampling = .linear, samplesPerVoxel: Float = 1, maximumSteps: Int = 512
        ) {
            self.mode = mode
            self.window = window
            self.opacity = opacity
            self.sampling = sampling
            self.samplesPerVoxel = samplesPerVoxel
            self.maximumSteps = maximumSteps
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case texture
        case pixelFormat
        case transform
        case settings
        case destination
        case depth
        case clearColor
        case commandBuffer
        case missingShader
        case pipeline(String)
        case encoder

        public var errorDescription: String? {
            switch self {
            case .texture: "Use a single-sample, hazard-tracked R16Float or R32Float 3D texture with shader-read usage."
            case .pixelFormat: "Use RGBA16Float or an 8-bit sRGB volume destination."
            case .transform: "Volume rendering requires finite, invertible transforms representable in Float32."
            case .settings: "Use a finite nonempty scalar window, opacity in 0...1, and a valid ray step budget."
            case .destination: "Use a matching single-sample, hazard-tracked 2D color render target."
            case .depth: "Use a separate, matching, single-sample shader-readable Depth32Float texture."
            case .clearColor: "Use finite nonnegative clear color and alpha in 0...1."
            case .commandBuffer: "Use an uncommitted command buffer on the volume device that retains its references."
            case .missingShader: "The scalar volume shader is missing."
            case .pipeline(let message): "The scalar volume pipeline failed. \(message)"
            case .encoder: "Could not create the scalar volume render encoder."
            }
        }
    }

    public let texture: any MTLTexture
    public let voxelToSpace: simd_double4x4
    public let pixelFormat: MTLPixelFormat
    private let colorTexture: any MTLTexture
    private let colorPipeline: any MTLRenderPipelineState
    private let depthPipeline: any MTLRenderPipelineState

    private struct Uniforms {
        var voxelFromClip: simd_float4x4
        var worldFromVoxel: simd_float4x4
        var voxelFromWorld: simd_float4x4
        // Window lower and upper bounds, isovalue, and opacity.
        var scalar: SIMD4<Float>
        // Near and ray-end clip depths, steps per voxel, and infinite-far flag.
        var ray: SIMD4<Float>
        // Step limit, mode, and interpolation mode.
        var flags: SIMD4<UInt32>
    }

    /// Converts NRRD scalars to Float32 and keeps the original physical voxel transform.
    public convenience init(
        device: any MTLDevice, volume: NRRDLoader.Volume, colorMap: ScalarColorLookupTable? = nil,
        pixelFormat: MTLPixelFormat = .rgba16Float
    ) throws {
        try self.init(texture: volume.makeTexture(device: device, storage: .float32),
                      voxelToSpace: volume.voxelToSpace, colorMap: colorMap, pixelFormat: pixelFormat)
    }

    /// Uses a scalar float texture with the supplied affine mapping voxel centers into physical space.
    public init(
        texture: any MTLTexture, voxelToSpace: simd_double4x4 = matrix_identity_double4x4,
        colorMap: ScalarColorLookupTable? = nil, pixelFormat: MTLPixelFormat = .rgba16Float
    ) throws {
        guard texture.textureType == .type3D, texture.pixelFormat == .r16Float || texture.pixelFormat == .r32Float,
              Self.readable(texture) else { throw Failure.texture }
        guard PostProcessPass.supportedPixelFormats.contains(pixelFormat) else { throw Failure.pixelFormat }
        try Self.validateAffine(voxelToSpace)
        self.texture = texture
        self.voxelToSpace = voxelToSpace
        self.pixelFormat = pixelFormat
        let device = texture.device
        colorTexture = try (colorMap ?? ScalarColorLookupTable(palette: .grayscale)).makeTexture(device: device)
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_scalar_volume_vertex") else {
                throw Failure.missingShader
            }
            func pipeline(usesDepth: Bool) throws -> any MTLRenderPipelineState {
                let constants = MTLFunctionConstantValues()
                var depth = usesDepth
                constants.setConstantValue(&depth, type: .bool, index: 0)
                let fragment = try library.makeFunction(name: "frostfield_scalar_volume_fragment", constantValues: constants)
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.label = "Frostfield scalar volume"
                descriptor.vertexFunction = vertex
                descriptor.fragmentFunction = fragment
                descriptor.colorAttachments[0].pixelFormat = pixelFormat
                MeshRenderState.transparent.configure(descriptor)
                return try device.makeRenderPipelineState(descriptor: descriptor)
            }
            colorPipeline = try pipeline(usesDepth: false)
            depthPipeline = try pipeline(usesDepth: true)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Composites premultiplied source-over color into the destination. Supply a
    /// premultiplied clear color for an empty frame, or leave it nil to load the target.
    /// Optional opaque depth clips the ray using the same camera and depth mode.
    /// Encode after the depth producer and keep textures on the command buffer's device.
    public func encode(
        camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward, placement: SceneTransform = .init(),
        settings: Settings = .init(), destination: any MTLTexture, opaqueDepth: (any MTLTexture)? = nil,
        clearColor: MTLClearColor? = nil, commandBuffer: any MTLCommandBuffer
    ) throws {
        let device = texture.device
        guard commandBuffer.device === device, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard destination.device === device, destination.pixelFormat == pixelFormat,
              destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              destination.hazardTrackingMode != .untracked else { throw Failure.destination }
        if let depth = opaqueDepth {
            guard depth.device === device, depth.textureType == .type2D, depth.pixelFormat == .depth32Float,
                  depth.width == destination.width, depth.height == destination.height, Self.readable(depth) else {
                throw Failure.depth
            }
        }
        if let clearColor {
            let colors = [clearColor.red, clearColor.green, clearColor.blue]
            let range = 0...Double(Float16.greatestFiniteMagnitude)
            guard colors.allSatisfy(range.contains), (0...1).contains(clearColor.alpha) else {
                throw Failure.clearColor
            }
        }

        try depthMode.requireProjectionDepth()
        var uniforms = try uniforms(camera: camera, depthMode: depthMode, placement: placement, settings: settings)
        let pass = MTLRenderPassDescriptor()
        let color = pass.colorAttachments[0]!
        color.texture = destination
        color.loadAction = clearColor == nil ? .load : .clear
        color.storeAction = .store
        if let clearColor { color.clearColor = clearColor }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield scalar volume"
        encoder.setRenderPipelineState(opaqueDepth == nil ? colorPipeline : depthPipeline)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.setFragmentTexture(colorTexture, index: 1)
        if let opaqueDepth { encoder.setFragmentTexture(opaqueDepth, index: 2) }
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func uniforms(
        camera: ShaderGraphCamera, depthMode: MeshDepthMode, placement: SceneTransform, settings: Settings
    ) throws -> Uniforms {
        let low = settings.window.lowerBound, high = settings.window.upperBound
        guard low.isFinite, high.isFinite, (high - low).isFinite, high > low,
              (0...1).contains(settings.opacity), (0.25...4).contains(settings.samplesPerVoxel),
              (1...4096).contains(settings.maximumSteps) else { throw Failure.settings }
        let mode: UInt32
        let threshold: Float
        switch settings.mode {
        case .maximumIntensity: mode = 0; threshold = 0
        case .isosurface(let value):
            guard value.isFinite else { throw Failure.settings }
            mode = 1
            threshold = value
        }
        let worldFromVoxel = Self.double(placement.matrix) * voxelToSpace
        try Self.validateAffine(worldFromVoxel)
        let voxelFromWorld = simd_inverse(worldFromVoxel)
        let worldFromClip = simd_inverse(Self.double(camera.viewMatrix))
            * simd_inverse(Self.double(camera.projectionMatrix))
        let near: Float = depthMode == .forward ? 0 : 1
        let end: Float = camera.far.isInfinite ? 0.5 : 1 - near
        return Uniforms(
            voxelFromClip: try Self.float(voxelFromWorld * worldFromClip),
            worldFromVoxel: try Self.float(worldFromVoxel), voxelFromWorld: try Self.float(voxelFromWorld),
            scalar: SIMD4(low, high, threshold, settings.opacity),
            ray: SIMD4(near, end, settings.samplesPerVoxel, camera.far.isInfinite ? 1 : 0),
            flags: SIMD4(UInt32(settings.maximumSteps), mode, settings.sampling.rawValue, 0)
        )
    }

    private static func readable(_ texture: any MTLTexture) -> Bool {
        texture.sampleCount == 1 && texture.usage.contains(.shaderRead) && !texture.isFramebufferOnly
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
    }

    private static func validateAffine(_ matrix: simd_double4x4) throws {
        let determinant = simd_determinant(matrix)
        guard SIMD4(matrix[0].w, matrix[1].w, matrix[2].w, matrix[3].w) == SIMD4(0, 0, 0, 1),
              determinant.isFinite, determinant != 0 else { throw Failure.transform }
        _ = try float(matrix)
    }

    private static func double(_ matrix: simd_float4x4) -> simd_double4x4 {
        simd_double4x4(columns: (SIMD4(matrix[0]), SIMD4(matrix[1]), SIMD4(matrix[2]), SIMD4(matrix[3])))
    }

    private static func float(_ matrix: simd_double4x4) throws -> simd_float4x4 {
        let columns = (0..<4).map { SIMD4<Float>(matrix[$0]) }
        guard columns.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
            throw Failure.transform
        }
        return simd_float4x4(columns: (columns[0], columns[1], columns[2], columns[3]))
    }
}
