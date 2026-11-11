import Foundation
import Metal
import simd
import Synchronization

/// GPU particles whose motion is a `BufferShaderGraph` and that draw as soft,
/// camera-facing quads into an HDR scene pass.
///
/// Each particle's fields follow one convention, so a graph author only writes
/// the motion:
/// - field 0: world position in xyz and age in seconds in w
/// - field 1: velocity in xyz and lifetime in seconds in w
/// - field 2 (optional): linear RGB in xyz and world-space diameter in w
///
/// Without field 2 particles use `defaultColor` and `defaultSize`. A particle
/// fades in and out over its lifetime; set its lifetime to zero to hide it.
/// The graph handles emission and respawning, for example by resetting a
/// particle whose age exceeds its lifetime. Its uniforms, such as a time step,
/// are the app's to set between frames. A simulation that reads the graph clock
/// (`time`, `deltaTime`, `frame`) always has live `uniforms`, so the app can
/// advance the clock with `ShaderUniforms.advanceFrame(by:)`.
public final class ShaderParticleSystem {
    public enum Blending: Sendable {
        /// Adds premultiplied radiance; order independent.
        case additive
        /// Premultiplied source-over. Particles are not sorted.
        case alpha
    }

    public static let maximumCapacity = 1_048_576

    public let capacity: Int
    public let simulation: BufferShaderGraph
    /// The values copied into each update: the ones passed in, live values the
    /// system created because the simulation reads the clock, or nil for the
    /// declared defaults.
    public let uniforms: ShaderUniforms?
    public let colorPixelFormat: MTLPixelFormat
    public let depthMode: MeshDepthMode
    private let kernel: ShaderBufferKernel
    private let buffers: [any MTLBuffer]
    private let drawPipelines: ParticleDrawPipelines
    private let depthState: any MTLDepthStencilState
    private let state = Mutex(0)
    private let style: SIMD4<Float>

    /// Prepares the simulation and draw pipelines and uploads the initial fields.
    ///
    /// - Parameters:
    ///   - simulation: A buffer graph with two or three fields following the convention above.
    ///   - initial: Exactly `capacity × simulation.fieldCount` finite float4 values, element by element.
    ///   - defaultColor: Linear RGB used without field 2.
    ///   - defaultSize: World-space diameter used without field 2.
    ///   - colorPixelFormat: The scene color format, usually `.rgba16Float`.
    ///   - depthMode: The depth convention used by the scene pass.
    public init(
        device: any MTLDevice, capacity: Int, simulation: BufferShaderGraph, initial: [SIMD4<Float>],
        uniforms: ShaderUniforms? = nil, blending: Blending = .additive,
        defaultColor: SIMD3<Float> = SIMD3(repeating: 1), defaultSize: Float = 0.05,
        colorPixelFormat: MTLPixelFormat = .rgba16Float, depthMode: MeshDepthMode = .forward
    ) throws {
        try depthMode.requireProjectionDepth()
        guard (1...Self.maximumCapacity).contains(capacity) else { throw Failure.capacity }
        guard (2...3).contains(simulation.fieldCount) else { throw Failure.fieldCount }
        guard initial.count == capacity * simulation.fieldCount,
              initial.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
            throw Failure.initialValues
        }
        guard defaultColor.x.isFinite, defaultColor.y.isFinite, defaultColor.z.isFinite,
              simd_reduce_min(defaultColor) >= 0, defaultSize.isFinite, defaultSize > 0 else { throw Failure.appearance }
        if let uniforms, !uniforms.matches(simulation.uniforms) { throw Failure.uniforms }
        kernel = try ShaderBufferKernel(graph: simulation, device: device)
        buffers = try (0..<2).map { index in
            guard let buffer = initial.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else {
                throw Failure.resource("particle buffer \(index)")
            }
            buffer.label = "Frostfield particles \(index)"
            return buffer
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield shader particles"
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        guard let vertex = library.makeFunction(name: "frostfield_shader_particle_vertex") else {
            throw Failure.pipeline("Frostfield's shader particle vertex function is missing.")
        }
        descriptor.vertexFunction = vertex
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        descriptor.depthAttachmentPixelFormat = MeshRenderer.depthPixelFormat
        let color = descriptor.colorAttachments[0]!
        color.isBlendingEnabled = true
        color.sourceRGBBlendFactor = .one
        color.sourceAlphaBlendFactor = .one
        color.destinationRGBBlendFactor = blending == .additive ? .one : .oneMinusSourceAlpha
        color.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        do {
            drawPipelines = try ParticleDrawPipelines(
                library: library, descriptor: descriptor, fragment: "frostfield_shader_particle_fragment"
            )
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = depthMode == .forward ? .less : .greater
        depth.isDepthWriteEnabled = false
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else { throw Failure.resource("depth state") }
        self.depthState = depthState
        self.capacity = capacity
        self.simulation = simulation
        // Without live values the clock would stay at zero and a deltaTime step would never move.
        self.uniforms = try uniforms ?? (simulation.program.readsClock ? ShaderUniforms(simulation.uniforms) : nil)
        self.colorPixelFormat = colorPixelFormat
        self.depthMode = depthMode
        style = SIMD4(defaultColor, defaultSize)
    }

    /// Runs one simulation step on the GPU, ping-ponging the particle buffers.
    /// The graph's uniforms are copied as they are now.
    public func encodeUpdate(commandBuffer: any MTLCommandBuffer) throws {
        try state.withLock { current in
            try kernel.encode(
                elementCount: capacity, from: buffers[current], to: buffers[1 - current],
                uniforms: uniforms, commandBuffer: commandBuffer
            )
            current = 1 - current
        }
    }

    /// Draws the current particles. The pass must load color in `colorPixelFormat`
    /// and a matching Depth32Float attachment prepared with the system's depth
    /// mode. Particles test depth without writing it. Encode after the scene pass
    /// and before post-processing. Supply `softParticles` to fade intersections
    /// against a separately stored opaque depth snapshot of the same view.
    public func encodeDraw(
        pass: MTLRenderPassDescriptor, camera: PerspectiveCamera, cameraTransform: SceneTransform,
        commandBuffer: any MTLCommandBuffer, softParticles: SoftParticleDepth? = nil
    ) throws {
        guard let color = pass.colorAttachments[0].texture, color.pixelFormat == colorPixelFormat,
              let depth = pass.depthAttachment.texture, depth.pixelFormat == MeshRenderer.depthPixelFormat,
              color.width == depth.width, color.height == depth.height else { throw Failure.attachments }
        let aspect = Float(color.width) / Float(color.height)
        let projection = try camera.projection(aspectRatio: aspect, depthMode: depthMode)
        var clipFromWorld = projection * simd_inverse(cameraTransform.matrix)
        let softDepth = try softParticles?.prepare(
            pass: pass, clipFromView: projection, depthMode: depthMode,
            device: kernel.device, commandBuffer: commandBuffer
        )
        let pipeline: any MTLRenderPipelineState
        do {
            pipeline = try drawPipelines.state(withSoftDepth: softDepth != nil)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        var axes = [
            SIMD4(cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)), 0),
            SIMD4(cameraTransform.rotation.act(SIMD3<Float>(0, 1, 0)), 0),
            style,
            SIMD4(Float(simulation.fieldCount), 0, 0, 0),
        ]
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.resource("draw encoder") }
        encoder.label = "Frostfield shader particles"
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setVertexBuffer(buffers[state.withLock { $0 }], offset: 0, index: 0)
        encoder.setVertexBytes(&clipFromWorld, length: MemoryLayout<simd_float4x4>.size, index: 1)
        encoder.setVertexBytes(&axes, length: MemoryLayout<SIMD4<Float>>.stride * axes.count, index: 2)
        softDepth?.bind(into: encoder)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: capacity)
        encoder.endEncoding()
    }

    /// Copies the most recently written fields after the GPU has finished with them.
    public func fields() -> [SIMD4<Float>] {
        let buffer = buffers[state.withLock { $0 }]
        return Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: SIMD4<Float>.self),
                                         count: capacity * simulation.fieldCount))
    }

    public enum Failure: LocalizedError, Equatable {
        case capacity
        case fieldCount
        case initialValues
        case appearance
        case uniforms
        case attachments
        case pipeline(String)
        case resource(String)

        public var errorDescription: String? {
            switch self {
            case .capacity: "A particle system holds 1 to \(ShaderParticleSystem.maximumCapacity) particles."
            case .fieldCount: "Particle simulations use two fields (position/age, velocity/lifetime) or three (plus color/size)."
            case .initialValues: "Supply one finite float4 per particle field, element by element."
            case .appearance: "The default color must be finite and nonnegative and the default size positive."
            case .uniforms: "Particle uniforms must match the simulation graph's declarations."
            case .attachments: "Draw particles into a pass with the system's color format and Depth32Float depth of the same size."
            case .pipeline(let detail): "Could not prepare the particle pipeline. \(detail)"
            case .resource(let name): "Metal could not allocate \(name)."
            }
        }
    }
}
