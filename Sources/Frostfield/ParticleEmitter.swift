import Foundation
import Metal
import simd

/// Fixed-capacity particles updated and drawn on one conventional Metal queue.
public final class ParticleEmitter {
    public static let maximumCapacity = 16_384
    public let capacity: Int
    public let depthMode: MeshDepthMode

    private let queue: any MTLCommandQueue
    private let particles: any MTLBuffer
    private let updatePipeline: any MTLComputePipelineState
    private let drawPipelines: ParticleDrawPipelines
    private let depthState: (any MTLDepthStencilState)?
    private let colorPixelFormat: MTLPixelFormat
    private let depthPixelFormat: MTLPixelFormat
    private let flow: SurfaceFlow?

    /// Settings for additive particles advected through a 3D curl field.
    public struct SurfaceFlow: Sendable {
        /// Field speed in scene units per second.
        public let speed: Float
        /// Spatial frequency in radians per scene unit.
        public let frequency: Float
        /// Launch normal drift in scene units per second.
        public let outwardSpeed: Float
        /// Uniformly sampled particle lifetimes in seconds.
        public let lifetime: ClosedRange<Float>
        /// Largest point diameter in drawable pixels.
        public let pointSize: Float
        /// Linear RGB in 0...1.
        public let color: SIMD3<Float>
        /// Additive emission multiplier in 0...2.
        public let intensity: Float

        public init(
            speed: Float = 0.75,
            frequency: Float = 2.5,
            outwardSpeed: Float = 0.18,
            lifetime: ClosedRange<Float> = 1.4...3.4,
            pointSize: Float = 6,
            color: SIMD3<Float> = SIMD3(0.2, 0.75, 1),
            intensity: Float = 1.25
        ) {
            self.speed = speed
            self.frequency = frequency
            self.outwardSpeed = outwardSpeed
            self.lifetime = lifetime
            self.pointSize = pointSize
            self.color = color
            self.intensity = intensity
        }
    }

    public convenience init(
        queue: any MTLCommandQueue,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat = .invalid,
        capacity: Int = 512,
        seed: UInt64,
        depthMode: MeshDepthMode = .forward
    ) throws {
        let values = try Self.seededState(capacity: capacity, seed: seed)
        try self.init(
            queue: queue, colorPixelFormat: colorPixelFormat, depthPixelFormat: depthPixelFormat,
            capacity: capacity, values: values, flow: nil, depthMode: depthMode
        )
    }

    /// Creates one particle per sample and respawns it at the same position.
    /// Provide samples in the coordinate space passed to `encodeDraw`.
    /// Source positions are bounded to ±1,000,000 per coordinate.
    /// Flow uses additive RGB with source-over alpha and supports HDR targets.
    public convenience init(
        queue: any MTLCommandQueue,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat = .invalid,
        surface: [MeshSurfaceSample],
        flow: SurfaceFlow = .init(),
        seed: UInt64,
        depthMode: MeshDepthMode = .forward
    ) throws {
        let values = try Self.seededFlowState(surface: surface, flow: flow, seed: seed)
        try self.init(
            queue: queue, colorPixelFormat: colorPixelFormat, depthPixelFormat: depthPixelFormat,
            capacity: surface.count, values: values, flow: flow, depthMode: depthMode
        )
    }

    private init(
        queue: any MTLCommandQueue,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat,
        capacity: Int,
        values: [SIMD4<Float>],
        flow: SurfaceFlow?,
        depthMode: MeshDepthMode
    ) throws {
        try depthMode.requireProjectionDepth()
        let device = queue.device
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        let shader = flow == nil ? "frostfield_particles" : "frostfield_surface_flow"
        let update = try Self.function("\(shader)_update", in: library)
        let vertex = try Self.function("\(shader)_vertex", in: library)

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = flow == nil ? "Frostfield particles" : "Frostfield surface flow particles"
        descriptor.vertexFunction = vertex
        descriptor.depthAttachmentPixelFormat = depthPixelFormat
        descriptor.inputPrimitiveTopology = .point
        guard let color = descriptor.colorAttachments[0] else {
            throw Failure.resource("particle pipeline color attachment")
        }
        color.pixelFormat = colorPixelFormat
        color.isBlendingEnabled = true
        color.sourceRGBBlendFactor = .one
        color.destinationRGBBlendFactor = flow == nil ? .oneMinusSourceAlpha : .one
        color.sourceAlphaBlendFactor = .one
        color.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        do {
            updatePipeline = try device.makeComputePipelineState(function: update)
            drawPipelines = try ParticleDrawPipelines(library: library, descriptor: descriptor, fragment: "\(shader)_fragment")
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }

        if depthPixelFormat == .invalid {
            depthState = nil
        } else {
            let depth = MTLDepthStencilDescriptor()
            depth.depthCompareFunction = depthMode == .forward ? .lessEqual : .greaterEqual
            depth.isDepthWriteEnabled = false
            guard let state = device.makeDepthStencilState(descriptor: depth) else {
                throw Failure.resource("particle depth state")
            }
            depthState = state
        }

        let buffer = values.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap {
                device.makeBuffer(bytes: $0, length: bytes.count, options: [.storageModeShared, .hazardTrackingModeTracked])
            }
        }
        guard let buffer else { throw Failure.resource("particle state buffer") }
        buffer.label = "Frostfield particle state"

        self.capacity = capacity
        self.depthMode = depthMode
        self.queue = queue
        self.colorPixelFormat = colorPixelFormat
        self.depthPixelFormat = depthPixelFormat
        self.flow = flow
        particles = buffer
    }

    /// Encodes the simulation before its draw. Pass zero while paused.
    /// Negative and nonfinite deltas skip the update. Positive deltas cap at 1/30 second.
    public func encodeUpdate(deltaTime: Float, commandBuffer: any MTLCommandBuffer) throws {
        try validate(commandBuffer)
        var step = Self.timeStep(deltaTime)
        guard step > 0 else { return }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Failure.encoder("compute")
        }
        var count = UInt32(capacity)
        encoder.setComputePipelineState(updatePipeline)
        encoder.setBuffer(particles, offset: 0, index: 0)
        encoder.setBytes(&step, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        if let flow {
            var dynamics = SIMD4(flow.speed, flow.frequency, flow.outwardSpeed, 0)
            encoder.setBytes(&dynamics, length: MemoryLayout<SIMD4<Float>>.size, index: 3)
        }
        let width = updatePipeline.threadExecutionWidth
        encoder.dispatchThreadgroups(
            MTLSize(width: (capacity + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Draws the current state. Use a retained-reference buffer from the emitter's
    /// queue, encode updates before draws, and submit buffers in simulation order.
    /// When overlaying another pass, load its color and optional depth attachments
    /// and use the projection and depth mode of that pass.
    public func encodeDraw(
        pass: MTLRenderPassDescriptor,
        clipFromWorld: simd_float4x4,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encodeDraw(pass: pass, clipFromWorld: clipFromWorld, softDepth: nil, commandBuffer: commandBuffer)
    }

    /// Draws with a separate opaque depth snapshot to soften intersections.
    /// Use the camera's view and projection that produced the snapshot and the
    /// emitter's depth mode. Both fountain and surface-flow particles use the fade.
    public func encodeDraw(
        pass: MTLRenderPassDescriptor, camera: ShaderGraphCamera,
        softParticles: SoftParticleDepth, commandBuffer: any MTLCommandBuffer
    ) throws {
        let prepared = try softParticles.prepare(
            pass: pass, clipFromView: camera.projectionMatrix, depthMode: depthMode,
            device: queue.device, commandBuffer: commandBuffer
        )
        try encodeDraw(
            pass: pass, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
            softDepth: prepared, commandBuffer: commandBuffer
        )
    }

    private func encodeDraw(
        pass: MTLRenderPassDescriptor, clipFromWorld: simd_float4x4, softDepth: SoftParticleDepth.Prepared?,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(commandBuffer)
        guard let color = pass.colorAttachments[0].texture,
              color.device === queue.device, color.pixelFormat == colorPixelFormat,
              color.textureType == .type2D, color.sampleCount == 1,
              pass.stencilAttachment.texture == nil,
              (1..<8).allSatisfy({ pass.colorAttachments[$0].texture == nil }) else {
            throw Failure.attachments
        }
        if depthPixelFormat == .invalid {
            guard pass.depthAttachment.texture == nil else { throw Failure.attachments }
        } else {
            guard let depth = pass.depthAttachment.texture,
                  depth.device === queue.device, depth.pixelFormat == depthPixelFormat,
                  depth.textureType == .type2D, depth.sampleCount == 1,
                  depth.width == color.width, depth.height == color.height else {
                throw Failure.attachments
            }
        }

        let (x, y, z, w) = clipFromWorld.columns
        guard Self.isFinite(x), Self.isFinite(y), Self.isFinite(z), Self.isFinite(w) else {
            throw Failure.transform
        }
        let pipeline: any MTLRenderPipelineState
        do {
            pipeline = try drawPipelines.state(withSoftDepth: softDepth != nil)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder("render")
        }

        var clip = clipFromWorld
        encoder.setRenderPipelineState(pipeline)
        if let depthState {
            encoder.setDepthStencilState(depthState)
        }
        encoder.setVertexBuffer(particles, offset: 0, index: 0)
        encoder.setVertexBytes(&clip, length: MemoryLayout<simd_float4x4>.size, index: 1)
        if let flow {
            var size = flow.pointSize
            var emission = SIMD4(flow.color, flow.intensity)
            encoder.setVertexBytes(&size, length: MemoryLayout<Float>.size, index: 2)
            encoder.setFragmentBytes(&emission, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        }
        softDepth?.bind(into: encoder)
        encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: capacity)
        encoder.endEncoding()
    }

    private func validate(_ buffer: any MTLCommandBuffer) throws {
        // The directly bound, hazard-tracked buffer stays on this queue. Metal orders
        // its compute writes and render reads, including across submitted buffers.
        guard buffer.commandQueue === queue else { throw Failure.wrongQueue }
        guard buffer.retainedReferences, buffer.status == .notEnqueued || buffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
    }

    package static func timeStep(_ deltaTime: Float) -> Float {
        guard deltaTime.isFinite, deltaTime > 0 else { return 0 }
        return min(deltaTime, 1.0 / 30)
    }

    // Two float4 records per particle: position/age, then velocity/lifetime.
    package static func seededState(capacity: Int, seed: UInt64) throws -> [SIMD4<Float>] {
        guard (1...maximumCapacity).contains(capacity) else { throw Failure.capacity(capacity) }
        var generator = SceneSeedGenerator(state: seed)
        var state: [SIMD4<Float>] = []
        state.reserveCapacity(capacity * 2)
        for _ in 0..<capacity {
            let velocity = SIMD3<Float>(
                (generator.nextUnit() - 0.5) * 1.25,
                0.8 + generator.nextUnit() * 0.6,
                (generator.nextUnit() - 0.5) * 1.25
            )
            let lifetime = 0.8 + generator.nextUnit() * 1.0
            let age = generator.nextUnit() * lifetime
            let position = velocity * age + SIMD3<Float>(0, -0.7 * age * age, 0)
            state.append(SIMD4<Float>(position, age))
            state.append(SIMD4<Float>(velocity, lifetime))
        }
        return state
    }

    // Three float4 records: position/age, launch position/lifetime, and normal/drift.
    package static func seededFlowState(
        surface: [MeshSurfaceSample], flow: SurfaceFlow, seed: UInt64
    ) throws -> [SIMD4<Float>] {
        guard (1...maximumCapacity).contains(surface.count) else { throw Failure.capacity(surface.count) }
        guard (0...100).contains(flow.speed), (0.001...100).contains(flow.frequency),
              (0...100).contains(flow.outwardSpeed), (0.1...60).contains(flow.lifetime.lowerBound),
              (flow.lifetime.lowerBound...60).contains(flow.lifetime.upperBound),
              (1...64).contains(flow.pointSize), (0...2).contains(flow.intensity),
              (0...1).contains(flow.color.x), (0...1).contains(flow.color.y), (0...1).contains(flow.color.z) else {
            throw Failure.flowSettings
        }
        var generator = SceneSeedGenerator(state: seed)
        var state: [SIMD4<Float>] = []
        state.reserveCapacity(surface.count * 3)
        for (index, sample) in surface.enumerated() {
            let position = sample.position
            let normal = sample.normal
            let normalScale = max(abs(normal.x), max(abs(normal.y), abs(normal.z)))
            let limit: ClosedRange<Float> = -1_000_000...1_000_000
            guard limit.contains(position.x), limit.contains(position.y), limit.contains(position.z),
                  isFinite(SIMD4(normal, 0)), normalScale > 0 else { throw Failure.surfaceSample(index) }

            let minimum = flow.lifetime.lowerBound
            let lifetime = minimum + generator.nextUnit() * (flow.lifetime.upperBound - minimum)
            let age = generator.nextUnit() * lifetime
            state.append(SIMD4(position, age))
            state.append(SIMD4(position, lifetime))
            state.append(SIMD4(simd_normalize(normal / normalScale), 0.8 + 0.4 * generator.nextUnit()))
        }
        return state
    }

    private static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    private static func function(_ name: String, in library: any MTLLibrary) throws -> any MTLFunction {
        guard let function = library.makeFunction(name: name) else { throw Failure.function(name) }
        return function
    }

    public enum Failure: LocalizedError, Equatable {
        case capacity(Int)
        case flowSettings
        case surfaceSample(Int)
        case library(String)
        case function(String)
        case pipeline(String)
        case resource(String)
        case wrongQueue
        case commandBuffer
        case attachments
        case transform
        case encoder(String)

        public var errorDescription: String? {
            switch self {
            case .capacity(let count): "Particle capacity \(count) is outside 1...\(ParticleEmitter.maximumCapacity)."
            case .flowSettings: "Use finite flow settings within the supported speed, frequency, lifetime, size and emission ranges."
            case .surfaceSample(let index): "Surface sample \(index) needs a bounded finite position and a finite, nonzero normal."
            case .library(let message): "Could not load Frostfield’s shader library: \(message)"
            case .function(let name): "The shader library does not contain \(name)."
            case .pipeline(let message): "Could not create a particle pipeline: \(message)"
            case .resource(let name): "Metal could not allocate \(name)."
            case .wrongQueue: "Use a command buffer from the particle emitter’s queue."
            case .commandBuffer: "Use an uncommitted command buffer with retained references."
            case .attachments: "Use one matching single-sample color texture and the configured depth texture, if any."
            case .transform: "The particle camera matrix must be finite."
            case .encoder(let kind): "Metal could not create a particle \(kind) encoder."
            }
        }
    }
}
