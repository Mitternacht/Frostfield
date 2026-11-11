import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// Native reference scene for animated assets, surface flow and screen effects.
@MainActor
final class LabOceanFixture {
    static let argument = "--cyber-ocean"
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let dolphin: LabDolphinAsset
    private let tunnel: MeshRenderer
    private let flow: ParticleEmitter
    private let effects: ScreenEffects
    private let tunnelID = UUID()
    private var sceneColor: (any MTLTexture)?
    private var time: Float = 0
    private var previousUpdate: CFTimeInterval?
    private var isPaused = false
    private var yaw: Float = 0
    private var pitch: Float = 0
    private var distance: Float = 5.4
    private var grainSeed: UInt32 = 0

    init(device: any MTLDevice, destinationFormat: MTLPixelFormat) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("ocean command queue") }
        self.device = device
        self.queue = queue
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3.3, near: 0.1, far: 85)
        let sourceFormat: MTLPixelFormat = .rgba16Float
        let dolphin = try LabDolphinAsset(device: device, pixelFormat: sourceFormat)
        self.dolphin = dolphin
        let sampler = try MeshSurfaceSampler(geometry: dolphin.bindGeometry)
        let world = dolphin.bindSurfaceWorldTransform
        let normals = simd_transpose(simd_inverse(world))
        let seeds = try sampler.samples(count: 4_096, seed: 42).map { sample in
            let position = world * SIMD4(sample.position, 1)
            let normal = normals * SIMD4(sample.normal, 0)
            return MeshSurfaceSample(
                position: SIMD3(position.x, position.y, position.z),
                normal: simd_normalize(SIMD3(normal.x, normal.y, normal.z)),
                textureCoordinate: sample.textureCoordinate
            )
        }
        flow = try ParticleEmitter(
            queue: queue, colorPixelFormat: sourceFormat, depthPixelFormat: MeshRenderer.depthPixelFormat,
            surface: seeds,
            flow: .init(
                speed: 0.72, frequency: 2.7, outwardSpeed: 0.28, lifetime: 1.6...4.2,
                pointSize: 5.5, color: SIMD3(0.22, 0.66, 1), intensity: 1.8
            ),
            seed: 43
        )
        effects = try ScreenEffects(queue: queue, destinationPixelFormat: destinationFormat)
        let library = try device.makeLibrary(source: Self.tunnelSource, options: nil)
        tunnel = try MeshRenderer(
            device: device, colorPixelFormat: sourceFormat, geometry: Self.tunnelGeometry(),
            material: .custom(library: library, fragment: "ocean_tunnel")
        )
    }

    func setPaused(_ paused: Bool) {
        guard isPaused != paused else { return }
        isPaused = paused
        previousUpdate = nil
    }

    func orbit(horizontal: Float, vertical: Float) {
        yaw = min(max(yaw - horizontal * 0.004, -0.62), 0.62)
        pitch = min(max(pitch - vertical * 0.004, -0.42), 0.42)
    }

    func zoom(by scale: Float) {
        guard scale.isFinite, scale > 0 else { return }
        distance = min(max(distance / scale, 3.2), 8.1)
    }

    func draw(
        pass drawablePass: MTLRenderPassDescriptor,
        drawable: any CAMetalDrawable,
        completed: @escaping @Sendable (String?) -> Void
    ) throws {
        guard let depth = drawablePass.depthAttachment.texture else { throw MeshRenderer.Failure.attachments }
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("ocean frame") }
        var delta: Float = 0
        if !isPaused {
            let now = CACurrentMediaTime()
            let elapsed = previousUpdate.map { now - $0 } ?? 1.0 / 60
            previousUpdate = now
            delta = Float(elapsed > 0.25 ? 1.0 / 60 : min(max(elapsed, 0), 1.0 / 30))
        }
        try encode(destination: drawable.texture, depth: depth, delta: delta, buffer: buffer)
        buffer.addCompletedHandler { command in
            let failure = command.status == .completed ? nil
                : command.error?.localizedDescription ?? "The ocean frame did not finish."
            completed(failure)
        }
        buffer.present(drawable)
        buffer.commit()
    }

    private func encode(
        destination: any MTLTexture, depth: any MTLTexture, delta: Float, buffer: any MTLCommandBuffer
    ) throws {
        let color = try colorTexture(width: destination.width, height: destination.height)
        if delta > 0 {
            time += delta
            grainSeed &+= 1
        }

        let aspect = Float(color.width) / Float(color.height)
        let radius = distance / max(0.69, min(1, aspect))
        let offset = SIMD3<Float>(
            sin(yaw) * cos(pitch) * radius,
            0.35 + sin(pitch) * radius,
            cos(yaw) * cos(pitch) * radius
        )
        let direction = simd_normalize(-offset)
        let pose = SceneTransform(
            translation: offset, rotation: simd_quatf(from: SIMD3<Float>(0, 0, -1), to: direction)
        )
        var graph = try dolphin.graph(camera: camera, cameraTransform: pose, time: time)
        graph.roots.append(SceneNode(
            id: tunnelID, mesh: tunnel, materialParameters: SIMD4(time, 0, 0, 1), castsShadow: false
        ))

        let scenePass = MTLRenderPassDescriptor()
        scenePass.colorAttachments[0].texture = color
        scenePass.colorAttachments[0].loadAction = .clear
        scenePass.colorAttachments[0].storeAction = .store
        scenePass.colorAttachments[0].clearColor = MTLClearColor(red: 0.002, green: 0.006, blue: 0.02, alpha: 1)
        scenePass.depthAttachment.texture = depth
        scenePass.depthAttachment.loadAction = .clear
        scenePass.depthAttachment.storeAction = .store
        scenePass.depthAttachment.clearDepth = 1

        try flow.encodeUpdate(deltaTime: delta, commandBuffer: buffer)
        try graph.encode(pass: scenePass, commandBuffer: buffer)
        scenePass.colorAttachments[0].loadAction = .load
        scenePass.depthAttachment.loadAction = .load
        scenePass.depthAttachment.storeAction = .dontCare
        let clipFromWorld = try camera.projection(aspectRatio: aspect) * simd_inverse(pose.matrix)
        try flow.encodeDraw(pass: scenePass, clipFromWorld: clipFromWorld, commandBuffer: buffer)
        try effects.encode(
            source: color, destination: destination,
            settings: ScreenEffects.Settings(
                bloomThreshold: 0.62, bloomStrength: 0.78, bloomRadius: 13,
                chromaticAberration: 1.25, grainStrength: 0.006, grainSeed: grainSeed
            ),
            commandBuffer: buffer
        )
    }

    #if os(macOS)
    func capture(to url: URL, frames: Int) throws {
        let width = 1200
        let height = 800
        let target = try LabImageCapture.target(device: device, width: width, height: height)
        let depth = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: width, height: height, mipmapped: false
        )
        depth.storageMode = .private
        depth.usage = .renderTarget
        guard let depthTarget = device.makeTexture(descriptor: depth) else {
            throw MeshRenderer.Failure.resource("ocean capture textures")
        }

        for _ in 0..<frames {
            try autoreleasepool {
                guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("ocean capture frame") }
                try encode(destination: target, depth: depthTarget, delta: 1.0 / 60, buffer: buffer)
                buffer.commit()
                buffer.waitUntilCompleted()
                guard buffer.status == .completed else {
                    throw buffer.error ?? MeshRenderer.Failure.resource("ocean capture completion")
                }
            }
        }

        try LabImageCapture.write(target, to: url)
    }
    #endif

    private func colorTexture(width: Int, height: Int) throws -> any MTLTexture {
        if let sceneColor, sceneColor.width == width, sceneColor.height == height { return sceneColor }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.renderTarget, .shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw MeshRenderer.Failure.resource("ocean color texture")
        }
        texture.label = "FrostfieldLab ocean color"
        sceneColor = texture
        return texture
    }

    private static func tunnelGeometry() throws -> MeshGeometry {
        let sectors = 80
        let rings = 52
        let stride = sectors + 1
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for ring in 0...rings {
            let v = Float(ring) / Float(rings)
            let z = 16 - v * 73
            let radius: Float = 7.2 + sin(v * 14) * 0.9
            let offset = SIMD2<Float>(sin(v * 7) * 0.85, cos(v * 8) * 0.43)
            for sector in 0...sectors {
                let u = Float(sector) / Float(sectors)
                let angle = u * 2 * .pi
                let radial = SIMD2<Float>(cos(angle), sin(angle))
                let xy = offset + radial * radius
                vertices.append(MeshVertex(
                    position: SIMD3(xy.x, xy.y, z), normal: SIMD3(-radial.x, -radial.y, 0),
                    textureCoordinate: SIMD2(u, v)
                ))
            }
        }
        for ring in 0..<rings {
            for sector in 0..<sectors {
                let a = UInt32(ring * stride + sector)
                let b = a + 1
                let c = a + UInt32(stride)
                indices += [a, b, c, b, c + 1, c]
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    private static let tunnelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct TunnelInput {
        float4 position [[position]];
        float2 uv [[user(frostfield_uv)]];
        float3 worldPosition [[user(frostfield_world_position)]];
    };

    float beam(float x, float width) {
        float d = abs(fract(x + 0.5) - 0.5);
        float smoothing = max(fwidth(x), 0.002);
        return 1.0 - smoothstep(width, width + smoothing, d);
    }

    fragment float4 ocean_tunnel(
        TunnelInput input [[stage_in]], constant float4& parameters [[buffer(0)]]
    ) {
        float u = input.uv.x;
        float v = input.uv.y;
        float time = parameters.x;
        float wave = sin(u * 31.416 + v * 13.0 - time * 0.8);
        float travel = v * 27.0 + time * 0.83 + wave * 0.38;
        float rings = beam(travel, 0.015);
        float veins = beam(u * 32.0 + sin(v * 15.0 + time * 0.42) * 0.24, 0.021);
        float pulse = pow(0.5 + 0.5 * sin(v * 62.0 + u * 12.566 + time * 2.2), 5.0);
        float depth = exp(-max(0.0, -input.worldPosition.z) * 0.026);
        float3 cyan = float3(0.055, 0.58, 1.8);
        float3 violet = float3(1.0, 0.035, 1.45);
        float3 hue = mix(cyan, violet, 0.5 + 0.5 * sin(u * 12.566 + v * 6.0));
        float3 base = float3(0.004, 0.009, 0.037) + hue * (0.012 + 0.027 * pulse);
        float3 light = hue * (rings * (0.30 + 0.68 * pulse) + veins * (0.07 + 0.17 * pulse));
        return float4(base + light * (0.18 + depth * 0.82), parameters.w);
    }
    """
}
