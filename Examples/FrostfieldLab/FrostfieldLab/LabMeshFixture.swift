import Frostfield
import MetalKit
import simd

final class LabMeshFixture {
    private let queue: any MTLCommandQueue
    private let renderer: MeshRenderer
    private let camera: PerspectiveCamera
    private let customMaterial: Bool
    private let lighting: Bool
    private let particles: ParticleEmitter?
    private let grade: ColorGrade?
    private let gradeStrength: Float
    private var sceneColor: (any MTLTexture)?

    static func isRequested(_ arguments: [String]) -> Bool {
        ["--mesh", "--custom-material", "--lit", "--particles", "--grade", "--grade-bypass"]
            .contains(where: arguments.contains)
    }

    init(device: any MTLDevice, pixelFormat: MTLPixelFormat, arguments: [String]) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("command queue") }
        self.queue = queue
        self.camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 100)
        self.customMaterial = arguments.contains("--custom-material")
        self.lighting = arguments.contains("--lit")
        let missingShader = arguments.contains("--simulate-bad-shader")
        let material: MeshMaterial
        if customMaterial || missingShader {
            let library = try device.makeLibrary(source: Self.materialSource, options: nil)
            material = .custom(library: library, fragment: missingShader ? "missing_fixture_fragment" : "lab_surface")
        } else {
            material = lighting ? .lambert : .unlit
        }

        let gradeEnabled = arguments.contains("--grade") || arguments.contains("--grade-bypass")
        let sceneFormat: MTLPixelFormat = gradeEnabled ? .rgba16Float : pixelFormat
        renderer = try MeshRenderer(device: device, colorPixelFormat: sceneFormat, geometry: .cube(), material: material)
        grade = gradeEnabled ? try ColorGrade(device: device, destinationPixelFormat: pixelFormat) : nil
        gradeStrength = arguments.contains("--grade-bypass") ? 0 : 0.72
        particles = arguments.contains("--particles")
            ? try ParticleEmitter(queue: queue, colorPixelFormat: sceneFormat, seed: 42)
            : nil
    }

    func draw(
        pass: MTLRenderPassDescriptor, drawable: any CAMetalDrawable, frame: Int,
        collectGPUTiming: Bool = false, completed: @escaping @Sendable (String?, Double?) -> Void
    ) throws {
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("command buffer") }
        let scenePass = try renderPass(for: pass, destination: drawable.texture)
        guard let target = scenePass.colorAttachments[0].texture else { throw MeshRenderer.Failure.attachments }

        let angle = Float(frame % 720) * .pi / 360
        let pitch = simd_quatf(angle: -0.38, axis: SIMD3(1, 0, 0))
        let yaw = simd_quatf(angle: 0.6 + angle, axis: SIMD3(0, 1, 0))
        let parameters: SIMD4<Float>
        if customMaterial {
            parameters = SIMD4(angle, 0, 0, 1)
        } else if lighting {
            parameters = SIMD4(-0.4, 0.8, 1, 0.9)
        } else {
            parameters = SIMD4(0.72, 0.86, 1, 1)
        }
        let aspect = Float(target.width) / Float(target.height)
        let scene = MeshScene(
            camera: camera,
            transform: SceneTransform(rotation: yaw * pitch),
            cameraTransform: SceneTransform(translation: SIMD3(0, 0, 2.8 / min(1, aspect))),
            materialParameters: parameters
        )
        try particles?.encodeUpdate(deltaTime: frame > 0 ? 1.0 / 60 : 0, commandBuffer: buffer)
        try renderer.encode(scene, pass: scenePass, commandBuffer: buffer)
        if let particles {
            let overlay = MTLRenderPassDescriptor()
            overlay.colorAttachments[0].texture = target
            overlay.colorAttachments[0].loadAction = .load
            overlay.colorAttachments[0].storeAction = .store
            let clipFromWorld = try camera.projection(aspectRatio: aspect)
                * simd_inverse(scene.cameraTransform.matrix)
            try particles.encodeDraw(pass: overlay, clipFromWorld: clipFromWorld, commandBuffer: buffer)
        }
        try grade?.encode(source: target, destination: drawable.texture, strength: gradeStrength, commandBuffer: buffer)
        buffer.addCompletedHandler { command in
            let error = command.status == .error
                ? command.error?.localizedDescription ?? "Metal could not finish the mesh frame." : nil
            #if os(visionOS)
            let gpuSeconds: Double? = nil
            #else
            var gpuSeconds: Double?
            if collectGPUTiming, command.status == .completed {
                let start = command.gpuStartTime
                let end = command.gpuEndTime
                if start.isFinite, end.isFinite, start > 0, end > start {
                    gpuSeconds = end - start
                }
            }
            #endif
            completed(error, gpuSeconds)
        }
        buffer.present(drawable)
        buffer.commit()
    }

    private func renderPass(
        for drawablePass: MTLRenderPassDescriptor, destination: any MTLTexture
    ) throws -> MTLRenderPassDescriptor {
        guard grade != nil else { return drawablePass }
        if sceneColor?.width != destination.width || sceneColor?.height != destination.height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: destination.width, height: destination.height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            sceneColor = queue.device.makeTexture(descriptor: descriptor)
            sceneColor?.label = "Frostfield grading input"
        }
        guard let sceneColor else { throw MeshRenderer.Failure.resource("offscreen scene color") }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = sceneColor
        pass.colorAttachments[0].clearColor = drawablePass.colorAttachments[0].clearColor
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = drawablePass.depthAttachment.texture
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        return pass
    }

    private static let materialSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Surface {
        float4 position [[position]];
        float3 normal [[user(frostfield_normal)]];
    };

    fragment float4 lab_surface(Surface input [[stage_in]], constant float4& values [[buffer(0)]]) {
        float3 hue = 0.55 + 0.4 * cos(values.x + float3(0.0, 2.1, 4.2));
        float brightness = 0.42 + 0.58 * abs(normalize(input.normal).z);
        return float4(hue * brightness, 1);
    }
    """
}
