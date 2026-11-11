import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PlanarRefractionTests {
    private let gpu: GeometryRenderFixture
    private let center = SIMD3<Float>(0, 0, -4)
    private let normal = simd_normalize(SIMD3<Float>(0.18, 0.12, 1))
    private let frame = SceneTransform(
        translation: SIMD3(4, -2, 7), rotation: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(1, 2, 1)))
    )

    init() throws { gpu = try GeometryRenderFixture(side: 96) }

    @Test("Same-view refraction clips the foreground, skips subtrees and aligns distorted texture sampling",
          arguments: [MeshDepthMode.forward, .reversed], [true, false])
    func oppositeSideCapture(mode: MeshDepthMode, perspective: Bool) throws {
        let target = try captureTarget()
        let ordinaryOutput = try gpu.makeTarget()
        let distortedOutput = try gpu.makeTarget()
        let camera = try camera(mode: mode, perspective: perspective)
        let first = planePoint(x: -0.78)
        let second = planePoint(x: 0.65)
        let firstUV = uv(first, camera: camera)
        let secondUV = uv(second, camera: camera)
        let red = SIMD3<Float>(0.875, 0.125, 0.0625)
        let green = SIMD3<Float>(0.0625, 0.75, 0.25)
        let blue = SIMD3<Float>(0.0625, 0.125, 0.9375)
        func alongRay(_ point: SIMD3<Float>, _ distance: Float) -> SIMD3<Float> {
            let direction = perspective ? simd_normalize(point) : SIMD3<Float>(0, 0, -1)
            return point + distance * direction
        }

        let surface = try makeSurface(sampling: target.color, mode: mode)
        let distorted = try makeSurface(sampling: target.color, mode: mode, offset: secondUV - firstUV)
        let refraction = try PlanarRefraction(plane: worldPlane(), refractor: surface.id)
        let view = try #require(try refraction.refractedView(camera, depthMode: mode))
        #expect(view.viewMatrix == camera.viewMatrix)
        for column in 0..<4 {
            #expect(view.projectionMatrix[column].x == camera.projectionMatrix[column].x)
            #expect(view.projectionMatrix[column].y == camera.projectionMatrix[column].y)
            #expect(view.projectionMatrix[column].w == camera.projectionMatrix[column].w)
        }

        let foreground = try marker(alongRay(first, -0.65), blue, mode: mode)
        let excluded = try marker(alongRay(second, 0.35), blue, mode: mode)
        var refractor = surface
        let surfaceFromLocal = simd_inverse(surface.transform.rotation)
        refractor.children = [try marker(
            surfaceFromLocal.act(alongRay(first, 0.35) - surface.transform.translation),
            blue, mode: mode, rotation: surfaceFromLocal
        )]
        let nodes = [
            try marker(alongRay(first, 1.2), red, mode: mode),
            try marker(alongRay(second, 1.2), green, mode: mode),
            foreground, excluded, refractor,
        ]
        let graph = SceneGraph(roots: [SceneNode(transform: frame, children: nodes)], camera: gpu.camera)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try #require(try refraction.encode(
            graph, camera: camera, color: target.color, depth: target.depth, depthMode: mode,
            excluding: [excluded.id], commandBuffer: command
        ))
        // The foreground draw disables CPU culling and is rejected by the GPU clip plane.
        #expect(statistics.drawnMeshCount == 3)

        for (node, output) in [(surface, ordinaryOutput), (distorted, distortedOutput)] {
            try SceneGraph(roots: [SceneNode(transform: frame, children: [node])], camera: gpu.camera).encode(
                pass: gpu.pass(color: output.color, depth: output.depth, depthMode: mode), commandBuffer: command,
                clipFromWorld: camera.projectionMatrix * camera.viewMatrix, cameraPosition: frame.translation,
                directionToCamera: perspective ? nil : frame.rotation.act(SIMD3(0, 0, 1)),
                depthMode: mode, graphCamera: camera
            )
        }
        try gpu.finish(command)

        let capture = try gpu.pixels(of: target.color)
        let ordinary = try gpu.pixels(of: ordinaryOutput.color)
        let distortion = try gpu.pixels(of: distortedOutput.color)
        for (point, expected) in [(firstUV, red), (secondUV, green)] {
            #expect(isNearly(pixel(capture, at: point), SIMD4(expected, 1), tolerance: 0.006))
            #expect(isNearly(pixel(ordinary, at: point), SIMD4(expected, 1), tolerance: 0.006))
        }
        #expect(isNearly(pixel(distortion, at: firstUV), SIMD4(green, 1), tolerance: 0.006))
        #expect(!capture.contains { $0.z > 0.9 })
    }

    @Test("Clip bias uses world units and an inactive capture clears reused targets")
    func biasAndInactiveCapture() async throws {
        let camera = try camera(mode: .reversed, perspective: true)
        let plane = try worldPlane()
        let ordinary = try PlanarRefraction(plane: plane, refractor: UUID())
        let biased = try PlanarRefraction(plane: plane, refractor: UUID(), clipBias: 0.5)
        let point = frame.translation + frame.rotation.act(center - normal * 0.25)
        let ordinaryView = try #require(try ordinary.refractedView(camera, depthMode: .reversed))
        let biasedView = try #require(try biased.refractedView(camera, depthMode: .reversed))
        let ordinaryClip = ordinaryView.projectionMatrix * ordinaryView.viewMatrix * SIMD4(point, 1)
        let biasedClip = biasedView.projectionMatrix * biasedView.viewMatrix * SIMD4(point, 1)
        #expect(ordinaryClip.z < ordinaryClip.w)
        #expect(biasedClip.z > biasedClip.w)
        #expect(throws: PlanarRefraction.Failure.clipBias) {
            try PlanarRefraction(plane: plane, refractor: UUID(), clipBias: .infinity)
        }
        #expect(throws: PlanarReflection.Failure.projection) {
            try biased.refractedView(camera, depthMode: .forward)
        }

        let reversedPlane = try Plane3D(normal: -plane.normal, through: frame.translation + frame.rotation.act(center))
        let inactive = try PlanarRefraction(plane: reversedPlane, refractor: UUID())
        let target = try captureTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(try inactive.encode(
            SceneGraph(roots: [], camera: gpu.camera), camera: camera, color: target.color, depth: target.depth,
            depthMode: .reversed, clearColor: .init(red: 0.25, green: 0.5, blue: 0.75, alpha: 0.5),
            commandBuffer: command
        ) == nil)
        try gpu.finish(command)
        #expect(try gpu.pixels(of: target.color).allSatisfy { $0 == SIMD4(0.25, 0.5, 0.75, 0.5) })
        let depth = try await TextureTransfer.read(
            from: target.depth, region: MTLRegionMake2D(0, 0, 1, 1), commandQueue: gpu.queue
        )
        #expect(depth.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) } == 0)
    }

    @Test("Another surface sampling the active target is rejected")
    func rejectsFeedback() throws {
        let camera = try camera(mode: .forward, perspective: true)
        let target = try captureTarget()
        let alias = try #require(target.color.makeTextureView(pixelFormat: .rgba16Float))
        let reader = try makeSurface(sampling: alias, mode: .forward)
        let graph = SceneGraph(roots: [SceneNode(transform: frame, children: [reader])], camera: gpu.camera)
        let refraction = try PlanarRefraction(plane: worldPlane(), refractor: UUID())
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PlanarReflection.Failure.feedback) {
            try refraction.encode(graph, camera: camera, color: target.color, depth: target.depth, commandBuffer: command)
        }
    }

    private func captureTarget() throws -> (color: any MTLTexture, depth: any MTLTexture) {
        func texture(_ format: MTLPixelFormat, storage: MTLStorageMode) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: gpu.side, height: gpu.side, mipmapped: false
            )
            descriptor.storageMode = storage
            descriptor.usage = [.renderTarget, .shaderRead]
            return try #require(gpu.device.makeTexture(descriptor: descriptor))
        }
        return (try texture(.rgba16Float, storage: .shared), try texture(.depth32Float, storage: .private))
    }

    private func camera(mode: MeshDepthMode, perspective: Bool) throws -> ShaderGraphCamera {
        var projection: simd_float4x4
        if perspective {
            projection = try gpu.camera.projection(aspectRatio: 1, depthMode: mode)
            projection[2].x = 0.1
            projection[2].y = -0.08
        } else {
            let depthScale: Float = mode == .forward ? -0.02 : 0.02
            projection = simd_float4x4(columns: (
                SIMD4(0.45, 0, 0, 0), SIMD4(0, 0.53, 0, 0), SIMD4(0, 0, depthScale, 0),
                SIMD4(0.09, -0.06, mode == .forward ? 0 : 1, 1)
            ))
        }
        return try ShaderGraphCamera(
            viewMatrix: simd_inverse(frame.matrix), projectionMatrix: projection,
            near: perspective ? gpu.camera.near : 0, far: 50
        )
    }

    private func makeSurface(
        sampling texture: any MTLTexture, mode: MeshDepthMode, offset: SIMD2<Float>? = nil
    ) throws -> SceneNode {
        let graph = try SurfaceShaderGraph { surface in
            let map = surface.texture("Refraction", addressing: .clampToEdge)
            return .init(baseColor: map.samplePlanarRefraction(distortion: offset.map { surface.constant($0) }).rgb)
        }
        let material = try PhysicallyBasedMaterial(
            surfaceGraph: graph, surfaceTextures: ["Refraction": texture], shadingModel: .unlit
        )
        let mesh = try gpu.unlit(
            MeshGeometry.plane(width: 3.2, height: 2.5), material: .physicallyBased(material), depthMode: mode
        )
        return SceneNode(
            transform: .init(translation: center, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: normal)), mesh: mesh
        )
    }

    private func marker(
        _ position: SIMD3<Float>, _ color: SIMD3<Float>, mode: MeshDepthMode,
        rotation: simd_quatf = simd_quatf(real: 1, imag: .zero)
    ) throws -> SceneNode {
        let mesh = try gpu.unlit(
            GeometryRenderFixture.square(0.17), material: .classic(ClassicMaterial(model: .basic, color: color)), depthMode: mode
        )
        return SceneNode(transform: .init(translation: position, rotation: rotation), mesh: mesh, frustumCulled: false)
    }

    private func planePoint(x: Float) -> SIMD3<Float> {
        SIMD3(x, 0, center.z - normal.x * x / normal.z)
    }

    private func worldPlane() throws -> Plane3D {
        try Plane3D(normal: frame.rotation.act(normal) * 7, through: frame.translation + frame.rotation.act(center))
    }

    private func uv(_ position: SIMD3<Float>, camera: ShaderGraphCamera) -> SIMD2<Float> {
        let clip = camera.projectionMatrix * SIMD4(position, 1)
        return SIMD2(0.5 + 0.5 * clip.x / clip.w, 0.5 - 0.5 * clip.y / clip.w)
    }

    private func pixel(_ image: [SIMD4<Float>], at uv: SIMD2<Float>) -> SIMD4<Float> {
        gpu.pixel(image, x: Int(uv.x * Float(gpu.side)), y: Int(uv.y * Float(gpu.side)))
    }
}
