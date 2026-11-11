import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CubeCameraTests {
    @Test("Metal samples six HDR faces with correctly projected off-axis detail")
    func projectsAllFaces() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 64)
        let target = try CubeCamera.makeTarget(device: fixture.device, size: 64)
        let captureLayers = SceneLayers(layer: 2)
        let capture = try CubeCamera(queue: fixture.queue, target: target, near: 0.25, far: 6, layers: captureLayers)
        let plane = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.65, height: 0.65)
        )
        let position = SIMD3<Float>(5, -3, 2)
        let faces: [(forward: SIMD3<Float>, offAxis: SIMD3<Float>, channel: Int)] = [
            (SIMD3(1, 0, 0), SIMD3(1, 0.27, -0.44), 0),
            (SIMD3(-1, 0, 0), SIMD3(-1, 0.27, 0.44), 0),
            (SIMD3(0, 1, 0), SIMD3(0.44, 1, -0.27), 1),
            (SIMD3(0, -1, 0), SIMD3(0.44, -1, 0.27), 1),
            (SIMD3(0, 0, 1), SIMD3(0.44, 0.27, 1), 2),
            (SIMD3(0, 0, -1), SIMD3(-0.44, 0.27, -1), 2),
        ]

        func panel(
            direction: SIMD3<Float>, forward: SIMD3<Float>, distance: Float, scale: Float = 1,
            tint: SIMD4<Float>, layers: SceneLayers = SceneLayers(layer: 2)
        ) -> SceneNode {
            SceneNode(
                transform: SceneTransform(
                    translation: position + distance * direction,
                    rotation: simd_quatf(from: SIMD3(0, 0, 1), to: -forward),
                    scale: SIMD3(repeating: scale)
                ),
                mesh: plane, materialParameters: tint, layers: layers
            )
        }

        var panels: [SceneNode] = []
        var directions: [SIMD3<Float>] = []
        var expected: [SIMD4<Float>] = []
        for (index, face) in faces.enumerated() {
            var centerTint = SIMD4<Float>(0, 0, 0, 1)
            centerTint[face.channel] = Float(index + 2)
            var markerTint = SIMD4<Float>(0, 0, 0, 1)
            markerTint[face.channel] = Float(index + 8)
            panels.append(panel(direction: face.forward, forward: face.forward, distance: 3, tint: centerTint))
            panels.append(panel(direction: face.offAxis, forward: face.forward, distance: 3, tint: markerTint))
            directions.append(contentsOf: [face.forward, face.offAxis, 2 * face.forward - face.offAxis])
            expected.append(contentsOf: [centerTint, markerTint, SIMD4(0, 0, 0, 1)])
        }

        let bright = SIMD4<Float>(40, 40, 40, 1)
        let x = faces[0].forward
        let z = faces[4].forward
        let excluded = SceneNode(children: [panel(direction: x, forward: x, distance: 2, tint: bright)])
        let hidden = SceneNode(
            children: [panel(direction: z, forward: z, distance: 2, tint: bright)], isVisible: false
        )
        panels.append(panel(direction: z, forward: z, distance: 2, tint: bright, layers: .default))
        panels.append(panel(direction: x, forward: x, distance: 0.1, scale: 0.1, tint: bright))
        panels.append(panel(direction: SIMD3(0, -1, -0.6), forward: faces[3].forward, distance: 8, tint: bright))
        directions.append(SIMD3(0, -1, -0.6))
        expected.append(SIMD4(0, 0, 0, 1))
        // A parent's layer does not filter its children.
        var graph = SceneGraph(
            roots: [SceneNode(children: panels), excluded, hidden],
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 2, far: 3),
            cameraTransform: SceneTransform(translation: SIMD3(20, 30, 40))
        )
        graph.projectionJitter = SIMD2(0.9, -0.8)
        try fixture.run { command in
            try capture.encode(graph, at: position, excluding: [excluded.id], commandBuffer: command)
        }

        let actual = try fixture.sampleCube(target, directions: directions)
        for (index, color) in actual.enumerated() {
            #expect(simd_length(color - expected[index]) < 0.04, "Cube direction \(directions[index])")
        }
        let firstMip = try fixture.sampleCube(target, directions: [faces[0].forward, faces[0].offAxis], mip: 1)
        #expect(firstMip[0].x > 1.8)
        #expect(firstMip[1].x > 7.8)
        #expect(graph.node(withID: excluded.id)?.isVisible == true)
        #expect(graph.cameraLayers == .default)
    }

    @Test("A captured cube feeds an independent static environment probe and can be refreshed")
    func preparesEnvironment() throws {
        let fixture = try RasterLightingFixture(width: 16, height: 16)
        let target = try CubeCamera.makeTarget(device: fixture.device, size: 16, mipmapped: false)
        let capture = try CubeCamera(queue: fixture.queue, target: target, near: 0.1, far: 5)
        var graph = SceneGraph(
            roots: [], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 5)
        )
        let initial = SIMD4<Float>(4, 2, 1, 1)
        let command = try #require(fixture.queue.makeCommandBuffer())
        try capture.encode(
            graph, at: .zero, clearColor: MTLClearColor(red: 4, green: 2, blue: 1, alpha: 1),
            commandBuffer: command
        )
        command.commit()

        let lighting = try EnvironmentLighting(
            queue: fixture.queue, cube: target, settings: .init(cubeSize: 8, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        )
        #expect(command.status == .completed)
        #expect(lighting.skyCube !== target)
        #expect(lighting.skyCube.width == 8)
        let direction = SIMD3<Float>(0, -1, 0.24)
        let diffuse = try fixture.sampleCube(lighting.diffuseCube, directions: [direction])
        let specular = try fixture.sampleCube(lighting.specularCube, directions: [direction], mip: 2)
        #expect(simd_length(diffuse[0] - initial) < 0.04)
        #expect(simd_length(specular[0] - initial) < 0.04)

        let background = try EnvironmentBackground(lighting: lighting)
        graph.environmentIntensity = 0.5
        try fixture.run { command in
            try capture.encode(graph, at: .zero, background: background, commandBuffer: command)
        }
        let refreshed = try fixture.sampleCube(target, directions: [direction])
        let frozen = try fixture.sampleCube(lighting.skyCube, directions: [direction])
        #expect(refreshed[0] == SIMD4(2, 1, 0.5, 1))
        #expect(frozen[0] == initial)
    }

    @Test("Capture validates texture usage, clip distances, position and queue before encoding")
    func validatesInputs() throws {
        let fixture = try RasterLightingFixture(width: 8, height: 8)
        let color = try fixture.colorTarget()
        #expect(throws: CubeCamera.Failure.target) {
            try CubeCamera(queue: fixture.queue, target: color, near: 0.1, far: 10)
        }
        #expect(throws: EnvironmentLighting.Failure.inputCube) {
            try EnvironmentLighting(queue: fixture.queue, cube: color)
        }
        let target = try CubeCamera.makeTarget(device: fixture.device, size: 8)
        #expect(throws: PerspectiveCamera.ValidationError.invalidClippingPlanes) {
            try CubeCamera(queue: fixture.queue, target: target, near: 10, far: 0.1)
        }
        let capture = try CubeCamera(queue: fixture.queue, target: target, near: 0.1, far: 10)
        let graph = SceneGraph(
            roots: [], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        )
        let otherQueue = try #require(fixture.device.makeCommandQueue())
        let otherCommand = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: CubeCamera.Failure.commandBuffer) {
            try capture.encode(graph, at: .zero, commandBuffer: otherCommand)
        }
        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: CubeCamera.Failure.position) {
            try capture.encode(graph, at: SIMD3(.infinity, 0, 0), commandBuffer: command)
        }
    }
}

extension RasterLightingFixture {
    func sampleCube(
        _ cube: any MTLTexture, directions: [SIMD3<Float>], mip: UInt32 = 0
    ) throws -> [SIMD4<Float>] {
        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void sample_cube(
                texturecube<float> input [[texture(0)]],
                constant float3* directions [[buffer(0)]],
                device float4* samples [[buffer(1)]],
                constant uint& mip [[buffer(2)]],
                uint i [[thread_position_in_grid]]
            ) {
                constexpr sampler cubeSampler(filter::nearest, mip_filter::nearest);
                samples[i] = float4(input.sample(cubeSampler, directions[i], level(float(mip))));
            }
            """, options: nil)
        let function = try #require(library.makeFunction(name: "sample_cube"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let result = try #require(device.makeBuffer(
            length: directions.count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared
        ))
        try run { command in
            let encoder = try #require(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(cube, index: 0)
            try directions.withUnsafeBytes { bytes in
                encoder.setBytes(try #require(bytes.baseAddress), length: bytes.count, index: 0)
            }
            encoder.setBuffer(result, offset: 0, index: 1)
            var mip = mip
            encoder.setBytes(&mip, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreadgroups(
                MTLSize(width: directions.count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }
        return Array(UnsafeBufferPointer(
            start: result.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: directions.count
        ))
    }
}
