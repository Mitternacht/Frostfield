import Foundation
import Frostfield
import Metal
import Testing
import simd

struct GLTFCameraTests {
    @Test
    func importsPerspectiveAndOrthographicProjectionsInBothDepthModes() throws {
        let data = try document()
        let plain = try GLTFMeshLoader.loadScene(data: data)
        #expect(plain.cameras.isEmpty)
        #expect(plain.nodes.allSatisfy { $0.cameraIndex == nil })

        let scene = try GLTFMeshLoader.loadScene(data: data, includeCameras: true)
        #expect(scene.nodes.map(\.cameraIndex) == [nil, 0, 1, 2, 0])
        #expect(scene.cameras.map(\.name) == ["Still", "Infinite", "Ortho"])
        let finite = scene.cameras[0]
        let infinite = scene.cameras[1]
        let ortho = scene.cameras[2]
        let authored = try finite.metalProjection(viewportAspectRatio: 1)
        #expect(abs(authored.columns.1.y / authored.columns.0.x - 1.6) < 0.0001)
        let automatic = try infinite.metalProjection(viewportAspectRatio: 1.25)
        #expect(abs(automatic.columns.1.y / automatic.columns.0.x - 1.25) < 0.0001)

        for mode in [MeshDepthMode.forward, .reversed] {
            let near: Float = mode == .forward ? 0 : 1
            let far = 1 - near
            let finiteProjection = try finite.metalProjection(viewportAspectRatio: 1, depthMode: mode)
            let infiniteProjection = try infinite.metalProjection(viewportAspectRatio: 1, depthMode: mode)
            let orthoProjection = try ortho.metalProjection(viewportAspectRatio: 1, depthMode: mode)
            #expect(abs(project(SIMD3(0, 0, -0.2), with: finiteProjection).z - near) < 0.0001)
            #expect(abs(project(SIMD3(0, 0, -20), with: finiteProjection).z - far) < 0.0001)
            #expect(abs(project(SIMD3(0, 0, -0.5), with: infiniteProjection).z - near) < 0.0001)
            #expect(abs(project(SIMD3(0, 0, -1_000_000), with: infiniteProjection).z - far) < 0.0001)
            #expect(simd_distance(project(SIMD3(2, 1, 0), with: orthoProjection), SIMD3(1, 1, near)) < 0.0001)
            #expect(abs(project(SIMD3(0, 0, -40), with: orthoProjection).z - far) < 0.0001)
        }
    }

    @Test
    func preparedCameraFollowsHierarchyWhileIgnoringScaleInItsViewBasis() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let source = try GLTFMeshLoader.loadScene(data: document(), includeCameras: true)
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: source, depthMode: .reversed)
        let appCamera = try PerspectiveCamera(verticalFieldOfView: 0.7, near: 0.1, far: 10)
        var graph = try prepared.graph(camera: appCamera)
        let ortho = try prepared.cameraView(nodeIndex: 3, in: graph, viewportAspectRatio: 1)
        #expect(ortho.depthMode == .reversed)
        #expect(abs(ortho.aspectRatio - 2) < 0.0001)
        #expect(simd_distance(ortho.cameraPosition, SIMD3(4, 2, 3)) < 0.0001)
        #expect(simd_distance(try #require(ortho.directionToCamera), SIMD3(1, 0, 0)) < 0.0001)
        #expect(simd_distance(project(SIMD3(4, 3, 1), with: ortho.clipFromWorld), SIMD3(1, 1, 1)) < 0.0001)

        let first = try prepared.cameraView(nodeIndex: 1, in: graph, viewportAspectRatio: 1)
        let reused = try prepared.cameraView(nodeIndex: 4, in: graph, viewportAspectRatio: 1)
        #expect(first.directionToCamera == nil)
        #expect(first.cameraPosition != reused.cameraPosition)
        graph.roots[0].transform.translation += SIMD3(0, 3, 0)
        let moved = try prepared.cameraView(nodeIndex: 3, in: graph, viewportAspectRatio: 1)
        #expect(simd_distance(moved.cameraPosition, ortho.cameraPosition + SIMD3(0, 3, 0)) < 0.0001)
        #expect(throws: PreparedGLTFScene.Failure.missingCameraNode(0)) {
            try prepared.cameraView(nodeIndex: 0, in: graph, viewportAspectRatio: 1)
        }
    }

    @Test
    func rejectsMissingCameraReferencesAndInvalidParameters() throws {
        #expect(throws: GLTFMeshLoader.Failure.missingCamera(7)) {
            try GLTFMeshLoader.loadScene(data: document(cameraIndex: 7), includeCameras: true)
        }
        #expect(throws: GLTFMeshLoader.Failure.invalidCamera(0, "invalid perspective field of view, aspect, or clipping planes")) {
            try GLTFMeshLoader.loadScene(data: document(far: 0.1), includeCameras: true)
        }
    }

    @Test("An imported orthographic camera keeps the reflection direction constant across a flat face")
    func orthographicShading() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let source = try GLTFMeshLoader.loadScene(data: document(), includeCameras: true)
        let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: source)
        let perspective = try PerspectiveCamera(verticalFieldOfView: 0.7, near: 0.1, far: 10)
        var graph = try prepared.graph(camera: perspective)
        let view = try prepared.cameraView(nodeIndex: 3, in: graph, viewportAspectRatio: 2)
        let direction = try #require(view.directionToCamera)
        let pbr = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0.23, directionToLight: direction, lightIntensity: 0.2, ambientIntensity: 0
        )
        let cube = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: .cube(), material: .physicallyBased(pbr))
        graph.roots.append(SceneNode(transform: .init(translation: view.cameraPosition - 3 * direction), mesh: cube))

        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 64, height: 32, mipmapped: false)
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 64, height: 32, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(
            pass: pass, commandBuffer: buffer, clipFromWorld: view.clipFromWorld, cameraPosition: view.cameraPosition,
            directionToCamera: view.directionToCamera, depthMode: view.depthMode
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)

        func pixel(_ x: Int) -> SIMD4<Float16> {
            var value = SIMD4<Float16>.zero
            color.getBytes(&value, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride, from: MTLRegionMake2D(x, 16, 1, 1), mipmapLevel: 0)
            return value
        }
        let center = pixel(32)
        #expect(center.x > 0.2)
        #expect(center == pixel(38))
    }

    private func document(cameraIndex: Int = 0, far: Double = 20) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "asset": ["version": "2.0"], "scenes": [["nodes": [0, 4]]],
            "nodes": [
                ["children": [1, 2, 3], "translation": [1, 2, 3], "scale": [2, 2, 3],
                 "rotation": [0, 0.70710678, 0, 0.70710678]],
                ["camera": cameraIndex], ["camera": 1], ["camera": 2, "translation": [0, 0, 1]],
                ["camera": 0],
            ],
            "cameras": [
                ["name": "Still", "type": "perspective",
                 "perspective": ["yfov": 0.7, "aspectRatio": 1.6, "znear": 0.2, "zfar": far]],
                ["name": "Infinite", "type": "perspective", "perspective": ["yfov": 0.6, "znear": 0.5]],
                ["name": "Ortho", "type": "orthographic", "orthographic": ["xmag": 2, "ymag": 1, "znear": 0, "zfar": 40]],
            ],
        ])
    }

    private func project(_ position: SIMD3<Float>, with matrix: simd_float4x4) -> SIMD3<Float> {
        let clip = matrix * SIMD4(position, 1)
        return SIMD3(clip.x, clip.y, clip.z) / clip.w
    }
}
