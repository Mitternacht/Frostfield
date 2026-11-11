import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessDepthTests {
    enum Projection: CaseIterable, Sendable {
        case perspective, orthographic
    }

    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Post-process unprojection and linear depth match cropped cameras in both depth modes",
          arguments: Projection.allCases, [false, true])
    func positionAndLinearDepth(projection: Projection, reversed: Bool) throws {
        let mode: MeshDepthMode = reversed ? .reversed : .forward
        let camera = try camera(projection, mode: mode)
        let width = 14, height = 9
        var stored: [Float] = []
        var expected: [SIMD4<Float>] = []
        let normal = simd_normalize(SIMD3<Float>(0.24, -0.15, 1))
        for row in 0..<height {
            for column in 0..<width {
                let uv = pixelUV(column, row, width: width, height: height)
                let point = pointOnPlane(uv, camera: camera, mode: mode, normal: normal, distance: 3)
                stored.append(projectedDepth(point, camera: camera))
                expected.append(SIMD4(
                    point.x / 32 + 0.5, point.y / 32 + 0.5, -point.z / 16,
                    (-point.z - camera.near) / (camera.far - camera.near)
                ))
            }
        }
        let graph = try PostProcessGraph { p in
            let depth = p.sceneDepth(at: p.screenUV)
            let inverse = p.inverseProjection(of: camera)
            let position = p.getViewPosition(at: p.screenUV, depth: depth, inverseProjection: inverse)
            let linear = p.linearDepth(
                at: p.screenUV, depth: depth, inverseProjection: inverse,
                near: p.uniform("Near", camera.near), far: p.uniform("Far", camera.far)
            )
            return p.vector(position.xy / 32 + 0.5, -position.z / 16, linear)
        }
        let restored = try JSONDecoder().decode(PostProcessGraph.self, from: JSONEncoder().encode(graph))
        #expect(restored == graph && restored.readsDepth)
        let result = try render(restored, width: width, height: height) { stored[$1 * width + $0] }
        for index in result.indices {
            #expect(PostProcessFixture.close(result[index], expected[index], tolerance: 0.001), "\(index): \(result[index])")
        }

        let endPlanes = try render(graph, width: 2, height: 1) { x, _ in
            if x == 0 { return reversed ? 1 : 0 }
            return reversed ? 0 : 1
        }
        #expect(endPlanes[0].w < 0.001 && endPlanes[1].w > 0.999)
    }

    @Test("Depth normals preserve both sides of a discontinuity and use texture edges safely",
          arguments: Projection.allCases, [false, true])
    func discontinuity(projection: Projection, reversed: Bool) throws {
        let mode: MeshDepthMode = reversed ? .reversed : .forward
        let camera = try camera(projection, mode: mode)
        let width = 14, height = 9
        let left = simd_normalize(SIMD3<Float>(0.24, -0.15, 1))
        let right = simd_normalize(SIMD3<Float>(-0.26, 0.12, 1))
        let clear: Float = reversed ? 0 : 1
        let graph = try PostProcessGraph { p in
            // The stencil should reconstruct actual texel centers even at subpixel queries.
            let uv = p.screenUV + p.texelSize * 0.25
            let normal = p.getNormalFromDepth(at: uv, inverseProjection: p.inverseProjection(of: camera), depthMode: mode)
            return p.vector(normal * 0.5 + 0.5, p.constant(Float(1)))
        }
        #expect(graph.readsDepth)
        let result = try render(graph, width: width, height: height) { x, y in
            let isolated = x == width - 1 && y == height - 1
            if x >= width - 3 && y >= height - 3 && !isolated { return clear }
            let uv = pixelUV(x, y, width: width, height: height)
            let normal = x < width / 2 ? left : right
            let point = pointOnPlane(uv, camera: camera, mode: mode, normal: normal, distance: x < width / 2 ? 3 : 6)
            return projectedDepth(point, camera: camera)
        }
        for (x, y) in [(0, 0), (6, 4), (7, 4), (13, 0), (0, 8), (10, 8)] {
            let expected = (x < width / 2 ? left : right) * 0.5 + 0.5
            let sample = result[y * width + x]
            #expect(PostProcessFixture.close(sample, SIMD4(expected, 1), tolerance: 0.0015), "\(x), \(y): \(sample)")
        }
        #expect(result[7 * width + 12] == SIMD4(0.5, 0.5, 0.5, 1))
        #expect(result[8 * width + 13] == SIMD4(0.5, 0.5, 0.5, 1))
    }

    @Test("Depth normals choose the smaller second-neighbor extrapolation error")
    func secondNeighbor() throws {
        let source = GLTFScene.Camera(sourceIndex: 0, name: nil, projection: .orthographic(
            halfWidth: 2.5, halfHeight: 1.5, near: 0.1, far: 10.1
        ))
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: source.metalProjection(viewportAspectRatio: 5 / 3),
            near: 0.1, far: 10.1
        )
        let graph = try PostProcessGraph { p in
            let normal = p.getNormalFromDepth(at: p.screenUV, inverseProjection: p.inverseProjection(of: camera))
            return p.vector(normal * 0.5 + 0.5, p.constant(Float(1)))
        }
        // The immediate right neighbor is nearer in depth, but its two-tap
        // extrapolation misses the center. The left plane predicts it exactly.
        let depths: [Float] = [0.3, 0.4, 0.5, 0.49, 0.49]
        let pixels = try render(graph, width: 5, height: 3) { x, _ in depths[x] }
        let expected = simd_normalize(SIMD3<Float>(1, 0, 1)) * 0.5 + 0.5
        #expect(PostProcessFixture.close(pixels[7], SIMD4(expected, 1), tolerance: 0.001))
    }

    @Test("Cleared infinite depth has no finite view position and an implicit stencil requires depth",
          arguments: [false, true])
    func infiniteAndMissingDepth(reversed: Bool) throws {
        let mode: MeshDepthMode = reversed ? .reversed : .forward
        let source = GLTFScene.Camera(sourceIndex: 0, name: nil, projection: .perspective(
            verticalFieldOfView: .pi / 3, aspectRatio: nil, near: 0.1, far: nil
        ))
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: source.metalProjection(viewportAspectRatio: 2, depthMode: mode),
            near: 0.1, far: .infinity
        )
        let graph = try PostProcessGraph { p in
            let position = p.getViewPosition(
                at: p.screenUV, depth: p.sceneDepth(at: p.screenUV), inverseProjection: p.inverseProjection(of: camera)
            )
            return p.vector(-position.z, p.constant(Float(0)), p.constant(Float(0)), p.constant(Float(1)))
        }
        let covered = projectedDepth(SIMD3(0, 0, -2), camera: camera)
        let pixels = try render(graph, width: 2, height: 1) { x, _ in
            x == 0 ? (reversed ? 0 : 1) : covered
        }
        #expect(pixels[0] == SIMD4(0, 0, 0, 1))
        #expect(PostProcessFixture.close(pixels[1], SIMD4(2, 0, 0, 1)))

        let implicit = try PostProcessGraph { p in
            p.getNormalFromDepth(at: p.screenUV, inverseProjection: p.inverseProjection(of: camera), depthMode: mode)
        }
        let pass = try PostProcessPass(graph: implicit, device: gpu.device)
        let color = try gpu.color(width: 2, height: 1)
        let output = try gpu.color(width: 2, height: 1)
        #expect(throws: PostProcessPass.Failure.missingDepth) {
            try gpu.run { try pass.encode(source: color, destination: output, commandBuffer: $0) }
        }
    }

    private func render(
        _ graph: PostProcessGraph, width: Int, height: Int, depth: (Int, Int) -> Float
    ) throws -> [SIMD4<Float>] {
        let pass = try PostProcessPass(graph: graph, device: gpu.device)
        let source = try gpu.color(width: width, height: height) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try gpu.color(width: width, height: height)
        let depth = try gpu.depth(width: width, height: height, value: depth)
        try gpu.run { try pass.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
        return try gpu.read(destination)
    }

    private func camera(_ kind: Projection, mode: MeshDepthMode) throws -> ShaderGraphCamera {
        let near: Float = 0.2, far: Float = 20
        let source: GLTFScene.Camera.Projection
        switch kind {
        case .perspective: source = .perspective(verticalFieldOfView: .pi / 3, aspectRatio: nil, near: near, far: far)
        case .orthographic: source = .orthographic(halfWidth: 4, halfHeight: 2.5, near: near, far: far)
        }
        let camera = GLTFScene.Camera(sourceIndex: 0, name: nil, projection: source)
        let crop = try CameraViewOffset(fullWidth: 20, fullHeight: 15, x: 2, y: 4, width: 14, height: 9)
        let projection = try camera.metalProjection(viewportAspectRatio: crop.fullAspectRatio, depthMode: mode)
        return try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: crop.applying(to: projection), near: near, far: far
        )
    }

    private func pixelUV(_ x: Int, _ y: Int, width: Int, height: Int) -> SIMD2<Float> {
        SIMD2((Float(x) + 0.5) / Float(width), (Float(y) + 0.5) / Float(height))
    }

    private func pointOnPlane(
        _ uv: SIMD2<Float>, camera: ShaderGraphCamera, mode: MeshDepthMode, normal: SIMD3<Float>, distance: Float
    ) -> SIMD3<Float> {
        let inverse = simd_inverse(camera.projectionMatrix)
        func point(at depth: Float) -> SIMD3<Float> {
            let value = inverse * SIMD4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1)
            return SIMD3(value.x, value.y, value.z) / value.w
        }
        let origin = point(at: mode == .reversed ? 1 : 0)
        let direction = point(at: 0.5) - origin
        let reference = SIMD3<Float>(0, 0, -distance)
        return origin + direction * (simd_dot(normal, reference - origin) / simd_dot(normal, direction))
    }

    private func projectedDepth(_ position: SIMD3<Float>, camera: ShaderGraphCamera) -> Float {
        let clip = camera.projectionMatrix * SIMD4(position, 1)
        return clip.z / clip.w
    }
}
