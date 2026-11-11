import Frostfield
import Metal
import Testing
import simd

struct LightProbeGridTests {
    @Test func `All eight anisotropic cells and nine SH coefficients match the pinned reference`() throws {
        let bounds = try AxisAlignedBox(minimum: SIMD3(2, 4, 6), maximum: SIMD3(6, 10, 14))
        let probes = try (0..<8).map { index in
            let i = Float(index)
            var coefficients = [SIMD3(2 + i * 0.25, 1.5 + i * 0.125, 0.75 + i * 0.5)]
            for coefficient in 1..<9 {
                let f = Float(coefficient) * 0.0625
                coefficients.append(SIMD3(index.isMultiple(of: 2) ? f : -f, Float(index % 3 - 1) * f * 0.5, -f * 0.5))
            }
            return try LightProbe(coefficients: coefficients, intensity: (i + 2) * 0.25)
        }
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(2, 2, 2), probes: probes, intensity: 1.5)
        #expect(try grid.probePosition(at: .zero) == SIMD3(3, 5.5, 8))
        #expect(try grid.probePosition(at: SIMD3(1, 1, 1)) == SIMD3(5, 8.5, 12))
        #expect(try grid.probePosition(at: SIMD3(0, 0, 1)) == SIMD3(3, 5.5, 12))
        let actual = try grid.irradiance(at: SIMD3(3.5, 7.75, 9.25), normal: SIMD3(2, -1, 2))
        #expect(simd_distance(actual, SIMD3(6.51466289, 4.33204470, 6.31547694)) < 0.000002)
    }

    @Test func `Normal bias and clamping work for singleton axes and outside positions`() throws {
        let bounds = try AxisAlignedBox(minimum: SIMD3(-2, -1, -1), maximum: SIMD3(2, 1, 1))
        let red = try LightProbe.hemisphere(skyColor: SIMD3(4, 0, 0), groundColor: SIMD3(4, 0, 0))
        let blue = try LightProbe.hemisphere(skyColor: SIMD3(0, 0, 2), groundColor: SIMD3(0, 0, 2))
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(2, 1, 1), probes: [red, blue])
        #expect(try grid.probePosition(at: SIMD3(1, 0, 0)) == SIMD3(1, 0, 0))
        let towardBlue = try grid.irradiance(at: SIMD3(-1, 0, 0), normal: SIMD3(17, 0, 0))
        let towardRed = try grid.irradiance(at: SIMD3(-1, 0, 0), normal: SIMD3(-1, 0, 0))
        #expect(simd_distance(towardBlue, SIMD3(2, 0, 1)) < 0.000001)
        #expect(simd_distance(towardRed, SIMD3(4, 0, 0)) < 0.000001)
        let outside = try grid.irradiance(at: SIMD3(.greatestFiniteMagnitude, 200, -300), normal: SIMD3(0, 1, 0))
        #expect(simd_distance(outside, SIMD3(0, 0, 2)) < 0.000001)
    }

    @Test func `Signed coefficients blend before the final nonnegative clamp`() throws {
        let bounds = try AxisAlignedBox(minimum: SIMD3(-2, -1, -1), maximum: SIMD3(2, 1, 1))
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 9)
        coefficients[0] = SIMD3(repeating: -2)
        let negative = try LightProbe(coefficients: coefficients)
        coefficients[0] = SIMD3(repeating: 2)
        let positive = try LightProbe(coefficients: coefficients)
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(2, 1, 1), probes: [negative, positive])
        let normal = SIMD3<Float>(0, 1, 0)
        #expect(try grid.irradiance(at: .zero, normal: normal) == .zero)
        #expect(try grid.irradiance(at: SIMD3(-1, 0, 0), normal: normal) == .zero)
        #expect(try simd_distance(grid.irradiance(at: SIMD3(0.5, 0, 0), normal: normal),
                                  SIMD3(repeating: 0.886227)) < 0.000001)
    }

    @Test func `Grid bounds, dimensions, count, position and intensity are validated`() throws {
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 1))
        let flat = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(1, 0, 1))
        let probe = try LightProbe(coefficients: Array(repeating: .zero, count: 9))
        #expect(throws: LightProbeGrid.Failure.bounds) {
            try LightProbeGrid(bounds: flat, resolution: SIMD3(repeating: 1), probes: [probe])
        }
        for resolution in [SIMD3(0, 1, 1), SIMD3(1, -1, 1), SIMD3(Int.max, Int.max, Int.max), SIMD3(257, 257, 1)] {
            #expect(throws: LightProbeGrid.Failure.resolution) {
                try LightProbeGrid(bounds: bounds, resolution: resolution, probes: [probe])
            }
        }
        #expect(throws: LightProbeGrid.Failure.probeCount(expected: 2)) {
            try LightProbeGrid(bounds: bounds, resolution: SIMD3(2, 1, 1), probes: [probe])
        }
        #expect(throws: LightProbe.Failure.intensity) {
            try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 1), probes: [probe], intensity: .nan)
        }
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 9)
        coefficients[0].x = Float.greatestFiniteMagnitude / 32
        let large = try LightProbe(coefficients: coefficients)
        #expect(throws: LightProbe.Failure.coefficientRange) {
            try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 1), probes: [large], intensity: 4)
        }
        let grid = try LightProbeGrid(bounds: bounds, resolution: SIMD3(repeating: 1), probes: [probe])
        #expect(throws: LightProbeGrid.Failure.probeIndex) { try grid.probePosition(at: SIMD3(0, 0, 1)) }
        #expect(throws: LightProbeGrid.Failure.position) {
            try grid.irradiance(at: SIMD3(.nan, 0, 0), normal: SIMD3(0, 0, 1))
        }
        #expect(throws: LightProbe.Failure.direction) { try grid.irradiance(at: .zero, normal: .zero) }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightProbeGridCaptureTests {
    @Test func `Native cube capture produces spatially different HDR irradiance from two walls`() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let panel = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 2, height: 2),
            material: .classic(try ClassicMaterial(model: .basic))
        )
        let left = SceneNode(transform: SceneTransform(
            translation: SIMD3(-2, 0, 0), rotation: simd_quatf(from: SIMD3(0, 0, 1), to: SIMD3(1, 0, 0))
        ), mesh: panel, materialParameters: SIMD4(4, 0, 0, 1))
        let right = SceneNode(transform: SceneTransform(
            translation: SIMD3(2, 0, 0), rotation: simd_quatf(from: SIMD3(0, 0, 1), to: SIMD3(-1, 0, 0))
        ), mesh: panel, materialParameters: SIMD4(0, 0, 4, 1))
        let graph = SceneGraph(roots: [left, right], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 8))
        let bounds = try AxisAlignedBox(minimum: SIMD3(-2, -1, -1), maximum: SIMD3(2, 1, 1))
        let grid = try await LightProbeGrid.capture(graph, bounds: bounds, resolution: SIMD3(2, 1, 1),
                                                   commandQueue: queue, faceSize: 16)
        let leftPoint = try grid.probePosition(at: .zero)
        let rightPoint = try grid.probePosition(at: SIMD3(1, 0, 0))
        #expect(leftPoint == SIMD3(-1, 0, 0))
        #expect(rightPoint == SIMD3(1, 0, 0))
        let normal = SIMD3<Float>(0, 1, 0)
        let leftLight = try grid.irradiance(at: leftPoint, normal: normal)
        let rightLight = try grid.irradiance(at: rightPoint, normal: normal)
        #expect(leftLight.x > 1 && leftLight.x > 2 * leftLight.z)
        #expect(rightLight.z > 1 && rightLight.z > 2 * rightLight.x)
        #expect(abs(leftLight.x - rightLight.z) < 0.01)
        let midpoint = try grid.irradiance(at: .zero, normal: normal)
        #expect(simd_distance(midpoint, (leftLight + rightLight) * 0.5) < 0.001)
        #expect(grid.probes[0].coefficients[0].x > grid.probes[0].coefficients[0].z)
        #expect(grid.probes[1].coefficients[0].z > grid.probes[1].coefficients[0].x)
    }

    @Test func `Capture bounds both the probe count and total cubemap texels`() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let graph = SceneGraph(roots: [], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 8))
        let bounds = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(repeating: 1))
        await #expect(throws: LightProbeGrid.Failure.captureBudget) {
            try await LightProbeGrid.capture(graph, bounds: bounds, resolution: SIMD3(17, 17, 17), commandQueue: queue)
        }
        await #expect(throws: LightProbeGrid.Failure.captureBudget) {
            try await LightProbeGrid.capture(graph, bounds: bounds, resolution: SIMD3(2, 2, 1),
                                             commandQueue: queue, faceSize: 1_024)
        }
        await #expect(throws: LightProbe.Failure.faceSize) {
            try await LightProbeGrid.capture(graph, bounds: bounds, resolution: SIMD3(1, 1, 1),
                                             commandQueue: queue, faceSize: 1_025)
        }
    }
}
