import Frostfield
import Metal
import Testing
import simd

struct LightProbeTests {
    @Test("A hemisphere interpolates sky and ground irradiance around a normalized world axis")
    func hemisphereEvaluation() throws {
        let sky = SIMD3<Float>(2, 0.5, 0)
        let ground = SIMD3<Float>(0, 0.25, 0.75)
        let axis = SIMD3<Float>(1, 2, -1)
        var probe = try LightProbe.hemisphere(skyColor: sky, groundColor: ground, direction: axis, intensity: 2)
        let fixtures = [(axis, 2 * sky), (-axis, 2 * ground), (SIMD3<Float>(1, 0, 1), sky + ground)]
        for (normal, expected) in fixtures {
            #expect(try simd_distance(probe.irradiance(at: normal), expected) < 0.000002)
        }
        #expect(probe.coefficients.dropFirst(4).allSatisfy { $0 == .zero })
        try probe.setIntensity(0)
        #expect(try probe.irradiance(at: axis) == .zero)
        #expect(throws: LightProbe.Failure.hemisphereColor) {
            try LightProbe.hemisphere(skyColor: sky, groundColor: SIMD3(0, -1, 0))
        }
        #expect(throws: LightProbe.Failure.direction) {
            try LightProbe.hemisphere(skyColor: sky, groundColor: ground, direction: .zero)
        }
    }

    @Test("Probe evaluation normalizes directions and scales a cosine-convolved first band")
    func evaluation() throws {
        var coefficients = [SIMD3<Float>](repeating: .zero, count: LightProbe.coefficientCount)
        coefficients[3] = SIMD3(1, 0.5, 2)
        var probe = try LightProbe(coefficients: coefficients, intensity: 0.5)
        let x = SIMD3<Float>(1, 0, 0)
        #expect(try simd_distance(probe.radiance(in: x * Float.greatestFiniteMagnitude),
                                  coefficients[3] * 0.2443015) < 0.000001)
        #expect(try simd_distance(probe.irradiance(at: -x * Float.leastNonzeroMagnitude),
                                  coefficients[3] * -0.511664) < 0.000001)
        #expect(try probe.irradiance(at: SIMD3(0, 5, 0)) == .zero)
        try probe.setIntensity(0)
        #expect(try probe.radiance(in: x) == .zero)
        #expect(throws: LightProbe.Failure.intensity) { try probe.setIntensity(-1) }
        #expect(probe.intensity == 0)
        #expect(throws: LightProbe.Failure.direction) { try probe.irradiance(at: .zero) }
        #expect(throws: LightProbe.Failure.direction) { try probe.radiance(in: SIMD3(0, .nan, 1)) }
    }

    @Test("A probe rejects missing, nonfinite and unrepresentably scaled coefficients")
    func validation() {
        #expect(throws: LightProbe.Failure.coefficients) { try LightProbe(coefficients: [.zero]) }
        var coefficients = [SIMD3<Float>](repeating: .zero, count: 9)
        coefficients[0].x = .infinity
        #expect(throws: LightProbe.Failure.coefficients) { try LightProbe(coefficients: coefficients) }
        coefficients[0].x = Float.greatestFiniteMagnitude / 32
        #expect(throws: LightProbe.Failure.coefficientRange) { try LightProbe(coefficients: coefficients, intensity: 8) }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightProbeProjectionTests {
    @Test("A private HDR cube captured on the queue yields constant radiance and π times irradiance")
    func capturedConstant() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let target = try CubeCamera.makeTarget(device: device, size: 8)
        let capture = try CubeCamera(queue: queue, target: target, near: 0.1, far: 12)
        let graph = SceneGraph(roots: [], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 12))
        let command = try #require(queue.makeCommandBuffer())
        let expected = SIMD3<Float>(3, 0.5, 1.25)
        try capture.encode(graph, at: .zero, clearColor: MTLClearColor(red: 3, green: 0.5, blue: 1.25, alpha: 1),
                           commandBuffer: command)
        command.commit()
        let probe = try await LightProbe(cube: target, commandQueue: queue, mipLevel: 1)
        #expect(command.status == .completed)
        #expect(simd_distance(probe.coefficients[0], expected * (4 * .pi).squareRoot()) < 0.00003)
        #expect(probe.coefficients.dropFirst().allSatisfy { simd_length($0) < 0.000001 })
        for direction: SIMD3<Float> in [SIMD3(1, 0, 0), SIMD3(0, -1, 0), SIMD3(1, 2, -3)] {
            #expect(try simd_distance(probe.radiance(in: direction), expected) < 0.00003)
            #expect(try simd_distance(probe.irradiance(at: direction), expected * .pi) < 0.00005)
        }
    }

    @Test("An asymmetric HDR cube agrees with the actual r187 generator after translating face axes")
    func asymmetricReference() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let cube = try texture(device, format: .rgba32Float, size: 3)
        for face in 0..<6 {
            var values = [SIMD4<Float>](repeating: .zero, count: 9)
            if face == 0 { values[0] = SIMD4(4, 0.25, 0, .nan) }
            if face == 2 { values[4] = SIMD4(0, 2, 1, 0) }
            if face == 5 { values[5] = SIMD4(0.5, 0, 3, 0) }
            upload(values, to: cube, face: face)
        }
        let probe = try await LightProbe(cube: cube, commandQueue: queue)
        // THREE.REVISION 187, actual fromCubeRenderTarget read using the Metal fixture
        // mirrored in X on every face, with the two X faces swapped for WebGPU.
        let reference: [SIMD3<Float>] = [
            SIMD3(0.223017100, 0.255633627, 0.332678974), SIMD3(0.157863238, 0.432296785, 0.211215166),
            SIMD3(0.107246586, 0.009866452, -0.303699914), SIMD3(0.203050423, 0.014799679, -0.202466609),
            SIMD3(0.256839727, 0.016052483, 0), SIMD3(0.171226484, 0.010701655, 0),
            SIMD3(-0.019497569, -0.276539345, 0.117392604), SIMD3(0.319621870, 0.016052483, 0.376692857),
            SIMD3(0.127943934, -0.465602277, -0.110581120),
        ]
        for (actual, expected) in zip(probe.coefficients, reference) {
            #expect(simd_distance(actual, expected) < 0.000002)
        }
        let fixtures: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = [
            (SIMD3(1, 0, 0), SIMD3(0.238164876, -0.087783986, -0.102510996), SIMD3(0.465154111, 0.110431958, 0.011116197)),
            (SIMD3(0, 1, 0), SIMD3(0.076301393, 0.624899189, 0.220430342), SIMD3(0.309125902, 0.937195233, 0.529336452)),
            (SIMD3(0, 0, -1), SIMD3(-0.001787749, -0.107144404, 0.316285141), SIMD3(0.078235914, 0.079450513, 0.663772006)),
            (SIMD3(-1, -1, 1), SIMD3(-0.093868109, -0.055124822, -0.131478108), SIMD3(-0.019160658, -0.034835315, 0.002483775)),
        ]
        for (direction, radiance, irradiance) in fixtures {
            #expect(try simd_distance(probe.radiance(in: direction), radiance) < 0.000002)
            #expect(try simd_distance(probe.irradiance(at: direction), irradiance) < 0.000002)
        }
    }

    @Test("RGBA8 samples decode according to their native linear or sRGB format")
    func sdrEncoding() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let byteColor = SIMD4<UInt8>(128, 64, 32, 0)
        let expected: [(MTLPixelFormat, SIMD3<Float>)] = [
            (.rgba8Unorm, SIMD3(128, 64, 32) / 255),
            (.rgba8Unorm_srgb, SIMD3(0.2158605, 0.0512695, 0.0144438)),
        ]
        for (format, color) in expected {
            let cube = try texture(device, format: format, size: 1)
            for face in 0..<6 { upload([byteColor], to: cube, face: face) }
            let probe = try await LightProbe(cube: cube, commandQueue: queue)
            #expect(try simd_distance(probe.radiance(in: SIMD3(0, 0, 1)), color) < 0.000002)
        }
    }

    @Test("Cube projection rejects absent mips, unsupported storage formats and nonfinite RGB")
    func cubeValidation() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let cube = try texture(device, format: .rgba32Float, size: 1)
        let unsupported = try texture(device, format: .rg16Float, size: 1)
        await #expect(throws: LightProbe.Failure.mip) {
            try await LightProbe(cube: cube, commandQueue: queue, mipLevel: 1)
        }
        await #expect(throws: LightProbe.Failure.cube) {
            try await LightProbe(cube: unsupported, commandQueue: queue)
        }
        upload([SIMD4<Float>(0, .nan, 0, 1)], to: cube, face: 0)
        await #expect(throws: LightProbe.Failure.cubeColor) {
            try await LightProbe(cube: cube, commandQueue: queue)
        }
    }

    private func texture(_ device: any MTLDevice, format: MTLPixelFormat, size: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: format, size: size, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        descriptor.hazardTrackingMode = .tracked
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func upload<T>(_ values: [T], to cube: any MTLTexture, face: Int) {
        let row = cube.width * MemoryLayout<T>.stride
        values.withUnsafeBytes {
            cube.replace(region: MTLRegionMake2D(0, 0, cube.width, cube.height), mipmapLevel: 0, slice: face,
                         withBytes: $0.baseAddress!, bytesPerRow: row, bytesPerImage: row * cube.height)
        }
    }
}
