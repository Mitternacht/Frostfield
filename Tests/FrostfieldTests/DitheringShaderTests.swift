import Foundation
import Frostfield
import Metal
import Testing

/// Compares the Metal functions of `FrostfieldDithering.h` with independent CPU references
/// of the Three.js r187 dithering chunks. The hash `fract( sin( sn ) * 43758.5453 )` turns
/// a one-ulp error in float32 into an unrelated result, so no float32 GPU can agree with a
/// Double evaluation of the whole formula at large coordinates. The suite therefore checks
/// each stage against Double arithmetic with the bound that stage can honestly meet, checks
/// the whole formula against Double only where float32 can resolve it, and checks the
/// whole grid against a float32 mirror that repeats the shader's operations.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DitheringShaderTests {
    private typealias Reference = R187DitheringReference

    // MARK: - GPU probes

    private static var source: String {
        get throws {
            """
            #include <metal_stdlib>
            using namespace metal;
            \(try DitheringShader.metalSource())

            kernel void probeRand(
                device const float2* coordinates [[buffer(0)]], device float4* results [[buffer(1)]],
                uint index [[thread_position_in_grid]]
            ) {
                float2 uv = coordinates[index];
                float dt = frostfieldR187RandDot(uv);
                float sn = frostfieldR187RandMod(dt);
                results[index] = float4(dt, sn, frostfieldR187RandFract(sn), frostfieldR187Rand(uv));
            }

            kernel void probeShift(
                device const float* positions [[buffer(0)]], device float4* results [[buffer(1)]],
                uint index [[thread_position_in_grid]]
            ) {
                results[index] = float4(frostfieldR187DitherShift(positions[index]), 0.0f);
            }

            kernel void probeDithering(
                device const float4* colors [[buffer(0)]], device const float2* coordinates [[buffer(1)]],
                device float4* results [[buffer(2)]], uint index [[thread_position_in_grid]]
            ) {
                results[index] = float4(frostfieldR187Dithering(colors[index].xyz, coordinates[index]), 0.0f);
            }

            kernel void probeSin(
                device const float* angles [[buffer(0)]], device float2* results [[buffer(1)]],
                uint index [[thread_position_in_grid]]
            ) {
                results[index] = float2(precise::sin(angles[index]), fast::sin(angles[index]));
            }

            kernel void probeTransfer(
                device const float4* values [[buffer(0)]], device float4* results [[buffer(1)]],
                uint index [[thread_position_in_grid]]
            ) {
                results[index * 2] = float4(frostfieldSRGBEncode(values[index].xyz), 0.0f);
                results[index * 2 + 1] = float4(frostfieldSRGBDecode(values[index].xyz), 0.0f);
            }
            """
        }
    }

    private struct Probe {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let library: any MTLLibrary

        init(mathMode: MTLMathMode) throws {
            device = try #require(MTLCreateSystemDefaultDevice())
            queue = try #require(device.makeCommandQueue())
            let options = MTLCompileOptions()
            options.mathMode = mathMode
            library = try device.makeLibrary(source: DitheringShaderTests.source, options: options)
        }

        /// The iOS Simulator's GPU cannot dispatch non-uniform threadgroups, so every dispatch
        /// is a whole number of threadgroups of this many threads and every buffer is padded
        /// with zeros to match. The padding threads compute results that are never read.
        private static let threadgroupSize = 64

        private static func padded(_ count: Int) -> Int {
            (count + threadgroupSize - 1) / threadgroupSize * threadgroupSize
        }

        func buffer<Element>(_ values: [Element]) throws -> any MTLBuffer {
            let length = Self.padded(values.count) * MemoryLayout<Element>.stride
            let buffer = try #require(device.makeBuffer(length: length, options: .storageModeShared))
            memset(buffer.contents(), 0, length)
            values.withUnsafeBytes { bytes in
                if let base = bytes.baseAddress { memcpy(buffer.contents(), base, bytes.count) }
            }
            return buffer
        }

        /// Runs one thread per input element with `inputs` bound from buffer 0 and the results
        /// in the next slot, `perInput` result elements per thread.
        func run<Output>(
            _ name: String, count: Int, inputs: [any MTLBuffer], as output: Output.Type, perInput: Int = 1
        ) throws -> [Output] {
            let pipeline = try device.makeComputePipelineState(function: try #require(library.makeFunction(name: name)))
            try #require(pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupSize)
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            for (slot, buffer) in inputs.enumerated() { encoder.setBuffer(buffer, offset: 0, index: slot) }
            let paddedCount = Self.padded(count)
            let results = try #require(device.makeBuffer(
                length: paddedCount * perInput * MemoryLayout<Output>.stride, options: .storageModeShared
            ))
            encoder.setBuffer(results, offset: 0, index: inputs.count)
            encoder.dispatchThreadgroups(
                MTLSize(width: paddedCount / Self.threadgroupSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Self.threadgroupSize, height: 1, depth: 1)
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed)
            let pointer = results.contents().bindMemory(to: Output.self, capacity: count * perInput)
            return Array(UnsafeBufferPointer(start: pointer, count: count * perInput))
        }

        func run<Element, Output>(
            _ name: String, _ values: [Element], as output: Output.Type, perInput: Int = 1
        ) throws -> [Output] {
            try run(name, count: values.count, inputs: [try buffer(values)], as: output, perInput: perInput)
        }
    }

    /// Pixel centers of every `stride`th pixel of a square image, in the order x fastest.
    private static func pixelCenters(side: Int, stride: Int) -> [SIMD2<Float>] {
        var centers: [SIMD2<Float>] = []
        for y in Swift.stride(from: 0, to: side, by: stride) {
            for x in Swift.stride(from: 0, to: side, by: stride) {
                centers.append(SIMD2(Float(x) + 0.5, Float(y) + 0.5))
            }
        }
        return centers
    }

    // MARK: - sin

    @Test func `precise sin on [0, PI] stays within a measured bound and the multiplier amplifies it`() throws {
        let probe = try Probe(mathMode: .fast)
        let count = 1 << 20
        let angles = (0..<count).map { Float($0) / Float(count) * Float(Reference.pi) }
        let results = try probe.run("probeSin", angles, as: SIMD2<Float>.self)
        var preciseMaximum = 0.0
        var fastMaximum = 0.0
        var preciseUlps = 0.0
        for (angle, result) in zip(angles, results) {
            let reference = sin(Double(angle))
            preciseMaximum = max(preciseMaximum, abs(Double(result.x) - reference))
            fastMaximum = max(fastMaximum, abs(Double(result.y) - reference))
            let ulp = Double(Float(abs(reference)).ulp)
            if ulp > 0 { preciseUlps = max(preciseUlps, abs(Double(result.x) - reference) / ulp) }
        }
        print("DITHER-MEASURE sin precise max abs error \(preciseMaximum) max ulps \(preciseUlps); fast max abs error \(fastMaximum); multiplier effect on grid position \(preciseMaximum * Reference.c)")
        #expect(preciseMaximum < 2e-7)
    }

    // MARK: - shift

    @Test func `shift equals mix of plus and minus 0.5 over 255 with sign pattern plus minus plus`() throws {
        let probe = try Probe(mathMode: .fast)
        var positions: [Float] = [0, 0.25, 0.5, 0.75, 1]
        var state: UInt64 = 0x9e37_79b9_7f4a_7c15
        for _ in 0..<1000 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            positions.append(Float(state >> 40) / Float(1 << 24))
        }
        let results = try probe.run("probeShift", positions, as: SIMD4<Float>.self)
        var maximum = 0.0
        for (position, result) in zip(positions, results) {
            let expected = Reference.shift(gridPosition: Double(position))
            for channel in 0..<3 { maximum = max(maximum, abs(Double(result[channel]) - expected[channel])) }
        }
        print("DITHER-MEASURE shift max abs error vs Double \(maximum) (one 8-bit step is \(1.0 / 255))")
        #expect(maximum < 1e-9)
        let half = 0.5 / 255
        #expect(abs(Double(results[0].x) - half) < 1e-9 && abs(Double(results[0].y) + half) < 1e-9 && abs(Double(results[0].z) - half) < 1e-9)
        #expect(abs(Double(results[4].x) + half) < 1e-9 && abs(Double(results[4].y) - half) < 1e-9 && abs(Double(results[4].z) + half) < 1e-9)
        #expect(abs(Double(results[2].x)) < 1e-9 && abs(Double(results[2].y)) < 1e-9 && abs(Double(results[2].z)) < 1e-9)
        #expect(results.allSatisfy { abs(Double($0.x) + Double($0.y)) < 1e-9 && $0.x == $0.z })
    }

    @Test func `dithering adds the shift of the rand grid position to every channel`() throws {
        let probe = try Probe(mathMode: .fast)
        let coordinates = Self.pixelCenters(side: 64, stride: 3)
        let colors = coordinates.indices.map { SIMD4<Float>(Float($0 % 7) / 7, 0.5, Float($0 % 5) / 5, 0) }
        let results = try probe.run("probeDithering", count: coordinates.count, inputs: [probe.buffer(colors), probe.buffer(coordinates)], as: SIMD4<Float>.self)
        let gridPositions = try probe.run("probeRand", coordinates, as: SIMD4<Float>.self)
        var maximum = 0.0
        for index in coordinates.indices {
            let color = SIMD3<Double>(Double(colors[index].x), Double(colors[index].y), Double(colors[index].z))
            let expected = color + Reference.shift(gridPosition: Double(gridPositions[index].w))
            for channel in 0..<3 { maximum = max(maximum, abs(Double(results[index][channel]) - expected[channel])) }
        }
        print("DITHER-MEASURE dithering color + shift max abs error \(maximum)")
        #expect(maximum < 1e-7)
    }

    // MARK: - rand

    @Test func `each stage of rand matches Double arithmetic within its stated float32 bound`() throws {
        let probe = try Probe(mathMode: .fast)
        let coordinates = Self.pixelCenters(side: 2048, stride: 7)
        let results = try probe.run("probeRand", coordinates, as: SIMD4<Float>.self)
        var dtWorst = 0.0
        var snWorst = 0.0
        var gridWorst = 0.0
        var rangeViolations = 0
        var composedMismatches = 0
        for (coordinate, result) in zip(coordinates, results) {
            let (x, y) = (Double(coordinate.x), Double(coordinate.y))
            let (dt, sn, grid, full) = (result.x, result.y, result.z, result.w)
            if grid != full { composedMismatches += 1 }

            // dt: a rounded product plus a rounded fused sum is within one ulp of the exact value.
            let dtReference = Reference.a * x + Reference.b * y
            dtWorst = max(dtWorst, abs(Double(dt) - dtReference) / Double(dt.ulp))

            // sn: the remainder is one fused operation, so it is within half an ulp of the exact
            // remainder of the dt the shader computed. A quotient that rounds to an integer may
            // leave it up to an ulp of dt outside 0...PI.
            let k = ((Double(dt) - Double(sn)) / Reference.pi).rounded()
            let snReference = Double(dt) - k * Reference.pi
            snWorst = max(snWorst, abs(Double(sn) - snReference) / Double(max(sn.ulp, Float.leastNormalMagnitude)))
            if Double(sn) < -2 * Double(dt.ulp) || Double(sn) > Reference.pi + 2 * Double(dt.ulp) { rangeViolations += 1 }

            // grid position: Double fract( sin( sn ) * c ) for the shader's sn.
            gridWorst = max(gridWorst, Reference.circularDistance(Double(grid), Reference.fract(sin(Double(sn)) * Reference.c)))
        }
        print("DITHER-MEASURE rand stages over \(coordinates.count) pixel centers: dt max error \(dtWorst) ulps; sn max error \(snWorst) ulps; sn range violations \(rangeViolations); grid position max circular error given the shader's sn \(gridWorst)")
        #expect(composedMismatches == 0, "the composed function must equal its stages bit for bit")
        #expect(dtWorst <= 1.0)
        #expect(snWorst <= 0.5)
        #expect(rangeViolations == 0)
        // c times the measured sin error (1.4e-7) plus half an ulp of the product (0.002).
        #expect(gridWorst < 0.008)
    }

    /// Error bound of the float32 evaluation of one coordinate against Double arithmetic with the
    /// same constants: a rounding of dt (at most one ulp) moves sin by as much, sin's own
    /// measured error of at most 1.4e-7 and half an ulp of the product 0.002 add a fixed amount,
    /// and the multiplier c turns all of it into an error of the grid position.
    private static func resolvableBound(x: Double, y: Double) -> Double {
        let dt = Float(Reference.a * x + Reference.b * y)
        return Reference.c * (Double(dt.ulp) + 1.4e-7) + 0.002
    }

    @Test func `the shift at a dense grid of coordinates matches Double within the bound float32 can meet`() throws {
        let probe = try Probe(mathMode: .fast)
        // Float32 resolves the hash only for small dt: one ulp of dt must stay well under
        // 1 / c, so the grids hug the origin. The two pixel centers (0.5, 0.5) and (1.5, 0.5) are
        // the only pixel centers whose bound stays below 0.25 of the range.
        var coordinates: [SIMD2<Float>] = [SIMD2(0.5, 0.5), SIMD2(1.5, 0.5)]
        for j in 0..<100 { for i in 0..<100 { coordinates.append(SIMD2(Float(i) * 0.001, Float(j) * 0.001)) } }
        for j in 0..<40 { for i in 0..<300 { coordinates.append(SIMD2(Float(i) * 0.01, Float(j) * 0.0075)) } }
        let colors = [SIMD4<Float>](repeating: .zero, count: coordinates.count)
        let shifts = try probe.run(
            "probeDithering", count: coordinates.count,
            inputs: [probe.buffer(colors), probe.buffer(coordinates)], as: SIMD4<Float>.self
        )
        var worstGrid = 0.0
        var violations = 0
        var worstRatio = 0.0
        var unresolvable = 0
        var signPatternViolations = 0
        for (coordinate, shift) in zip(coordinates, shifts) {
            let (x, y) = (Double(coordinate.x), Double(coordinate.y))
            let bound = Self.resolvableBound(x: x, y: y)
            if bound >= 0.25 { unresolvable += 1 }
            let reference = Reference.shift(gridPosition: Reference.rand(x: x, y: y))
            // The shift is linear in the grid position with slope 1 / 255 per unit, except where
            // the grid position wraps from 1 to 0, where the shift jumps by a full step.
            let gpuGrid = (1 - Double(shift.x) * 255 / 0.5) / 2
            let referenceGrid = (1 - reference.x * 255 / 0.5) / 2
            let distance = Reference.circularDistance(gpuGrid, referenceGrid)
            worstGrid = max(worstGrid, distance)
            worstRatio = max(worstRatio, distance / bound)
            if distance > bound { violations += 1 }
            if !(shift.x == shift.z && abs(Double(shift.x) + Double(shift.y)) < 1e-9) { signPatternViolations += 1 }
        }
        print("DITHER-MEASURE shift vs Double over \(coordinates.count) coordinates with dt < 64: worst grid position error \(worstGrid), worst error / bound \(worstRatio), violations \(violations); a grid position error of 1 is \(1.0 / 255) of color")
        #expect(violations == 0)
        #expect(unresolvable == 0, "every coordinate of this grid must be one float32 can resolve")
        #expect(signPatternViolations == 0)
    }

    @Test func `over a full pixel grid the shader equals a float32 mirror and the Double reference decorrelates`() throws {
        let probe = try Probe(mathMode: .fast)
        let side = 2048
        let coordinates = Self.pixelCenters(side: side, stride: 1)
        let results = try probe.run("probeRand", coordinates, as: SIMD4<Float>.self)
        var decimalAgreement = 0
        var doubleAgreement = 0
        var mirrorAgreement = 0
        var mirrorWorst = 0.0
        for (coordinate, result) in zip(coordinates, results) {
            let (x, y) = (Double(coordinate.x), Double(coordinate.y))
            let grid = Double(result.w)
            if Reference.circularDistance(grid, Reference.rand(x: x, y: y)) <= 0.01 { doubleAgreement += 1 }
            if Reference.circularDistance(grid, Reference.randWithDecimalConstants(x: x, y: y)) <= 0.01 { decimalAgreement += 1 }
            let mirror = Reference.circularDistance(grid, Double(Reference.mirrorRand(x: coordinate.x, y: coordinate.y)))
            mirrorWorst = max(mirrorWorst, mirror)
            if mirror <= 0.01 { mirrorAgreement += 1 }
        }
        let total = Double(coordinates.count)
        print("DITHER-MEASURE \(side)x\(side) pixel centers: within 0.01 of the float32 mirror \(Double(mirrorAgreement) / total) (worst \(mirrorWorst)); within 0.01 of Double with float32 constants \(Double(doubleAgreement) / total); within 0.01 of Double with decimal constants \(Double(decimalAgreement) / total); chance level 0.02")
        #expect(Double(mirrorAgreement) / total >= 0.999)
        #expect(mirrorWorst < 0.02)
        // Float32 cannot follow the Double formula at these coordinates: agreement stays at the
        // level of two independent uniform values falling within 0.01 of each other.
        #expect(Double(doubleAgreement) / total < 0.05)
    }

    @Test func `the grid position is close to uniform on 0 to 1 and the shift stays within half a code`() throws {
        let probe = try Probe(mathMode: .fast)
        let coordinates = Self.pixelCenters(side: 2048, stride: 3)
        let results = try probe.run("probeRand", coordinates, as: SIMD4<Float>.self)
        let grid = results.map { Double($0.w) }
        let mean = grid.reduce(0, +) / Double(grid.count)
        let variance = grid.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(grid.count)
        var bins = [Int](repeating: 0, count: 16)
        for value in grid { bins[min(Int(value * 16), 15)] += 1 }
        let expected = Double(grid.count) / 16
        let binRange = bins.map { Double($0) / expected }
        print("DITHER-MEASURE grid position over \(grid.count) pixels: mean \(mean), variance \(variance) (uniform: 0.5, 0.08333), 16 bins relative to uniform min \(binRange.min()!) max \(binRange.max()!), range \(grid.min()!) to \(grid.max()!)")
        #expect(abs(mean - 0.5) < 0.01)
        #expect(abs(variance - 1.0 / 12) < 0.005)
        #expect(binRange.allSatisfy { abs($0 - 1) < 0.1 })
        #expect(grid.allSatisfy { (0..<1).contains($0) })
        let shifts = try probe.run("probeShift", results.map(\.w), as: SIMD4<Float>.self)
        #expect(shifts.allSatisfy { abs(Double($0.x)) <= 0.5 / 255 + 1e-9 })
    }

    @Test func `fast and safe math modes give identical bits`() throws {
        let fast = try Probe(mathMode: .fast)
        let safe = try Probe(mathMode: .safe)
        let coordinates = Self.pixelCenters(side: 2048, stride: 5)
        let fastResults = try fast.run("probeRand", coordinates, as: SIMD4<Float>.self)
        let safeResults = try safe.run("probeRand", coordinates, as: SIMD4<Float>.self)
        let differing = zip(fastResults, safeResults).filter { $0 != $1 }.count
        print("DITHER-MEASURE fast vs safe math: \(differing) of \(coordinates.count) pixels differ")
        #expect(differing == 0)
    }

    // MARK: - sRGB transfer

    @Test func `sRGB encode and decode match the piecewise definition and invert each other`() throws {
        let probe = try Probe(mathMode: .fast)
        var values: [SIMD4<Float>] = []
        for index in 0...4096 {
            let value = Float(index) / 4096
            values.append(SIMD4(value, value * value, 1 - value, 0))
        }
        values.append(SIMD4(-0.002, -0.5, 4, 0))
        let results = try probe.run("probeTransfer", values, as: SIMD4<Float>.self, perInput: 2)
        func encode(_ x: Double) -> Double {
            let m = abs(x)
            return (m <= 0.0031308 ? m * 12.92 : 1.055 * pow(m, 1 / 2.4) - 0.055) * (x < 0 ? -1 : 1)
        }
        func decode(_ x: Double) -> Double {
            let m = abs(x)
            return (m <= 0.04045 ? m / 12.92 : pow((m + 0.055) / 1.055, 2.4)) * (x < 0 ? -1 : 1)
        }
        var encodeWorst = 0.0
        var decodeWorst = 0.0
        for (index, value) in values.enumerated() {
            for channel in 0..<3 {
                let input = Double(value[channel])
                encodeWorst = max(encodeWorst, abs(Double(results[index * 2][channel]) - encode(input)) / max(1, abs(encode(input))))
                decodeWorst = max(decodeWorst, abs(Double(results[index * 2 + 1][channel]) - decode(input)) / max(1, abs(decode(input))))
            }
        }
        print("DITHER-MEASURE sRGB transfer max error: encode \(encodeWorst) (in 8-bit codes \(encodeWorst * 255)), decode \(decodeWorst)")
        #expect(encodeWorst < 1e-5)
        #expect(decodeWorst < 1e-5)
    }
}
