import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathBlueDitherTests {
    /// Each stripe covers the left half of one column of image pixels.
    /// The binary output records the horizontal primary ray jitter.
    private struct StripeTrace {
        let fixture: PathTestFixture
        let tracer: PathTracer
        let width = 128
        let height: Int
        let camera: PathTracer.CameraModel

        init(height: Int = 64) throws {
            self.height = height
            camera = .init(projection: .orthographic(height: Float(height)))
            fixture = try PathTestFixture()
            var vertices: [MeshVertex] = []
            var indices: [UInt32] = []
            for column in 0..<width {
                let x = Float(column - width / 2)
                let y = Float(height) / 2
                let start = UInt32(vertices.count)
                for point in [SIMD3(x, -y, -3), SIMD3(x + 0.5, -y, -3), SIMD3(x + 0.5, y, -3), SIMD3(x, y, -3)] {
                    vertices.append(MeshVertex(position: point, normal: SIMD3(0, 0, 1)))
                }
                indices.append(contentsOf: [start, start + 1, start + 2, start, start + 2, start + 3])
            }
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: MeshGeometry(vertices: vertices, indices: indices),
                material: .physicallyBased(try .init(alphaMode: .opaque, shadingModel: .unlit))
            )
            let graph = SceneGraph(roots: [SceneNode(mesh: mesh)], camera: fixture.camera)
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .software))
            tracer = try PathTracer(queue: fixture.queue, scene: scene)
        }

        func render(
            sampler: PathTracer.Sampler, seed: UInt32 = 31, batches: [Int] = [1],
            x: Int = 0, y: Int = 0, tileWidth: Int = 128, tileHeight: Int? = nil
        ) throws -> [Float] {
            let tileHeight = tileHeight ?? height
            let accumulation = try tracer.makeAccumulation(width: tileWidth, height: tileHeight)
            let destination = try fixture.output(width: tileWidth, height: tileHeight)
            let region = try PathTracer.Region(imageWidth: width, imageHeight: height, x: x, y: y)
            var history: PathTracer.History?
            for batch in batches {
                let command = try #require(fixture.queue.makeCommandBuffer())
                history = try tracer.encode(
                    cameraModel: camera, cameraTransform: .init(), destination: destination,
                    accumulation: accumulation, previous: history, commandBuffer: command,
                    samplesPerPixel: batch, settings: .init(maxBounces: 1, seed: seed, sampler: sampler), region: region
                )
                try fixture.finish(command)
            }
            return try fixture.readback(destination).map(\.x)
        }
    }

    @Test("The original samplers retain their traced primary-ray footprints")
    func originalFootprints() throws {
        let trace = try StripeTrace()
        // Captured from the original shader on the same render fixture.
        let originals: [(PathTracer.Sampler, UInt64)] = [
            (.independent, 0xb8f007123be1ac21),
            (.sobol, 0xdb8fa8fd00d6b131),
        ]
        for (sampler, expected) in originals {
            let pixels = try trace.render(sampler: sampler)
            #expect(pixels.allSatisfy { $0 == 0 || $0 == 1 })
            let hits = pixels.filter { $0 == 1 }.count
            #expect((3_900...4_300).contains(hits))
            let fingerprint = pixels.reduce(UInt64(14_695_981_039_346_656_037)) { hash, value in
                (hash ^ (value == 1 ? 1 : 0)) &* 1_099_511_628_211
            }
            #expect(fingerprint == expected, "\(sampler): \(String(fingerprint, radix: 16))")
        }
    }

    @Test("Blue-dithered camera rays shift error out of the low spatial frequencies")
    func spatialPower() throws {
        let trace = try StripeTrace(height: 128)
        let blue = try trace.render(sampler: .blueDither)
        let white = try trace.render(sampler: .independent)
        func firstPeriod(_ pixels: [Float]) -> [Float] {
            (0..<64).flatMap { y in pixels[(y * trace.width)..<(y * trace.width + 64)] }
        }
        let bluePeriod = firstPeriod(blue)
        #expect(bluePeriod.allSatisfy { $0 == 0 || $0 == 1 })
        #expect(bluePeriod.filter { $0 == 1 }.count == 2_048)
        let bluePower = lowFrequencyPower(bluePeriod)
        let whitePower = lowFrequencyPower(firstPeriod(white))
        print("Path sampler low-band power relative to white expectation: blue \(bluePower), PCG \(whitePower)")
        #expect(whitePower > 0.5)
        #expect(bluePower < 0.1 && bluePower < 0.2 * whitePower)
        #expect(blue.enumerated().allSatisfy { index, value in
            value == bluePeriod[((index / trace.width) % 64) * 64 + (index % trace.width) % 64]
        })
        #expect(try trace.render(sampler: .blueDither, seed: 32) != blue)
    }

    @Test("Blue-dithered output is deterministic across progressive batches and tile boundaries")
    func batchAndRegion() throws {
        let trace = try StripeTrace(height: 128)
        let full = try trace.render(sampler: .blueDither, batches: [4])
        let split = try trace.render(sampler: .blueDither, batches: [1, 3])
        #expect(full == split)
        // This window crosses both periods and does not begin at a threadgroup boundary.
        let tile = try trace.render(sampler: .blueDither, batches: [2, 2], x: 57, y: 59, tileWidth: 17, tileHeight: 13)
        let window = (59..<(59 + 13)).flatMap { y in full[(y * trace.width + 57)..<(y * trace.width + 57 + 17)] }
        #expect(tile == window)
    }

    /// Fourier power in nonzero modes of radial frequency at most four cycles
    /// per period, divided by the expectation for independent binary samples.
    private func lowFrequencyPower(_ values: [Float]) -> Double {
        var energy = 0.0
        var modeCount = 0
        for u in -4...4 {
            for v in -4...4 where (1...16).contains(u * u + v * v) {
                var real = 0.0, imaginary = 0.0
                for (index, value) in values.enumerated() {
                    let centered = Double(value) - 0.5
                    let phase = 2 * Double.pi * Double(u * (index % 64) + v * (index / 64)) / 64
                    real += centered * cos(phase)
                    imaginary += centered * sin(phase)
                }
                energy += real * real + imaginary * imaginary
                modeCount += 1
            }
        }
        return energy / (0.25 * Double(values.count * modeCount))
    }
}
