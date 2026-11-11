import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import simd

/// Renders a white, rotated half-plane over transparent black with the 90° camera
/// of `GeometryRenderFixture` and measures aliasing against an 8 × 8 supersampled
/// render of the same scene.
struct EdgeScene {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let side: Int
    let angle: Float
    let mesh: MeshRenderer
    var graph: SceneGraph

    /// `angle` rotates the half-plane's boundary away from vertical. Its boundary
    /// passes through the view center at depth 2, where the view spans ±2 units.
    init(side: Int = 64, angle: Float, cameraX: Float = 0) throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        self.side = side
        self.angle = angle
        mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(6), material: .unlit
        )
        let rotation = simd_quatf(angle: angle, axis: SIMD3(0, 0, 1))
        let center = rotation.act(SIMD3<Float>(-6, 0, 0)) + SIMD3(0, 0, -2)
        // The unlit material shades a +Z normal as (0.22, 0.22, 1) × tint.
        let node = SceneNode(
            transform: SceneTransform(translation: center, rotation: rotation),
            mesh: mesh, materialParameters: SIMD4(1 / 0.22, 1 / 0.22, 1, 1)
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50)
        graph = SceneGraph(roots: [node], camera: camera, cameraTransform: SceneTransform(translation: SIMD3(cameraX, 0, 0)))
    }

    /// Color with shader-read usage and stored, shader-readable depth.
    func makeTarget(side: Int? = nil) throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let side = side ?? self.side
        let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false)
        color.storageMode = .shared
        color.usage = [.renderTarget, .shaderRead]
        let depth = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: side, height: side, mipmapped: false)
        depth.storageMode = .private
        depth.usage = [.renderTarget, .shaderRead]
        return (try #require(device.makeTexture(descriptor: color)), try #require(device.makeTexture(descriptor: depth)))
    }

    func encode(_ graph: SceneGraph, into target: (color: any MTLTexture, depth: any MTLTexture), commandBuffer: any MTLCommandBuffer) throws {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target.color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = target.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1
        try graph.encode(pass: pass, commandBuffer: commandBuffer)
    }

    func run(_ body: (any MTLCommandBuffer) throws -> Void) throws {
        let buffer = try #require(queue.makeCommandBuffer())
        try body(buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(String(describing: buffer.error))")
    }

    /// One sample per pixel at the pixel center.
    func render(_ graph: SceneGraph? = nil) throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let target = try makeTarget()
        try run { try encode(graph ?? self.graph, into: target, commandBuffer: $0) }
        return target
    }

    /// The box-filtered mean of an 8 × 8 grid of samples per pixel.
    func supersampledReference(_ graph: SceneGraph? = nil) throws -> [SIMD4<Float>] {
        let factor = 8
        let target = try makeTarget(side: side * factor)
        try run { try encode(graph ?? self.graph, into: target, commandBuffer: $0) }
        let fine = try Self.read(target.color)
        var result = [SIMD4<Float>](repeating: .zero, count: side * side)
        for y in 0..<side {
            for x in 0..<side {
                var sum = SIMD4<Float>.zero
                for v in 0..<factor {
                    for u in 0..<factor { sum += fine[(y * factor + v) * side * factor + x * factor + u] }
                }
                result[y * side + x] = sum / Float(factor * factor)
            }
        }
        return result
    }

    static func read(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var halves = [SIMD4<Float16>](repeating: .zero, count: texture.width * texture.height)
        try halves.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return halves.map(SIMD4<Float>.init)
    }

    /// Pixels within `radius` of a pixel the reference edge partially covers.
    func edgeBand(_ reference: [SIMD4<Float>], radius: Int = 2) -> [Int] {
        var band = Set<Int>()
        for y in 0..<side {
            for x in 0..<side where reference[y * side + x].w > 0.01 && reference[y * side + x].w < 0.99 {
                for v in max(0, y - radius)...min(side - 1, y + radius) {
                    for u in max(0, x - radius)...min(side - 1, x + radius) { band.insert(v * side + u) }
                }
            }
        }
        return band.sorted()
    }

    /// Root-mean-square RGBA error over the chosen pixels.
    static func rmse(_ image: [SIMD4<Float>], _ reference: [SIMD4<Float>], over pixels: [Int]) -> Float {
        let sum = pixels.reduce(Float(0)) { total, index in
            let difference = image[index] - reference[index]
            return total + simd_dot(difference, difference) / 4
        }
        return (sum / Float(pixels.count)).squareRoot()
    }

    /// Writes an image, scaled up 4× without filtering, when FROSTFIELD_TEST_IMAGES
    /// names a directory (pass TEST_RUNNER_FROSTFIELD_TEST_IMAGES to xcodebuild).
    static func save(_ pixels: [SIMD4<Float>], side: Int, name: String) {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_TEST_IMAGES"] else { return }
        let scale = 4
        let width = side * scale
        var bytes = [UInt8](repeating: 0, count: width * width * 4)
        func encode(_ value: Float) -> UInt8 {
            let linear = min(max(value, 0), 1)
            let encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
            return UInt8((encoded * 255).rounded())
        }
        for y in 0..<width {
            for x in 0..<width {
                let pixel = pixels[(y / scale) * side + x / scale]
                let offset = (y * width + x) * 4
                bytes[offset] = encode(pixel.x)
                bytes[offset + 1] = encode(pixel.y)
                bytes[offset + 2] = encode(pixel.z)
                bytes[offset + 3] = 255
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                width: width, height: width, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ),
              let destination = CGImageDestinationCreateWithURL(
                URL(fileURLWithPath: directory).appendingPathComponent(name + ".png") as CFURL, "public.png" as CFString, 1, nil
              ) else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct FXAATests {
    @Test("FXAA reduces edge aliasing against an 8 × 8 supersampled reference", arguments: [Float(0.25), 0.7])
    func edgeAliasing(angle: Float) throws {
        let scene = try EdgeScene(angle: angle)
        let reference = try scene.supersampledReference()
        let raw = try scene.render()
        let fxaa = try FXAA(device: scene.device)
        let output = try scene.makeTarget().color
        try scene.run { try fxaa.encode(source: raw.color, destination: output, commandBuffer: $0) }

        let rawPixels = try EdgeScene.read(raw.color)
        let filtered = try EdgeScene.read(output)
        let band = scene.edgeBand(reference)
        let before = EdgeScene.rmse(rawPixels, reference, over: band)
        let after = EdgeScene.rmse(filtered, reference, over: band)
        print("[rasterlook] FXAA angle \(angle): edge RMSE \(before) -> \(after) over \(band.count) pixels")
        EdgeScene.save(reference, side: scene.side, name: "fxaa-\(angle)-reference")
        EdgeScene.save(rawPixels, side: scene.side, name: "fxaa-\(angle)-raw")
        EdgeScene.save(filtered, side: scene.side, name: "fxaa-\(angle)-fxaa")
        #expect(after < 0.7 * before, "FXAA edge RMSE \(after) versus \(before) without it")

        // Pixels whose whole 5 × 5 neighborhood is flat stay bit-identical.
        var flatCount = 0
        for y in 2..<(scene.side - 2) {
            for x in 2..<(scene.side - 2) {
                let center = rawPixels[y * scene.side + x]
                let flat = (-2...2).allSatisfy { v in (-2...2).allSatisfy { u in rawPixels[(y + v) * scene.side + x + u] == center } }
                guard flat else { continue }
                flatCount += 1
                #expect(filtered[y * scene.side + x] == center, "flat pixel \(x), \(y)")
            }
        }
        #expect(flatCount > scene.side * scene.side / 2)
    }

    @Test("FXAA's edge-end search alone, without sub-pixel blending, reduces aliasing", arguments: [Float(0.25), 0.7])
    func edgeSearch(angle: Float) throws {
        let scene = try EdgeScene(angle: angle)
        let reference = try scene.supersampledReference()
        let band = scene.edgeBand(reference)
        let raw = try scene.render()
        let before = EdgeScene.rmse(try EdgeScene.read(raw.color), reference, over: band)
        let output = try scene.makeTarget().color
        try scene.run {
            try FXAA(device: scene.device).encode(
                source: raw.color, destination: output, settings: .init(subpixelBlending: 0), commandBuffer: $0
            )
        }
        let after = EdgeScene.rmse(try EdgeScene.read(output), reference, over: band)
        print("[rasterlook] FXAA without sub-pixel blending, angle \(angle): edge RMSE \(before) -> \(after)")
        // Measured on an M4 Max: 0.49 and 0.72 of the one-sample error; 1 without the search.
        #expect(after < 0.8 * before, "FXAA edge search RMSE \(after) versus \(before)")
    }

    @Test("Each FXAA preset's edge-end search blends a stair step by its closed-form amount")
    func presetSearchReach() throws {
        // Dark above row 16 and white below; row 16 is white left of column 64 and dark from it
        // on, so the horizontal edge steps down one row at x = 64 and runs on to the right.
        let fixture = try PostProcessFixture()
        let width = 128, height = 32
        let source = try fixture.color(width: width, height: height) { x, y in
            y >= 17 || (y == 16 && x < 64) ? SIMD4(1, 1, 1, 1) : SIMD4(0, 0, 0, 1)
        }
        // FXAA 3.11's step tables. Every step but the last is sampled; the last only extends
        // the span when the end is not found.
        let tables: [(FXAA.Quality, [Float])] = [
            (.medium, [1, 1.5, 2, 4, 12]),
            (.high, [1, 1.5, 2, 2, 2, 2, 2, 2, 2, 2, 4, 8]),
            (.extreme, [1, 1, 1, 1, 1, 1.5, 2, 2, 2, 2, 4, 8]),
        ]
        for (quality, steps) in tables {
            let output = try fixture.color(width: width, height: height)
            try fixture.run {
                try FXAA(device: fixture.device).encode(
                    source: source, destination: output, settings: .init(quality: quality, subpixelBlending: 0), commandBuffer: $0
                )
            }
            let pixels = try fixture.read(output)
            let sampled = steps.dropLast().reduce(into: [Float]()) { $0.append(($0.last ?? 0) + $1) }
            let reach = steps.reduce(0, +)
            for distance in [3, 7, 11, 15, 21] {
                // The pixel at column 64 + d searches left along the row 16/17 boundary; the
                // bilinear luma there jumps once a sample lies past x = 64, which is d + 0.5
                // pixels away. To the right the edge never ends within the reach.
                let found = sampled.first { $0 > Float(distance) + 0.5 }
                let expected: Float = found.map { 0.5 - $0 / ($0 + reach) } ?? 0
                // The pixel blends with the white pixel below by the offset.
                let actual = pixels[16 * width + 64 + distance].x
                #expect(abs(actual - expected) <= 6e-3, "\(quality) at distance \(distance): \(actual), expected \(expected)")
            }
        }
    }

    @Test("FXAA leaves a smooth gradient untouched")
    func smoothGradient() throws {
        let fixture = try PostProcessFixture()
        // A horizontal ramp from 0.25 to 0.75 with steps far below FXAA's contrast threshold.
        let source = try fixture.color(width: 64, height: 16) { x, _ in
            let value: Float = 0.25 + 0.5 * Float(x) / 63
            return SIMD4<Float>(value, value, value, 1)
        }
        let target = try fixture.color(width: 64, height: 16)
        let fxaa = try FXAA(device: fixture.device)
        try fixture.run { try fxaa.encode(source: source, destination: target, commandBuffer: $0) }
        #expect(try fixture.read(target) == fixture.read(source))
    }

    @Test("A post-process chain runs tone mapping then FXAA like the standalone passes")
    func chainStages() throws {
        let scene = try EdgeScene(angle: 0.5)
        let hdr = try scene.render().color
        let side = scene.side
        func srgbTarget() throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: side, height: side, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = .renderTarget
            return try #require(scene.device.makeTexture(descriptor: descriptor))
        }
        func bytes(_ texture: any MTLTexture) -> [UInt8] {
            var bytes = [UInt8](repeating: 0, count: side * side * 4)
            bytes.withUnsafeMutableBytes { raw in
                texture.getBytes(raw.baseAddress!, bytesPerRow: side * 4, from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0)
            }
            return bytes
        }
        let display = HDRDisplay.Settings(exposureStops: 0.5, toneMapping: .acesFilmic)
        let chain = try PostProcessChain(
            queue: scene.queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(display), .fxaa(.init())]
        )
        let chained = try srgbTarget()
        try scene.run { try chain.encode(source: hdr, destination: chained, commandBuffer: $0) }

        let mapped = try scene.makeTarget().color
        let manual = try srgbTarget()
        try scene.run { buffer in
            try HDRDisplay(device: scene.device, destinationPixelFormat: .rgba16Float)
                .encode(source: hdr, destination: mapped, settings: display, commandBuffer: buffer)
            try FXAA(device: scene.device, destinationPixelFormat: .bgra8Unorm_srgb)
                .encode(source: mapped, destination: manual, commandBuffer: buffer)
        }
        #expect(bytes(chained) == bytes(manual))
        // FXAA produced intermediate coverage on the edge.
        #expect(bytes(chained).enumerated().contains { $0.offset % 4 == 1 && $0.element > 20 && $0.element < 235 })

        #expect(throws: FXAA.Failure.invalidSettings) {
            try chain.replaceEffect(at: 1, with: .fxaa(.init(edgeThreshold: 0)))
        }
        #expect(throws: HDRDisplay.Failure.invalidHeadroom) {
            try chain.replaceEffect(at: 0, with: .toneMapping(.init(displayHeadroom: 100)))
        }
        // A headroom above 1 cannot be written to the chain's sRGB destination.
        try chain.setEnabled(false, at: 1)
        try chain.replaceEffect(at: 0, with: .toneMapping(.init(displayHeadroom: 2)))
        let buffer = try #require(scene.queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.headroomNeedsExtendedRange) {
            try chain.encode(source: hdr, destination: chained, commandBuffer: buffer)
        }
    }

    @Test("FXAA quality presets and settings are validated before encoding")
    func validation() throws {
        let fixture = try PostProcessFixture()
        let source = try fixture.color(width: 4, height: 4)
        let target = try fixture.color(width: 4, height: 4)
        let fxaa = try FXAA(device: fixture.device)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: FXAA.Failure.invalidSettings) {
            try fxaa.encode(source: source, destination: target, settings: .init(subpixelBlending: 2), commandBuffer: buffer)
        }
        #expect(throws: FXAA.Failure.invalidSettings) {
            try fxaa.encode(source: source, destination: target, settings: .init(edgeThreshold: .nan), commandBuffer: buffer)
        }
        #expect(throws: FXAA.Failure.destination) {
            try fxaa.encode(source: source, destination: source, commandBuffer: buffer)
        }
        let small = try fixture.color(width: 2, height: 2)
        #expect(throws: FXAA.Failure.sizeMismatch) {
            try fxaa.encode(source: source, destination: small, commandBuffer: buffer)
        }
        #expect(throws: FXAA.Failure.destinationFormat) {
            try FXAA(device: fixture.device, destinationPixelFormat: .r8Unorm)
        }
        // Every preset produces a finite image.
        for quality in [FXAA.Quality.medium, .high, .extreme] {
            let scene = try EdgeScene(side: 32, angle: 0.4)
            let raw = try scene.render()
            let output = try scene.makeTarget().color
            try scene.run { try FXAA(device: scene.device).encode(source: raw.color, destination: output, settings: .init(quality: quality), commandBuffer: $0) }
            #expect(try EdgeScene.read(output).allSatisfy { $0.x.isFinite && $0.w >= 0 && $0.w <= 1 })
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TemporalAntiAliasingTests {
    @Test("Halton (2, 3) jitter offsets are centered and distinct")
    func haltonJitter() {
        // Halton (2, 3) points 1 and 2 are (1/2, 1/3) and (1/4, 2/3).
        #expect(simd_distance(TemporalAntiAliasing.jitter(at: 0), SIMD2(0, -1.0 / 6)) < 1e-6)
        #expect(simd_distance(TemporalAntiAliasing.jitter(at: 1), SIMD2(-0.25, 1.0 / 6)) < 1e-6)
        #expect(abs(TemporalAntiAliasing.jitter(at: 2).y - (1.0 / 9 - 0.5)) < 1e-6)
        let offsets = (0..<16).map(TemporalAntiAliasing.jitter(at:))
        #expect(Set(offsets.map { [$0.x, $0.y] }).count == 16)
        #expect(offsets.allSatisfy { abs($0.x) < 0.5 && abs($0.y) < 0.5 })
        let mean = offsets.reduce(.zero, +) / 16
        #expect(abs(mean.x) < 0.04 && abs(mean.y) < 0.04)
    }

    @Test("Projection jitter moves the rendered image by the offset, x right and y down")
    func jitterDirection() throws {
        // A vertical boundary at the view center lies on the line between columns 31 and 32.
        var scene = try EdgeScene(side: 64, angle: 0)
        let still = try EdgeScene.read(scene.render().color)
        #expect(still[32 * 64 + 31].w == 1 && still[32 * 64 + 32].w == 0)
        scene.graph.projectionJitter = SIMD2(0.6, 0)
        let right = try EdgeScene.read(scene.render().color)
        #expect(right[32 * 64 + 32].w == 1, "Moving the image right covers column 32")
        scene.graph.projectionJitter = SIMD2(-0.6, 0)
        let left = try EdgeScene.read(scene.render().color)
        #expect(left[32 * 64 + 31].w == 0, "Moving the image left uncovers column 31")

        // A horizontal boundary: rotate the half-plane a quarter turn so it covers the top half.
        var horizontal = try EdgeScene(side: 64, angle: -.pi / 2)
        let top = try EdgeScene.read(horizontal.render().color)
        #expect(top[31 * 64 + 32].w == 1 && top[32 * 64 + 32].w == 0)
        horizontal.graph.projectionJitter = SIMD2(0, 0.6)
        let down = try EdgeScene.read(horizontal.render().color)
        #expect(down[32 * 64 + 32].w == 1, "Moving the image down covers row 32")

        scene.graph.projectionJitter = SIMD2(1.5, 0)
        #expect(throws: MeshRenderer.Failure.projectionJitter) { _ = try scene.render() }
    }

    @Test("TAA converges a jittered edge toward the supersampled reference", arguments: [Float(0.25), 0.7])
    func convergence(angle: Float) throws {
        var scene = try EdgeScene(angle: angle)
        let reference = try scene.supersampledReference()
        let band = scene.edgeBand(reference)
        let single = try EdgeScene.read(scene.render().color)
        let singleError = EdgeScene.rmse(single, reference, over: band)

        let taa = try TemporalAntiAliasing(device: scene.device)
        let output = try scene.makeTarget().color
        let clipFromWorld = try scene.graph.clipFromWorld(width: scene.side, height: scene.side)
        var errors: [Float] = []
        for _ in 0..<32 {
            let jitter = taa.nextJitter
            scene.graph.projectionJitter = jitter
            let frame = try scene.makeTarget()
            try scene.run { buffer in
                try scene.encode(scene.graph, into: frame, commandBuffer: buffer)
                try taa.encode(
                    source: frame.color, depth: frame.depth, clipFromWorld: clipFromWorld, jitter: jitter,
                    destination: output, commandBuffer: buffer
                )
            }
            errors.append(EdgeScene.rmse(try EdgeScene.read(output), reference, over: band))
        }
        let resolved = try EdgeScene.read(output)
        print("[rasterlook] TAA angle \(angle): single \(singleError), after 1/8/16/32 frames \(errors[0]) \(errors[7]) \(errors[15]) \(errors[31])")
        EdgeScene.save(single, side: scene.side, name: "taa-\(angle)-single")
        EdgeScene.save(resolved, side: scene.side, name: "taa-\(angle)-resolved")
        EdgeScene.save(reference, side: scene.side, name: "taa-\(angle)-reference")
        #expect(errors[31] < 0.4 * singleError, "TAA RMSE \(errors[31]) versus \(singleError) for one sample")
        #expect(errors[15] < errors[0])
        #expect(taa.accumulatedFrameCount == 32)

        // Equal-weight accumulation without clipping is progressive supersampling.
        let accumulation = try TemporalAntiAliasing(device: scene.device, settings: .accumulation)
        for _ in 0..<64 {
            let jitter = accumulation.nextJitter
            scene.graph.projectionJitter = jitter
            let frame = try scene.makeTarget()
            try scene.run { buffer in
                try scene.encode(scene.graph, into: frame, commandBuffer: buffer)
                try accumulation.encode(
                    source: frame.color, depth: frame.depth, clipFromWorld: clipFromWorld, jitter: jitter,
                    destination: output, commandBuffer: buffer
                )
            }
        }
        let supersampled = EdgeScene.rmse(try EdgeScene.read(output), reference, over: band)
        print("[rasterlook] SSAA accumulation angle \(angle): 64 frames \(supersampled)")
        #expect(supersampled < 0.15 * singleError, "Accumulated RMSE \(supersampled) versus \(singleError)")
    }

    @Test("Accumulation keeps the exact mean of over a thousand jittered frames")
    func longAccumulation() throws {
        var scene = try EdgeScene(angle: 0.25)
        let reference = try scene.supersampledReference()
        let band = scene.edgeBand(reference)
        let accumulation = try TemporalAntiAliasing(device: scene.device, settings: .accumulation)
        let output = try scene.makeTarget().color
        let clipFromWorld = try scene.graph.clipFromWorld(width: scene.side, height: scene.side)
        let frameCount = 1024
        var sum = [SIMD4<Double>](repeating: .zero, count: scene.side * scene.side)
        var report: [String] = []
        var worst: Float = 0
        for frame in 1...frameCount {
            let jitter = accumulation.nextJitter
            scene.graph.projectionJitter = jitter
            let target = try scene.makeTarget()
            try scene.run { buffer in
                try scene.encode(scene.graph, into: target, commandBuffer: buffer)
                try accumulation.encode(
                    source: target.color, depth: target.depth, clipFromWorld: clipFromWorld, jitter: jitter,
                    destination: output, commandBuffer: buffer
                )
            }
            for (index, pixel) in try EdgeScene.read(target.color).enumerated() { sum[index] += SIMD4<Double>(pixel) }
            guard [64, 256, 1024].contains(frame) else { continue }
            let resolved = try EdgeScene.read(output)
            let mean = sum.map { SIMD4<Float>($0 / Double(frame)) }
            worst = zip(resolved, mean).reduce(0) { max($0, simd_reduce_max(simd_abs($1.0 - $1.1))) }
            report.append("\(frame) frames: RMSE \(EdgeScene.rmse(resolved, reference, over: band)), exact mean \(EdgeScene.rmse(mean, reference, over: band)), worst pixel off by \(worst)")
            // Half-float output stores values up to 1 to within 2.5e-4.
            #expect(worst <= 1e-3, "after \(frame) frames the accumulated image differs from the exact mean by \(worst)")
        }
        print("[rasterlook] accumulation: " + report.joined(separator: "; "))
        #expect(accumulation.accumulatedFrameCount == frameCount)
    }

    @Test("Catmull-Rom history keeps a sub-pixel pan sharper than bilinear resampling would")
    func subpixelPan() throws {
        var scene = try EdgeScene(angle: 0.3)
        let taa = try TemporalAntiAliasing(device: scene.device)
        let output = try scene.makeTarget().color
        // 0.4 pixel per frame: at depth 2 the view spans 4 units over 64 pixels. Each frame
        // reads history 0.4 pixel away from a texel center, where the filter matters.
        let step: Float = 0.4 / 16
        var errors: [Float] = []
        var singles: [Float] = []
        for _ in 0..<40 {
            let jitter = taa.nextJitter
            scene.graph.projectionJitter = jitter
            let clipFromWorld = try scene.graph.clipFromWorld(width: scene.side, height: scene.side)
            let target = try scene.makeTarget()
            try scene.run { buffer in
                try scene.encode(scene.graph, into: target, commandBuffer: buffer)
                try taa.encode(
                    source: target.color, depth: target.depth, clipFromWorld: clipFromWorld, jitter: jitter,
                    destination: output, commandBuffer: buffer
                )
            }
            var still = scene
            still.graph.projectionJitter = .zero
            let reference = try still.supersampledReference()
            let band = still.edgeBand(reference)
            errors.append(EdgeScene.rmse(try EdgeScene.read(output), reference, over: band))
            singles.append(EdgeScene.rmse(try EdgeScene.read(still.render().color), reference, over: band))
            scene.graph.cameraTransform.translation.x += step
        }
        // The last 20 frames, after the history has settled into the motion.
        let error = errors[20...].reduce(0, +) / 20
        let single = singles[20...].reduce(0, +) / 20
        print("[rasterlook] TAA 0.4-pixel pan, mean of frames 21-40: single \(single), resolved \(error)")
        EdgeScene.save(try EdgeScene.read(output), side: scene.side, name: "taa-pan")
        // Measured on an M4 Max: 0.084 with Catmull-Rom history, 0.105 with one bilinear tap.
        #expect(error < 0.85 * single, "panned TAA RMSE \(error) versus \(single) for one sample")
    }

    @Test("Reset discards history: the next resolve is the current frame")
    func reset() throws {
        var scene = try EdgeScene(angle: 0.4)
        let taa = try TemporalAntiAliasing(device: scene.device)
        let output = try scene.makeTarget().color
        let clipFromWorld = try scene.graph.clipFromWorld(width: scene.side, height: scene.side)
        func frame() throws -> [SIMD4<Float>] {
            let jitter = taa.nextJitter
            scene.graph.projectionJitter = jitter
            let target = try scene.makeTarget()
            try scene.run { buffer in
                try scene.encode(scene.graph, into: target, commandBuffer: buffer)
                try taa.encode(
                    source: target.color, depth: target.depth, clipFromWorld: clipFromWorld, jitter: jitter,
                    destination: output, commandBuffer: buffer
                )
            }
            return try EdgeScene.read(target.color)
        }
        for _ in 0..<12 { _ = try frame() }
        let converged = try EdgeScene.read(output)
        #expect(converged.contains { $0.w > 0.05 && $0.w < 0.95 })

        taa.reset()
        #expect(taa.accumulatedFrameCount == 0 && taa.frameIndex == 0)
        #expect(taa.nextJitter == TemporalAntiAliasing.jitter(at: 0))
        let current = try frame()
        #expect(try EdgeScene.read(output) == current)
        #expect(taa.accumulatedFrameCount == 1)
        // A single aliased frame has no partial coverage.
        #expect(current.allSatisfy { $0.w == 0 || $0.w == 1 })

        // Switching to accumulation changes the history format, which starts it over.
        for _ in 0..<4 { _ = try frame() }
        taa.settings = .accumulation
        let restarted = try frame()
        #expect(try EdgeScene.read(output) == restarted)
        #expect(taa.accumulatedFrameCount == 1)
    }

    @Test("Reprojection keeps history aligned when the camera moves")
    func cameraMotion() throws {
        var scene = try EdgeScene(angle: 0.3)
        let taa = try TemporalAntiAliasing(device: scene.device)
        let output = try scene.makeTarget().color
        func frame() throws {
            let jitter = taa.nextJitter
            scene.graph.projectionJitter = jitter
            let clipFromWorld = try scene.graph.clipFromWorld(width: scene.side, height: scene.side)
            let target = try scene.makeTarget()
            try scene.run { buffer in
                try scene.encode(scene.graph, into: target, commandBuffer: buffer)
                try taa.encode(
                    source: target.color, depth: target.depth, clipFromWorld: clipFromWorld, jitter: jitter,
                    destination: output, commandBuffer: buffer
                )
            }
        }
        for _ in 0..<24 { try frame() }

        // Move the camera 0.21 units right: at depth 2 the edge shifts 3.36 pixels left.
        scene.graph.cameraTransform.translation.x = 0.21
        scene.graph.projectionJitter = .zero
        let reference = try scene.supersampledReference()
        let band = scene.edgeBand(reference)
        let singleError = EdgeScene.rmse(try EdgeScene.read(scene.render().color), reference, over: band)
        try frame()
        let movedError = EdgeScene.rmse(try EdgeScene.read(output), reference, over: band)
        for _ in 0..<8 { try frame() }
        let settledError = EdgeScene.rmse(try EdgeScene.read(output), reference, over: band)
        print("[rasterlook] TAA camera motion: single \(singleError), first moved frame \(movedError), 8 frames later \(settledError)")
        EdgeScene.save(try EdgeScene.read(output), side: scene.side, name: "taa-moved")
        #expect(movedError < 0.6 * singleError, "Reprojected history should stay anti-aliased: \(movedError) vs \(singleError)")
        #expect(settledError < 0.5 * singleError)
    }

    @Test("TAA validates its inputs before encoding")
    func validation() throws {
        let scene = try EdgeScene(side: 8, angle: 0.4)
        let taa = try TemporalAntiAliasing(device: scene.device)
        let frame = try scene.makeTarget()
        let output = try scene.makeTarget().color
        let clipFromWorld = try scene.graph.clipFromWorld(width: 8, height: 8)
        let buffer = try #require(scene.queue.makeCommandBuffer())
        #expect(throws: TemporalAntiAliasing.Failure.jitter) {
            try taa.encode(source: frame.color, depth: frame.depth, clipFromWorld: clipFromWorld, jitter: SIMD2(.nan, 0), destination: output, commandBuffer: buffer)
        }
        #expect(throws: TemporalAntiAliasing.Failure.transform) {
            try taa.encode(source: frame.color, depth: frame.depth, clipFromWorld: simd_float4x4(), jitter: .zero, destination: output, commandBuffer: buffer)
        }
        #expect(throws: TemporalAntiAliasing.Failure.destination) {
            try taa.encode(source: frame.color, depth: frame.depth, clipFromWorld: clipFromWorld, jitter: .zero, destination: frame.color, commandBuffer: buffer)
        }
        let large = try scene.makeTarget(side: 16)
        #expect(throws: TemporalAntiAliasing.Failure.sizeMismatch) {
            try taa.encode(source: frame.color, depth: large.depth, clipFromWorld: clipFromWorld, jitter: .zero, destination: output, commandBuffer: buffer)
        }
        taa.settings.currentFrameWeight = 0
        #expect(throws: TemporalAntiAliasing.Failure.invalidSettings) {
            try taa.encode(source: frame.color, depth: frame.depth, clipFromWorld: clipFromWorld, jitter: .zero, destination: output, commandBuffer: buffer)
        }
        #expect(taa.accumulatedFrameCount == 0)
    }
}
