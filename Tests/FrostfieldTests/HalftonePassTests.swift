import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HalftonePassTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("All five pinned shapes match rotated grids and preserve nonuniform coverage")
    func shapeReference() throws {
        let width = 23, height = 17
        let rgb = SIMD3<Float>(0.375, 0.625, 0.75)
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = [0, 0.25, 0.5, 0.75, 1][(x + y * 2) % 5]
            return SIMD4(rgb * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let original = try gpu.read(source)
        let effect = try HalftonePass(device: gpu.device)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        let pipelineCount = effect.builtPipelineCount
        var distinctFrames = Set<[SIMD4<Float>]>()
        for shape in HalftonePass.Shape.allCases {
            effect.settings = .init(shape: shape, radius: 4.75, rotations: SIMD3(0.21, -0.45, 0.91))
            try gpu.run { try chain.encode(source: source, destination: destination, commandBuffer: $0) }
            let output = try gpu.read(destination)
            distinctFrames.insert(output)
            for (index, actual) in output.enumerated() {
                let alpha = original[index].w
                let screen = reference(pixel(index, width: width, height: height), settings: effect.settings) { _, channel in
                    rgb[channel]
                }
                let expected = SIMD4(screen * alpha, alpha)
                #expect(PostProcessFixture.close(actual, expected, tolerance: 0.004),
                        "shape \(shape), pixel \(index): \(actual) vs \(expected)")
                #expect(actual.w == alpha)
            }
        }
        #expect(distinctFrames.count == HalftonePass.Shape.allCases.count)
        #expect(effect.builtPipelineCount == pipelineCount)
    }

    @Test("Nine-tap density, scatter and blend modes match a premultiplied HDR fixture")
    func sampleAndStyleReference() throws {
        let width = 17, height = 11
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = x < 2 ? 0.25 : 0.5
            let rgb = SIMD3<Float>(Float(x + y) / 24, 0.25 + Float(y) / 32, 1.25 + Float(x) / 16)
            return SIMD4(rgb * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let input = try gpu.read(source)
        let effect = try HalftonePass(device: gpu.device)
        for mode in HalftonePass.BlendMode.allCases {
            effect.settings = .init(
                shape: .square, radius: 4.25, rotations: .zero, scatter: 0.63,
                blending: 0.43, blendMode: mode, grayscale: mode == .darker
            )
            let settings = effect.settings
            try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
            let output = try gpu.read(destination)
            for (index, actual) in output.enumerated() {
                let sample = input[index]
                let screen = reference(pixel(index, width: width, height: height), settings: settings) { center, channel in
                    let phase = phase(center)
                    var color = bilinear(input, width: width, height: height, point: center)
                    for tap in 0..<8 {
                        let angle = phase + Float(tap) * .pi / 4
                        let point = center + SIMD2(cos(angle), sin(angle)) * (settings.radius * 0.66)
                        color += bilinear(input, width: width, height: height, point: point)
                    }
                    return min(color[channel] / color.w, 1)
                }
                let straight = SIMD3(sample.x, sample.y, sample.z) / sample.w
                let bounded = simd_min(straight, SIMD3(repeating: 1))
                let styled: SIMD3<Float>
                switch mode {
                case .linear: styled = bounded * (1 - settings.blending) + screen * settings.blending
                case .multiply: styled = mix(screen, screen * bounded, settings.blending)
                case .add: styled = mix(screen, simd_min(screen + bounded, SIMD3(repeating: 1)), settings.blending)
                case .lighter: styled = mix(screen, simd_max(screen, bounded), settings.blending)
                case .darker: styled = mix(screen, simd_min(screen, bounded), settings.blending)
                }
                var rgb = styled + straight - bounded
                if settings.grayscale { rgb = SIMD3(repeating: (rgb.x + rgb.y + rgb.z) / 3) }
                let expected = SIMD4(rgb * sample.w, sample.w)
                #expect(PostProcessFixture.close(actual, expected, tolerance: 0.018),
                        "mode \(mode), pixel \(index): \(actual) vs \(expected)")
                #expect(actual.w == sample.w)
            }
            if !settings.grayscale { #expect(output.contains { $0.z > 1 }) }
        }

        effect.settings.isEnabled = false
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination) == input)
        effect.settings = .init(blending: 0)
        try gpu.run { try effect.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination) == input)

        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: source, destination: source, commandBuffer: command)
        }
        effect.settings.rotations.x = .infinity
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, destination: destination, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.self) {
            try HalftonePass(device: gpu.device, settings: .init(radius: 0))
        }
    }

    private func pixel(_ index: Int, width: Int, height: Int) -> SIMD2<Float> {
        SIMD2(Float(index % width) + 0.5, Float(height - index / width) - 0.5)
    }

    private func phase(_ center: SIMD2<Float>) -> Float {
        let center = floor(center)
        let seed = (center.x * 12.9898).addingProduct(center.y, 78.233)
        let hash = sin(seed) * 43758.5453
        return (hash - floor(hash)) * (2 * .pi)
    }

    private func reference(
        _ p: SIMD2<Float>, settings: HalftonePass.Settings, density: (SIMD2<Float>, Int) -> Float
    ) -> SIMD3<Float> {
        let radius = settings.radius
        var color = SIMD3<Float>.zero
        for channel in 0..<3 {
            let angle = settings.rotations[channel]
            let normal = SIMD2(cos(angle), sin(angle))
            let tangent = SIMD2(-normal.y, normal.x)
            let projected = SIMD2(simd_dot(p, normal), simd_dot(p, tangent))
            let fromLow = simd_abs(projected) - floor(simd_abs(projected) / radius) * radius
            let sign = SIMD2<Float>(projected.x < 0 ? -1 : 1, projected.y < 0 ? -1 : 1)
            let nearX = fromLow.x < radius * 0.5
            let nearY = fromLow.y < radius * 0.5
            let shift = SIMD2(nearX ? -fromLow.x : radius - fromLow.x,
                              nearY ? -fromLow.y : radius - fromLow.y) * sign
            var corner = p + normal * shift.x + tangent * shift.y
            if settings.scatter > 0 {
                let scatterAngle = phase(corner)
                corner += SIMD2(cos(scatterAngle), sin(scatterAngle)) * (settings.scatter * radius / 4)
            }
            let other = SIMD2(nearX ? sign.x : -sign.x, nearY ? sign.y : -sign.y) * radius
            let across = normal * other.x
            let along = tangent * other.y
            for center in [corner, corner + across, corner + along, corner + across + along] {
                let delta = p - center
                var distance = simd_length(delta)
                let level = density(center, channel)
                let extent: Float
                switch settings.shape {
                case .dot: extent = pow(level, 1.125) * radius
                case .ellipse:
                    extent = pow(level, 1.125) * radius
                    if distance > 0 {
                        let projected = abs(simd_dot(delta / distance, normal))
                        distance = distance * 0.79289322 + projected * distance * 0.41421356
                    }
                case .line:
                    extent = pow(level, 1.5) * radius
                    distance = simd_length(normal * simd_dot(delta, normal))
                case .square, .diamond:
                    let tilt = settings.shape == .diamond ? Float.pi / 4 : 0
                    let azimuth = distance > 0 ? atan2(delta.y, delta.x) - angle - tilt : 0
                    let alignment = max(abs(cos(azimuth)), abs(sin(azimuth)))
                    extent = pow(level, 1.4) * radius * (2 - alignment)
                }
                color[channel] += min(max((extent - distance) / min(radius / 2, 1.25), 0), 1)
            }
        }
        return simd_min(color, SIMD3(repeating: 1))
    }

    private func bilinear(_ input: [SIMD4<Float>], width: Int, height: Int, point: SIMD2<Float>) -> SIMD4<Float> {
        let last = SIMD2(Float(width - 1), Float(height - 1))
        let topDown = SIMD2(point.x, Float(height) - point.y)
        let clamped = simd_clamp(topDown - 0.5, .zero, last)
        let low = floor(clamped)
        let x = Int(low.x), y = Int(low.y)
        let nextX = min(x + 1, width - 1), nextY = min(y + 1, height - 1)
        let fraction = clamped - low
        let top = input[y * width + x] * (1 - fraction.x) + input[y * width + nextX] * fraction.x
        let bottom = input[nextY * width + x] * (1 - fraction.x) + input[nextY * width + nextX] * fraction.x
        return top * (1 - fraction.y) + bottom * fraction.y
    }

    private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
        a + (b - a) * t
    }
}
