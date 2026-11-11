import Foundation
import Frostfield
import Metal
import simd

/// A still snapshot of a wind-driven sea, with a dense patch near the camera and iceberg.
enum LabIcebergOcean {
    private struct Wave {
        let waveVector: SIMD2<Float>
        let bendVector: SIMD2<Float>
        let groupVector: SIMD2<Float>
        let amplitude: Float
        let phase: Float
        let fadeStart: Float
        let fadeEnd: Float

        init(wavelength: Float, amplitude: Float, direction: Float, phase: Float, fadeStart: Float, fadeEnd: Float) {
            let vector = SIMD2(cos(direction), sin(direction)) * (2 * .pi / wavelength)
            let across = SIMD2(-vector.y, vector.x)
            waveVector = vector
            bendVector = across * 0.30 + vector * 0.10
            groupVector = vector * 0.24 - across * 0.13
            self.amplitude = amplitude
            self.phase = phase
            self.fadeStart = fadeStart
            self.fadeEnd = fadeEnd
        }
    }

    // Long swell crosses wind waves from a more oblique direction.
    private static let swells = [
        Wave(wavelength: 28.6, amplitude: 0.216, direction: 1.13, phase: 0.72, fadeStart: 185, fadeEnd: 430),
        Wave(wavelength: 19.3, amplitude: 0.120, direction: 0.74, phase: 3.01, fadeStart: 100, fadeEnd: 290),
        Wave(wavelength: 12.7, amplitude: 0.093, direction: 1.35, phase: 5.12, fadeStart: 75, fadeEnd: 161),
        Wave(wavelength: 8.4, amplitude: 0.071, direction: 1.54, phase: 2.04, fadeStart: 51, fadeEnd: 110),
        Wave(wavelength: 5.83, amplitude: 0.047, direction: 2.26, phase: 0.81, fadeStart: 37, fadeEnd: 75),
        Wave(wavelength: 4.43, amplitude: 0.037, direction: 1.88, phase: 3.28, fadeStart: 33, fadeEnd: 64),
        Wave(wavelength: 3.08, amplitude: 0.026, direction: 2.01, phase: 1.47, fadeStart: 28, fadeEnd: 53),
        Wave(wavelength: 2.34, amplitude: 0.019, direction: 2.47, phase: 4.19, fadeStart: 25, fadeEnd: 41),
        Wave(wavelength: 1.65, amplitude: 0.010, direction: 1.74, phase: 0.42, fadeStart: 19, fadeEnd: 32),
    ]

    static func make(device: any MTLDevice, iceberg: MeshGeometry) throws -> SceneNode {
        let material = try PhysicallyBasedMaterial(
            metallic: 0, roughness: 0.14, lightIntensity: 0, ambientIntensity: 0,
            normalTexture: .init(texture: try rippleNormals(device: device), textureCoordinateSet: .secondary),
            alphaMode: .opaque, transmission: .init(factor: 0, indexOfRefraction: 1.34),
            surfaceGraph: try surfaceGraph()
        )
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry(contact: Contact(iceberg: iceberg)),
            material: .physicallyBased(material)
        )
        return SceneNode(mesh: mesh)
    }

    /// Returns height in meters at world X and Z, relative to the mean waterline.
    static func waveHeight(at point: SIMD2<Float>) -> Float { swell(at: point).height }

    private static func swell(at point: SIMD2<Float>) -> (height: Float, gradient: SIMD2<Float>) {
        let local = point - SIMD2<Float>(3, 6)
        let radius = simd_length(local)
        let radialDirection = radius > 0 ? local / radius : .zero
        var height: Float = 0
        var gradient = SIMD2<Float>.zero
        for wave in swells where radius < wave.fadeEnd {
            let fadeSpan = wave.fadeEnd - wave.fadeStart
            let t = simd_clamp((radius - wave.fadeStart) / fadeSpan, 0, 1)
            let weight = 1 - t * t * (3 - 2 * t)
            let weightSlope = -6 * t * (1 - t) / fadeSpan
            let bend = simd_dot(point, wave.bendVector) + wave.phase * 1.41
            let group = simd_dot(point, wave.groupVector) - wave.phase * 0.67
            let phase = simd_dot(point, wave.waveVector) + wave.phase + 0.86 * sin(bend)
            let phaseGradient = wave.waveVector + (0.86 * cos(bend)) * wave.bendVector
            let envelope = 0.75 + 0.25 * cos(group)
            let envelopeGradient = (-0.25 * sin(group)) * wave.groupVector

            // The harmonic steepens individual crests. The envelope and bend break long periodic runs.
            let crest = wave.amplitude * (cos(phase) + 0.17 * cos(2 * phase))
            let crestSlope = -wave.amplitude * (sin(phase) + 0.34 * sin(2 * phase))
            let bandHeight = envelope * crest
            height += weight * bandHeight
            gradient += weight * (envelope * crestSlope * phaseGradient + crest * envelopeGradient)
            gradient += weightSlope * bandHeight * radialDirection
        }
        return (height, gradient)
    }

    /// Extracts the line where the iceberg crosses the displaced water. The water mesh carries
    /// the nearest distance in its primary U coordinate and its ripple coordinates separately.
    private struct Contact {
        private struct Edge {
            let start: SIMD2<Float>
            let delta: SIMD2<Float>
            let inverseLengthSquared: Float

            func distanceSquared(to point: SIMD2<Float>) -> Float {
                let offset = point - start
                let t = simd_clamp(simd_dot(offset, delta) * inverseLengthSquared, 0, 1)
                return simd_length_squared(offset - t * delta)
            }
        }

        private static let range: Float = 1.7
        private let edges: [Edge]
        private let lower: SIMD2<Float>
        private let upper: SIMD2<Float>

        init(iceberg: MeshGeometry) {
            let positions = iceberg.vertices.map { SIMD2($0.position.x, $0.position.z) }
            let offsets = zip(iceberg.vertices, positions).map { $0.position.y - LabIcebergOcean.waveHeight(at: $1) }
            let indices = iceberg.indices
            var edges: [Edge] = []
            var lower = SIMD2<Float>(repeating: .infinity)
            var upper = SIMD2<Float>(repeating: -.infinity)

            func crossing(_ a: Int, _ b: Int) -> SIMD2<Float>? {
                let first = offsets[a]
                let second = offsets[b]
                guard (first > 0) != (second > 0) else { return nil }
                return positions[a] + (positions[b] - positions[a]) * (first / (first - second))
            }

            for offset in stride(from: 0, to: indices.count, by: 3) {
                let a = Int(indices[offset])
                let b = Int(indices[offset + 1])
                let c = Int(indices[offset + 2])
                let above = offsets[a] > 0
                guard above != (offsets[b] > 0) || above != (offsets[c] > 0) else { continue }
                let crossings = [crossing(a, b), crossing(b, c), crossing(c, a)].compactMap { $0 }
                guard crossings.count == 2 else { continue }
                let delta = crossings[1] - crossings[0]
                let lengthSquared = simd_length_squared(delta)
                guard lengthSquared > 1e-8 else { continue }
                edges.append(Edge(start: crossings[0], delta: delta, inverseLengthSquared: 1 / lengthSquared))
                lower = simd_min(lower, simd_min(crossings[0], crossings[1]))
                upper = simd_max(upper, simd_max(crossings[0], crossings[1]))
            }
            self.edges = edges
            self.lower = lower - SIMD2(repeating: Self.range)
            self.upper = upper + SIMD2(repeating: Self.range)
        }

        func normalizedDistance(at point: SIMD2<Float>) -> Float {
            guard point.x >= lower.x, point.x <= upper.x, point.y >= lower.y, point.y <= upper.y else { return 1 }
            var nearest = Self.range * Self.range
            for edge in edges {
                nearest = min(nearest, edge.distanceSquared(to: point))
            }
            return sqrt(nearest) / Self.range
        }
    }

    private static func axis(from start: Float, to end: Float, segments: Int) -> [Float] {
        let extent: Float = 6_000
        let step = (end - start) / Float(segments)

        func outer(from edge: Float, direction: Float) -> [Float] {
            var coordinate = edge
            var stride = step * 1.12
            var points: [Float] = []
            while abs(coordinate) < extent {
                coordinate = direction * min(extent, abs(coordinate + direction * stride))
                points.append(coordinate)
                stride *= 1.18
            }
            return points
        }

        let center = (0...segments).map { index in start + Float(index) * step }
        return outer(from: start, direction: -1).reversed() + center + outer(from: end, direction: 1)
    }

    private static func geometry(contact: Contact) throws -> MeshGeometry {
        let xs = axis(from: -21, to: 29, segments: 125)
        let zs = axis(from: -18, to: 27, segments: 125)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(xs.count * zs.count)
        indices.reserveCapacity((xs.count - 1) * (zs.count - 1) * 6)
        for z in zs {
            for x in xs {
                let point = SIMD2(x, z)
                let wave = swell(at: point)
                let normal = simd_normalize(SIMD3(-wave.gradient.x, 1, -wave.gradient.y))
                vertices.append(MeshVertex(
                    position: SIMD3(x, wave.height, z), normal: normal,
                    textureCoordinate: SIMD2(contact.normalizedDistance(at: point), 0),
                    secondaryTextureCoordinate: point / 19
                ))
            }
        }
        for row in 0..<(zs.count - 1) {
            for column in 0..<(xs.count - 1) {
                let a = UInt32(row * xs.count + column)
                let c = a + UInt32(xs.count)
                if (row + column).isMultiple(of: 2) {
                    indices.append(contentsOf: [a, c, a + 1, a + 1, c, c + 1])
                } else {
                    indices.append(contentsOf: [a, c, c + 1, a, c + 1, a + 1])
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    private static func rippleNormals(device: any MTLDevice) throws -> any MTLTexture {
        let modes: [(direction: SIMD2<Float>, amplitude: Float, phase: Float)] = [
            (SIMD2(-8, 26), 0.090, 0.8), (SIMD2(-19, 30), 0.060, 2.6),
            (SIMD2(-17, 49), 0.034, 4.1), (SIMD2(-47, 62), 0.027, 1.2),
            (SIMD2(-21, 75), 0.014, 5.3), (SIMD2(-68, 91), 0.008, 3.4),
        ]
        let ripple = try ProceduralSurface(device: device, resolution: 512, heightScale: 0.0065) { uv in
            let turn = 2 * Float.pi
            // Integer mode vectors and periodic warps keep the larger tile seamless.
            let warp = SIMD2<Float>(
                0.010 * sin(turn * simd_dot(uv, SIMD2(3, 2))),
                0.009 * sin(turn * simd_dot(uv, SIMD2(2, -3)) + 1.28)
            )
            let gust = 0.68 + 0.32 * cos(turn * simd_dot(uv, SIMD2(2, 3)) + 0.72)
            let height = modes.reduce(Float(0.5)) { result, mode in
                result + gust * mode.amplitude * sin(turn * simd_dot(uv + warp, mode.direction) + mode.phase)
            }
            return .init(baseColor: .zero, height: height)
        }
        return ripple.normalTexture
    }

    private static func surfaceGraph() throws -> SurfaceShaderGraph {
        var nodes: [MaterialGraph.Node] = []
        func add(_ node: MaterialGraph.Node) -> Int {
            nodes.append(node)
            return nodes.count - 1
        }
        func scale(_ coordinate: Int, _ factor: Float) -> Int {
            add(.multiply(coordinate, add(.scalar(factor))))
        }
        func ramp(_ value: Int, _ lower: Float, _ upper: Float) -> Int {
            add(.smoothstep(edge0: add(.scalar(lower)), edge1: add(.scalar(upper)), value: value))
        }
        func complement(_ value: Int) -> Int { add(.subtract(add(.scalar(1)), value)) }

        let x = add(.worldPosition(.x))
        let y = add(.worldPosition(.y))
        let z = add(.worldPosition(.z))
        let windward = add(.add(x, scale(z, 0.54)))
        let acrossWind = add(.subtract(z, scale(x, 0.54)))
        let broad = add(.noise(x: scale(windward, 0.12), y: scale(y, 0.61), z: scale(acrossWind, 0.26)))
        let navy = add(.color(SIMD3(0.012, 0.028, 0.042)))
        let slate = add(.color(SIMD3(0.024, 0.059, 0.071)))
        let water = add(.mix(navy, slate, factor: broad))
        let distance = add(.uv(.u))
        let shallows = add(.multiply(complement(ramp(distance, 0.07, 0.52)), ramp(broad, 0.25, 0.76)))
        let nearIce = add(.color(SIMD3(0.030, 0.098, 0.096)))
        let waterColor = add(.mix(water, nearIce, factor: shallows))

        // Smaller waves leave broken foam lines at the ice edge. Isolated offshore flecks sit on higher crests.
        let gust = add(.noise(x: scale(windward, 0.39), y: scale(y, 1.30), z: scale(acrossWind, 0.95)))
        let lacing = add(.noise(x: scale(windward, 4.4), y: scale(y, 2.1), z: scale(acrossWind, 9.8)))
        let shoreline = complement(ramp(distance, 0.045, 0.27))
        let edgeLace = add(.multiply(ramp(gust, 0.34, 0.68), ramp(lacing, 0.28, 0.65)))
        let contact = add(.multiply(shoreline, edgeLace))
        let drift = complement(ramp(distance, 0.19, 0.56))
        let driftStreaks = add(.multiply(ramp(gust, 0.60, 0.79), ramp(lacing, 0.57, 0.78)))
        let wash = scale(add(.multiply(drift, driftStreaks)), 0.43)
        let height = ramp(y, 0.24, 0.43)
        let whitecaps = scale(add(.multiply(height, driftStreaks)), 0.80)
        let detachedFoam = add(.add(wash, whitecaps))
        let foamCoverage = add(.add(contact, detachedFoam))
        let foam = add(.clamp01(foamCoverage))
        let white = add(.color(SIMD3(0.71, 0.78, 0.76)))
        let color = add(.mix(waterColor, white, factor: foam))
        let sheltered = add(.scalar(0.090))
        let wind = add(.scalar(0.136))
        let ripple = add(.mix(sheltered, wind, factor: broad))
        let foamRoughness = add(.scalar(0.62))
        let roughness = add(.mix(ripple, foamRoughness, factor: foam))

        return try SurfaceShaderGraph(nodes: nodes, outputs: .init(baseColor: color, roughness: roughness))
    }
}
