import Frostfield
import Metal
import simd

/// A procedural Arctic seascape. Every surface and light is authored in code.
enum LabIcebergScene {
    private static let skyRadius: Float = 7_800
    private static let sunDirection = simd_normalize(SIMD3<Float>(-0.75, 0.49, 0.46))

    static func makeTracer(queue: any MTLCommandQueue, camera: PerspectiveCamera) throws -> PathTracer {
        let device = queue.device
        let ice = try iceMaterial(device: device)
        let iceGeometry = try LabIcebergGeometry.make()
        let iceberg = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: iceGeometry, material: .physicallyBased(ice)
        )
        let chips = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: .uvSphere(radius: 0.5, segments: 7, rings: 4), material: .physicallyBased(ice)
        )
        let sky = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: .uvSphere(radius: skyRadius, segments: 52, rings: 28),
            material: .physicallyBased(skyMaterial())
        )
        let chipAxis = simd_normalize(SIMD3<Float>(0.13, 0.98, 0.10))
        let calvedPieces: [(SIMD3<Float>, SIMD3<Float>, Float)] = [
            (SIMD3(5.1, -0.045, 1.45), SIMD3(0.73, 0.30, 0.47), 0.4),
            (SIMD3(5.58, -0.072, 2.40), SIMD3(0.35, 0.20, 0.26), -0.5),
            (SIMD3(-5.12, -0.085, 2.67), SIMD3(0.59, 0.32, 0.32), 0.8),
            (SIMD3(-4.15, -0.055, 3.91), SIMD3(0.24, 0.14, 0.27), -0.9),
            (SIMD3(3.34, -0.065, 4.12), SIMD3(0.31, 0.17, 0.19), 0.5),
            (SIMD3(-5.63, -0.030, -0.80), SIMD3(0.41, 0.21, 0.28), 1.4),
        ]
        var roots = [
            SceneNode(mesh: iceberg),
            try LabIcebergOcean.make(device: device, iceberg: iceGeometry),
            SceneNode(mesh: sky),
        ]
        roots += calvedPieces.map { location, scale, angle in
            let height = LabIcebergOcean.waveHeight(at: SIMD2(location.x, location.z))
            return SceneNode(transform: .init(
                translation: SIMD3(location.x, height + location.y, location.z),
                rotation: simd_quatf(angle: angle, axis: chipAxis), scale: scale
            ), mesh: chips)
        }
        let scene = try PathTracingScene(queue: queue, graph: SceneGraph(roots: roots, camera: camera), textureSize: 512)
        let daylight = try PathTracer.AreaLight(
            center: sunDirection * 120, aim: .zero, size: SIMD2(7.2, 5.7),
            radiance: SIMD3(465, 441, 399), cameraVisible: false
        )
        return try PathTracer(queue: queue, scene: scene, lights: [daylight])
    }

    private static func iceMaterial(device: any MTLDevice) throws -> PhysicallyBasedMaterial {
        let g = Graph()
        let x = g.node(.worldPosition(.x))
        let y = g.node(.worldPosition(.y))
        let z = g.node(.worldPosition(.z))
        let macro = g.noise(x, y, z, scale: SIMD3(0.73, 0.86, 0.89))
        let chips = g.noise(x, y, z, scale: SIMD3(3.9, 4.7, 4.4))
        let crystals = g.noise(x, y, z, scale: SIMD3(15, 19, 18))
        let exposed = g.noise(x, y, z, scale: SIMD3(5.5, 0.88, 5.8))
        let high = g.ramp(y, from: 1.0, to: 3.8)

        // Fold a sparse bedding field through the fractures so its lines vary in spacing and angle.
        let layer = g.add(g.scale(y, 9.6), g.add(g.scale(x, 2.05), g.scale(macro, 2.3)))
        let lamina = g.node(.noise(x: g.scale(x, 0.38), y: layer, z: g.scale(z, 0.54)))
        let bedding = g.ramp(lamina, from: 0.63, to: 0.80)
        let seams = g.ramp(g.add(exposed, g.scale(chips, 0.20)), from: 0.69, to: 0.89)
        let weathering = g.add(g.scale(macro, 0.47), g.add(g.scale(chips, 0.28), g.scale(crystals, 0.13)))
        let crust = g.ramp(g.add(weathering, g.scale(high, 0.22)), from: 0.37, to: 0.54)
        let clear = g.node(.multiply(seams, g.node(.subtract(g.scalar(1), crust))))
        let wet = g.node(.subtract(g.scalar(1), g.ramp(y, from: 0.06, to: 0.34)))

        let denseIce = g.mix(
            g.color(SIMD3(0.53, 0.70, 0.76)), g.color(SIMD3(0.77, 0.86, 0.88)), macro
        )
        let firn = g.mix(
            g.color(SIMD3(0.80, 0.85, 0.86)), g.color(SIMD3(0.97, 0.98, 0.96)), crystals
        )
        let blueSeams = g.mix(denseIce, g.color(SIMD3(0.38, 0.65, 0.72)), clear)
        let weathered = g.mix(blueSeams, firn, crust)
        let banded = g.mix(weathered, g.color(SIMD3(0.32, 0.40, 0.43)), g.scale(bedding, 0.22))
        let color = g.mix(banded, g.color(SIMD3(0.31, 0.49, 0.53)), g.scale(wet, 0.23))

        let matte = g.mix(g.scalar(0.48), g.scalar(0.77), crystals)
        let polished = g.mix(matte, g.scalar(0.22), clear)
        let roughness = g.mix(polished, g.scalar(0.31), g.scale(wet, 0.54))
        let bodyTransmission = g.mix(g.scalar(0.33), g.scalar(0.015), crust)
        let transmission = g.mix(bodyTransmission, g.scalar(0.98), clear)

        return try PhysicallyBasedMaterial(
            roughness: 0.48, lightIntensity: 0, ambientIntensity: 0,
            normalTexture: .init(texture: try iceRelief(device: device)), alphaMode: .opaque,
            transmission: .init(
                factor: 0.86, indexOfRefraction: 1.31, thickness: 1,
                attenuationDistance: 6.1, attenuationColor: SIMD3(0.49, 0.77, 0.87)
            ),
            surfaceGraph: .init(nodes: g.nodes, outputs: .init(
                baseColor: color, roughness: roughness, transmissionMask: transmission
            ))
        )
    }

    private static func iceRelief(device: any MTLDevice) throws -> any MTLTexture {
        let surface = try ProceduralSurface(device: device, resolution: 512, heightScale: 0.016) { uv in
            let fractured = tileNoise(uv, columns: 27, rows: 15, seed: 89)
            let granular = tileNoise(uv, columns: 65, rows: 81, seed: 317)
            let etched = tileNoise(uv, columns: 118, rows: 33, seed: 71)
            let ridge = 1 - abs(2 * etched - 1)
            return .init(baseColor: .zero, height: 0.26 * fractured + 0.19 * granular + 0.12 * ridge)
        }
        return surface.normalTexture
    }

    private static func tileNoise(_ uv: SIMD2<Float>, columns: Int, rows: Int, seed: UInt32) -> Float {
        let point = uv * SIMD2(Float(columns), Float(rows))
        let x = Int(point.x)
        let y = Int(point.y)
        let cell = point - SIMD2(Float(x), Float(y))
        let t = cell * cell * (3 - 2 * cell)
        func value(_ i: Int, _ j: Int) -> Float {
            var hash = UInt32(i % columns) &* 0x8da6b343
            hash ^= UInt32(j % rows) &* 0xd8163841
            hash ^= seed
            hash = (hash ^ (hash >> 16)) &* 0x7feb352d
            hash = (hash ^ (hash >> 15)) &* 0x846ca68b
            return Float(hash >> 8) / 16_777_216
        }
        let a = simd_mix(value(x, y), value(x + 1, y), t.x)
        let b = simd_mix(value(x, y + 1), value(x + 1, y + 1), t.x)
        return simd_mix(a, b, t.y)
    }

    private static func skyMaterial() throws -> PhysicallyBasedMaterial {
        let g = Graph()
        let x = g.scale(g.node(.worldPosition(.x)), 1 / skyRadius)
        let y = g.scale(g.node(.worldPosition(.y)), 1 / skyRadius)
        let z = g.scale(g.node(.worldPosition(.z)), 1 / skyRadius)
        let elevation = g.node(.clamp01(y))
        let air = g.mix(
            g.color(SIMD3(0.72, 0.79, 0.85)), g.color(SIMD3(0.21, 0.35, 0.53)),
            g.ramp(elevation, from: 0.018, to: 0.70)
        )

        // Evaluate cloud banks in angular coordinates. Near the horizon, a uniform haze
        // blends their contrast down instead of stretching the nearest texture cells.
        let coarse = g.noise(x, y, z, scale: SIMD3(4.1, 16, 4.1))
        let middle = g.noise(x, y, z, scale: SIMD3(11.1, 37, 10.7))
        let fine = g.noise(x, y, z, scale: SIMD3(31, 68, 29))
        let grain = g.noise(x, y, z, scale: SIMD3(70, 126, 74))
        let density = g.add(g.scale(coarse, 0.56), g.add(g.scale(middle, 0.24), g.add(
            g.scale(fine, 0.13), g.scale(grain, 0.07)
        )))
        let haze = g.ramp(elevation, from: 0.006, to: 0.14)
        let cover = g.node(.multiply(g.ramp(density, from: 0.31, to: 0.61), haze))
        let cloud = g.mix(
            g.color(SIMD3(0.43, 0.51, 0.61)), g.color(SIMD3(1.16, 1.18, 1.19)),
            g.ramp(density, from: 0.45, to: 0.78)
        )
        let weather = g.mix(air, cloud, cover)
        let sunCosine = g.add(g.scale(x, sunDirection.x), g.add(g.scale(y, sunDirection.y), g.scale(z, sunDirection.z)))
        let sunGlow = g.node(.multiply(g.color(SIMD3(0.47, 0.35, 0.19)), g.ramp(sunCosine, from: 0.90, to: 1)))
        let sky = g.add(weather, sunGlow)
        let radiance = g.mix(
            g.color(SIMD3(0.006, 0.020, 0.031)), sky, g.ramp(y, from: -0.007, to: 0.002)
        )

        return try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque, isDoubleSided: true,
            transmission: .init(factor: 0, indexOfRefraction: 1),
            surfaceGraph: .init(nodes: g.nodes, outputs: .init(emission: radiance))
        )
    }

    private final class Graph {
        private(set) var nodes: [MaterialGraph.Node] = []
        func node(_ value: MaterialGraph.Node) -> Int { nodes.append(value); return nodes.count - 1 }
        func scalar(_ value: Float) -> Int { node(.scalar(value)) }
        func color(_ value: SIMD3<Float>) -> Int { node(.color(value)) }
        func add(_ a: Int, _ b: Int) -> Int { node(.add(a, b)) }
        func scale(_ value: Int, _ amount: Float) -> Int { node(.multiply(value, scalar(amount))) }
        func mix(_ a: Int, _ b: Int, _ factor: Int) -> Int { node(.mix(a, b, factor: factor)) }

        func ramp(_ value: Int, from lower: Float, to upper: Float) -> Int {
            node(.smoothstep(edge0: scalar(lower), edge1: scalar(upper), value: value))
        }

        func noise(_ x: Int, _ y: Int, _ z: Int, scale factor: SIMD3<Float>) -> Int {
            node(.noise(x: scale(x, factor.x), y: scale(y, factor.y), z: scale(z, factor.z)))
        }
    }
}
