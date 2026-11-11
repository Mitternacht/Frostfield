import Foundation
import Frostfield
import Metal
import simd

/// A rangefinder built from dimensions and functions. No model or image files are loaded.
struct LabProceduralCamera {
    let device: any MTLDevice
    let light: SIMD3<Float>
    let intensity: Float
    let environment: PhysicallyBasedMaterial.Environment?
    var generatedFinish = false
    var bakedGrip = false

    private static let faceForward = simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))

    func graph(camera: PerspectiveCamera) throws -> SceneGraph {
        let body: PhysicallyBasedMaterial
        if generatedFinish {
            body = try shaderFinish()
        } else {
            body = try ProceduralSurface(
                device: device, style: .leatherette, resolution: 512, seed: 35, baseColor: SIMD3(0.033, 0.024, 0.019)
            ).material(directionToLight: light, lightIntensity: intensity, ambientIntensity: 0.18,
                       receivesShadows: true, environment: environment, normalScale: 0.55)
        }
        let alloy = try ProceduralSurface(
            device: device, style: .brushedMetal, resolution: 512, seed: 90, baseColor: SIMD3(0.62, 0.56, 0.47)
        ).material(directionToLight: light, lightIntensity: intensity, ambientIntensity: 0.18,
                   receivesShadows: true, environment: environment, normalScale: 0.52)
        let charcoal = try material(SIMD3(0.032, 0.038, 0.041), metallic: 0.82, roughness: 0.35)
        let black = try material(SIMD3(0.013, 0.016, 0.019), roughness: 0.61)
        let polished = try material(SIMD3(0.73, 0.68, 0.60), metallic: 1, roughness: 0.18)
        let copper = try material(SIMD3(0.67, 0.22, 0.09), metallic: 0.91, roughness: 0.29)
        let lettering = try material(SIMD3(0.34, 0.20, 0.095), metallic: 0.1, roughness: 0.5)
        let modelMark = try material(SIMD3(0.29, 0.32, 0.32), roughness: 0.59)
        let window = try material(
            SIMD3(0.023, 0.086, 0.094), metallic: 0.57, roughness: 0.19,
            clearcoat: .init(factor: 0.88, roughness: 0.16)
        )

        var nodes = [
            try box(SIMD3(2.96, 1.22, 0.80), bevel: 0.117, at: SIMD3(0, 0.72, 0), repeats: SIMD2(2, 1), using: body),
            try box(SIMD3(2.96, 0.33, 0.84), bevel: 0.067, at: SIMD3(0, 1.39, 0), using: alloy),
            try box(SIMD3(2.96, 0.15, 0.85), bevel: 0.045, at: SIMD3(0, 0.106, 0), using: alloy),
            try box(SIMD3(0.53, 0.20, 0.070), bevel: 0.032, at: SIMD3(-1.025, 1.393, 0.423), using: charcoal),
            try box(SIMD3(0.426, 0.122, 0.024), bevel: 0.011, at: SIMD3(-1.025, 1.395, 0.462), using: window),
            try box(SIMD3(0.34, 0.17, 0.058), bevel: 0.027, at: SIMD3(-0.42, 1.396, 0.426), using: charcoal),
            try box(SIMD3(0.262, 0.105, 0.026), bevel: 0.010, at: SIMD3(-0.42, 1.397, 0.457), using: window),
            try box(SIMD3(0.16, 0.26, 0.055), bevel: 0.023, at: SIMD3(1.11, 0.65, 0.414), using: charcoal),
            try box(SIMD3(0.47, 0.048, 0.36), bevel: 0.015, at: SIMD3(0.075, 1.568, -0.055), using: charcoal),
        ]
        for side: Float in [-1, 1] {
            nodes.append(try box(
                SIMD3(0.062, 0.037, 0.34), bevel: 0.012,
                at: SIMD3(0.075 + side * 0.18, 1.603, -0.055), using: alloy
            ))
        }
        nodes += try rear(leather: body, charcoal: charcoal, black: black, window: window, alloy: alloy)
        nodes += try optics(alloy: alloy, black: black, charcoal: charcoal, polished: polished, copper: copper)
        nodes += try topControls(alloy: alloy, black: black, charcoal: charcoal, polished: polished, copper: copper)
        nodes += try lugs(alloy: polished, cord: copper)
        nodes.append(try mark("FROSTFIELD", at: SIMD3(0.065, 1.344, 0.421), pixel: 0.017, using: lettering))
        nodes.append(try mark("R-35", at: SIMD3(0.760, 0.281, 0.402), pixel: 0.022, using: modelMark))
        return SceneGraph(roots: nodes, camera: camera)
    }

    private func shaderFinish() throws -> PhysicallyBasedMaterial {
        var nodes: [MaterialGraph.Node] = []
        func node(_ operation: MaterialGraph.Node) -> Int {
            nodes.append(operation)
            return nodes.count - 1
        }

        let frequency = node(.parameter(.x))
        let u = node(.multiply(node(.uv(.u)), node(.scalar(64))))
        let v = node(.multiply(node(.uv(.v)), node(.scalar(285))))
        let z = node(.multiply(node(.worldPosition(.z)), node(.scalar(37))))
        let scratch = node(.noise(x: node(.multiply(u, frequency)), y: node(.multiply(v, frequency)), z: z))
        let finish = node(.mix(
            node(.color(SIMD3(0.027, 0.084, 0.123))),
            node(.color(SIMD3(0.128, 0.257, 0.311))), factor: scratch
        ))
        let roughness = node(.mix(node(.scalar(0.24)), node(.scalar(0.49)), factor: scratch))
        let metallic = node(.scalar(0.76))
        let graph = try SurfaceShaderGraph(nodes: nodes, outputs: .init(baseColor: finish, metallic: metallic, roughness: roughness))
        return try PhysicallyBasedMaterial(
            directionToLight: light, lightIntensity: intensity, ambientIntensity: 0.18, receivesShadows: true,
            alphaMode: .opaque, environment: environment,
            clearcoat: .init(factor: 0.30, roughness: 0.19), surfaceGraph: graph
        )
    }

    private func rear(
        leather: PhysicallyBasedMaterial, charcoal: PhysicallyBasedMaterial,
        black: PhysicallyBasedMaterial, window: PhysicallyBasedMaterial, alloy: PhysicallyBasedMaterial
    ) throws -> [SceneNode] {
        let faceBack = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        return [
            try box(SIMD3(2.62, 1.03, 0.050), bevel: 0.022, at: SIMD3(0.019, 0.68, -0.399), using: black),
            try box(SIMD3(2.52, 0.93, 0.032), bevel: 0.014, at: SIMD3(0.019, 0.68, -0.420),
                    repeats: SIMD2(2, 1), using: leather),
            try box(SIMD3(0.51, 0.23, 0.088), bevel: 0.041, at: SIMD3(-1.018, 1.392, -0.433), using: black),
            try box(SIMD3(0.38, 0.134, 0.024), bevel: 0.011, at: SIMD3(-1.018, 1.394, -0.480), using: window),
            try box(SIMD3(0.55, 0.34, 0.043), bevel: 0.019, at: SIMD3(0.22, 0.69, -0.437), using: charcoal),
            try box(SIMD3(0.42, 0.214, 0.022), bevel: 0.010, at: SIMD3(0.22, 0.69, -0.460), using: alloy),
            try lathe([(0.085, 0), (0.105, 0.015), (0.105, 0.038), (0.091, 0.052)],
                      at: SIMD3(1.156, 1.20, -0.375), rotation: faceBack * Self.faceForward, using: alloy),
        ]
    }

    private func optics(
        alloy: PhysicallyBasedMaterial, black: PhysicallyBasedMaterial,
        charcoal: PhysicallyBasedMaterial, polished: PhysicallyBasedMaterial, copper: PhysicallyBasedMaterial
    ) throws -> [SceneNode] {
        let x: Float = -0.31
        let y: Float = 0.71
        let front = Self.faceForward
        let glass = try material(
            SIMD3(0.55, 0.82, 0.79), roughness: 0.115,
            transmission: .init(
                factor: 0.95, indexOfRefraction: 1.53, thickness: 0.075,
                attenuationDistance: 0.22, attenuationColor: SIMD3(0.48, 0.80, 0.85)
            )
        )
        let sensor = try ProceduralSurface(device: device, resolution: 256, heightScale: 0.002) { uv in
            let position = uv - SIMD2(repeating: 0.5)
            let radius = simd_length(position)
            let angle = atan2(position.y, position.x)
            let sector = Float.pi / 4
            let bladeAngle = angle - floor(angle / sector + 0.5) * sector
            let aperture = 0.134 / cos(bladeAngle)
            let iris = min(max((radius - aperture) * 150, 0), 1)
            let groove = 0.5 + 0.5 * cos(radius * 228 + bladeAngle * 3)
            let glassColor = SIMD3<Float>(0.07, 0.144, 0.14)
            let bladeColor = SIMD3<Float>(0.041, 0.060, 0.085) * (0.69 + 0.31 * groove)
            return .init(
                baseColor: glassColor * (1 - iris) + bladeColor * iris,
                height: 0.38 + 0.08 * groove * iris, roughness: 0.24 + 0.10 * iris, metallic: 0.76
            )
        }.material(directionToLight: light, lightIntensity: intensity, ambientIntensity: 0.18,
                   receivesShadows: true, environment: environment)
        let gripRadius: Float = 0.574
        let gripDepth: Float = 0.021
        let gripGeometry = try knurl(radius: gripRadius, height: 0.228, teeth: 72, depth: gripDepth)
        let grip = bakedGrip
            ? try LabAssetBaking.grip(
                gripGeometry, radius: gripRadius, depth: gripDepth, device: device,
                light: light, intensity: intensity, environment: environment
            )
            : (gripGeometry, charcoal)
        return [
            try lathe([(0.595, 0), (0.649, 0.015), (0.660, 0.030), (0.660, 0.058), (0.647, 0.074), (0.606, 0.092)],
                      at: SIMD3(x, y, 0.396), rotation: front, using: alloy),
            try lathe([(0.594, 0), (0.611, 0.030), (0.611, 0.071), (0.575, 0.106), (0.571, 0.405), (0.53, 0.455)],
                      at: SIMD3(x, y, 0.484), rotation: front, using: charcoal),
            try node(grip.0, at: SIMD3(x, y, 0.599), rotation: front, using: grip.1),
            try lathe([(0.574, 0), (0.579, 0.003), (0.579, 0.018), (0.574, 0.022)],
                      at: SIMD3(x, y, 0.827), rotation: front, using: copper, closed: false),
            try lathe([(0.526, 0), (0.563, 0.016), (0.565, 0.096), (0.555, 0.112), (0.534, 0.120), (0.503, 0.120)],
                      at: SIMD3(x, y, 0.928), rotation: front, using: polished, closed: false),
            try lathe([(0.514, 0), (0.514, 0.032), (0.488, 0.046), (0.472, 0.050), (0.462, 0.056), (0.449, 0.061), (0.430, 0.070)],
                      at: SIMD3(x, y, 0.969), rotation: front, using: black, closed: false),
            try lathe([(0.443, 0), (0.443, 0.013)], at: SIMD3(x, y, 0.945), rotation: front, using: sensor),
            try lathe([(0.426, 0), (0.414, 0.015), (0.380, 0.028), (0.312, 0.040), (0.212, 0.050), (0.102, 0.057), (0.012, 0.060)],
                      at: SIMD3(x, y, 0.982), rotation: front, using: glass),
        ]
    }

    private func topControls(
        alloy: PhysicallyBasedMaterial, black: PhysicallyBasedMaterial, charcoal: PhysicallyBasedMaterial,
        polished: PhysicallyBasedMaterial, copper: PhysicallyBasedMaterial
    ) throws -> [SceneNode] {
        var nodes: [SceneNode] = []
        for x: Float in [-1.03, 1.03] {
            let z: Float = x < 0 ? -0.042 : 0.032
            nodes += [
                try lathe([(0.237, 0), (0.252, 0.025), (0.252, 0.044), (0.225, 0.054)],
                          at: SIMD3(x, 1.533, z), using: black),
                try lathe([(0.223, 0), (0.241, 0.021), (0.240, 0.112), (0.220, 0.135), (0.207, 0.139)],
                          at: SIMD3(x, 1.583, z), using: alloy),
                try node(knurl(radius: 0.238, height: 0.087, teeth: 46, depth: 0.016),
                         at: SIMD3(x, 1.603, z), using: polished),
                try lathe([(0.182, 0), (0.183, 0.014), (0.175, 0.020)],
                          at: SIMD3(x, 1.715, z), using: charcoal),
            ]
        }
        let lever = try MeshGeometry.extrude(contour: [
            SIMD2(-0.162, -0.037), SIMD2(0.073, -0.040), SIMD2(0.108, -0.068),
            SIMD2(0.192, -0.051), SIMD2(0.210, -0.007), SIMD2(0.184, 0.050),
            SIMD2(0.099, 0.060), SIMD2(0.050, 0.037), SIMD2(-0.162, 0.040),
        ], depth: 0.036)
        let leverRotation = simd_quatf(angle: -0.34, axis: SIMD3(0, 1, 0)) * Self.faceForward.inverse
        nodes += [
            try lathe([(0.092, 0), (0.098, 0.012), (0.096, 0.041), (0.087, 0.051)],
                      at: SIMD3(1.03, 1.732, 0.032), using: copper),
            try node(lever, at: SIMD3(-0.961, 1.748, -0.042), rotation: leverRotation, using: polished),
            try lathe([(0.070, 0), (0.075, 0.015), (0.070, 0.040)],
                      at: SIMD3(0.622, 1.551, -0.16), using: charcoal),
        ]
        return nodes
    }

    private func lugs(alloy: PhysicallyBasedMaterial, cord: PhysicallyBasedMaterial) throws -> [SceneNode] {
        let start = SIMD3<Float>(-1.46, 1.01, 0.08)
        let turn = SIMD3<Float>(-1.75, 0.88, 0.105)
        let end = SIMD3<Float>(-1.46, 0.79, 0.08)
        let first = try ProceduralPath.cubicBezier(
            from: start, control1: SIMD3(-1.56, 1.06, 0.092), control2: SIMD3(-1.73, 1.01, 0.11), to: turn, segments: 14
        )
        let second = try ProceduralPath.cubicBezier(
            from: turn, control1: SIMD3(-1.766, 0.777, 0.101), control2: SIMD3(-1.55, 0.727, 0.093), to: end, segments: 14
        )
        let loop = try MeshGeometry.tube(along: first + second.dropFirst(), radius: 0.021)
        return [
            try box(SIMD3(0.14, 0.30, 0.16), bevel: 0.036, at: SIMD3(-1.468, 0.900, 0.08), using: alloy),
            try node(loop, using: cord),
        ]
    }

    private func knurl(radius: Float, height: Float, teeth: Int, depth: Float) throws -> MeshGeometry {
        try MeshGeometry.parametricSurface(uSegments: teeth * 6, vSegments: 8, wrapU: true) { uv in
            let angle = -2 * Float.pi * uv.x
            let ridge = 0.5 + 0.5 * cos(Float(teeth) * angle)
            let rim = min(1, min(uv.y, 1 - uv.y) * 9)
            let r = radius + depth * ridge * rim
            return SIMD3(r * cos(angle), height * uv.y, r * sin(angle))
        }
    }

    private func material(
        _ color: SIMD3<Float>, metallic: Float = 0, roughness: Float,
        clearcoat: PhysicallyBasedMaterial.Clearcoat? = nil, transmission: PhysicallyBasedMaterial.Transmission? = nil
    ) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            metallic: metallic, roughness: roughness, directionToLight: light, lightIntensity: intensity,
            ambientIntensity: 0.18, receivesShadows: true, baseColorFactor: SIMD4(color, 1),
            alphaMode: .opaque, environment: environment, clearcoat: clearcoat, transmission: transmission
        )
    }

    private func box(
        _ size: SIMD3<Float>, bevel: Float, at position: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(real: 1, imag: .zero), repeats: SIMD2<Float> = SIMD2(repeating: 1),
        using material: PhysicallyBasedMaterial
    ) throws -> SceneNode {
        var geometry = try MeshGeometry.bevelBox(size: size, bevel: bevel, segments: 5)
        if repeats != SIMD2(repeating: 1) {
            geometry = try MeshGeometry(vertices: geometry.vertices.map { vertex in
                MeshVertex(
                    position: vertex.position, normal: vertex.normal,
                    textureCoordinate: vertex.textureCoordinate * repeats, tangent: vertex.tangent
                )
            }, indices: geometry.indices)
        }
        return try node(geometry, at: position, rotation: rotation, using: material)
    }

    private func lathe(
        _ profile: [(radius: Float, y: Float)], at position: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(real: 1, imag: .zero),
        using material: PhysicallyBasedMaterial, closed: Bool = true, top: Bool = false
    ) throws -> SceneNode {
        let geometry = try MeshGeometry.lathe(
            profile: profile.map { LatheProfilePoint(radius: $0.radius, y: $0.y) },
            segments: 96, capBottom: closed, capTop: closed || top
        )
        return try node(geometry, at: position, rotation: rotation, using: material)
    }

    private func node(
        _ geometry: MeshGeometry, at position: SIMD3<Float> = .zero,
        rotation: simd_quatf = simd_quatf(real: 1, imag: .zero), using material: PhysicallyBasedMaterial,
        castsShadow: Bool = true
    ) throws -> SceneNode {
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        return SceneNode(transform: .init(translation: position, rotation: rotation), mesh: renderer, castsShadow: castsShadow)
    }

    // Pixel strokes become geometry, so the face carries no image or font dependency.
    private static let glyphs: [Character: [UInt8]] = [
        "F": [31, 16, 16, 30, 16, 16, 16], "R": [30, 17, 17, 30, 20, 18, 17],
        "O": [14, 17, 17, 17, 17, 17, 14], "S": [15, 16, 16, 14, 1, 1, 30],
        "T": [31, 4, 4, 4, 4, 4, 4], "I": [14, 4, 4, 4, 4, 4, 14],
        "E": [31, 16, 16, 30, 16, 16, 31], "L": [16, 16, 16, 16, 16, 16, 31],
        "D": [30, 17, 17, 17, 17, 17, 30], "-": [0, 0, 0, 14, 0, 0, 0],
        "3": [30, 1, 1, 14, 1, 1, 30], "5": [31, 16, 16, 30, 1, 1, 30],
    ]

    private func mark(
        _ text: String, at position: SIMD3<Float>, pixel: Float, using material: PhysicallyBasedMaterial
    ) throws -> SceneNode {
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for (letter, character) in text.enumerated() {
            guard let rows = Self.glyphs[character] else { continue }
            for (row, bits) in rows.enumerated() {
                for column in 0..<5 where bits & (1 << (4 - column)) != 0 {
                    let x = Float(letter * 6 + column) * pixel
                    let y = Float(6 - row) * pixel
                    let start = UInt32(vertices.count)
                    let corners: [SIMD3<Float>] = [
                        SIMD3(x, y, 0), SIMD3(x + pixel, y, 0),
                        SIMD3(x + pixel, y + pixel, 0), SIMD3(x, y + pixel, 0),
                    ]
                    vertices.append(contentsOf: corners.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) })
                    indices += [start, start + 1, start + 2, start, start + 2, start + 3]
                }
            }
        }
        return try node(.init(vertices: vertices, indices: indices), at: position, using: material, castsShadow: false)
    }
}
