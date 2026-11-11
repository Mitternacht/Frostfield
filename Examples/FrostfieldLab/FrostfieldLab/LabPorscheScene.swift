import CoreGraphics
import CoreText
import Foundation
import Frostfield
import Metal
import simd

/// A studio portrait of a code-built 992.2 GT3. Dimensions are in meters.
enum LabPorscheScene {
    static func makeTracer(queue: any MTLCommandQueue, camera: PerspectiveCamera) throws -> PathTracer {
        let device = queue.device
        let paint = try bodyPaint()
        var nodes = try LabPorscheBody.nodes(device: device, paint: paint)
        nodes += try LabPorscheWheels.nodes(device: device)
        nodes += try LabPorscheTrim.nodes(device: device, paint: paint)
        nodes += try cabin(device: device)
        nodes += try registrationPlate(device: device)
        nodes.append(try studio(device: device))

        let scene = try PathTracingScene(queue: queue, graph: SceneGraph(roots: nodes, camera: camera), textureSize: 512)
        // Long studio banks pull continuous highlights along the crown and door.
        let lamps: [PathTracer.AreaLight] = try [
            .init(center: SIMD3(-0.53, 4.1, 0.16), halfU: SIMD3(0, 0, 3.25), halfV: SIMD3(-0.59, 0, 0),
                  radiance: SIMD3(5.6, 5.36, 5.03), cameraVisible: false),
            .init(center: SIMD3(4.45, 2.16, -0.47), halfU: SIMD3(0, 0, 3.36), halfV: SIMD3(0, 0.68, 0),
                  radiance: SIMD3(4.28, 4.66, 5.04), cameraVisible: false),
            .init(center: SIMD3(-2.7, 2.2, 4.75), aim: SIMD3(0, 0.51, 1.49), size: SIMD2(3.3, 2.1),
                  radiance: SIMD3(4.45, 4.28, 4.04), cameraVisible: false),
            .init(center: SIMD3(-3.8, 3.48, -3.78), aim: SIMD3(0, 0.81, -0.4), size: SIMD2(4.1, 0.92),
                  radiance: SIMD3(6.7, 7.15, 7.7), cameraVisible: false),
            .init(center: SIMD3(3.65, 2.40, 6.3), aim: SIMD3(0.4, 0.4, 0.7), size: SIMD2(3.65, 1.75),
                  radiance: SIMD3(1.70, 1.77, 1.86), cameraVisible: false),
            .init(center: SIMD3(-2.0, 4.0, -4.3), aim: SIMD3(-7.2, 1.6, -11.1), size: SIMD2(5.8, 2.5),
                  radiance: SIMD3(5.1, 5.5, 6.5), cameraVisible: false),
        ]
        return try PathTracer(queue: queue, scene: scene, lights: lamps, environment: SIMD3(0.16, 0.176, 0.194))
    }

    private static func bodyPaint() throws -> PhysicallyBasedMaterial {
        let finish = try SurfaceShaderGraph(nodes: [
            .worldPosition(.x), .worldPosition(.y), .worldPosition(.z),
            .scalar(263), .multiply(0, 3), .multiply(1, 3), .multiply(2, 3),
            .noise(x: 4, y: 5, z: 6), .scalar(0.53), .scalar(0.92),
            .smoothstep(edge0: 8, edge1: 9, value: 7),
            .color(SIMD3(0.021, 0.076, 0.172)), .color(SIMD3(0.036, 0.113, 0.217)),
            .mix(11, 12, factor: 10), .scalar(0.245), .scalar(0.287), .mix(14, 15, factor: 7),
        ], outputs: .init(baseColor: 13, roughness: 16))
        return try PhysicallyBasedMaterial(
            metallic: 0.54, roughness: 0.27, lightIntensity: 0, ambientIntensity: 0,
            alphaMode: .opaque, clearcoat: .init(factor: 0.82, roughness: 0.115), surfaceGraph: finish
        )
    }

    private static func studio(device: any MTLDevice) throws -> SceneNode {
        let concrete = try SurfaceShaderGraph(nodes: [
            .worldPosition(.x), .worldPosition(.z), .scalar(0), .scalar(3.0),
            .multiply(0, 3), .multiply(1, 3), .noise(x: 4, y: 2, z: 5),
            .scalar(125), .multiply(0, 7), .multiply(1, 7), .noise(x: 8, y: 2, z: 9),
            .scalar(0.79), .scalar(0.21), .multiply(6, 11), .multiply(10, 12), .add(13, 14),
            .color(SIMD3(0.049, 0.053, 0.061)), .color(SIMD3(0.071, 0.075, 0.084)), .mix(16, 17, factor: 15),
            .scalar(0.31), .scalar(0.43), .mix(19, 20, factor: 15),
        ], outputs: .init(baseColor: 18, roughness: 21))
        let surface = try PhysicallyBasedMaterial(
            metallic: 0.15, roughness: 0.38, lightIntensity: 0, ambientIntensity: 0,
            alphaMode: .opaque, surfaceGraph: concrete
        )
        return try mesh(device, geometry: cyclorama(), material: surface)
    }

    private static func cyclorama() throws -> MeshGeometry {
        struct Ring {
            let radius: Float
            let height: Float
            let angle: Float
        }
        let steps = 112
        var rings = [Ring(radius: 0.02, height: 0, angle: 0), Ring(radius: 10, height: 0, angle: 0)]
        let bendSteps = 15
        for index in 1...bendSteps {
            let angle = Float(index) / Float(bendSteps) * (.pi / 2)
            rings.append(Ring(radius: 10 + 5.8 * sin(angle), height: 5.8 * (1 - cos(angle)), angle: angle))
        }
        rings.append(Ring(radius: 15.8, height: 11, angle: .pi / 2))
        var vertices = [MeshVertex(position: .zero, normal: SIMD3(0, 1, 0))]
        var indices: [UInt32] = []
        for ring in rings {
            for step in 0..<steps {
                let angle = 2 * Float.pi * Float(step) / Float(steps)
                let x = cos(angle)
                let z = sin(angle)
                vertices.append(MeshVertex(
                    position: SIMD3(x * ring.radius, ring.height, z * ring.radius),
                    normal: SIMD3(-x * sin(ring.angle), cos(ring.angle), -z * sin(ring.angle))
                ))
            }
        }
        for step in 0..<steps {
            let next = (step + 1) % steps
            indices += [0, UInt32(1 + next), UInt32(1 + step)]
            for ring in 0..<(rings.count - 1) {
                let a = UInt32(1 + ring * steps + step)
                let b = UInt32(1 + ring * steps + next)
                let c = a + UInt32(steps)
                let d = b + UInt32(steps)
                indices += [a, b, c, b, d, c]
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    private static func registrationPlate(device: any MTLDevice) throws -> [SceneNode] {
        let centerY: Float = 0.402
        let nose = LabPorscheBody.frontFasciaZ(x: 0, y: centerY)
        let holder = try trim(SIMD3(0.010, 0.011, 0.013), roughness: 0.34)
        let face = try PhysicallyBasedMaterial(
            baseColorTexture: plateTexture(device: device), roughness: 0.42,
            lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque
        )
        return try [
            mesh(device, geometry: .bevelBox(size: SIMD3(0.431, 0.091, 0.025), bevel: 0.006, segments: 3),
                 material: holder, at: SIMD3(0, centerY, nose + 0.003)),
            mesh(device, geometry: .plane(width: 0.419, height: 0.080),
                 material: face, at: SIMD3(0, centerY, nose + 0.016)),
        ]
    }

    private static func plateTexture(device: any MTLDevice) throws -> any MTLTexture {
        let width = 512
        let height = 104
        let rowBytes = width * 4
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm_srgb, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw MeshRenderer.Failure.resource("registration plate texture")
        }

        var texels = [UInt8](repeating: 0, count: rowBytes * height)
        try texels.withUnsafeMutableBytes { pixels in
            guard let address = pixels.baseAddress, let context = CGContext(
                data: address, width: width, height: height, bitsPerComponent: 8, bytesPerRow: rowBytes,
                space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { throw MeshRenderer.Failure.resource("registration plate drawing") }

            func ink(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat) -> CGColor {
                CGColor(srgbRed: red, green: green, blue: blue, alpha: 1)
            }
            func lettering(_ value: String, size: CGFloat, center: CGFloat, baseline: CGFloat, color: CGColor) {
                let font = CTFontCreateWithName("DINAlternate-Bold" as CFString, size, nil)
                let text = NSAttributedString(string: value, attributes: [
                    NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font,
                    NSAttributedString.Key(rawValue: kCTForegroundColorAttributeName as String): color,
                ])
                let line = CTLineCreateWithAttributedString(text)
                let advance = CTLineGetTypographicBounds(line, nil, nil, nil)
                context.textPosition = CGPoint(x: center - advance / 2, y: baseline)
                CTLineDraw(line, context)
            }
            context.setFillColor(ink(0.93, 0.937, 0.94))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.setFillColor(ink(0.06, 0.20, 0.48))
            context.fill(CGRect(x: 0, y: 0, width: 49, height: height))
            context.setFillColor(ink(0.92, 0.74, 0.19))
            for step in 0..<12 {
                let angle = Double(step) * 2 * .pi / 12
                context.fillEllipse(in: CGRect(x: 22 + 12 * cos(angle), y: 68 + 12 * sin(angle), width: 3.1, height: 3.1))
            }
            lettering("D", size: 25, center: 24.5, baseline: 23, color: ink(0.96, 0.96, 0.96))
            lettering("S  FF 911", size: 70, center: 278, baseline: 24, color: ink(0.042, 0.049, 0.055))
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: address, bytesPerRow: rowBytes
            )
        }
        texture.label = "Procedural Porsche registration plate"
        return texture
    }

    private static func cabin(device: any MTLDevice) throws -> [SceneNode] {
        let leather = try trim(SIMD3(0.032, 0.036, 0.044), roughness: 0.52)
        let fabric = try trim(SIMD3(0.069, 0.073, 0.084), roughness: 0.79)
        let pianoBlack = try trim(SIMD3(0.008, 0.010, 0.014), roughness: 0.24)
        let brushed = try trim(SIMD3(0.18, 0.19, 0.21), roughness: 0.34, metallic: 0.70)
        let liner = try trim(SIMD3(0.016, 0.018, 0.022), roughness: 0.75)
        func box(_ size: SIMD3<Float>, at point: SIMD3<Float>, radius: Float,
                 surface: PhysicallyBasedMaterial, tilt: Float = 0) throws -> SceneNode {
            try mesh(device, geometry: .bevelBox(size: size, bevel: radius, segments: 2), material: surface,
                     at: point, rotation: simd_quatf(angle: tilt, axis: SIMD3(1, 0, 0)))
        }
        var nodes = try [
            box(SIMD3(1.38, 0.09, 1.44), at: SIMD3(0, 0.40, -0.02), radius: 0.042, surface: liner),
            box(SIMD3(1.31, 0.145, 0.39), at: SIMD3(0, 0.756, 0.72), radius: 0.068, surface: leather, tilt: -0.12),
            box(SIMD3(0.24, 0.18, 0.73), at: SIMD3(0, 0.54, 0.22), radius: 0.054, surface: pianoBlack, tilt: 0.14),
            box(SIMD3(0.27, 0.093, 0.024), at: SIMD3(0, 0.786, 0.517), radius: 0.011, surface: pianoBlack, tilt: -0.22),
        ]
        for side: Float in [-1, 1] {
            let x = side * 0.34
            nodes += try [
                box(SIMD3(0.44, 0.12, 0.51), at: SIMD3(x, 0.487, 0.044), radius: 0.057, surface: leather),
                box(SIMD3(0.26, 0.07, 0.41), at: SIMD3(x, 0.536, 0.056), radius: 0.032, surface: fabric),
                box(SIMD3(0.43, 0.39, 0.15), at: SIMD3(x, 0.695, -0.206), radius: 0.071, surface: leather, tilt: -0.17),
                box(SIMD3(0.25, 0.37, 0.09), at: SIMD3(x, 0.709, -0.136), radius: 0.041, surface: fabric, tilt: -0.17),
                box(SIMD3(0.23, 0.225, 0.12), at: SIMD3(x, 0.978, -0.276), radius: 0.053, surface: leather, tilt: -0.12),
                box(SIMD3(0.08, 0.275, 0.13), at: SIMD3(x + 0.155, 0.687, -0.145), radius: 0.039, surface: leather, tilt: -0.16),
                box(SIMD3(0.08, 0.275, 0.13), at: SIMD3(x - 0.155, 0.687, -0.145), radius: 0.039, surface: leather, tilt: -0.16),
                box(SIMD3(0.078, 0.24, 0.94), at: SIMD3(side * 0.674, 0.68, 0.03), radius: 0.035, surface: liner),
            ]
        }
        let wheelCenter = SIMD3<Float>(-0.34, 0.773, 0.452)
        let wheelAngle: Float = -0.43
        let wheelRotation = simd_quatf(angle: wheelAngle, axis: SIMD3(1, 0, 0))
        let rimPoints = (0...56).map { index -> SIMD3<Float> in
            let angle = 2 * Float.pi * Float(index) / 56
            return wheelCenter + wheelRotation.act(SIMD3(0.159 * cos(angle), 0.157 * sin(angle), 0))
        }
        nodes.append(try mesh(device, geometry: .tube(along: rimPoints, radius: 0.013,
                                                      radialSegments: 10, capped: false), material: leather))
        nodes += try [
            box(SIMD3(0.12, 0.086, 0.053), at: wheelCenter, radius: 0.024, surface: pianoBlack, tilt: wheelAngle),
            box(SIMD3(0.26, 0.039, 0.024), at: wheelCenter, radius: 0.011, surface: brushed, tilt: wheelAngle),
            box(SIMD3(0.036, 0.117, 0.025), at: wheelCenter + wheelRotation.act(SIMD3(0, -0.079, 0)),
                radius: 0.012, surface: brushed, tilt: wheelAngle),
        ]
        return nodes
    }

    private static func trim(_ color: SIMD3<Float>, roughness: Float, metallic: Float = 0) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(metallic: metallic, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
                                    baseColorFactor: SIMD4(color, 1), alphaMode: .opaque)
    }

    private static func mesh(
        _ device: any MTLDevice, geometry: MeshGeometry, material: PhysicallyBasedMaterial,
        at point: SIMD3<Float> = .zero, rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    ) throws -> SceneNode {
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float,
                                        geometry: geometry, material: .physicallyBased(material))
        return SceneNode(transform: .init(translation: point, rotation: rotation), mesh: renderer)
    }
}
