import Foundation
import Frostfield
import Metal
import simd

enum LabPorscheTrim {
    private static let lightCenter = SIMD2<Float>(0.669, 1.813)
    private static let lightRadii = SIMD2<Float>(0.151, 0.225)
    private static let centerIntake = rounded([
        SIMD2(-0.552, 0.165), SIMD2(0.552, 0.165), SIMD2(0.474, 0.353), SIMD2(-0.474, 0.353),
    ])
    private static let sideIntake = rounded([
        SIMD2(0.566, 0.166), SIMD2(0.819, 0.167), SIMD2(0.836, 0.205), SIMD2(0.826, 0.425),
        SIMD2(0.797, 0.452), SIMD2(0.515, 0.426), SIMD2(0.464, 0.392), SIMD2(0.487, 0.328),
    ])
    // Outlets are described in X/Z, unlike the apron openings in X/Y.
    private static let hoodDuct = rounded([
        SIMD2(0.124, 2.054), SIMD2(0.403, 2.052), SIMD2(0.432, 2.076),
        SIMD2(0.128, 2.100), SIMD2(0.115, 2.086),
    ])

    static func headlightContains(x: Float, z: Float) -> Bool {
        let rise = z - lightCenter.y
        let across = abs(x) - lightCenter.x + 0.12 * rise
        return simd_length_squared(SIMD2(across, rise) / lightRadii) < 0.96 * 0.96
    }

    static func frontIntakeContains(x: Float, y: Float) -> Bool {
        contains(SIMD2(x, y), in: centerIntake) || contains(SIMD2(abs(x), y), in: sideIntake)
    }

    static func hoodDuctContains(x: Float, z: Float) -> Bool {
        contains(SIMD2(abs(x), z), in: hoodDuct)
    }

    private static func rounded(_ corners: [SIMD2<Float>]) -> [SIMD2<Float>] {
        corners.indices.flatMap { index in
            let a = corners[index]
            let b = corners[(index + 1) % corners.count]
            return [a + (b - a) * 0.13, b + (a - b) * 0.13]
        }
    }

    private static func contains(_ p: SIMD2<Float>, in loop: [SIMD2<Float>]) -> Bool {
        loop.indices.allSatisfy { index in
            let start = loop[index]
            let edge = loop[(index + 1) % loop.count] - start
            let d = p - start
            return edge.x * d.y - edge.y * d.x >= 0
        }
    }

    static func nodes(device: any MTLDevice, paint: PhysicallyBasedMaterial) throws -> [SceneNode] {
        var parts = Parts()
        for side: Float in [-1, 1] {
            try headlight(side: side, parts: &parts)
            try intake(sideIntake, side: side, slats: true, parts: &parts)
            try hoodOutlet(side: side, parts: &parts)
            try mirror(side: side, parts: &parts)
            try wingMount(side: side, parts: &parts)
            try rocker(side: side, parts: &parts)
        }
        try intake(centerIntake, side: 1, slats: false, parts: &parts)
        try splitter(parts: &parts)
        try wing(parts: &parts)
        try tailLamp(parts: &parts)
        try crest(parts: &parts)
        return try parts.nodes(device: device, paint: paint)
    }

    private static func headlight(side: Float, parts: inout Parts) throws {
        func position(_ unit: SIMD2<Float>, offset: Float) -> SIMD3<Float> {
            let z = lightCenter.y + unit.y * lightRadii.y
            let x = side * (lightCenter.x + unit.x * lightRadii.x - 0.12 * (z - lightCenter.y))
            return SIMD3(x, LabPorscheBody.frontDeckY(x: x, z: z) + offset, z)
        }
        func profile(_ a: SIMD2<Float>, _ b: SIMD2<Float>) throws -> MeshGeometry {
            try .parametricSurface(uSegments: 64, vSegments: 3, wrapU: true) { uv in
                let angle = 2 * Float.pi * uv.x * side
                let section = a + (b - a) * uv.y
                return position(SIMD2(cos(angle), sin(angle)) * section.x, offset: section.y)
            }
        }

        // The rubber seat covers the panel seam and the graphite surround curves into the housing.
        parts.add(try profile(SIMD2(0.93, 0.004), SIMD2(1.054, 0.002)), .rubber)
        parts.add(try profile(SIMD2(0.919, 0.002), SIMD2(0.971, 0.005)), .projector)
        parts.add(try profile(SIMD2(0.60, -0.060), SIMD2(0.928, 0.003)), .projector)
        parts.add(try profile(SIMD2(0.008, -0.065), SIMD2(0.605, -0.059)), .carbon)

        func oval(center: SIMD2<Float>, radii: SIMD2<Float>, depth: Float, rise: Float) throws -> MeshGeometry {
            try .parametricSurface(uSegments: 32, vSegments: 6, wrapU: true) { uv in
                let angle = 2 * Float.pi * uv.x * side
                let radius = 0.009 + 0.991 * uv.y
                let local = center + radius * radii * SIMD2(cos(angle), sin(angle))
                return position(local, offset: depth + rise * (1 - radius * radius))
            }
        }

        // Four distinct optic packages form the 992.2 daytime signature.
        for row: Float in [-1, 1] {
            for column: Float in [-1, 1] {
                let center = SIMD2(column * (row > 0 ? 0.390 : 0.360), row * 0.350)
                parts.add(try oval(center: center, radii: SIMD2(0.242, 0.164), depth: -0.046, rise: 0.015), .alloy)
                parts.add(try oval(center: center, radii: SIMD2(0.215, 0.139), depth: -0.031, rise: 0.011), .projector)
                let emitterCenter = center + SIMD2(0, row * 0.016)
                parts.add(try oval(center: emitterCenter, radii: SIMD2(0.156, 0.026), depth: -0.019, rise: 0.002), .led)
            }
        }
        parts.add(try oval(center: .zero, radii: SIMD2(0.228, 0.251), depth: -0.060, rise: 0.024), .projector)

        let cover = try MeshGeometry.parametricSurface(uSegments: 80, vSegments: 14, wrapU: true) { uv in
            let angle = 2 * Float.pi * uv.x * side
            let radius = 0.004 + 0.931 * uv.y
            return position(
                radius * SIMD2(cos(angle), sin(angle)),
                offset: 0.005 + 0.009 * (1 - radius * radius / (0.935 * 0.935))
            )
        }
        parts.add(cover, .cover)
    }

    private static func intake(
        _ contour: [SIMD2<Float>], side: Float, slats: Bool, parts: inout Parts
    ) throws {
        let midpoint = contour.reduce(.zero, +) / Float(contour.count)
        let front = SIMD3<Float>(0, 0, 1)
        func position(_ point: SIMD2<Float>, depth: Float) -> SIMD3<Float> {
            let x = side * point.x
            return SIMD3(x, point.y, LabPorscheBody.frontFasciaZ(x: x, y: point.y) + depth)
        }
        func outline(_ scale: Float, _ depth: Float) -> [SIMD3<Float>] {
            contour.map { position(midpoint + ($0 - midpoint) * scale, depth: depth) }
        }

        parts.add(try band(outer: outline(1.092, 0.004), inner: outline(1.005, 0.006), facing: front), .carbon)
        parts.add(try band(outer: outline(1.005, 0.006), inner: outline(0.87, -0.066), facing: front), .rubber)
        parts.add(try face(outline(1.018, -0.090), facing: front), .rubber)

        let aperture = contour.map { midpoint + ($0 - midpoint) * 0.885 }
        let xMin = aperture.map(\.x).min()!
        let xMax = aperture.map(\.x).max()!
        let yMin = aperture.map(\.y).min()!
        let yMax = aperture.map(\.y).max()!
        let pitch: Float = slats ? 0.0115 : 0.0107
        let across = sqrt(Float(3)) * pitch
        let rise = pitch * 1.5
        let hexagon = (0..<6).map { i -> SIMD2<Float> in
            let angle = Float(i) * (.pi / 3) + .pi / 6
            return SIMD2(cos(angle), sin(angle))
        }

        var grille = Faces()
        for row in 0...Int((yMax - yMin) / rise) {
            let y = yMin + Float(row) * rise
            for column in 0...Int((xMax - xMin) / across) {
                let x = xMin + (Float(column) + (row.isMultiple(of: 2) ? 0 : 0.5)) * across
                let center = SIMD2(x, y)
                let outer = hexagon.map { center + $0 * (pitch * 0.965) }
                guard outer.allSatisfy({ contains($0, in: aperture) }) else { continue }
                let inner = hexagon.map { center + $0 * (pitch * 0.77) }
                grille.band(
                    outer: outer.map { position($0, depth: -0.051) },
                    inner: inner.map { position($0, depth: -0.051) }, facing: front
                )
            }
        }
        parts.add(try grille.geometry(), .grille)

        if slats {
            for y: Float in [0.245, 0.318, 0.386] {
                let interval = horizontalSpan(of: aperture, at: y)
                guard let interval, interval.upperBound - interval.lowerBound > 0.04 else { continue }
                let vane = try MeshGeometry.parametricSurface(uSegments: 22, vSegments: 4) { uv in
                    let t = side > 0 ? uv.x : 1 - uv.x
                    let x = interval.lowerBound + (interval.upperBound - interval.lowerBound) * t
                    let lift = 0.012 * sin(.pi * t)
                    let height = y + (uv.y - 0.5) * (0.016 + lift)
                    let profile = 0.008 * sin(.pi * uv.y) - 0.012 * uv.y
                    return position(SIMD2(x, height), depth: -0.027 + profile)
                }
                parts.add(vane, .carbon)
            }
        } else {
            for x: Float in [-0.337, 0, 0.337] {
                let y: Float = 0.254
                parts.add(try box(
                    size: SIMD3(0.008, 0.153, 0.010), bevel: 0.002,
                    at: position(SIMD2(x, y), depth: -0.046)
                ), .carbon)
            }
        }
    }

    private static func horizontalSpan(of loop: [SIMD2<Float>], at y: Float) -> ClosedRange<Float>? {
        let intersections = loop.indices.compactMap { index -> Float? in
            let a = loop[index]
            let b = loop[(index + 1) % loop.count]
            guard (a.y <= y) != (b.y <= y) else { return nil }
            return a.x + (b.x - a.x) * ((y - a.y) / (b.y - a.y))
        }
        guard let low = intersections.min(), let high = intersections.max() else { return nil }
        return low...high
    }

    private static func hoodOutlet(side: Float, parts: inout Parts) throws {
        let center = hoodDuct.reduce(.zero, +) / Float(hoodDuct.count)
        func outline(scale: Float, height: Float) -> [SIMD3<Float>] {
            hoodDuct.map { point in
                let p = center + (point - center) * scale
                return SIMD3(side * p.x, LabPorscheBody.frontDeckY(x: side * p.x, z: p.y) + height, p.y)
            }
        }
        parts.add(try band(outer: outline(scale: 1.11, height: 0.002),
                           inner: outline(scale: 0.85, height: -0.014), facing: SIMD3(0, 1, 0)), .carbon)
        parts.add(try face(outline(scale: 0.87, height: -0.019), facing: SIMD3(0, 1, 0)), .rubber)
    }

    private static func splitter(parts: inout Parts) throws {
        let lip = try MeshGeometry.parametricSurface(uSegments: 100, vSegments: 24, wrapV: true) { uv in
            let x = (uv.x - 0.5) * 1.773
            let flank = pow(abs(x) / 0.887, 6)
            let angle = 2 * Float.pi * uv.y
            let y = 0.147 + 0.029 * flank + 0.014 * sin(angle)
            let nose = LabPorscheBody.frontFasciaZ(x: x, y: 0.165)
            let z = nose - 0.045 - 0.14 * flank + (0.077 + 0.048 * flank) * cos(angle)
            return SIMD3(x, y, z)
        }
        parts.add(lip, .carbon)
    }

    private static func rocker(side: Float, parts: inout Parts) throws {
        let sill = try MeshGeometry.parametricSurface(uSegments: 62, vSegments: 16, wrapV: true) { uv in
            let lengthwise = 2 * uv.x - 1
            let ends = pow(abs(lengthwise), 2.4)
            let angle = 2 * Float.pi * uv.y
            let x = side * (0.754 + 0.035 * ends + 0.015 * cos(angle))
            let y = 0.138 + 0.006 * ends - side * 0.012 * sin(angle)
            let z = 0.007 + 0.864 * lengthwise
            return SIMD3(x, y, z)
        }
        parts.add(sill, .carbon)
    }

    private static func mirror(side: Float, parts: inout Parts) throws {
        let center = SIMD3<Float>(side * 0.948, 0.956, 0.475)
        let mount = try MeshGeometry.tube(
            along: [SIMD3(side * 0.703, 0.903, 0.424), SIMD3(side * 0.79, 0.919, 0.467),
                    SIMD3(side * 0.869, 0.946, 0.489)],
            radius: 0.018, radialSegments: 16
        )
        parts.add(mount, .carbon)

        func point(angle: Float, r: Float, depth: Float) -> SIMD3<Float> {
            let horizontal = cos(angle)
            let vertical = sin(angle)
            let x = side * (0.119 * r * horizontal)
            let y = r * vertical * 0.048 * (1 - 0.14 * horizontal) + 0.010 * horizontal * r
            let z = depth + 0.029 * r * horizontal + 0.014 * r * vertical
            return center + SIMD3(x, y, z)
        }

        let housing = try MeshGeometry.parametricSurface(uSegments: 56, vSegments: 16, wrapU: true) { uv in
            let angle = -2 * Float.pi * uv.x * side
            let t = uv.y
            let radius = 0.014 + 0.986 * sin(t * 0.5 * .pi)
            let depth = 0.071 - 0.145 * t * t
            return point(angle: angle, r: radius, depth: depth)
        }
        parts.add(housing, .paint)
        let bezel = try MeshGeometry.parametricSurface(uSegments: 56, vSegments: 3, wrapU: true) { uv in
            point(angle: 2 * .pi * uv.x * side, r: 0.861 + 0.146 * uv.y, depth: -0.077 + 0.004 * uv.y)
        }
        parts.add(bezel, .carbon)
        let glass = try MeshGeometry.parametricSurface(uSegments: 56, vSegments: 8, wrapU: true) { uv in
            let r = 0.008 + 0.86 * uv.y
            return point(angle: 2 * .pi * uv.x * side, r: r, depth: -0.078 - 0.004 * (1 - r * r))
        }
        parts.add(glass, .mirror)
    }

    private static func wingMount(side: Float, parts: inout Parts) throws {
        let x = side * 0.442
        let base = LabPorscheBody.rearDeckY(x: x, z: -1.660) - 0.010
        let silhouette: [SIMD2<Float>] = [
            SIMD2(-1.663, base), SIMD2(-1.605, base + 0.006), SIMD2(-1.768, 1.184),
            SIMD2(-1.824, 1.274), SIMD2(-1.901, 1.312), SIMD2(-1.997, 1.306),
            SIMD2(-2.022, 1.271), SIMD2(-1.961, 1.259), SIMD2(-1.914, 1.274),
            SIMD2(-1.863, 1.247), SIMD2(-1.825, 1.173), SIMD2(-1.723, base + 0.005),
        ]
        let support = try MeshGeometry.extrude(contour: silhouette, depth: 0.021)
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))
        parts.add(try moved(support, to: SIMD3(x, 0, 0), rotation: rotation), .carbon)
        let footing = try box(
            size: SIMD3(0.067, 0.018, 0.153), bevel: 0.008,
            at: SIMD3(x, base + 0.017, -1.679)
        )
        parts.add(footing, .rubber)
        parts.add(try box(size: SIMD3(0.036, 0.034, 0.079), bevel: 0.006,
                          at: SIMD3(x, 1.255, -1.960)), .carbon)
    }

    private static func wing(parts: inout Parts) throws {
        let wing = try MeshGeometry.parametricSurface(uSegments: 100, vSegments: 38, wrapV: true) { uv in
            let x = (uv.x - 0.5) * 1.665
            let tip = pow(abs(x) / 0.833, 3)
            let angle = 2 * Float.pi * uv.y
            let chord = 0.151 - 0.012 * tip
            let camber = 0.024 * cos(angle)
            let y = 1.224 + 0.017 * tip - camber + 0.017 * sin(angle) * (1 + 0.36 * cos(angle))
            let z = -1.951 - 0.035 * tip + chord * cos(angle)
            return SIMD3(x, y, z)
        }
        parts.add(wing, .paint)
        let tip: [SIMD2<Float>] = rounded([
            SIMD2(-2.168, 1.175), SIMD2(-2.125, 1.178), SIMD2(-1.841, 1.199),
            SIMD2(-1.827, 1.267), SIMD2(-2.126, 1.307), SIMD2(-2.172, 1.286),
        ])
        let endPlate = try MeshGeometry.extrude(contour: tip, depth: 0.014)
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))
        for side: Float in [-1, 1] {
            parts.add(try moved(endPlate, to: SIMD3(side * 0.831, 0, 0), rotation: rotation), .carbon)
        }
    }

    private static func tailLamp(parts: inout Parts) throws {
        func bar(width: Float, offset: Float, height: Float, lift: Float) throws -> MeshGeometry {
            try .parametricSurface(uSegments: 100, vSegments: 5) { uv in
                let x = (0.5 - uv.x) * width
                let ends = pow(abs(x) / 0.835, 3)
                let y = 0.640 + offset + 0.021 * ends + height * (uv.y - 0.5)
                return SIMD3(x, y, LabPorscheBody.rearFasciaZ(x: x, y: y) - lift)
            }
        }
        parts.add(try bar(width: 1.695, offset: 0, height: 0.067, lift: 0.004), .projector)
        parts.add(try bar(width: 1.667, offset: 0.013, height: 0.012, lift: 0.007), .red)
    }

    private static func crest(parts: inout Parts) throws {
        let shield: [SIMD2<Float>] = [
            SIMD2(-0.022, -0.025), SIMD2(0.022, -0.025), SIMD2(0.021, -0.007), SIMD2(0.017, 0.010),
            SIMD2(0.008, 0.020), SIMD2(0, 0.026), SIMD2(-0.008, 0.020), SIMD2(-0.017, 0.010), SIMD2(-0.021, -0.007),
        ]
        func mount(_ points: [SIMD2<Float>], scale: Float = 1, lift: Float = 0.006) -> [SIMD3<Float>] {
            points.map { p in
                let x = p.x * scale
                let z = 2.149 + p.y * scale
                return SIMD3(x, LabPorscheBody.frontDeckY(x: x, z: z) + lift, z)
            }
        }
        let up = SIMD3<Float>(0, 1, 0)
        parts.add(try face(mount(shield, scale: 1.055, lift: 0.005), facing: up), .rubber)
        parts.add(try face(mount(shield), facing: up), .gold)
        for y: Float in [-0.008, 0.004] {
            for side: Float in [-1, 1] {
                let x = side * 0.012
                parts.add(try face(mount([
                    SIMD2(x - 0.006, y - 0.003), SIMD2(x + 0.006, y - 0.003),
                    SIMD2(x + 0.006, y + 0.001), SIMD2(x - 0.006, y + 0.001),
                ], lift: 0.0067), facing: up), y < 0 ? .badgeRed : .projector)
            }
        }
        parts.add(try face(mount([
            SIMD2(-0.005, 0.009), SIMD2(0.003, 0.007), SIMD2(0.005, -0.002),
            SIMD2(0.002, -0.006), SIMD2(0.004, -0.014), SIMD2(0.001, -0.015),
            SIMD2(-0.003, -0.009), SIMD2(-0.005, -0.005),
        ], lift: 0.0067), facing: up), .projector)
    }

    private static func box(size: SIMD3<Float>, bevel: Float, at center: SIMD3<Float>) throws -> MeshGeometry {
        try moved(.bevelBox(size: size, bevel: bevel, segments: 3), to: center)
    }

    private static func moved(
        _ geometry: MeshGeometry, to center: SIMD3<Float>,
        rotation: simd_quatf = simd_quatf(real: 1, imag: .zero)
    ) throws -> MeshGeometry {
        try MeshGeometry(vertices: geometry.vertices.map { v in
            MeshVertex(position: center + rotation.act(v.position), normal: rotation.act(v.normal),
                       textureCoordinate: v.textureCoordinate)
        }, indices: geometry.indices)
    }

    private static func face(_ points: [SIMD3<Float>], facing: SIMD3<Float>) throws -> MeshGeometry {
        var faces = Faces()
        let center = points.reduce(.zero, +) / Float(points.count)
        for index in points.indices {
            faces.triangle(center, points[index], points[(index + 1) % points.count], facing: facing)
        }
        return try faces.geometry()
    }

    private static func band(
        outer: [SIMD3<Float>], inner: [SIMD3<Float>], facing: SIMD3<Float>
    ) throws -> MeshGeometry {
        var faces = Faces()
        faces.band(outer: outer, inner: inner, facing: facing)
        return try faces.geometry()
    }

    private struct Faces {
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []

        mutating func triangle(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, facing: SIMD3<Float>) {
            let normal = simd_cross(b - a, c - a)
            guard simd_length_squared(normal) > 1e-20 else { return }
            let reversed = simd_dot(normal, facing) < 0
            let n = simd_normalize(reversed ? -normal : normal)
            let base = UInt32(vertices.count)
            vertices.append(MeshVertex(position: a, normal: n))
            vertices.append(MeshVertex(position: reversed ? c : b, normal: n))
            vertices.append(MeshVertex(position: reversed ? b : c, normal: n))
            indices += [base, base + 1, base + 2]
        }

        mutating func band(outer: [SIMD3<Float>], inner: [SIMD3<Float>], facing: SIMD3<Float>) {
            for index in outer.indices {
                let next = (index + 1) % outer.count
                triangle(outer[index], outer[next], inner[index], facing: facing)
                triangle(inner[index], outer[next], inner[next], facing: facing)
            }
        }

        func geometry() throws -> MeshGeometry { try MeshGeometry(vertices: vertices, indices: indices) }
    }

    private enum Finish: CaseIterable {
        case paint, rubber, carbon, grille, alloy, projector, cover, led, mirror, red, gold, badgeRed

        func material(paint: PhysicallyBasedMaterial) throws -> PhysicallyBasedMaterial {
            func material(
                _ color: SIMD3<Float>, _ roughness: Float, metal: Float = 0, emission: SIMD3<Float> = .zero,
                transmission: PhysicallyBasedMaterial.Transmission? = nil
            ) throws -> PhysicallyBasedMaterial {
                try PhysicallyBasedMaterial(
                    metallic: metal, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
                    baseColorFactor: SIMD4(color, 1), emissiveFactor: emission, alphaMode: .opaque,
                    transmission: transmission
                )
            }
            return switch self {
            case .paint: paint
            case .rubber: try material(SIMD3(0.0055, 0.0064, 0.0075), 0.55)
            case .carbon: try material(SIMD3(0.018, 0.022, 0.027), 0.28, metal: 0.16)
            case .grille: try material(SIMD3(0.032, 0.034, 0.037), 0.39, metal: 0.48)
            case .alloy: try material(SIMD3(0.16, 0.20, 0.27), 0.25, metal: 0.61)
            case .projector: try material(SIMD3(0.011, 0.017, 0.024), 0.27, metal: 0.21)
            case .cover: try material(SIMD3(0.985, 0.995, 1), 0.019,
                                      transmission: .init(factor: 1, indexOfRefraction: 1.23))
            case .led: try material(SIMD3(0.64, 0.79, 0.91), 0.17, emission: SIMD3(0.76, 0.90, 1.0))
            case .mirror: try material(SIMD3(0.67, 0.72, 0.80), 0.051, metal: 0.98)
            case .red: try material(SIMD3(0.40, 0.012, 0.015), 0.20, emission: SIMD3(0.62, 0.017, 0.020))
            case .gold: try material(SIMD3(0.64, 0.34, 0.070), 0.29, metal: 0.65)
            case .badgeRed: try material(SIMD3(0.36, 0.014, 0.015), 0.28, metal: 0.12)
            }
        }
    }

    private struct Parts {
        var meshes: [Finish: [MeshGeometry]] = [:]

        mutating func add(_ mesh: MeshGeometry, _ finish: Finish) {
            meshes[finish, default: []].append(mesh)
        }

        func nodes(device: any MTLDevice, paint: PhysicallyBasedMaterial) throws -> [SceneNode] {
            try Finish.allCases.compactMap { finish in
                guard let components = meshes[finish] else { return nil }
                var vertices: [MeshVertex] = []
                var indices: [UInt32] = []
                for component in components {
                    let offset = UInt32(vertices.count)
                    vertices += component.vertices
                    indices += component.indices.map { $0 + offset }
                }
                let geometry = try MeshGeometry(vertices: vertices, indices: indices)
                let renderer = try MeshRenderer(
                    device: device, colorPixelFormat: .rgba16Float,
                    geometry: geometry, material: .physicallyBased(finish.material(paint: paint))
                )
                return SceneNode(mesh: renderer)
            }
        }
    }
}
