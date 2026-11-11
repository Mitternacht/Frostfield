import Frostfield
import Metal
import simd

/// GT3 wheel assemblies, modeled in meters around the local X axle.
/// Porsche MY2025 specifications: 255/35 ZR 20 and 315/30 ZR 21,
/// with 408 mm and 380 mm cast iron brake discs.
enum LabPorscheWheels {
    private typealias V2 = SIMD2<Float>
    private typealias V3 = SIMD3<Float>
    private typealias Profile = [SIMD2<Float>]
    private static let turn: Float = 2 * .pi

    private struct Axle {
        let z: Float
        let radius: Float
        let halfWidth: Float
        let rim: Float
        let disc: Float
        let caliperAngle: Float
        let tireSize: String

        var height: Float { radius - 0.003 }
        var lip: Float { halfWidth - 0.017 }
        var discFace: Float { lip - 0.059 }
    }

    static func nodes(device: any MTLDevice) throws -> [SceneNode] {
        let axles = [
            Axle(z: 1.2275, radius: 0.332, halfWidth: 0.132, rim: 0.254, disc: 0.204,
                 caliperAngle: -1.22, tireSize: "255/35 ZR 20"),
            Axle(z: -1.2275, radius: 0.348, halfWidth: 0.161, rim: 0.2667, disc: 0.190,
                 caliperAngle: 1.28, tireSize: "315/30 ZR 21"),
        ]
        let rubber = try finish(V3(0.024, 0.026, 0.029), roughness: 0.68)
        let embossedRubber = try finish(V3(0.042, 0.044, 0.047), roughness: 0.52)
        let silver = try finish(V3(0.64, 0.65, 0.67), metallic: 0.91, roughness: 0.245)
        let barrel = try finish(V3(0.19, 0.20, 0.22), metallic: 0.78, roughness: 0.33)
        let lock = try finish(V3(0.056, 0.060, 0.069), metallic: 0.62, roughness: 0.29)
        let iron = try rotorFinish()
        let red = try finish(V3(0.49, 0.015, 0.009), roughness: 0.29, coat: 0.43)
        let ink = try finish(V3(0.74, 0.75, 0.73), roughness: 0.43)

        return try axles.flatMap { axle in
            let profile = tireProfile(axle)
            let parts = try [
                renderer(tire(axle, profile: profile), rubber, device: device),
                renderer(tireLettering(axle, profile: profile), embossedRubber, device: device),
                renderer(forging(axle), silver, device: device),
                renderer(wheelBarrel(axle), barrel, device: device),
                renderer(disc(axle), iron, device: device),
                renderer(hardware(axle), lock, device: device),
                renderer(caliper(axle), red, device: device),
                renderer(inkDetails(axle), ink, device: device),
            ]
            return [-1.0 as Float, 1].map { side in
                SceneNode(
                    transform: .init(
                        translation: V3(side * 0.793, axle.height, axle.z),
                        rotation: simd_quatf(angle: side > 0 ? 0 : .pi, axis: V3(0, 1, 0))
                    ),
                    children: parts.map { SceneNode(mesh: $0) }
                )
            }
        }
    }

    private static func renderer(_ mesh: Mesh, _ material: PhysicallyBasedMaterial, device: any MTLDevice) throws -> MeshRenderer {
        try MeshRenderer(device: device, colorPixelFormat: .rgba16Float,
                         geometry: MeshGeometry(vertices: mesh.vertices, indices: mesh.indices),
                         material: .physicallyBased(material))
    }

    private static func finish(
        _ color: V3, metallic: Float = 0, roughness: Float, coat: Float = 0
    ) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            metallic: metallic, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(color, 1), alphaMode: .opaque,
            clearcoat: coat > 0 ? .init(factor: coat, roughness: 0.22) : nil
        )
    }

    private static func rotorFinish() throws -> PhysicallyBasedMaterial {
        let graph = try SurfaceShaderGraph(nodes: [
            .uv(.u), .scalar(420), .multiply(0, 1), .cosine(2), .scalar(0.5),
            .multiply(3, 4), .add(5, 4),
            .color(V3(0.20, 0.204, 0.207)), .color(V3(0.30, 0.303, 0.307)), .mix(7, 8, factor: 6),
            .scalar(0.27), .scalar(0.39), .mix(10, 11, factor: 6),
        ], outputs: .init(baseColor: 9, roughness: 12))
        return try PhysicallyBasedMaterial(
            metallic: 0.75, roughness: 0.34, lightIntensity: 0, ambientIntensity: 0,
            alphaMode: .opaque, surfaceGraph: graph
        )
    }

    private static func tireProfile(_ axle: Axle) -> Profile {
        let w = axle.halfWidth
        let r = axle.radius
        let bead = axle.rim
        let shoulder: Profile = [
            V2(0.80 * w, bead - 0.004), V2(0.84 * w, bead + 0.005),
            V2(0.895 * w, bead + 0.014), V2(0.968 * w, bead + 0.021),
            V2(0.992 * w, bead + 0.036), V2(w, r - 0.032),
            V2(0.991 * w, r - 0.021), V2(0.966 * w, r - 0.014),
            V2(0.904 * w, r - 0.008), V2(0.815 * w, r - 0.0043),
            V2(0.726 * w, r - 0.0018),
        ]
        // Four longitudinal channels and staggered transverse shoulder cuts.
        let crown: Profile = [
            V2(0.58 * w, r - 0.0008), V2(0.566 * w, r - 0.0047),
            V2(0.507 * w, r - 0.0045), V2(0.486 * w, r - 0.0006),
            V2(0.23 * w, r), V2(0.211 * w, r - 0.0045),
            V2(0.162 * w, r - 0.0045), V2(0.14 * w, r),
            V2(-0.14 * w, r), V2(-0.162 * w, r - 0.0045),
            V2(-0.211 * w, r - 0.0045), V2(-0.23 * w, r),
            V2(-0.486 * w, r - 0.0006), V2(-0.507 * w, r - 0.0045),
            V2(-0.566 * w, r - 0.0047), V2(-0.58 * w, r - 0.0008),
        ]
        return shoulder + crown + shoulder.reversed().map { V2(-$0.x, $0.y) }
    }

    private static func tire(_ axle: Axle, profile: Profile) -> Mesh {
        var mesh = Mesh()
        mesh.lathe(profile, segments: 168, ground: axle.height) { angle, point in
            let lateral = abs(point.x) / axle.halfWidth
            let shoulder = smooth(lateral, from: 0.55, to: 0.72)
            let onTread = smooth(point.y, from: axle.radius - 0.021, to: axle.radius - 0.005)
            let step = 49 * angle / turn + lateral * 1.55 + (point.x < 0 ? 0.34 : 0)
            let groove = 1 - smooth(abs(step - round(step)), from: 0.045, to: 0.20)
            return -0.004 * shoulder * onTread * groove
        }
        return mesh
    }

    private static func forging(_ axle: Axle) -> Mesh {
        let lip = axle.lip
        let r = axle.rim
        var mesh = Mesh()
        mesh.lathe([
            V2(lip - 0.008, r - 0.017), V2(lip - 0.001, r - 0.015), V2(lip + 0.003, r - 0.010),
            V2(lip + 0.004, r - 0.003), V2(lip + 0.001, r + 0.005), V2(lip - 0.005, r + 0.008),
            V2(lip - 0.019, r + 0.007), V2(lip - 0.026, r - 0.004), V2(lip - 0.019, r - 0.015),
        ], segments: 104)
        mesh.lathe([
            V2(lip - 0.031, 0.030), V2(lip - 0.028, 0.034), V2(lip - 0.028, 0.047),
            V2(lip - 0.034, 0.056), V2(lip - 0.044, 0.070), V2(lip - 0.060, 0.069),
            V2(lip - 0.064, 0.051), V2(lip - 0.055, 0.030),
        ], segments: 56)

        // Five paired spokes share broad roots, then diverge into narrow swept ends.
        let stations: [(radius: Float, axial: Float, offset: Float, width: Float, depth: Float)] = [
            (0.050, lip - 0.052, 0.11, 0.031, 0.020),
            (0.068, lip - 0.047, 0.14, 0.030, 0.022),
            (0.092, lip - 0.042, 0.19, 0.024, 0.025),
            (0.125, lip - 0.034, 0.237, 0.019, 0.025),
            (0.169, lip - 0.026, 0.255, 0.017, 0.025),
            (0.206, lip - 0.018, 0.265, 0.016, 0.021),
            (r - 0.025, lip - 0.012, 0.268, 0.017, 0.017),
            (r - 0.007, lip - 0.008, 0.270, 0.019, 0.015),
        ]
        let section: Profile = [
            V2(0.50, -0.36), V2(0.50, 0.34), V2(0.27, 0.50), V2(-0.27, 0.50),
            V2(-0.50, 0.32), V2(-0.50, -0.32), V2(-0.25, -0.50), V2(0.25, -0.50),
        ]
        for group in 0..<5 {
            let pair = Float(group) * turn / 5 + 0.13
            for hand: Float in [-1, 1] {
                let rings: [[V3]] = stations.map { station in
                    let angle = pair + hand * station.offset
                    let center = radialPoint(station.axial, station.radius, angle)
                    let tangent = V3(0, -sin(angle), cos(angle))
                    return section.map { center + V3($0.x * station.depth, 0, 0) + tangent * ($0.y * station.width) }
                }
                mesh.tube(rings)
            }
        }
        return mesh
    }

    private static func wheelBarrel(_ axle: Axle) -> Mesh {
        let r = axle.rim
        var mesh = Mesh()
        mesh.lathe([
            V2(axle.lip - 0.016, r - 0.011), V2(axle.lip - 0.022, r - 0.002),
            V2(axle.lip - 0.044, r + 0.002), V2(-axle.halfWidth + 0.025, r + 0.002),
            V2(-axle.halfWidth + 0.014, r - 0.008), V2(-axle.halfWidth + 0.023, r - 0.016),
            V2(axle.lip - 0.048, r - 0.018),
        ], segments: 72)
        return mesh
    }

    private static func disc(_ axle: Axle) -> Mesh {
        let face = axle.discFace
        let r = axle.disc
        var mesh = Mesh()
        let inner = r * 0.54
        // The backing ring closes the dimples and creates a separate vent channel.
        mesh.lathe([
            V2(face - 0.006, inner), V2(face - 0.002, inner + 0.001),
            V2(face - 0.002, r - 0.0018), V2(face - 0.003, r),
            V2(face - 0.011, r), V2(face - 0.012, r - 0.005),
            V2(face - 0.017, r - 0.005), V2(face - 0.018, r),
            V2(face - 0.027, r), V2(face - 0.030, r - 0.003),
            V2(face - 0.030, inner), V2(face - 0.025, inner - 0.004),
        ], segments: 72)
        let bands = 4
        let sectors = 36
        let extent = turn / Float(sectors)
        let pitch = (r - inner) / Float(bands)
        for band in 0..<bands {
            let middle = inner + (Float(band) + 0.5) * pitch
            for sector in 0..<sectors {
                let angle = (Float(sector) + 0.5) * extent
                let dimpleAngle = angle + (Float(band) - 1.5) * 0.027
                let outer = (0..<8).map { index -> V2 in
                    let t = Float(index) * turn / 8
                    let p = V2(cos(t), sin(t))
                    let q = p / max(abs(p.x), abs(p.y))
                    return V2(angle + q.x * extent * 0.5, middle + q.y * pitch * 0.5)
                }
                let hole = (0..<8).map { index -> V2 in
                    let t = Float(index) * turn / 8
                    let bore: Float = 0.00235
                    return V2(dimpleAngle + cos(t) * bore / middle, middle + sin(t) * bore)
                }
                for i in 0..<8 {
                    let j = (i + 1) % 8
                    func point(_ axial: Float, _ ar: V2) -> V3 { radialPoint(axial, ar.y, ar.x) }
                    func uv(_ ar: V2) -> V2 { V2(ar.y / r, ar.x / turn) }
                    mesh.quad([point(face, outer[i]), point(face, outer[j]), point(face, hole[j]), point(face, hole[i])],
                              normal: V3(1, 0, 0), uv: [uv(outer[i]), uv(outer[j]), uv(hole[j]), uv(hole[i])])
                    let tangent = V3(0, -sin(dimpleAngle), cos(dimpleAngle))
                    let inward = radialPoint(0, 1, dimpleAngle)
                    let t = (Float(i) + 0.5) * turn / 8
                    let normal = simd_normalize(V3(0.4, 0, 0) - tangent * cos(t) - inward * sin(t))
                    mesh.quad([point(face, hole[i]), point(face, hole[j]), point(face - 0.004, hole[j]), point(face - 0.004, hole[i])],
                              normal: normal)
                }
            }
        }
        return mesh
    }

    private static func hardware(_ axle: Axle) -> Mesh {
        let face = axle.lip - 0.020
        let bell = axle.discFace - 0.001
        let r = axle.disc
        var mesh = Mesh()
        mesh.lathe([
            V2(bell - 0.005, 0.036), V2(bell + 0.006, 0.045), V2(bell + 0.004, 0.068),
            V2(bell - 0.001, r * 0.53), V2(bell - 0.004, r * 0.55),
            V2(bell - 0.021, r * 0.55), V2(bell - 0.022, 0.036),
        ], segments: 56)
        mesh.lathe([
            V2(face - 0.001, 0.0008), V2(face - 0.001, 0.0245), V2(face - 0.002, 0.0305),
            V2(face - 0.007, 0.033), V2(face - 0.014, 0.033), V2(face - 0.016, 0.025),
            V2(face - 0.015, 0.0008),
        ], segments: 60) { angle, point in
            point.y > 0.028 ? 0.00165 * cos(12 * angle) : 0
        }
        for index in 0..<10 {
            let center = radialPoint(0, r * 0.49, Float(index) * turn / 10)
            mesh.lathe([
                V2(bell + 0.0015, 0.0006), V2(bell + 0.0015, 0.0033),
                V2(bell, 0.0045), V2(bell - 0.0015, 0.0048), V2(bell - 0.003, 0.001),
            ], segments: 12, offset: center)
        }
        let valve = radialPoint(0, axle.rim - 0.026, 1.18)
        mesh.lathe([
            V2(axle.lip - 0.013, 0.001), V2(axle.lip - 0.013, 0.0033),
            V2(axle.lip - 0.021, 0.0035), V2(axle.lip - 0.029, 0.0050), V2(axle.lip - 0.032, 0.001),
        ], segments: 16, offset: valve)
        return mesh
    }

    private static func caliper(_ axle: Axle) -> Mesh {
        let front = axle.discFace + 0.018
        let outer = axle.disc + 0.023
        let outline: Profile = [
            V2(0.002, -0.039), V2(0.005, -0.029), V2(0.004, -0.010), V2(-0.001, 0),
            V2(-0.013, 0.003), V2(-0.037, 0.001), V2(-0.057, -0.008), V2(-0.059, -0.030),
            V2(-0.050, -0.042), V2(-0.038, -0.042), V2(-0.035, -0.015), V2(-0.009, -0.017),
        ]
        let rings: [[V3]] = (0...18).map { segment in
            let t = Float(segment) / 18
            let angle = axle.caliperAngle + (t - 0.5) * 0.83
            let end = smooth(abs(t - 0.5), from: 0.33, to: 0.53)
            let ribs = 0.0008 * (cos(t * turn * 3.1) + 1)
            return outline.map { point in
                let x = front + point.x - 0.009 * end + ribs
                let radius = outer + point.y - 0.014 * end
                return radialPoint(x, radius, angle)
            }
        }
        var mesh = Mesh()
        mesh.tube(rings, capEnds: true)
        return mesh
    }

    private static func tireLettering(_ axle: Axle, profile: Profile) -> Mesh {
        var mesh = Mesh()
        let sidewallRadius = axle.radius - 0.035
        func sidewall(_ radius: Float) -> Float {
            for i in 0..<10 where radius <= profile[i + 1].y {
                let fraction = (radius - profile[i].y) / (profile[i + 1].y - profile[i].y)
                return profile[i].x + fraction * (profile[i + 1].x - profile[i].x)
            }
            return profile[10].x
        }
        for (text, angle, hand): (String, Float, Float) in [
            ("PILOT SPORT CUP 2", -0.23, 1), (axle.tireSize, .pi + 0.20, -1),
        ] {
            mesh.lettering(text, pitch: 0.00155) { u, v in
                let radius = sidewallRadius + hand * v
                let theta = angle - hand * u / sidewallRadius
                var point = radialPoint(sidewall(radius) + 0.00055, radius, theta)
                point.y = max(point.y, -axle.height)
                return point
            }
        }
        return mesh
    }

    private static func inkDetails(_ axle: Axle) -> Mesh {
        var mesh = Mesh()
        mesh.lettering("GT3", pitch: 0.0013) { u, v in
            V3(axle.lip - 0.0205, v, -u)
        }
        // The mark runs down the long axis of the cast caliper body.
        let radius = axle.disc + 0.0075
        mesh.lettering("PORSCHE", pitch: 0.0012) { u, v in
            radialPoint(axle.discFace + 0.0244, radius + v, axle.caliperAngle - u / radius)
        }
        return mesh
    }

    private static func radialPoint(_ axial: Float, _ radius: Float, _ angle: Float) -> V3 {
        V3(axial, radius * cos(angle), radius * sin(angle))
    }

    private static func smooth(_ x: Float, from low: Float, to high: Float) -> Float {
        let t = simd_clamp((x - low) / (high - low), 0, 1)
        return t * t * (3 - 2 * t)
    }

    private struct Mesh {
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []

        mutating func lathe(
            _ profile: Profile, segments: Int, ground: Float? = nil, offset: V3 = .zero,
            radiusAdjustment: ((Float, V2) -> Float)? = nil
        ) {
            let stride = profile.count
            let base = UInt32(vertices.count)
            for row in 0...segments {
                let u = Float(row) / Float(segments)
                let angle = row == segments ? 0 : u * turn
                for i in profile.indices {
                    let point = profile[i]
                    let before = profile[(i + stride - 1) % stride]
                    let after = profile[(i + 1) % stride]
                    let edge = simd_normalize(point - before) + simd_normalize(after - point)
                    let profileNormal = simd_normalize(V2(edge.y, -edge.x))
                    var normal = radialPoint(profileNormal.x, profileNormal.y, angle)
                    let radius = point.y + (radiusAdjustment?(angle, point) ?? 0)
                    var position = radialPoint(point.x, radius, angle) + offset
                    if let ground, position.y < -ground {
                        position.y = -ground
                        normal = V3(0, -1, 0)
                    }
                    vertices.append(MeshVertex(position: position, normal: normal,
                                               textureCoordinate: V2(u, Float(i) / Float(stride))))
                }
            }
            for row in 0..<segments {
                for i in profile.indices {
                    let next = (i + 1) % stride
                    let a = base + UInt32(row * stride + i)
                    let b = base + UInt32(row * stride + next)
                    let c = base + UInt32((row + 1) * stride + next)
                    let d = base + UInt32((row + 1) * stride + i)
                    indices += [a, b, c, a, c, d]
                }
            }
        }

        mutating func tube(_ rings: [[V3]], capEnds: Bool = false) {
            let count = rings[0].count
            let centers = rings.map { $0.reduce(.zero, +) / Float(count) }
            let area = rings[0].indices.reduce(V3.zero) { sum, i in
                let next = (i + 1) % count
                return sum + simd_cross(rings[0][i] - centers[0], rings[0][next] - centers[0])
            }
            let winding: Float = simd_dot(area, centers[1] - centers[0]) > 0 ? 1 : -1
            for row in 0..<(rings.count - 1) {
                for i in 0..<count {
                    let next = (i + 1) % count
                    let points = [rings[row][i], rings[row][next], rings[row + 1][next], rings[row + 1][i]]
                    let normal = winding * simd_normalize(simd_cross(points[1] - points[0], points[3] - points[0]))
                    quad(points, normal: normal)
                }
            }
            if capEnds {
                for (row, neighbor) in [(0, 1), (rings.count - 1, rings.count - 2)] {
                    let normal = simd_normalize(centers[row] - centers[neighbor])
                    // The U-shaped caliper end is split along the open rotor slot.
                    let points = rings[row]
                    for section in [[0, 1, 2, 3, 4, 11], [4, 5, 10, 11], [5, 6, 7, 8, 9, 10]] {
                        for i in 1..<(section.count - 1) {
                            triangle(points[section[0]], points[section[i]], points[section[i + 1]], normal: normal)
                        }
                    }
                }
            }
        }

        mutating func quad(_ p: [V3], normal: V3, uv: [V2] = [.zero, .zero, .zero, .zero]) {
            let first = UInt32(vertices.count)
            for i in 0..<4 {
                vertices.append(MeshVertex(position: p[i], normal: normal, textureCoordinate: uv[i]))
            }
            if simd_dot(simd_cross(p[1] - p[0], p[2] - p[0]), normal) > 0 {
                indices += [first, first + 1, first + 2, first, first + 2, first + 3]
            } else {
                indices += [first, first + 2, first + 1, first, first + 3, first + 2]
            }
        }

        mutating func triangle(_ a: V3, _ b: V3, _ c: V3, normal: V3) {
            let first = UInt32(vertices.count)
            vertices += [a, b, c].map { MeshVertex(position: $0, normal: normal) }
            if simd_dot(simd_cross(b - a, c - a), normal) > 0 {
                indices += [first, first + 1, first + 2]
            } else {
                indices += [first, first + 2, first + 1]
            }
        }

        mutating func lettering(_ text: String, pitch: Float, project: (Float, Float) -> V3) {
            let origin = -Float(text.count * 6 - 1) * pitch * 0.5
            for (index, character) in text.enumerated() {
                guard let bitmap = LabPorscheWheels.glyphs[character] else { continue }
                for (row, bits) in bitmap.enumerated() {
                    for column in 0..<5 where bits & (16 >> column) != 0 {
                        let u = origin + Float(index * 6 + column) * pitch
                        let v = Float(3 - row) * pitch
                        quad([project(u, v), project(u + pitch, v), project(u + pitch, v - pitch), project(u, v - pitch)],
                             normal: V3(1, 0, 0))
                    }
                }
            }
        }
    }

    // Compact block forms give the raised sidewall codes and painted caliper their own geometry.
    private static let glyphs: [Character: [UInt8]] = [
        "P": [30, 17, 17, 30, 16, 16, 16], "I": [14, 4, 4, 4, 4, 4, 14],
        "L": [16, 16, 16, 16, 16, 16, 31], "O": [14, 17, 17, 17, 17, 17, 14],
        "T": [31, 4, 4, 4, 4, 4, 4], "S": [15, 16, 16, 14, 1, 1, 30],
        "R": [30, 17, 17, 30, 20, 18, 17], "C": [15, 16, 16, 16, 16, 16, 15],
        "U": [17, 17, 17, 17, 17, 17, 14], "Z": [31, 1, 2, 4, 8, 16, 31],
        "H": [17, 17, 17, 31, 17, 17, 17], "E": [31, 16, 16, 30, 16, 16, 31],
        "G": [14, 17, 16, 23, 17, 17, 15],
        "0": [14, 17, 19, 21, 25, 17, 14], "1": [4, 12, 4, 4, 4, 4, 14],
        "2": [14, 17, 1, 2, 4, 8, 31], "3": [30, 1, 1, 14, 1, 1, 30],
        "5": [31, 16, 16, 30, 1, 1, 30], "/": [1, 2, 2, 4, 8, 8, 16],
    ]
}
