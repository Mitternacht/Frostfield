import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Writes LM-63 text for synthetic luminaires.
enum IESFixture {
    static func text(
        format: String = "IES:LM-63-2019", type: Int = 1, lumens: Double = -1, multiplier: Double = 1,
        ballast: Double = 1, secondFactor: Double = 1, tilt: String = "NONE", tiltBlock: String? = nil,
        vertical: [Double], horizontal: [Double], candela: (Double, Double) -> Double
    ) -> String {
        var lines = [
            format,
            "[TEST] Frostfield synthetic profile",
            "[MANUFAC] Frostfield",
            "[MORE] Test fixture",
            "TILT=\(tilt)",
        ]
        if let tiltBlock {
            lines.append(contentsOf: tiltBlock.split(whereSeparator: \.isNewline).map(String.init))
        }
        lines.append(contentsOf: [
            "1 \(lumens) \(multiplier) \(vertical.count) \(horizontal.count) \(type) 2 0.1 0.1 0.05",
            "\(ballast) \(secondFactor) 25",
            vertical.map { String($0) }.joined(separator: " "),
            horizontal.map { String($0) }.joined(separator: " "),
        ])
        for h in horizontal {
            // Values wrap at ten per line, as luminaire exporters write them.
            let values = vertical.map { String(format: "%.4f", candela($0, h)) }
            stride(from: 0, to: values.count, by: 10).forEach { start in
                lines.append(values[start..<min(start + 10, values.count)].joined(separator: " "))
            }
        }
        return lines.joined(separator: "\r\n") + "\r\n"
    }

    /// A quadrant-symmetric type C downlight, 1,000 cos V (1 + 0.5 sin² V cos 2H) cd below the
    /// horizon: elongated along the 0°–180° plane and single-valued at the nadir.
    static func quadrantDownlight(
        format: String = "IES:LM-63-2019", multiplier: Double = 1, ballast: Double = 1, secondFactor: Double = 1,
        tilt: String = "NONE", tiltBlock: String? = nil
    ) -> String {
        text(
            format: format, multiplier: multiplier, ballast: ballast, secondFactor: secondFactor,
            tilt: tilt, tiltBlock: tiltBlock,
            vertical: Array(stride(from: 0.0, through: 90, by: 7.5)), horizontal: [0, 22.5, 45, 67.5, 90]
        ) { v, h in shape(v, h) }
    }

    static func asymmetricDownlight() -> String {
        text(
            vertical: Array(stride(from: 0.0, through: 90, by: 7.5)),
            horizontal: Array(stride(from: 0.0, through: 360, by: 15))
        ) { v, h in
            let theta = v * .pi / 180
            let phi = h * .pi / 180
            return 1_000 * cos(theta) * (1 + 0.7 * sin(theta) * sin(phi))
        }
    }

    static func shape(_ v: Double, _ h: Double) -> Double {
        let sine = sin(v * .pi / 180)
        return 1_000 * cos(v * .pi / 180) * (1 + 0.5 * sine * sine * cos(2 * h * .pi / 180))
    }
}

struct IESProfileTests {
    @Test("An invalid photometric type returns a header error", arguments: ["1e30", "-1e30", "1.5"])
    func invalidPhotometricType(type: String) {
        let text = "IES:LM-63-2019\nTILT=NONE\n1 -1 1 1 1 \(type) 2 0 0 0\n1 1 0\n0\n0\n1\n"
        #expect(throws: IESProfile.Failure.header("the photometric type must be 1 (C), 2 (B) or 3 (A)")) {
            try IESProfile(string: text)
        }
    }

    @Test("A written LM-63-2019 file parses with keywords, multipliers, tabulated values and bilinear interpolation")
    func parseQuadrantFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("frostfield-ies-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("quadrant.ies")
        try IESFixture.quadrantDownlight(multiplier: 2, ballast: 0.9, secondFactor: 1.10001).write(to: url, atomically: true, encoding: .utf8)
        let profile = try IESProfile(contentsOf: url)

        #expect(profile.format == "IES:LM-63-2019" && profile.photometricType == .c)
        #expect(profile.keywords["MANUFAC"] == "Frostfield\nTest fixture")
        #expect(profile.lumensPerLamp == nil && profile.ratedLumens == nil && profile.lampCount == 1 && profile.inputWatts == 25)
        #expect(profile.verticalAngles.count == 13 && profile.horizontalAngles == [0, 22.5, 45, 67.5, 90])
        // LM-63-2019 reads the second factor as the file generation type, so only 2 × 0.9 applies.
        let scale: Float = 1.8
        for v in profile.verticalAngles {
            for h in profile.horizontalAngles {
                let written = Float(String(format: "%.4f", IESFixture.shape(Double(v), Double(h))))!
                #expect(abs(profile.intensity(verticalAngle: v, horizontalAngle: h) - written * scale) < 1e-3)
            }
        }
        #expect(abs(profile.maximumIntensity - 1_000 * scale) < 1e-3)
        // Bilinear between V 30°–37.5° and H 0°–22.5°.
        func at(_ v: Double, _ h: Double) -> Double { Double(Float(String(format: "%.4f", IESFixture.shape(v, h)))!) * 1.8 }
        let expected = 0.5 * (0.25 * at(37.5, 0) + 0.75 * at(30, 0)) + 0.5 * (0.25 * at(37.5, 22.5) + 0.75 * at(30, 22.5))
        #expect(abs(Double(profile.intensity(verticalAngle: 31.875, horizontalAngle: 11.25)) - expected) < 1e-2)
        // Quadrant symmetry about the 0°–180° and 90°–270° planes; nothing above the horizon.
        let reference = profile.intensity(verticalAngle: 40, horizontalAngle: 30)
        for h: Float in [150, 210, 330, -30, 390] {
            #expect(abs(profile.intensity(verticalAngle: 40, horizontalAngle: h) - reference) < 1e-3)
        }
        #expect(profile.intensity(verticalAngle: 95, horizontalAngle: 0) == 0)
        // Directions in the light frame: −Z is the nadir, +X is H 0°, +Y is H 90°.
        let tilt = Float(30 * Double.pi / 180)
        #expect(abs(profile.intensity(toward: SIMD3(sin(tilt), 0, -cos(tilt))) - profile.intensity(verticalAngle: 30, horizontalAngle: 0)) < 1e-2)
        #expect(abs(profile.intensity(toward: SIMD3(0, sin(tilt), -cos(tilt))) - profile.intensity(verticalAngle: 30, horizontalAngle: 90)) < 1e-2)
        #expect(profile.intensity(toward: SIMD3(0, 0, 1)) == 0)
        // The renderers' resampled table agrees with the file's interpolation.
        for (theta, phi) in [(0.3, 0.2), (1.0, 2.5), (1.4, 4.0), (0.05, 5.9)] as [(Float, Float)] {
            let exact = profile.intensity(toward: SIMD3(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta)))
            #expect(abs(profile.tableIntensity(theta: theta, phi: phi) / exact - 1) < 0.005, "θ \(theta), φ \(phi)")
        }
    }

    @Test("Integrated flux matches the header lumens of consistent files")
    func luminousFlux() throws {
        // A Lambertian downlight emits π I₀; 1,000 lm needs I₀ = 1,000 / π.
        let lambert = try IESProfile(string: IESFixture.text(
            lumens: 1_000, vertical: Array(stride(from: 0.0, through: 90, by: 2.5)), horizontal: [0]
        ) { v, _ in 1_000 / .pi * cos(v * .pi / 180) })
        #expect(lambert.ratedLumens == 1_000)
        #expect(abs(lambert.luminousFlux / 1_000 - 1) < 0.002, "\(lambert.luminousFlux) lm")
        // An isotropic 100 cd source emits 400π lm.
        let sphere = try IESProfile(string: IESFixture.text(
            lumens: 400 * .pi, vertical: Array(stride(from: 0.0, through: 180, by: 10)), horizontal: [0]
        ) { _, _ in 100 })
        #expect(abs(sphere.luminousFlux / Float(400 * Double.pi) - 1) < 1e-4)
        // An independent midpoint quadrature of the file's own interpolation over the sphere.
        let quadrant = try IESProfile(string: IESFixture.quadrantDownlight())
        var integral = 0.0
        let steps = 720
        for i in 0..<steps {
            let theta = (Double(i) + 0.5) / Double(steps) * .pi
            for j in 0..<(2 * steps) {
                let phi = (Double(j) + 0.5) / Double(2 * steps) * 2 * .pi
                let direction = SIMD3<Float>(Float(sin(theta) * cos(phi)), Float(sin(theta) * sin(phi)), Float(-cos(theta)))
                integral += Double(quadrant.intensity(toward: direction)) * sin(theta) * (.pi / Double(steps)) * (.pi / Double(steps))
            }
        }
        #expect(abs(Double(quadrant.luminousFlux) / integral - 1) < 0.002, "\(quadrant.luminousFlux) vs \(integral)")
    }

    @Test("Symmetric horizontal ranges and photometric types B and A map directions as LM-63 describes")
    func symmetriesAndTypes() throws {
        let vertical = Array(stride(from: 0.0, through: 180, by: 15))
        func profile(_ horizontal: [Double], _ candela: @escaping (Double, Double) -> Double) throws -> IESProfile {
            try IESProfile(string: IESFixture.text(vertical: vertical, horizontal: horizontal, candela: candela))
        }
        // Bilateral about the 0°–180° plane: I(H) = I(360° − H).
        let bilateral = try profile(Array(stride(from: 0.0, through: 180, by: 30))) { _, h in 100 + h }
        #expect(abs(bilateral.intensity(verticalAngle: 45, horizontalAngle: 300) - 160) < 1e-3)
        // Bilateral about the 90°–270° plane: I(H) = I(180° − H).
        let lateral = try profile(Array(stride(from: 90.0, through: 270, by: 30))) { _, h in 100 + h }
        #expect(abs(lateral.intensity(verticalAngle: 45, horizontalAngle: 30) - 250) < 1e-3)
        #expect(abs(lateral.intensity(verticalAngle: 45, horizontalAngle: 330) - 310) < 1e-3)
        // A full table without a 360° column wraps from its last angle back to 0°.
        let full = try profile(Array(stride(from: 0.0, through: 330, by: 30))) { _, h in 100 + h }
        #expect(abs(full.intensity(verticalAngle: 45, horizontalAngle: 345) - (430 + 100) / 2) < 1e-3)

        let asymmetric = try IESProfile(string: IESFixture.asymmetricDownlight())
        let tilt = Float.pi / 6
        #expect(asymmetric.intensity(toward: SIMD3(0, sin(tilt), -cos(tilt))) > 1_100)
        #expect(asymmetric.intensity(toward: SIMD3(0, -sin(tilt), -cos(tilt))) < 600)

        // Type B: vertical angles tilt about local X, horizontal angles lean toward +X; laterally symmetric.
        let typeB = try IESProfile(string: IESFixture.text(
            type: 2, vertical: Array(stride(from: -90.0, through: 90, by: 10)), horizontal: Array(stride(from: 0.0, through: 90, by: 10))
        ) { v, h in 500 + 2 * v + h })
        let angle = Float(20 * Double.pi / 180)
        #expect(abs(typeB.intensity(toward: SIMD3(0, sin(angle), -cos(angle))) - 540) < 0.05)
        #expect(abs(typeB.intensity(toward: SIMD3(-sin(angle), 0, -cos(angle))) - 520) < 0.05)
        #expect(typeB.intensity(toward: SIMD3(0, 0, 1)) == 0)
        // Type A: horizontal angles turn about local Y, vertical angles are latitude toward +Y.
        let typeA = try IESProfile(string: IESFixture.text(
            type: 3, vertical: Array(stride(from: -90.0, through: 90, by: 10)), horizontal: Array(stride(from: -90.0, through: 90, by: 10))
        ) { v, h in 500 + 2 * v + h })
        #expect(abs(typeA.intensity(toward: SIMD3(sin(angle), 0, -cos(angle))) - 520) < 0.05)
        #expect(abs(typeA.intensity(toward: SIMD3(-sin(angle), 0, -cos(angle))) - 480) < 0.05)
        #expect(abs(typeA.intensity(toward: SIMD3(0, sin(angle), -cos(angle))) - 540) < 0.05)
    }

    @Test("Older revisions apply the ballast-lamp factor, and malformed files fail with specific errors")
    func revisionsAndErrors() throws {
        let old = try IESProfile(string: IESFixture.quadrantDownlight(format: "IESNA:LM-63-1995", secondFactor: 0.5))
        let modern = try IESProfile(string: IESFixture.quadrantDownlight(format: "IESNA:LM-63-2002", secondFactor: 0.5))
        #expect(abs(old.maximumIntensity - 500) < 1e-3 && abs(modern.maximumIntensity - 1_000) < 1e-3)

        let vertical: [Double] = [0, 45, 90]
        func parse(_ text: String) -> IESProfile.Failure? {
            do { _ = try IESProfile(string: text); return nil } catch { return error as? IESProfile.Failure }
        }
        if case .tilt? = parse(IESFixture.text(tilt: "INCLUDE", vertical: vertical, horizontal: [0]) { _, _ in 1 }) {} else {
            Issue.record("TILT=INCLUDE without a tilt curve should fail")
        }
        #expect(parse("IES:LM-63-2019\n1 -1 1 1 1 1 2 0 0 0\n1 1 0\n0\n0\n1\n") == .missingTilt)
        if case .angles? = parse(IESFixture.text(vertical: [0, 90, 45], horizontal: [0]) { _, _ in 1 }) {} else {
            Issue.record("decreasing vertical angles should fail")
        }
        if case .angles? = parse(IESFixture.text(vertical: vertical, horizontal: [10, 20]) { _, _ in 1 }) {} else {
            Issue.record("a type C table must start at 0° or 90°")
        }
        if case .values? = parse(IESFixture.text(vertical: vertical, horizontal: [0]) { _, _ in 0 }) {} else {
            Issue.record("an all-zero table should fail")
        }
        if case .header? = parse(IESFixture.text(type: 4, vertical: vertical, horizontal: [0]) { _, _ in 1 }) {} else {
            Issue.record("photometric type 4 should fail")
        }
        var truncated = IESFixture.text(vertical: vertical, horizontal: [0]) { _, _ in 1 }
        truncated = String(truncated.dropLast(10))
        if case .values? = parse(truncated) {} else { Issue.record("a short candela table should fail") }
        // IES profiles fit point and spot lights; cookies only spots.
        let profile = try IESProfile(string: IESFixture.quadrantDownlight())
        #expect(throws: PunctualLight.Failure.profileKind) { try PunctualLight(kind: .directional, iesProfile: profile) }
        _ = try PunctualLight(kind: .point(), iesProfile: profile)
    }

    @Test("Rig power with a profile divides by the profile's solid angle, so total flux stays as authored")
    func rigFlux() throws {
        let profile = try IESProfile(string: IESFixture.quadrantDownlight())
        var light = LightRig.Light(
            id: "ies", kind: .point(), position: SIMD3(0, 2, 0), target: .zero, emission: .power(1_000), iesProfile: profile
        )
        // The test integrates the normalized distribution itself.
        var solidAngle = 0.0
        var spotSolidAngle = 0.0
        let steps = 360
        let angleStep = Double.pi / Double(steps)
        let inner: Float = 0.3
        let outer: Float = 0.6
        for i in 0..<steps {
            let theta = (Double(i) + 0.5) * angleStep
            let cone = min(max((cos(theta) - cos(Double(outer))) / (cos(Double(inner)) - cos(Double(outer))), 0), 1)
            let projectedArea = sin(theta) * angleStep * angleStep
            for j in 0..<(2 * steps) {
                let phi = (Double(j) + 0.5) * angleStep
                let direction = SIMD3<Float>(Float(sin(theta) * cos(phi)), Float(sin(theta) * sin(phi)), Float(-cos(theta)))
                let weightedArea = Double(profile.intensity(toward: direction) / profile.maximumIntensity) * projectedArea
                solidAngle += weightedArea
                spotSolidAngle += weightedArea * cone * cone
            }
        }
        #expect(abs(Double(try light.intensity()) * solidAngle / 1_000 - 1) < 0.003)
        #expect(abs(try #require(try light.power()) - 1_000) < 1e-3)
        light.emission = .intensity(profile.maximumIntensity)
        #expect(abs(try #require(try light.power()) / profile.luminousFlux - 1) < 1e-5)
        light.kind = .spot(innerAngle: inner, outerAngle: outer)
        #expect(abs(Double(try #require(try light.power())) / (Double(profile.maximumIntensity) * spotSolidAngle) - 1) < 0.003)
        light.emission = .power(1_000)
        #expect(abs(Double(try light.intensity()) * spotSolidAngle / 1_000 - 1) < 0.003)
        light.kind = .directional()
        #expect(throws: LightRig.Failure.lightTables("ies")) { try LightRig(lights: [light]) }
    }

    @Test("An included lamp tilt is kept and does not scale the measured candela")
    func includedTilt() throws {
        let plain = try IESProfile(string: IESFixture.quadrantDownlight())
        let tilted = try IESProfile(string: IESFixture.quadrantDownlight(
            tilt: "INCLUDE", tiltBlock: "1\n3\n0 45 90\n1 0.5 0.25"
        ))
        let curve = try #require(tilted.lampTilt)
        #expect(curve.orientation == .vertical)
        #expect(curve.angles == [0, 45, 90])
        #expect(curve.factors == [1, 0.5, 0.25])
        #expect(plain.lampTilt == nil)
        #expect(plain.candela == tilted.candela)
        #expect(plain.intensity(verticalAngle: 30, horizontalAngle: 45) == tilted.intensity(verticalAngle: 30, horizontalAngle: 45))

        let named = IESFixture.quadrantDownlight(tilt: "lamp.tilt")
        #expect(throws: IESProfile.Failure.unsupportedTilt("lamp.tilt")) {
            try IESProfile(string: named)
        }
        let external = try IESProfile(string: named, tilt: "TILT=INCLUDE\n2\n2\n0 90\n1 0.25\n")
        #expect(external.lampTilt == IESProfile.LampTilt(orientation: .horizontal, angles: [0, 90], factors: [1, 0.25]))
        #expect(external.candela == plain.candela)
        #expect(throws: IESProfile.Failure.tilt("a TILT=NONE file has no tilt document")) {
            try IESProfile(string: IESFixture.quadrantDownlight(), tilt: "1\n1\n0\n1\n")
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct IESRenderTests {
    private let fixture: PathCameraFixture

    init() throws {
        fixture = try PathCameraFixture()
    }

    @Test("Both renderers light a plane with I(θ, φ) cos θ / d² from an IES point or spot light",
          arguments: [false, true])
    func illuminanceMatchesProfile(asymmetric: Bool) throws {
        let profile = try IESProfile(string: asymmetric ? IESFixture.asymmetricDownlight() : IESFixture.quadrantDownlight())
        let albedo: Float = 0.8
        let floor = try fixture.lambert(.plane(width: 10, height: 10), albedo: albedo)
        let floorNode = SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: floor)
        let side = 48
        let height: Float = 5
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        let cameraTransform = SceneTransform(translation: SIMD3(0, height, 0), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)))
        let lightPosition = SIMD3<Float>(0.3, 1.5, -0.2)
        // A slight tilt makes the profile's axis and horizontal angles both matter.
        let target = SIMD3<Float>(0.6, 0, 0.1)

        for kind in [LightRig.Light.Kind.point(), .spot(innerAngle: 0.5, outerAngle: 1.2)] {
            let light = LightRig.Light(
                id: "ies", kind: kind, position: lightPosition, target: target,
                color: SIMD3(1, 0.9, 0.8), emission: .intensity(profile.maximumIntensity / 400), iesProfile: profile
            )
            let rig = try LightRig(lights: [light])
            var graph = SceneGraph(roots: [floorNode] + (try rig.sceneNodes()), camera: camera)
            graph.cameraTransform = cameraTransform
            let raster = try fixture.base.readback(fixture.rasterize(graph, width: side, height: side).color)
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
            let tracer = try PathTracer(queue: fixture.queue, scene: scene)
            let traced = try fixture.trace(
                tracer, model: .init(projection: .perspective(verticalFieldOfView: .pi / 3), near: 0.1, far: 30),
                transform: cameraTransform, width: side, height: side, samples: 64, lighting: rig.pathLighting()
            )
            if !asymmetric {
                PathCameraFixture.writeImage(raster, width: side, height: side, name: "ies-raster-\(light.kind == .point() ? "point" : "spot")", scale: 0.5)
                PathCameraFixture.writeImage(traced, width: side, height: side, name: "ies-traced-\(light.kind == .point() ? "point" : "spot")", scale: 0.5)
            }

            let rotation = try light.transform().rotation
            let right = rotation.act(SIMD3<Float>(1, 0, 0))
            let up = rotation.act(SIMD3<Float>(0, 1, 0))
            let forward = rotation.act(SIMD3<Float>(0, 0, -1))
            let scale = try light.intensity() / profile.maximumIntensity
            var checked = 0
            var worst: (raster: Float, traced: Float) = (0, 0)
            for row in 0..<side {
                for column in 0..<side {
                    // The pixel center's ray from the camera hits the floor at y = 0.
                    let ndc = SIMD2((Float(column) + 0.5) / Float(side) * 2 - 1, 1 - (Float(row) + 0.5) / Float(side) * 2)
                    let tangent = tan(Float.pi / 6)
                    let point = SIMD3(ndc.x * tangent * height, 0, -ndc.y * tangent * height)
                    let delta = point - lightPosition
                    let distance = simd_length(delta)
                    let toSurface = delta / distance
                    let local = SIMD3(simd_dot(toSurface, right), simd_dot(toSurface, up), -simd_dot(toSurface, forward))
                    var intensity = profile.intensity(toward: local) * scale
                    if case .spot(let inner, let outer, _) = kind {
                        let cone = simd_clamp((simd_dot(toSurface, forward) - cos(outer)) / (cos(inner) - cos(outer)), 0, 1)
                        intensity *= cone * cone
                    }
                    let illuminance = intensity * (lightPosition.y / distance) / (distance * distance)
                    let expected = albedo / .pi * illuminance
                    guard expected > 0.05 else { continue }
                    checked += 1
                    let index = row * side + column
                    let rasterError = abs(raster[index].x / expected - 1)
                    let tracedError = abs(traced[index].x / expected - 1)
                    worst = (max(worst.raster, rasterError), max(worst.traced, tracedError))
                    // Green carries the light's 0.9 tint in both renderers.
                    #expect(abs(raster[index].y / (expected * 0.9) - 1) < 0.005)
                }
            }
            #expect(checked > (kind == .point() ? 600 : 250), "\(checked) lit pixels")
            #expect(worst.raster < 0.005, "\(kind): raster worst relative error \(worst.raster)")
            #expect(worst.traced < 0.03, "\(kind): traced worst relative error \(worst.traced)")
        }
    }
}
