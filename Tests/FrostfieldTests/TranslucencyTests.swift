import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Diffuse transmission, rough (EON) diffuse, clearcoat normal maps and the raster
/// translucency approximations, checked in both renderers against closed forms.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TranslucencyTests {
    private let fixture: MediaFixture

    init() throws {
        fixture = try MediaFixture()
    }

    /// A camera `distance` from the origin along (sin angle, 0, cos angle), looking at the origin.
    private func pose(angle: Float, distance: Float = 3) -> SceneTransform {
        SceneTransform(
            translation: SIMD3(sin(angle), 0, cos(angle)) * distance,
            rotation: simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
        )
    }

    @Test("A thin diffuse-transmission sheet transmits factor x color x cos / pi of back light and reflects the rest, in both renderers")
    func thinSheet() throws {
        let width = 17
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.3, near: 0.1, far: 30)
        let base = SIMD3<Float>(0.5, 0.5, 0.5)
        let factor: Float = 0.6
        let tint = SIMD3<Float>(1, 0.5, 0.25)
        let material = try PhysicallyBasedMaterial(
            roughness: 0.7, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(base, 1), alphaMode: .opaque,
            isDoubleSided: true, specular: .init(factor: 0),
            diffuseTransmission: .init(factor: factor, colorFactor: tint)
        )
        let sheet = try fixture.renderer(try MeshGeometry.plane(width: 4, height: 4), material)
        let irradiance: Float = 3
        let angle = Float.pi / 6
        for behind in [true, false] {
            // Light arriving from behind the sheet (toward -Z side) or in front of it.
            let toLight = SIMD3<Float>(sin(angle), 0, behind ? -cos(angle) : cos(angle))
            let light = try PunctualLight(kind: .directional, intensity: irradiance)
            let graph = SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, 0)), mesh: sheet),
                SceneNode(transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -toLight)), light: light),
            ], camera: camera)
            let expected = (behind ? factor * tint : (1 - factor) * base) * irradiance * cos(angle) / .pi
            let traced = try fixture.trace(
                graph, camera: camera, pose: pose(angle: 0), width: width, height: width, samples: 16,
                settings: .init(maxBounces: 1, seed: 2),
                lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph))
            )
            let raster = try fixture.rasterize(
                SceneGraph(roots: graph.roots, camera: camera, cameraTransform: pose(angle: 0)), width: width, height: width
            )
            let tracedCenter = MediaFixture.center(traced, width: width, height: width, radius: 2)
            let rasterCenter = MediaFixture.center(raster, width: width, height: width, radius: 2)
            #expect(simd_reduce_max(abs(tracedCenter - expected) / expected) < 0.01, "traced \(behind): \(tracedCenter) vs \(expected)")
            #expect(simd_reduce_max(abs(rasterCenter - expected) / expected) < 0.01, "raster \(behind): \(rasterCenter) vs \(expected)")
        }

        // Under uniform light both sides reach the eye: (1 - factor) base + factor tint.
        let furnace = try fixture.trace(
            SceneGraph(roots: [SceneNode(mesh: sheet)], camera: camera), camera: camera, pose: pose(angle: 0.4),
            width: width, height: width, samples: 256, settings: .init(maxBounces: 1, seed: 8),
            lighting: .init(environment: SIMD3(repeating: 1))
        )
        let uniform = (1 - factor) * base + factor * tint
        let measured = MediaFixture.center(furnace, width: width, height: width, radius: 3)
        #expect(simd_reduce_max(abs(measured - uniform) / uniform) < 0.015, "\(measured) vs \(uniform)")
        let filled = try fixture.rasterize(SceneGraph(roots: [SceneNode(mesh: try fixture.renderer(
            try MeshGeometry.plane(width: 4, height: 4),
            try PhysicallyBasedMaterial(
                roughness: 0.7, lightIntensity: 0, ambientIntensity: 1, baseColorFactor: SIMD4(base, 1), alphaMode: .opaque,
                isDoubleSided: true, specular: .init(factor: 0), diffuseTransmission: .init(factor: factor, colorFactor: tint)
            )
        ))], camera: camera, cameraTransform: pose(angle: 0.4)), width: width, height: width)
        let fill = MediaFixture.center(filled, width: width, height: width, radius: 3)
        #expect(simd_reduce_max(abs(fill - uniform) / uniform) < 0.01, "raster fill \(fill) vs \(uniform)")
    }

    @Test("Rough diffuse reflection follows the EON reference at sample angles in both renderers and keeps a white surface white")
    func orenNayar() throws {
        let width = 17
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.1, near: 0.1, far: 30)
        let albedo = SIMD3<Float>(0.9, 0.5, 0.2)
        let irradiance: Float = 2
        // (light zenith, light azimuth, view zenith) in radians; the view azimuth is zero.
        let configurations: [(Float, Float, Float)] = [
            (0, 0, 0), (.pi / 3, 0, .pi / 3), (.pi / 3, .pi, .pi / 3), (.pi / 6, .pi / 2, .pi / 4), (1.2, 0.3, 0.9),
        ]
        for roughness in [0.5, 1] as [Float] {
            let material = try PhysicallyBasedMaterial(
                roughness: 0.5, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(albedo, 1), alphaMode: .opaque,
                specular: .init(factor: 0), diffuseRoughness: .init(factor: roughness)
            )
            let plane = try fixture.renderer(try MeshGeometry.plane(width: 6, height: 6), material)
            for (lightZenith, lightAzimuth, viewZenith) in configurations {
                let toLight = SIMD3<Float>(sin(lightZenith) * cos(lightAzimuth), sin(lightZenith) * sin(lightAzimuth), cos(lightZenith))
                let toView = SIMD3<Float>(sin(viewZenith), 0, cos(viewZenith))
                let graph = SceneGraph(roots: [
                    SceneNode(mesh: plane),
                    SceneNode(
                        transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -toLight)),
                        light: try PunctualLight(kind: .directional, intensity: irradiance)
                    ),
                ], camera: camera)
                let expected = SIMD3<Float>((0..<3).map { channel in
                    Float(MediaReference.eon(
                        albedo: Double(albedo[channel]), roughness: Double(roughness),
                        incoming: SIMD3<Double>(toLight), outgoing: SIMD3<Double>(toView)
                    ) * Double(irradiance * toLight.z))
                })
                let lambert = albedo * irradiance * toLight.z / .pi
                let traced = try fixture.trace(
                    graph, camera: camera, pose: pose(angle: viewZenith), width: width, height: width, samples: 16,
                    settings: .init(maxBounces: 1, seed: 4),
                    lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph))
                )
                let raster = try fixture.rasterize(
                    SceneGraph(roots: graph.roots, camera: camera, cameraTransform: pose(angle: viewZenith)), width: width, height: width
                )
                let tracedCenter = MediaFixture.center(traced, width: width, height: width, radius: 1)
                let rasterCenter = MediaFixture.center(raster, width: width, height: width, radius: 1)
                let label = "r \(roughness), light \(lightZenith)/\(lightAzimuth), view \(viewZenith)"
                #expect(simd_reduce_max(abs(tracedCenter - expected) / expected) < 0.01, "traced \(label): \(tracedCenter) vs \(expected)")
                #expect(simd_reduce_max(abs(rasterCenter - expected) / expected) < 0.01, "raster \(label): \(rasterCenter) vs \(expected)")
                if lightZenith > 0.5, lightAzimuth == 0 {
                    // Retro-reflection brightens a rough surface beyond Lambert.
                    #expect(tracedCenter.x > lambert.x * 1.05, "\(label): \(tracedCenter.x) vs Lambert \(lambert.x)")
                }
            }
        }

        // EON is energy-preserving: a white rough sphere in a white furnace stays white.
        let sphereCamera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 30)
        let white = try PhysicallyBasedMaterial(
            roughness: 0.5, lightIntensity: 0, ambientIntensity: 1, alphaMode: .opaque,
            specular: .init(factor: 0), diffuseRoughness: .init(factor: 1)
        )
        let sphere = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3.5)), mesh: try fixture.renderer(
                try MeshGeometry.uvSphere(radius: 1, segments: 64, rings: 32), white
            )),
        ], camera: sphereCamera)
        let furnace = try fixture.trace(
            sphere, camera: sphereCamera, width: 33, height: 33, samples: 256, settings: .init(maxBounces: 1, seed: 6),
            lighting: .init(environment: SIMD3(repeating: 1))
        )
        let mean = furnace.map { SIMD3($0.x, $0.y, $0.z) }.reduce(.zero, +) / Float(furnace.count)
        #expect(simd_reduce_max(abs(mean - 1)) < 0.005, "furnace mean \(mean)")
        // Raster fill light uses EON's exact directional albedo, which is one for a white surface.
        let filled = try fixture.rasterize(sphere, width: 33, height: 33)
        let litPixels = filled.filter { $0.x > 0 }
        #expect(!litPixels.isEmpty && litPixels.allSatisfy { abs($0.x - 1) < 0.002 })
    }

    @Test("A clearcoat normal map tilts only the coat in the path tracer, matching raster at the mirror configuration",
          arguments: [false, true])
    func clearcoatNormalMap(baked: Bool) throws {
        let width = 65
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        let tilt: Float = 0.35
        // A tangent-space normal tilted toward +U (+X on the plane).
        let encoded = SIMD3<Float>(sin(tilt), 0, cos(tilt)) * 0.5 + 0.5
        let texel = (0..<3).map { UInt8((encoded[$0] * 255).rounded()) } + [255]
        let map = try fixture.dataTexture(width: 4, height: 4, pixels: Array([[UInt8]](repeating: texel, count: 16).joined()))
        let decoded = simd_normalize(SIMD3<Float>(texel[0...2].map { Float($0) / 255 * 2 - 1 }))
        func material(coatMap: Bool, baseMap: Bool) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                normalTexture: baseMap ? .init(texture: map) : nil, alphaMode: .opaque,
                clearcoat: .init(factor: 1, roughness: 0.3, normalTexture: coatMap ? .init(texture: map) : nil),
                specular: .init(factor: 0)
            )
        }
        // The camera looks down -Z at the plane's center; the light sits at the mirror direction of the tilted normal.
        let mirrored = simd_normalize(2 * decoded.z * decoded - SIMD3(0, 0, 1))
        func render(_ material: PhysicallyBasedMaterial, toLight: SIMD3<Float>) throws -> (traced: Float, raster: Float) {
            let graph = SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: try fixture.renderer(
                    try MeshGeometry.plane(width: 4, height: 4), material
                )),
                SceneNode(
                    transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -toLight)),
                    light: try PunctualLight(kind: .directional, intensity: 1)
                ),
            ], camera: camera)
            let traced = try fixture.trace(
                graph, camera: camera, width: width, height: width, samples: 16, settings: .init(maxBounces: 1, seed: 9),
                lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph)),
                options: baked ? .init(textures: .baked(size: 16)) : .init()
            )
            let raster = try fixture.rasterize(graph, width: width, height: width)
            MediaFixture.save(traced, width: width, height: width, name: "coat-\(toLight.x)-traced")
            return (traced[width / 2 * width + width / 2].x, raster[width / 2 * width + width / 2].x)
        }
        let mapped = try render(try material(coatMap: true, baseMap: false), toLight: mirrored)
        let unmapped = try render(try material(coatMap: false, baseMap: false), toLight: mirrored)
        let baseOnly = try render(try material(coatMap: false, baseMap: true), toLight: mirrored)
        // F(cos) D(1) V NoL for the GGX coat at alpha 0.09 seen along its mirror direction.
        let alpha: Float = 0.09
        let cosine = decoded.z
        let fresnel = 0.04 + 0.96 * pow(1 - cosine, 5)
        let lambda = (alpha * alpha + (1 - alpha * alpha) * cosine * cosine).squareRoot()
        let analytic = fresnel / (.pi * alpha * alpha) * 0.5 / (2 * cosine * lambda) * cosine
        #expect(abs(mapped.traced / analytic - 1) < 0.03, "traced \(mapped.traced) vs analytic \(analytic)")
        #expect(abs(mapped.traced / mapped.raster - 1) < 0.03, "traced \(mapped.traced) vs raster \(mapped.raster)")
        // Without the coat map the highlight sits at the vertex normal's mirror direction, even
        // when the base layer has a normal map, as KHR_materials_clearcoat specifies.
        #expect(unmapped.traced < 0.02 * mapped.traced && unmapped.raster < 0.02 * mapped.raster)
        #expect(baseOnly.traced < 0.02 * mapped.traced && baseOnly.raster < 0.02 * mapped.raster)
        let flat = try render(try material(coatMap: false, baseMap: true), toLight: SIMD3(0, 0, 1))
        #expect(abs(flat.traced / flat.raster - 1) < 0.03, "flat coat \(flat.traced) vs raster \(flat.raster)")
    }

    @Test("Raster subsurface and diffuse-transmission volumes: zero radius is diffuse, scattering wraps the terminator, thickness dims back light")
    func rasterApproximation() throws {
        let width = 65
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 30)
        let color = SIMD3<Float>(0.9, 0.6, 0.4)
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 96, rings: 48)
        let light = try PunctualLight(kind: .directional, intensity: 3)
        func image(_ material: PhysicallyBasedMaterial) throws -> [SIMD4<Float>] {
            try fixture.rasterize(SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -3.5)), mesh: try fixture.renderer(sphere, material)),
                // Light from +X: the terminator runs down the sphere's middle column.
                SceneNode(transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: SIMD3(-1, 0, 0))), light: light),
            ], camera: camera), width: width, height: width)
        }
        func material(base: SIMD3<Float>, subsurface: PhysicallyBasedMaterial.Subsurface?) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 0.6, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(base, 1),
                alphaMode: .opaque, specular: .init(factor: 0), subsurface: subsurface
            )
        }
        let diffuse = try image(material(base: color, subsurface: nil))
        let zero = try image(material(base: SIMD3(repeating: 1), subsurface: .init(color: color, radius: .zero, scale: 1)))
        #expect(PathTestFixture.maximumDifference(diffuse, zero) < 2e-3)
        // A long red mean free path wraps red light past the terminator; blue stays near diffuse.
        let wrapped = try image(material(
            base: SIMD3(repeating: 1), subsurface: .init(color: color, radius: SIMD3(1, 0.2, 0.05), scale: 0.2)
        ))
        MediaFixture.save(wrapped, width: width, height: width, name: "raster-subsurface-wrap")
        let row = width / 2 * width
        // Columns just past the terminator (x slightly negative on the sphere's silhouette).
        let shadowSide = (width / 2 - 4)...(width / 2 - 2)
        let redWrap = shadowSide.map { wrapped[row + $0].x }.reduce(0, +)
        let blueWrap = shadowSide.map { wrapped[row + $0].z }.reduce(0, +)
        #expect(shadowSide.allSatisfy { diffuse[row + $0].x == 0 })
        #expect(redWrap > 0.05 && redWrap > 4 * blueWrap * color.x / color.z, "red \(redWrap), blue \(blueWrap)")
        // Energy-conserving wrap keeps the lit hemisphere's total red within 10% of diffuse.
        // The energy-conserving wrap max(0, cos + w) / (1 + w)^2 halves red where the light is
        // normal (w = 1 for a diffusion length beyond the radius) and leaves blue (w near 0) diffuse.
        let edge = (width / 2 + 20)...(width / 2 + 22)
        let redRatio = edge.map { wrapped[row + $0].x }.reduce(0, +) / edge.map { diffuse[row + $0].x }.reduce(0, +)
        let blueRatio = edge.map { wrapped[row + $0].z }.reduce(0, +) / edge.map { diffuse[row + $0].z }.reduce(0, +)
        #expect(redRatio > 0.45 && redRatio < 0.62, "red \(redRatio)")
        #expect(blueRatio > 0.95 && blueRatio < 1.01, "blue \(blueRatio)")

        // Back light through a diffuse-transmission volume dims with its thickness.
        func backLit(thickness: Float) throws -> Float {
            let sheet = try PhysicallyBasedMaterial(
                roughness: 0.6, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, isDoubleSided: true,
                transmission: .init(factor: 0, thickness: thickness, attenuationDistance: 0.05, attenuationColor: SIMD3(0.8, 0.5, 0.3)),
                specular: .init(factor: 0), diffuseTransmission: .init(factor: 1),
                volumeScatter: .init(multiscatterColor: SIMD3(0.9, 0.7, 0.5))
            )
            let pixels = try fixture.rasterize(SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: try fixture.renderer(try MeshGeometry.plane(width: 3, height: 3), sheet)),
                // Light from behind the sheet, traveling toward the camera (+Z).
                SceneNode(transform: .init(rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))),
                          light: try PunctualLight(kind: .directional, intensity: 3)),
            ], camera: camera), width: width, height: width)
            return pixels[row + width / 2].x
        }
        let thin = try backLit(thickness: 0.01)
        let thick = try backLit(thickness: 1)
        // Red attenuates by the diffusion coefficient sqrt(3 sigma_a (sigma_a + sigma_s)), far below
        // its extinction, because its multiple-scattering color is nearly white.
        let extinction = -log(0.8) / 0.05
        let albedo = MediaReference.singleScatteringAlbedo(0.9)
        let absorption = extinction * (1 - albedo)
        let transport = min(extinction, (3 * absorption * (absorption + extinction * albedo)).squareRoot())
        let ratio = Double(thick / thin)
        #expect(abs(ratio - exp(-transport * 0.99)) < 0.01, "thin \(thin), thick \(thick), expected ratio \(exp(-transport * 0.99))")
        #expect(ratio < 0.8)
    }

    @Test("Material validation rejects out-of-range translucency parameters and conflicting volumes")
    func validation() throws {
        typealias Material = PhysicallyBasedMaterial
        #expect(throws: Material.Failure.invalidDiffuseTransmission) { try Material.DiffuseTransmission(factor: 1.1) }
        #expect(throws: Material.Failure.invalidDiffuseTransmission) { try Material.DiffuseTransmission(colorFactor: SIMD3(1, -0.1, 1)) }
        #expect(throws: Material.Failure.invalidDiffuseRoughness) { try Material.DiffuseRoughness(factor: -0.1) }
        #expect(throws: Material.Failure.invalidVolumeScatter) { try Material.VolumeScatter(anisotropy: 1) }
        #expect(throws: Material.Failure.invalidVolumeScatter) { try Material.VolumeScatter(multiscatterColor: SIMD3(2, 0, 0)) }
        #expect(throws: Material.Failure.invalidSubsurface) { try Material.Subsurface(radius: SIMD3(1, .nan, 1)) }
        #expect(throws: Material.Failure.invalidSubsurface) { try Material.Subsurface(scale: -1) }
        #expect(throws: Material.Failure.invalidSubsurface) { try Material.Subsurface(anisotropy: -1) }
        #expect(throws: Material.Failure.volumeScatterWithoutVolume) {
            try Material(volumeScatter: .init(multiscatterColor: SIMD3(repeating: 0.5)))
        }
        #expect(throws: Material.Failure.volumeScatterWithoutVolume) {
            try Material(transmission: .init(factor: 1, thickness: 0), volumeScatter: .init(multiscatterColor: SIMD3(repeating: 0.5)))
        }
        #expect(throws: Material.Failure.subsurfaceConflict) {
            try Material(transmission: .init(factor: 1, thickness: 1), subsurface: .init())
        }
        #expect(throws: Material.Failure.subsurfaceConflict) {
            try Material(diffuseTransmission: .init(factor: 0.5), subsurface: .init())
        }
        let subsurface = try Material.Subsurface(radius: SIMD3(1, 0.5, 0.25), scale: 0.02)
        #expect(subsurface.meanFreePath == SIMD3(0.02, 0.01, 0.005))
        let liveVolume = try Material(
            transmission: .init(factor: 1, thickness: 1),
            volumeScatter: .init(multiscatterColor: SIMD3(repeating: 0.5))
        ).withLiveFactors()
        let live = try #require(liveVolume.liveFactors)
        try live.update { $0.transmissionFactor = 0 }
        #expect(live.current.transmissionFactor == 0)
        try live.update { $0.transmissionFactor = 1 }
        #expect(live.current.transmissionFactor == 1)
        // Unlit materials ignore every optical layer.
        #expect(try Material(shadingModel: .unlit, subsurface: subsurface).subsurface != nil)
    }
}
