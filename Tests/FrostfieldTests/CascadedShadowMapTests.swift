import Foundation
import Frostfield
import Metal
import Testing
import simd

struct CascadedShadowSplitTests {
    @Test("Practical splits blend logarithmic and uniform distances and end at the far distance")
    func practicalSplits() throws {
        let splits = try DirectionalShadowMap.practicalSplits(near: 0.1, far: 100, count: 4, lambda: 0.75)
        let expected: [Double] = (1...4).map { index in
            let fraction = Double(index) / 4
            return 0.75 * 0.1 * pow(1000, fraction) + 0.25 * (0.1 + 99.9 * fraction)
        }
        #expect(splits.count == 4)
        for (split, value) in zip(splits, expected) {
            #expect(abs(Double(split) - value) < 1e-4)
        }
        #expect(splits.last == 100)
        let uniform = try DirectionalShadowMap.practicalSplits(near: 1, far: 9, count: 2, lambda: 0)
        #expect(uniform == [5, 9])
        let logarithmic = try DirectionalShadowMap.practicalSplits(near: 1, far: 100, count: 2, lambda: 1)
        #expect(abs(logarithmic[0] - 10) < 1e-4)
        #expect(try DirectionalShadowMap.practicalSplits(near: 1, far: 9, count: 1, lambda: 0.5) == [9])
    }

    @Test(
        "Practical splits reject ranges and parameters they cannot divide",
        arguments: [
            (Float(0), Float(100), 4, Float(0.75)), (-1, 100, 4, 0.75), (10, 10, 2, 0.5), (10, 5, 2, 0.5),
            (.nan, 100, 2, 0.5), (1, .infinity, 2, 0.5), (1, 100, 0, 0.5), (1, 100, 5, 0.5),
            (1, 100, 2, -0.1), (1, 100, 2, 1.1), (1, 100, 2, .nan),
        ] as [(Float, Float, Int, Float)]
    )
    func invalidSplits(near: Float, far: Float, count: Int, lambda: Float) {
        #expect(throws: DirectionalShadowMap.Failure.invalidSplits) {
            try DirectionalShadowMap.practicalSplits(near: near, far: far, count: count, lambda: lambda)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CascadedShadowMapTests {
    private let fixture: RasterLightingFixture

    init() throws {
        fixture = try RasterLightingFixture(width: 512, height: 512)
    }

    @Test("Cascades are validated, must be fitted before drawing, and keep single-map updates separate")
    func validation() throws {
        let queue = fixture.queue
        #expect(throws: DirectionalShadowMap.Failure.invalidCascades) {
            try DirectionalShadowMap(
                queue: queue, resolution: 64, directionToLight: SIMD3(0, 1, 0), cascades: .init(count: 5)
            )
        }
        #expect(throws: DirectionalShadowMap.Failure.invalidCascades) {
            try DirectionalShadowMap(
                queue: queue, resolution: 64, directionToLight: SIMD3(0, 1, 0), cascades: .init(blendFraction: 0.6)
            )
        }
        let map = try DirectionalShadowMap(
            queue: queue, resolution: 64, directionToLight: SIMD3(0, 1, 0), cascades: .init(count: 2)
        )
        #expect(throws: DirectionalShadowMap.Failure.cascaded) {
            try map.update(directionToLight: SIMD3(0, 1, 0), halfExtent: 1)
        }
        let single = try DirectionalShadowMap(queue: queue, resolution: 64, directionToLight: SIMD3(0, 1, 0), halfExtent: 1)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 50)
        #expect(throws: DirectionalShadowMap.Failure.notCascaded) {
            try single.fitCascades(camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1)
        }
        #expect(throws: DirectionalShadowMap.Failure.invalidCamera) {
            try map.fitCascades(camera: camera, cameraTransform: SceneTransform(), aspectRatio: 0)
        }

        let ground = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .shadowedLambert
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: ground)], camera: camera
        )
        #expect(throws: DirectionalShadowMap.Failure.cascadesNotFitted) {
            _ = try fixture.render(graph, shadowMap: map)
        }
        try map.fitCascades(camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1)
        #expect(map.cascadeSplits.count == 2)
        #expect(map.cascadeSplits.last == 50)
        _ = try fixture.render(graph, shadowMap: map)
        // Re-aiming refits the same camera; sphere radii, and so texel sizes, depend only on the camera.
        let texels = map.texelSizes
        try map.update(directionToLight: SIMD3(1, 1, 0))
        #expect(map.texelSizes == texels)
        #expect(throws: DirectionalShadowMap.Failure.notCascaded) {
            try single.update(directionToLight: SIMD3(0, 1, 0))
        }
    }

    enum Receiver: Sendable {
        case shadowedLambert, physicallyBased, classicLambert
    }

    @Test(
        "Cascades keep near and far shadow edges sharp where one map covering both is coarse near the camera",
        arguments: [Receiver.shadowedLambert, .physicallyBased, .classicLambert]
    )
    func penumbraWidth(receiver: Receiver) throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 200)
        let cameraTransform = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2, 0), pitch: -.pi / 12)
        let directionToLight = simd_normalize(SIMD3<Float>(1, 1, 0))
        let device = fixture.device
        let material: MeshMaterial = switch receiver {
        case .shadowedLambert: .shadowedLambert
        case .physicallyBased: .physicallyBased(try .init(roughness: 0.7, receivesShadows: true))
        case .classicLambert:
            .classic(try .init(model: .lambert, ambient: SIMD3(repeating: 0.16 * .pi), receivesShadows: true))
        }
        let ground = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 400, height: 400),
            material: material
        )
        func plate(width: Float, depth: Float) throws -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: width, height: depth),
                material: .unlit
            )
        }
        // Plates at height h cast their shadow h units toward -X with this light.
        let near = SceneNode(
            transform: SceneTransform(
                translation: SIMD3(0.5, 1, -4), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            ),
            mesh: try plate(width: 0.6, depth: 0.6)
        )
        let far = SceneNode(
            transform: SceneTransform(
                translation: SIMD3(6, 6, -60), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            ),
            mesh: try plate(width: 8, depth: 30)
        )
        var graph = SceneGraph(
            roots: [
                SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: ground, castsShadow: false),
                near, far,
            ],
            camera: camera
        )
        graph.cameraTransform = cameraTransform

        let single = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 2048, directionToLight: directionToLight,
            center: SIMD3(0, 0, -40), halfExtent: 50
        )
        // Four 1024² cascades hold as many texels as the single 2048² map.
        let cascaded = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 1024, directionToLight: directionToLight,
            cascades: .init(count: 4, splitLambda: 0.75, maximumDistance: 100)
        )
        try cascaded.fitCascades(camera: camera, cameraTransform: cameraTransform, aspectRatio: 1)

        let singlePixels = try fixture.pixels(of: fixture.render(graph, shadowMap: single).color)
        let cascadedPixels = try fixture.pixels(of: fixture.render(graph, shadowMap: cascaded).color)
        fixture.writeImage(singlePixels, name: "csm-single-\(receiver)")
        fixture.writeImage(cascadedPixels, name: "csm-cascaded-\(receiver)")

        // Shadowed Lambert reads 0.16 + 0.84 · cos(45°) · visibility; PBR adds its direct light
        // to an ambient fill. Read both levels at the band's middle and just outside it.
        func edgeWidths(_ pixels: [SIMD4<Float>], center: SIMD3<Float>, halfWidth: Float) -> [Int] {
            let left = fixture.project(center - SIMD3(halfWidth + 0.25 * halfWidth, 0, 0), camera: camera, transform: cameraTransform)
            let right = fixture.project(center + SIMD3(halfWidth + 0.25 * halfWidth, 0, 0), camera: camera, transform: cameraTransform)
            let row = Int(left.y)
            let middle = fixture.project(center, camera: camera, transform: cameraTransform)
            let shadowed = pixels[Int(middle.y) * fixture.width + Int(middle.x)].x
            let lit = pixels[row * fixture.width + Int(left.x)].x
            if receiver == .shadowedLambert {
                #expect(abs(shadowed - 0.16) < 0.01)
                #expect(abs(lit - (0.16 + 0.84 * directionToLight.y)) < 0.01)
            }
            #expect(lit > shadowed + 0.1)
            var runs: [Int] = []
            var run = 0
            for x in Int(left.x)...Int(right.x) {
                let value = pixels[row * fixture.width + x].x
                let fraction = (value - shadowed) / (lit - shadowed)
                if fraction > 0.1, fraction < 0.9 {
                    run += 1
                } else if run > 0 {
                    runs.append(run)
                    run = 0
                }
            }
            if run > 0 { runs.append(run) }
            return runs
        }
        let nearShadow = SIMD3<Float>(-0.5, 0, -4)
        let farShadow = SIMD3<Float>(0, 0, -60)
        let singleNear = edgeWidths(singlePixels, center: nearShadow, halfWidth: 0.3)
        let cascadedNear = edgeWidths(cascadedPixels, center: nearShadow, halfWidth: 0.3)
        let singleFar = edgeWidths(singlePixels, center: farShadow, halfWidth: 4)
        let cascadedFar = edgeWidths(cascadedPixels, center: farShadow, halfWidth: 4)

        // A shadow-map texel spreads its comparison over 0.8 texel between the 10% and 90% levels.
        // Near the camera that is about one pixel in the cascade, so the ramp covers one or two
        // pixels depending on where the edge falls in the texel grid; the single map's is wider.
        #expect(singleNear.count == 2 && singleNear.allSatisfy { $0 >= 3 })
        let singleNearest = singleNear.min() ?? 0
        #expect(cascadedNear.allSatisfy { $0 <= 2 && 2 * $0 <= singleNearest })
        #expect(cascadedFar.allSatisfy { $0 <= 2 })
        #expect(singleFar.allSatisfy { $0 <= 2 })
    }

    @Test("Sub-texel camera moves leave cascaded shadow edges fixed in the world",
          arguments: [false, true])
    func stableUnderTranslation(_ highPrecision: Bool) throws {
        let fixture = try RasterLightingFixture(width: 256, height: 256)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 60)
        let height: Float = 8
        let device = fixture.device
        let ground = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 100, height: 100),
            material: .shadowedLambert
        )
        let plate = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1.3, height: 0.9), material: .unlit
        )
        // Rotated casters give edges at several angles to the shadow texel grid.
        var roots = [SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: ground, castsShadow: false)]
        for (index, offset) in [SIMD3<Float>(-1.5, 1.2, -0.5), SIMD3(1.4, 0.8, 1.1), SIMD3(0.3, 1.6, -2.2)].enumerated() {
            let flat = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            let turn = simd_quatf(angle: Float(index) * 0.55 + 0.3, axis: SIMD3(0, 1, 0))
            roots.append(SceneNode(transform: SceneTransform(translation: offset, rotation: turn * flat), mesh: plate))
        }
        var graph = SceneGraph(roots: roots, camera: camera)
        graph.highPrecision = highPrecision
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.3, 1, 0.2),
            cascades: .init(count: 3, splitLambda: 0.75, maximumDistance: 20)
        )
        // One pixel at the ground's depth, looking straight down.
        let pixel = 2 * height * tan(camera.verticalFieldOfView / 2) / Float(fixture.width)
        func render(offset: Float) throws -> [SIMD4<Float>] {
            let transform = RasterLightingFixture.cameraTransform(position: SIMD3(offset, height, 0), pitch: -.pi / 2)
            graph.cameraTransform = transform
            try map.fitCascades(camera: camera, cameraTransform: transform, aspectRatio: 1)
            return try fixture.pixels(of: fixture.render(graph, shadowMap: map).color)
        }
        let before = try render(offset: 0)
        let after = try render(offset: pixel)
        let variant = highPrecision ? "relative" : "world"
        fixture.writeImage(before, name: "csm-stable-\(variant)-before")
        fixture.writeImage(after, name: "csm-stable-\(variant)-after")
        // The ground (view distance 8) lies in the third cascade; the move is a fraction of its texel.
        let cascadeIndex = try #require(map.cascadeSplits.firstIndex { $0 >= height })
        let texelFraction = pixel / map.texelSizes[cascadeIndex]
        #expect(cascadeIndex == 2)
        #expect(texelFraction > 0.2 && texelFraction < 0.8)

        // The camera moved one pixel along +X, so the ground moved one pixel left. The
        // nearer plates move farther with parallax, so compare gray ground pixels only.
        var largest: Float = 0
        var edgePixels = 0
        for y in 0..<fixture.height {
            for x in 0..<(fixture.width - 1) {
                let a = before[y * fixture.width + x + 1]
                let b = after[y * fixture.width + x]
                guard abs(a.x - a.y) < 0.01, abs(b.x - b.y) < 0.01 else { continue }
                largest = max(largest, simd_reduce_max(abs(a - b)))
                // Partially shadowed gray ground, not the green-tinted plates.
                let fraction = (a.x - 0.16) / (0.84 * simd_normalize(SIMD3<Float>(0.3, 1, 0.2)).y)
                if abs(a.x - a.y) < 0.01, fraction > 0.05, fraction < 0.95 { edgePixels += 1 }
            }
        }
        #expect(edgePixels > 100)
        #expect(largest < 0.002)
    }

    /// Shadowed Lambert ground under a light straight overhead reads 0.16 + 0.84 · visibility.
    private static func visibility(_ pixel: SIMD4<Float>) -> Float {
        (pixel.x - 0.16) / 0.84
    }

    /// A camera 2 units above the ground, pitched down, a shadowed Lambert ground, and
    /// horizontal unlit plates that cast onto it. The light is straight overhead.
    private func overheadScene(
        camera: PerspectiveCamera, cameraTransform: SceneTransform,
        plates: [(center: SIMD3<Float>, size: SIMD2<Float>)], depthMode: MeshDepthMode = .forward
    ) throws -> SceneGraph {
        let ground = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 200, height: 200),
            material: .shadowedLambert, depthMode: depthMode
        )
        var roots = [SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: ground, castsShadow: false)]
        for plate in plates {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: .plane(width: plate.size.x, height: plate.size.y), material: .unlit, depthMode: depthMode
            )
            roots.append(SceneNode(
                transform: SceneTransform(translation: plate.center, rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
                mesh: mesh
            ))
        }
        var graph = SceneGraph(roots: roots, camera: camera)
        graph.cameraTransform = cameraTransform
        return graph
    }

    /// The ground point under the center of row `y` and its distance along the view axis.
    private func groundRow(
        _ y: Int, x: Int, camera: PerspectiveCamera, transform: SceneTransform
    ) -> (point: SIMD3<Float>, viewDistance: Float) {
        let ray = fixture.ray(x: x, y: y, camera: camera, transform: transform)
        let point = ray.origin + (-ray.origin.y / ray.direction.y) * ray.direction
        let forward = transform.rotation.act(SIMD3<Float>(0, 0, -1))
        return (point, simd_dot(point - transform.translation, forward))
    }

    @Test("Custom splits use the capped view range and a rejected refit preserves the previous layout")
    func customPerspectiveSplits() throws {
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 64, directionToLight: SIMD3(0, 1, 0),
            cascades: .init(count: 3, maximumDistance: 20)
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 4, far: 40)
        let pose = SceneTransform()
        try map.fitCascades(camera: camera, cameraTransform: pose, aspectRatio: 1, splitFractions: [0.125, 0.75])
        #expect(map.cascadeSplits == [6, 16, 20])
        let texels = map.texelSizes
        for fractions: [Float] in [[0.2], [0, 0.7], [0.5, 0.5], [0.8, 0.2], [0.5, 1], [0.5, .nan]] {
            #expect(throws: DirectionalShadowMap.Failure.invalidSplits) {
                try map.fitCascades(camera: camera, cameraTransform: pose, aspectRatio: 1, splitFractions: fractions)
            }
            #expect(map.cascadeSplits == [6, 16, 20])
            #expect(map.texelSizes == texels)
        }
        try map.update(directionToLight: SIMD3(1, 1, 0))
        #expect(map.cascadeSplits == [6, 16, 20])
        #expect(map.texelSizes == texels)
    }

    @Test("Shifted orthographic views use custom cascade splits and fade at the fitted distance", arguments: [
        MeshDepthMode.forward, .reversed,
    ])
    func orthographicCascades(depthMode: MeshDepthMode) throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 100)
        let pose = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2, 0), pitch: -.pi / 6)
        let graph = try overheadScene(
            camera: camera, cameraTransform: pose, plates: [(SIMD3(0, 6, -8), SIMD2(40, 30))], depthMode: depthMode
        )
        let reversed = depthMode == .reversed
        let projection = simd_float4x4(columns: (
            SIMD4(1 / Float(6), 0, 0, 0), SIMD4(0, 1 / Float(3), 0, 0),
            SIMD4(0, 0, reversed ? 0.05 : -0.05, 0), SIMD4(-0.35, 0, reversed ? 1 : 0, 1)
        ))
        let view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection, near: 0, far: 20
        )
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0, 1, 0),
            cascades: .init(count: 2, maximumDistance: 7, blendFraction: 0.25)
        )
        try map.fitCascades(view: view)
        #expect(map.cascadeSplits == [3.5, 7])
        try map.fitCascades(view: view, splitFractions: [0.25])
        #expect(map.cascadeSplits == [1.75, 7])

        let color = try fixture.colorTarget()
        let depth = try fixture.depthTarget()
        let towardCamera = pose.rotation.act(SIMD3(0, 0, 1))
        try fixture.run { buffer in
            try graph.encode(
                pass: fixture.pass(color: color, depth: depth, depthMode: depthMode), commandBuffer: buffer,
                clipFromWorld: projection * view.viewMatrix, cameraPosition: pose.translation,
                directionToCamera: towardCamera, depthMode: depthMode, shadowMap: map, graphCamera: view
            )
        }
        let pixels = try fixture.pixels(of: color)
        let band: Float = (7 - 1.75) * 0.25
        var fadedRows = 0
        var shadowedRows = 0
        var outsideRows = 0
        var largestError: Float = 0
        for y in 0..<fixture.height {
            let localY = 3 * (1 - 2 * (Float(y) + 0.5) / Float(fixture.height))
            let rayOrigin = pose.translation + pose.rotation.act(SIMD3(0, localY, 0))
            let distance = rayOrigin.y / towardCamera.y
            guard distance > 0.4, distance < 8.5 else { continue }
            let expected = min(max((distance - (7 - band)) / band, 0), 1)
            for x in [20, fixture.width / 2, fixture.width - 21] {
                let measured = Self.visibility(pixels[y * fixture.width + x])
                largestError = max(largestError, abs(measured - expected))
            }
            if expected > 0.05, expected < 0.95 { fadedRows += 1 }
            if distance < 7 - band { shadowedRows += 1 }
            if distance > 7 { outsideRows += 1 }
        }
        #expect(fadedRows > 20)
        #expect(shadowedRows > 20)
        #expect(outsideRows > 20)
        #expect(largestError < 0.01)
    }

    @Test(
        "The last cascade fades to unshadowed over its blend band and ends at the maximum distance",
        arguments: [1, 2]
    )
    func lastCascadeFade(count: Int) throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 100)
        let transform = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2, 0), pitch: -.pi / 12)
        // A plate 3 units up shades the ground from z = −2 to z = −40, well past the cutoff.
        let graph = try overheadScene(
            camera: camera, cameraTransform: transform, plates: [(SIMD3(0, 3, -21), SIMD2(8, 38))]
        )
        let cascades = DirectionalShadowMap.Cascades(count: count, splitLambda: 0.75, maximumDistance: 10, blendFraction: 0.2)
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 1024, directionToLight: SIMD3(0, 1, 0), cascades: cascades
        )
        try map.fitCascades(camera: camera, cameraTransform: transform, aspectRatio: 1)
        let pixels = try fixture.pixels(of: fixture.render(graph, shadowMap: map).color)
        fixture.writeImage(pixels, name: "csm-last-fade-\(count)")
        let lastNear = count == 1 ? camera.near : map.cascadeSplits[count - 2]
        let band = (10 - lastNear) * 0.2
        var faded = 0
        var beyond = 0
        var largest: Float = 0
        for y in 0..<fixture.height {
            let (point, distance) = groundRow(y, x: fixture.width / 2, camera: camera, transform: transform)
            guard distance > 3, distance < 14, point.z < -2.5 else { continue }
            let expected = min(max((distance - (10 - band)) / band, 0), 1)
            let measured = Self.visibility(pixels[y * fixture.width + fixture.width / 2])
            #expect(abs(measured - expected) < 0.01, "view distance \(distance)")
            largest = max(largest, abs(measured - expected))
            if expected > 0.05, expected < 0.95 { faded += 1 }
            if distance > 10 { beyond += 1 }
        }
        #expect(faded >= 5)
        #expect(beyond >= 5)
        fixture.note("last cascade fade, \(count) cascades: \(faded) rows in the band, \(beyond) beyond, largest error \(largest)")
    }

    @Test("Inside the first cascade's blend band, shading mixes both cascades by the band's linear weight")
    func blendBandMixesCascades() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 1, far: 100)
        let transform = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2, 0), pitch: -.pi / 12)
        // A plate 6 units up lies above the first cascade's depth range but inside the second's,
        // so the first cascade sees the ground lit and the second sees it shadowed.
        let plateCenter = SIMD3<Float>(0, 6, -5.5)
        let graph = try overheadScene(
            camera: camera, cameraTransform: transform, plates: [(plateCenter, SIMD2(8, 9))]
        )
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 1024, directionToLight: SIMD3(0, 1, 0),
            cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 12, blendFraction: 0.5, casterMargin: 0)
        )
        try map.fitCascades(camera: camera, cameraTransform: transform, aspectRatio: 1)
        func lightDepth(_ point: SIMD3<Float>, cascade: Int) -> Float {
            let clip = map.cascadeViews[cascade] * SIMD4(point, 1)
            return clip.z / clip.w
        }
        try #require(lightDepth(plateCenter, cascade: 0) < 0)
        try #require((0...1).contains(lightDepth(plateCenter, cascade: 1)))
        try #require((0...1).contains(lightDepth(SIMD3(0, 0, -3), cascade: 0)))

        let pixels = try fixture.pixels(of: fixture.render(graph, shadowMap: map).color)
        fixture.writeImage(pixels, name: "csm-blend-band")
        // The band covers the last half of the first cascade's range, which starts at the near plane.
        let split = map.cascadeSplits[0]
        let bandStart = split - 0.5 * (split - camera.near)
        var inBand = 0
        var largest: Float = 0
        for y in 0..<fixture.height {
            let (_, distance) = groundRow(y, x: fixture.width / 2, camera: camera, transform: transform)
            guard distance > 2.2, distance < 6.5 else { continue }
            let expected = distance < bandStart ? 1 : distance < split ? 1 - (distance - bandStart) / (split - bandStart) : 0
            let measured = Self.visibility(pixels[y * fixture.width + fixture.width / 2])
            #expect(abs(measured - expected) < 0.01, "view distance \(distance)")
            largest = max(largest, abs(measured - expected))
            if expected > 0.05, expected < 0.95 { inBand += 1 }
        }
        #expect(inBand >= 10)
        fixture.note("blend band: split \(split), band start \(bandStart), \(inBand) rows in the band, largest error \(largest)")
    }

    @Test("Contact hardening gives the same penumbra width in metres in near and far cascades")
    func contactHardeningAcrossCascades() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.5, far: 100)
        let transform = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2, 0), pitch: -0.35)
        let height: Float = 1
        let angularRadius: Float = 0.15
        // Plates 1 unit above the ground near the camera and far from it.
        let nearShadow = SIMD3<Float>(0, 0, -3)
        let farShadow = SIMD3<Float>(0, 0, -12)
        let graph = try overheadScene(
            camera: camera, cameraTransform: transform,
            plates: [(nearShadow + SIMD3(0, height, 0), SIMD2(1, 0.8)), (farShadow + SIMD3(0, height, 0), SIMD2(2, 4))]
        )
        let map = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 1024, directionToLight: SIMD3(0, 1, 0),
            cascades: .init(count: 2, splitLambda: 0.5, maximumDistance: 20), lightAngularRadius: angularRadius
        )
        try map.fitCascades(camera: camera, cameraTransform: transform, aspectRatio: 1)
        let forward = transform.rotation.act(SIMD3<Float>(0, 0, -1))
        let nearDistance = simd_dot(nearShadow - transform.translation, forward)
        let farDistance = simd_dot(farShadow - transform.translation, forward)
        // Each shadow lies inside one cascade, away from the blend band.
        try #require(nearDistance < 0.9 * map.cascadeSplits[0] && farDistance > 1.1 * map.cascadeSplits[0])
        let pixels = try fixture.pixels(of: fixture.render(graph, shadowMap: map).color)
        fixture.writeImage(pixels, name: "csm-contact-hardening")

        /// The mean 10%–90% width, in metres along X, of both side edges over rows crossing a shadow.
        func meanWidth(center: SIMD3<Float>, halfWidth: Float, halfDepth: Float) -> (width: Float, rows: Int) {
            let top = Int(fixture.project(center - SIMD3(0, 0, 0.6 * halfDepth), camera: camera, transform: transform).y)
            let bottom = Int(fixture.project(center + SIMD3(0, 0, 0.6 * halfDepth), camera: camera, transform: transform).y)
            var widths: [Float] = []
            for y in top...bottom {
                let row = (0..<fixture.width).map { x in
                    (x: groundRow(y, x: x, camera: camera, transform: transform).point.x,
                     v: Self.visibility(pixels[y * fixture.width + x]))
                }
                for side: Float in [-1, 1] {
                    // Walk outward from the shadow's middle until the ground is lit.
                    let samples = row.filter { $0.x * side >= 0 && abs($0.x) < halfWidth + 0.6 }
                        .sorted { abs($0.x) < abs($1.x) }
                    func crossing(_ level: Float) -> Float? {
                        for (inner, outer) in zip(samples, samples.dropFirst()) where inner.v < level && outer.v >= level {
                            return abs(inner.x) + (level - inner.v) / (outer.v - inner.v) * (abs(outer.x) - abs(inner.x))
                        }
                        return nil
                    }
                    if let low = crossing(0.1), let high = crossing(0.9) { widths.append(high - low) }
                }
            }
            return (widths.reduce(0, +) / Float(max(widths.count, 1)), widths.count)
        }
        let near = meanWidth(center: nearShadow, halfWidth: 0.5, halfDepth: 0.4)
        let far = meanWidth(center: farShadow, halfWidth: 1, halfDepth: 2)
        // A disk filter of radius h·tan θ spreads a straight edge's 10%–90% over 1.374 radii.
        let predicted = 1.374 * height * tan(angularRadius)
        #expect(near.rows >= 20 && far.rows >= 20)
        fixture.note(
            "contact hardening: near \(near.width) m over \(near.rows) edges, far \(far.width) m over \(far.rows), "
                + "predicted \(predicted) m, texels \(map.texelSizes)"
        )
        #expect(abs(near.width - far.width) < 0.15 * predicted, "near \(near.width) m, far \(far.width) m")
        #expect(abs(near.width - predicted) < 0.25 * predicted, "near \(near.width) m, predicted \(predicted) m")
        #expect(abs(far.width - predicted) < 0.25 * predicted, "far \(far.width) m, predicted \(predicted) m")
    }
}
