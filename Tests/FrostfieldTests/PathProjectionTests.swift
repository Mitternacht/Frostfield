import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathProjectionTests {
    private let fixture: PathCameraFixture

    init() throws {
        fixture = try PathCameraFixture()
    }

    /// Emissive one-meter cubes five meters out along each axis, each its own color.
    private static let markers: [(axis: SIMD3<Float>, color: SIMD3<Float>)] = [
        (SIMD3(1, 0, 0), SIMD3(1, 0, 0)), (SIMD3(-1, 0, 0), SIMD3(0, 1, 1)),
        (SIMD3(0, 1, 0), SIMD3(0, 1, 0)), (SIMD3(0, -1, 0), SIMD3(1, 0, 1)),
        (SIMD3(0, 0, -1), SIMD3(0, 0, 1)), (SIMD3(0, 0, 1), SIMD3(1, 1, 0)),
    ]

    private func tracer(_ nodes: [SceneNode]) throws -> PathTracer {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.01, far: 100)
        let scene = try PathTracingScene(queue: fixture.queue, graph: SceneGraph(roots: nodes, camera: camera), textureSize: 16)
        return try PathTracer(queue: fixture.queue, scene: scene)
    }

    /// The equirectangular pixel a camera-space direction lands on: longitude from −Z toward +X, latitude toward +Y.
    private static func equirectangular(_ direction: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let d = simd_normalize(direction)
        let longitude = atan2(d.x, -d.z)
        let latitude = asin(d.y)
        return SIMD2((0.5 + longitude / (2 * .pi)) * Float(width), (0.5 - latitude / .pi) * Float(height))
    }

    @Test("An equirectangular camera puts each axis marker at its predicted pixel and turns with the camera")
    func equirectangularAxes() throws {
        let tracer = try tracer(try Self.markers.map { marker in
            SceneNode(
                transform: .init(translation: marker.axis * 5),
                mesh: try fixture.lambert(.cube(), albedo: 0, emission: marker.color)
            )
        })
        let width = 128
        let height = 64
        let model = PathTracer.CameraModel(projection: .equirectangular, near: 0.01, far: 100)
        func color(_ image: [SIMD4<Float>], at point: SIMD2<Float>) -> SIMD3<Float> {
            let column = min(max(Int(point.x), 0), width - 1)
            let row = min(max(Int(point.y), 0), height - 1)
            let pixel = image[row * width + column]
            return SIMD3(pixel.x, pixel.y, pixel.z)
        }
        for yaw in [Float(0), .pi / 2] {
            let rotation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
            let image = try fixture.trace(
                tracer, model: model, transform: .init(rotation: rotation), width: width, height: height, samples: 16
            )
            PathCameraFixture.writeImage(image, width: width, height: height, name: "equirect-yaw\(Int(yaw * 100))")
            for marker in Self.markers {
                // The marker's direction in camera space; poles and the seam are checked a pixel inside.
                let local = rotation.inverse.act(marker.axis)
                var point = Self.equirectangular(local, width: width, height: height)
                point.y = min(max(point.y, 0.5), Float(height) - 0.5)
                if point.x >= Float(width) - 0.5 { point.x = Float(width) - 0.5 }
                let seen = color(image, at: point)
                #expect(simd_distance(seen, marker.color) < 0.02, "yaw \(yaw): \(marker.axis) at \(point) shows \(seen)")
            }
            // Directions between markers see the black background.
            #expect(color(image, at: Self.equirectangular(SIMD3(1, 0, -1), width: width, height: height)) == .zero)
        }
    }

    @Test("Omnidirectional stereo offsets each eye's view of a near marker by the ODS disparity")
    func omnidirectionalStereo() throws {
        let tracer = try tracer([SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)),
            mesh: try fixture.lambert(.uvSphere(radius: 0.04, segments: 24, rings: 12), albedo: 0, emission: SIMD3(1, 1, 1))
        )])
        let side = 256
        let separation: Float = 1
        let model = PathTracer.CameraModel(projection: .stereoEquirectangular(interpupillaryDistance: separation), near: 0.01, far: 100)
        let image = try fixture.trace(tracer, model: model, width: side, height: side, samples: 64)
        PathCameraFixture.writeImage(image, width: side, height: side, name: "ods-top-bottom")
        func centroid(rows: Range<Int>) -> SIMD2<Float> {
            var sum = SIMD3<Float>.zero
            for row in rows {
                for column in 0..<side {
                    let value = image[row * side + column].x
                    sum += value * SIMD3(Float(column) + 0.5, Float(row) + 0.5, 1)
                }
            }
            return SIMD2(sum.x, sum.y) / sum.z
        }
        // A ray tangent to the eye circle of radius r passes a point at distance d ahead when sin(longitude) = ∓r / d.
        let longitude = asin(separation / 2 / 2)
        let left = centroid(rows: 0..<(side / 2))
        let right = centroid(rows: (side / 2)..<side)
        let expectedLeft = (0.5 + longitude / (2 * .pi)) * Float(side)
        let expectedRight = (0.5 - longitude / (2 * .pi)) * Float(side)
        #expect(abs(left.x - expectedLeft) < 0.3, "left eye column \(left.x) vs \(expectedLeft)")
        #expect(abs(right.x - expectedRight) < 0.3, "right eye column \(right.x) vs \(expectedRight)")
        // Each eye keeps the horizon on its own middle row.
        #expect(abs(left.y - Float(side) / 4) < 0.3 && abs(right.y - 3 * Float(side) / 4) < 0.3, "\(left) \(right)")

        // The stereo layout needs an even height; lenses belong to perspective projections.
        let odd = try tracer.makeAccumulation(width: 8, height: 7)
        let target = try fixture.base.output(width: 8, height: 7)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.camera) {
            try tracer.encode(cameraModel: model, cameraTransform: .init(), destination: target, accumulation: odd, commandBuffer: buffer)
        }
        var lensed = PathTracer.CameraModel(projection: .equirectangular)
        lensed.lens = .init(apertureRadius: 0.01, focusDistance: 2)
        #expect(throws: PathTracer.Failure.camera) {
            try tracer.encode(cameraModel: lensed, cameraTransform: .init(), destination: target, accumulation: odd, commandBuffer: buffer)
        }
        #expect(throws: PathTracer.Failure.settings) {
            try tracer.encode(
                cameraModel: .init(projection: .equirectangular), cameraTransform: .init(), destination: target,
                accumulation: odd, commandBuffer: buffer, settings: .init(apertureRadius: 0.1)
            )
        }
    }

    @Test("An orthographic camera images equal squares at different depths at the same size")
    func orthographic() throws {
        let squares = try [SIMD3<Float>(-1, 0, -3), SIMD3(1, 0, -9)].map { position in
            SceneNode(
                transform: .init(translation: position),
                mesh: try fixture.lambert(.plane(width: 1, height: 1), albedo: 0, emission: SIMD3(1, 1, 1))
            )
        }
        let tracer = try tracer(squares)
        let side = 64
        let model = PathTracer.CameraModel(projection: .orthographic(height: 4), near: 0.01, far: 100)
        let image = try fixture.trace(tracer, model: model, width: side, height: side, samples: 16)
        PathCameraFixture.writeImage(image, width: side, height: side, name: "orthographic")
        // 16 pixels per meter: the squares cover columns 8–23 and 40–55 and rows 24–39 exactly.
        for (index, pixel) in image.enumerated() {
            let column = index % side
            let row = index / side
            let inside = (24..<40).contains(row) && ((8..<24).contains(column) || (40..<56).contains(column))
            #expect(abs(pixel.x - (inside ? 1 : 0)) < 1e-3, "pixel \(column), \(row): \(pixel.x)")
        }
        // The far plane clips the second square.
        let clipped = try fixture.trace(
            tracer, model: .init(projection: .orthographic(height: 4), near: 0.01, far: 5), width: side, height: side, samples: 1
        )
        #expect(clipped[32 * side + 48].x == 0 && clipped[32 * side + 16].x == 1)
    }

    private func rasterEye(_ nodes: [SceneNode], view: PhysicalCamera.StereoView, side: Int) throws -> [SIMD4<Float>] {
        let graph = SceneGraph(roots: nodes, camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.05, far: 50))
        let color = try fixture.base.output(width: side, height: side, usage: [.renderTarget, .shaderRead])
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: side, height: side, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(fixture.device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        try graph.encode(
            pass: pass, commandBuffer: buffer, clipFromWorld: view.clipFromWorld(aspectRatio: 1),
            cameraPosition: view.cameraTransform.translation
        )
        try fixture.base.finish(buffer)
        return try fixture.base.readback(color)
    }

    @Test("A stereo pair converges at its convergence distance and shows the predicted disparity elsewhere")
    func stereoPair() throws {
        let markers = try [SIMD3<Float>(0, 0, -2), SIMD3(0, 0.5, -4)].map { position in
            SceneNode(
                transform: .init(translation: position),
                mesh: try fixture.lambert(.uvSphere(radius: 0.02, segments: 16, rings: 8), albedo: 0, emission: SIMD3(1, 1, 1))
            )
        }
        let tracer = try tracer(markers)
        let side = 128
        let camera = try PhysicalCamera(lens: PhysicalLens(focalLength: 50, fNumber: 8, focusDistance: 2), near: 0.05, far: 50)
        let separation: Float = 0.2
        let pair = try camera.stereoPair(cameraTransform: .init(), eyeSeparation: separation, convergenceDistance: 2, aspectRatio: 1)
        #expect(simd_distance(pair.left.cameraTransform.translation, SIMD3(-0.1, 0, 0)) < 1e-6)
        #expect(simd_distance(pair.right.cameraTransform.translation, SIMD3(0.1, 0, 0)) < 1e-6)
        let tanX: Float = 0.24
        for (view, side_) in [(pair.left, Float(-1)), (pair.right, Float(1))] {
            var model = view.cameraModel
            model.lens = nil
            let image = try fixture.trace(tracer, model: model, transform: view.cameraTransform, width: side, height: side, samples: 64)
            PathCameraFixture.writeImage(image, width: side, height: side, name: side_ < 0 ? "stereo-left" : "stereo-right")
            func centroid(rows: Range<Int>) -> SIMD2<Float> {
                var sum = SIMD3<Float>.zero
                for row in rows {
                    for column in 0..<side { sum += image[row * side + column].x * SIMD3(Float(column) + 0.5, Float(row) + 0.5, 1) }
                }
                return SIMD2(sum.x, sum.y) / sum.z
            }
            // An eye at x = ±e/2 sees a point at depth D at ndc ∓(e/2)/(D tanX) ± (e/2)/(c tanX).
            func expectedColumn(depth: Float) -> Float {
                let ndc = -side_ * (separation / 2) / (depth * tanX) + side_ * (separation / 2) / (2 * tanX)
                return (ndc + 1) / 2 * Float(side)
            }
            let near = centroid(rows: 48..<side)
            let far = centroid(rows: 0..<48)
            #expect(abs(near.x - 64) < 0.2 && abs(near.y - 64) < 0.2, "converged marker at \(near)")
            #expect(abs(far.x - expectedColumn(depth: 4)) < 0.2, "far marker column \(far.x) vs \(expectedColumn(depth: 4))")
            #expect(abs(far.y - (1 - 0.5 / (4 * tanX)) / 2 * Float(side)) < 0.2)

            // The raster eye uses the same asymmetric frustum.
            let raster = try rasterEye(markers, view: view, side: side)
            func rasterCentroid(rows: Range<Int>) -> SIMD2<Float> {
                var sum = SIMD3<Float>.zero
                for row in rows {
                    for column in 0..<side { sum += raster[row * side + column].x * SIMD3(Float(column) + 0.5, Float(row) + 0.5, 1) }
                }
                return SIMD2(sum.x, sum.y) / sum.z
            }
            #expect(simd_distance(rasterCentroid(rows: 48..<side), near) < 0.3, "raster converged marker \(rasterCentroid(rows: 48..<side))")
            #expect(simd_distance(rasterCentroid(rows: 0..<48), far) < 0.3, "raster far marker \(rasterCentroid(rows: 0..<48))")
        }
    }
}
