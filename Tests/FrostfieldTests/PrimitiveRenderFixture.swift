import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
import simd

/// Renders primitive scene graphs into a square RGBA16Float target with the
/// default 90° camera at the origin looking down −Z. At depth d a world offset
/// x lands `side / 2 × x / d` pixels from the center, so one world unit at unit
/// depth spans `side / 2` pixels.
struct PrimitiveRenderFixture {
    let base: GeometryRenderFixture
    var device: any MTLDevice { base.device }
    var side: Int { base.side }
    var camera: PerspectiveCamera { base.camera }
    /// Pixels per world unit at unit depth.
    var pixelsPerUnit: Float { Float(side) / 2 }

    init(side: Int = 64) throws {
        base = try GeometryRenderFixture(side: side)
    }

    /// The framebuffer pixel coordinate (top-left origin, pixel centers at +0.5)
    /// that a camera-space point projects to.
    func pixel(_ point: SIMD3<Float>) -> SIMD2<Float> {
        let depth = -point.z
        return SIMD2(
            Float(side) / 2 + pixelsPerUnit * point.x / depth,
            Float(side) / 2 - pixelsPerUnit * point.y / depth
        )
    }

    /// The camera-space point at depth d that projects to a pixel coordinate.
    func point(pixel: SIMD2<Float>, depth: Float) -> SIMD3<Float> {
        SIMD3(
            (pixel.x - Float(side) / 2) * depth / pixelsPerUnit,
            (Float(side) / 2 - pixel.y) * depth / pixelsPerUnit,
            -depth
        )
    }

    func lines(_ geometry: LineGeometry, _ material: LineMaterial) throws -> ScenePrimitive {
        .lines(try LineRenderer(device: device, geometry: geometry, material: material))
    }

    func graph(_ roots: [SceneNode], cameraTransform: SceneTransform = SceneTransform()) -> SceneGraph {
        SceneGraph(roots: roots, camera: camera, cameraTransform: cameraTransform)
    }

    func render(_ graph: SceneGraph, name: String? = nil) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        let result = try base.render(graph)
        if let name { dump(result.pixels, name: name) }
        return result
    }

    /// Renders through the explicit-view encode with forward or reversed finite
    /// depth, optionally into a viewport of the target.
    func render(
        _ graph: SceneGraph, depthMode: MeshDepthMode, viewport: MTLViewport? = nil,
        ordering: SceneRenderOrdering = .init(), name: String? = nil
    ) throws -> [SIMD4<Float>] {
        let target = try base.makeTarget()
        let pass = base.pass(color: target.color, depth: target.depth)
        pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
        let camera = graph.camera
        let y = 1 / tan(camera.verticalFieldOfView / 2)
        let near = camera.near
        let far = camera.far
        // Reversed depth maps near to 1 and far to 0.
        let (z, w): (Float, Float) = depthMode == .forward
            ? (far / (near - far), near * far / (near - far))
            : (near / (far - near), far * near / (far - near))
        let projection = simd_float4x4(columns: (
            SIMD4(y, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(0, 0, z, -1), SIMD4(0, 0, w, 0)
        ))
        let buffer = try #require(base.queue.makeCommandBuffer())
        try graph.encode(
            pass: pass, commandBuffer: buffer,
            clipFromWorld: projection * graph.cameraTransform.matrix.inverse,
            cameraPosition: graph.cameraTransform.translation, viewport: viewport, depthMode: depthMode, ordering: ordering
        )
        try base.finish(buffer)
        let pixels = try base.pixels(of: target.color)
        if let name { dump(pixels, name: name) }
        return pixels
    }

    func alpha(_ pixels: [SIMD4<Float>], x: Int, y: Int) -> Float {
        pixels[y * side + x].w
    }

    /// Writes the image to `FROSTFIELD_TEST_IMAGE_DIRECTORY` when that variable is
    /// set (pass `TEST_RUNNER_FROSTFIELD_TEST_IMAGE_DIRECTORY` to xcodebuild), for inspection.
    func dump(_ pixels: [SIMD4<Float>], name: String) {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_TEST_IMAGE_DIRECTORY"] else { return }
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        for (index, pixel) in pixels.enumerated() {
            // Composite premultiplied linear color over black, then encode as sRGB.
            for channel in 0..<3 {
                let linear = Double(min(max(pixel[channel], 0), 1))
                let encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
                bytes[index * 4 + channel] = UInt8((encoded * 255).rounded())
            }
            bytes[index * 4 + 3] = 255
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else { return }
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}
