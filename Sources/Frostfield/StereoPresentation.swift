import Foundation
import Metal
import simd

/// Stereo views and the r187 image combines for anaglyph, parallax barrier, and side-by-side.
///
/// Render the scene once with each returned camera. Anaglyph eyes share one screen plane and
/// the Dubois combine mixes their colors. Barrier eyes use the `StereoCamera` frustum and
/// alternate rows. Side-by-side eyes use that frustum at half aspect; draw each into one half
/// of the encode viewport. The color combines stop before tone mapping and output encoding.
public enum StereoPresentation {
    public enum Eye: Sendable { case left, right }
    public enum Combine: Int, Sendable { case anaglyph = 0, parallaxBarrier = 1 }

    /// Dubois left-eye matrix, stored in the column order Three.js uploads to `mat3`.
    public static let anaglyphLeft = simd_float3x3(columns: (
        SIMD3(0.456100, -0.0400822, -0.0152161),
        SIMD3(0.500484, -0.0378246, -0.0205971),
        SIMD3(0.176381, -0.0157589, -0.00546856)
    ))
    /// Dubois right-eye matrix, in the same column order.
    public static let anaglyphRight = simd_float3x3(columns: (
        SIMD3(-0.0434706, 0.378476, -0.0721527),
        SIMD3(-0.0879388, 0.73364, -0.112961),
        SIMD3(-0.00155529, -0.0184503, 1.2264)
    ))

    public struct View: Sendable {
        public let position: SIMD3<Float>
        public let clipFromWorld: simd_float4x4
    }

    public enum Failure: LocalizedError, Equatable {
        case separation
        case distance
        case focus
        case zoom
        case aspect
        case transform
        case count
        case pipeline
        case encoder
        case command

        public var errorDescription: String? {
            switch self {
            case .separation: "Eye separation must be finite and positive."
            case .distance: "The zero-parallax plane must be a finite positive distance."
            case .focus: "Stereo focus must be finite and positive."
            case .zoom: "Stereo zoom must be finite and positive."
            case .aspect: "The stereo aspect scale must be finite and positive."
            case .transform: "The camera transform must be invertible."
            case .count: "A stereo combine accepts at most 65,536 pixels."
            case .pipeline: "Metal could not compile the stereo combine."
            case .encoder: "Metal could not create the stereo combine encoder."
            case .command: "The stereo combine command buffer did not complete."
            }
        }
    }

    /// Left and right off-axis views of one screen, matching `AnaglyphEffect`.
    public static func anaglyphEyes(
        camera: PerspectiveCamera, transform: SceneTransform = .init(), aspectRatio: Float,
        separation: Float = 0.064, planeDistance: Float = 0.5, depthMode: MeshDepthMode = .forward
    ) throws -> (left: View, right: View) {
        guard separation.isFinite, separation > 0 else { throw Failure.separation }
        guard planeDistance.isFinite, planeDistance > 0 else { throw Failure.distance }
        let basis = cameraBasis(transform)
        let half = separation / 2
        let leftEye = basis.position - basis.right * half
        let rightEye = basis.position + basis.right * half
        let center = basis.position - basis.forward * planeDistance
        let halfHeight = planeDistance * tan(camera.verticalFieldOfView / 2)
        let halfWidth = halfHeight * aspectRatio
        let bottomLeft = center - basis.right * halfWidth - basis.up * halfHeight
        let bottomRight = center + basis.right * halfWidth - basis.up * halfHeight
        let topLeft = center - basis.right * halfWidth + basis.up * halfHeight
        let left = try CameraUtils.frameCorners(
            eye: leftEye, bottomLeft: bottomLeft, bottomRight: bottomRight, topLeft: topLeft,
            near: camera.near, far: camera.far, depthMode: depthMode
        )
        let right = try CameraUtils.frameCorners(
            eye: rightEye, bottomLeft: bottomLeft, bottomRight: bottomRight, topLeft: topLeft,
            near: camera.near, far: camera.far, depthMode: depthMode
        )
        return (View(position: leftEye, clipFromWorld: left.clipFromWorld), View(position: rightEye, clipFromWorld: right.clipFromWorld))
    }

    /// One eye of `StereoCamera`, used by the side-by-side and parallax-barrier effects.
    /// `aspectScale` is 0.5 for side-by-side and 1 for the barrier.
    public static func stereoEye(
        camera: PerspectiveCamera, transform: SceneTransform = .init(), aspectRatio: Float, eye: Eye,
        separation: Float = 0.064, focus: Float = 10, zoom: Float = 1, aspectScale: Float = 1,
        depthMode: MeshDepthMode = .forward
    ) throws -> View {
        guard separation.isFinite, separation > 0 else { throw Failure.separation }
        guard focus.isFinite, focus > 0 else { throw Failure.focus }
        guard zoom.isFinite, zoom > 0 else { throw Failure.zoom }
        guard aspectScale.isFinite, aspectScale > 0 else { throw Failure.aspect }
        let sign: Float = eye == .left ? -1 : 1
        var eyeMatrix = transform.matrix
        let position4 = eyeMatrix * SIMD4(sign * separation / 2, 0, 0, 1)
        let position = SIMD3(position4.x, position4.y, position4.z)
        var projection = try camera.projection(aspectRatio: aspectRatio * aspectScale, depthMode: depthMode)
        let aspect = aspectRatio * aspectScale
        let shift = (separation / focus) * zoom / (2 * tan(camera.verticalFieldOfView / 2) * aspect)
        projection.columns.0.x *= zoom
        projection.columns.1.y *= zoom
        projection.columns.2.x = eye == .left ? shift : -shift
        eyeMatrix.columns.3 = position4
        let determinant = simd_determinant(eyeMatrix)
        guard determinant.isFinite, abs(determinant) > 1e-12 else { throw Failure.transform }
        return View(position: position, clipFromWorld: projection * simd_inverse(eyeMatrix))
    }

    /// Dubois anaglyph, clamped, with alpha from the stronger source.
    public static func anaglyph(left: SIMD4<Float>, right: SIMD4<Float>) -> SIMD4<Float> {
        let color = simd_clamp(anaglyphLeft * left.rgb + anaglyphRight * right.rgb, SIMD3(repeating: 0), SIMD3(repeating: 1))
        return SIMD4(color, max(left.w, right.w))
    }

    /// Left image when `mod(fragmentY, 2) > 1`, otherwise the right image.
    public static func parallaxBarrier(left: SIMD4<Float>, right: SIMD4<Float>, fragmentY: Float) -> SIMD4<Float> {
        glslMod(fragmentY, 2) > 1 ? left : right
    }

    public struct Pixel: Sendable {
        public var combine: Combine
        public var left: SIMD4<Float>
        public var right: SIMD4<Float>
        public var fragmentY: Float

        public init(combine: Combine, left: SIMD4<Float>, right: SIMD4<Float>, fragmentY: Float = 0) {
            self.combine = combine
            self.left = left
            self.right = right
            self.fragmentY = fragmentY
        }

        var packed: Packed {
            Packed(combine: SIMD4(Float(combine.rawValue), fragmentY, 0, 0), left: left, right: right)
        }

        struct Packed {
            var combine, left, right: SIMD4<Float>
        }
    }

    /// Combines pixels on the GPU and waits for the colors.
    public static func evaluateOnGPU(_ pixels: [Pixel], device: any MTLDevice) throws -> [SIMD4<Float>] {
        guard !pixels.isEmpty else { return [] }
        guard pixels.count <= 65_536 else { throw Failure.count }
        guard let queue = device.makeCommandQueue() else { throw Failure.encoder }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: metalSource, options: nil) } catch { throw Failure.pipeline }
        guard let function = library.makeFunction(name: "frostfield_stereo_combine") else { throw Failure.pipeline }
        let pipeline: any MTLComputePipelineState
        do { pipeline = try device.makeComputePipelineState(function: function) } catch { throw Failure.pipeline }
        let packed = pixels.map(\.packed)
        guard let source = packed.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }), let destination = device.makeBuffer(
            length: pixels.count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared
        ), let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeComputeCommandEncoder() else {
            throw Failure.encoder
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        var count = UInt32(pixels.count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 64)
        encoder.dispatchThreadgroups(
            MTLSize(width: (pixels.count + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw Failure.command }
        return Array(UnsafeBufferPointer(
            start: destination.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: pixels.count
        ))
    }

    private static func cameraBasis(_ transform: SceneTransform) -> (
        right: SIMD3<Float>, up: SIMD3<Float>, forward: SIMD3<Float>, position: SIMD3<Float>
    ) {
        let matrix = transform.matrix
        func axis(_ column: SIMD4<Float>) -> SIMD3<Float> {
            let vector = SIMD3(column.x, column.y, column.z)
            let length = simd_length(vector)
            return length > 0 ? vector / length : vector
        }
        return (
            axis(matrix.columns.0), axis(matrix.columns.1), axis(matrix.columns.2),
            SIMD3(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
        )
    }

    private static func glslMod(_ value: Float, _ divisor: Float) -> Float {
        value - divisor * floor(value / divisor)
    }

    private static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct StereoPixel { float4 combine, left, right; };

    static float frostfield_mod(float value, float divisor) {
        return value - divisor * floor(value / divisor);
    }

    constant float3x3 anaglyphLeft = float3x3(
        float3(0.456100, -0.0400822, -0.0152161),
        float3(0.500484, -0.0378246, -0.0205971),
        float3(0.176381, -0.0157589, -0.00546856)
    );
    constant float3x3 anaglyphRight = float3x3(
        float3(-0.0434706, 0.378476, -0.0721527),
        float3(-0.0879388, 0.73364, -0.112961),
        float3(-0.00155529, -0.0184503, 1.2264)
    );

    kernel void frostfield_stereo_combine(
        device const StereoPixel* pixels [[buffer(0)]],
        device float4* colors [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= count) { return; }
        StereoPixel pixel = pixels[id];
        float mode = pixel.combine.x;
        float4 color;
        if (mode < 0.5f) {
            float3 mixed = clamp(
                anaglyphLeft * pixel.left.rgb + anaglyphRight * pixel.right.rgb, 0.0f, 1.0f
            );
            color = float4(mixed, max(pixel.left.a, pixel.right.a));
        } else {
            color = frostfield_mod(pixel.combine.y, 2.0f) > 1.0f ? pixel.left : pixel.right;
        }
        colors[id] = color;
    }
    """
}

private extension SIMD4 where Scalar == Float {
    var rgb: SIMD3<Float> { SIMD3(x, y, z) }
}
