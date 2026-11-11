import Foundation
import Metal
import simd

/// The camera depth range shared by the logarithmic mesh color and attribute pipelines.
struct LogarithmicRasterDepth {
    /// Effective near, inverse log2 range, log2 near, and finite far. Zero for orthographic views.
    let range: SIMD4<Float>
    private let cullingDepth: SIMD2<Double>?

    init(
        view: GraphViewContext, pass: MTLRenderPassDescriptor, viewport: MTLViewport? = nil
    ) throws {
        let depth = pass.depthAttachment!
        guard depth.loadAction != .clear || depth.clearDepth == 1,
              viewport.map({ $0.znear == 0 && $0.zfar == 1 }) ?? true else {
            throw MeshRenderer.Failure.logarithmicDepthPass
        }
        let camera = view.viewMatrix
        guard view.hasProjection, camera[0].w == 0, camera[1].w == 0, camera[2].w == 0,
              abs(camera[3].w - 1) <= 8 * Float.ulpOfOne else {
            throw MeshRenderer.Failure.logarithmicDepthCamera
        }
        try self.init(projection: view.projectionMatrix, near: view.near, far: view.far)
    }

    /// Validate and prepare the same authored clip range for raster and its depth consumers.
    init(projection: simd_float4x4, near authoredNear: Float, far: Float) throws {
        guard far.isFinite, projection[0].w == 0, projection[1].w == 0,
              projection[0].z == 0, projection[1].z == 0, projection[2].z < 0 else {
            throw MeshRenderer.Failure.logarithmicDepthCamera
        }
        let nearPlane = Double(authoredNear)
        let farPlane = Double(far)
        let distanceRange = farPlane - nearPlane
        func matches(_ depthScale: Double, _ depthOffset: Double) -> Bool {
            func close(_ actual: Float, _ expected: Double) -> Bool {
                abs(Double(actual) - expected) <= 2e-6 * max(abs(expected), Double(Float.leastNormalMagnitude))
            }
            return close(projection[2].z, depthScale) && close(projection[3].z, depthOffset)
        }
        if projection[2].w == 0, projection[3].w == 1 {
            guard matches(-1 / distanceRange, -nearPlane / distanceRange) else {
                throw MeshRenderer.Failure.logarithmicDepthCamera
            }
            range = .zero
            cullingDepth = nil
            return
        }
        let near = max(authoredNear, 1e-6)
        let depthOffset = nearPlane * farPlane / distanceRange
        guard projection[2].w == -1, projection[3].w == 0, authoredNear > 0, far > near,
              matches(-farPlane / distanceRange, -depthOffset) else {
            throw MeshRenderer.Failure.logarithmicDepthCamera
        }
        let inverseLogRange = Float(1 / log2(farPlane / Double(near)))
        range = SIMD4(near, inverseLogRange, log2(near), far)
        cullingDepth = SIMD2(-nearPlane / distanceRange, depthOffset)
    }

    /// Recover finite near and far half-spaces when the Float forward clip matrix
    /// rounded its far plane to infinity. The raster projection and ordering stay forward.
    func cullingClip(_ forward: simd_float4x4) -> simd_float4x4 {
        guard let cullingDepth else { return forward }
        var result = forward
        for column in 0..<4 {
            let offset = column == 3 ? cullingDepth.y : 0
            result[column].z = Float(Double(forward[column].w) * cullingDepth.x + offset)
        }
        return result
    }

    func projectedDepth(at clip: SIMD4<Float>) -> Double {
        if range.w == 0 { return Double(clip.z) / Double(clip.w) }
        return log2(Double(clip.w) / Double(range.x)) * Double(range.y)
    }

    func bind(into encoder: any MTLRenderCommandEncoder) {
        var parameters = range
        encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 29)
    }
}

extension MeshDepthMode {
    /// Effects that reconstruct or compare projection depth require a matching attachment.
    func requireProjectionDepth() throws {
        guard self != .logarithmic else { throw MeshRenderer.Failure.logarithmicDepthConsumer }
    }
}
