import Foundation
import Metal
import simd

public extension SceneGraph {
    /// The color format of the surface attribute prepass. Each texel holds the world-space
    /// shading normal as two snorm16 octahedral coordinates in the first word, then the
    /// square roots of the RGB specular reflectance and the perceptual roughness as unorm8
    /// values in the second.
    static let surfaceAttributesPixelFormat: MTLPixelFormat = .rg32Uint

    /// Encodes an opt-in prepass that writes every opaque and masked mesh's shading
    /// normal, roughness and specular reflectance, with its own depth, for
    /// `AmbientOcclusion.encodeVisibility` and `ScreenSpaceReflections`.
    ///
    /// The pass needs a single-sample `surfaceAttributesPixelFormat` color attachment and
    /// a Depth32Float depth attachment of the same size, cleared to 1 for forward depth or
    /// to 0 for reversed depth. Use the same depth mode as the color pass.
    /// Use a depth texture separate from the color pass's: both passes write equal depth.
    /// PBR meshes write their mapped normal, perceptual roughness and the split-sum
    /// reflectance `(F0·A + F90·B)·occlusion` at the view angle, from the environment's
    /// lookup table or Karis's analytic fit without one, scaled by the material's occlusion
    /// map. Unlit PBR and non-PBR meshes write their normal with zero reflectance.
    /// The prepass draws opaque and cutout meshes. Its single sample follows fragment
    /// discards when the color pass uses alpha-to-coverage. Primitives and blended meshes
    /// do not write attributes. Culling and level of detail match `encode`.
    /// Supply the same `indirectDraws` node arguments as the corresponding color pass.
    /// Each mesh prepares its pipeline on first use. See `MeshRenderer.prepareSurfaceAttributes()`.
    @discardableResult
    func encodeSurfaceAttributes(
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer,
        depthMode: MeshDepthMode = .forward,
        scissor: MTLScissorRect? = nil,
        indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneDrawStatistics {
        try Self.validateSurfaceAttributes(pass: pass, device: commandBuffer.device)
        return try MeshRenderer.encode(
            self, pass: pass, commandBuffer: commandBuffer, shadowMap: nil, depthMode: depthMode,
            surfaceAttributes: true, scissor: scissor, indirectDraws: indirectDraws
        )
    }

    /// Encodes the surface attribute prepass for a caller-supplied view, as
    /// `encode(pass:commandBuffer:clipFromWorld:cameraPosition:...)` does for color.
    @discardableResult
    func encodeSurfaceAttributes(
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer,
        clipFromWorld: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        directionToCamera: SIMD3<Float>? = nil,
        viewport: MTLViewport? = nil,
        depthMode: MeshDepthMode = .forward,
        graphCamera: ShaderGraphCamera? = nil,
        scissor: MTLScissorRect? = nil,
        indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneDrawStatistics {
        try Self.validateSurfaceAttributes(pass: pass, device: commandBuffer.device)
        return try MeshRenderer.encode(
            self, pass: pass, commandBuffer: commandBuffer, shadowMap: nil,
            view: (clipFromWorld, cameraPosition, directionToCamera), viewport: viewport, depthMode: depthMode,
            graphCamera: graphCamera, surfaceAttributes: true, scissor: scissor, indirectDraws: indirectDraws
        )
    }

    private static func validateSurfaceAttributes(pass: MTLRenderPassDescriptor, device: any MTLDevice) throws {
        guard let color = pass.colorAttachments[0].texture, let depth = pass.depthAttachment.texture,
              color.device === device, color.pixelFormat == surfaceAttributesPixelFormat,
              color.textureType == .type2D, color.sampleCount == 1, depth.sampleCount == 1,
              color.usage.contains(.renderTarget),
              (1..<8).allSatisfy({ pass.colorAttachments[$0].texture == nil }) else {
            throw MeshRenderer.Failure.surfaceAttributes
        }
    }
}

/// The perspective projection that screen-space passes invert. A view-space point
/// (x, y, −d) has NDC (sx·x/d − kx, sy·y/d − ky), and 1/d is affine in stored depth:
/// `1/d = farInverseDistance + inverseDistanceRange · w`, where w is 0 at the cleared depth
/// and 1 at the near plane. Off-center (asymmetric) frustums are supported.
struct ScreenSpaceProjection {
    /// sx and sy.
    let scale: SIMD2<Float>
    /// kx and ky, the projection's third column in x and y.
    let center: SIMD2<Float>
    let farInverseDistance: Float
    let inverseDistanceRange: Float
    /// The view distance of the near plane.
    let near: Float

    /// A Frostfield camera's projection for this aspect ratio, in either depth mode.
    init?(camera: PerspectiveCamera, aspectRatio: Float) {
        guard let projection = try? camera.projection(aspectRatio: aspectRatio) else { return nil }
        let farInverse = 1 / Double(camera.far)
        self.init(
            scale: SIMD2(projection.columns.0.x, projection.columns.1.y), center: .zero,
            farInverse: farInverse, range: 1 / Double(camera.near) - farInverse
        )
    }

    /// A caller-supplied view-to-clip matrix with Metal's 0...1 depth, whose depth mapping
    /// matches `depthMode`: forward depth stores 0 at the near plane, reversed depth 1.
    /// Returns nil for a matrix that is not a perspective projection of that form.
    init?(clipFromView: simd_float4x4, depthMode: MeshDepthMode) {
        let columns = [clipFromView.columns.0, clipFromView.columns.1, clipFromView.columns.2, clipFromView.columns.3]
        guard columns.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }),
              clipFromView.columns.2.w < 0 else { return nil }
        // Clip w must be the view distance up to a uniform scale, which leaves NDC unchanged.
        let w = -Double(clipFromView.columns.2.w)
        let largest = columns.map { simd_reduce_max(abs($0)) }.max() ?? 0
        let zeros: [Float] = [
            clipFromView.columns.0.y, clipFromView.columns.0.z, clipFromView.columns.0.w,
            clipFromView.columns.1.x, clipFromView.columns.1.z, clipFromView.columns.1.w,
            clipFromView.columns.3.x, clipFromView.columns.3.y, clipFromView.columns.3.w,
        ]
        guard zeros.allSatisfy({ abs($0) <= 1e-6 * largest }) else { return nil }
        // Depth = (a·z + b) / −z with z = −d, so 1/d = (depth + a) / b.
        let a = Double(clipFromView.columns.2.z) / w
        let b = Double(clipFromView.columns.3.z) / w
        guard b != 0 else { return nil }
        let farInverse = depthMode == .reversed ? a / b : (1 + a) / b
        let range = depthMode == .reversed ? 1 / b : -1 / b
        // Allow rounding at an infinite far plane.
        self.init(
            scale: SIMD2(Float(Double(clipFromView.columns.0.x) / w), Float(Double(clipFromView.columns.1.y) / w)),
            center: SIMD2(Float(Double(clipFromView.columns.2.x) / w), Float(Double(clipFromView.columns.2.y) / w)),
            farInverse: abs(farInverse) <= 1e-6 * abs(range) ? 0 : farInverse, range: range
        )
    }

    private init?(scale: SIMD2<Float>, center: SIMD2<Float>, farInverse: Double, range: Double) {
        let near = 1 / (farInverse + range)
        guard scale.x.isNormal, scale.y.isNormal, scale.x > 0, scale.y > 0, center.x.isFinite, center.y.isFinite,
              farInverse.isFinite, farInverse >= 0, range.isFinite, range > 0,
              Float(farInverse).isFinite, Float(range).isNormal, Float(near).isNormal else { return nil }
        self.scale = scale
        self.center = center
        farInverseDistance = Float(farInverse)
        inverseDistanceRange = Float(range)
        self.near = Float(near)
    }

    /// The reciprocal scales and centers, then the depth mapping and near distance, with `extra` last.
    func constants(extra: Float = 0) -> [SIMD4<Float>] {
        [SIMD4(1 / scale.x, 1 / scale.y, center.x, center.y), SIMD4(farInverseDistance, inverseDistanceRange, near, extra)]
    }

    /// The rotation part of a rigid world-from-view transform, or nil when it is not one.
    static func viewFromWorld(_ worldFromView: simd_float4x4) -> simd_float3x3? {
        let rotation = simd_float3x3(
            SIMD3(worldFromView.columns.0.x, worldFromView.columns.0.y, worldFromView.columns.0.z),
            SIMD3(worldFromView.columns.1.x, worldFromView.columns.1.y, worldFromView.columns.1.z),
            SIMD3(worldFromView.columns.2.x, worldFromView.columns.2.y, worldFromView.columns.2.z)
        )
        let check = rotation.transpose * rotation
        let identity = matrix_identity_float3x3
        for column in 0..<3 {
            let difference = check[column] - identity[column]
            guard difference.x.isFinite, difference.y.isFinite, difference.z.isFinite,
                  simd_reduce_max(abs(difference)) < 1e-3 else { return nil }
        }
        guard simd_determinant(rotation) > 0 else { return nil }
        return rotation.transpose
    }
}
