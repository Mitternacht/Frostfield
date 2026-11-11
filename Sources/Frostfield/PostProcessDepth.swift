import simd

public extension ShaderGraphBuilder {
    /// The inverse of the camera's Metal projection as a graph matrix.
    /// Use the projection that produced the source depth, including any view
    /// offset, jitter and forward or reversed depth mapping.
    func inverseProjection(of camera: ShaderGraphCamera) -> ShaderValue<simd_float4x4> {
        let inverse = simd_inverse(camera.projectionMatrix)
        return matrix(
            constant(inverse.columns.0), constant(inverse.columns.1),
            constant(inverse.columns.2), constant(inverse.columns.3)
        )
    }

    /// Unprojects top-left normalized screen coordinates and raw Metal depth.
    /// The right-handed view looks along negative Z. Nonfinite positions,
    /// including a cleared infinite far plane, return zero.
    func getViewPosition(
        at coordinates: ShaderValue<SIMD2<Float>>, depth: ShaderValue<Float>,
        inverseProjection: ShaderValue<simd_float4x4>
    ) -> ShaderValue<SIMD3<Float>> {
        add(
            .viewPositionFromDepth(coordinates: coordinates.id, depth: depth.id, inverseProjection: inverseProjection.id),
            operands: [coordinates, depth, inverseProjection]
        )
    }

    /// Reconstructs a view-space normal from the bound source depth.
    /// Samples actual texel centers and favors the smoother side on each axis.
    /// Texture edges use one-sided stencils. Cleared depth and insufficient
    /// samples return zero. Use the source attachment's forward or reversed
    /// projection depth. Logarithmic depth is rejected at graph construction.
    func getNormalFromDepth(
        at coordinates: ShaderValue<SIMD2<Float>>, inverseProjection: ShaderValue<simd_float4x4>,
        depthMode: MeshDepthMode = .forward
    ) -> ShaderValue<SIMD3<Float>> {
        if depthMode == .logarithmic {
            rejectDeclaration("Depth normal reconstruction requires forward or reversed projection depth.")
        }
        return add(
            .normalFromDepth(coordinates: coordinates.id, inverseProjection: inverseProjection.id,
                             reversedDepth: depthMode == .reversed),
            operands: [coordinates, inverseProjection]
        )
    }

    /// Linear view-axis depth in 0...1, with zero at near and one at far.
    /// The inverse projection supplies the source depth direction and handles
    /// perspective and orthographic projections. Clip distances are in view
    /// units and must be finite, with 0 ≤ near < far. Invalid inputs return zero.
    func linearDepth(
        at coordinates: ShaderValue<SIMD2<Float>>, depth: ShaderValue<Float>,
        inverseProjection: ShaderValue<simd_float4x4>, near: ShaderValue<Float>, far: ShaderValue<Float>
    ) -> ShaderValue<Float> {
        add(
            .linearDepth(coordinates: coordinates.id, depth: depth.id, inverseProjection: inverseProjection.id,
                         near: near.id, far: far.id),
            operands: [coordinates, depth, inverseProjection, near, far]
        )
    }
}

enum PostDepthGraphSource {
    static let math = """
    static inline float4 frostfield_post_view_homogeneous(float2 uv, float depth, float4x4 inverseProjection) {
        if (!all(frostfield_graph_finite(uv)) || !(depth >= 0.0f && depth <= 1.0f)) return float4(0);
        return inverseProjection * float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    }

    static inline bool frostfield_post_try_view_position(
        float2 uv, float depth, float4x4 inverseProjection, thread float3& position
    ) {
        float4 view = frostfield_post_view_homogeneous(uv, depth, inverseProjection);
        if (view.w == 0.0f || !all(frostfield_graph_finite(view))) return false;
        position = view.xyz / view.w;
        return all(frostfield_graph_finite(position));
    }

    static inline float3 frostfield_post_view_position(float2 uv, float depth, float4x4 inverseProjection) {
        float3 position;
        return frostfield_post_try_view_position(uv, depth, inverseProjection, position) ? position : float3(0);
    }

    static inline float frostfield_post_linear_depth(
        float2 uv, float depth, float4x4 inverseProjection, float near, float far
    ) {
        if (!(near >= 0.0f && far > near) || !frostfield_graph_finite(far)) return 0.0f;
        float4 view = frostfield_post_view_homogeneous(uv, depth, inverseProjection);
        if (!all(frostfield_graph_finite(view))) return 0.0f;
        if (view.w == 0.0f) return view.z != 0.0f ? 1.0f : 0.0f;
        return saturate((-view.z / view.w - near) / (far - near));
    }
    """

    static let normalHook = """
    static inline bool frostfield_post_has_depth(float depth, float clearDepth) {
        return depth >= 0.0f && depth <= 1.0f && depth != clearDepth;
    }

    static inline float frostfield_post_read_depth(depth2d<float> depth, int2 pixel, int2 size, float clearDepth) {
        return any(pixel < 0) || any(pixel >= size) ? clearDepth : depth.read(uint2(pixel));
    }

    // (First neighbor's depth, error extrapolating its surface back to center).
    // At a border or silhouette, a missing second sample falls back to first difference.
    static inline float2 frostfield_post_depth_side(
        depth2d<float> depth, int2 pixel, int2 step, int2 size, float center, float clearDepth
    ) {
        float first = frostfield_post_read_depth(depth, pixel + step, size, clearDepth);
        if (!frostfield_post_has_depth(first, clearDepth)) return float2(first, 3.0f);
        float second = frostfield_post_read_depth(depth, pixel + 2 * step, size, clearDepth);
        float estimate = frostfield_post_has_depth(second, clearDepth) ? 2.0f * first - second : first;
        return float2(first, abs(estimate - center));
    }

    static inline float3 frostfield_graph_depth_normal(
        thread const FrostfieldGraphResources& r, float2 uv, float4x4 inverseProjection, bool reversed
    ) {
        if (!all(frostfield_graph_finite(uv))) return float3(0);
        int2 size(r.depth.get_width(), r.depth.get_height());
        float2 inverseSize = 1.0f / float2(size);
        int2 pixel = min(int2(saturate(uv) * float2(size)), size - 1);
        float clearDepth = reversed ? 0.0f : 1.0f;
        float center = r.depth.read(uint2(pixel));
        if (!frostfield_post_has_depth(center, clearDepth)) return float3(0);

        float2 left = frostfield_post_depth_side(r.depth, pixel, int2(-1, 0), size, center, clearDepth);
        float2 right = frostfield_post_depth_side(r.depth, pixel, int2(1, 0), size, center, clearDepth);
        float2 top = frostfield_post_depth_side(r.depth, pixel, int2(0, -1), size, center, clearDepth);
        float2 bottom = frostfield_post_depth_side(r.depth, pixel, int2(0, 1), size, center, clearDepth);
        bool useLeft = left.y < right.y;
        bool useBottom = bottom.y < top.y;
        float depthX = useLeft ? left.x : right.x;
        float depthY = useBottom ? bottom.x : top.x;
        if (!frostfield_post_has_depth(depthX, clearDepth) || !frostfield_post_has_depth(depthY, clearDepth)) {
            return float3(0);
        }

        float2 centerUV = (float2(pixel) + 0.5f) * inverseSize;
        float stepX = useLeft ? -1.0f : 1.0f;
        float stepY = useBottom ? 1.0f : -1.0f;
        float3 origin, x, y;
        if (!frostfield_post_try_view_position(centerUV, center, inverseProjection, origin)
            || !frostfield_post_try_view_position(centerUV + float2(stepX, 0) * inverseSize, depthX, inverseProjection, x)
            || !frostfield_post_try_view_position(centerUV + float2(0, stepY) * inverseSize, depthY, inverseProjection, y)) {
            return float3(0);
        }

        // Convert the chosen one-sided differences to view-space right and up.
        float3 horizontal = frostfield_graph_unit((x - origin) * stepX);
        float3 vertical = frostfield_graph_unit((origin - y) * stepY);
        return frostfield_graph_unit(cross(horizontal, vertical));
    }
    """
}
