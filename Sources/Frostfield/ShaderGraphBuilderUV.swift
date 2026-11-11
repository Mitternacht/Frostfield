import simd

extension ShaderGraphBuilder {
    /// Projects a unit direction into a top-left equirectangular panorama.
    /// +X is the center, +Z moves right, and +Y is the top pole.
    /// Y is clamped to −1...1. At either pole U is 0.5.
    public func equirectUV(direction: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD2<Float>> {
        let horizontal = direction.x.abs().max(direction.z.abs())
        let longitude = direction.z.atan2(direction.x) * (0.5 / Float.pi) + 0.5
        let u = select(horizontal.greaterThan(constant(Float(0))), longitude, constant(Float(0.5)))
        let v = direction.y.clamped(-1, 1).acos() * (1 / Float.pi)
        return vector(u, v)
    }

    /// Projects a normal into a top-left matcap texture. Both vectors are in
    /// view space, and the direction points from the surface to the viewer.
    /// The projection normalizes both vectors and uses +X as its horizontal
    /// axis for a vertical view direction. A zero view direction uses +Z.
    public func matcapUV(
        normalInView: ShaderValue<SIMD3<Float>>, viewDirectionInView: ShaderValue<SIMD3<Float>>
    ) -> ShaderValue<SIMD2<Float>> {
        let zero = constant(Float(0))
        let normalizedView = viewDirectionInView.normalized()
        let view = select(normalizedView.length().greaterThan(zero), normalizedView, constant(SIMD3<Float>(0, 0, 1)))
        let horizontal = vector(view.z, zero, zero - view.x).normalized()
        let right = select(horizontal.length().greaterThan(zero), horizontal, constant(SIMD3<Float>(1, 0, 0)))
        let up = view.cross(right)
        let normal = normalInView.normalized()
        return vector(right.dot(normal) * 0.495 + 0.5, 0.5 - up.dot(normal) * 0.495)
    }

    /// Uses the surface graph's shading normal and viewer direction, transformed
    /// into the camera's view space. Requires the surface stage and its view matrix.
    public func matcapUV() -> ShaderValue<SIMD2<Float>> {
        let worldToView = viewMatrix.upperLeft
        return matcapUV(normalInView: worldToView * normal, viewDirectionInView: worldToView * viewDirection)
    }

    /// Maps top-left local UVs into a row-major sprite atlas. The grid gives
    /// columns and rows. Frames are zero-based, floored, and wrap in both
    /// directions. UVs default to the primary mesh coordinates and frame to 0.
    ///
    /// Supply finite values. Grid dimensions round down to whole cells and are
    /// clamped to at least one cell. Rows are reduced as needed to keep the
    /// total at or below 2²⁴, the exact Float integer range. Local UVs pass
    /// through unclamped so the caller can choose the atlas's edge behavior.
    public func spritesheetUV(
        _ grid: ShaderValue<SIMD2<Float>>,
        uv: ShaderValue<SIMD2<Float>>? = nil,
        frame: ShaderValue<Float>? = nil
    ) -> ShaderValue<SIMD2<Float>> {
        let maximumCells: Float = 16_777_216
        let columns = grid.x.floor().clamped(1, maximumCells)
        let rowLimit = (constant(maximumCells) / columns).floor()
        let rows = grid.y.floor().max(1).min(rowLimit)
        let count = columns * rows
        let tile = (frame ?? constant(Float(0))).floor().mod(count).clamped(constant(Float(0)), count - 1)
        let column = tile.mod(columns)
        let row = (tile / columns).floor()
        let local = uv ?? uv0
        return vector((local.x + column) / columns, (local.y + row) / rows)
    }
}
