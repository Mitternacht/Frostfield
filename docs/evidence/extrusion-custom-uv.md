# Custom extrusion UVs

Pass a `ShapeExtrusionUVGenerator` to straight, beveled or curve extrusion. A cap callback sees a triangle in emitted order, with the cap side and triangle index. A wall callback sees a quad in front-start, front-end, back-end, back-start order, with its ring, edge and axial interval. Both receive the output mesh's positions, normals and default UV0. Holes have their own ring indices. A collapsed bevel quad can contribute one triangle.

```swift
let uv = ShapeExtrusionUVGenerator(cap: { face in
    func mirror(_ vertex: MeshVertex) -> SIMD2<Float> {
        SIMD2(1 - vertex.textureCoordinate.x, vertex.textureCoordinate.y)
    }
    return (mirror(face.corners.0), mirror(face.corners.1), mirror(face.corners.2))
})
let mesh = try MeshGeometry.extrude([profile], ShapeExtrusion(depth: 1), uvGenerator: uv)
```

Omitted callbacks use native UVs for their surfaces. A callback can return values outside 0...1. It must return finite coordinates, and its own errors propagate. Changed coordinates split indexed vertices only as needed at UV seams and regenerate MikkTSpace tangents in the extrusion's V-down convention. Triangle order, winding, normals and material group ranges are retained. An identity mapping retains the original vertex and index layout and tangents. The custom path checks a conservative budget of 1,048,576 index corners before callbacks run.

Extruded `MeshGeometry.text` forwards the same generator for its glyph profiles. The Swift API follows r187's per-triangle cap and per-quad wall callback granularity using typed corner records. [Three.js r187 ExtrudeGeometry](https://github.com/mrdoob/three.js/blob/r187/src/geometries/ExtrudeGeometry.js#L658-L692)

## Verification

- Geometry checks cover rotated cap UVs and tangent signs, mirrored beveled walls and hole-ring metadata, and identity mapping.
- A curved sweep assigns a separate UV island per wall quad and preserves its exact cap and wall group ranges, corner positions and normals.
- Tests cover nonfinite values, callback errors, and custom UV forwarding through extruded text.
- Metal readback samples a two-color texture on the cap. Mirroring UV0 swaps the two sampled colors.
- The pinned r187 callback probe confirms four cap calls and eight wall calls for a two-step square and one returned quad tuple reused by both wall triangles.
- Xcode 27.2 ran 37 focused tests in seven suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.
