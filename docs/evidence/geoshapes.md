# Shape and modeling tools

Frostfield builds 3D text, filled and extruded 2D shapes, projected decals,
convex hulls and boolean solids as ordinary indexed `MeshGeometry`, so the
results use the same materials, shadows, picking, path tracing and glTF export
as imported meshes. Everything here runs on the CPU in Double precision and
writes Float vertices.

| Capability | Public API | Implementation | Verified by |
| --- | --- | --- | --- |
| 3D text from any vector font, with Core Text kerning, ligatures, fallback and multi-line alignment | `TextOutline(_:font:size:alignment:lineSpacing:curveTolerance:)`, `TextOutline(_:unitsPerPoint:...)` for attributed strings, `MeshGeometry.text(_:font:size:extrusion:...)` | One `CTLine` per line; `CTRunGetPositions` places `CTFontCreatePathForGlyph` outlines; each glyph is resolved with the nonzero rule | `TextGeometryTests` |
| 2D shapes with holes from `CGPath`, including self-intersecting and overlapping subpaths | `PlanarShape.shapes(from:fillRule:tolerance:)`, `PlanarShape.shapes(rings:fillRule:)`, `PlanarShape(outer:holes:)` | Wang's bound flattens quadratic and cubic Béziers; crossings split at Float-rounded points; an edge stays where the fill rule fills one side only; holes go to the smallest enclosing outer ring | `ShapeGeometryTests` |
| Hole-aware triangulation that keeps every boundary vertex | `PolygonTriangulator.triangulate(outer:holes:)` (package) | Eberly bridges and earcut-style ear clipping with fallbacks; collinear points it drops are reinserted by splitting the edge through them; the result is rejected unless triangle areas sum to the polygon area | `PolygonTriangulatorTests` |
| Flat and extruded shape meshes with rounded bevels, crease-angle normals, tangents and UVs | `MeshGeometry.shape(_:textureMapping:)`, `MeshGeometry.extrude(_:_:)`, `ShapeExtrusion`, `ShapeTextureMapping` | Walls stand on the outline; caps inset by a quarter-ellipse bevel; edges that shrink to nothing merge as straight-skeleton edge events | `ShapeGeometryTests`, `TextGeometryTests` |
| Decals clipped to an oriented projector box | `DecalProjector`, `MeshGeometry.decal(projector:localToWorld:maximumAngle:surfaceOffset:bvh:)` | Optional `MeshBVH` box query, Sutherland–Hodgman clipping against six planes, back-face cut-off, surface offset, tangents along +U | `DecalGeometryTests` |
| Convex hulls | `ConvexHull(points:)`, `MeshGeometry.convexHull(_:)` | Quickhull with a 64-ulp tolerance. A simpler hull replaces the first result only if it still contains the input within that tolerance. | `ConvexHullTests` |
| CSG union, subtraction and intersection of closed meshes | `MeshBoolean.evaluate(_:_:_:)`, `MeshBooleanOperand`, `MeshGeometry.union(_:)`, `.subtracting(_:)`, `.intersection(_:)`, `MeshGeometry.signedVolume` | `MeshBVH` candidate pairs, shared edge–plane intersection points, per-triangle planar graphs triangulated with holes, patch flood fill, same/opposite coplanar rule, generalized winding number | `MeshBooleanTests` |
| Merge indexed geometry | `MeshGeometry.merged(_:useGroups:)`, `MeshGeometry.MergePart` | Bake finite affine transforms and concatenate indexed triangles. Mirrored parts reverse triangle winding and tangent handedness. Use the returned ranges or request one material group per part. | `MeshMergingTests` |

## What the tests measure

- **Triangulation.** Random concave star polygons with 25 square holes cover
  their shoelace area to 1e-9 relative, with exactly `V + 2H − 2`
  counterclockwise triangles that use every vertex. A square with five
  collinear boundary points keeps all of them, each boundary segment an edge
  of exactly one triangle. Crossing rings throw instead of returning
  overlapping triangles.
- **Shapes.** Two overlapping 2 × 2 squares resolve to area 7 under the
  nonzero rule and to two L shapes of area 3 under even-odd. A pentagram fills
  its center under nonzero and leaves the center pentagon, whose area comes
  from the closed-form circumradius `R cos 72° / cos 36°`, empty under
  even-odd. A circle hole flattened at tolerances 0.01 and 0.0001 removes
  between the exact Bézier area, integrated with Gauss–Legendre quadrature,
  and that area less `perimeter × tolerance`.
- **Extrusion.** A plain extrusion is closed by position and its signed volume
  equals area × depth to 1e-6 relative. A beveled square prism matches the sum
  of prismatoid slabs of its bevel profile to 1e-5 for 1, 3 and 8 segments.
  Bevel normals on a circular outline are radial and follow the ellipse normal
  `(t sin φ, s cos φ)` to 2e-3. An L shape's beveled cap is its exact 0.1
  offset, reflex corner included. A 20° notch, whose exact miter of
  `1 / sin 10°` times the inset would cut through the bar below it, is
  beveled with the reflex miter limited to twice the inset. Every triangle's
  tangent points along +U with handedness along +V.
- **Text.** Helvetica “Ag”: `A` and `g` each have one outer ring and one
  counter hole, each glyph extrudes to a closed solid, no cap triangle covers
  a counter, and the mesh XY bounds equal `CTLineGetBoundsWithOptions(.useGlyphPathBounds)`
  to the 1/1,000-em flattening tolerance. Line width, ascent and descent equal
  `CTLineGetTypographicBounds`. “AV” is narrower than its summed advances
  because Core Text kerns it. The system UI font (`.SFNS-Regular`, a variable
  font) extrudes `Bg8&@` to closed solids with the right counter counts. An
  emoji glyph without an outline throws. A sheared font matches Core Text's
  glyph-path bounds with the font matrix applied once.
  [Core Text glyph path documentation](https://developer.apple.com/documentation/coretext/ctfontcreatepathforglyph(_:_:_:))
- **Rendered text.** The flat “Ag” mesh is drawn by the Metal unlit pipeline
  into a 160 × 160 target and compared with `CTLineDraw` of the same line,
  antialiased at five times the resolution with subpixel positioning. Wherever
  Core Text covers at least 3/4 or at most 1/4 of the subpixel around a GPU
  pixel center, the GPU agrees: zero disagreements over more than 2,500 inside
  and 20,000 outside pixels. Total coverage agrees within 1%.
- **Decals.** A projector rotated 30° over a plane cuts exactly its 1 × 0.5
  footprint (area to 1e-5); every UV equals `(x / w + 0.5, 0.5 − y / h)` in
  projector space to 1e-5 and tangents follow the projector's X axis. On a
  scaled sphere only faces toward the projector are kept, the BVH and linear
  paths agree, and `maximumAngle: .pi` roughly doubles the area by reaching the
  back.
- **Hulls.** 2,000 random points in a box, three seeds: every point is behind
  every face plane to within 8 × tolerance, every vertex is behind every face
  (convex), each edge has one reversed twin and `V − E + F = 2`. All 400
  random sphere points are vertices, with `F = 2V − 4`. A 5 × 5 × 5 lattice
  with duplicates gives exactly 8 vertices, 12 triangles and volume 1. A
  corner only `1e-5` above a face is retained. A thin square frustum retains
  all eight vertices and matches its analytic volume.
- **CSG.** Unit cubes offset by (0.5, 0.5, 0.5) give union 1.875,
  intersection 0.125 and difference 0.875 to 1e-6; shifted by 0.5 along X,
  with four coplanar face pairs, 1.5, 0.5 and 0.5 with surface area equal to
  the exact box. Touching cubes merge into a 2 × 1 × 1 box of area 10.
  Intersecting a cube with itself turned 45° gives `2(√2 − 1)`. Cube and a
  96 × 48 sphere of radius 0.65 satisfy `V(A ∪ B) + V(A ∩ B) = V(A) + V(B)`
  and `V(A − B) = V(A) − V(A ∩ B)` to 1e-5, and the intersection lies below
  the analytic ball-and-cube value `4/3 πR³ − 6 · πh²(3R − h)/3` by less than
  the faceted sphere's own volume deficit. Random rigid placements of cubes,
  spheres, rounded boxes and cylinders, three seeds, keep both identities and
  closed results. A sphere turned by exactly one segment, coincident with
  itself only to Float rounding, merges as coplanar. Engraving and embossing
  “Ag” into a plaque changes its volume by exactly half the letters' volume.
  Cut faces carry the source face's UVs, which the cavity wall checks
  analytically, and subtracted faces flip normals and tangent handedness.
- Every result above is checked for closure by position: each edge is used
  exactly twice, in opposite directions.

Each suite was also run against deliberately broken code: collinear
reinsertion disabled, even-odd treated as nonzero, bevel normal weights
swapped, the front cap on the wrong profile level, the reflex miter limit
removed, glyph pen positions ignored, a stretched line width,
Quickhull stopped early or keeping non-extreme vertices, decal V flipped or
back faces kept, coplanar same/opposite swapped, edge splits not shared,
inside/outside inverted, flipped attributes, and a reversed front cap. Every
one makes the matching suite fail.

## Historical measurements

The original geometry revision `fad058c` was measured in Release on an Apple
M4 Max Mac Studio (Mac16,9, 36 GB) running macOS 27.2. These are the best of
three runs while other worktrees were building on the same machine.

| Work | Size | Time |
| --- | --- | --- |
| `TextOutline`, Helvetica 72 pt | “The quick brown fox jumps over the lazy dog”, 43 glyphs | 6.7 ms |
| `MeshGeometry.text`, Helvetica, depth 0.2, bevel 0.02 em × 4 segments | 34,400 triangles, 23,366 vertices | 12.6 ms |
| Same, system UI font, bevel 0.01 em | 36,352 triangles | 11.8 ms |
| `PolygonTriangulator`, random star polygon | 1,000 / 10,000 vertices | 1.2 ms / 163 ms |
| `ConvexHull`, random points in a cube | 100,000 points | 8.7 ms |
| `ConvexHull`, random points on a sphere, all vertices | 20,000 points | 107 ms |
| `decal`, 256 × 128 UV sphere, linear / with `MeshBVH` | 65,024 triangles | 19.7 ms / 4.3 ms |
| `subtracting`, unit cube minus 96 × 48 sphere | 9,024 + 12 triangles → 4,312 | 48 ms |
| `MeshBoolean` union of two offset 128 × 64 spheres | 16,128 + 16,128 → 24,788 | 174 ms |
| Engrave Helvetica “Ag”, extruded without bevel, into a 12-triangle plaque | 2 glyphs | 7.3 ms |

A pangram in the system font rejects a 0.02 em bevel: the crotch where the
`r` arm meets its stem is about 0.03 em thick, so an exact inset of 0.02 em
from both sides would split the glyph, and `extrude` reports
`bevelTooLarge` instead of producing crossing caps.

## Limits

- Text lines break only at line separators; there is no paragraph wrapping,
  vertical text or path-following layout. Glyphs that overlap each other, as
  in tight script fonts, become separate interpenetrating solids; overlapping
  contours inside one glyph are merged.
- Rings or shapes that touch at a vertex can produce extrusions that share a
  wall edge. These meshes render, but CSG rejects the non-manifold edge. The
  touching-hole test verifies the extrusion volume and the rejection.
- Bevels inset the caps, so a bevel larger than half a stroke makes rings
  cross or collapse and throws `bevelTooLarge`; only edge-collapse events of
  the straight skeleton are handled, not split events. Reflex corners use a
  miter limited to twice the inset, a one-chord stand-in for the round corner
  of an exact offset. There is no extrusion
  along a path or per-step depth subdivision.
- Ring resolution and triangulation are quadratic in the worst case. Resolving
  a large path can take longer than triangulating rings that are already valid.
- `ConvexHull` returns triangles; coplanar faces are not merged into polygons.
- Decals are clipped to triangles facing the projector; they do not follow
  skinned or morphed vertices unless built from the deformed geometry.
- CSG needs closed, consistently oriented two-manifold operands without
  self-intersections. Points within 1e-6 of the larger of the operands'
  extent and largest coordinate are one point. A result that touches itself
  along an edge or at a vertex is reported as `nonManifoldResult`. Winding
  numbers cost one pass over the other operand's triangles per patch. Hollow
  operations, symmetric difference, multi-operation evaluation, configurable
  attribute lists, and per-operand material groups remain open.
  [three-bvh-csg API](https://github.com/gkjohnson/three-bvh-csg)
- No GPU path: all builders run on the calling thread.
