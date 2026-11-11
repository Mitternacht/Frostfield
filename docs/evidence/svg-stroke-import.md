# SVG stroke surfaces

Each SVG element exposes its inherited stroke paint, local width,
opacity, cap, join, and miter limit. The style comes from presentation
attributes, simple stylesheets, and inline declarations. `strokedShapes`
outlines the source path in local coordinates and then applies the
element's affine transform. Nonuniform scale and skew therefore
transform the cap and join with the path. Open subpaths keep their
caps, closed outlines preserve their interior holes, and overlapping
stroke contours resolve as a union. `strokeGeometry` triangulates the
outline in the XY plane. The optional `yUp` flag reflects Y.
[SVG stroke properties](https://www.w3.org/TR/SVG2/painting.html#StrokeProperties)

Supported caps are butt, round, and square. Supported joins are
miter, miter-clip, bevel, and round, with the authored miter limit.
Core Graphics outlines the standard joins. Miter-clip tessellates a
ribbon and unions its coverage. It follows the pinned r187 clip
fraction along both miter edges. Arcs joins report an explicit
unsupported error. Dash patterns and non-scaling stroke are outside
this import. Paint references remain available as metadata for
caller-supplied materials.

The default mesh uses planar cap UVs. Call
`strokeGeometry(textureCoordinates: .path)` for a textured ribbon.
Each subpath maps its sampled segment indices across U from 0 to 1,
with V spanning left to right. Closed paths have a U seam at their
authored start. Open caps and corners use the local stroke width
before the affine transform. The ribbon retains overlapping
coverage at crossings and very short joins. Use the default outline
for a single layer of translucent coverage. Curve sampling follows
the native tolerance, so UV spacing on curves can differ from r187's
curve sampler. Frostfield returns indexed mesh data suitable for
Metal upload.

The importer bounds each stroke to 16,384 source path elements.
Ribbon tessellation admits at most 4,096 sampled centerline points
and 65,536 emitted vertices. Unioned outlines are limited to 8,192
edges.

## Verification

Four pinned SVG fixtures contained nine stroked paths. The six join
and cap examples matched the r187 stroke helper within 0.015 source
units in bounds. Their areas and the wide curve's area matched within
0.006%. The wide curve's butt endpoints differed by up to 0.813 units
because the r187 polyline approximates the end tangent. Close-path
restart bounds matched exactly. The four-square restart example
resolves to a 104 × 104 outer boundary and four 46 × 46 holes. The
r187 triangle sum includes shared stroke coverage twice.
[Three.js r187 SVG stroke helper](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/SVGLoader.js)
[Three.js r187 SVG fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/svg)

A Metal render samples the imported translucent stroke, its open
center, and the background. Focused tests also check cap area under
nonuniform scale, joins, miter limits and empty strokes. The miter-clip
fixture matches the pinned helper's left-turn bounds and area. The
native right turn preserves mirrored coverage where the pinned helper
collapses a flipped clip triangle. A separate Metal test reads three
texture samples at pinned r187 UVs on unequal stroke segments. Tests
also cover curved ribbons, closed seams and independent subpaths.

The combined SVG and shape suite passed 34 test declarations on Mac
and iPad Simulator. The Xcode 27.2 MCP timed out resolving the package
destination. Verification used `xcodebuild` with Metal API and shader
validation.
