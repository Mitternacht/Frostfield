# SVG paths and filled surfaces

`SVGLoader.load` reads UTF-8 or ASCII SVG from local files, data or text.
It returns immutable Core Graphics paths in document order and source
coordinates. It accepts absolute and relative M, L, H, V, C, S, Q, T, A
and Z commands, repeated arguments, adjacent arc flags, rounded
rectangles, circles, ellipses, polygons, polylines and lines.
Elliptical arcs use short cubic segments with the authored endpoint.
[SVG path data](https://www.w3.org/TR/SVG2/paths.html)
[Three.js r187 SVGLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/SVGLoader.js)

Matrix, translate, scale, rotate and skew transforms accumulate through
nested elements in SVG order. Coordinates retain positive Y downward.
Call `filledShapes(yUp: true)` or `fillGeometry(yUp: true)` to reflect
them into a Y-up scene. The root viewBox is retained as metadata.
Absolute shape lengths support px, in, cm, mm, pt and pc, with a
configurable default of 90 dots per inch.

Presentation attributes and inline style provide inherited solid fill,
currentColor, visibility, fill rule and opacity. Colors are stored as
straight-alpha linear sRGB. The flattened elements carry the product
of ancestor and element opacity and fill opacity. Fill geometry
preserves the authored nonzero or even-odd rule, closes open fill
contours, and triangulates regions with holes. Unresolved paint
references are retained for the caller.
[SVG fill and stroke properties](https://www.w3.org/TR/SVG2/painting.html)

Percentage shape lengths, viewport layout, text, masks and filters
are outside this import.
[SVG stroke surfaces](svg-stroke-import.md)
[SVG local instances and stylesheets](svg-local-style-import.md)
[SVG gradient fills](svg-gradient-import.md)
The reader recognizes the standard SVG 1.1 public declaration in the
prolog and removes it before XML parsing. It rejects custom DTDs and
internal subsets. External entity resolution is disabled. The
original r187 `tiger.svg` and `roundJoinPrecisionIssue.svg` imported
directly with 240 and two paths respectively. The Mac run passed 24
tests with Metal API Validation, including the original-file probe and
a persistent declaration and entity regression.
It limits input to 8 MiB, nesting to 64, XML nodes to 32,768, commands
per path to 16,384, and document path elements to 262,144. Each fill
is limited to 4,096 flattened points and 8,192 edges after splitting.

## Verification

Eight pinned r187 assets matched 83 paths and 101 subpaths on Mac and
iPad Simulator. The maximum difference in bounds was 0.00236 source
units across transformed ellipses. The eight area comparisons differed
by at most 0.021%. For the emoji, the reference uses nonzero winding
over the pinned loader's sampled contours. Summing the loader's
overlapping shape areas counts some regions twice. An actual Metal
render verifies translucent imported fill, its open hole, and the
transparent background. The focused run passed 18 test declarations
on each target.
[Three.js r187 SVG fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/svg)

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API
and shader validation enabled.
