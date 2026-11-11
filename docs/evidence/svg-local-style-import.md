# SVG local instances and stylesheets

The SVG loader resolves local `use` elements through either `href`
or `xlink:href`. Definitions can appear before or after the instance.
The loader expands referenced shapes, groups and symbols in document
order, preserving the original element IDs and applying the instance's
x and y, transform, and inherited style. A definition only contributes
geometry when referenced. External and missing references are skipped.
Cyclic references throw, and expansion has its own 64-level depth and
65,536-visit limits.
[SVG use elements](https://www.w3.org/TR/SVG2/struct.html#UseElement)

Embedded CSS accepts simple class and ID selectors, comma-separated
selector lists, and multiple classes per element. Styles in defs are
included. Presentation attributes, classes in authored order, IDs, and
inline styles apply in that order, as in the pinned SVGLoader. The
stylesheet subset shares the same paint, opacity, visibility, fill rule,
and stroke properties as inline styles. Stylesheets are bounded to
256 KiB and 4,096 selectors. External stylesheets, at-rules, complex
selectors, and symbol viewport sizing are outside this import.
[SVG styling](https://www.w3.org/TR/SVG2/styling.html)
[Three.js r187 SVGLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/SVGLoader.js)

## Verification

Six pinned r187 assets matched all 19 paths, bounds, fill colors,
stroke colors, and opacities on macOS and iPad Simulator. The fixture
run passed 27 test declarations on each target. A Metal fixture
samples two differently styled uses of one definition and the
transparent gap between them. The committed tests cover style
precedence, CDATA inside defs, forward references, transform order,
cyclic references, and the depth limit.
[Three.js r187 SVG fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/svg)

Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API
and shader validation enabled.
