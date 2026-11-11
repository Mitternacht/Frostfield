# SVG gradient fills

The SVG loader resolves local linear and radial gradient fills into
typed paints. Definitions may inherit stops and omitted attributes
through a local `href` or `xlink:href`. Stops accept color, opacity and
fractional or percentage offsets. Presentation attributes, inline
style and supported class and ID selectors also apply to stops.
Offsets are clamped and kept in authored order. Cyclic references
throw. Missing paint servers retain the original unresolved paint.
[SVG paint servers](https://www.w3.org/TR/SVG2/pservers.html)

Gradients support object bounding boxes, user space, a full affine
gradient transform, pad, repeat and reflect. User-space percentages
use the root viewBox or absolute root width and height. Radial
gradients preserve their focal point and their ellipse under
nonuniform scale. Stops interpolate in sRGB by default, with
`color-interpolation="linearRGB"` available.

Call `gradientColor(at:)` to sample at an authored document point, or
`prepareGradientFill` for a matching fill mesh and straight-alpha
linear RGBA16Float texture. Both include the element's combined fill
opacity. The texture samples the cap UVs and can be drawn with a
textured mesh material and normal alpha blending. The square texture
resolution defaults to 256 and can range from 1 to 1,024. Texture
filtering can soften abrupt stops and narrow repeat bands at low
resolutions. Preparation bounds the document to 2,048 gradient
definitions, 1,024 stops per definition and 16,384 stops overall.

Three.js r187 uses a 256-pixel canvas for each gradient axis and a
texture matrix. Frostfield uses the full inverse affine and a radial
distance parameter for spreading. Elliptical bounds, skewed gradients,
user-space percentages and percentage stop offsets can therefore
produce different samples from the reference. Gradient strokes remain
unresolved, as they do in r187. SVG2 focal start radius, nested
viewport layout and external paint references are outside this import.
[Three.js r187 SVGLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/SVGLoader.js)

## Verification

The pinned SVGLoader rendered the reference gradient textures from
`defs4.svg` and `blueprint.svg`. Across 44 bilinearly sampled reference
pixels from 11 gradient paints, the largest absolute linear channel
difference was 0.0093 on macOS and iPad Simulator. The focused
reference run passed 32 test declarations on each target. The
committed tests cover placement, units, stop opacity, interpolation,
radial focus, spread, invalid references and actual Metal pixels
from the reflected mesh.
[Three.js r187 SVG fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/svg)

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API
and shader validation enabled.
