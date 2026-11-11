# Shader texture bump maps

Declare a linear height texture with `texture(_:content:)` and pass
`ShaderTexture.bumpMap` to a surface graph's normal output. The red
channel stores height. The node takes UV0 or UV1, an affine texture
transform, and a typed signed scale. A graph uniform can change the
strength after preparation.

Raster shading uses forward differences over pixel UV derivatives.
Path tracing uses the matching UV offsets of its ray cone, with its
native or baked texture sampling. Both share the stock material's
surface-gradient calculation. The result is expressed in the
surface graph's UV0 tangent frame. A height map using UV1 requires
authored secondary coordinates and a valid graph tangent frame.
A graph normal output replaces the stock base shading normal.
[Three.js r187 BumpMapNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/BumpMapNode.js)
[Native PBR height sampling](pbr-bump-maps.md)

The node accepts the authored UV sets and affine transforms.
Three.js TSL also allows arbitrary procedural texture coordinates
and a caller-supplied starting normal. The native graph archive
preserves the node and its texture, transform and scale inputs.

The focused suite passed 56 tests on macOS and iPad Simulator with
Metal API and shader validation. It reads an analytic ramp through
a mirrored UV1 transform and an independently oriented UV0 frame,
checks a live sign change and an archive round trip, and validates
linear input and missing secondary coordinates. First-hit normal
AOVs match the analytic ramp using native and baked texture
sampling. macOS exercised software and Metal ray tracing. The
simulator exercised software ray tracing.

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData.
