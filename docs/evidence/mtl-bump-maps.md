# Wavefront height maps

Standalone MTL and OBJ scene import retain `bump` and `map_bump` as
native height maps. The parsed material stores the encoded source
texture, signed `-bm` strength, and `-s`, `-o`, and `-clamp`
sampling settings. The vertical transform converts from Wavefront's
bottom-left origin to Frostfield's top-left UVs. The existing
material preparation loads the red channel as linear data and
supplies the texture to PBR.
[Three.js r187 MTLLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/MTLLoader.js)
[Native PBR height sampling](pbr-bump-maps.md)

A valid `norm` texture takes precedence when both are present.
The height source remains available in the imported material.
Preparation skips inactive height textures. Portable glTF and USDZ
export report an active height texture and require conversion to a
tangent-space normal map. A normal texture that overrides a height
map can be exported.

A prepared OBJ produced the analytic signed normal tilt from a red
ramp with authored scaling and offset. A normal texture on the same
OBJ superseded that tilt. The standalone MTL fixture loaded the same
image as both an sRGB color texture and a linear height texture.
The existing mixed-transparency fixture retained its exact RGBA
codes. A 44-test focused run passed on macOS and iPad Simulator
with Metal API and shader validation. Xcode 27.2 MCP timed out
resolving the destination, so verification used `xcodebuild`.
