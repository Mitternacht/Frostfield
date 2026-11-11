# Raw Metal mesh programs

Use `MeshMaterial.rawMetal` with a `RawMeshProgram` to supply both Metal
stages for a raster mesh. Each function declares its own varyings and
argument indices. Native indexed draws supply `vertex_id` and four
float4 values per vertex for position, normal, UV0/UV1 and tangent.
Vertex buffers 1 through 3 carry object-to-clip, normal and
object-to-world matrices. Fragment buffer 0 carries per-draw material
parameters. Shaders can read additional caller buffers, textures and
samplers. The caller synchronizes changes to those resources with GPU use.

Frostfield reflects active arguments when preparing the pipeline. It
checks function stages, required bindings, reserved layouts, device
identity, buffer ranges and alignment, texture types and resource access.
It checks texture feedback against attachments before beginning a draw.
Ordinary indexed raster color passes support both standalone meshes and
scene graphs. For a vertex shader that moves the silhouette beyond its
mesh, supply conservative node bounds or disable frustum culling.
Generated shadow, selection and surface-attribute passes use separate
material contracts. Path tracing and scene archives require an
interchangeable material representation.

The Metal test shader moves a square with a caller buffer and passes
a custom varying into a fragment that samples a bound texture and
sampler. Pixel readback matches between standalone and graph encodes.
The validation fixture exercises missing resources, reversed function
stages, indirect resource buffers, integer texture mismatch, attachment
feedback and auxiliary pass rejection. With Metal API Validation, 52
tests across six suites passed on macOS and iPhone 18 Pro Simulator,
including the existing custom-fragment, MRT, archive and raster-state
tests. The Xcode bridge timed out resolving the package run destination.
The runs used `xcodebuild` with internal DerivedData.

[Three.js r187 RawShaderMaterial](https://github.com/mrdoob/three.js/blob/r187/src/materials/RawShaderMaterial.js)
· [Metal resource binding reflection](https://developer.apple.com/documentation/metal/mtlpipelineoption/bindinginfo)
