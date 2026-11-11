# Logarithmic raster depth

Prepare a supported `MeshRenderer` with `depthMode: .logarithmic`
and use the same mode on the scene color and surface-attribute passes.
Standalone meshes use their prepared mode. Depth clears to one and uses
the forward depth comparison convention. A loaded depth attachment must
contain values from the same depth mode and camera range. Logarithmic
passes use the full viewport depth range.

Perspective clipping, jitter and projected X/Y use the forward camera
matrix. The fragment writes
`log2(viewDistance / near) / log2(far / near)` to the depth attachment,
with the near plane clamped to 1e-6 for the logarithm. This is the r187
node renderer's formula. The WebGL renderer uses a separate `log2(1+w)`
formula. Orthographic views keep linear depth. An explicit view
supplies a matching `ShaderGraphCamera` with a finite forward projection
and affine view matrix.
[Three.js r187 NodeMaterial](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/NodeMaterial.js#L640) ·
[Three.js r187 depth functions](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ViewportDepthNode.js#L257) ·
[Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

At large ratios, the Float forward projection can round away the far
plane. Culling reconstructs finite perspective clip planes from the
camera distances and clip W. The fragment also discards depths beyond
far for objects that cross the plane or disable culling. Transparent
sorting uses logarithmic view depth, avoiding ties from the rounded
forward projection.

The mesh pipelines cover unlit, textured, Lambert, shadowed Lambert,
and PBR materials including surface graphs. The color and
surface-attribute passes use the same mapping and graph cutouts.
Directional and punctual shadow maps use their own depth conventions.
Logarithmic mode currently requires one sample and one color target.
Classic and custom shaders, primitive and particle draws, indirect
command bundles, depth bias, and depth-reconstructing post effects need
a compatible pipeline. Scene archives currently reject logarithmic mesh
preparation, and selecting its depth mode is a runtime operation.

## Native checks

Metal API Validation passed 72 focused tests in 14 suites on macOS and
the iPhone 18 Pro Simulator. Readbacks verify overlapping far geometry at
a 10¹⁰ clip ratio in either draw order, finite-frustum culling, the shader
far cut with culling disabled, and forward, reversed, and logarithmic
standalone depths through a translated and rotated view. A generated PBR
cutout writes the same depth in color and surface attributes. Stock
shadowed PBR changes the camera depth while keeping the stored directional
shadow map identical. An explicit orthographic camera stores linear
depth. Invalid clip modes, clear depth, and unsupported consumers report
errors before GPU encoding. The Xcode MCP timed out resolving the package
destination, so these checks used `xcodebuild`.
