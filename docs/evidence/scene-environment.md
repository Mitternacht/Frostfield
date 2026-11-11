# Scene image-based lighting

Assign a prepared `EnvironmentLighting` to `SceneGraph.environment`
to light raster PBR materials that have no environment of their own.
The scene uses its `environmentIntensity` and `environmentEulerRotation`.
The Euler vector is in radians, with the default Three.js XYZ order
`Rx * Ry * Rz`. The existing scalar `environmentRotation` adds to Y.
The shaders sample with the inverse rotation. An authored material
environment takes precedence with its own intensity and yaw. Existing
scenes without a shared source can still use the graph's intensity and
scalar yaw to adjust authored material maps.
[Three.js r187 scene fallback](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/MeshStandardNodeMaterial.js#L99) ·
[Three.js r187 environment controls](https://github.com/mrdoob/three.js/blob/r187/src/nodes/accessors/MaterialProperties.js)

The environment-enabled color and surface-attribute pipelines prepare
once per eligible mesh and pass on first use. Subsequent source
replacements only bind the new cube maps and BRDF table. An in-flight
command buffer keeps the environment it encoded. Sources must be
prepared on the scene's Metal device. Unlit and classic materials keep
their own shading behavior.

Archives use version 16 when a scene environment is present and version
18 when `environmentEulerRotation` is nonzero. Older captures read zero
Euler angles and retain their original yaw. The scene and materials can
refer to the same archived texture set, and reloading restores shared
ownership.

Raster background drawing and the standalone screen-space reflection
fallback take their own sky inputs and controls.

## Path tracing

`PathTracer.Lighting(queue:graph:)` snapshots the scene's visible
rectangle and analytic lights and takes its environment when no explicit
sky is supplied. A caller-supplied sky keeps its intensity and yaw.
`Lighting.skyEulerRotation` reports the effective orientation. The path
tracer uses its inverse for radiance and PDFs and the forward rotation
for importance samples. `EnvironmentMap.sample` and `pdf` expose the
same transform for CPU queries. `backgroundVisible` still controls what
the camera sees on a miss, independently of environment light on surfaces.

The first snapshot for an `EnvironmentLighting` samples mip zero of its
linear radiance cube with a Metal compute pass. The output is a 2:1
half-float panorama of four texels per face-width in longitude and two in
latitude, bounded to 1,024 × 512. The existing path map prepares its
luminance sampling distribution from that panorama. Prepare the first
snapshot outside the frame loop. Later snapshots for the same source
reuse the immutable path map. Intensity, orientation, visibility, light,
or source edits yield a different lighting snapshot and require a new
accumulation. Both the software BVH and the Metal intersector use the same
map and sampling table.

## Native checks

Metal API Validation passed 62 focused tests in 12 suites on Mac and
the iPhone 18 Pro Simulator. GPU readback verified live replacement
with an older unretained frame still queued, shader-graph fallback
with yaw and intensity, material precedence and the legacy no-source
controls, surface-attribute LUT selection, and classic and unlit
color. Replacing a source reused the warmed pipeline count in both
passes. The archive test restored the shared texture set and reproduced
the original pixels. Xcode MCP timed out resolving the package
destination, so verification used `xcodebuild`.

The path snapshot passed 30 focused tests in seven suites on Mac with
Metal API Validation. The iPhone 18 Pro Simulator passed 29, with the
Metal intersector case skipped by its hardware capability check.
The converted cube's panorama pixels and sampling densities matched
a directly authored directional panorama. Both path backends matched
its camera radiance and diffuse surface energy, and hiding the camera
background preserved the surface pixels. Repeated snapshots extended
an accumulation and an edited lighting snapshot required a restart.
Xcode MCP timed out resolving the package destination, so verification
used `xcodebuild`.

The XYZ fixture used the matrix returned by the tagged r187 Euler code.
Metal readback verified the expected cube face after rotation, preserved
material precedence and pipeline count, and matched archived pixels.
Importance-sampled directions and their reciprocal PDFs matched the r187
matrix. The software BVH and Metal intersector matched camera radiance
and diffuse surface energy from an independently rotated panorama.
Xcode 27.2 ran 82 focused test declarations in 12 suites on Mac and
81 on iPhone 18 Pro Simulator, which skipped the Metal intersector case.
The Xcode bridge timed out resolving the package destination, so these
checks ran through `xcodebuild`.
