# Physically based retroreflection

`PhysicallyBasedMaterial(retroreflectivity:)` accepts a finite value from zero
through one. Zero preserves ordinary reflection. At one the base specular
GGX lobe evaluates the outgoing view direction mirrored about the shading
normal, so its peak returns toward the light. Intermediate values blend the
two lobes and their direct diffuse Fresnel terms. Both the material's
directional light and the scene's directional, point and spot lights use
this response with their usual visibility. Native GGX and layer factors
still govern lobe strength.
[Three.js material r187](https://github.com/mrdoob/three.js/blob/r187/src/materials/MeshPhysicalMaterial.js#L486-L511) ·
[Three.js direct shader r187](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/lights_physical_pars_fragment.glsl.js#L533-L570)

The filtered base environment reflection blends from the ordinary
mirror lookup to one toward the view direction, biased toward the
reflection normal at high roughness. Rectangle LTC and clearcoat use their
ordinary reflection paths, as in Three.js. The native control is a material
scalar. Three.js also offers a node input override.
[Three.js environment shader r187](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/envmap_physical_pars_fragment.glsl.js#L42-L95) ·
[Three.js node material r187](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/MeshPhysicalNodeMaterial.js)

Scene archive v23 stores nonzero strength. Older archives load with the
default, and captures with default strength retain their previous archive
version. Live factor copies preserve the authored strength. glTF export
reports dropped retroreflection.

All 56 focused tests in eight suites passed on arm64 macOS and the iPhone
18 Pro Simulator with Metal validation. GPU fixtures compared both
directional light paths against the CPU GGX response at the ordinary
and retroreflective peak directions. They also verified intermediate
strength, filtered HDR environment colors, and archive round-trip pixels.
The test used internal `xcodebuild` DerivedData after the Xcode bridge
reported a full volume.

## Path tracing

The same material controls the base GGX reflection in software BVH and
Metal ray tracing. Reflection evaluation uses the mirrored outgoing view
in the shading frame. The tracer samples visible GGX normals from both
the ordinary and mirrored views. One mixture PDF drives scattering and
MIS for sampled lights, environment rays, and light hits. An ordinary
proposal remains available for thin and thick transmission, including
index-matched straight-through events. The authored strength also blends
the direct diffuse Fresnel terms.
[PBRT path integration](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer)

Native path-traced area lights use this BSDF for both strategies. The
separate raster rectangle-light integration retains its ordinary LTC
response, matching the r187 raster rectangle path.
[Three.js rectangle shader r187](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/lights_physical_pars_fragment.glsl.js#L452-L499)

With Metal validation, 28 focused tests in seven suites passed on arm64
macOS and the iPhone 18 Pro Simulator. The Mac ran both software and
Metal ray tracing. The Simulator ran the software BVH. The GPU cases
checked punctual GGX peaks and intermediate strength, and compared
BSDF-only and MIS estimates of area lighting with independent CPU
quadrature. Thin and index-matched transmission kept the same
throughput within the sampling tolerance. On Mac, zero-strength output
bytes matched the pre-change PCG and Sobol fixtures on both traversal
backends. The broader Mac run passed 49 tests in nine suites.
