# Physically based light maps

A `PhysicallyBasedMaterial.LightMap` holds linear RGB irradiance and an
atlas transform. It samples UV1 by default and can use UV0. Metal uses
linear filtering and clamps sampling to the atlas edges. Alpha is
ignored. `MeshLightmapBaker.pbrLightMapBinding()` exposes the same
irradiance texture and UV set returned by a native direct or progressive
bake. `lightMapIntensity` defaults to one, and zero disables the map.

The raster material adds the irradiance multiplied by intensity to its
indirect diffuse response. It applies the diffuse albedo and reciprocal
π once, including the native material's diffuse energy and occlusion
terms. Ambient and direct light keep their own intensity. Baked and live lighting add, so a
source authored in both contributes twice. Unlit materials ignore the
map. The optional PBR binding uses the existing texture-handle table so
it fits the Simulator's native texture and sampler limits.
[Three.js light-map shader r187](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/lights_fragment_maps.glsl.js),
[Three.js physical diffuse response r187](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/lights_physical_pars_fragment.glsl.js)

Scene archive v22 stores the texture, UV set, transform and intensity.
Materials without light-map settings still use their previous archive
version. Live factor changes preserve the binding and its intensity.
glTF export reports the dropped map through `materialLayersDropped`.

## Path tracing

Software BVH and Metal ray tracing evaluate the same light-map binding at
camera and scattered hits within the configured scatter budget. Path
materials use the existing texture table with clamped UVs and ray-cone mip
selection. A direct binding samples the source texture and its mip chain.
The fallback resamples it into the scene's RGBA16Float texture array and
builds its mip chain. Path texture samples use the existing half-float
range and precision.

The map adds a diffuse reflection with the material's albedo and the
irradiance-to-radiance conversion of reciprocal π. Dielectric
reflectance, metalness, transmission, and coat attenuate this lobe.
For rough diffuse materials the tracer uses the hemispherical albedo
because a scalar irradiance per RGB channel carries no incident direction.
The term is separate from authored emission, emitter selection, and the
first-hit emission and albedo outputs. Multiple importance sampling still
weights authored light transport. Keep the map's baked sources out of the
live lighting rig to avoid counting the same source twice.
[PBRT diffuse reflection](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection) ·
[PBRT path integration](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer)

## Raster verification

All 63 focused tests in nine suites passed on arm64 macOS and the iPhone 18
Pro Simulator with Metal validation. GPU readback checked UV0, UV1, atlas
transforms, authored mip blending, HDR irradiance and the independent live
lighting term. The baked atlas also round-tripped through archive v22 and
rendered through the generated shadowed PBR shader with an optical layer.
The test used `xcodebuild` with internal DerivedData after the Xcode bridge
encountered a full volume at its configured DerivedData location.

## Path verification

All 34 focused tests in seven suites passed on arm64 macOS and the iPhone
18 Pro Simulator with Metal validation. The Mac exercised both software BVH
and Metal ray tracing. The Simulator exercised software BVH. GPU tests checked
the direct texture table and RGBA16Float atlas fallback, both UV sets, transform
clamping, disabled intensity, and the zero diffuse term for a pure metal.
The native baker fixture combined independently authored live light with the
map and checked albedo and emission output. A mirror fixture verified the
expected irradiance contribution at a later hit on a sampled emissive surface
with MIS enabled and disabled. The test used internal `xcodebuild` DerivedData
after the Xcode bridge reported a full volume.
