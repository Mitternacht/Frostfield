# Planar water

`PlanarWater` creates raster water materials for authored planes. Build the material
with Metal capture textures and draw it on an ordinary mesh node. Encode
`PlanarReflection` and, for flowing water, `PlanarRefraction` before the main scene.
Use the same oriented world plane, source camera, depth mapping and viewport coverage
for the capture and water draw. Pass the water node UUID to exclude it from each capture.
Reflection and refraction use separate tracked RGBA16Float color targets and separate
depth attachments. The capture APIs own clipping and any optional background, mipmaps
or multisampling. The caller sets the resolution and capture cadence.

```swift
let waterMaterial = try PlanarWater.flowing(
    reflection: reflectionColor, refraction: refractionColor,
    firstNormal: normal0, secondNormal: normal1, flowMap: flowTexture
)
// After building the water mesh node with waterMaterial and before drawing it:
try waterMaterial.surfaceUniforms?.setFrame(.init(time: elapsedSeconds))
try PlanarReflection(plane: waterPlane, reflector: waterNode.id).encode(
    scene, camera: sourceCamera, color: reflectionColor, depth: reflectionDepth,
    commandBuffer: commandBuffer
)
try PlanarRefraction(plane: waterPlane, refractor: waterNode.id).encode(
    scene, camera: sourceCamera, color: refractionColor, depth: refractionDepth,
    commandBuffer: commandBuffer
)
```

`flowing` is the Water2 model. It advances a 0.15-unit phase and a half-cycle-staggered
phase on the live material clock, then crossfades two normal samples with a triangle
weight. Direction comes from an authored top-left UV0 vector or a linear RG flow map.
The decoded normal uses the mesh tangent frame for the Fresnel term. Distortion offsets
both captures in top-left viewport UV before reflection's horizontal correction.
The offset is weighted by forward projective depth, including under a reversed-Z
projection. The shader mixes premultiplied capture colors and multiplies them by
linear tint. It supports ordinary planar mesh transforms and independent capture resolution.
[Three.js r187 Water2](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Water2.js)

`reflective` is the shared Water and WaterMesh model for a horizontal Y-up surface.
It samples four moving normal layers in world XZ at the r187 spatial scales and
phase rates, computes a distance-dependent distortion, and mixes water scatter with
reflected radiance and an authored sun highlight. It accepts linear tint, sun color,
sun direction, distortion, wave scale and surface opacity. The caller's clock drives
motion without rebuilding either material or pipeline.
[Three.js r187 Water](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Water.js),
[Three.js r187 WaterMesh](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/WaterMesh.js)

## Fog, shadows and custom shading

Set either factory's `options.fog` to `PlanarWater.Fog` with a linear RGB color.
The linear variant uses smoothstep between two view-axis distances. The exponential-
squared variant uses density in inverse world units. Both blend the surface color
before display conversion and preserve alpha. Fog is local to the water material.
[Three.js r187 fog fragment](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/fog_fragment.glsl.js)

Set `ReflectionOptions.receivesSunShadows` and pass a matching `DirectionalShadowMap`
to the main scene encode. The existing renderer samples that map at the water
surface. Visibility scales scatter and diffuse sunlight, leaving the reflected
image and reflected highlight intact. The map follows the renderer's configured
filtering, bias and cascades. This input covers the directional light.
[Three.js r187 Water](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Water.js)

The `flowing` factory accepts a typed `shade` closure. It can replace the water
color, coverage and object-space vertex offset while retaining the live capture
bindings and flow clock. The inputs include both capture samplers, both normal
maps, the decoded flow direction, phase, world normal, Fresnel and distorted UV.
Additional textures are declared in the graph and passed by name.

```swift
let stylized = try PlanarWater.flowing(
    reflection: reflectionColor, refraction: refractionColor,
    firstNormal: normal0, secondNormal: normal1, additionalTextures: ["Foam": foam]
) { surface, water in
    let through = water.refraction.samplePlanarRefraction(at: water.coordinates).rgb
    let detail = surface.texture("Foam").sample(surface.uv0).rgb
    return .init(baseColor: through.mix(detail, by: water.phase / 0.15), opacity: water.opacity)
}
```

The factories use raster viewport coordinates. They route composed RGB through
the material's opt-in HDR unlit graph emission, preserving radiance up to the
RGBA16Float limit while keeping authored premultiplied coverage. Custom Water2
graphs still supply their straight water color through `baseColor`. The factory
routes that output to HDR radiance. The caller schedules captures, supplies the
scene view and can choose capture lighting and background.

Metal readbacks compare the expected mix from two independently captured scenes.
Separate fixtures cover normal and grazing Fresnel, moving phase and wrap, RG flow
advection, four-wave motion, linear and exponential-squared fog, a real directional
caster and a custom Water2 graph with an additional texture. HDR fixtures verify
above-one reflection and refraction after real capture and water composition, and
above-one reflective water with partial alpha. Archive v27 preserves the general
unlit graph emission opt-in.

## Depth-safe viewport refraction

`PlanarWater.flowingViewport` is the Water2Mesh variant. It applies a normal-based
UV offset at constant strength to the reflection and refraction lookups. The
reflection still uses a caller-encoded planar capture. Refraction samples the
main view's opaque color. When the shifted lookup reaches an opaque fragment in
front of the water, `ShaderGraphBuilder.viewportSafeUV(at:)` uses the unshifted
screen pixel instead. A farther opaque fragment or clear depth keeps the shifted
lookup. The depth and color snapshots must cover the same attachment and view.
[Three.js r187 Water2Mesh](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Water2Mesh.js),
[Three.js r187 viewportSafeUV](https://github.com/mrdoob/three.js/blob/r187/src/nodes/utils/ViewportUtils.js)

```swift
let waterMaterial = try PlanarWater.flowingViewport(
    reflection: reflectionColor, firstNormal: normal0, secondNormal: normal1
)
try waterMaterial.surfaceUniforms?.setFrame(.init(time: elapsedSeconds))
// Encode the planar reflection with the water node excluded, then set the
// snapshot for the main scene's opaque and transparent draws.
scene.viewportSnapshot = try SceneViewportSnapshot(
    color: snapshotColor, depth: snapshotDepth, source: .opaquePass
)
```

For a pre-rendered opaque scene, supply the same-view color and depth with
`.stored`. The shader uses whole-attachment UVs even when the water draws in
an offset viewport. The live snapshot is a per-pass binding. Archive v28
preserves the graph, and the caller supplies a snapshot again after loading.

Metal readback verifies that a red foreground occluder remains visible in its
own pixels without appearing in displaced refraction on the water. A blue
object behind the water still appears through the distorted lookup. The fixture
uses both projected depth directions with an offset viewport and non-default
depth range, plus logarithmic perspective depth.
