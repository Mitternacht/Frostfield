# Planar reflection

`PlanarReflection` captures a scene through an oriented `Plane3D` into caller-owned color and depth textures. Supply the main view's split `ShaderGraphCamera`, including any projection crop or jitter already applied to the camera. The reflected render uses the authored positive half-space, excludes the reflector's whole subtree, and accepts additional exclusions for other mirrors. The draw uses the graph's usual material, light, layer and geometry inputs. A back-facing or inactive mirror clears its targets.

```swift
let mirror = PlanarReflection(plane: plane, reflector: mirrorNodeID)
try mirror.encode(
    scene, camera: graphCamera, color: reflectionColor, depth: reflectionDepth,
    depthMode: .reversed, generateMipmaps: true, commandBuffer: commandBuffer
)
```

Bind the tracked RGBA16Float color target as a surface-graph texture. `samplePlanarReflection()` projects viewport UV into the capture, so a mirror mesh can route the result to emission or mix it into a material. Encode the main surface after the capture on the same tracked command buffer. `reflectedView(_:depthMode:)` exposes the virtual camera's oblique projection for consumers that need to unproject the stored Depth32Float texture.

```swift
let mirrorSurface = try SurfaceShaderGraph { graph in
    let capture = graph.texture("Reflection", addressing: .clampToEdge)
    return .init(baseColor: capture.samplePlanarReflection().rgb)
}
let material = try PhysicallyBasedMaterial(
    surfaceGraph: mirrorSurface, surfaceTextures: ["Reflection": reflectionColor], shadingModel: .unlit
)
```

The virtual camera reflects position and basis and flips camera X to preserve winding. The conjugated projection preserves the original viewing window after the sampling flip. The mirror's positive half-space replaces the original near plane, with the opposite clip plane fit to the far corner and normalized for Metal's forward or reversed depth. The oblique projection shifts the far plane as part of this construction. Perspective and orthographic inputs are supported. [Three.js r187 ReflectorNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/utils/ReflectorNode.js), [Eric Lengyel, Oblique View Frustum Depth Projection and Clipping](https://terathon.com/lengyel/Lengyel-Oblique.pdf)

Supply a prepared `EnvironmentBackground` to fill the capture behind the
geometry. Directional sources use the reflected camera and its original
viewing window before oblique clipping. Geometry then loads that color and
uses the oblique depth projection. Background intensity and orientation
remain separate from scene image-based lighting. An inactive mirror clears
its outputs.

For multisampling, prepare reusable `PlanarReflection.MultisampleTargets`
and use the same sample count for the reflected mesh, primitive, and
background pipelines. Metal resolves color to the single-sample capture.
Depth selects the nearest covered sample, using minimum forward depth
or maximum reversed depth. Color mips are built from the resolved level
zero. Reuse the targets in queue order and check native sample-count and
depth-resolve support before allocating. Transmission keeps its existing
single-sample capture requirement.
[Three.js r187 Reflector](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Reflector.js#L75) ·
[Metal color resolve](https://developer.apple.com/documentation/metal/mtlstoreaction/multisampleresolve) ·
[Metal depth resolve](https://developer.apple.com/documentation/metal/mtlrenderpassdepthattachmentdescriptor/depthresolvefilter)

For multiple mirrors, build a tree of `PlanarReflection.Capture` values.
A child captures a mirror from its parent's reflected pose. Give each
occurrence its own color and depth storage, including when one mirror
appears under several source views. The same base scene can supply
separate prepared material bindings for each view.

```swift
let bThroughA = PlanarReflection.Capture(mirrorB, color: bThroughAColor, depth: bThroughADepth)
let a = PlanarReflection.Capture(mirrorA, color: aColor, depth: aDepth, children: [bThroughA])
let b = PlanarReflection.Capture(mirrorB, color: bColor, depth: bDepth)
let rootColors = try PlanarReflection.encodeBounces(
    [a, b], camera: graphCamera, depthMode: .reversed, commandBuffer: commandBuffer
) { path, childColors in
    try sceneWithMirrorBindings(path: path, colors: childColors)
}
// Bind rootColors in the source view and encode it after the captures.
```

The provider is called after its direct child passes have been encoded.
Bind their color textures with `samplePlanarReflection()` and use the
application's fallback material for terminal mirrors. The reflected pose
inherits the parent view's unclipped projection window before applying
its own oblique plane. Each pass excludes all reflector subtrees on its
path. An inactive capture clears its own attachments and skips its
children. Bindings that read another capture's storage fail unless they
read the color output of a direct child.

The explicit plan allows four levels and thirty-two total captures. It
supports a shared environment background and each view's configured
sample count, clear color, and mip generation. Prepare compatible
materials and pipelines for each view.

`Capture.Attachments` adds directional and punctual shadow maps and
optical transmission to individual views. Construct maps on the capture
queue and give each view its own map set and scratch texture. Cascades
fit the active reflected camera's unclipped viewing window and use the
map's configured split scheme. On-demand maps keep their own update
requests and hold their command buffer lease until completion. For
transmission, supply a tracked shader-readable RGBA16Float
texture with the capture dimensions and a complete mip chain. Each view
copies and filters its own opaque image before drawing transmitting
geometry. Transmission uses single sampling. The plan rejects shared
attachments and graph reads from other views' scratch storage.

`PlanarReflectorSet` can plan this tree automatically. Register authored
node IDs with their oriented mesh-local planes and give the set its own
render queue and capture size. The set uses resolved world transforms,
hierarchy visibility, camera layers, facing and conservative frustum
culling. It replaces each active node with a projected HDR mirror
material that shares its authored geometry. The source graph keeps its
original materials, which also serve as fallbacks for terminal or
inactive views.

```swift
let reflectors = try PlanarReflectorSet(
    queue: queue, width: 512, height: 512,
    mirrors: [.init(nodeID: mirrorNodeID, localPlane: localPlane)], maximumDepth: 2
)
let reflected = try reflectors.encode(scene, camera: graphCamera, commandBuffer: commandBuffer)
try reflected.graph.encode(
    pass: mainPass, commandBuffer: commandBuffer,
    clipFromWorld: graphCamera.projectionMatrix * graphCamera.viewMatrix,
    cameraPosition: cameraPosition, graphCamera: graphCamera
)
```

Use the source camera, including any authored jitter, for the main draw.
Submit that draw on the same command buffer and finish it before encoding
another frame with the set. The set caches color, depth and prepared
bindings by active view path. Shallow views are planned first within the
configured depth, 32-capture and 256 MiB logical storage limits. Automatic
mirrors use static planar geometry and single-sample RGBA16Float pipelines.
The explicit capture tree remains available for custom mirror shaders and
multisampled capture. Recursive layer refraction and water are separate
capabilities.

## Blurred reflector materials

`PlanarReflectionBlur` filters the captured HDR color on the capture's
command buffer. Supply shader-readable Depth32Float for depth scaling
and separate shader-readable RGBA16Float and optional R16Float outputs.
The filter alternates between tracked scratch textures through five
four-tap passes. Radius is in capture pixels per axis. The proximity
response uses stored device depth, with the same near-to-far convention
for forward and reversed depth. Near pixels contract the blur taps and
far pixels spread farther. The optional R16Float output stores the
proximity weight for material mixing.

```swift
let blur = try PlanarReflectionBlur(
    device: queue.device, settings: .init(radius: SIMD2(8, 3), depthScale: 1)
)
try blur.encode(
    source: reflectionColor, depth: reflectionDepth, destination: blurredColor,
    depthFactor: reflectionProximity, depthMode: .reversed, commandBuffer: commandBuffer
)
let mix = try PlanarReflectionMix(blurMix: 1, strength: 0.8, mirror: 0.7)
let reflector = try SurfaceShaderGraph { s in
    let sharp = s.texture("Sharp", addressing: .clampToEdge)
    let blurred = s.texture("Blurred", addressing: .clampToEdge)
    let proximity = s.texture("Proximity", content: .data, addressing: .clampToEdge, filtering: .nearest)
    return .init(baseColor: s.baseColor * mix.multiplier(sharp: sharp, blurred: blurred, depthFactor: proximity))
}
```

Bind the captures to those texture names when preparing a PBR material.
`PlanarReflectionMix` uses the current material roughness or an explicit
shader value, so a roughness texture can vary the sharp-to-blurred mix.
The mirror, strength and contrast controls apply in linear space. An
optional linear red distortion map displaces the projected lookup in
clip units before the homogeneous divide. An explicit camera supplies
the graph's view and projection matrices when encoding a custom view.
Set a mirror's `appearance` to use the same blur and mix in
`PlanarReflectorSet`. The mirror node needs an authored PBR material
with fixed factors and without a surface graph. The generated overlay
preserves its material maps, roughness, and lighting model. Each view
writes its own blur and proximity targets before its parent can sample
them. The owner counts these targets and the shared scratch against its
capture budget. Mirror descriptors without an appearance use the sharp
projected material.

The normal-map projection uses the authored RGB tangent-space normal
map, its UV set and transform, and the material's scalar normal UV scale.
The automatic owner takes the projected depth from that mirror's
pre-clip camera. An explicit graph can supply the normal texture and
projected depth to `PlanarReflectionMix.multiplier`. The normal offset
moves sharp and blurred color while the depth weight follows the red-map
distortion coordinate. Use `usesAuthoredNormalMap: false` when a material
uses a packed or object-space normal map, a custom sampler with no
description, or sampler settings the typed surface texture cannot
represent. That material still uses its original normal for lighting.

The filter and material controls follow drei 10.7.9 at commit
`66b9bda91eee0ae98837e16c053888c34341421e`. The native radius is measured
in texels. [Convolution](https://github.com/pmndrs/drei/blob/66b9bda91eee0ae98837e16c053888c34341421e/src/materials/ConvolutionMaterial.tsx) ·
[Material](https://github.com/pmndrs/drei/blob/66b9bda91eee0ae98837e16c053888c34341421e/src/materials/MeshReflectorMaterial.tsx)

## Verification

- The GPU fixture uses a tilted mirror, a rotated and translated scene hierarchy, and off-center source cameras. Red and green reflected markers match analytic mirror-ray positions both in the captured image and on the surface graph's mirror, for perspective and orthographic cameras in both depth modes.
- A nearer blue marker on the wrong side disables CPU culling and is clipped on the GPU. A second marker belongs to the excluded mirror subtree. Neither appears in the capture.
- The back face clears color, depth and generated mip levels. Validation covers feedback through a view of the color target, explicit exclusions, target usage and mismatched camera depth mapping.
- Xcode 27.2 ran 40 focused tests in seven suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. Xcode MCP timed out resolving the package destination, so the runs used headless `xcodebuild`.

The follow-up passed 41 focused tests in eight suites on Mac with Metal
API Validation. The iPhone 18 Pro Simulator passed 40. Its GPU reports
4× raster sampling but lacks the Apple GPU family capability for native
depth resolve, so the multisample fixture was skipped there. On Mac, the
4× capture resolved partially covered edges against the background,
selected the nearest depth in both conventions, and generated the final
mip from the resolved image. Analytic sky-ray readbacks passed on both
destinations for tilted, off-center perspective and orthographic views.
An inactive multisample capture resolved its clear values. Xcode MCP
timed out resolving the package destination, so these runs used
`xcodebuild`.

The bounce fixture reads the same mirror from two virtual cameras. A
right-angle bounce through mirror A reaches a red marker in B's child
capture, while the source camera sees a green marker in B's root
capture. The final view contains both colors in the expected mirrors
with forward and reversed depth. Additional checks cover ancestor
exclusion, inactive child clears, sibling texture aliases, reused
attachments, duplicate reflector IDs, and the depth limit. The focused
run covered 60 test declarations in nine suites on Mac and iPhone 18
Pro Simulator. The simulator skipped the existing native multisample
depth resolve fixture. The Xcode bridge timed out resolving the package
destination, so focused checks ran through `xcodebuild`.

The view-local attachment checks compared lit and shadowed pixels from
independent on-demand cascaded directional and spot maps. In the
A-to-B bounce, glass tinted the red marker while each B view's opaque
backdrop retained its own untinted color. Forward and reversed depth
passed. Validation rejected shared scratch, scratch bound as a graph
texture, missing mips, reused shadow maps, and a wrong command queue.
Xcode 27.2 passed 45 focused tests in nine suites on Mac and 44 on
iPhone 18 Pro Simulator, which skipped the existing native multisample
depth-resolve case. The transmission regression reads its final mip
through an explicit blit because the simulator's direct texture
readback did not return that mip. Xcode MCP timed out resolving the
package destination, so verification used `xcodebuild`.

The automatic fixture used a mirror under nonuniform scale and another
mirror with different main and reflected source views. GPU readback
showed the expected colors under both depth conventions. One level
showed the authored fallback where two levels showed the second bounce.
A mirror outside the main view was captured through a reflection, while
hidden, off-layer, and fully offscreen nodes were omitted. Checks also
covered planar geometry validation, reuse, and overlapping encodes.
Xcode 27.2 passed 49 focused tests on Mac and 48 on iPhone 18 Pro
Simulator, which skipped the existing native multisample depth-resolve
case. Xcode MCP timed out resolving the package destination, so
verification used `xcodebuild`.

The blurred material fixture projected two black-and-white edges at
physically different depths into a tilted reflection. On both depth
conventions, the near edge stayed sharp and the far edge spread into
neighboring texels. The material readback matched the roughness and
proximity weights, and the red distortion map shifted a dark sample
onto a white region at the requested half strength. Validation covered
invalid depth thresholds, missing depth, output formats, and aliases.
Xcode 27.2 passed 53 focused tests on Mac and 52 on iPhone 18 Pro
Simulator, which skipped the existing native MSAA depth-resolve case.
The Xcode bridge timed out resolving the package destination, so
verification used `xcodebuild`.

Automatic material readback verified two filtered views of the same
mirror in one command buffer. The direct view kept the green marker and
the second bounce kept the red marker. The direct result matched its
GPU-written blur texture, and authored roughness zero restored the sharp
pixels. A separate tilted capture verified that an authored RGB normal
map moved the reflected sample onto the bright side of an edge using
pre-clip projected depth. Both checks ran with forward and reversed
depth. Xcode 27.2 passed 46 focused tests on Mac and 45 on the iPhone
18 Pro Simulator, which skipped the existing native MSAA depth-resolve
case. Xcode MCP timed out resolving the package destination, so
verification used `xcodebuild`.
