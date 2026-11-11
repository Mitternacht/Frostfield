# Soft particles

`SoftParticleDepth` adds an optional scene-intersection fade to `ShaderParticleSystem` billboards and both `ParticleEmitter` modes. Store the opaque scene's depth and copy it to a separate, tracked, shader-readable Depth32Float texture before drawing particles. Use the same full-size single-sample view and depth mapping for the snapshot and the particle pass. Metal orders the tracked copy and draw when encoded on the same queue. The renderer validates the snapshot's size, format and distinct storage from active attachments.

```swift
// The opaque pass used depth storeAction .store.
try TextureTransfer.encodeCopy(from: sceneDepth, to: opaqueDepthSnapshot, commandBuffer: commandBuffer)
let softness = try SoftParticleDepth(texture: opaqueDepthSnapshot, distance: 1, contrast: 2)
try particles.encodeDraw(
    pass: particlePass, camera: perspectiveCamera, cameraTransform: cameraTransform,
    commandBuffer: commandBuffer, softParticles: softness
)
```

The shader reconstructs particle and scene view Z, measures their positive separation in view units, and applies r187's symmetric contrast curve. The result scales both premultiplied radiance and alpha. The defaults are a one-unit fade distance and a contrast exponent of two. A contrast of one gives a linear fade. Clear depth leaves normal opacity, and the existing live depth test continues to occlude particles. Forward and reversed depth use the corresponding Metal projection. `ParticleEmitter` accepts a `ShaderGraphCamera` for soft draws, including orthographic projection. `ShaderParticleSystem` uses its existing perspective camera entry point. [Three.js r187 SoftParticles](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/utils/SoftParticles.js)

## Graph-controlled soft particles

Use `softParticleOpacity` in a raster surface graph for a transparent mesh
billboard or other custom particle surface. Opacity, fade distance and contrast
accept graph expressions. The helper samples the full attachment by default,
and an explicit `viewportDepth` expression can choose a different source.

```swift
let particleSurface = try SurfaceShaderGraph { graph in
    .init(opacity: graph.softParticleOpacity(
        opacity: graph.uniform("Opacity", Float(0.8)),
        distance: graph.uniform("Soft distance", Float(0.5))
    ))
}
let material = try PhysicallyBasedMaterial(
    alphaMode: .blend, surfaceGraph: particleSurface, shadingModel: .unlit
)
scene.viewportSnapshot = try SceneViewportSnapshot(depth: opaqueDepth, source: .opaquePass)
```

Supply a snapshot with the same camera projection, depth mode and viewport
depth range as the particle draw. The graph reconstructs the particle's view
depth from the raster fragment and applies the same contrast curve as the
native emitters. Perspective and orthographic views support forward and
reversed depth. Logarithmic graph passes decode their stored perspective
depth and keep orthographic depth linear. Standard depth coefficients are
required. Clear samples and invalid live distance or contrast values leave
opacity unchanged.

Archives preserve the graph. Attach a fresh snapshot after loading. Custom
mesh billboards use the ordinary surface pipeline.
[Viewport capture](surface-viewport-snapshot.md)

## Points and sprites

Set `PointsMaterial.softParticleFade` or `SpriteMaterial.softParticleFade`
to a `SoftParticleDepth.Fade` and choose alpha or additive blending. The
native point or sprite quad samples `SceneGraph.viewportSnapshot` depth
in attachment coordinates. The fade multiplies texture and material
opacity before alpha testing. Depth-tested particles still use the live
depth attachment for ordinary occlusion. Both forward and reversed
projection depth are supported.

An automatic opaque snapshot uses a single-sample pass. A multisampled
point or sprite pass can instead sample a matching, separately captured
single-sample depth texture. Scene archives version 20 preserve point
and sprite fade settings. Restore a depth snapshot for the live view
after loading.

## Verification

- A real rasterized sloped plane and a scene-depth copy drive an intersecting billboard. The Metal readback matches the linear and quadratic opacity curves at six distances in both depth modes and checks full occlusion and unchanged clear-depth pixels.
- The fountain and HDR additive flow fixtures match the same fade against a rendered plane using explicit orthographic cameras with both depth mappings.
- Validation rejects a separate texture view of active depth storage, mismatched dimensions, unreadable depth and a projection with the wrong depth mapping.
- Xcode 27.2 ran 17 focused tests in four suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. The Xcode MCP workspace call timed out resolving the package destination, so the runs used headless `xcodebuild`.
- The graph fixture compares six sloped-depth pixels to the analytic fade in
  forward, reversed and logarithmic depth, including a rebased origin at
  10⁹ world units. A cropped orthographic viewport verifies attachment UVs
  and its custom depth range. Live settings change opacity, distance and
  contrast without rebuilding the renderer. Archive version 19 restores an
  explicit depth expression. The fixture rejects oblique projections and
  path tracing.
- With Metal API Validation, all 39 focused tests passed on Mac and iPad Pro
  (M5) Simulator before the final precision fixture. The final three suites
  passed all 12 tests on both destinations after that fixture was added.
- Points and sprites matched six analytic intersection samples in forward
  and reversed depth with texture alpha and authored cutoff. The tests
  cover alpha and additive blending, unchanged clear-depth pixels, a real
  4× MSAA color resolve from separately captured depth, and archive
  round trips. All 24 focused tests in five suites passed on Mac and iPad
  Pro (M5) Simulator with Metal API Validation using headless Xcode.
