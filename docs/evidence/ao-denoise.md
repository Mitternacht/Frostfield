# Depth-guided AO denoise

Use `PostProcessGraph.denoise(camera:depthMode:normalTextureName:)` to filter a noisy ambient-occlusion image carried in linear RGB, or another premultiplied linear frame. Use it with `PostProcessPass` or `PostProcessChain.Effect.graph`. The lower-level `ShaderGraphBuilder.denoise` returns a typed RGBA value that can feed other nodes in the same post graph.

```swift
let denoise = try PostProcessGraph.denoise(camera: graphCamera, depthMode: .reversed)
let chain = try PostProcessChain(queue: queue, effects: [.graph(denoise)])
let controls = try chain.uniforms(at: 0)
try controls?.set("DepthPhi", to: Float(0.5))
try chain.encode(source: noisyAO, depth: sceneDepth, destination: filteredAO, commandBuffer: commandBuffer)
```

Use the projection and depth mapping that produced `sceneDepth`. Omit the optional normal texture to reconstruct view-space normals using the existing discontinuity-aware depth stencil. Supplied normals are signed view-space RGB in a tracked, source-size RGBA16Float or RGBA32Float data texture, read from level zero. A zero normal marks an invalid sample. Cleared depth, invalid normals and zero source alpha leave the center unchanged and exclude neighboring taps. The filter evaluates linear unassociated RGB and restores the center's stored alpha.

The kernel uses r187's 16-tap, two-ring spiral and its luminance, tangent-plane depth and cosine-normal weighting equations. Frostfield uses a deterministic per-pixel rotation with four selectable phases and samples matched integer texel positions. It skips off-image and cleared neighbors. The radius and three weight parameters default to five. The graph preset exposes them as live uniforms along with the sample phase. Frostfield's GTAO bent-normal and visibility buffer has its own resolve and filter. [Three.js r187 DenoiseNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/DenoiseNode.js)

## Verification

- Metal readback checks a noisy two-depth AO fixture in forward and reversed depth with perspective and orthographic projections. Interior absolute error falls below 45% of the unfiltered input while the depth edge, clear pixels, masked pixel and original alpha are preserved.
- Same-depth readback verifies that signed orthogonal normals and tighter luminance weights stop cross-edge blending, that a zero normal preserves the center and excludes it from neighboring results, and that a constant source region is unchanged.
- A two-texel fixture isolates one neighbor and compares the Metal result against the pinned weighting equations. The chain fixture exercises a second deterministic phase and bypass on invalid radius.
- Validation covers graph serialization and stage use, texture size, missing normal or depth input, in-place and aliased output, and the chain's full-resolution depth requirement.
- Xcode 27.2 ran 31 focused tests in five suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation. The Xcode MCP workspace call timed out resolving the package destination, so the runs used headless `xcodebuild`.
