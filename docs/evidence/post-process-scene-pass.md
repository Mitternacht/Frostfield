# Scene capture for post-processing

`PostProcessScenePass` captures linear RGBA16Float color and Depth32Float
in one draw. Each pass can select its own `SceneLayers` and mesh or
primitive material overrides. The source graph keeps its camera layers
and materials. Omitting a pass setting inherits the source graph's value.

```swift
let capture = PostProcessScenePass(queue: queue)
capture.layers = SceneLayers(layer: 2)
capture.overrideMaterial = try SceneMaterialOverride(material: .unlit)
try capture.setResolutionScale(0.5)

let frame = try chain.encode(
    scene: graph, using: capture, destination: output, commandBuffer: command
)
```

The pass uses the output aspect ratio and rounds each scaled dimension to
at least one texel. The chain starts with matching color and depth at the
capture size and resamples the processed color into the output. A standalone
capture exposes both textures and its split camera for custom post graphs.
The returned projection includes any jitter in the captured frame. Use
separate pass instances to preserve multiple simultaneous captures. A pass
reuses its attachments when the next capture has the same dimensions.

Metal readback checks layer filtering, both material override types, a scaled
65 × 41 scene with its unscaled camera aspect, depth sampling in forward and
reversed modes, and two successive captures in a single command buffer.
An independent 1 × 1 capture checks explicit camera jitter and sRGB output.
All 18 focused tests passed on macOS and iPad Pro (M5) Simulator with Metal API
Validation. The Xcode MCP timed out resolving the initial package destination.
Verification used Xcode 27.2 `xcodebuild` with internal DerivedData.

[Three.js r187 scene pass source](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/PassNode.js)
