# Scene tools, archives, video, and audio

| Capability | API | Evidence |
| --- | --- | --- |
| Hierarchy editing and queries | `SceneNode.name`, `metadata`, `SceneGraph.node(named:)`, `forEachVisibleNode`, `add(contentsOf:toParent:)`, `removeNode(withID:)`, `attach(_:to:)`, `lookAt(_:worldTarget:up:forward:)`, and local/world coordinate conversion | Tests cover duplicate IDs, parent lookup, bulk insertion, mirrored transforms, and world-pose preservation. Attaching under a parent that would require shear throws and leaves the graph intact. |
| Visibility and 32 layers | `SceneNode.isVisible`, `SceneNode.layers`, `SceneGraph.cameraLayers`, and the explicit layer mask on `pick` | Rendered tests compare hidden and off-layer color, directional shadows, and punctual lights with the corresponding nodes removed. A hidden parent hides descendants. A visible parent and child may belong to different layers. |
| glTF node names and visibility | `PreparedGLTFScene`, `GLTFExporter.export(_:sceneName:nodeNames:animations:includesCamera:onlyVisible:)` | Names survive import and export. Scene graph export can omit hidden subtrees and reports the animation tracks those omissions drop. |
| Asset loading | `AssetLoader` provides URL modification, progress events, cancellation, coalescing, and a memory-budgeted cache | The tests cover concurrent requests, cancellation and retry for the same key, eviction order and overflow, URL modifiers, and texture identity per Metal device. |
| Scene archives | `SceneArchive(capturing:animations:lightRig:commandQueue:)`, `write(to:)`, `init(contentsOf:)`, `load(device:commandQueue:colorPixelFormat:)` | Recreated renderers produce identical pixels and draw counts for the fixture. Recapture produces the original manifest and sidecar bytes. A second fixture covers vertex colors, index groups, alpha maps, render state, projection jitter, and shader graph matrices and branching. Translucent layers also retain their factors and shared texture bindings. |
| Shader graph JSON | `Codable` on `MaterialGraph`, `SurfaceShaderGraph`, `PostProcessGraph`, `TextureShaderGraph`, and `BufferShaderGraph` | Decoding revalidates graph declarations and operations. Archive tests also cover graph uniforms and textures in a prepared material. |
| Video texture and frames | `VideoTexture` updates a stable Metal texture from an `AVPlayerItem`. `VideoTextureCache` wraps CoreVideo frames. | A three-frame H.264 fixture verifies texture orientation and linear sRGB samples. The same prepared textured and shader graph materials follow all three frames. Additional tests verify opaque black on every mip before the first frame and removal of the player item output when released. |
| Positional audio | `SceneAudio`, `AudioDistanceModel`, `AudioCone`, and `AudioPanning` | Offline stereo renders check equal-power gains and attenuation at several azimuths and distances, cone gain, HRTF channel response after rotating the camera, and playback rate for positional and non-positional sources. |
| Spectrum analysis | `SceneAudio.makeAnalyser`, `AudioAnalyser.floatFrequencyData()`, `frequencyData()`, and `averageFrequency()` | Tests compare Blackman-window spectral bins and exponential smoothing with their closed-form values. |

## Archive data and limits

Scene archives store a JSON manifest and a little-endian binary sidecar. Nodes are
stored in pre-order with parent indices. The depth limit is 1,024, and the tests
round-trip a 1,000-level hierarchy on a task stack. Matching byte ranges share
one sidecar view. Shared renderers, textures, environments, uniform sets, and
instance sets stay shared after load.

The archive captures point clouds, lines, sprites, supported indexed meshes,
skinned and morphed meshes,
instance sets, levels of detail, transform and glTF node animation channels,
classic and PBR materials, and supported lights. A fixture renders all nine classic shading models before and
after reload and compares every pixel and the recaptured archive. A primitive
fixture also checks identical pixels and byte-for-byte recapture for textured
points, dashed lines, and rotated sprites, with sharing between nodes.
GPU textures keep their stored format and mip chain, so some archives need
device-specific format support. Tests exercise BC1 and ASTC. On simulators
the archive rejects mipmapped ASTC because Metal can return level-zero
blocks for smaller mips. Single-level ASTC still round-trips. The loader checks
dimensions, lengths, indices, and keyframes before use. Invalid archive records
for an oversized cube array, excessive depth, or malformed animation keys
throw errors.

Version 2 adds typed animation tracks and shared live PBR materials. Scenes
that only need the earlier schema still write version 1, and the loader reads
both versions. Typed bindings reconnect to graph nodes, the graph camera and
environment, live materials, shader uniform sets and light rig IDs. glTF camera
bindings retain their source camera index for use with an external prepared glTF
scene. Custom bindings retain their names. Tests apply restored tracks through
the mixer, verify authored and current live factors independently, and compare
exact key samples for a split additive quaternion spline.

Version 3 adds IES profiles and light cookies for node lights and light rigs.
The sidecar stores the validated LM-63 source and the cookie's linear RGB
texels. Shared resources load once and stay attached when animation changes
the light. Scenes that only need versions 1 or 2 still use those versions.

Capture rejects custom Metal shader libraries and samplers with neither a
recorded descriptor nor an explicit description.
The graph JSON format stores supported custom shader function source through
`ShaderFunction`. Framebuffer-only, memoryless, and multisampled textures
cannot be captured.

## Media behavior

Video uses BGRA8 with an explicit sRGB transfer request. A wrapped frame shares
its IOSurface with Metal. `VideoTexture.update` encodes a blit into the caller's
command buffer and regenerates mips when requested. The caller supplies frame
dimensions, playback and render scheduling. HDR and wide-gamut input is
converted to the output gamut. A track's preferred transform remains the
caller's responsibility.

Spatial audio uses the graph camera as its listener. Positional sources use
`AVAudioEnvironmentNode`, including Apple's HRTF option. Stereo positional input
is averaged to mono. The API covers volume, loops, attenuation, panning, and
rate. Detune, playback offsets, filters, Doppler, and occlusion are open.

## Verification

Xcode 27.2 on an arm64 Mac with Metal API Validation. The 49 tests in the
scene editing, visibility, primitive picking, glTF export, asset loading, video,
and spatial audio suites passed. The archive and graph coding suites passed
12 focused tests. Xcode MCP opened the Lab project, and its build was
blocked by the configured DerivedData volume being full. The Swift package
workspace open timed out, so test runs used `xcodebuild` with an internal
DerivedData path.

The typed archive, existing scene archive, and animation mixer suites ran
30 focused tests on macOS. These cover staged failures for missing targets,
malformed key data and duplicate mutable rig IDs as well as clip playback.

The photometric archive and related suites passed 44 focused tests each on
macOS and the iPad Pro M5 simulator. The round-trip fixture produced identical
raster and path-traced pixels and exercised restored node and rig animation.

```sh
TEST_RUNNER_MTL_DEBUG_LAYER=1 xcodebuild test -scheme Frostfield-Package \
  -destination 'platform=macOS,arch=arm64' \
  -only-testing:FrostfieldTests/SceneNodeConveniencesTests \
  -only-testing:FrostfieldTests/SceneVisibilityRenderingTests \
  -only-testing:FrostfieldTests/SceneArchiveTests \
  -only-testing:FrostfieldTests/SceneCodingTests \
  -only-testing:FrostfieldTests/AssetLoaderTests \
  -only-testing:FrostfieldTests/VideoTextureTests \
  -only-testing:FrostfieldTests/SpatialAudioTests
```
