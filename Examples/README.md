# Frostfield examples

Open [FrostfieldLab.xcodeproj](FrostfieldLab/FrostfieldLab.xcodeproj) in Xcode 27
with the Metal Toolchain installed. Use the `FrostfieldLab` scheme and a native
destination running OS 27 or later.

| Destination | Window host |
| --- | --- |
| **My Mac** | SwiftUI window with an AppKit Metal layer |
| iPhone or iPad | UIKit window with a Metal layer |
| Apple Vision Pro or its simulator | SwiftUI window with a UIKit Metal layer and an opt-in immersive scene |

For a local Mac build signed ad hoc, run this from the repository root.

```sh
frostfield_build_dir="$(mktemp -d)"
xcodebuild -project Examples/FrostfieldLab/FrostfieldLab.xcodeproj \
  -scheme FrostfieldLab -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$frostfield_build_dir" \
  CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY=- build &&
open "$frostfield_build_dir/Build/Products/Debug/FrostfieldLab.app"
```

For an iPhone, iPad, or Apple Vision Pro, select a local development team in
Signing & Capabilities.

All three hosts link the local Frostfield package and render the same triangle.
On visionOS the lab presents the triangle in a standard window. To exercise
startup failures, add either launch argument to the scheme's Run action.

- `--simulate-no-metal` shows the unavailable device state.
- `--simulate-bad-shader` requests a missing vertex function and shows the shader error.

The triangle draws on demand by default. Use `--continuous` to exercise the draw
loop while a window is active. Add `--trace-render-schedule` to log each window's
drawable size, activity changes, and submission count to standard error.

Add these arguments in the scheme's Run settings or pass them after `open -n … --args`.

| Argument | Fixture |
| --- | --- |
| `--mesh` | Indexed cube under a perspective camera |
| `--custom-material --continuous` | Rotating cube with a caller MSL fragment and changing color |
| `--lit --particles --grade --continuous` | Lambert lighting, a seeded GPU fountain, and a warm color pass |
| `--grade-bypass` | Offscreen render with the color grade set to zero |
| `--swiftui-mesh` | Reusable SwiftUI host with a native Rotate Cube button |
| `--swiftui-graph` | Reusable SwiftUI scene graph host with a native Rotate Scene button |
| `--scene-graph` | Three nested cubes. Click or tap a cube to highlight it, or select the background to clear. |
| `--asset-scene` | Textured OBJ octahedra with looping scene keyframes |
| `--gltf-scene` | GLB node hierarchy, embedded texture, metallic-roughness factors, and looping scene keyframes |
| `--pbr-scene` | Textured metallic-roughness shading with a directional light |
| `--shadow-scene` | Three cubes casting directional shadows onto a plane |
| `--pbr-shadow-scene` | Procedural metallic-roughness spheres that cast and receive directional shadows |
| `--immersive-scene` | visionOS launcher with an Open Immersive Preview button |
| `--cyber-ocean` | Animated glTF dolphin, surface flow particles, procedural neon tunnel, bloom, chromatic aberration, and grain |
| `--pbr-studio` | Blender-exported antique camera with HDR lighting, soft shadows, depth of field, and neutral tone mapping |
| `--scale-lab` | Procedural material samples and a field of up to 100,000 objects, with raster and path-traced views |

In a visionOS Simulator Debug build, pair `--immersive-scene` with
`--immersive-autostart` to open the immersive fixture during a smoke check.

For your own app, import `Frostfield` to prepare a `MeshRenderer`, then update
its `MeshScene` as the camera, transform, or material parameters change.
Import `FrostfieldSwiftUI` to present it with
`FrostfieldSceneView(renderer:scene:)`. [SwiftUI example source](FrostfieldLab/FrostfieldLab/LabSwiftUIScene.swift)

For multiple meshes, group `SceneNode` values under a `SceneGraph`. Pass the
same prepared `MeshRenderer` to each node that shares geometry and material.
Use `graph.encode(pass:commandBuffer:)` for drawing and
`graph.pick(at:aspectRatio:)` for selection. [Scene graph fixture](FrostfieldLab/FrostfieldLab/LabGraphFixture.swift)

To display a graph in SwiftUI, retain a command queue and pass it to
`FrostfieldSceneView(commandQueue:graph:)`. Build any directional shadow map
on the same queue. Set `drawableResolution(_:)` to choose a fixed pixel
ratio, and use `onDrawableSizeChange(_:)` to observe completed dimensions.
SwiftUI frame modifiers still choose the view's size in points.
[Drawable resolution evidence](../docs/evidence/drawable-resolution.md)

Use `OBJMeshLoader` or `GLTFMeshLoader` to load geometry before preparing a
renderer. Load an image with `TextureLoader(device:)` and pass the texture to
`MeshMaterial.textured` or `PhysicallyBasedMaterial`. Apply a `SceneAnimation`
to the base graph at the current playback time before encoding the result.
Use `GLTFMeshLoader.loadScene(data:includeMaterials:)` for static nodes and
opt-in material metadata. Include `includeAnimations: true` and
`includeSkins: true` for animated meshes. `PreparedGLTFScene` prepares
renderers once and produces graph snapshots at a requested time. The material
path supports PNG/JPEG PBR maps, samplers, and alpha modes on the primary UV
and secondary UV sets. [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)

The Cyber Ocean scene uses drag to orbit, scroll or pinch to zoom, and a native
Pause/Play control. A Mac lab executable can render a 1200 × 800 PNG without a
window using `--capture-cyber-ocean OUTPUT.png [FRAMES]`.
[Render evidence and asset credit](../docs/evidence/cyber-ocean.md)

Launch with `--procedural-camera` to orbit a Swift-built rangefinder. Add
`--studio-detail` for a lens close-up. On the Mac, use
`--capture-procedural-camera OUTPUT.png [YAW_RADIANS]` for a still.
[Source, sample code, and render evidence](../docs/evidence/procedural-camera.md)

Launch with `--path-traced-camera` to render the rangefinder with progressive
path tracing. A camera move restarts sampling. Add `--path-light-room` to
see a matte object and a mirror under a rectangular ceiling emitter. On Mac,
use `--capture-path-traced-camera OUTPUT.png [SAMPLES_PER_PIXEL] [YAW_RADIANS]`.
[Path tracing source and evidence](../docs/evidence/path-tracing.md)

Launch with `--path-light-controls` to relight a path-traced camera with a
generated metal finish. Select an emitter, then adjust power, size, angle,
or tint. Exposure changes the display without clearing the accumulated
light. Add `--shader-finish` to a procedural-camera capture to see the same
material graph in raster PBR. [Studio controls and capture commands](../docs/evidence/lights-and-shaders.md)

Launch with `--iceberg` to orbit a procedural iceberg in open water. On the
Mac, use `--capture-iceberg OUTPUT.png [SAMPLES_PER_PIXEL] [YAW_RADIANS]` for
a 1920 × 1152 still. [Iceberg render and source](../docs/evidence/iceberg-ocean.md)

Launch with `--porsche` to orbit a procedural 911 GT3 in a photography studio.
On the Mac, use `--capture-porsche OUTPUT.png [SAMPLES_PER_PIXEL] [YAW_RADIANS]`
for a 2560 × 1536 still. Add `--path-preview` for a 1280 × 768 proof.
[Car render and source](../docs/evidence/porsche-911.md)

Add `--studio-detail` for a lens close-up, `--studio-sky` to display the
environment background, and `--studio-light-rig` to load the glTF spot and
point lights. `--studio-no-ao` bypasses screen-space contact occlusion.
On the Mac, use `--studio-asset=/path/to/model.glb` to preview your own local
asset in the same studio. Drag to turn the subject. Scroll or pinch to zoom.
[Studio evidence and asset credits](../docs/evidence/pbr-studio.md)

For Blender assets, export glTF 2.0 as GLB with normals and tangents, and
enable Punctual Lights when the scene uses Sun, Point, or Spot lights. Load
with `includeMaterials: true` and `includeLights: true`. Load the scene's
HDR environment separately as Radiance RGBE or a linear floating-point
panorama. Use baked maps for appearance authored with Blender procedural
nodes. [Blender glTF exporter documentation](https://github.com/KhronosGroup/glTF-Blender-IO/blob/main/docs/blender_docs/scene_gltf2.rst)

Enable Cameras in Blender and pass `includeCameras: true` to the loader
to keep authored shots. `PreparedGLTFScene.cameraView(nodeIndex:in:viewportAspectRatio:)`
resolves the selected camera node in the current graph, including animated
ancestors. Pass its clip matrix, camera position, optional orthographic
direction, and depth mode to the explicit-view `SceneGraph.encode` overload.
Fit the viewport to the returned aspect ratio when it differs from the target.
[glTF cameras](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#cameras)

If a prepared mesh reports `requiresTransmissionBackdrop`, render the graph
into tracked linear RGBA16Float color with a store-capable depth target and
pass a separate same-sized texture with a full mip chain as
`transmissionBackdrop:`. Use the HDR output for post-processing or tone
mapping. The studio fixture uses this path for imported glass. The direct
SwiftUI drawable adapter uses its SDR path.
[Transmission extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_transmission)

The app and package share a Swift package access identifier so the fixture can
call the renderer without making it a public API. Use a normal checkout directory
name such as `Frostfield`, without a trailing `.git`. See the
[triangle evidence](../docs/evidence/triangle-fixture.md) for checked destinations.

See the [lifecycle results](../docs/evidence/render-lifecycle.md),
[windowed metrics recipe](../docs/evidence/render-metrics.md),
[Mac offscreen profile](../docs/evidence/offscreen-profile.md), and
[remaining tasks](../tasks/todo.md).
