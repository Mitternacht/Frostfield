# Materials and the shader binding contract

Status: implemented for FF-08 (`Material`, `UnlitParameters`, `MaterialError`,
`MeshRenderer.makeUnlitMaterial(color:)`,
`MeshRenderer.makeMaterial(fragmentFunction:library:parameters:)`, and
`MeshRenderer.drawFrame(mesh:material:transform:camera:target:)`). The API is
experimental: there is no stable release.

A material is a **fragment function** that shades what Frostfield's mesh vertex
stage rasterizes, plus a **parameter block** the app updates. Frostfield ships two
built-in fragment functions, the normal diagnostic (`ff_mesh_normal_fragment`, used by
`drawFrame(mesh:transform:camera:target:)`) and the unlit material
(`ff_unlit_fragment`). An app supplies its own function from any `MTLLibrary` made by
the renderer's device, such as its `makeDefaultLibrary(bundle: .main)` or a library
compiled from source. Every material, built-in or app-supplied, is drawn the same way.
There is no material graph.

## Using a material

```swift
// Once, before interaction: each call checks the function and prepares
// (or reuses) its pipeline.
var glow = try renderer.makeMaterial(
    fragmentFunction: "my_glow_fragment",
    library: try device.makeDefaultLibrary(bundle: .main),
    parameters: MyGlowParameters(color: [1, 0.5, 0.2, 1], time: 0, padding: (0, 0, 0))
)
var flat = try renderer.makeUnlitMaterial(color: [0.2, 0.6, 0.9, 1])

// Every frame: change the values and draw. Nothing is compiled or allocated.
glow.parameters.time = Float(animationTime)
_ = renderer.drawFrame(mesh: mesh, material: glow, transform: transform, camera: camera) { target }
```

`Material<Parameters>` is a value. Its `parameters` are copied into the frame's
commands when the frame is encoded, so changing them after `drawFrame` returns affects
only later frames, never a frame the GPU is still running.

## The fragment function contract

### Inputs

The function may read these varyings through its `[[stage_in]]` struct. Metal matches
a fragment input to the vertex output by its `[[user(...)]]` name and type, **not by
the member name**, so declare your own struct with any member names and only the
members you need:

```metal
struct MyVaryings {
    float4 position [[position]];                  // window position and depth
    float3 normal [[user(ff_model_normal)]];       // model-space normal, interpolated, not normalized
    float3 modelPosition [[user(ff_model_position)]]; // model-space position, interpolated
};
```

Frostfield's own header declares the same outputs as `FFMeshVaryings`; an app does not
need that header. A function without `[[stage_in]]` is also valid (the unlit material
has none).

### Parameter block

The function may declare **one** read-only parameter block at fragment buffer index 0:

```metal
fragment float4 my_glow_fragment(MyVaryings in [[stage_in]],
                                 constant MyGlowParameters &parameters [[buffer(0)]])
```

- Its size must equal `MemoryLayout<Parameters>.size` of the Swift type passed to
  `makeMaterial`, and be **smaller than 4096 bytes** (Frostfield copies it with
  `setFragmentBytes`, which Metal documents for data smaller than 4 KB).
- An empty Swift type (size 0) means the function declares no parameter block.
- Declare the record once in a C header that both Swift (through a bridging header or
  a C target) and the shader include, with fixed-width scalars, SIMD types, and
  explicit padding. Frostfield checks the size; it cannot check field offsets. A
  Swift `struct { SIMD4<Float>; Float }` is 20 bytes while the matching MSL struct is
  32, which the size check reports.

The lab's `LabShaderTypes.h` and `LabShaders.metal`
([`Examples/FrostfieldLab/FrostfieldLab`](../Examples/FrostfieldLab/FrostfieldLab))
are a complete example of this pattern.

### Nothing else

Frostfield binds nothing else to the fragment stage: no textures, samplers, other
buffers, or writable buffers. A function that declares any of them is rejected, even if
it does not use them.

### Output

The function returns the color for color attachment 0. There is **no blending and no
color conversion**: the returned value, alpha included, is written as-is. With a
non-sRGB format such as `bgra8Unorm` (the lab's `MTKView` format), the stored values are
the returned values; with an `_srgb` format Metal encodes on write. The return type
must suit the target's color format (`float4` or `half4` for unorm formats).

## Rendering and lifetime rules shared by every material

| Rule | Value |
| --- | --- |
| Vertex stage | `ff_mesh_vertex`: `projection * view * model` from the frame slot's uniform buffer |
| Winding and culling | Counter-clockwise front faces, back faces culled |
| Depth | `less`, writes on, a depth attachment of the renderer's `depthPixelFormat` is required |
| Frames in flight | The renderer's frame slots; a busy slot skips the frame instead of waiting |
| Parameters | Copied into the frame's commands at encode time |
| Pipelines | Created when a material is made, never while drawing |

Pipelines are cached per renderer by **library object and function name**: making a
second material from the same function in the same `MTLLibrary` object reuses the
pipeline. A new library object, even one compiled from the same source, is a new key.
`MeshRenderer.resourceStatistics.renderPipelinesCreated` counts every pipeline Metal
created, including one created and then rejected by the binding check.

A material can only be drawn by the renderer that made it; drawing it with another
renderer stops on a precondition before anything is encoded.

## Errors

`makeMaterial` and `makeUnlitMaterial` throw `MaterialError` before any frame uses the
function. Each description names the function or value and the fix.

| Case | Cause |
| --- | --- |
| `libraryDeviceMismatch` | The library was made by another device. |
| `functionMissing` | No function with that name; the error lists the library's fragment functions. |
| `notAFragmentFunction` | The function is a vertex, kernel, or other function. Checked before Metal sees it: Metal API validation aborts on a non-fragment function. |
| `pipelineCreationFailed` | Metal's own pipeline error, for example a `[[stage_in]]` member the vertex stage does not write (`user(ff_uv)`), or an output type the color format cannot store. The description includes Metal's message and the available varyings. |
| `unsupportedBinding` | A texture, sampler, extra buffer, or writable buffer argument. |
| `parameterSizeMismatch` | The shader's parameter block and the Swift type differ in size. |
| `parametersNotDeclared` | Non-empty Swift parameters, but the function declares no parameter block. |
| `parametersMissing` | The function declares a parameter block, but the Swift parameters are empty. |
| `parametersTooLarge` | The Swift parameters are 4096 bytes or larger. |

## Evidence

Package tests (`MaterialTests.swift`) on My Mac (Apple M5 GPU) with Metal API
validation, scheme `Frostfield (Frostfield Workspace)`:

- Every material (built-in normal diagnostic, unlit, and two app functions compiled at
  test time from the declaration above) covers exactly the same pixels in six cube
  poses, and an app copy of the normal diagnostic is byte-identical to the built-in
  one.
- An app function's `[[user(ff_model_position)]]` matches a CPU ray-cast entry point
  within 0.012 model units on every unambiguous pixel of three poses.
- Parameters, including alpha, reach the shader unchanged; two frames in flight keep
  the parameters each was encoded with.
- 120 frames that change the unlit color and an app material's parameters, alternating
  materials, show each frame's values and leave every resource counter unchanged
  (3 pipelines).
- Each `MaterialError` case is produced by a real function and leaves the renderer
  usable.

The lab's material fixture shows the same on the iOS Simulator. These are Mac GPU tests
and simulator runs, not physical-device evidence.
