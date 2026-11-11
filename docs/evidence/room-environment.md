# Room environment lighting

`RoomEnvironment.prepare(queue:)` captures a reusable HDR room at the
world origin. Assign the resulting `EnvironmentLighting` to a scene
graph to light raster PBR or prepare the path tracer's environment.

```swift
let room = try RoomEnvironment.prepare(queue: queue)
graph.environment = room
let lighting = try PathTracer.Lighting(queue: queue, graph: graph, backgroundVisible: false)
```

The fixture uses the Three.js r187 room dimensions, inward-facing shell,
white boxes, point light and six thin emissive boxes. The emitters use
linear white radiance values of 50, 50, 17, 43, 20 and 100. The fixture
uses Frostfield's native PBR shading and point-light attenuation.
It captures the six faces through `CubeCamera` and prepares the diffuse
and GGX maps through `EnvironmentLighting`. The default captures 256
texels per face with near and far distances of 0.1 and 100.
[Three.js r187 RoomEnvironment](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/environments/RoomEnvironment.js) ·
[Three.js r187 PMREMGenerator](https://github.com/mrdoob/three.js/blob/r187/src/extras/PMREMGenerator.js)

Use `RoomEnvironment.makeScene(device:)` to inspect or edit the source
geometry and lighting before a custom capture. The convenience `prepare`
waits for capture and filtering. Call it during setup and reuse the
prepared environment across frames.

The native fixtures sample all six panel radiances in the default
256-texel cube, check reflected floor light, and use the same room
environment to light a mirror in raster PBR and the path tracer. Each
mirror produces an HDR pixel above 20 and goes dark at zero environment
intensity. All 13 focused tests passed on macOS and iPhone 18 Pro
Simulator with Metal API Validation enabled.

The pinned source file's SHA-256 is
`55f466192cc84298755a424c5e040345006b2ee1455589b3b54126c2ea4123f4`.
[Three.js MIT notice](../../LICENSES/Three-RoomEnvironment.txt)
