# Changelog

## Unreleased

### Added

- Initial Frostfield architecture and development plan.
- Ordered implementation tasks, acceptance criteria, and validation protocol.
- Claude implementation handoff and contribution guidance.
- Swift package baseline with a runtime GPU capability survey (FF-01).
- Experimental `TriangleRenderer`: a diagnostic triangle through a conventional Metal
  command queue with bounded frames in flight, typed construction errors, and a shader
  library packaged in the Frostfield resource bundle (FF-02).
- `FrostfieldLab`, a UIKit `MTKView` example app, and a combined Xcode workspace (FF-02).
- Experimental `RenderSchedule`: requested versus continuous drawing, per-scene
  pause while inactive, hidden, or in the background, redraw only after
  invalidation, frame-slot backpressure without waiting, and drawable size from
  local bounds and display scale (FF-03). A requested frame that finds no free slot
  or no drawable is deferred rather than waited for; with nothing in flight, a
  missing drawable is retried once, then waits for the next invalidation
  (`frameDidEnd(_:framesInFlight:)`). The UIKit sample host still acquires
  drawables synchronously on the main thread through `MTKView.currentDrawable`;
  whether that blocks on a device is unmeasured, and FF-03's drawable criterion is
  open ([Checkpoint A](docs/evidence/checkpoint-a-host.md#drawable-acquisition-on-the-main-thread)).
- FF-04 experiment comparing conventional Metal with the Metal 4 command path
  (`Examples/Experiments/Metal4Comparison`, not shipped) and the
  [Metal backend decision](docs/decisions/metal-backend.md): conventional Metal stays
  the production path.
- FF-05 experiment comparing a Satin-backed render of an indexed mesh with a
  minimal renderer (`Examples/Experiments/SatinComparison`, an isolated package, not
  shipped) and the [renderer foundation decision](docs/decisions/renderer-foundation.md):
  keep the minimal renderer.
- Experimental `Transform`, `PerspectiveProjection`, and `Camera` with documented
  [coordinate conventions](docs/coordinates.md) (right-handed, column-major column
  vectors, Metal depth [0, 1], validated projection inputs with `ProjectionError`),
  `TriangleRenderer.drawFrame(transform:camera:target:)`, the shared
  `FFTransformUniforms` record, `ResourceStatistics`, and the lab's camera fixture
  (FF-06).
- Experimental `IndexedGeometry` with creation-time validation (`GeometryError`), the
  procedural `IndexedGeometry.cube(size:)`, and `MeshRenderer`/`Mesh`: indexed meshes
  uploaded once and drawn with depth testing, back-face culling, and normal
  diagnostic coloring; the lab's cube fixture (FF-07).
- Experimental materials: `Material<Parameters>`, the built-in unlit material
  (`MeshRenderer.makeUnlitMaterial(color:)`, `UnlitParameters`), app-supplied fragment
  functions (`MeshRenderer.makeMaterial(fragmentFunction:library:parameters:)`) under a
  documented [binding contract](docs/materials.md), typed `MaterialError` diagnostics
  checked through Metal's pipeline reflection before any frame is encoded, a
  per-renderer pipeline cache, and `drawFrame(mesh:material:transform:camera:target:)`.
  Parameter updates copy values into the frame's commands and create no pipeline. The
  lab's material fixture shades the cube with the unlit material and a time-varying
  material from the app's own shader library (FF-08).
- [v0.1 core qualification record](docs/evidence/v0.1-core-qualification.md)
  (Checkpoint C preparation): a clean consumer package resolved Frostfield alone from
  git at the candidate revision and rendered with its packaged shaders on a Mac and on
  the iOS Simulator; labelled Mac characterization; every physical-device measurement
  listed as not taken.

No stable public API or release is available yet. The limited v0.1 is not qualified:
physical-device rendering, performance, and budgets have not been measured.

### Roadmap (not implemented)

- Physical-device qualification of the v0.1 core.
- Lighting, particles, post-processing, public texture output, a SwiftUI adapter, and
  GlassWeave interoperability.
