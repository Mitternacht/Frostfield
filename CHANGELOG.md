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
  invalidation, slot and drawable backpressure without waiting, and drawable size
  from local bounds and display scale (FF-03).
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

No stable public API or release is available yet. Physical-device rendering has not
been verified.
