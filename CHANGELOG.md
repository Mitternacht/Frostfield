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

No stable public API or release is available yet. Physical-device rendering has not
been verified.
