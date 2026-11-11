# Axis view helper

`AxisViewHelper.Layout` projects six signed world axes from the camera's local-to-world rotation into a top-left pixel square. The default square is 128 pixels. The caller supplies its origin so a native host can place it within any pane. The layout exposes marker positions and resolves circular hits in the same back-to-front order used by the overlay. The three positive stems use the r187 axis colors, with dimmed negative endpoints. [Three.js r187 ViewHelper](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/helpers/ViewHelper.js).

Prepare one `AxisViewHelper` per device and target format. Encode after the scene color pass, or after tone mapping when drawing directly in display colors. The helper alpha blends into an existing linear HDR or sRGB render target and leaves scene depth alone. It takes a caller-owned command buffer, and it uses no auxiliary frame textures or persistent per-frame uploads.

```swift
let layout = try AxisViewHelper.Layout(
    cameraRotation: orbit.cameraTransform.rotation,
    origin: SIMD2(color.width - 128, color.height - 128),
    markerRadius: 9
)
// Prepare labels once and keep them with the helper.
let labels: [AxisViewHelper.Axis: AxisViewHelper.Label] = [
    .positiveX: try .init("X", device: helper.device),
    .positiveY: try .init("Y", device: helper.device),
    .positiveZ: try .init("Z", device: helper.device)
]
try helper.encode(layout: layout, labels: labels, destination: color, commandBuffer: commandBuffer)

// Handle an input in the same top-left pixel coordinates.
if let axis = layout.hitTest(at: pointerPixel) {
    try orbit.orient(to: axis, transition: .constantSpeed())
}
```

Labels are optional on each signed endpoint. Supply a Core Text font or use the system font, and choose a linear color for the ink. The renderer scales a prepared glyph mask with the overlay and fits it into the marker. Labels and discs composite together, so an obscured rear marker's text stays behind the front marker. Marker radius controls both the displayed discs and the hit area.

Axis selection retains the presented orbit center, distance and orthographic scale. It replaces pending pan, zoom and rotation input and honors the controller's angular limits. The original `orient(to:animated:)` uses the controller's damping or snaps immediately. `.constantSpeed()` advances at 2π radians per second by default, with a custom positive rate available. It rotates along the orbit's no-roll yaw and pitch path. A compound turn follows that path with duration `hypot(yawDelta, pitchDelta) / rate`, so its timing can differ from r187's quaternion path. Call `update(deltaTime:)` for either animation. Accepted orbit input returns to the normal controller timing. The host owns input routing and frame scheduling.

## Verification

The pinned r187 runtime fixtures verified the six marker placements for the identity view, the effect of camera roll, all six selections with a fixed orbit radius, custom positive-axis labels, and the midpoint and finish of a quarter turn at 0.125 and 0.25 seconds. At the poles, r187's look-at fallback introduces a small position offset while aligning the final orientation. Frostfield uses the orbit controller's cardinal pitch and keeps the position and orientation aligned.

The Apple silicon Mac and the iPad Pro M5 simulator each passed 49 focused tests across six suites with Metal validation requested. Native pixel checks confirm signed markers, front marker precedence, the stems, unchanged surrounding pixels, sRGB output, glyph orientation and ink color, and occlusion of a rear label. Timed tests exercise the default quarter turn, a compound turn with angular limits, split updates, and transactional recovery after failed orbit input. The Xcode bridge could not create its arena on the full external volume, so the native Xcode command-line runner used internal DerivedData.
