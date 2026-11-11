# Editable arrow helper

`DebugHelpers.addArrow(..., to:parent:)` attaches a shaft and cone. Set
the attachment's direction, origin, length, color or head dimensions, then
call `update(in:)` to apply those values together. Values use the parent's
coordinates. The direction is normalized internally. A nil head length
uses 20% of the current arrow length. A nil base diameter uses 20% of the
resolved head length.

The attachment keeps the root, shaft and cone node identities. Direction
and length changes reuse both renderers. Color changes update the cone tint
and replace the shaft's colored line geometry. A head that fills the full
length hides the shaft. Keep both children attached to the helper.

This follows the authored controls of [Three.js r187 ArrowHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/ArrowHelper.js).
Frostfield retains its 16-sided cone and requires finite positive
dimensions, with the head length at most the total length.

The focused suite passed 64 tests on macOS arm64 and 64 on an iPhone 18 Pro
simulator with Metal validation. The rendered fixture checks vertical and
horizontal placement, shaft and cone color readback, exact tip position,
buffer reuse, stable node identities and transactional validation.
