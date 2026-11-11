# Reference-style temporal reprojection

Initialize `TemporalAntiAliasing` with `.referenceTRAA` and supply the raster
motion attachment to `encode`. The preset uses 32 Halton jitter offsets and
the r187 motion, variance and luminance resolve. Tune
`settings.traa?.maxVelocityLength`, `useSubpixelCorrection` and
`edgeDepthDifference` for the view. Motion speed is measured in input
pixels after the raster jitter difference is removed. The minimum current
weight is 0.05, and motion speed and optional fractional-pixel correction
increase it. Three-by-three color variance limits the history, and luminance
weighting reduces flicker while retaining HDR.
[Three.js r187 TRAANode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/TRAANode.js)
[Three.js r187 TAAUtils](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/utils/TAAUtils.js)

The Frostfield mode follows the motion vector and projected prior depth of
the current pixel. Its edge policy uses the range of stored depth in the
current neighborhood. If the nearest prior depth differs, a depth edge can
recover history only when its raw previous footprint has a matching depth.
Jitter-corrected color taps also need a matching depth. This policy keeps
history when a foreground tap survives near an edge and rejects history
when the surface is missing. The reference selects the closest neighboring
velocity and can allow history across depth mismatches at edges.

Variance clipping operates in premultiplied linear RGB. Alpha remains within
the current neighborhood's coverage range, with transparent black at zero
coverage. Both forward and reversed depth use the same thresholds. Omit the
motion texture for the camera-only resolve. The standard raster motion
resolve remains the default, and switching the motion resolve mode seeds a
new history.

The GPU fixtures check the combined velocity, subpixel, variance and
luminance equations against a CPU reference. They also exercise both depth
orders, strict and relaxed edge thresholds, rejection without matching
geometry, invalid settings, and exact camera-only behavior.

All 17 focused reference TRAA, motion TAA, camera-only TAA and raster
upscaler tests passed on macOS and iPhone 18 Pro Simulator with Metal API
validation. Xcode MCP reported a full external DerivedData volume, so
verification reused the internal Xcode command-line build caches.
