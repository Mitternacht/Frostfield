# Finite-radius path spotlights

`PathTracer.AnalyticLight(..., spotRadius:)` accepts a radius in world units for an authored spot light. Zero uses the original apex sample. A positive radius samples a uniform disk perpendicular to the light's travel direction. The disk center moves along that direction by `radius / max(tan(outerAngle), 1e-5)`. An initializer taking a resolved scene light retains its authored local frame, including roll. A radius change invalidates the accumulation history. [Pinned PhysicalSpotLight](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/objects/PhysicalSpotLight.js), [aperture sampling](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/nodes/lights.wgsl.js).

Distance falloff, the range window, the cone, IES and cookie projection use the sampled direction from the aperture to the receiver. Shadow rays end at the sampled point, so a nearer blocker can soften a shadow while geometry beyond the disk cannot block it. The native analytic light keeps the existing squared cosine cone and multiplies it by IES and cookie transmission. The reference chooses IES in place of its cone attenuation. [Pinned light dispatch](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/webgpu/LightsInfoNode.js).

Finite spots remain analytic samples and camera or reflection rays pass through their aperture. The pinned implementation also limits ray intersections to its planar rectangle and disk area emitters. Those reference area emitters intersect only from the front. Frostfield's separately authored planar panels retain their native opaque dark back. Raster and the scene light rig continue to use the authored spot apex.

## Verification

The pinned runtime fixture checks default radius, copy semantics, packed radius and disk area, and the direction and offset of the aperture. The native tests compare on-axis lighting with the analytic disk average, verify a partial shadow and a blocker beyond the sampling plane, and integrate the authored cone, IES and colored cookie independently on a rotated light. They also exercise parameter validation, invisibility and lighting-history reset.

The zero-radius Mac readbacks matched the baseline byte for byte for plain, colored-cookie, quadrant IES and asymmetric IES scenes. The comparison covered 9,793 pixels.

The Apple silicon Mac and the iPad Pro M5 simulator passed 34 focused tests each across nine suites with Metal validation requested. The Mac aperture and shadow fixture passed with both software and Metal ray traversal. The Xcode bridge could not create its arena on the full external volume, so the native command-line runner used internal DerivedData.
