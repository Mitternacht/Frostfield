# Coordinate conventions

Status: implemented for FF-06 (`Transform`, `Camera`, `PerspectiveProjection`,
`TriangleRenderer.drawFrame(transform:camera:target:)`). Every Frostfield type
that places geometry uses these rules. Tests that check them are named below.

## Spaces and handedness

| Space | Axes | Notes |
| --- | --- | --- |
| Model and world | Right-handed: +x right, +y up, +z toward the viewer | One world unit has no fixed physical size. |
| View (camera) | Right-handed; the camera looks along **-z** with +y up | A point in front of the camera has negative view-space z. |
| Clip (Metal) | After the perspective divide: x and y in [-1, 1] with **+y up**; depth in **[0, 1]** | Metal's normalized device coordinates. The viewport then maps +y up to the top of the target. |

A positive rotation angle turns counter-clockwise when you look from the
positive end of the rotation axis toward the origin. A quarter turn about +z
takes +x to +y; about +x it takes +y to +z; about +y it takes +z to +x.

## Matrices

- Matrices are `simd_float4x4`, stored **column-major**, and multiply **column
  vectors** on the right: `p' = M * p`.
- In a product the rightmost matrix applies first:
  `clip = projection * view * model * position`.
- `Transform.matrix` is `T * R * S`: scale along the model's own axes, then
  rotate about the model's origin, then translate. The translation is the last
  column. Rotations use the normalized quaternion.
- `Camera.viewMatrix` is the inverse of the camera's rigid placement
  (`orientation`, then `position`). `Camera` has no scale.
- The shared record `FFTransformUniforms` (in `FrostfieldShaderTypes.h`) holds
  `modelViewProjection` at byte offset 0. Swift's `columns.N` and MSL's
  `float4x4[N]` refer to the same 16 bytes at offset 16 × N.

## Projection and depth

`PerspectiveProjection(verticalFieldOfView:near:far:)` builds a symmetric
right-handed perspective projection for Metal:

```
yScale = 1 / tan(verticalFieldOfView / 2)
xScale = yScale / aspectRatio          // aspectRatio = target width / height in pixels
zScale = far / (near - far)

columns: (xScale, 0, 0, 0), (0, yScale, 0, 0), (0, 0, zScale, -1), (0, 0, zScale * near, 0)
```

Clip-space w is the distance in front of the camera. After the divide, a point
at distance `near` has depth 0 and a point at distance `far` has depth 1.
Geometry nearer than `near` or farther than `far` is clipped (Metal's default
depth-clip mode). The field of view is vertical; a wider target sees more
horizontally rather than stretching.

## Validation

The projection rejects values that would produce a degenerate or inverted
frustum before any drawing, with a `ProjectionError` that names the value and
the fix:

| Input | Requirement | Error |
| --- | --- | --- |
| `verticalFieldOfView` | Radians, greater than 0 and less than π. A value between π and 180 adds a hint that it looks like degrees. | `invalidFieldOfView` |
| `near` | Finite and greater than 0 | `invalidNearPlane` |
| `far` | Finite and greater than `near` | `invalidFarPlane` |
| `aspectRatio` | Finite and greater than 0, checked by `matrix(aspectRatio:)` and `viewProjectionMatrix(aspectRatio:)` | `invalidAspectRatio` |

The renderer computes the aspect ratio from the target's pixel size and draws
nothing for an empty target, so it never builds a projection from an invalid
ratio.

## Winding

The diagnostic triangle's corners (red apex, green lower left, blue lower right)
run counter-clockwise seen from +z. `drawFrame(transform:camera:target:)` does
not cull faces, so the triangle is visible from both sides. Front-face winding
and culling for closed meshes are defined with the indexed geometry in FF-07.

## Evidence

Package tests on My Mac (Apple M5 GPU) with Metal API validation, scheme
`Frostfield (Frostfield Workspace)`:

- `TransformTests`, `PerspectiveProjectionTests`, `CameraTests`: composition
  order, right-handed rotations, near → 0 and far → 1, frustum edges for three
  aspect ratios, the view matrix as the inverse placement, and every validation
  error.
- `TransformUniformsLayoutTests` and `TransformUniformsRoundTripTests`: host
  size, alignment, offset, and stride of `FFTransformUniforms`, and a GPU kernel,
  compiled at test time from the shared header, that copies each matrix element
  back by MSL `[column][row]` indexing. Distinctive values (10 × column + row +
  0.5) come back unchanged, and MSL reports size 64 and alignment 16.
- `TransformRenderingTests`: twelve deterministic poses rendered off screen and
  read back at corner positions computed by hand, geometry just inside and just
  outside the near and far planes, and resource counters that stay constant over
  60 camera and transform updates.
