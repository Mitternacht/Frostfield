# Primitives, vertex colors and material states

This page covers the Three.js primitive geometries, material groups, vertex
colors, the `Material` base state (blending, depth, color writes, polygon
offset, sides, alpha hashing, alpha to coverage, render order) and the non-PBR
Three.js material models. "Three.js" means r187 unless stated otherwise.

## Primitive geometry

Every generator returns ordinary indexed `MeshGeometry` with analytic normals,
top-left UVs and a tangent along increasing U whose handedness points
`cross(normal, tangent) × w` along increasing V. Vertex order, positions,
normals, triangle order and corner order follow the Three.js geometry of the
same name; V is one minus the Three.js V. Two differences are deliberate:

- Zero-area triangles that Three.js emits are left out: the capsule's
  2 × radialSegments pole triangles (and, at zero height, its
  2 × radialSegments × heightSegments middle band), and a ring's center slivers
  at zero inner radius. Vertex arrays keep the Three.js counts, so every
  remaining index has the Three.js value.
- Capsule pole vertices take the U of the wedge they close. Three.js offsets
  them half a wedge the other way, so each pole triangle's apex sits one wedge
  off.

| Capability | API | Three.js | Verified by |
| --- | --- | --- | --- |
| Segmented box with one material group per face | `MeshGeometry.box(width:height:depth:widthSegments:heightSegments:depthSegments:)` | `BoxGeometry` | `PrimitiveGeometryTests.box`, `.threeParity` |
| Cylinder and truncated cone, open ends, theta range, groups 0 side, 1 top, 2 bottom | `MeshGeometry.cylinder(...)`, `MeshGeometry.cone(...)` | `CylinderGeometry`, `ConeGeometry` | `PrimitiveGeometryTests.cylinder`, `.threeParity` |
| Capsule with arc-length V, seam at −X | `MeshGeometry.capsule(radius:height:capSegments:radialSegments:heightSegments:)` | `CapsuleGeometry` | `PrimitiveGeometryTests.capsule`, `.threeParity` |
| Torus with arc and tube sweep | `MeshGeometry.torus(radius:tube:radialSegments:tubularSegments:arc:thetaStart:thetaLength:)` | `TorusGeometry` | `PrimitiveGeometryTests.torus`, `.threeParity` |
| (p, q) torus knot with an analytic-derivative frame and an exactly closed seam | `MeshGeometry.torusKnot(radius:tube:tubularSegments:radialSegments:p:q:)` | `TorusKnotGeometry` | `PrimitiveGeometryTests.torusKnot` |
| Ring, disk and sectors | `MeshGeometry.ring(...)`, `MeshGeometry.circle(...)` | `RingGeometry`, `CircleGeometry` | `PrimitiveGeometryTests.flatShapes`, `.threeParity` |
| Sphere with phi and theta ranges | `MeshGeometry.sphere(radius:widthSegments:heightSegments:phiStart:phiLength:thetaStart:thetaLength:)` | `SphereGeometry` | `PrimitiveGeometryTests.sphere` |
| Polyhedra with detail, flat at detail 0 and smooth above, seam-corrected UVs | `MeshGeometry.polyhedron(vertices:faces:radius:detail:)`, `.tetrahedron`, `.octahedron`, `.icosahedron`, `.dodecahedron` | `PolyhedronGeometry` family | `PrimitiveGeometryTests.polyhedra` |
| Material groups and multi-material nodes | `MeshGeometryGroup`, `MeshGeometry.withGroups(_:)`, `MeshGeometry.subset(_:)`, `SceneNode.grouped(_:materials:device:colorPixelFormat:...)` | `BufferGeometry.groups` with a material array | `PrimitiveGeometryTests.groups`, `.groupedRendering` |

`PrimitiveGeometryTests.threeParity` compares the box, cylinder (three
variants, including a cone tip and an open, bottomless one), cone, capsule
(three variants, including zero height), torus (with arc and tube sweep) and
ring (with and without an inner radius) against line-by-line Swift ports of
the r187 generators in `Tests/FrostfieldTests/ThreeGeometryReference.swift`:
same vertex count, positions and normals within 1e-5, U and flipped V within
1e-6, groups equal, and indices equal to the reference's once its zero-area
triangles are removed. The sphere and polyhedra are checked against closed-form
values only. The other tests check vertex and triangle counts against the
Three.js formulas, the capsule's count less its dropped triangles included;
welded edges for closed shapes (each edge used twice, in opposite directions);
signed divergence-theorem volumes, which are positive only for outward winding;
and each vertex tangent against the UV gradient of its triangles. Measured on
an M4 Max with macOS 27.2:

- Sphere area against 4πr²: −0.80% at 32 × 16 segments and −0.20% at 64 × 32
  (the test bound is 0.5%).
- Icosahedron at detail 12 (3,380 triangles): −0.18% against 4πr².
- Torus at 64 × 256: −0.046% against 4π²Rr. The outer half of its tube
  (`thetaStart` −π/2, `thetaLength` π): −0.019% against
  2πr(πR + 2r). Capsule at 32 cap and 256 radial segments: −0.019% (volume −0.034% against 4πr³/3 + πr²h)
  against 4πr² + 2πrh.
- Frustum, cone and cylinder areas and volumes within 0.1% at 512 segments;
  ring, sector and circle areas equal the polygon formulas within 1e-5.
- Platonic solids at detail 0 match closed-form surface areas within 1e-5 of
  relative error.
- The (2, 3) torus knot's last ring equals its first ring bit for bit, and its
  volume is within 0.5% of π r² L times the 32-gon factor, with the curve length
  L integrated independently.
- Generation in a Debug test build: a 1,046,528-triangle sphere in 1.09 s and a
  262,144-triangle knot in 0.27 s.

## Vertex colors and alpha maps

| Capability | API | Three.js | Verified by |
| --- | --- | --- | --- |
| Per-vertex linear RGBA | `MeshVertex.color`, `MeshGeometry.hasVertexColors` | `vertexColors`, `color` attribute | `VertexColorTests.rasterInterpolation` |
| glTF `COLOR_0` import: float, normalized unsigned byte and short, VEC3 or VEC4 | `GLTFMeshLoader` | `GLTFLoader` | `VertexColorTests.gltfImport`, `.gltfSceneRoundTrip` |
| glTF `COLOR_0` export as float VEC4 | `GLTFExporter` | `GLTFExporter` | `VertexColorTests.gltfSceneRoundTrip` |
| Path-traced vertex colors | `PathTracingScene` | none in core | `VertexColorTests.pathTracerMatchesRaster`, `.pathTracerClampsAndAlpha` |
| PBR alpha map (green channel) with its own UV set and transform | `PhysicallyBasedMaterial(alphaTexture:)` | `alphaMap` | `VertexColorTests.pbrAlphaMap` |

Physically based and classic materials multiply vertex colors into the base
color and alpha, as glTF requires. A mesh with colors draws through the
specialized vertex stage, which reads them from vertex buffer 10. The path
tracer packs each vertex color as four halves into the unused w lanes of the
triangle record, so triangles stay 160 bytes.

The tracer applies the color the way raster PBR does:
`saturate(parameters × instance tint × factor × vertex color) × texture`. For a
colored mesh it keeps the unclamped tint product and saturates it per vertex
color in the shader; meshes without colors keep the earlier packing exactly.
`VertexColorTests.pathTracerClampsAndAlpha` checks colors above one, material
parameters above one and a base color texture, raster against tracer within
0.002. With the default `.materialAlpha` mode, raster draws vertex alpha below
one as translucency, which the tracer cannot show, so `PathTracingScene` throws
`.unsupported(_, "material alpha coverage")` unless every vertex reaches alpha
one after the tint.

Inside a vertex-colored square, the 16-sample unlit trace differs from raster by
at most 0.00073. Lit with a point light, both renderers stay within 2% of the
closed-form diffuse plus GGX radiance for the colored albedo.

## Material state

| Capability | API | Three.js | Verified by |
| --- | --- | --- | --- |
| Normal, additive, subtractive, multiply and custom blending with separate alpha equations and a constant color | `MeshRenderState.blending`, `MeshRenderState.BlendFunction` | `blending`, `blendSrc`, `blendDst`, `blendEquation`, `blendColor` | `RenderStateTests.blendModes` |
| Depth test, depth writes and compare function, mirrored for reversed depth | `depthTest`, `depthWrite`, `depthCompare` | `depthTest`, `depthWrite`, `depthFunc` | `RenderStateTests.depthAndColorWrites` |
| Per-channel color write mask | `colorWriteMask` | `colorWrite` | `RenderStateTests.depthAndColorWrites` |
| Polygon offset (factor, units, clamp) in forward and reversed depth | `depthOffset` | `polygonOffset`, `polygonOffsetFactor`, `polygonOffsetUnits` | `RenderStateTests.polygonOffset`, `.polygonOffsetMagnitudes` |
| Front, back and double sides | `side` | `side` | `RenderStateTests.sides` |
| Transparent bucket sorted back to front | `transparent` | `transparent` | `RenderStateTests.transparentSorting` |
| Render order for opaque and transparent draws, and group order from mesh-less nodes | `SceneNode.renderOrder`, `ResolvedSceneNode.groupOrder` | `Object3D.renderOrder`, `Group` render order | `RenderStateTests.opaqueRenderOrder`, `.transparentSorting`, `.groupRenderOrder` |
| Hashed alpha test | `alphaHash` | `alphaHash` | `RenderStateTests.alphaHash` |
| MSAA renderers and alpha to coverage, with Three.js's alpha-test ramp | `MeshRenderer(sampleCount:)`, `alphaToCoverage` | `antialias`, `alphaToCoverage` with `alphaTest` | `RenderStateTests.multisampling`, `.alphaToCoverage`, `.alphaToCoverageCutoff` |
| Path tracer honors or rejects the state | `PathTracingScene` | none in core | `RenderStateTests.pathTracerRenderState` |

Each blend mode renders a quad of color (0.9, 0.4, 0.1) at alpha 0.7 over a
cleared (0.2, 0.5, 0.8, 0.6) and matches the premultiplied equation within
0.004 in RGBA16Float.

Depth follows `WebGLRenderer`: with `depthTest` false nothing writes depth, as
when WebGL disables `GL_DEPTH_TEST`. `depthCompare: .always` with the test on
keeps writing depth, which is what `WebGPURenderer` does for `depthTest` false.
The default compare stays `.less`; Three.js defaults to `LessEqualDepth`.

`polygonOffsetMagnitudes` reads the depth attachment back. On a quad turned 40°
about Y, `factor` 4 moves depth by 4 × the measured window-space slope within
2%; on a camera-facing quad, `units` 1,000 moves it by 1,000 × 2^(e − 23) within
1%, where e is the exponent of the quad's depth, and `factor` alone moves it not
at all. A clamp of a quarter of the sloped bias limits the shift to the clamp
within 1%. Each holds with the opposite sign under reversed depth. Measured:
forward depth, slope 8.757e-4 per pixel and a factor-4 shift of 3.5029e-3
(ratio 0.99998); depth 0.9686, so 2^(e − 23) = 5.96e-8, and a units-1,000 shift
of exactly 5.9605e-5; clamp 8.757e-4 gives 8.757e-4. Reversed depth: the same
slope and shift (ratio 1.0000005), depth 0.0314 with 2^(e − 23) = 3.73e-9 and a
units-1,000 shift of exactly −3.7253e-6. Metal's units term therefore matches
the Direct3D and Vulkan definition for 32-bit float depth on the M4 Max.

The hashed alpha test, on a 128 × 128 target (16,384 pixels), keeps
9.91%, 29.73%, 49.61%, 70.09% and 90.22% of the pixels at alpha 0.1, 0.3, 0.5, 0.7 and 0.9 (test bound
±3 points, for classic and PBR), and the pattern is the same between frames.

With 4× MSAA on the M4 Max, alpha to coverage resolves alpha 0.25, 0.5 and 0.75
to exactly 1, 2 and 3 covered samples in every pixel of a 64 × 64 target, for
classic and PBR materials. Covered samples store alpha one (Metal's
alpha-to-one), so the resolved pixel is premultiplied color with alpha equal to
coverage; Three.js under WebGL, which has no alpha-to-one, resolves alpha ×
coverage. With an alpha cutoff (PBR `.mask` or a classic `alphaTest`), coverage
is Three.js's `smoothstep(cutoff, cutoff + fwidth(a), a)`: on a 64-pixel alpha
ramp with cutoff 0.5, columns up to 31 are empty, column 32 is partly covered
and every column from 33 is fully covered, and a constant alpha of 0.75 over a
cutoff of 0.5 covers every sample. Alpha to coverage combined with blending
throws, since blending would apply alpha a second time.

Group order: a node without a mesh or levels of detail whose `renderOrder` is
nonzero orders every mesh below it first, like a Three.js `Group`; the nearest
such ancestor wins. Unlike a Three.js `Group`, a mesh-less node with render
order 0 passes its ancestors' order through, as an `Object3D` does, so plain
transform nodes in an imported hierarchy do not reset it.

The path tracer renders surfaces as physically present. PBR `.blend` uses
stochastic opacity. The tracer throws `.unsupported` for other blending,
color write masks other than `.all`, alpha hashing, `.back` sides, and
depth tests other than `.less` or `.lessEqual`. It traces both faces for
`.double` and culls back faces for `.front`, taken from the render state rather
than the material. Depth writes, polygon offset, the transparent flag and
render order only order raster draws, and alpha to coverage converges to the
tracer's per-sample cutoff, so those are accepted.

Without a supplied state, a renderer keeps its previous behavior: physically
based `.blend` materials use `MeshRenderState.transparent`, double-sided ones
draw both sides, and everything else is opaque with `.less`. A 96 × 96 frame
of the existing PBR paths (textured, normal-mapped, masked, blended,
material-alpha, layered, shadowed, surface-graph and instanced draws, with
point and spot lights) is byte-identical to the base commit 157bb26.

## Classic materials

`ClassicMaterial` models are drawn with `MeshMaterial.classic`.

| Model | Three.js | Verified by |
| --- | --- | --- |
| `.basic` with map, alpha map, alpha test, light map, occlusion map and a reflection cube with multiply, mix or add | `MeshBasicMaterial` | `ClassicMaterialTests.basicMaps`, `.alpha` |
| `.lambert` with scene lights, ambient, emissive and shadows | `MeshLambertMaterial` | `ClassicMaterialTests.lambertAndPhong`, `.shadowMaterial` |
| `.phong(specular:shininess:)`, default specular `0x111111` decoded from sRGB | `MeshPhongMaterial` | `ClassicMaterialTests.lambertAndPhong`, `.phongDefaults` |
| `.toon(gradient:)` | `MeshToonMaterial` | `ClassicMaterialTests.toonBands` |
| `ToonShaders.directional`, `.hatching`, `.dotted`, `evaluateOnGPU` | `ToonShader1`, `ToonShaderHatching`, `ToonShaderDotted` before output color encoding | `ToonShaderTests` |
| `.matcap(_:)` | `MeshMatcapMaterial` | `ClassicMaterialTests.matcap`, `.matcapViewDirection` |
| `.normal` | `MeshNormalMaterial` | `ClassicMaterialTests.normalDepthDistance` |
| `.depth` | `MeshDepthMaterial` (`BasicDepthPacking`) | `ClassicMaterialTests.normalDepthDistance` |
| `.distance(reference:near:far:)` | `MeshDistanceMaterial` | `ClassicMaterialTests.normalDepthDistance` |
| `.shadow` | `ShadowMaterial` | `ClassicMaterialTests.shadowMaterial` |

Lighting follows the Three.js formulas, with Lambertian terms divided by π.
Phong's default specular is 0.005605 linear, which is Three.js `0x111111` after
its sRGB decode. Toon bands from a three-texel gradient change at N·L = ±1/3.
The matcap tests read the center, right, left, top and bottom texels of a 3 × 3
map for normals tilted 40°, and, at three off-center pixels of a camera-facing
quad, the texel that r187's `meshmatcap` formula gives for each pixel's own view
direction (a 256 × 256 lookup, within half a texel). Phong is checked at three
pixels against the Blinn-Phong formula evaluated with each pixel's exact view
direction. Depth and distance materials match the projection and distance
formulas within 0.002. A map on the secondary UV set needs secondary
coordinates on every vertex, and the alpha map rejects formats without a green
channel (R8, BC4, EAC R11).

## How the tests were checked

Every new test was run against deliberately broken code and failed, then passed
once the code was restored. In the first round: blending disabled, alpha to
coverage off, depth compare ignored, back-side culling swapped, polygon offset
zeroed and its reversed-depth sign removed, alpha hashing off, render order
ignored, depth sorting removed, straight-color coverage output removed, toon
coordinates clamped, the matcap V flipped, the camera view basis frozen, vertex
colors dropped in raster, path tracing and glTF import, alpha maps ignored (PBR
and classic), the light map's 1/π removed, depth inversion removed,
shadow-material alpha ignored, and the Phong normalization removed.

After review, 24 more breakages, each caught: the alpha-to-coverage-with-blending
check removed; depth writes kept with the depth test off; polygon offset factor
and units swapped, its clamp dropped and its reversed sign removed; the PBR and
classic coverage ramps removed and the ramp replaced by a hard step;
alpha-to-one off; group order not inherited and not sorted; tracer render-state
validation removed and its sides taken from the material; the tracer tint
clamped before the vertex color, its shader saturation removed and its
material-alpha check ignoring vertex alpha; the old Phong default; R8 accepted
as an alpha map; the classic secondary-UV check removed; the matcap view
direction fixed to the camera axis; the cylinder's row-major index order; the
torus ignoring `thetaStart`; the Three.js capsule pole offsets; and the box's
earlier corner rotation. Frames of the capsule seam and poles, the half-swept
torus, and a soft-alpha cutout with and without alpha to coverage were rendered
and inspected.

## Limits

- Classic materials are raster-only; `PathTracingScene` rejects them. They
  ignore rectangle lights, as Three.js does, and have no normal, bump,
  specular or displacement maps. The environment map uses reflection mapping
  only.
- The depth material writes nonlinear window depth (`BasicDepthPacking`), and
  the distance material writes an unpacked value; neither has RGBA packing.
- Alpha hashing hashes world position, so a moving object gets a new pattern.
  Shadow passes ignore hashing, alpha maps and vertex alpha; masked PBR
  materials still cut shadows by base color alpha.
- Alpha to coverage needs `sampleCount` above one and `blending` `.none`.
  Transmission still needs single-sample attachments.
- The path tracer rejects the render state it cannot represent rather than
  approximating it, and picking tests both faces of back-sided meshes.
- The unlit, textured, Lambert, shadowed Lambert and custom fragments ignore
  vertex colors. Path-traced vertex colors are half precision. Without vertex
  colors, the tracer still clamps the tint before the material factor
  (`saturate(parameters × instance) × factor`), while raster saturates the
  product; the two differ only for tints above one.
- A mesh-less node with render order 0 does not reset an outer group order, as
  a Three.js `Group` with render order 0 would.
- `SceneNode.grouped` creates one renderer per group with compacted vertices,
  so vertices shared between groups are uploaded once per group.
- Torus knots require coprime p and q from 1 to 64; Three.js accepts any values.
- glTF export writes `COLOR_0` as float VEC4 even when the source used bytes.

All tests ran with `xcodebuild` and Metal API Validation on an M4 Max Mac with
macOS 27.2. Nothing here was measured on iPhone, iPad, the iOS Simulator or
Vision Pro, and no Three.js frames were rendered for comparison; the geometry
parity is against Swift ports of the r187 generators, not Three.js itself.

## Raster scissor

Both scene color entry points and the surface-attribute prepass accept a
top-left `MTLScissorRect` in attachment pixels. A caller-supplied viewport
controls projection into the attachment, and the scissor limits raster
coverage within it. The same scissor applies to opaque, transparent and
transmissive color draws and to raster primitives. An empty rectangle
skips the camera draws, with pass load and store actions and shadow-map
work preserved.

`RenderStateTests` checks every pixel of a scissored viewport containing a
mesh and point cloud, rejects an overflowing rectangle before encoding,
and verifies an empty rectangle still clears the attachment. The 27
focused render-state, primitive and transmission tests passed on macOS
with Metal API Validation.
