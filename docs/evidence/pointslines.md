# Points, lines, sprites and debug helpers

Scene nodes can draw point clouds, lines and sprites alongside meshes. The scene
graph draws them in the same color pass as meshes. It supports frustum culling,
forward or reversed depth, group order, render order and back-to-front blending.
Points, lines and sprites can use multisampled color and depth attachments when
their prepared sample counts match. Edge, wireframe and debug helper generators
produce ordinary line geometry.

## Capabilities

| Capability | Three.js | Frostfield API | Raster path | Verified by |
| --- | --- | --- | --- | --- |
| Point clouds with per-point color and size, square or disc shape, map and alpha test | `Points`, `PointsMaterial` (`size`, `sizeAttenuation`, `map`, `alphaTest`, `vertexColors`) | `PointCloud`, `PointsMaterial`, `PointsRenderer`, `ScenePrimitive.points` | One instanced draw of camera-facing quads, so sizes have no point-size limit; world sizes scale with the projection's focal length and 1/depth | `PointsAndSpritesTests` |
| One-pixel lines, strips, loops and separate segments | `Line`, `LineLoop`, `LineSegments`, `LineBasicMaterial` | `LineGeometry(positions:colors:topology:)`, `LineMaterial.Width.hairline` | Metal `.line` primitives | `LineRenderingTests`, `LineShapeTests` |
| Dashes measured in accumulated world length | `LineDashedMaterial`, `computeLineDistances()` | `LineMaterial.Dash`, `LineSegment.startDistance`, `LineSegment.endDistance` | Fragment pattern on a perspective-correct line distance | `LineRenderingTests`, `LineShapeTests` |
| Wide lines in pixels or world units, with miter, round and bevel joins, butt, round and square caps, and dashes | `Line2`, `LineSegments2`, `LineMaterial` (`linewidth`, `worldUnits`, `dashed`) | `LineMaterial.Width.pixels`, `.world`, `LineMaterial.Join`, `LineMaterial.Cap`, `LineRenderer` | A strip of three quads per segment evaluates the stroke in the fragment. Adjacent translucent segments composite to the union of the stroke at their join. | `LineRenderingTests`, `LineShapeTests` |
| Sprites with rotation, anchor, atlas region, world or pixel size | `Sprite`, `SpriteMaterial` (`map`, `rotation`, `sizeAttenuation`), `Sprite.center` | `Sprite`, `SpriteMaterial`, `SpriteRenderer`, `ScenePrimitive.sprites` | One instanced draw of quads parallel to the image plane | `PointsAndSpritesTests` |
| Scene culling, opaque and blended ordering, overlays, tint and statistics | render lists, `Object3D.frustumCulled`, `Object3D.renderOrder`, `Material.depthTest` | `SceneNode(primitive:)`, `SceneNode.primitive`, `PrimitiveBlending`, `SceneDrawStatistics.drawnPrimitiveCount`, `.culledPrimitiveCount` | Primitives share mesh color ordering. Primitives with depth testing disabled draw last in node order. | `PrimitiveSceneGraphTests` |
| Multisample attachments and color resolve | `RenderTarget.samples`, `resolveColorBuffer` | `PointsRenderer.sampleCount`, `LineRenderer.sampleCount`, `SpriteRenderer.sampleCount` | Prepare a supported sample count and supply matching color and depth attachments. Resolve color before the single-sample effects chain. | `RenderStateTests` |
| Picking points, segments and sprites | `Raycaster.params.Points`, `Raycaster.params.Line`, `LineSegments2.raycast` | `SceneGraph.pickPrimitive(at:drawableSize:tolerance:)`, `PrimitiveHit`, `PrimitivePickingTolerance` | CPU scan in screen space against the drawn size or width plus a pixel or world tolerance | `PrimitivePickingTests` |
| Crease and boundary edges, wireframes | `EdgesGeometry`, `WireframeGeometry` | `LineGeometry.edges(of:thresholdAngle:color:)`, `LineGeometry.wireframe(of:color:)` | Edges use 10⁻⁴ position welding. Wireframes deduplicate exact endpoint positions and omit zero-length edges. | `EdgesAndHelpersTests` |
| Helpers | `AxesHelper`, `GridHelper`, `PolarGridHelper`, `Box3Helper`, `BoxHelper`, `ArrowHelper`, `CameraHelper`, `DirectionalLightHelper`, `SpotLightHelper`, `PointLightHelper`, `RectAreaLightHelper`, `PlaneHelper`, `SkeletonHelper` | `LineGeometry.axes`, `.grid`, `.polarGrid`, `.box`, `.boundingBox(of:in:color:)`, `.cameraFrustum`, `.light(_:size:)`, `.plane`, `.skeleton(in:joints:)`, `DebugHelpers.node`, `.arrow`, `.plane` | Line geometry with the Three.js colors converted to linear; the arrow head and plane fill are unlit meshes | `EdgesAndHelpersTests` |

## Measured against closed-form values

All values come from the tests above, rendered into RGBA16Float targets with a
90° camera, so one world unit at unit depth spans half the target height.

- A 4-pixel line centered on row 40.3 of a 128-pixel target covers 4 ± 0.02
  pixels in each of 34 interior columns, with its coverage centroid within 0.02
  pixels of 40.3. Total coverage equals length × width within 0.5 pixels².
- A 3-pixel line rising half a pixel per pixel covers 3 / cos θ within 0.05 per
  column, so width is measured perpendicular to the line.
- 0.25-unit world-width lines at depths 2 and 8 are 8 and 2 pixels wide within 0.02.
- On a line receding from depth 2 to 10, dash transitions cross half coverage
  within 0.35 pixels of the projected world-distance boundaries at depths 3, 4, 5
  and 6. Interpolating the distance linearly on screen would put the first
  boundary about 15 pixels away. Vertex colors at depth 6 match the world-space
  parameter within 0.01 where the screen parameter differs by 0.33.
- At a right-angle join of an 8-pixel stroke, the miter covers the corner pixel at
  (+3.5, +3.5) from the vertex; bevel and round joins leave it empty. A round
  join's pixel at (+2.5, +2.5) has the analytic 96% coverage. With a 50%-alpha
  stroke every join pixel is 0.5 ± 0.01 and no pixel in the image exceeds 0.5, so
  the two segments never shade the same pixel. Three.js fat lines overlap their
  round ends at joins instead.
- A 20° interior angle exceeds the default miter limit of 4 and draws as a bevel;
  with a limit of 10 the tip covers more than 20 pixels past the vertex.
- A segment crossing the near plane, in either direction, renders identically in
  forward and reversed depth, with no pixels outside its visible half.
- Unit-wide square points at depths 4 and 8 cover exactly 64 and 16 pixels;
  6-pixel points cover 36 at both depths. A 16-pixel antialiased disc's total
  alpha is within 1% of π × 8². A material size of 2 pixels whose graph size is
  6, then 4, then 0 covers 36, 16, and 0 pixels on the same renderer. macOS
  passed `PointsAndSpritesTests` (13 tests, 0.446 seconds). The iPad Pro (M5)
  simulator passed that suite and `LineRenderingTests` (25 tests, 3.347 seconds).
  Both runs had Metal API Validation on. A material width of 2 pixels whose graph
  width is 4, then 2, covers 4 and 2 pixels in an interior column. A hairline
  with a graph width of 8 stays one row. macOS passed `LineRenderingTests`
  (13 tests, 0.443 seconds). The iPad simulator passed that suite and
  `PointsAndSpritesTests` (26 tests, 4.162 seconds), again with Metal API Validation.
- Under a rotated and translated camera, a 2.5 × 1.25 unit sprite at distance 5
  covers exactly the 16 × 8 pixel rectangle about its projection; a quarter turn
  covers 8 × 16, and a top-left anchor hangs it right and down.
- In an orthographic view, unit points cover 8 × 8 pixels at depths 4 and 40.
- `LineGeometry.edges(of: .cube())` returns 12 unit edges at 1° and none above
  90°. A 30° fold keeps its shared edge at 29.4° and drops it at 30.6°. The cube's
  wireframe has 18 edges. Wireframe tests keep all three edges of a triangle
  narrower than 10⁻⁴ and preserve distinct edges only 10⁻⁵ apart.
- Picking returns the exact point, segment or sprite index. A 2-pixel line with
  1 pixel of tolerance is hit 1.9 pixels from its centerline and missed at 2.1.

On the original `d8a0c67` revision, forty-one deliberate breakages each failed
at least one test, and every test failed under at least one. The recovered
review tests also cover dash offsets, caps, short joins, viewport offsets,
sprite alpha testing and transmission backdrop capture.

## Integrated checks

The Xcode 27.2 (27B52) bridge opened and edited the package. Its build and test
actions could not write to DerivedData on the full external SSD, so validation
used `xcodebuild` with internal DerivedData and Metal API Validation.

- iPad Pro 13-inch (M5) simulator, iOS 27.2. All 69 focused tests passed,
  including exact half-alpha joins and 4× point, line and sprite pipelines.
- M4 Max Mac, macOS 27.2 (26B123). The full suite passed with 765 tests
  reported in 150 suites. The macOS FrostfieldLab build also passed.
- In the Mac Debug stress fixture, the recovered wide-line shader took
  100.20–102.60 ms to draw 100,000 long dashed segments at 1920 × 1080.
  Trimming the single-sample stroke bounds and skipping dash calculations
  for uncovered fragments brought the same fixture to 83.80–84.42 ms in
  the full and focused runs. The existing 100 ms test threshold stayed in
  place.

## Earlier isolated-lane performance

`PrimitivePerformanceTests` takes the median GPU time of seven frames after one
warm-up, into a 1920 × 1080 RGBA16Float target with depth, on an M4 Max Mac with
Metal API Validation on. These ranges come from the original lane at `d8a0c67`,
before the recovered wide-line changes. They cover three runs (two for the
random walk).

| Workload | GPU time |
| --- | --- |
| 1,000,000 points, 2-pixel opaque squares | 3.0–3.5 ms |
| 1,000,000 points, 0.05-unit antialiased discs | 4.0–4.4 ms |
| 100,000-segment random-walk strip, 3-pixel dashed miter, about 3.7 pixels per segment | 0.67–0.69 ms |
| The same material on 100,000 random segments averaging 697 pixels | 25.3–27.1 ms |
| The same 100,000 random segments as hairlines | 2.0 ms |
| 100,000 rotated, blended sprites | 0.41–0.42 ms |

The random-segment case shades about 3 × 10⁸ quad pixels and shows the cost of
very long wide lines. The test's 100 ms bound is only a regression guard.

## Limits

- Primitives are unlit, do not cast or receive shadows, and are ignored by the
  path tracer and by glTF import and export (no glTF `POINTS` or `LINES` modes).
- Renderers are immutable. Changing positions, colors or materials needs a new
  renderer; only the node's tint and transform change per frame. A uniform on a
  point, sprite, or line color graph, including a point's graph size or an
  analytic line's graph width, edits without a new renderer. Hairlines ignore
  a graph width and stay one pixel wide.
- Blending sorts whole nodes. Points and sprites within one renderer draw in
  array order.
- Pixel-sized points, sprites and lines cull by their anchor points, as Three.js
  does, so a shape whose anchor leaves the view can disappear while its edge
  would still be visible. Set `cullingBounds` to avoid that.
- Hairlines use Metal line rasterization without analytic antialiasing, joins
  or caps. Opaque analytic stroke and disc edges use hard cutoffs. The
  primitive material APIs have no alpha-to-coverage control.
- Transmission and post-processing require single-sample attachments.
  The prepared `DebugHelpers` nodes use single-sample pipelines.
- World-unit widths taper linearly on screen along each segment, so their
  distance field is approximate where a segment is strongly foreshortened.
  Stroke coverage is combined for neighboring segments at a join. Separate
  or nonadjacent strokes can still overlap. A segment pointing straight at
  the camera draws only its caps.
- Dashes are measured in world line distance only; there are no screen-space dash units.
- Picking returns the nearest primitive and does not test occlusion by meshes.
- The helpers are static geometry. `BoxHelper` and `SkeletonHelper` results must be
  rebuilt when their targets move; the camera and light helpers can follow their
  targets as child nodes. There is no `HemisphereLightHelper`, because Frostfield
  has no hemisphere light. The point light helper draws three circles rather than
  a wireframe sphere, and the directional helper has no target line.

The original lane passed 39 correctness tests on an iPhone 17 simulator with
iOS 27.2 and built for the visionOS 27.2 simulator. Physical iPhone, iPad and
Vision Pro qualification and matched Three.js comparison remain open.
