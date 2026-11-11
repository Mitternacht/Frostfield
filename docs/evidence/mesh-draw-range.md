# Indexed mesh draw ranges

Set a prepared mesh's `drawRange` between encodes. All nodes using that
renderer use the same range. Start and count refer to index-buffer elements.
An omitted count uses the rest of the buffer. The renderer clamps the range
to the available indices and submits complete triangles from its first index.
An empty range leaves the pass's load and store actions in effect.

```swift
mesh.drawRange = try MeshDrawRange(start: 6, count: 12)
// Later, restore all of the prepared indices.
mesh.drawRange = .all
```

Color, raster shadows, named and packed attributes, and selection depth use
the same draw range. Existing buffers are reused. Picking reports the
triangle index relative to the original buffer, and collision uses the
selected triangles. Scene archives store nondefault ranges in version 9.
Native multi-material meshes use one prepared renderer per material group,
so each range addresses the indices of that prepared renderer. GPU indirect
arguments supply their own range and require the renderer's default range.

Path scene preparation captures the range. Use `updateGeometry(for:in:)`
after changing a nonempty range. Prepare a new path scene when a mesh starts
or stops contributing triangles. A separately supplied path replacement
defines its own complete geometry. Use native scene archives to preserve
the editable range alongside its full source mesh.

[Three.js r187 BufferGeometry](https://github.com/mrdoob/three.js/blob/r187/src/core/BufferGeometry.js) · [Three.js r187 mesh raycasting](https://github.com/mrdoob/three.js/blob/r187/src/objects/Mesh.js) · [Metal indexed draw](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/drawindexedprimitives(type:indexcount:indextype:indexbuffer:indexbufferoffset:instancecount:))

`MeshDrawRangeTests` compares raster and software-path pixels before and
after an edit, reads back directional shadow depth, and checks picking,
collision, archive round trips, shifted index starts and empty ranges.
The nine focused suites passed 56 tests on the M4 Max Mac and 49 on iPhone
18 Pro Simulator with Metal API Validation. The simulator skipped seven
indexed indirect tests. Xcode used `xcodebuild` after the bridge timed out
resolving the package run destination.
## Point and line draw ranges

`GeometryDrawRange` is the shared window type. `MeshDrawRange` remains a
source-compatible name for indexed meshes. `PointsRenderer.drawRange`
selects source vertices without copying point or color buffers.
`LineRenderer.setDrawRange(_:)` rebuilds only the segment buffer. A strip
gets caps at the new ends, a loop closes across the selected vertices,
and independent pairs start at the first selected vertex. Existing
distance-based dash patterns retain their phase at the range start.

Both picking paths follow the visible range. Point hits keep their source
index. Line hits index `LineRenderer.drawnSegments`, whose endpoints
reference source positions. Conservative bounds still cover the original
geometry. Native archive version 11 preserves nondefault primitive ranges
and the original data.

The nine focused suites passed 62 tests on both Apple Silicon Mac and
iPhone 18 Pro Simulator with Metal API Validation. The tests read
point and line pixels, check both pickers, and exercise a range edit
while an unretained command buffer still holds an earlier draw. Native
archive recapture is byte-identical and the older version rejects the
new range fields.
[Three.js BufferGeometry](https://github.com/mrdoob/three.js/blob/r187/src/core/BufferGeometry.js) · [Point picking](https://github.com/mrdoob/three.js/blob/r187/src/objects/Points.js) · [Line picking](https://github.com/mrdoob/three.js/blob/r187/src/objects/Line.js)

## Material groups

Pass a source-index `drawRange` to `SceneNode.grouped` or keep a
`GroupedMesh` to edit its range after placing its node in a graph. The
range intersects each material group's original index window before
assembling complete triangles. This preserves group-local triangle starts
when the range begins between two indices of an earlier group. Changes
reuse each group's prepared renderer. The scene archive stores each
child's effective range.

All 52 focused tests passed on Apple Silicon Mac and iPhone 18 Pro
Simulator with Metal API Validation. The grouped fixture compares color
pixels from both renderers, checks picking before and after archive
restore, and proves that an unaligned global window assembles a full
triangle in the next group. The Xcode bridge timed out resolving the
package destination, so verification used `xcodebuild`.
[Three.js WebGL group draw range](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js)
