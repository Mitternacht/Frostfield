# GPU-written mesh draw arguments

`MeshIndirectDraws` describes up to 512 ordered byte offsets into a Metal
buffer. Each offset selects one native 20-byte
`MTLDrawIndexedPrimitivesIndirectArguments` record. Offsets must be aligned to
four bytes and fit completely inside the buffer. Repeated offsets submit the
same arguments again. This matches the ordered-offset model in Three.js r187
`BufferGeometry.setIndirect`.
[Three.js r187 BufferGeometry](https://github.com/mrdoob/three.js/blob/r187/src/core/BufferGeometry.js) ·
[Metal indexed indirect arguments](https://developer.apple.com/documentation/metal/mtldrawindexedprimitivesindirectarguments)

Encode and end the argument producer before using the buffer in the same
Metal command buffer. Use a hazard-tracked source. Synchronize producers on
other queues before the consumer reads them. The renderer validates the
arguments with a Metal compute pass into private scratch storage, then reads
that storage with Metal's indexed indirect draw API. The source and scratch
stay alive through completion even with unretained references. The caller
keeps the render attachments alive when using unretained references.
[Metal resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)

```swift
let draws = try MeshIndirectDraws(buffer: gpuArguments, byteOffsets: [64, 4])
try renderer.encode(scene, pass: pass, commandBuffer: commandBuffer, indirectDraws: draws)
```

This entry point draws one `MeshScene` through its normal forward-depth
color pipeline. It uses the renderer's UInt32 triangle index storage. The GPU
can select any in-range index start and count and, with a prepared
`InstanceSet`, any in-range base instance and instance count. Without an
instance set, a draw can use one instance at base zero. Zero counts or
out-of-range vertex and instance indices produce empty draws.

`PrimitiveIndirectDraws` is the unindexed counterpart for one point, line or
sprite node. `SceneGraph.primitiveIndirectDraws` maps a node ID to ordered
offsets in a hazard-tracked buffer of `MTLDrawPrimitivesIndirectArguments`.
A compute kernel keeps a record only when its vertex count matches the
prepared topology (6 for a point or sprite quad, 8 for a wide-line strip,
or two vertices per hairline segment), its start and base instance are zero,
and its instance count fits the prepared range. Anything else becomes an
empty draw. The color pass submits the validated records and reports them in
`indirectArgumentDrawCount` instead of the CPU point, segment or sprite counts.

A signed base vertex shifts each index in the selected range. The first
indirect draw prepares and caches a bounds tree from the mesh's immutable
indices. A record with base vertex zero uses the existing range checks and
reads no index data. For a nonzero base vertex, the GPU combines cached
bounds for complete 64-index blocks and reads only the partial blocks at
the edges. It checks the shifted minimum and maximum in signed 64-bit
arithmetic before submitting the record. The producer's records stay on
the GPU.
[Metal signed baseVertex](https://developer.apple.com/documentation/metal/mtldrawindexedprimitivesindirectarguments/basevertex)

Check `MeshIndirectDraws.isSupported(on:)` on the active device. These records
are GPU-readable arguments consumed by individual Metal draws. Batched mesh
bundles remain the separate CPU-encoded indirect command buffer path.
[Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

## Scene graph passes

Pass a dictionary of node IDs and arguments to a graph encode. Each entry
must target an ordinary mesh in that graph. Mesh batches and level-of-detail
nodes use their own submission paths. An instance set requires a renderer
prepared for instancing. One graph encode accepts up to 16,384 records.

```swift
let arguments = [node.id: draws]
let statistics = try graph.encode(pass: pass, commandBuffer: commandBuffer, indirectDraws: arguments)
```

Color and shadow passes use the same validated arguments for a node.
Pass the same table to `encodeSurfaceAttributes` and
`SelectionOutline.encodeSelection` when those passes should use the same
geometry. The normal node transform, layer, visibility, culling and render
order still apply. Culling bounds conservatively include the authored mesh
and instance set. The argument table belongs to the encode. Picking,
collision, path tracing and exports continue using the authored scene.

`SceneDrawStatistics.indirectArgumentDrawCount` counts submitted records
across color, surface attributes and shadows. It includes records the GPU
turns into empty draws. GPU instance counts stay on the GPU, and the
existing instance count reports draws whose counts are known to the CPU.

## Verification

Xcode 27.2 ran 14 focused tests on arm64 macOS with Metal API Validation.
A compute fixture writes records to private storage and switches which
textured triangle is visible between two encodes in one unretained command
buffer. Tests also check native field offsets, invalid host layouts, malformed
GPU records, nonzero base instance, repeated offsets and blended order. The
run included the existing direct MeshRenderer and BatchedMesh ICB tests.
Xcode MCP timed out resolving the workspace's initial run destination, so
verification used `xcodebuild` with isolated DerivedData.

The iPadOS 27.2 simulator passed the record-layout and device capability
test with Metal API Validation. Its device rejected indirect submission
and rendered the regular direct draw. The three indirect execution tests
were skipped on that simulator.
[Metal Simulator capabilities](https://developer.apple.com/documentation/metal/developing-metal-apps-that-run-in-simulator)

The graph follow-up passed 17 focused macOS tests with Metal API Validation.
It checks ownership errors, parent transforms, visibility and camera layers,
instance ranges and transparent render order. The GPU-selected triangle also
wrote the expected directional shadow, surface normal and selection depth.
The iPadOS simulator passed the two layout, ownership and capability tests
and skipped five GPU execution tests.

The signed base-vertex follow-up passed 19 focused macOS tests with Metal API
Validation. The selected ranges shifted to different triangles in graph and
standalone draws. Out-of-range shifts in either partial block or in a full
block drew nothing, as did shifts at the signed integer extremes. The
iPadOS simulator passed the two layout and ownership/capability tests and
skipped seven GPU execution tests.
