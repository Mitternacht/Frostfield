# Graph-displaced picking acceleration

`MeshRenderer.preparePickingAcceleration(progress:)` accepts unskinned
meshes with constant, uniform-driven or clock-driven graph position
offsets. It snapshots the live inputs and active index range. It reads a matching completed raster vertex buffer if available,
or runs the same Metal deformation with the captured inputs. A pending
command never supplies vertices to another queue. Preparation awaits
Metal completion before the cancellable BVH build moves off the caller's
actor.

Scene ray queries reuse the active hierarchy while its revision matches.
A query after a uniform or relevant clock change reads the new pose and
refits the existing topology. The graph reports original mesh triangle
indices when using a restricted draw range. The renderer discards its
cached hierarchy after a geometry or range edit. An asynchronous build
that observes a superseding edit throws `pickingGeometryChanged`. A caller
holding an earlier returned hierarchy can continue to query that snapshot.

Metal textures can change contents without a new uniform revision. A
graph that reads a texture from its vertex offset still evaluates the
current texture for direct picking and rejects prewarming. A graph
combined with skinning still uses the existing palette-specific direct
picking path.

The implementation uses completed Metal buffers and a separate shared
readback buffer. [Apple Metal completion handlers](https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler%28_%3A%29) · [Metal buffer copies](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy%28from%3Asourceoffset%3Ato%3Adestinationoffset%3Asize%3A%29)

The focused regression suite passed 66 tests on macOS arm64 and 66 on the
iPhone 18 Pro simulator with Metal validation. It compared posed vertices,
normals and rendered pixels, preserved original triangle IDs through
draw-range edits, and tested unsubmitted raster work, stale async builds
after uniform or clock edits, and live texture replacement. Xcode's bridge
reported a full workspace volume, so verification used the existing
internal DerivedData with `xcodebuild`.
