# Scene occlusion queries

`SceneGraph.encodeWithOcclusionQueries` renders the usual color pass with
Metal visibility tests for the requested node IDs. It accepts the graph
camera or an explicit raster view. Draw order, depth mode, transparency,
viewport and scissor follow the color renderer. A batch is queried as one
node across all its draw runs. Results describe the draws in this encode
and do not change which nodes render.
[Three.js r187 renderer](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js) ·
[Metal visibility result mode](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/setvisibilityresultmode(_:offset:))

```swift
let frame = try graph.encodeWithOcclusionQueries(
    for: [targetID], pass: pass, commandBuffer: command
)
command.commit()
command.waitUntilCompleted()
let visibility = try frame.results()[targetID]
```

`results()` throws while the command buffer is pending or when execution
fails. A requested ID is visible when any of its draws has a passing
fragment, occluded when it has an encoded draw but zero passing fragments,
or not drawn when no color draw was encoded. Hidden nodes, frustum-culled
nodes and missing IDs are examples of the third case.

Each queried drawable has a unique result offset. A node with both mesh
and primitive draws combines their results. Split transmission passes
own separate result buffers, and the frame combines visibility across
them. The caller's pass descriptor stays unmodified. Each frame accepts
up to 65,536 requested IDs and retains its buffers until completion,
including on command buffers with unretained references.
[Metal visibility result buffer](https://developer.apple.com/documentation/metal/mtlrenderpassdescriptor/visibilityresultbuffer)

`SceneOcclusionQueryTests` compares queried and ordinary raster output,
covers an occluded mesh followed by an unqueried visible draw, and reads
visibility for an indirect batch on supported hardware. Separate fixtures
exercise a node that draws in both transmission passes and independent
results from two encodes on one unretained command buffer.

All three query tests passed in the combined Xcode 27.2 macOS and iPhone
18 Pro Simulator runs at `2813a8b`, with Metal API Validation. The Mac used
the native batch bundle in the raster comparison. The simulator used direct
batch submission. The Xcode bridge timed out resolving the Swift package
and the Lab's configured external DerivedData volume was full. The runs
used `xcodebuild` with internal DerivedData.
