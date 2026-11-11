# Surface graph opacity, alpha test and masks

`SurfaceShaderGraph` exposes three optional scalar outputs on its typed
builder and node-array initializer. They operate on stock PBR coverage.

```swift
let shader = try SurfaceShaderGraph { graph in
    let pattern = graph.texture("Coverage", content: .data)
    return .init(
        opacity: pattern.sample(graph.uv0).x,
        alphaTest: graph.uniform("Cutoff", Float(0.5)),
        mask: graph.uv0.x.lessThan(graph.constant(Float(0.75)))
    )
}
```

`opacity` multiplies the sampled alpha and `alphaTest` replaces the material
cutoff. Both clamp to zero through one. A graph cutoff rejects equality.
`mask` discards when its value is zero. Nonfinite outputs leave the authored
coverage inputs in effect. Graphs that omit these outputs retain the prior
coverage rules, including the authored mask cutoff comparison.
[Three.js r187 `NodeMaterial.setupDiffuseColor`](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/NodeMaterial.js#L775)

Raster color and surface attributes use the same coverage decisions. Opaque
and masked materials write full opacity where they pass. `.materialAlpha`
writes premultiplied color and alpha without compositing through the
surface. `.blend` uses sorted raster blending and stochastic path coverage.
An alpha-to-coverage render state uses the graph cutoff for its coverage
ramp. A path-traced first-hit guide chooses the dominant layer at alpha
one half after applying graph cutouts.

The path tracer evaluates the coverage outputs during candidate intersection
tests for camera, bounce and shadow rays. Coverage uses the same sampled
surface inputs, texture resources and live uniforms as shading. Separate
coverage code avoids running the graph's other output branches during
intersection tests. The existing path graph input restrictions apply, including
screen-space inputs and screen derivatives. The path renderer still rejects the
material's standalone alpha-map binding and raster alpha hashing.

Scene archives preserve these outputs in version 7. Earlier graphs load
with the new slots unset. A scene with both coverage and spherical light
probes uses version 7 and preserves both features. CPU picking remains a
geometry query. Interchange export retains warnings for surface graph channels it
cannot represent.

## Verification

The GPU fixtures distinguish a cutoff equality, a lit visible region, and
a false mask in raster color and path first-hit albedo. They compare
written alpha and stochastic path blending with reference colors, update a
live mask on a prepared scene to move its path shadow, and round-trip a
graph texture and cutoff through a scene archive.

The five focused suites passed 34 tests each on the M4 Max Mac and the
iPad Pro 13-inch (M5) Simulator with Metal API validation. The Mac cutout
fixture exercised both Metal ray tracing and the software BVH. The simulator
used the software BVH. Xcode 27.2 MCP timed out resolving the package run
destination, so verification used `xcodebuild` with isolated DerivedData.

## Raster shadow coverage

Directional and punctual shadow maps evaluate graph opacity, alpha test,
and masks before writing depth. The shader uses the same material
textures, alpha map, UV transforms, tint, vertex data and live graph
uniforms as the color pass. It evaluates coverage outputs against the
light projection and that map's pixel coordinates. Position offsets,
skinning, morphing and instance transforms use the ordinary mesh vertex
path. Three.js r187 similarly forwards the source mask and position to
its shadow material and renders through the shadow camera.
[Three.js shadow material override](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L3458)
[Three.js shadow pass camera](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/ShadowNode.js#L602)

Depth maps apply binary cutouts. Blended and transmissive meshes follow
the existing caster policy. Coverage can use light-pass screen inputs
and derivatives. If it reads camera position, view direction, clip
distances, or camera matrices, encoding a raster shadow for that mesh
throws `MeshRenderer.Failure.graphShadowCameraInput` with the caster and
graph node. Color graph outputs can still use those inputs. The selection
depth pass and CPU picking remain separate from raster shadow coverage.

Directional, spot and point fixtures read the stored depth at the light's
projected caster position and compare the lit receiver before and after
live cutoff and mask edits. The caster moves through both a vertex graph
and an instance transform. An opacity-only fixture keeps the authored
cutoff, including equality. A separate fixture verifies the camera-input
error and draws the same graph with casting disabled. All 26 focused
tests passed on the M4 Max Mac and the iPad Pro 13-inch (M5) Simulator
with Metal API validation. Xcode 27.2 MCP timed out opening the package,
so verification used `xcodebuild` with isolated DerivedData.
