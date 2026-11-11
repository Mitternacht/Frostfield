# Mesh topology processing

`MeshGeometry.weldedWithRemap` selects the lowest-indexed matching referenced
vertex as a representative. Tolerances cover position distance, normal and
tangent angles, UVs and linear RGBA. Separate vertex data can participate in the
match. For example, an asset tool can compare each pair's skin influences
before merging and then copy the survivors using `sourceVertexIndices`.
Material groups follow surviving triangles. If all grouped triangles disappear
while ungrouped triangles remain, welding reports an error so the result cannot
silently begin drawing the ungrouped triangles.

`creasedNormalsWithRemap` uses corner-angle weighting across manifold edges
that join by position. This keeps smooth normals consistent across UV seams
while each side retains its own attributes. `splittingEdgesWithRemap` can keep
the original shading on vertices that do not split. Both keep the original
vertex slots and append any split copies with their source indices. Regenerate
tangents after changing normals if the material uses a normal map.

`tessellatedWithRemap` splits long edges at their midpoints. Adjacent triangles
subdivide the shared position even when different UV or normal data requires
separate vertex indices. The result retains triangle winding and material
groups, interpolates vertex colors, and reports each midpoint's endpoints so
asset tools can carry their own channels. The pass budget bounds work. At that
limit some edges may still exceed the target length.

`MeshWeldingTests` checks seam preservation, material ranges, topology of a
closed tessellated sphere, UVs, color and source remaps, and limits on edge
length, triangle count and floating-point precision. `estimatedGPUByteCount`
reports the packed base mesh, optional color buffer and UInt32 index size.

The comparison targets the r187 welding, crease-normal and modifier
operations. [BufferGeometryUtils](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/BufferGeometryUtils.js) ·
[EdgeSplitModifier](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/modifiers/EdgeSplitModifier.js) ·
[TessellateModifier](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/modifiers/TessellateModifier.js)
