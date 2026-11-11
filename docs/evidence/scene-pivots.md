# Scene node pivots

Set `SceneNode.pivot` to a point in the object's local coordinates. The matrix
applies scale and rotation around that point, then applies the authored
translation. The point itself maps to `transform.translation + pivot` in the
parent's space. A nil pivot uses the local origin. The same matrix feeds child
transforms, raster draws, picking, lights and path-scene refits.
[Three.js r187 Object3D](https://github.com/mrdoob/three.js/blob/r187/src/core/Object3D.js)

Animation tracks keep their existing local translation, rotation and scale
channels. A track edits those components while the authored pivot stays on
the node. World-preserving attachment solves for the new translation with the
same pivot and retains the existing restriction against shear. The gizmo and CCD
solver use the pivot as the rotation anchor. Look-at aims from the object's
current origin, which can move around the pivot as the new rotation is applied.

Native scene archive version 8 stores node pivots. Earlier versions load with
nil pivots, and captures without pivots still use the earliest version required
for their contents. For glTF interchange, a TRS container holds translation
plus pivot, rotation and scale. Its offset child holds the mesh, light,
instances and authored child nodes. Translation animation keys include the same
offset so both keyed and interpolated poses survive export. A glTF import
retains the portable hierarchy. Use the native archive to retain an editable
pivot property. USDZ stores the pivot as a paired translation operation around
the rotation and scale, including when those operations have time samples.
[OpenUSD transform operations](https://openusd.org/release/api/class_usd_geom_xformable.html)

`ScenePivotTests` checks matrix composition and coordinate conversion, attach,
look-at, gizmo and CCD movement, old and new archive manifests, interpolated
glTF poses, ModelIO reads of USDZ key poses, and matching raster, world-ray
picking and path-traced movement.

The ten focused suites passed 75 tests on the Apple Silicon Mac and 75 on
the iPad Pro M5 simulator, with Metal API Validation enabled.
