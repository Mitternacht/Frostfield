# Scene material overrides

Assign a `SceneMaterialOverride` to `SceneGraph.overrideMaterial` to
replace the raster material on the graph's indexed meshes. Set it to nil
to restore their authored materials. An exclusion set keeps selected node
IDs on their authored material, including nodes that share a mesh renderer
with selected draws. A copy of the graph can use the override for a single
auxiliary view.

The material and render state apply to color and surface-attribute draws.
The pass can use a different format or depth direction from the authored
pipeline. Named surface outputs require an unblended PBR override. The
override shares mesh, skin and morph buffers, uses the current draw range
and batch palette, and keeps native geometry deformations. It keeps the
authored material for picking and shadow-caster coverage. Batch color
submission uses direct runs while overridden so fragments that cannot run
inside Metal indirect command buffers still render.

Use `prepare(in:colorPixelFormat:depthMode:sampleCount:)` to compile the
standard color variant for existing meshes and every level of detail before
the first frame. New source meshes or layouts prepare when used. The render
owner serializes access to the override and the source graph. Exclude
meshes whose raw Metal vertex programs must remain active. Scene archives
and the path tracer continue to use authored materials.

Use `SceneGraph.primitiveMaterialOverride` for point, line and sprite color
draws. Each takes its native material type and can be set independently.
The override keeps vertex colors, per-point sizes, sprite records, line
distances and the live point or line draw range. The selected material
determines shape and size, depth testing and blend order. The original
renderers keep their authored state for picking. Override textures cannot
share storage with the color attachment or its resolve image.

```swift
var inspection = graph
inspection.overrideMaterial = try SceneMaterialOverride(material: .classic(ClassicMaterial(model: .normal)))
inspection.primitiveMaterialOverride = try ScenePrimitiveMaterialOverride(
    points: PointsMaterial(size: .pixels(6), color: SIMD4(1, 0.7, 0.25, 1)),
    lines: LineMaterial(width: .pixels(2)),
    sprites: SpriteMaterial(),
    excluding: [cameraMarkerID]
)
```

`ScenePrimitiveMaterialOverride.prepare(in:colorPixelFormat:sampleCount:)`
warms the current geometry variants while sharing their existing GPU buffers.

[Three.js r187 scene override](https://github.com/mrdoob/three.js/blob/r187/src/scenes/Scene.js#L100-L107) ·
[Three.js r187 material opt-out](https://github.com/mrdoob/three.js/blob/r187/src/materials/Material.js#L399-L405)

## Verification

Metal readback checked a shared mesh with a per-node exclusion, an
RGBA8Unorm reversed-depth view of a mesh prepared for forward RGBA16Float,
and the restoration of the original frame. Geometry cases checked live
batch replacement, authored graph displacement, and two concurrent
dynamic geometry revisions after clearing the override. The shadow fixture
verified that an authored blended mesh stays out of the caster set under
an opaque color override. The named PBR fixture read beauty, normal and
roughness from a mesh prepared for an ordinary color pass.

All 60 focused tests passed on Mac and the iPad Pro M5 simulator. The
Xcode workspace bridge timed out resolving the initial run destination.
The runs used Xcode 27.2 `xcodebuild` with internal DerivedData and
Metal API Validation.

The primitive override follow-up passed 60 focused tests on both
destinations. GPU readback checked live point and line ranges after
preparation, wider point and line pixels, a shared line renderer excluded
on one node, identical output in forward and reversed depth, culling at
the edge of the view, and a textured sprite drawing over opaque geometry
with unretained command references. A feedback attempt failed before
encoding, and the same command buffer rendered the authored material
after clearing the override.
