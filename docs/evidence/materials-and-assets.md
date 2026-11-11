# Material extensions and compressed assets

`PhysicallyBasedMaterial` covers the glTF material extensions in raster PBR and
the path tracer, and `GLTFMeshLoader` reads compressed and extended assets.
`GLTFMeshLoader.supportedExtensions` lists every extension the loader honors and
the loading option each one needs.

## Material extensions

| Capability | API | Raster | Path tracer | Verified by |
| --- | --- | --- | --- | --- |
| Texture transforms on all 17 maps, including normal-map tangent frames and alpha-mask shadows (`KHR_texture_transform`) | `PhysicallyBasedMaterial.TextureTransform`, `TextureBinding.transform`, `baseColorTextureTransform` | ✓ | ✓ | `TextureTransformTests` |
| Emissive strength above one (`KHR_materials_emissive_strength`) | `emissiveStrength` | ✓ | ✓ | `MaterialExtensionsTests` |
| Specular factor and color (`KHR_materials_specular`) | `PhysicallyBasedMaterial.Specular` | ✓ | ✓ | `MaterialExtensionsTests` |
| Sheen with a tabulated base-layer albedo (`KHR_materials_sheen`) | `PhysicallyBasedMaterial.Sheen` | ✓ | ✓ | `MaterialExtensionsTests` |
| Anisotropic GGX (`KHR_materials_anisotropy`) | `PhysicallyBasedMaterial.Anisotropy` | ✓ | ✓ | `MaterialExtensionsTests` |
| Thin-film iridescence (`KHR_materials_iridescence`) | `PhysicallyBasedMaterial.Iridescence` | ✓ | ✓ | `MaterialExtensionsTests` |
| Dispersion (`KHR_materials_dispersion`) | `Transmission(..., dispersion:)` | ignored | ✓ | `MaterialExtensionsTests` |
| Unlit shading (`KHR_materials_unlit`) | `ShadingModel.unlit` | ✓ | ✓ | `MaterialExtensionsTests` |
| Material variants (`KHR_materials_variants`) | `GLTFScene.selectingMaterialVariant(_:)`, `selectingMaterialVariant(named:)` | — | — | `GLTFMaterialExtensionTests` |
| glTF sampler wrap and filter modes, per axis | `PhysicallyBasedMaterial.TextureSampling`, `TextureBinding.sampling`, `baseColorSampling` | ✓ | ✓ (magnification only) | `PathTextureSamplingTests`, `PathImportedMaterialTests` |
| Blended coverage | `AlphaMode.blend` | sorted blending | stochastic opacity | `PathImportedMaterialTests` |
| Material alpha | `AlphaMode.materialAlpha` | opaque, written alpha | opaque, written alpha | `PathImportedMaterialTests` |

The tests compare rendered pixels with closed-form values or with the other
renderer:

- Emission equals factor × strength exactly for strengths 0, 1 and 6.
- A specular factor of zero gives exactly zero highlight; 0.5 halves it; the
  color tints F0 per channel.
- The grazing sheen highlight matches the closed-form Charlie × Neubelt value
  within 1% in raster and 4% in the path tracer. Raster image-lit sheen matches
  the path-traced directional albedo under a uniform sky within 6%.
- Anisotropic highlights stretch along the tangent and turn with `rotation`.
- Iridescence colors agree between renderers. A 170 nm film reads blue and a
  250 nm film reads red, as interference predicts.
- Checker texels land on the predicted pixels under offset, rotation and scale.
  Rotated normal maps light within 1% of the analytic value.
- A dispersive glass ball shows blue-only fringes in the path tracer.
- Clamp-to-edge, repeat and mirrored-repeat addressing, in all nine U and V
  combinations, put the same checker texels on the same pixels in both
  renderers over UVs from -1 to 2. A magnified 2 × 2 checker keeps hard texel
  edges with nearest filtering in both renderers and blends with linear.
- A clearcoat factor map that clamps U and mirrors V, beside a repeating base
  color map, traces each pixel exactly as a uniform map of the expected texel
  does. Each of the 14 base and layer maps packs its own addressing and filter
  bits.
- A nearest map twice as wide as the path tracer's texture array averages each
  pair of one-texel columns instead of dropping one, and keeps a hard edge on
  its magnified axis.
- An imported glTF quad whose sampler clamps U and mirrors V traces with every
  core, clearcoat, specular and sheen map, and its first-hit albedo guide
  matches the raster pixels.
- A black blended cover over a white emissive backdrop converges to the
  transmitted fraction 1 − alpha within 0.015 at 64 samples per pixel; alpha
  zero is invisible and alpha one is opaque. Shadow rays through an alpha 0.5
  cover carry half the direct light within 3%. The first-hit guides show the
  cover wherever its alpha is at least one half and the backdrop otherwise, and
  three denoiser passes cut the 4-sample error by more than 60%.
- A material-alpha cover hides the backdrop and writes premultiplied color and
  alpha within 0.01 of raster, for factor and map alpha, and casts an opaque
  shadow, as the raster shadow map does.

Limits: raster ignores dispersion. Raster sheen under image lighting uses
irradiance × color × E, which is exact only for a uniform sky. Anisotropic image
lighting uses a bent normal. One material's layer maps can use at most six
distinct samplers. The path tracer still rejects clearcoat normal maps.

The path tracer samples textures it has resampled into a fixed-size array, so it
reads a `TextureSampling` description instead of the Metal sampler: per-axis
addressing and the magnification filter apply, and mip levels always blend
linearly; the minification and mip filters apply only to raster.
Nearest-magnified maps keep the nearest texel on each axis the array magnifies,
so their texel edges stay hard, and filter linearly on each axis it shrinks.
Shrinking any map to less than half its size reads only two source texels per
array texel along that axis, so the rest drop out. The resampling clamps at the
source's edges, so linear repeat seams on very small textures blend over a
narrower band than in raster. A binding with a `sampler` but no `sampling`
description is rejected, because Metal sampler states cannot be inspected; glTF
material preparation always supplies both.

Blended surfaces trace as stochastic opacity, not sorted compositing. Every
camera, bounce and shadow ray accepts a candidate hit with probability alpha
(base color factor × tint × map alpha), using a hash of the ray, instance and
triangle, so the converged image is unbiased alpha coverage and costs noise at
low sample counts. Raster shadow maps skip blended casters, while the path
tracer's shadow rays pass 1 − alpha of the light. The first-hit guides accept a
blended surface where its alpha is at least one half, so the denoiser sees one
layer per region rather than per-pixel noise.

Material-alpha surfaces follow raster: they hide what lies behind them and cast
opaque shadows, and a camera ray that first hits one scales its radiance and
coverage by the alpha, so the output alpha matches raster's written alpha. Seen
by a reflection or refraction, they are opaque. Surface graphs cannot change
traced coverage in either mode.

## Compressed and extended assets

| Capability | API | Verified by |
| --- | --- | --- |
| Quantized positions, normals, tangents, UVs and morph offsets (`KHR_mesh_quantization`) | `GLTFMeshLoader.loadScene` | `GLTFQuantizationTests` |
| Meshopt buffer views: attribute codec v0 and v1, triangle and sequence codecs, octahedral, quaternion, exponential and color filters (`EXT_meshopt_compression`, `KHR_meshopt_compression`) | `MeshoptDecoder` | `MeshoptDecoderTests` |
| KTX2 with RGBA8, RGBA16F, BC1/3/7, ETC2 and LDR ASTC levels, stored or zlib (`KHR_texture_basisu` without Basis payloads) | `TextureLoader.load(data:interpretation:mipmapped:)` | `KTX2LoaderTests` |
| WebP and AVIF through ImageIO (`EXT_texture_webp`, `EXT_texture_avif`) | `TextureLoader`, `GLTFMeshLoader` | `TextureLoaderTests`, `GLTFExtensionPolicyTests` |
| GPU instancing transforms (`EXT_mesh_gpu_instancing`) | `loadScene(..., includeGPUInstancing: true)`, `GLTFScene.Node.instances` | `GLTFExtensionPolicyTests` |
| Optional Draco data falls back to uncompressed accessors | `GLTFMeshLoader.Failure.dracoCompressionUnsupported` | `GLTFExtensionPolicyTests` |

The meshopt decoder is pure Swift and bounds-checks every read. Before its tests
were written it was compared with meshoptimizer 1.3.0 on 720 random streams:
unfiltered output matched exactly and filtered output stayed within the
specification's one-unit allowance. All 300 truncated streams were rejected.
Nothing from that library ships in Frostfield.

A KTX2 texture keeps its PNG or JPEG source as `GLTFScene.Texture.fallbackImage`,
and material preparation switches to it on a GPU that cannot sample the block
format. A rendered ASTC base color matches an RGBA8 one within two code values,
and the path tracer bakes an ASTC emissive map to the expected radiance.

Limits: Basis Universal payloads (BasisLZ, UASTC) are not transcoded, so such
textures use their PNG or JPEG source or fail. KTX2 has no Zstandard, cube,
array or 3D support. WebP and AVIF depend on the system's ImageIO decoders.
Draco is not decoded. Imported `EXT_mesh_gpu_instancing` nodes expand into one
draw node per instance that shares the node's renderers; they do not use
`InstanceSet` yet.

## Export

`GLTFExporter` writes the same extensions back out, from imported scenes and from
scene graphs: texture transforms, every material extension above, material
variants, KTX2, WebP and AVIF images through their texture extensions with the
PNG or JPEG fallback as the core source, and GPU instances from `GLTFScene` nodes
or `InstanceSet`. `GLTFExtensionExportTests` round-trips each one through GLB and
`.gltf`. Scene graph export writes each map's `TextureSampling` as its glTF
sampler and falls back to the default sampler, with a warning, only for a custom
Metal sampler without a description. The exporter warns for data glTF cannot
hold: instance tints, a selected variant, levels of detail beyond the finest,
and morph targets, which are not exported yet.

All tests ran with `xcodebuild` and Metal API Validation on an M4 Max Mac with
macOS 27.2. The Xcode MCP bridge could not build because its DerivedData folder
is on a full external drive.
