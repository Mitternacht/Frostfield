# Asset Studio Lab

Launch FrostfieldLab with `--asset-studio` to inspect the authored rangefinder
in the existing studio renderer. Drag to orbit the camera around the fixed
subject. On macOS, the app executable also accepts
`--capture-asset-studio OUTPUT.png [YAW_RADIANS]` for a 1440×960 still.

The lens grip uses `MeshGeometry.unwrappedUVAtlas` to generate a 512×512 UV1
atlas with a four-texel gutter. `MeshTextureBaker` bakes the authored wear
colors into that atlas. The drawn mesh samples the baked base color and keeps
the original knurled silhouette, normals, and metal finish. The same render
shows the surrounding brushed alloy, leather, and transmissive lens.
[Automatic UV atlas API](mesh-uv-atlas.md)

`MeshLightmapBaker` bakes directional irradiance into the studio sweep's 512×512
UV0 texture using the placed camera as the opaque caster geometry. The sweep
samples the bake as diffuse radiance. Asset Studio moves the view instead of
rotating the asset so the baked shadows and subject stay aligned. The subject
retains the studio's runtime lighting and HDR display pass.

Xcode 27.2 built the automatic-atlas fixture for macOS, iPhone 18 Pro
Simulator, and visionOS Simulator. Native macOS captures at the default yaw
and 0.42 radians completed with Metal API Validation. Compared with the earlier authored-UV renders,
pixel differences stayed inside the grip region in both views. The captures
show the baked floor shadow staying aligned with the fixed subject. The
integrated macOS capture matched the isolated default capture byte for byte.

The default bake runs at fixture startup and samples six authored light
directions against the fixed opaque caster set.

Add `--studio-bake-indirect` to the capture command to bake one diffuse
bounce into the sweep at 48 samples per covered texel.
`--studio-bake-reference` captures the direct term using the same atlas.
The indirect variant uses each opaque piece's PBR base-color and emissive
factors and maps, including the grip's baked UV1 texture. The rendered
subject keeps its full PBR materials. The bake approximates opaque surfaces
as Lambertian, retains the matte proxy for the optional shader finish, and
excludes the transmissive lens. Metallic-roughness, specular reflection,
and transmission are outside this diffuse bake.

## Shared floor and backdrop atlas

```sh
FrostfieldLab --capture-asset-studio shared-direct.png --studio-bake-shared-reference
FrostfieldLab --capture-asset-studio shared-indirect.png --studio-bake-shared-indirect
```

Both variants pack the floor and backdrop into one 320×320 UV1 atlas with
four-texel gutters. With `--studio-sky`, the rear receiver is a flat floor
extension. The two opaque receivers exchange diffuse light with each other
and with the rangefinder's textured transport surfaces. They render as
separate meshes sampling the same irradiance texture. The indirect variant
adds 48 paths per covered texel at one diffuse bounce.

The initial macOS capture with Metal API Validation changed 79,405 of
81,368 covered atlas texels. The 1440×960 direct and indirect renders
differed at 19,948 RGB pixels. The existing one-receiver indirect capture
matched its reference byte for byte.

At `ebff184` all three captures reproduced those PNGs byte for byte.
Independent UV1 coverage masks verified changes at 42,459 of 44,384
interior floor texels and 32,299 of 32,301 interior backdrop texels.
The aggregate count also includes the atlas gutters. The Lab built on
Mac, iPad Simulator and Vision Pro Simulator.

The `5f0f4e5` integration added cutout transport to the baker and reproduced the
single- and shared-receiver indirect captures byte for byte. The per-receiver
counts and linear radiance also matched. The dedicated bake fixtures
verify the cutout behavior. [Cutout baking evidence](static-lightmap.md#alpha-tested-transport)

The `3622df3` package reproduced both indirect PNGs byte for byte.
Lab builds passed for Mac, iPad Simulator and Vision Pro Simulator.

The `48a4e72` package reproduced the single- and shared-receiver indirect
PNGs byte for byte with Metal API Validation. It retained the shared atlas
coverage of 79,405 changed texels, including 42,459 on the floor and 32,299
on the backdrop. The single-receiver bake retained 12,302 changed texels.
The Lab built for Mac, iPad Simulator and Vision Pro Simulator.

The `d5971cd` package reproduced both indirect PNGs and the same coverage
with Metal API Validation. Lab builds passed for all three destinations.

The `ad959d2` package reproduced both indirect PNGs and the same coverage
after adding mutable casters and rolling lightmap previews. The Lab built
for Mac, iPad Simulator and Vision Pro Simulator.

The `69f0438` package reproduced both indirect PNGs and their coverage with
Metal API Validation after integrating custom irradiance, shader BVH queries,
and immutable GPU deformation poses. The Lab built for all three destinations.

## Metal-authored irradiance

```sh
FrostfieldLab --capture-asset-studio shader-irradiance.png --studio-bake-shared-shader
```

This variant uses the shared 320×320 UV1 atlas. Six direct light directions
retain the asset's baked shadows. A Metal buffer graph adds irradiance from
an authored warm bounce card using 16 stratified card samples. The graph
uses the receiver position, normal and sample index with cosine and distance
weighting. Halton UV sampling resolves the chart edges. The floor and
backdrop bind the result as a native PBR light map and disable their live
analytic and ambient light, since the sweep's direct light is already baked.
[Metal irradiance source](static-lightmap.md#caller-defined-irradiance) ·
[PBR light maps](pbr-light-map.md)

The Mac capture at `92f36dc` rendered at 1440×960 with Metal API Validation.
Against a direct-only bake at the same 16 Halton positions, the graph changed
22,543 of 82,536 covered atlas texels. Interior coverage checks counted
16,131 changed floor texels and 4,790 changed backdrop texels.

The integrated `b55a4e7` capture reproduced those bake counts and radiance
measurements. Its final PNG differed at three of 1,382,400 pixels, by one
8-bit channel step per pixel. Both other bake variants matched their
reference PNGs byte for byte. A native Metal shader swap restored the
shared-shader reference exactly with the previous GTAO implementation.

The `a1822b9` package specializes default GTAO separately from optional
clipping and tap spacing. All three captures then reproduced their
reference PNGs byte for byte with Metal API Validation.

The integrated `76350a4` Lab reproduced the bake measurements for all
three capture variants with Metal API Validation. Each 1440×960 PNG
changed three pixels by one 8-bit channel step per pixel. Replacing
the directional shadow receiver shaders with the versions from
`922c10a` restored the shared-shader reference byte for byte. That
comparison isolates the rounding change to the directional receiver
controls introduced in `c8d2f26`.

## Single-receiver verification

Native macOS captures with Metal API Validation reproduced the earlier
direct reference PNG byte for byte. Against that reference, the textured
bake changed 10,504 RGB pixels at 1440×960, compared with 7,050 for the
earlier constant proxy. The two indirect renders differed at 10,607 pixels
within the visible floor region. The textured bake added light at 12,302
of 66,816 floor atlas texels.

The integrated Xcode 27.2 build at `c783bcf` reproduced both PNGs byte for
byte with Metal API Validation. The Lab also built for iOS Simulator and
visionOS Simulator.

The `b6b8781` build repeated the textured indirect capture with the same
texel measurements and byte-identical PNG after the BVH and raycast changes.
The `2813a8b` build also reproduced the existing single-receiver capture
byte for byte after integrating shared lightmap atlases.
The `969c199` build reproduced the textured indirect PNG byte for byte
with Metal API Validation after adding point and spot light baking, relighting,
and IES and cookie sampling. The existing studio fixture still used its six
baked directional sources and 48 diffuse samples per covered texel.

The `ad18bca` build reproduced the floor bake at 48 samples and 12,302
of 66,816 changed texels, with identical mean and peak linear radiance.
Compared with `969c199`, the 1440×960 capture changed one 8-bit RGB code
in each of two pixels. A second capture matched the first byte for byte.
Both captures used Metal API Validation.

The `5760918` build reproduced the indirect capture byte for byte after
integrating texture tools, raster upload controls, and post-process timing.
The bake retained the same covered-texel count and linear radiance with
Metal API Validation enabled.

The `b423a0a` build reproduced the floor bake at 48 samples and 12,302
changed texels with identical mean and peak linear radiance. The PNG
matched `969c199` byte for byte. Compared with `5760918`, one 8-bit RGB
code changed in each of two pixels. The capture used Metal API Validation.

The `dae857d` build reproduced the same floor bake measurements with
Metal API Validation enabled. In the 1440×960 capture, one pixel changed
by one 8-bit green code relative to `b423a0a`. Compared with `5760918`,
three pixels changed, each by at most one code in any channel.

The `63b3c4b` build reproduced the `dae857d` indirect capture byte for
byte. The bake used 48 samples and changed 12,302 of 66,816 texels, with
the same mean and peak linear radiance. Metal API Validation was enabled.

The `31829dd` build reproduced the `63b3c4b` indirect capture byte for
byte with Metal API Validation. Covered texels and mean and peak linear
radiance also matched.

The `4be085b` build reproduced the `31829dd` indirect capture byte for
byte. With Metal API Validation enabled, it matched the 48-sample bake's
covered texels and mean and peak linear radiance.

The `1121bf6` capture again matched the preceding image byte for byte.
The 48-sample bake retained the same covered texels and linear radiance
with Metal API Validation enabled.

The `2ee32cb` capture matched that image byte for byte with Metal API
Validation enabled. Its 48-sample bake again changed 12,302 of 66,816
texels and reproduced the mean and peak linear radiance.

The `ab55d25` capture reproduced the image, covered texel count, and linear
radiance with Metal API Validation enabled.

The `c92d1fc` capture kept the same image hash and 48-sample bake results.

The `d76674a` capture reproduced the same image bytes and indirect bake
with Metal API Validation enabled.

The `7e8a411` capture matched the prior image and 48-sample bake with
Metal API Validation enabled.

The `45d435e` capture reproduced the same image bytes and indirect bake.

The `44611e3` Lab built on macOS, iOS Simulator and visionOS Simulator.
The macOS textured indirect capture reproduced the image bytes, 48-sample
floor bake, 12,302 changed texels and linear radiance with Metal API Validation.

The `bd65f32` build reproduced the same PNG and floor bake with Metal API
Validation enabled. The Lab also built for iOS and visionOS Simulator.

The `2d89fd4` Lab retained the same default display output after adding
wide-gamut profiles. The macOS textured indirect capture reproduced the
PNG byte for byte, with the same 48-sample bake, 12,302 changed texels
and linear radiance under Metal API Validation. iOS and visionOS
Simulator Lab builds passed.

The `36fdcde` build repeated the 48-sample indirect floor bake with
Metal API Validation and reproduced the preceding PNG byte for byte.
The bake changed 12,302 of 66,816 floor atlas texels. The Lab also
built for iPad and Vision Pro simulators.

The `78c7c9d` integration kept the same 48-sample texel and radiance
measurements and produced the same PNG with Metal API Validation.
The Lab built on Mac and for iPad and Vision Pro simulators.

The `bb0acd5` build repeated the 48-sample bake and produced a
byte-identical PNG with Metal API Validation. Lab builds also passed
for the iPad and Vision Pro simulators.

The `e76ac58` integration repeated the 48-sample floor bake with
Metal API Validation and reproduced the preceding PNG byte for byte.
The Lab also built for iPad and Vision Pro simulators.

The `7789983` integration repeated the 48-sample floor bake with
Metal API Validation and reproduced the preceding PNG byte for byte.
The Lab also built for iPad and Vision Pro simulators.

The `5194b80` integration reproduced the 48-sample PNG byte for byte,
with 12,302 of 66,816 floor texels receiving additional radiance. The Lab
built for Mac, iPad Simulator and Vision Pro Simulator. This regression
uses the existing scene and settings. The mesh-emitter opt-in and LTC
response are checked separately in their focused Metal fixtures.
[Emissive sampling evidence](path-emissive-mis.md) · [Rectangle lighting evidence](rectangle-ltc.md)

The `b711721` integration reproduced the 48-sample indirect bake and PNG
byte for byte with Metal API Validation. Lab builds passed on Mac,
iPad Simulator and Vision Pro Simulator.

The `2677984` integration reproduced the same bake and PNG byte for byte
with Metal API Validation. The Lab built again for all three targets.

The `61d06b3` integration reproduced the 48-sample bake and PNG byte for
byte. Mac, iPad Simulator and Vision Pro Simulator Lab builds passed.

The `9ef92aa` integration reproduced the reference PNG byte for byte with
Metal API Validation. The 48-sample bake added radiance to 12,302 of 66,816
floor texels. The Lab built for Mac, iPad Simulator and Vision Pro Simulator.

The `4367c12` captures finished after that checkpoint was recorded. The
shared shader, shared indirect and single-receiver PNGs and their bake
metrics matched the preceding verified checkpoint byte for byte. Each image
still differs from the earlier `rig-sss-sun-refraction` capture in three
pixels by one 8-bit channel step. That difference comes from the directional
shadow change and is not new here.

The `91e5c13` integration, with layered colored glass shadows, RGB sun
radiance, Water2Mesh-style viewport refraction and overlapping fog boxes,
repeated all three captures with Metal API Validation. The PNGs and bake
metrics again matched the preceding checkpoint byte for byte, including the
48-sample bake with 12,302 of 66,816 floor texels receiving radiance. The Lab
built for Mac, iPad Simulator and Vision Pro Simulator in Xcode 27.2.

The integration through `0d83e34` added mesh-shaped fog, graph-authored colored
shadow casters, array-camera encoding, native KTX2 cube-array export, opt-in
dithering in the HDR display pass and per-node encode hooks. It
repeated all three captures with Metal API Validation. The PNGs again matched the
previous checkpoint byte for byte, and the Lab built for Mac, iPad Simulator and
Vision Pro Simulator in Xcode 27.2.
