# Raster PBR layer argument buffers

The iPad Pro 13-inch (M5) Simulator reports Metal argument-buffer Tier 1.
In `TextureTransformTests.layerMapTransform`, direct Metal sampling returned
the expected alpha values of `0` and `1`. The scalar-only raster reference
gave `0.018371582` on both sides. The mapped draw returned zero on both
sides with either UV offset. The same inputs on the M4 Max Mac worked
with Tier 2.

Tier 1 argument buffers have a device-specific memory layout. Raster PBR
now encodes its extension texture array once during material preparation
through `MTLArgumentEncoder`, with the same starting ID as the shader's
array. Each draw retains the existing texture residency and sampler bindings.
[Apple argument-buffer layout and encoding](https://developer.apple.com/documentation/metal/improving-cpu-performance-by-using-argument-buffers#Encode-resources-into-argument-buffers)

After the change, the iPad raster samples matched the Mac exactly. The
unshifted pair was `0, 0.018371582` and the shifted pair was
`0.018371582, 0`. The original assertions stayed unchanged.

The focused material extension, UV transform, and packed normal suites
passed 20 tests on each destination with Metal API validation.
