# Wide-gamut HDR display output

Prepare `HDRDisplay` for sRGB, Display P3 or extended linear Rec. 2020
primaries. Source textures contain premultiplied linear sRGB. The display
pass applies exposure and the selected sRGB tone curve, then transforms
the result into the output primaries. With `.none`, it converts signed
linear source values, including colors outside the sRGB gamut.

Apply `HDRDisplay.presentationColorSpace` to the layer or image that
presents the texture. RGBA16Float uses the matching extended linear
profile. An sRGB storage format applies the same transfer curve for
sRGB and Display P3, and 8-bit linear storage uses the corresponding
linear profile. Rec. 2020 uses RGBA16Float. Alpha stays linear.

Metal readback compares P3 and Rec. 2020 with independent double-precision
XYZ transforms using the pinned Three.js matrices. It checks a pure
P3 primary encoded outside sRGB, signed HDR values, neutral highlights,
premultiplication and the 8-bit P3 transfer. The focused Mac and iPhone
18 Pro Simulator runs passed 48 tests each across five suites with Metal
API Validation, including existing sRGB tone and HDR loading coverage.
The Xcode bridge timed out resolving the package destination. These
tests used internal DerivedData with `xcodebuild`.

[Three.js r187 color management](https://github.com/mrdoob/three.js/blob/r187/src/math/ColorManagement.js)
· [Three.js r187 wide-gamut color spaces](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/ColorSpaces.js)
· [Apple layer color space](https://developer.apple.com/documentation/quartzcore/cametallayer/colorspace)
