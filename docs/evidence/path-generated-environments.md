# Generated HDR environments and visible solar disks

`PathTracer.EnvironmentMap(device:width:height:generator:)` prepares a
linear HDR equirectangular image and its importance table in one operation.
The callback receives a top-left integer coordinate, the image dimensions,
normalized texel-center UV, and a unit direction in Frostfield's panorama
convention. It runs on the calling thread in row-major order. A thrown
callback error aborts construction. Native panoramas use a 2:1 aspect.
[Reference procedural texture](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/textures/ProceduralEquirectTexture.js)

Pass a `VerticalGradient` to make a zenith-to-nadir map. The default colors
are white and black, and the default exponent is two. The blend weight is
`((direction.y + 1) / 2)^exponent`. Both constructors use the same
half-float storage and luminance/solid-angle importance tables as authored
HDR maps.
[Reference gradient texture](https://github.com/gkjohnson/three-gpu-pathtracer/blob/80d188848fd7ca0ae4c565392d0487d9770d5aa6/src/textures/GradientEquirectTexture.js)

`SkyModel.sunLight()` now makes its finite directional disk visible to
escaping camera and scattered rays. The spectral panorama remains disk-free,
and the same analytic emitter still provides direct illumination. Its
uniform radiance equals the spectral beam illuminance divided by the
projected disk solid angle, `π sin²(radius)`. Direct-light and scattered
samples use reciprocal power-heuristic weights. Geometry can occlude the
disk. Hiding the background also hides the disk from camera rays.
Other analytic lights preserve their existing behavior unless a finite
directional emitter opts in with `diskVisible`.

The path disk has uniform radiance. Spectral limb darkening in
`AerialPerspective` is a separate raster effect. The visible disk does not
change the first-hit geometry outputs of a ray that escapes the scene.

## Verification

Twenty-four focused tests passed on Apple Silicon macOS and the iPhone 18
Pro simulator with Metal API Validation. The generated pixels matched the
HDR oracle, and the procedural and authored images produced identical
importance samples and densities. Path readback verified the spectral disk's
radiance, angular edge, camera hiding and occlusion. A near-delta mirror
matched a constant-radiance reference, and diffuse lighting matched the
analytical irradiance with the disk visible or direct-only. The high-intensity
fixture also exercised a directional sample above half-float range whose
reflected value fits. The existing spectral sky and directional light
regressions passed.

The Xcode bridge build encountered a full external DerivedData volume, so
verification used `xcodebuild` with internal caches.
