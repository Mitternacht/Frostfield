import Foundation
import Metal

// Graph equivalents of the Three.js post-processing shaders. Each preset declares
// its parameters as uniforms named in its documentation, so they can be edited live
// through `PostProcessPass.uniforms` or `PostProcessChain.uniforms(at:)`. Colors are
// premultiplied linear RGBA; presets that are not linear in color unpremultiply
// first and premultiply their result.

/// The output pixel's position in pixels, `screenUV / texelSize`, divided exactly.
private func pixelCoordinates(_ p: ShaderGraphBuilder) -> ShaderValue<SIMD2<Float>> {
    let divide = p.function(
        "pixel_coordinates", returns: SIMD2<Float>.self, parameters: [.init("uv", .vector2), .init("texel", .vector2)],
        body: "return uv / texel;"
    )
    return divide(p.screenUV, p.texelSize)
}

private func pixelBlockCenter(_ pixel: ShaderValue<SIMD2<Float>>, block: ShaderValue<Float>) -> ShaderValue<SIMD2<Float>> {
    let corner = (pixel / block).floor() * block
    return corner + (block * Float(0.5)).floor() + Float(0.5)
}

extension PostProcessGraph {
    /// Three.js `SepiaShader`: the sepia matrix scaled by "Amount", with each straight
    /// channel clamped to 1 as Three.js does.
    public static func sepia(amount: Float = 1) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let a = p.uniform("Amount", amount)
            let color = p.sceneColor(at: p.screenUV)
            let c = color.rgb
            let one = p.constant(Float(1))
            let r = c.dot(p.vector(one - a * Float(0.607), a * Float(0.769), a * Float(0.189)))
            let g = c.dot(p.vector(a * Float(0.349), one - a * Float(0.314), a * Float(0.168)))
            let b = c.dot(p.vector(a * Float(0.272), a * Float(0.534), one - a * Float(0.869)))
            return p.vector(r, g, b).min(p.vector(color.a, color.a, color.a))
        }
    }

    /// Three.js `HueSaturationShader`: "Hue" from −1 to 1 rotates hue by up to ±180°
    /// about the gray axis, and "Saturation" from −1 (gray) to 1 scales chroma.
    public static func hueSaturation(hue: Float = 0, saturation: Float = 0) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let angle = p.uniform("Hue", hue) * Float.pi
            let s = angle.sin()
            let c = angle.cos()
            let root3 = Float(3).squareRoot()
            let weights = (p.vector(c * Float(2), -(s * root3) - c, s * root3 - c) + Float(1)) / Float(3)
            let color = p.sceneColor(at: p.screenUV).rgb
            let rotated = p.vector(
                color.dot(weights), color.dot(weights.swizzle("zxy", as: SIMD3<Float>.self)),
                color.dot(weights.swizzle("yzx", as: SIMD3<Float>.self))
            )
            let saturationValue = p.uniform("Saturation", saturation)
            let average = (rotated.x + rotated.y + rotated.z) / Float(3)
            let increase = Float(1) - p.constant(Float(1)) / (Float(1.001) - saturationValue)
            let factor = p.select(saturationValue.greaterThan(p.constant(Float(0))), increase, -saturationValue)
            return rotated + (p.vector(average, average, average) - rotated) * factor
        }
    }

    /// Three.js `BrightnessContrastShader` on straight color: add "Brightness" (−1 to 1),
    /// then expand about ½ by `1 / (1 − Contrast)` for positive "Contrast" or compress
    /// by `1 + Contrast` for negative values, from −1 to just below 1.
    public static func brightnessContrast(brightness: Float = 0, contrast: Float = 0) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            let contrastValue = p.uniform("Contrast", contrast)
            let shifted = color.rgb / color.a + p.uniform("Brightness", brightness) - Float(0.5)
            let expand = shifted / (Float(1) - contrastValue)
            let compress = shifted * (Float(1) + contrastValue)
            let graded = p.select(contrastValue.greaterThan(p.constant(Float(0))), expand, compress) + Float(0.5)
            return p.vector(graded * color.a, color.a)
        }
    }

    /// Three.js `ColorCorrectionShader` on straight color:
    /// `Multiply · pow(color + Add, Power)` with vector3 uniforms.
    public static func colorCorrection(
        power: SIMD3<Float> = SIMD3(2, 2, 2), multiply: SIMD3<Float> = .one, add: SIMD3<Float> = .zero
    ) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            let straight = color.rgb / color.a
            let corrected = p.uniform("Multiply", multiply) * (straight + p.uniform("Add", add)).pow(p.uniform("Power", power))
            return p.vector(corrected * color.a, color.a)
        }
    }

    /// Three.js `VignetteShader` on straight color:
    /// `mix(color, 1 − Darkness, dot(uv, uv))` with `uv = (screenUV − ½) · Offset`.
    public static func vignette(offset: Float, darkness: Float) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            let uv = (p.screenUV - Float(0.5)) * p.uniform("Offset", offset)
            let target = Float(1) - p.uniform("Darkness", darkness)
            let straight = (color.rgb / color.a).mix(p.vector(target, target, target), by: uv.dot(uv))
            return p.vector(straight * color.a, color.a)
        }
    }

    /// Three.js `RGBShiftShader`: red from `screenUV + offset`, green and alpha from the
    /// pixel, blue from `screenUV − offset`, with `offset = Amount · (cos Angle, −sin Angle)`
    /// in normalized frame units. The sign keeps Three.js's on-screen direction under
    /// Frostfield's top-left coordinates.
    public static func rgbShift(amount: Float = 0.005, angle: Float = 0) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let angleValue = p.uniform("Angle", angle)
            let offset = p.vector(angleValue.cos(), -angleValue.sin()) * p.uniform("Amount", amount)
            let red = p.sceneColor(at: p.screenUV + offset)
            let center = p.sceneColor(at: p.screenUV)
            let blue = p.sceneColor(at: p.screenUV - offset)
            return p.vector(red.r, center.g, blue.b, center.a)
        }
    }

    /// Three.js `DigitalGlitch` (used by `GlitchPass`): hashed block displacement, line
    /// tears, an RGB split and snow.
    ///
    /// Uniforms: "Bypass" (above ½ passes the frame through), "Amount", "Angle", "Seed",
    /// "SeedX", "SeedY", "DistortionX", "DistortionY" and "ColumnSize". The random
    /// displacement texture of Three.js is replaced by a hash of 64 × 64 blocks scaled
    /// by `Seed²`. Drive the uniforms each frame with `GlitchRandomizer`.
    public static func glitch() throws -> PostProcessGraph {
        try makeGlitch(usesDisplacement: false)
    }

    /// Three.js r187 `DigitalGlitch` with a sampled displacement map. Bind a
    /// linear texture named "Displacement". Its red channel is sampled with
    /// nearest filtering and clamp-to-edge addressing.
    ///
    /// `GlitchRandomizer` drives the same uniforms as `glitch()`. "SnowStrength"
    /// defaults to one and can be set to zero for band and channel distortion alone.
    /// The shifted channels and snow use the warped center pixel's alpha.
    public static func glitchWithDisplacement() throws -> PostProcessGraph {
        try makeGlitch(usesDisplacement: true)
    }

    private static func makeGlitch(usesDisplacement: Bool) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let seed = p.uniform("Seed", Float(0.02))
            let seeds = p.vector(seed, p.uniform("SeedX", Float(0.02)), p.uniform("SeedY", Float(0.02)),
                                 p.uniform("ColumnSize", Float(0.05)))
            let distortion = p.vector(p.uniform("DistortionX", Float(0.5)), p.uniform("DistortionY", Float(0.6)))
            var parameters: [ShaderFunction.Parameter] = [.init("uv", .vector2), .init("s", .vector4), .init("d", .vector2)]
            if usesDisplacement { parameters.append(.init("disp", .scalar)) }
            let hash = usesDisplacement ? ""
                : "float disp = fract(sin(dot(floor(q * s.x * s.x * 64.0f), float2(12.9898f, 78.233f))) * 43758.5453f);"
            let coordinates = p.function(
                "glitch_coordinates", returns: SIMD2<Float>.self, parameters: parameters,
                body: """
                float2 q = uv;
                \(hash)
                if (q.y < d.x + s.w && q.y > d.x - s.w * s.x) { q.y = s.y > 0.0f ? 1.0f - (q.y + d.y) : d.y; }
                if (q.x < d.y + s.w && q.x > d.y - s.w * s.x) { q.x = s.z > 0.0f ? d.x : 1.0f - (q.x + d.x); }
                q.x += disp * s.y * (s.x / 5.0f);
                q.y += disp * s.z * (s.x / 5.0f);
                return q;
                """
            )
            let snow = p.function(
                "glitch_snow", returns: Float.self,
                parameters: [.init("pixel", .vector2), .init("seed", .scalar)],
                body: """
                float2 cell = floor(pixel / 0.5f);
                return fract(sin(dot(float2(cell.x * seed, cell.y * seed * 50.0f), float2(12.9898f, 78.233f))) * 43758.5453f) * 0.2f;
                """
            )
            let amount = p.uniform("Amount", Float(0.01))
            let angle = p.uniform("Angle", Float(0))
            let q: ShaderValue<SIMD2<Float>>
            if usesDisplacement {
                // The pinned bands and hash use bottom-left coordinates. Convert
                // to Metal's top-left coordinates only at texture lookup boundaries.
                let one = Float(1)
                let uv = p.vector(p.screenUV.x, one - p.screenUV.y)
                let lookup = uv * seed * seed
                let map = p.texture("Displacement", content: .data, addressing: .clampToEdge, filtering: .nearest)
                let displacement = map.sample(p.vector(lookup.x, one - lookup.y)).r
                let warped = coordinates(uv, seeds, distortion, displacement)
                q = p.vector(warped.x, one - warped.y)
            } else {
                q = coordinates(p.screenUV, seeds, distortion)
            }
            let offset = p.vector(angle.cos(), -angle.sin()) * amount
            let red = p.sceneColor(at: q + offset)
            let center = p.sceneColor(at: q)
            let blue = p.sceneColor(at: q - offset)
            let pixel = usesDisplacement
                ? p.vector(p.screenCoordinate.x, p.viewportSize.y - p.screenCoordinate.y) : pixelCoordinates(p)
            let noise = snow(pixel, seed) * amount * Float(200)
            let glitched: ShaderValue<SIMD4<Float>>
            if usesDisplacement {
                let strength = p.uniform("SnowStrength", Float(1)).max(Float(0))
                let rgb = p.vector((red.r / red.a) * center.a, center.g, (blue.b / blue.a) * center.a)
                glitched = p.vector(rgb + noise * strength * center.a, center.a)
            } else {
                glitched = p.vector(red.r, center.g, blue.b, center.a) + noise
            }
            return p.select(p.uniform("Bypass", Float(0)), p.sceneColor(at: p.screenUV), glitched)
        }
    }

    /// An RGB halftone like Three.js `HalftonePass` with circular dots: each channel is
    /// screened on its own grid of `2 · Radius`-pixel cells rotated by "RotateR",
    /// "RotateG" or "RotateB" radians. A cell samples the scene at its center and draws an
    /// antialiased dot covering that channel's straight (unpremultiplied) value as a
    /// fraction of the cell area. "Blending" mixes from the original (0) to the screen
    /// (1), premultiplied by the pixel's alpha. Alpha is kept.
    public static func halftone(
        radius: Float = 4, rotations: SIMD3<Float> = SIMD3(Float.pi / 12, Float.pi / 6, Float.pi / 4), blending: Float = 1
    ) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let cell = p.uniform("Radius", radius) * Float(2)
            let pixel = pixelCoordinates(p)
            let original = p.sceneColor(at: p.screenUV)
            var channels: [ShaderValue<Float>] = []
            for (index, name) in ["RotateR", "RotateG", "RotateB"].enumerated() {
                let angle = p.uniform(name, rotations[index])
                let c = angle.cos()
                let s = angle.sin()
                let rotated = p.vector(pixel.x * c + pixel.y * s, pixel.y * c - pixel.x * s)
                let center = ((rotated / cell).floor() + Float(0.5)) * cell
                let back = p.vector(center.x * c - center.y * s, center.x * s + center.y * c)
                let sample = p.sceneColor(at: back * p.texelSize)
                let straight = [sample.r, sample.g, sample.b][index] / sample.a
                let value = p.select(sample.a.greaterThan(p.constant(Float(0))), straight, p.constant(Float(0))).saturated()
                let dot = cell * (value / Float.pi).squareRoot()
                let distance = (rotated - center).length()
                channels.append(Float(1) - distance.smoothstep(dot - Float(0.5), dot + Float(0.5)))
            }
            let screened = p.vector(channels[0], channels[1], channels[2])
            return original.rgb.mix(screened * original.a, by: p.uniform("Blending", blending))
        }
    }

    /// Three.js `DotScreenShader`: `average · 10 − 5 + 4 · sin(x) · sin(y)` over the
    /// rotated, scaled point `(screenUV · TextureSize − Center)`, where `average` is the
    /// mean straight RGB, as gray premultiplied by the source alpha, which is kept.
    /// "TextureSize" defaults to 256 × 256 as in `DotScreenPass`.
    public static func dotScreen(
        center: SIMD2<Float> = SIMD2(0.5, 0.5), angle: Float = 1.57, scale: Float = 1,
        textureSize: SIMD2<Float> = SIMD2(256, 256)
    ) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let angleValue = p.uniform("Angle", angle)
            let s = angleValue.sin()
            let c = angleValue.cos()
            let uv = p.vector(p.screenUV.x, Float(1) - p.screenUV.y)
            let texel = uv * p.uniform("TextureSize", textureSize) - p.uniform("Center", center)
            let point = p.vector(texel.x * c - texel.y * s, texel.x * s + texel.y * c) * p.uniform("Scale", scale)
            let pattern = point.x.sin() * point.y.sin() * Float(4)
            let color = p.sceneColor(at: p.screenUV)
            let average = (color.r + color.g + color.b) / (color.a * Float(3))
            let gray = (average * Float(10) - Float(5) + pattern) * color.a
            return p.vector(gray, gray, gray)
        }
    }

    /// Nearest-texel pixelation like Three.js `RenderPixelatedPass`: every
    /// "PixelSize"-pixel block shows the source texel at its center (offset
    /// `floor(size / 2)` from the block's top-left), including alpha.
    public static func pixelate(size: Float = 6) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let block = p.uniform("PixelSize", size).max(p.constant(Float(1)))
            return p.sceneColor(at: pixelBlockCenter(pixelCoordinates(p), block: block) * p.texelSize)
        }
    }

    /// CRT image styling from the Three.js r187 retro example. The graph samples
    /// barrel-warped color in "PixelSize"-pixel blocks, then applies ordered
    /// "DitherStrength" and "ColorLevels" to straight HDR RGB. A color level count
    /// of zero bypasses quantization. The sampled alpha is preserved.
    ///
    /// "Curvature" is 0...0.49. "ScanlineDensity" multiplies the frame height to set
    /// the sine frequency. "ScanlineSpeed" scrolls using `ShaderFrame.time`.
    /// "ScanlineIntensity", "VignetteIntensity" and "VignetteSmoothness" are 0...1.
    public static func retroCRT(
        pixelSize: Int = 4, curvature: Float = 0.02, colorLevels: Int = 32, ditherStrength: Float = 1,
        scanlineIntensity: Float = 0.3, scanlineDensity: Float = 1, scanlineSpeed: Float = 0,
        vignetteIntensity: Float = 0.3, vignetteSmoothness: Float = 0.6
    ) throws -> PostProcessGraph {
        guard (1...1024).contains(pixelSize), (0...4096).contains(colorLevels),
              (0...0.49).contains(curvature), (0...4).contains(scanlineDensity), scanlineSpeed.isFinite,
              [ditherStrength, scanlineIntensity, vignetteIntensity, vignetteSmoothness].allSatisfy({ (0...1).contains($0) })
        else {
            throw PostProcessEffectError.invalidSettings(
                "Retro CRT needs pixel size 1...1024, color levels 0...4096, curvature 0...0.49, density 0...4, finite speed and normalized strengths."
            )
        }
        return try PostProcessGraph { p in
            let one = p.constant(Float(1))
            let zero = p.constant(Float(0))
            let centered = p.screenUV * Float(2) - Float(1)
            let curve = p.uniform("Curvature", curvature).clamped(0, 0.49)
            let radiusSquared = centered.dot(centered)
            let warped = (centered / (one - radiusSquared * curve)) * (one - curve * Float(2)) * Float(0.5) + Float(0.5)
            let uv = p.select(curve.greaterThan(zero), warped, p.screenUV)
            let block = p.uniform("PixelSize", Float(pixelSize)).floor().clamped(1, 1024)
            let snapped = pixelBlockCenter(uv * p.viewportSize, block: block) * p.texelSize
            let color = p.sceneColor(at: p.select(block.greaterThan(one), snapped, uv))

            let levels = p.uniform("ColorLevels", Float(colorLevels)).floor().clamped(0, 4096)
            let steps = levels.max(one)
            let straight = color.rgb / color.a
            let dithered = p.bayerDither(straight, pixel: p.screenCoordinate, levels: steps)
            let dither = p.uniform("DitherStrength", ditherStrength).clamped(0, 1)
            let quantized = p.posterize(straight.mix(dithered, by: dither), steps: steps).max(.zero)
            let rgb = p.select(levels.greaterThan(zero), quantized * color.a, color.rgb)

            let softness = p.uniform("VignetteSmoothness", vignetteSmoothness).clamped(0, 1)
            let outer = p.constant(Float(1.42))
            let radial = radiusSquared.squareRoot().smoothstep(outer * (one - softness), outer)
            let vignette = one - p.uniform("VignetteIntensity", vignetteIntensity).clamped(0, 1) * radial

            let density = p.uniform("ScanlineDensity", scanlineDensity).clamped(0, 4)
            let phase = (p.screenUV.y - p.time * p.uniform("ScanlineSpeed", scanlineSpeed)) * p.viewportSize.y * density
            let scanline = phase.sin() * Float(0.5) + Float(0.5)
            let scan = one - scanline * p.uniform("ScanlineIntensity", scanlineIntensity).clamped(0, 1)
            return p.vector(rgb * vignette * scan, color.a)
        }
    }

    /// Three.js `SobelOperatorShader` on Rec. 709 luminance: the gradient magnitude
    /// `sqrt(Gx² + Gy²)` of the 3 × 3 Sobel kernels with one-pixel taps, written as gray
    /// RGB with alpha 1. Three.js reads the red channel of a prior luminance pass.
    public static func sobel() throws -> PostProcessGraph {
        try PostProcessGraph { p in
            func tap(_ x: Float, _ y: Float) -> ShaderValue<Float> {
                p.sceneColor(at: p.screenUV + p.texelSize * SIMD2(x, y)).rgb.luminance()
            }
            let tl = tap(-1, -1), t = tap(0, -1), tr = tap(1, -1)
            let l = tap(-1, 0), r = tap(1, 0)
            let bl = tap(-1, 1), b = tap(0, 1), br = tap(1, 1)
            let gx = (tr + r * Float(2) + br) - (tl + l * Float(2) + bl)
            let gy = (bl + b * Float(2) + br) - (tl + t * Float(2) + tr)
            let magnitude = (gx * gx + gy * gy).squareRoot()
            return p.vector(magnitude, magnitude, magnitude, p.constant(Float(1)))
        }
    }

    /// Three.js `FreiChenShader`: the square root of edge-subspace energy over
    /// total neighborhood energy. One-texel taps use the length of sampled RGB.
    /// The result is opaque gray. A zero-energy neighborhood returns black.
    public static func freiChen() throws -> PostProcessGraph {
        try PostProcessGraph { p in
            func tap(_ x: Float, _ y: Float) -> ShaderValue<Float> {
                p.sceneColor(at: p.screenUV + p.texelSize * SIMD2(x, y)).rgb.length()
            }
            let tl = tap(-1, -1), t = tap(0, -1), tr = tap(1, -1)
            let l = tap(-1, 0), c = tap(0, 0), r = tap(1, 0)
            let bl = tap(-1, 1), b = tap(0, 1), br = tap(1, 1)
            let a = Float(0.3535533845424652)
            let half = Float(0.5)

            // The first four r187 masks, with Y converted from bottom-up UVs to
            // top-down Metal coordinates. Pairwise differences cancel flat color exactly.
            let vertical = ((bl - tl) + (br - tr)) * a + (b - t) * half
            let horizontal = ((bl - br) + (tl - tr)) * a + (l - r) * half
            let diagonal = ((l - r) + (t - b)) * a + (br - tl) * half
            let antidiagonal = ((r - l) + (t - b)) * a + (bl - tr) * half
            let straight = p.vector(vertical, horizontal)
            let diagonals = p.vector(diagonal, antidiagonal)
            let edgeEnergy = straight.dot(straight) + diagonals.dot(diagonals)

            // The nine masks form an orthonormal basis, so the sum of their
            // squared responses is the squared norm of the original neighborhood.
            let left = p.vector(tl, l, bl), middle = p.vector(t, c, b), right = p.vector(tr, r, br)
            let energy = left.dot(left) + middle.dot(middle) + right.dot(right)
            // Graph division returns zero when its denominator is zero.
            let strength = (edgeEnergy / energy).squareRoot()
            return p.vector(strength, strength, strength, p.constant(Float(1)))
        }
    }

    /// Composites scene, bloom and lens dirt as in the Three.js r187 lens effects example.
    /// Bind the separate strength-weighted HDR glow as "Bloom" and the dirt texture
    /// as "Dirt". Pass that texture's dimensions in `dirtSize` to center-crop it to the
    /// frame's aspect ratio. "Intensity" is a live nonnegative uniform.
    ///
    /// Source alpha is preserved. Bloom and dirt are added in linear RGB and scaled
    /// by source coverage once, so fully transparent pixels stay transparent black.
    public static func lensDirt(dirtSize: SIMD2<Int>, intensity: Float = 1) throws -> PostProcessGraph {
        guard dirtSize.x > 0, dirtSize.y > 0, intensity.isFinite, intensity >= 0 else {
            throw PostProcessEffectError.invalidSettings("Lens dirt needs positive texture dimensions and finite nonnegative intensity.")
        }
        let dirtAspect = Float(dirtSize.x) / Float(dirtSize.y)
        return try PostProcessGraph { p in
            let source = p.sceneColor(at: p.screenUV)
            let bloom = p.texture("Bloom", addressing: .clampToEdge).sample(p.screenUV).rgb
            let aspect = (p.viewportSize.x / p.viewportSize.y) / dirtAspect
            let one = p.constant(Float(1))
            let crop = p.vector(aspect.min(one), (one / aspect).min(one))
            let dirtUV = (p.screenUV - Float(0.5)) * crop + Float(0.5)
            let dirt = p.texture("Dirt", addressing: .clampToEdge).sample(dirtUV).rgb
            let gain = p.uniform("Intensity", intensity).max(p.constant(Float(0)))
            let scattered = bloom + bloom * dirt * gain
            return p.vector(source.rgb + scattered * source.a, source.a)
        }
    }

    /// Three.js `FilmShader`: `color + color · clamp(0.1 + noise, 0, 1)` mixed by
    /// "Intensity", with noise hashed from `fract(screenUV + Time)`, then gray when
    /// "Grayscale" exceeds ½. Change "Time" every frame to animate the grain.
    public static func film(intensity: Float = 0.5, grayscale: Bool = false) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let random = p.function(
                "film_random", returns: Float.self, parameters: [.init("co", .vector2)],
                body: "return fract(sin(dot(co, float2(12.9898f, 78.233f))) * 43758.5453f);"
            )
            let base = p.sceneColor(at: p.screenUV).rgb
            let noise = random((p.screenUV + p.uniform("Time", Float(0))).fract())
            let grained = base + base * (noise + Float(0.1)).clamped(0, 1)
            let color = base.mix(grained, by: p.uniform("Intensity", intensity))
            let luminance = color.dot(p.constant(SIMD3<Float>(0.2125, 0.7154, 0.0721)))
            return p.select(p.uniform("Grayscale", Float(grayscale ? 1 : 0)), p.vector(luminance, luminance, luminance), color)
        }
    }

    /// Three.js `LuminosityHighPassShader`: the source scaled by
    /// `smoothstep(Threshold, Threshold + SmoothWidth, luminance)`, fading to transparent black.
    public static func luminosityHighPass(threshold: Float = 1, smoothWidth: Float = 1) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let color = p.sceneColor(at: p.screenUV)
            let thresholdValue = p.uniform("Threshold", threshold)
            let factor = color.rgb.luminance().smoothstep(thresholdValue, thresholdValue + p.uniform("SmoothWidth", smoothWidth))
            return color * factor
        }
    }

    /// Blends the frame with the previous frame's result held in a texture named
    /// "Previous": `mix(scene, previous, Persistence)` in premultiplied RGBA. Bind
    /// `PostProcessHistory.texture` to "Previous" and capture the result after this stage.
    public static func motionTrail(persistence: Float = 0.8) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let previous = p.texture("Previous", addressing: .clampToEdge, filtering: .nearest).sample(p.screenUV)
            return p.sceneColor(at: p.screenUV).mix(previous, by: p.uniform("Persistence", persistence))
        }
    }
}

/// Per-frame uniform values for the hash and displacement glitch graphs, generated
/// like Three.js `GlitchPass`. Strong bursts are followed by weaker jitter during
/// the first fifth of the interval and then bypass. `goWild` stays active every frame.
public struct GlitchRandomizer: Sendable {
    public var goWild: Bool
    private var state: UInt64
    private var frame = 0
    private var trigger = 0

    /// A deterministic sequence for a seed.
    public init(seed: UInt64 = 0x9E37_79B9_7F4A_7C15, goWild: Bool = false) {
        self.goWild = goWild
        state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        trigger = 120 + Int(nextRandom() * 120)
    }

    private mutating func nextRandom() -> Float {
        Self.nextRandom(state: &state)
    }

    private static func nextRandom(state: inout UInt64) -> Float {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return Float(state >> 40) / Float(1 << 24)
    }

    /// Creates a deterministic square linear R32Float noise map for
    /// `PostProcessGraph.glitchWithDisplacement()`. Its seed is independent of
    /// the frame scheduler. The supported side length is 1 through 1024.
    public static func makeDisplacementTexture(
        device: any MTLDevice, size: Int = 64, seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    ) throws -> any MTLTexture {
        guard (1...1024).contains(size) else {
            throw PostProcessEffectError.invalidSettings("Glitch displacement size must be 1 through 1024.")
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: size, height: size, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw PostProcessEffectError.allocation }
        var state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        let values = (0..<(size * size)).map { _ in nextRandom(state: &state) }
        values.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: size * MemoryLayout<Float>.stride
            )
        }
        texture.label = "Frostfield glitch displacement"
        return texture
    }

    private mutating func random(_ low: Float, _ high: Float) -> Float { low + (high - low) * nextRandom() }

    /// Advances one frame and writes the glitch graph's uniforms.
    public mutating func advance(_ uniforms: ShaderUniforms) throws {
        try uniforms.set("Seed", to: nextRandom())
        try uniforms.set("Bypass", to: Float(0))
        if frame % trigger == 0 || goWild {
            try uniforms.set("Amount", to: nextRandom() / 30)
            try uniforms.set("Angle", to: random(-.pi, .pi))
            try uniforms.set("SeedX", to: random(-1, 1))
            try uniforms.set("SeedY", to: random(-1, 1))
            try uniforms.set("DistortionX", to: random(0, 1))
            try uniforms.set("DistortionY", to: random(0, 1))
            frame = 0
            trigger = 120 + Int(nextRandom() * 120)
        } else if frame % trigger < trigger / 5 {
            try uniforms.set("Amount", to: nextRandom() / 90)
            try uniforms.set("Angle", to: random(-.pi, .pi))
            try uniforms.set("DistortionX", to: random(0, 1))
            try uniforms.set("DistortionY", to: random(0, 1))
            try uniforms.set("SeedX", to: random(-0.3, 0.3))
            try uniforms.set("SeedY", to: random(-0.3, 0.3))
        } else {
            try uniforms.set("Bypass", to: Float(1))
        }
        frame += 1
    }
}
