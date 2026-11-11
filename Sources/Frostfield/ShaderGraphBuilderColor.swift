import simd

extension ShaderGraphBuilder {
    /// Screen blends each RGB channel in the supplied working space.
    /// Values pass through without color conversion or clamping.
    public func blendScreen(
        _ base: ShaderValue<SIMD3<Float>>, _ blend: ShaderValue<SIMD3<Float>>
    ) -> ShaderValue<SIMD3<Float>> {
        (base.oneMinus() * blend.oneMinus()).oneMinus()
    }

    /// Overlay multiplies the channels below a base of 0.5 and screens those
    /// at or above it. Values stay in the supplied working space.
    public func blendOverlay(
        _ base: ShaderValue<SIMD3<Float>>, _ blend: ShaderValue<SIMD3<Float>>
    ) -> ShaderValue<SIMD3<Float>> {
        let dark = base * 2 * blend
        let light = (base.oneMinus() * 2 * blend.oneMinus()).oneMinus()
        let threshold = constant(Float(0.5))
        return vector(
            select(base.x.lessThan(threshold), dark.x, light.x),
            select(base.y.lessThan(threshold), dark.y, light.y),
            select(base.z.lessThan(threshold), dark.z, light.z)
        )
    }

    /// Applies the r187 ACES filmic curve to linear-sRGB input. Exposure is a
    /// linear multiplier and defaults to 1. The linear-sRGB result is clamped
    /// to 0...1 after the output matrix.
    public func acesFilmicToneMapping(
        _ color: ShaderValue<SIMD3<Float>>, exposure: ShaderValue<Float>? = nil
    ) -> ShaderValue<SIMD3<Float>> {
        // The same Stephen Hill fit and matrix columns used by HDRDisplay.
        let input = constant(simd_float3x3(columns: (
            SIMD3(0.59719, 0.07600, 0.02840),
            SIMD3(0.35458, 0.90834, 0.13383),
            SIMD3(0.04823, 0.01566, 0.83777)
        )))
        let output = constant(simd_float3x3(columns: (
            SIMD3(1.60475, -0.10208, -0.00327),
            SIMD3(-0.53108, 1.10813, -0.07276),
            SIMD3(-0.07367, -0.00605, 1.07602)
        )))
        let exposed = color * (exposure ?? constant(Float(1))) / 0.6
        let working = input * exposed
        let numerator = working * (working + 0.0245786) - 0.000090537
        let denominator = working * (working * 0.983729 + 0.432951) + 0.238081
        return (output * divideChannels(numerator, by: denominator)).saturated()
    }

    /// Floors each component after scaling by `steps`, then divides by
    /// `steps`. The input can be signed or HDR. A zero divisor gives zero.
    public func posterize<T: ShaderValueType>(
        _ source: ShaderValue<T>, steps: ShaderValue<Float>
    ) -> ShaderValue<T> {
        let scaled: ShaderValue<T> = add(.multiply(source.id, steps.id), operands: [source, steps])
        return scaled.floor() / steps
    }

    /// Posterizes a vector with independent step counts per component.
    public func posterize<V: ShaderVectorType>(
        _ source: ShaderValue<V>, steps: ShaderValue<V>
    ) -> ShaderValue<V> {
        divideChannels((source * steps).floor(), by: steps)
    }

    /// Expands a scalar source and posterizes it at independent step counts.
    public func posterize<V: ShaderVectorType>(
        _ source: ShaderValue<Float>, steps: ShaderValue<V>
    ) -> ShaderValue<V> {
        posterize(steps.splat(source), steps: steps)
    }

    /// Applies the r187 color decision list to straight RGBA in the supplied
    /// working space, typically log-encoded color. Luma comes from the original
    /// RGB. The final RGB has a zero floor, and alpha is preserved.
    public func cdl(
        _ color: ShaderValue<SIMD4<Float>>,
        slope: ShaderValue<SIMD3<Float>>? = nil,
        offset: ShaderValue<SIMD3<Float>>? = nil,
        power: ShaderValue<SIMD3<Float>>? = nil,
        saturation: ShaderValue<Float>? = nil,
        luminanceCoefficients: ShaderValue<SIMD3<Float>>? = nil
    ) -> ShaderValue<SIMD4<Float>> {
        let rgb = color.rgb
        let one = constant(SIMD3<Float>(repeating: 1))
        let zero = constant(SIMD3<Float>.zero)
        let luma = luminanceCoefficients.map { rgb.dot($0) } ?? rgb.luminance()
        let pregraded = (rgb * (slope ?? one) + (offset ?? zero)).max(zero)
        // Substitute one for zero bases before pow and restore those channels to zero.
        let positive = pregraded.greaterThan(zero)
        let powered = (pregraded + positive.oneMinus()).pow(power ?? one) * positive
        let saturated = (powered - luma) * (saturation ?? constant(Float(1))) + luma
        return vector(saturated.max(zero), color.a)
    }

    private func divideChannels<V: ShaderVectorType>(
        _ numerator: ShaderValue<V>, by denominator: ShaderValue<V>
    ) -> ShaderValue<V> {
        let quotients = ["x", "y", "z", "w"].prefix(V.shaderType.componentCount).map { component in
            numerator.swizzle(component, as: Float.self) / denominator.swizzle(component, as: Float.self)
        }
        return add(.combine(quotients.map(\.id)), operands: quotients)
    }
}
