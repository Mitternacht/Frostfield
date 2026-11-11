import Foundation
import simd

// Control flow, patterns, derivatives, bit operations, dithering and matrices for
// `ShaderGraphBuilder`. Each method records the matching `MaterialGraph.Node`.

/// One iteration's result for `ShaderGraphBuilder.loop(count:initial:until:)`.
public struct ShaderLoopStep<Value: ShaderGraphValue> {
    /// The value the next iteration starts from, and the loop's result after the last one.
    public var next: ShaderValue<Value>
    /// The loop stops after this iteration when the scalar exceeds 0.5.
    public var exit: ShaderValue<Float>

    public init(next: ShaderValue<Value>, exit: ShaderValue<Float>) {
        self.next = next
        self.exit = exit
    }
}

extension ShaderGraphBuilder {
    // MARK: Control flow

    /// Evaluates only the chosen value: `then` where the condition exceeds 0.5, else
    /// `else`. Nodes the closures create that nothing else reads run only in their arm,
    /// as Three.js `If(...).Else(...)`. Texture samples and derivatives always run
    /// outside the arms.
    ///
    /// ```swift
    /// let color = s.branch(s.uv0.x.greaterThan(s.constant(0.5))) {
    ///     s.fractalNoise(s.worldPosition * 8).xxx
    /// } else: {
    ///     s.baseColor
    /// }
    /// ```
    public func branch<T: ShaderGraphValue>(
        _ condition: ShaderValue<Float>,
        then ifTrue: () throws -> ShaderValue<T>,
        else ifFalse: () throws -> ShaderValue<T>
    ) rethrows -> ShaderValue<T> {
        let whenTrue = try ifTrue()
        let whenFalse = try ifFalse()
        return add(
            .branch(condition: condition.id, ifTrue: whenTrue.id, ifFalse: whenFalse.id),
            operands: [condition, whenTrue, whenFalse]
        )
    }

    /// An else-if chain: the value of the first case whose condition exceeds 0.5, or
    /// `otherwise`. Each case becomes a nested branch in the previous one's else arm.
    public func branch<T: ShaderGraphValue>(
        _ cases: [(condition: ShaderValue<Float>, value: ShaderValue<T>)], otherwise: ShaderValue<T>
    ) -> ShaderValue<T> {
        cases.reversed().reduce(otherwise) { fallback, item in
            add(
                .branch(condition: item.condition.id, ifTrue: item.value.id, ifFalse: fallback.id),
                operands: [item.condition, item.value, fallback]
            )
        }
    }

    /// Runs `count` iterations (1...`MaterialGraph.maximumLoopIterations`). The body
    /// receives the zero-based iteration as a scalar and the running value, starting
    /// at `initial`, and returns the next value, as Three.js `Loop` with an accumulator.
    ///
    /// ```swift
    /// // Sum of four octaves of noise with halving amplitude.
    /// let sum = s.loop(count: 4, initial: s.constant(Float(0))) { octave, total in
    ///     let scale = s.constant(Float(2)).pow(octave)
    ///     return total + s.noise(s.worldPosition * scale) / scale
    /// }
    /// ```
    public func loop<T: ShaderGraphValue>(
        count: Int, initial: ShaderValue<T>,
        _ body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderValue<T>
    ) rethrows -> ShaderValue<T> {
        let id = reserve()
        let next = try body(add(.loopIndex(loop: id)), add(.loopValue(loop: id)))
        noteOperands([initial, next])
        fill(id, with: .loop(count: count, initial: initial.id, next: next.id, exit: nil))
        return ShaderValue(id: id, builder: self)
    }

    /// Runs up to `count` iterations and stops after the first iteration whose `exit`
    /// exceeds 0.5, as Three.js `Break()`. Texture samples and derivatives cannot depend
    /// on a loop that exits early; the graph rejects them.
    public func loop<T: ShaderGraphValue>(
        count: Int, initial: ShaderValue<T>,
        until body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderLoopStep<T>
    ) rethrows -> ShaderValue<T> {
        let id = reserve()
        let step = try body(add(.loopIndex(loop: id)), add(.loopValue(loop: id)))
        noteOperands([initial, step.next, step.exit])
        fill(id, with: .loop(count: count, initial: initial.id, next: step.next.id, exit: step.exit.id))
        return ShaderValue(id: id, builder: self)
    }

    // MARK: Oscillators and coordinates

    /// A wave with period 1 and range 0...1 that starts at 0. Pass `time` for Three.js
    /// `oscSine()` and its siblings.
    public func oscillator<T: ShaderValueType>(_ kind: MaterialGraph.Oscillator, _ t: ShaderValue<T>) -> ShaderValue<T> {
        add(.oscillator(kind, t.id), operands: [t])
    }

    /// Rotates coordinates by `angle` radians about `center`, as Three.js `rotateUV`.
    public func rotateUV(
        _ uv: ShaderValue<SIMD2<Float>>, angle: ShaderValue<Float>, center: SIMD2<Float> = SIMD2(0.5, 0.5)
    ) -> ShaderValue<SIMD2<Float>> {
        let pivot = constant(center)
        return add(.rotateUV(uv.id, angle: angle.id, center: pivot.id), operands: [uv, angle, pivot])
    }

    /// Bulges coordinates about `center` by `strength`, as Three.js `spherizeUV`.
    public func spherizeUV(
        _ uv: ShaderValue<SIMD2<Float>>, strength: ShaderValue<Float>, center: SIMD2<Float> = SIMD2(0.5, 0.5)
    ) -> ShaderValue<SIMD2<Float>> {
        let pivot = constant(center)
        return add(.spherizeUV(uv.id, strength: strength.id, center: pivot.id), operands: [uv, strength, pivot])
    }

    /// Distance from `center` in x and the angle in turns, 0...1, in y.
    public func polar(_ uv: ShaderValue<SIMD2<Float>>, center: SIMD2<Float> = SIMD2(0.5, 0.5)) -> ShaderValue<SIMD2<Float>> {
        let pivot = constant(center)
        return add(.polar(uv.id, center: pivot.id), operands: [uv, pivot])
    }

    /// 1 on odd unit cells and 0 on even ones. Three.js `checker(uv)` is `checker(uv * 2)`.
    public func checker(_ coordinates: ShaderValue<SIMD2<Float>>) -> ShaderValue<Float> {
        add(.checker(coordinates.id), operands: [coordinates])
    }

    /// Triplanar weights that sum to 1. Sharpness 1 matches Three.js `triplanarTexture`.
    public func triplanarWeights(normal: ShaderValue<SIMD3<Float>>, sharpness: Float = 1) -> ShaderValue<SIMD3<Float>> {
        let k = constant(sharpness)
        return add(.triplanarWeights(normal: normal.id, sharpness: k.id), operands: [normal, k])
    }

    /// Samples three textures projected along x, y and z, as Three.js
    /// `triplanarTextures(x, y, z, scale, position, normal)`; scale the position first.
    public func triplanar(
        x: ShaderTexture, y: ShaderTexture, z: ShaderTexture,
        position: ShaderValue<SIMD3<Float>>, normal: ShaderValue<SIMD3<Float>>, sharpness: Float = 1
    ) -> ShaderValue<SIMD4<Float>> {
        let k = constant(sharpness)
        if x.builder !== self || y.builder !== self || z.builder !== self {
            // Slots only identify textures in the builder that declared them.
            noteOperands([ForeignArgument()])
        }
        return add(
            .triplanarSample(slots: [x.slot, y.slot, z.slot], position: position.id, normal: normal.id, sharpness: k.id),
            operands: [position, normal, k]
        )
    }

    /// A direction expressed in the surface's UV0 tangent frame: its components along
    /// `tangent` (+U), `bitangent` (+V) and `normal`, so x and y follow the texture
    /// coordinates in both renderers. Surface stage.
    public func tangentSpace(_ direction: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD3<Float>> {
        vector(direction.dot(tangent), direction.dot(bitangent), direction.dot(normal))
    }

    // MARK: Dithering

    /// Jimenez's interleaved gradient noise of pixel coordinates, in 0...1.
    public func interleavedGradientNoise(_ pixel: ShaderValue<SIMD2<Float>>) -> ShaderValue<Float> {
        add(.interleavedGradientNoise(pixel.id), operands: [pixel])
    }

    /// Adds a per-pixel offset of `(ign − 0.5) / levels`, so quantizing the result to
    /// `levels` steps per unit trades banding for unbiased, fine-grained noise.
    public func dither<T: ShaderValueType>(
        _ value: ShaderValue<T>, pixel: ShaderValue<SIMD2<Float>>, levels: Float = 255
    ) -> ShaderValue<T> {
        let offset = (interleavedGradientNoise(pixel) - 0.5) / levels
        let spread: ShaderValue<T> = add(
            .swizzle(offset.id, String(repeating: "x", count: T.shaderType.componentCount)), operands: [offset]
        )
        return value + spread
    }

    /// Adds the r187 four-pixel ordered dither before RGB quantization.
    /// Supply pixel coordinates and a finite positive number of levels.
    /// The default is 32 levels per channel.
    public func bayerDither(
        _ color: ShaderValue<SIMD3<Float>>, pixel: ShaderValue<SIMD2<Float>>,
        levels: ShaderValue<Float>? = nil
    ) -> ShaderValue<SIMD3<Float>> {
        let cell = pixel.floor()
        let tile = cell - (cell / 4).floor() * 4
        let phase = (tile.x + 1) * (tile.y + 1) * (17.0 / 16.0)
        let offset = (phase.fract() - 0.5) / (levels ?? constant(32))
        return color + vector(offset, offset, offset)
    }

    /// Concatenates two vector2 values into a vector4.
    public func vector(_ xy: ShaderValue<SIMD2<Float>>, _ zw: ShaderValue<SIMD2<Float>>) -> ShaderValue<SIMD4<Float>> {
        add(.combine([xy.id, zw.id]), operands: [xy, zw])
    }

    /// Concatenates a vector2 and two scalars into a vector4.
    public func vector(
        _ xy: ShaderValue<SIMD2<Float>>, _ z: ShaderValue<Float>, _ w: ShaderValue<Float>
    ) -> ShaderValue<SIMD4<Float>> {
        add(.combine([xy.id, z.id, w.id]), operands: [xy, z, w])
    }

    // MARK: Matrices

    /// A matrix from three vector3 or four vector4 columns.
    public func matrix(_ c0: ShaderValue<SIMD3<Float>>, _ c1: ShaderValue<SIMD3<Float>>, _ c2: ShaderValue<SIMD3<Float>>)
        -> ShaderValue<simd_float3x3> {
        add(.matrix([c0.id, c1.id, c2.id]), operands: [c0, c1, c2])
    }

    public func matrix(
        _ c0: ShaderValue<SIMD4<Float>>, _ c1: ShaderValue<SIMD4<Float>>,
        _ c2: ShaderValue<SIMD4<Float>>, _ c3: ShaderValue<SIMD4<Float>>
    ) -> ShaderValue<simd_float4x4> {
        add(.matrix([c0.id, c1.id, c2.id, c3.id]), operands: [c0, c1, c2, c3])
    }
}

/// Marks a texture from another builder so the graph initializer rejects it.
private struct ForeignArgument: ShaderArgument {
    var shaderNodeID: MaterialGraph.NodeID { -1 }
}

extension ShaderTexture {
    /// Samples this texture along all three axes, blended by `triplanarWeights`, as
    /// Three.js `triplanarTexture`. Scale the position for the texture's world size.
    public func triplanar(
        position: ShaderValue<SIMD3<Float>>, normal: ShaderValue<SIMD3<Float>>, sharpness: Float = 1
    ) -> ShaderValue<SIMD4<Float>> {
        let k = builder.constant(sharpness)
        return builder.add(
            .triplanarSample(slots: [slot], position: position.id, normal: normal.id, sharpness: k.id),
            operands: [position, normal, k]
        )
    }

    /// Parallax occlusion mapping into this texture's red-channel height field: 1 is the
    /// surface and 0 lies `depth` UV units below. `view` is the direction toward the
    /// viewer in tangent space, such as `s.tangentSpace(s.viewDirection)`. Marches
    /// `steps` layers and returns the coordinates where the view ray meets the height field.
    public func parallaxOcclusion(
        _ coordinates: ShaderValue<SIMD2<Float>>, view: ShaderValue<SIMD3<Float>>,
        depth: ShaderValue<Float>, steps: Int = 16
    ) -> ShaderValue<SIMD2<Float>> {
        builder.add(
            .parallaxOcclusion(slot: slot, coordinates: coordinates.id, view: view.id, depth: depth.id, steps: steps),
            operands: [coordinates, view, depth]
        )
    }
}

extension ShaderValue where Value: ShaderValueType {
    /// A periodic wave of this value. See `MaterialGraph.Oscillator`.
    public func oscillated(_ kind: MaterialGraph.Oscillator) -> Self { make(.oscillator(kind, id)) }
    /// Change per pixel to the right, as Three.js `dFdx`. Raster surface and post-process stages.
    public func dFdx() -> Self { make(.derivative(.dx, id)) }
    /// Change per pixel downward, as Three.js `dFdy`. Raster surface and post-process stages.
    public func dFdy() -> Self { make(.derivative(.dy, id)) }
    /// `|dFdx| + |dFdy|`, as Three.js `fwidth`. Raster surface and post-process stages.
    public func fwidth() -> Self { make(.derivative(.width, id)) }
    /// A 0...1 hash of this value's exact bits.
    public func hash() -> ShaderValue<Float> { make(.hash(id)) }
}

extension ShaderValue where Value == Float {
    /// A bitwise operation on unsigned 24-bit integers. See `MaterialGraph.BitOperation`.
    public func bits(_ operation: MaterialGraph.BitOperation, _ other: Self) -> Self {
        make(.bits(operation, id, other.id), with: other)
    }
}

extension ShaderValue where Value: ShaderMatrixType {
    /// The transpose.
    public func transposed() -> Self { make(.matrixUnary(.transpose, id)) }
    /// The inverse, or zero when the matrix is singular.
    public func inverse() -> Self { make(.matrixUnary(.inverse, id)) }
    /// The determinant.
    public func determinant() -> ShaderValue<Float> { make(.matrixUnary(.determinant, id)) }
    /// One column; validation rejects an index outside the matrix.
    public func column(_ index: Int) -> ShaderValue<Value.Column> { make(.matrixColumn(id, index)) }
}

extension ShaderValue where Value == simd_float4x4 {
    /// The upper-left 3 × 3.
    public var upperLeft: ShaderValue<simd_float3x3> { make(.matrixUnary(.upperLeft, id)) }

    /// Transforms a point with w = 1 and drops w, for affine matrices such as `modelMatrix`.
    public func transformPoint(_ point: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD3<Float>> {
        (self * builder.vector(point, builder.constant(Float(1)))).xyz
    }

    /// Transforms a direction with w = 0.
    public func transformDirection(_ direction: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD3<Float>> {
        (self * builder.vector(direction, builder.constant(Float(0)))).xyz
    }
}

/// The matrix product of two matrices of the same size.
public func * <M: ShaderMatrixType>(left: ShaderValue<M>, right: ShaderValue<M>) -> ShaderValue<M> {
    left.make(.matrixMultiply(left.id, right.id), with: right)
}

/// A matrix times a column vector of matching size.
public func * <M: ShaderMatrixType>(left: ShaderValue<M>, right: ShaderValue<M.Column>) -> ShaderValue<M.Column> {
    left.make(.matrixMultiply(left.id, right.id), with: right)
}
