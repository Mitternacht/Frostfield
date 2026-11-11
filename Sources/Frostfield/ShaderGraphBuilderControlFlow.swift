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
    /// When this exceeds 0.5 the iteration keeps the accumulator, as Three.js `Continue()`.
    /// Nodes that only `next` reads do not run on a continued iteration.
    public var skip: ShaderValue<Float>?

    public init(next: ShaderValue<Value>, exit: ShaderValue<Float>, skip: ShaderValue<Float>? = nil) {
        self.next = next
        self.exit = exit
        self.skip = skip
    }
}

extension ShaderGraphBuilder {
    // MARK: Control flow

    /// Evaluates only the chosen value: `then` where the condition exceeds 0.5, else
    /// `else`. Nodes the closures create that nothing else reads run only in their arm,
    /// as Three.js `If(...).Else(...)`. Implicit-LOD texture samples and derivatives
    /// always run outside the arms. An explicit mip level or explicit UV derivatives
    /// may stay inside the arm that reads them.
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

    /// Like `branch(_:then:else:)`, with a boolean condition instead of a float above 0.5.
    public func branch<T: ShaderGraphValue>(
        _ condition: ShaderValue<Bool>,
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

    /// The value of the first case whose match equals `selector`, otherwise the fallback.
    /// Only the chosen arm's private nodes run, as Three.js `Switch`. Equality is exact.
    ///
    /// ```swift
    /// let color = s.switch(s.uniform("Kind", Float(0)), cases: [
    ///     (match: s.constant(0), value: { s.constant(SIMD3<Float>(1, 0, 0)) }),
    ///     (match: s.constant(1), value: { s.baseColor }),
    /// ], otherwise: { s.constant(SIMD3<Float>(0, 0, 1)) })
    /// ```
    public func `switch`<S: ShaderValueType, T: ShaderGraphValue>(
        _ selector: ShaderValue<S>,
        cases: [(match: ShaderValue<S>, value: () throws -> ShaderValue<T>)],
        otherwise: () throws -> ShaderValue<T>
    ) throws -> ShaderValue<T> {
        guard (1...MaterialGraph.maximumSwitchCases).contains(cases.count) else {
            rejectDeclaration("A switch needs 1–\(MaterialGraph.maximumSwitchCases) cases.")
            return try otherwise()
        }
        var matches: [MaterialGraph.NodeID] = []
        var values: [MaterialGraph.NodeID] = []
        var operands: [any ShaderArgument] = [selector]
        for item in cases {
            matches.append(item.match.id)
            operands.append(item.match)
            let value = try item.value()
            values.append(value.id)
            operands.append(value)
        }
        let fallback = try otherwise()
        operands.append(fallback)
        return add(
            .switchSelect(selector: selector.id, matches: matches, values: values, fallback: fallback.id),
            operands: operands
        )
    }

    /// Yields `value`, and discards the fragment where `condition` exceeds 0.5.
    /// Raster coverage and the path tracer both reject that fragment. Surface stage.
    public func discard<T: ShaderGraphValue>(_ condition: ShaderValue<Float>, _ value: ShaderValue<T>) -> ShaderValue<T> {
        add(.discard(condition: condition.id, value: value.id), operands: [condition, value])
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
        try makeLoop(count: count, iterations: nil, initial: initial, body)
    }

    /// Like `loop(count:initial:_:)`, but the iteration count is `iterations` truncated
    /// toward zero and clamped to 0...`count`. A nonfinite count runs no iterations.
    /// `count` is the most the loop may run, so the graph can still bound its work.
    public func loop<T: ShaderGraphValue>(
        count: Int, iterations: ShaderValue<Float>, initial: ShaderValue<T>,
        _ body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderValue<T>
    ) rethrows -> ShaderValue<T> {
        try makeLoop(count: count, iterations: iterations, initial: initial, body)
    }

    private func makeLoop<T: ShaderGraphValue>(
        count: Int, iterations: ShaderValue<Float>?, initial: ShaderValue<T>,
        _ body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderValue<T>
    ) rethrows -> ShaderValue<T> {
        let id = reserve()
        let next = try body(add(.loopIndex(loop: id)), add(.loopValue(loop: id)))
        var operands: [any ShaderArgument] = [initial, next]
        if let iterations { operands.append(iterations) }
        noteOperands(operands)
        fill(id, with: .loop(
            count: count, iterations: iterations?.id, initial: initial.id, next: next.id, exit: nil, skip: nil
        ))
        return ShaderValue(id: id, builder: self)
    }

    /// Runs up to `count` iterations and stops after the first iteration whose `exit`
    /// exceeds 0.5, as Three.js `Break()`. Set `ShaderLoopStep.skip` to continue an
    /// iteration without storing `next`. Texture samples and derivatives cannot depend
    /// on a loop that exits early; the graph rejects them.
    public func loop<T: ShaderGraphValue>(
        count: Int, initial: ShaderValue<T>,
        until body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderLoopStep<T>
    ) rethrows -> ShaderValue<T> {
        try makeExitingLoop(count: count, iterations: nil, initial: initial, until: body)
    }

    /// The early-exit loop with a dynamic iteration count. See `loop(count:iterations:initial:_:)`.
    public func loop<T: ShaderGraphValue>(
        count: Int, iterations: ShaderValue<Float>, initial: ShaderValue<T>,
        until body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderLoopStep<T>
    ) rethrows -> ShaderValue<T> {
        try makeExitingLoop(count: count, iterations: iterations, initial: initial, until: body)
    }

    private func makeExitingLoop<T: ShaderGraphValue>(
        count: Int, iterations: ShaderValue<Float>?, initial: ShaderValue<T>,
        until body: (_ index: ShaderValue<Float>, _ value: ShaderValue<T>) throws -> ShaderLoopStep<T>
    ) rethrows -> ShaderValue<T> {
        let id = reserve()
        let step = try body(add(.loopIndex(loop: id)), add(.loopValue(loop: id)))
        var operands: [any ShaderArgument] = [initial, step.next, step.exit]
        if let iterations { operands.append(iterations) }
        if let skip = step.skip { operands.append(skip) }
        noteOperands(operands)
        fill(id, with: .loop(
            count: count, iterations: iterations?.id, initial: initial.id, next: step.next.id,
            exit: step.exit.id, skip: step.skip?.id
        ))
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
    /// Reinterprets these bits as a signed 32-bit integer.
    public func bitcast() -> ShaderValue<Int32> { make(.bitcast(id)) }
    /// The number of set bits in this value's representation.
    public func countOneBits() -> ShaderValue<Int32> { make(.countOneBits(id)) }
    /// Truncates toward zero. A nonfinite value becomes zero.
    public func toInt() -> ShaderValue<Int32> { make(.numericCast(id)) }
}

extension ShaderValue where Value == Int32 {
    /// Reinterprets these bits as a float.
    public func bitcast() -> ShaderValue<Float> { make(.bitcast(id)) }
    /// The number of set bits in this two's-complement value.
    public func countOneBits() -> ShaderValue<Int32> { make(.countOneBits(id)) }
    /// The nearest float, which is exact through ±16,777,216.
    public func toFloat() -> ShaderValue<Float> { make(.numericCast(id)) }
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

public func / (left: ShaderValue<Int32>, right: ShaderValue<Int32>) -> ShaderValue<Int32> {
    left.make(.divide(left.id, right.id), with: right)
}

/// A matrix times a column vector of matching size.
public func * <M: ShaderMatrixType>(left: ShaderValue<M>, right: ShaderValue<M.Column>) -> ShaderValue<M.Column> {
    left.make(.matrixMultiply(left.id, right.id), with: right)
}

/// A value of a declared struct. Read fields with `field`.
public struct ShaderStructValue: ShaderArgument {
    public let id: MaterialGraph.NodeID
    /// Index of the struct declaration this value has.
    public let structure: Int
    let builder: ShaderGraphBuilder

    public var shaderNodeID: MaterialGraph.NodeID { id }

    /// A scalar, vector, matrix, int, or bool field.
    public func field<T: ShaderValueType>(_ name: String, as type: T.Type = T.self) -> ShaderValue<T> {
        let field = fieldIndex(named: name)
        if let field, builder.structs.indices.contains(structure) {
            switch builder.structs[structure].fields[field].content {
            case .value(let declared) where declared != T.shaderType:
                builder.rejectDeclaration("Field \(name) is \(declared.rawValue), not \(T.shaderType.rawValue).")
            case .structure:
                builder.rejectDeclaration("Field \(name) is a struct. Read it by passing that struct to field(_:as:).")
            case .value:
                break
            }
        }
        return builder.add(.structField(value: id, field: field ?? 0), operands: [self])
    }

    /// A field that is itself a struct.
    public func field(_ name: String, as structure: ShaderStruct) -> ShaderStructValue {
        let field = fieldIndex(named: name)
        var nested = structure.index
        if let field, builder.structs.indices.contains(self.structure) {
            if case .structure(let index) = builder.structs[self.structure].fields[field].content, index == structure.index {
                nested = index
            } else {
                builder.rejectDeclaration("Field \(name) is not struct \(structure.name).")
            }
        }
        builder.noteOperands([self])
        return ShaderStructValue(
            id: builder.append(.structField(value: id, field: field ?? 0)), structure: nested, builder: builder
        )
    }

    private func fieldIndex(named name: String) -> Int? {
        guard builder.structs.indices.contains(structure) else {
            builder.rejectDeclaration("Struct \(structure) is not declared on this graph.")
            return nil
        }
        guard let index = builder.structs[structure].fields.firstIndex(where: { $0.name == name }) else {
            builder.rejectDeclaration("Struct \(builder.structs[structure].name) has no field \(name).")
            return nil
        }
        return index
    }
}

extension ShaderStructValue: ShaderBuilderOwned {
    var owner: ShaderGraphBuilder { builder }
}

extension ShaderGraphBuilder {
    /// Declares a struct. Nested fields must name a struct this builder already declared.
    @discardableResult
    public func defineStruct(_ name: String, _ fields: [ShaderStruct.Field]) -> ShaderStruct {
        let structure = ShaderStruct(index: structs.count, name: name, fields: fields)
        addStruct(structure)
        return structure
    }

    /// Builds a struct from one value per field name.
    public func make(_ structure: ShaderStruct, _ values: [String: any ShaderArgument]) -> ShaderStructValue {
        noteOperands(Array(values.values))
        if !(structs.indices.contains(structure.index) && structs[structure.index] == structure) {
            rejectDeclaration("Struct \(structure.name) belongs to a different graph.")
        }
        var ordered: [MaterialGraph.NodeID] = []
        var seen = Set<String>()
        for field in structure.fields {
            guard let value = values[field.name] else {
                rejectDeclaration("Struct \(structure.name) is missing field \(field.name).")
                ordered.append(0)
                continue
            }
            seen.insert(field.name)
            ordered.append(value.shaderNodeID)
        }
        if values.count != seen.count {
            rejectDeclaration("Struct \(structure.name) received a field it does not declare.")
        }
        return ShaderStructValue(
            id: append(.constructStruct(index: structure.index, fields: ordered)),
            structure: structure.index, builder: self
        )
    }

    /// Chooses one struct value. Both arms have to be the same struct.
    public func branch(
        _ condition: ShaderValue<Float>,
        then ifTrue: () throws -> ShaderStructValue,
        else ifFalse: () throws -> ShaderStructValue
    ) rethrows -> ShaderStructValue {
        try structBranch(condition, then: ifTrue, else: ifFalse)
    }

    /// Like `branch(_:then:else:)` for structs, with a boolean condition.
    public func branch(
        _ condition: ShaderValue<Bool>,
        then ifTrue: () throws -> ShaderStructValue,
        else ifFalse: () throws -> ShaderStructValue
    ) rethrows -> ShaderStructValue {
        try structBranch(condition, then: ifTrue, else: ifFalse)
    }

    private func structBranch<Condition: ShaderArgument>(
        _ condition: Condition,
        then ifTrue: () throws -> ShaderStructValue,
        else ifFalse: () throws -> ShaderStructValue
    ) rethrows -> ShaderStructValue {
        let whenTrue = try ifTrue()
        let whenFalse = try ifFalse()
        if whenTrue.structure != whenFalse.structure {
            rejectDeclaration("The arms of a struct branch must be the same struct.")
        }
        noteOperands([condition, whenTrue, whenFalse])
        return ShaderStructValue(
            id: append(.branch(
                condition: condition.shaderNodeID, ifTrue: whenTrue.id, ifFalse: whenFalse.id
            )),
            structure: whenTrue.structure, builder: self
        )
    }
}
