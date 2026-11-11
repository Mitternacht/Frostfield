import Foundation
import Synchronization

/// The clock a graph's `time`, `deltaTime` and `frame` inputs read.
public struct ShaderFrame: Hashable, Sendable {
    /// Seconds since the clock started. Graphs read it as a 32-bit float.
    public var time: Double
    /// Seconds since the previous frame.
    public var deltaTime: Float
    /// Frames since the clock started. Graphs read it modulo 2²⁴, exact as a float.
    public var index: UInt64

    public init(time: Double = 0, deltaTime: Float = 0, index: UInt64 = 0) {
        self.time = time
        self.deltaTime = deltaTime
        self.index = index
    }

    /// The float4 slot renderers bind after the declared uniforms.
    var packed: SIMD4<Float> { SIMD4(Float(time), deltaTime, Float(index & 0xFF_FFFF), 0) }
}

/// Live values for a graph's declared uniforms and its clock.
///
/// Materials and passes keep a reference to this object and copy its current
/// values into each encoded draw or dispatch, so editing a value never rebuilds
/// a pipeline. Several materials may share one instance. Every successful change
/// to a declared value advances `revision`, and every clock change advances
/// `frameRevision`. Progressive accumulation tracks the uniforms used by
/// traced surface shading. Vertex deformation also watches its graph clock
/// when the vertex stage reads it.
public final class ShaderUniforms: Sendable {
    public let declarations: [ShaderUniform]
    private let indices: [String: Int]
    private let state: Mutex<State>

    private struct State {
        var values: [SIMD4<Float>]
        var valueRevisions: [UInt64]
        var revision: UInt64
        var frame = ShaderFrame()
        var frameRevision: UInt64 = 0
    }

    /// Starts every uniform at its declared default.
    public init(_ declarations: [ShaderUniform]) throws {
        guard declarations.count <= MaterialGraph.maximumUniformCount else {
            throw Failure.tooManyUniforms(limit: MaterialGraph.maximumUniformCount)
        }
        var indices: [String: Int] = [:]
        for (index, declaration) in declarations.enumerated() {
            guard indices.updateValue(index, forKey: declaration.name) == nil else {
                throw Failure.duplicateName(declaration.name)
            }
            guard Self.isFinite(declaration.defaultValue) else { throw Failure.nonfinite(declaration.name) }
        }
        self.declarations = declarations
        self.indices = indices
        state = Mutex(State(
            values: declarations.map(\.defaultValue), valueRevisions: Array(repeating: 0, count: declarations.count), revision: 0
        ))
    }

    /// Starts every uniform the graph declares at its default.
    public convenience init(graph: SurfaceShaderGraph) throws {
        try self.init(graph.uniforms)
    }

    /// Increases after every change to a declared value. Equal revisions from one
    /// instance mean equal declared values; the clock has its own `frameRevision`.
    public var revision: UInt64 { state.withLock { $0.revision } }

    /// Increases after every clock change. Equal frame revisions from one instance mean equal clocks.
    public var frameRevision: UInt64 { state.withLock { $0.frameRevision } }

    /// The current value in the leading components of a float4, or nil for an unknown name.
    public func value(named name: String) -> SIMD4<Float>? {
        guard let index = indices[name] else { return nil }
        return state.withLock { $0.values[index] }
    }

    public func set(_ name: String, to value: Float) throws {
        try set(name, value: SIMD4(value, 0, 0, 0), type: .scalar)
    }

    public func set(_ name: String, to value: SIMD2<Float>) throws {
        try set(name, value: SIMD4(value.x, value.y, 0, 0), type: .vector2)
    }

    public func set(_ name: String, to value: SIMD3<Float>) throws {
        try set(name, value: SIMD4(value, 0), type: .vector3)
    }

    public func set(_ name: String, to value: SIMD4<Float>) throws {
        try set(name, value: value, type: .vector4)
    }

    /// Restores every declared default and advances the revision.
    public func reset() {
        state.withLock { state in
            state.values = declarations.map(\.defaultValue)
            state.revision &+= 1
            state.valueRevisions = Array(repeating: state.revision, count: declarations.count)
        }
    }

    /// A consistent copy of all values, one float4 per declaration, and their revision.
    public func snapshot() -> (values: [SIMD4<Float>], revision: UInt64) {
        state.withLock { ($0.values, $0.revision) }
    }

    /// The values renderers bind: one float4 per declaration, then the clock as
    /// (time, deltaTime, frame, 0), with the revision of the selected values
    /// and the clock's independent revision.
    func bufferSnapshot(reading mask: UInt32 = .max) -> (values: [SIMD4<Float>], revision: UInt64, frameRevision: UInt64) {
        state.withLock { state in
            let revision: UInt64
            if mask == .max {
                revision = state.revision
            } else {
                revision = state.valueRevisions.enumerated().reduce(0) { last, entry in
                    let (index, stamp) = entry
                    return mask & (UInt32(1) << index) != 0 ? max(last, stamp) : last
                }
            }
            return (state.values + [state.frame.packed], revision, state.frameRevision)
        }
    }

    /// The revision a consumer compares to decide whether its output is stale: the
    /// declared values' revision, and the clock's when the consumer's graph reads it.
    func revision(readingClock: Bool) -> ShaderUniformRevision {
        state.withLock { ShaderUniformRevision(values: $0.revision, frame: readingClock ? $0.frameRevision : nil) }
    }

    /// The clock that `time`, `deltaTime` and `frame` graph inputs read.
    public var frame: ShaderFrame { state.withLock { $0.frame } }

    /// Sets the clock. Time and delta time must be finite.
    public func setFrame(_ frame: ShaderFrame) throws {
        guard frame.time.isFinite, frame.deltaTime.isFinite else { throw Failure.nonfinite("frame") }
        state.withLock { state in
            guard state.frame != frame else { return }
            state.frame = frame
            state.frameRevision &+= 1
        }
    }

    /// Advances the clock by one frame of `deltaTime` seconds.
    public func advanceFrame(by deltaTime: Float) throws {
        guard deltaTime.isFinite else { throw Failure.nonfinite("frame") }
        state.withLock { state in
            state.frame = ShaderFrame(
                time: state.frame.time + Double(deltaTime), deltaTime: deltaTime, index: state.frame.index &+ 1
            )
            state.frameRevision &+= 1
        }
    }

    /// Whether these values satisfy a graph's declarations in order and type.
    public func matches(_ declarations: [ShaderUniform]) -> Bool {
        declarations.count == self.declarations.count
            && zip(declarations, self.declarations).allSatisfy { $0.name == $1.name && $0.type == $1.type }
    }

    private func set(_ name: String, value: SIMD4<Float>, type: MaterialGraph.ValueType) throws {
        guard let index = indices[name] else { throw Failure.unknownName(name) }
        let declared = declarations[index].type
        guard declared == type else { throw Failure.type(name: name, expected: declared, actual: type) }
        guard Self.isFinite(value) else { throw Failure.nonfinite(name) }
        state.withLock { state in
            guard state.values[index] != value else { return }
            state.values[index] = value
            state.revision &+= 1
            state.valueRevisions[index] = state.revision
        }
    }

    private static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    public enum Failure: LocalizedError, Equatable {
        case tooManyUniforms(limit: Int)
        case duplicateName(String)
        case unknownName(String)
        case type(name: String, expected: MaterialGraph.ValueType, actual: MaterialGraph.ValueType)
        case nonfinite(String)

        public var errorDescription: String? {
            switch self {
            case .tooManyUniforms(let limit): "A shader may declare at most \(limit) uniforms."
            case .duplicateName(let name): "Shader uniform \"\(name)\" is declared more than once."
            case .unknownName(let name): "The shader does not declare a uniform named \"\(name)\"."
            case .type(let name, let expected, let actual):
                "Shader uniform \"\(name)\" is a \(expected.rawValue), not a \(actual.rawValue)."
            case .nonfinite(let name): "Shader uniform \"\(name)\" needs finite values."
            }
        }
    }
}

/// A consumer's view of a `ShaderUniforms` revision: the declared values' revision,
/// plus the clock's when the consumer reads the clock.
struct ShaderUniformRevision: Hashable, Sendable {
    var values: UInt64
    var frame: UInt64?
}
