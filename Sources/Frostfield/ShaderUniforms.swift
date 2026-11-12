import Foundation
import Synchronization
import simd

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

/// The transform a uniform object hook sees for one draw or dispatch.
public struct ShaderObjectContext: Equatable, Sendable {
    public var world: simd_float4x4

    public init(world: simd_float4x4) {
        self.world = world
    }

    public var position: SIMD3<Float> {
        SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
    }
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
///
/// The GPU buffer is one float4 per value slot, in declaration order, then the
/// clock. A scalar or vector is one slot and a matrix is one slot per column.
/// A set of only scalar and vector uniforms therefore has the same layout as
/// one float4 per declaration.
public final class ShaderUniforms: Sendable {
    public let declarations: [ShaderUniform]
    private let indices: [String: Int]
    /// Start of each declaration in the flat slot array.
    private let slotOffsets: [Int]
    /// Float4 slots occupied by the declared values, before the clock.
    let valueSlotCount: Int
    private let state: Mutex<State>
    private let updateHooks = Mutex(UpdateHooks())

    private struct UpdateHooks: Sendable {
        var frame: (@Sendable (ShaderUniforms) -> Void)?
        var object: (@Sendable (ShaderUniforms, ShaderObjectContext) -> Void)?
    }

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
        var offsets: [Int] = []
        var cursor = 0
        for (index, declaration) in declarations.enumerated() {
            guard indices.updateValue(index, forKey: declaration.name) == nil else {
                throw Failure.duplicateName(declaration.name)
            }
            if declaration.type != .int {
                guard declaration.columns.allSatisfy(Self.isFinite) else { throw Failure.nonfinite(declaration.name) }
            }
            offsets.append(cursor)
            cursor += declaration.type.uniformSlotCount
        }
        self.declarations = declarations
        self.indices = indices
        slotOffsets = offsets
        valueSlotCount = cursor
        state = Mutex(State(
            values: declarations.flatMap(\.columns),
            valueRevisions: Array(repeating: 0, count: declarations.count), revision: 0
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

    /// The first slot of a declared uniform, or nil for an unknown name.
    /// Read the remaining columns of a matrix with `matrix3(named:)` or `matrix4(named:)`.
    public func value(named name: String) -> SIMD4<Float>? {
        slots(named: name, type: nil)?.first
    }

    /// The current 3 × 3 uniform, or nil when `name` is unknown or a different type.
    public func matrix3(named name: String) -> simd_float3x3? {
        guard let slots = slots(named: name, type: .matrix3) else { return nil }
        return simd_float3x3(
            SIMD3(slots[0].x, slots[0].y, slots[0].z),
            SIMD3(slots[1].x, slots[1].y, slots[1].z),
            SIMD3(slots[2].x, slots[2].y, slots[2].z)
        )
    }

    /// The current 4 × 4 uniform, or nil when `name` is unknown or a different type.
    public func matrix4(named name: String) -> simd_float4x4? {
        guard let slots = slots(named: name, type: .matrix4) else { return nil }
        return simd_float4x4(columns: (slots[0], slots[1], slots[2], slots[3]))
    }

    public func set(_ name: String, to value: Float) throws {
        try replace(name, with: [SIMD4(value, 0, 0, 0)], type: .scalar)
    }

    public func set(_ name: String, to value: SIMD2<Float>) throws {
        try replace(name, with: [SIMD4(value.x, value.y, 0, 0)], type: .vector2)
    }

    public func set(_ name: String, to value: SIMD3<Float>) throws {
        try replace(name, with: [SIMD4(value, 0)], type: .vector3)
    }

    public func set(_ name: String, to value: SIMD4<Float>) throws {
        try replace(name, with: [value], type: .vector4)
    }

    public func set(_ name: String, to value: simd_float3x3) throws {
        try replace(name, with: value.shaderColumns.map { SIMD4($0, 0) }, type: .matrix3)
    }

    public func set(_ name: String, to value: simd_float4x4) throws {
        try replace(name, with: value.shaderColumns, type: .matrix4)
    }

    public func setInteger(_ name: String, _ value: Int32) throws {
        try replace(name, with: [SIMD4(Float(bitPattern: UInt32(bitPattern: value)), 0, 0, 0)], type: .int)
    }

    public func setBoolean(_ name: String, _ value: Bool) throws {
        try replace(name, with: [SIMD4(value ? 1 : 0, 0, 0, 0)], type: .bool)
    }

    /// The current signed integer uniform, or nil when `name` is unknown or a different type.
    public func integer(named name: String) -> Int32? {
        guard let slots = slots(named: name, type: .int) else { return nil }
        return Int32(bitPattern: slots[0].x.bitPattern)
    }

    /// The current boolean uniform, or nil when `name` is unknown or a different type.
    public func boolean(named name: String) -> Bool? {
        guard let slots = slots(named: name, type: .bool) else { return nil }
        return slots[0].x != 0
    }

    /// Restores every declared default and advances the revision.
    public func reset() {
        state.withLock { state in
            state.values = declarations.flatMap(\.columns)
            state.revision &+= 1
            state.valueRevisions = Array(repeating: state.revision, count: declarations.count)
        }
    }

    /// A consistent copy of every value slot, in declaration order, and their revision.
    public func snapshot() -> (values: [SIMD4<Float>], revision: UInt64) {
        state.withLock { ($0.values, $0.revision) }
    }

    /// The values renderers bind: every value slot, then the clock as
    /// (time, deltaTime, frame, 0), with the revision of the selected values
    /// and the clock's independent revision.
    /// Runs before each uniform snapshot. A raster draw passes the node matrix.
    /// A path dispatch passes the camera matrix. The hook may edit values or the clock.
    /// It must not snapshot these uniforms again.
    public func setFrameUpdate(_ hook: (@Sendable (ShaderUniforms) -> Void)?) {
        updateHooks.withLock { $0.frame = hook }
    }

    /// Runs before each raster draw and path dispatch that reads these uniforms.
    public func setObjectUpdate(_ hook: (@Sendable (ShaderUniforms, ShaderObjectContext) -> Void)?) {
        updateHooks.withLock { $0.object = hook }
    }

    func bufferSnapshot(
        reading mask: UInt32 = .max, object: ShaderObjectContext? = nil
    ) -> (values: [SIMD4<Float>], revision: UInt64, frameRevision: UInt64) {
        let hooks = updateHooks.withLock { $0 }
        hooks.frame?(self)
        if let object { hooks.object?(self, object) }
        return state.withLock { state in
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

    private func slots(named name: String, type: MaterialGraph.ValueType?) -> [SIMD4<Float>]? {
        guard let index = indices[name] else { return nil }
        if let type, declarations[index].type != type { return nil }
        let start = slotOffsets[index]
        let count = declarations[index].type.uniformSlotCount
        return state.withLock { Array($0.values[start..<(start + count)]) }
    }

    private func replace(_ name: String, with columns: [SIMD4<Float>], type: MaterialGraph.ValueType) throws {
        guard let index = indices[name] else { throw Failure.unknownName(name) }
        let declared = declarations[index].type
        guard declared == type else { throw Failure.type(name: name, expected: declared, actual: type) }
        let stored = columns.map { type.storedColumn($0) }
        if type != .int {
            guard stored.allSatisfy(Self.isFinite) else { throw Failure.nonfinite(name) }
        }
        let start = slotOffsets[index]
        state.withLock { state in
            let current = state.values[start..<(start + stored.count)]
            let unchanged = type == .int
                ? current.elementsEqual(stored, by: Self.sameBits)
                : current.elementsEqual(stored)
            guard !unchanged else { return }
            for (offset, column) in stored.enumerated() {
                state.values[start + offset] = column
            }
            state.revision &+= 1
            state.valueRevisions[index] = state.revision
        }
    }

    private static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    private static func sameBits(_ lhs: SIMD4<Float>, _ rhs: SIMD4<Float>) -> Bool {
        lhs.x.bitPattern == rhs.x.bitPattern && lhs.y.bitPattern == rhs.y.bitPattern
            && lhs.z.bitPattern == rhs.z.bitPattern && lhs.w.bitPattern == rhs.w.bitPattern
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
