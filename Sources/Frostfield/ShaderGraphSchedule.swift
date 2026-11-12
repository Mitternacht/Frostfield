import Foundation

/// Places every node reachable from a set of outputs into the scope where the
/// generated Metal evaluates it: the function body, one arm of a `.branch`, or the
/// body of a `.loop`.
///
/// The rules, applied in dependency order so the result is deterministic:
/// - A node that reads a loop's `loopIndex` or `loopValue`, directly or through
///   other nodes, belongs to that loop's body. Loops nest by the same rule, and a
///   value from inside a loop may only reach the outside through that loop's
///   `next` or `exit`.
/// - Every other node sinks to the deepest scope that contains all of its uses,
///   so a node that only one branch arm reads runs only in that arm, but it is
///   never moved into a loop body it does not depend on: loop-invariant work runs
///   once, before the loop.
/// - Texture samples, derivatives, parallax and triplanar nodes, calls to declared
///   functions whose bodies take derivatives or cross-lane values, and loops that
///   contain any of them, are never moved into branch arms, and may not depend on
///   a loop that can exit early, so implicit screen-space derivatives stay defined.
/// - One evaluation may run at most `MaterialGraph.maximumTotalIterations` loop
///   iterations and parallax layers, counting each loop's count times the counts
///   of the loops around it.
struct ShaderSchedule {
    typealias NodeID = MaterialGraph.NodeID
    typealias Failure = MaterialGraph.Failure

    enum ScopeKind: Equatable {
        case root
        case branchArm
        case loopBody
    }

    struct Scope {
        let kind: ScopeKind
        let parent: Int
        let depth: Int
        /// How many times one evaluation runs this scope at most: the product of the
        /// counts of the loops around it.
        let repetitions: Int
    }

    private(set) var scopes: [Scope] = [Scope(kind: .root, parent: 0, depth: 0, repetitions: 1)]
    private(set) var trueScope: [NodeID: Int] = [:]
    private(set) var falseScope: [NodeID: Int] = [:]
    private(set) var bodyScope: [NodeID: Int] = [:]
    /// The arm that stores a loop's next value. Absent when the loop has no continue.
    private(set) var updateScope: [NodeID: Int] = [:]
    /// One scope per switch arm, with the fallback last.
    private(set) var caseScopes: [NodeID: [Int]] = [:]
    /// Reachable nodes per scope, in dependency order.
    private var members: [Int: [NodeID]] = [:]

    func nodes(in scope: Int) -> [NodeID] { members[scope] ?? [] }

    init(program: ShaderProgram, outputs: [NodeID]) throws {
        let nodes = program.nodes
        let order = program.order(for: outputs)

        // Loops whose iteration each node reads without passing through the loop itself.
        var free = [[NodeID]](repeating: [], count: nodes.count)
        var loops: [NodeID] = []
        for id in order {
            switch nodes[id] {
            case .loopIndex(let loop), .loopValue(let loop):
                free[id] = [loop]
            case .loop(_, let iterations, let initial, let next, let exit, let skip):
                if let loop = free[initial].first(where: { $0 == id }) {
                    throw Failure.loopValueEscapes(node: initial, loop: loop)
                }
                if let iterations, let loop = free[iterations].first(where: { $0 == id }) {
                    throw Failure.loopValueEscapes(node: iterations, loop: loop)
                }
                var body = free[next]
                if let exit { body = Self.union(body, free[exit]) }
                if let skip { body = Self.union(body, free[skip]) }
                var outside = free[initial]
                if let iterations { outside = Self.union(outside, free[iterations]) }
                free[id] = Self.union(outside, body.filter { $0 != id })
                loops.append(id)
            default:
                free[id] = ShaderProgram.inputs(of: nodes[id]).reduce(into: []) { $0 = Self.union($0, free[$1]) }
            }
        }
        for output in outputs {
            if let loop = free[output].first { throw Failure.loopValueEscapes(node: output, loop: loop) }
        }

        // Loop K encloses loop L when L reads K's iteration, directly or through loops
        // L reads. A loop that encloses itself mixes values from two loop bodies.
        var enclosing: [NodeID: [NodeID]] = [:]
        for loop in loops { enclosing[loop] = free[loop] }
        var changed = true
        while changed {
            changed = false
            for loop in loops {
                let current = enclosing[loop]!
                let expanded = current.reduce(current) { Self.union($0, enclosing[$1] ?? []) }
                if expanded != current {
                    enclosing[loop] = expanded
                    changed = true
                }
            }
        }
        for loop in loops where enclosing[loop]!.contains(loop) {
            throw Failure.loopValueEscapes(node: loop, loop: loop)
        }
        // The innermost loop a node must run inside: the loop it reads that every other
        // loop it reads encloses.
        var required = [NodeID?](repeating: nil, count: nodes.count)
        for id in order where !free[id].isEmpty {
            guard let inner = free[id].first(where: { candidate in
                free[id].allSatisfy { $0 == candidate || enclosing[candidate]!.contains($0) }
            }) else {
                throw Failure.loopValueEscapes(node: id, loop: free[id][0])
            }
            required[id] = inner
        }

        // Nodes whose implicit derivatives need every pixel of a quad to run them together.
        let quadFunctions = Self.quadDependentFunctions(program.declarations.functions)
        var uniform = [Bool](repeating: false, count: nodes.count)
        var bodyIsUniform: Set<NodeID> = []
        for id in order {
            switch nodes[id] {
            case .sampleTexture, .triplanarSample, .parallaxOcclusion, .bumpMap, .bumpHeight, .derivative: uniform[id] = true
            case .function(let index, _): uniform[id] = quadFunctions.indices.contains(index) && quadFunctions[index]
            case .loop: uniform[id] = bodyIsUniform.contains(id)
            default: break
            }
            guard uniform[id], let loop = required[id] else { continue }
            if case .loop(_, _, _, _, let exit?, _) = nodes[loop] {
                _ = exit
                throw Failure.divergentDerivative(node: id, loop: loop)
            }
            bodyIsUniform.insert(loop)
        }

        // Uses of each node: the scope of an ordinary consumer, a branch arm, or a loop body.
        var uses = [[UseSite]](repeating: [], count: nodes.count)
        for consumer in order {
            switch nodes[consumer] {
            case .switchSelect(let selector, let matches, let values, let fallback):
                uses[selector].append(.scope(of: consumer))
                for match in matches { uses[match].append(.scope(of: consumer)) }
                for (index, value) in values.enumerated() { uses[value].append(.switchArm(consumer, index)) }
                uses[fallback].append(.switchArm(consumer, values.count))
            case .discard(let condition, let value):
                uses[condition].append(.scope(of: consumer))
                uses[value].append(.scope(of: consumer))
            case .loop(_, let iterations, let initial, let next, let exit, let skip):
                uses[initial].append(.scope(of: consumer))
                if let iterations { uses[iterations].append(.scope(of: consumer)) }
                uses[next].append(skip == nil ? .body(consumer) : .update(consumer))
                if let exit { uses[exit].append(.body(consumer)) }
                if let skip { uses[skip].append(.body(consumer)) }
            default:
                for (position, input) in ShaderProgram.inputs(of: nodes[consumer]).enumerated() {
                    let site: UseSite
                    switch nodes[consumer] {
                    case .branch where position == 1: site = .trueArm(consumer)
                    case .branch where position == 2: site = .falseArm(consumer)
                    default: site = .scope(of: consumer)
                    }
                    uses[input].append(site)
                }
            }
        }
        let roots = Set(outputs)

        var home = [Int](repeating: 0, count: nodes.count)
        var iterations = 0
        for id in order.reversed() {
            var scopesOfUse: [Int] = roots.contains(id) ? [0] : []
            for site in uses[id] {
                switch site {
                case .scope(let consumer): scopesOfUse.append(home[consumer])
                case .trueArm(let branch): scopesOfUse.append(trueScope[branch]!)
                case .falseArm(let branch): scopesOfUse.append(falseScope[branch]!)
                case .body(let loop): scopesOfUse.append(bodyScope[loop]!)
                case .update(let loop): scopesOfUse.append(updateScope[loop]!)
                case .switchArm(let selection, let index): scopesOfUse.append(caseScopes[selection]![index])
                }
            }
            let deepest = scopesOfUse.dropFirst().reduce(scopesOfUse.first ?? 0) { lowestCommonAncestor($0, $1) }
            let floor = required[id].map { bodyScope[$0]! } ?? 0
            // Walk from the required scope toward the uses, stopping before any loop
            // body (invariant work stays outside) and, for derivative work, any branch arm.
            var path: [Int] = []
            var scope = deepest
            while scope != floor {
                guard scope != 0 else { throw Failure.loopValueEscapes(node: id, loop: required[id] ?? id) }
                path.append(scope)
                scope = scopes[scope].parent
            }
            var placed = floor
            for candidate in path.reversed() {
                let kind = scopes[candidate].kind
                if kind == .loopBody || (uniform[id] && kind == .branchArm) { break }
                placed = candidate
            }
            home[id] = placed
            members[placed, default: []].append(id)

            switch nodes[id] {
            case .branch:
                trueScope[id] = try addScope(.branchArm, parent: placed, owner: id)
                falseScope[id] = try addScope(.branchArm, parent: placed, owner: id)
            case .switchSelect(_, _, let values, _):
                var arms: [Int] = []
                for _ in 0...values.count { arms.append(try addScope(.branchArm, parent: placed, owner: id)) }
                caseScopes[id] = arms
            case .loop(let count, _, _, _, _, let skip):
                let body = try addScope(.loopBody, parent: placed, owner: id, count: count)
                bodyScope[id] = body
                if skip != nil { updateScope[id] = try addScope(.branchArm, parent: body, owner: id) }
                iterations += scopes[placed].repetitions * count
            case .parallaxOcclusion(_, _, _, _, let steps):
                iterations += scopes[placed].repetitions * (steps + 1)
            default:
                break
            }
            guard iterations <= MaterialGraph.maximumTotalIterations else {
                throw Failure.tooManyIterations(node: id, iterations: iterations)
            }
        }
        // Members were collected consumers first; emit them dependencies first.
        for key in members.keys { members[key]!.reverse() }
    }

    private enum UseSite {
        case scope(of: NodeID)
        case trueArm(NodeID)
        case falseArm(NodeID)
        case body(NodeID)
        case update(NodeID)
        case switchArm(NodeID, Int)
    }

    private mutating func addScope(_ kind: ScopeKind, parent: Int, owner: NodeID, count: Int = 1) throws -> Int {
        let depth = scopes[parent].depth + 1
        guard depth <= MaterialGraph.maximumNestingDepth else { throw Failure.nestingTooDeep(owner) }
        // Saturate so a deep nest of large loops reports the limit instead of overflowing.
        let repetitions = min(scopes[parent].repetitions * count, MaterialGraph.maximumTotalIterations + 1)
        scopes.append(Scope(kind: kind, parent: parent, depth: depth, repetitions: repetitions))
        return scopes.count - 1
    }

    /// Declared functions whose bodies take screen-space derivatives or exchange values
    /// between lanes, directly or by calling another such declared function, so a call
    /// must run in the same control flow for every pixel of a quad. The match is
    /// textual and conservative: any mention of `dfdx`, `dfdy`, `fwidth`, `quad_` or
    /// `simd_` counts.
    static func quadDependentFunctions(_ functions: [ShaderFunction]) -> [Bool] {
        let tokens = ["dfdx", "dfdy", "fwidth", "quad_", "simd_"]
        var marked = functions.map { function in tokens.contains { function.body.contains($0) } }
        var changed = true
        while changed {
            changed = false
            for index in functions.indices where !marked[index] {
                let body = functions[index].body
                if functions.indices.contains(where: { marked[$0] && body.contains("_fn_\(functions[$0].name)") }) {
                    marked[index] = true
                    changed = true
                }
            }
        }
        return marked
    }

    private func lowestCommonAncestor(_ a: Int, _ b: Int) -> Int {
        var a = a
        var b = b
        while scopes[a].depth > scopes[b].depth { a = scopes[a].parent }
        while scopes[b].depth > scopes[a].depth { b = scopes[b].parent }
        while a != b {
            a = scopes[a].parent
            b = scopes[b].parent
        }
        return a
    }

    /// The sorted union of two small sorted sets.
    private static func union(_ a: [NodeID], _ b: [NodeID]) -> [NodeID] {
        if b.isEmpty { return a }
        if a.isEmpty { return b }
        return Array(Set(a).union(b)).sorted()
    }
}
