import Foundation
import simd

/// Weighted accumulation of one typed property, following Three.js `PropertyMixer`.
///
/// Continuous values average by weight and fill weight below one with the rest
/// value; additive offsets then add, and quaternions blend like node rotations.
/// Booleans and strings select like `PropertyMixer._select`: each contribution
/// replaces the running value when its weight is at least half the accumulated
/// weight, and the rest value wins when the total weight is at most one half.
/// An additive action's boolean or string replaces that result when the action's
/// weight is at least one half, the last such action winning, as in Three.js. A
/// lighter additive value has no effect, where Three.js would restore the rest value.
struct PropertyBlend {
    let property: AnimationProperty
    private(set) var kind: AnimationValue.Kind
    private var sum: [Double]
    private var weight = 0.0
    private var offset: [Double]
    private var hasOffset = false
    private var rotation = RotationBlend()
    private var selected: AnimationValue?
    private var additiveSelected: AnimationValue?

    init(property: AnimationProperty, kind: AnimationValue.Kind) {
        self.property = property
        self.kind = kind
        sum = Array(repeating: 0, count: kind.componentCount)
        offset = sum
    }

    mutating func add(_ value: AnimationValue, weight contribution: Double, additive: Bool) throws {
        guard kind.accepts(value.kind) else { throw AnimationMixer.Failure.valueKindMismatch(property.description) }
        switch value {
        case .quaternion(let rotation):
            self.rotation.add(rotation, weight: contribution, additive: additive)
            if !additive { weight += contribution } else { hasOffset = true }
        case .boolean, .string:
            guard !additive else {
                if contribution >= 0.5 { additiveSelected = value }
                return
            }
            if weight == 0 {
                selected = value
                weight = contribution
            } else {
                weight += contribution
                if contribution / weight >= 0.5 { selected = value }
            }
        default:
            let components = value.components ?? []
            for index in components.indices {
                if additive { offset[index] += components[index] * contribution } else { sum[index] += components[index] * contribution }
            }
            if additive { hasOffset = true } else { weight += contribution }
        }
    }

    /// The blended value of the rest value's kind.
    func resolve(rest: AnimationValue) throws -> AnimationValue {
        guard rest.kind.accepts(kind) else { throw AnimationMixer.Failure.valueKindMismatch(property.description) }
        guard weight > 0 || hasOffset || additiveSelected != nil else { return rest }
        let value: AnimationValue
        switch rest {
        case .quaternion(let restRotation):
            value = .quaternion(rotation.resolve(rest: restRotation))
        case .boolean, .string:
            if let additiveSelected { return additiveSelected }
            guard let selected, weight > 0.5 else { return rest }
            value = selected
        default:
            let restComponents = rest.components ?? []
            let blended = restComponents.indices.map { index in
                let base = weight >= 1 ? sum[index] / weight
                    : sum[index] + restComponents[index] * (1 - weight)
                return base + offset[index]
            }
            value = .continuous(rest.kind, blended)
        }
        guard value.isFinite else { throw AnimationMixer.Failure.invalidPropertyValue(property.description) }
        return value
    }
}

/// Blends for many properties, in first-contribution order.
struct PropertyBlends {
    private(set) var order: [AnimationProperty] = []
    private(set) var blends: [AnimationProperty: PropertyBlend] = [:]

    var isEmpty: Bool { blends.isEmpty }

    mutating func add(_ value: AnimationValue, for property: AnimationProperty, kind: AnimationValue.Kind, weight: Double, additive: Bool) throws {
        if blends[property] == nil {
            order.append(property)
            blends[property] = PropertyBlend(property: property, kind: kind)
        }
        try blends[property]!.add(value, weight: weight, additive: additive)
    }

    subscript(property: AnimationProperty) -> PropertyBlend? { blends[property] }
}

extension AnimationMixer {
    /// Accumulates every playing action's typed tracks that match a filter.
    func blendTracks(targetNode: UUID? = nil, where include: (AnimationProperty) -> Bool) throws -> PropertyBlends {
        var result = PropertyBlends()
        for action in actions {
            let weight = Double(action.effectiveWeight)
            guard weight > 0 else { continue }
            let clip = action.clip
            let groupSource = targetNode.flatMap { action.objectGroup?.sourceNode(for: $0) }
            for index in clip.tracks.indices {
                let track = clip.tracks[index]
                let property: AnimationProperty
                if action.objectGroup != nil, case .node(let source, let field) = track.property {
                    guard let targetNode, source == groupSource else { continue }
                    property = .node(targetNode, field)
                } else {
                    property = track.property
                }
                guard include(property) else { continue }
                try result.add(
                    try clip.trackValue(index, at: action.clipTime), for: property, kind: track.valueKind,
                    weight: weight, additive: clip.blendMode == .additive
                )
            }
        }
        return result
    }

    /// Returns the blend of one property, from every playing action with a track
    /// for it, over a rest value of its kind. Returns the rest value when no playing
    /// action animates the property. Use it for glTF camera parameters, custom
    /// properties, or to inspect any property.
    public func value(of property: AnimationProperty, rest: AnimationValue) throws -> AnimationValue {
        let targetNode: UUID? = if case .node(let id, _) = property { id } else { nil }
        let blends = try blendTracks(targetNode: targetNode) { $0 == property }
        return try blends[property]?.resolve(rest: rest) ?? rest
    }

    /// Writes the blended values of every shader uniform and live material factor
    /// that a track of any action in the mixer targets, playing or not.
    ///
    /// Like graph properties, these blend toward a rest value: a uniform's declared
    /// default, or a material's authored factor (`LiveFactors.rest`). A property that
    /// no playing action animates returns to its rest value, so stopping an action
    /// restores what it changed, as Three.js restores a binding's original state.
    /// Factors that no track targets keep their current values. Throws without
    /// writing a material when its blended factors are invalid.
    public func applyLiveProperties() throws {
        var targets: [AnimationProperty] = []
        var seen = Set<AnimationProperty>()
        for action in actions {
            for track in action.clip.tracks where track.property.isLiveProperty && seen.insert(track.property).inserted {
                targets.append(track.property)
            }
        }
        guard !targets.isEmpty else { return }
        let blends = try blendTracks { $0.isLiveProperty }

        var materials: [ObjectIdentifier: (live: PhysicallyBasedMaterial.LiveFactors, properties: [AnimationProperty.Material])] = [:]
        var materialOrder: [ObjectIdentifier] = []
        for property in targets {
            switch property {
            case .uniform(let uniforms, let name):
                guard let declaration = uniforms.declarations.first(where: { $0.name == name }) else {
                    throw ShaderUniforms.Failure.unknownName(name)
                }
                let value = declaration.defaultValue
                let rest: AnimationValue? = switch declaration.type {
                case .scalar: .number(value.x)
                case .vector2: .vector2(SIMD2(value.x, value.y))
                case .vector3: .vector3(SIMD3(value.x, value.y, value.z))
                case .vector4: .vector4(value)
                case .matrix3, .matrix4, .int, .bool, .structure: nil
                }
                guard let rest else { throw Failure.valueKindMismatch(property.description) }
                switch try blends[property]?.resolve(rest: rest) ?? rest {
                case .number(let number): try uniforms.set(name, to: number)
                case .vector2(let vector): try uniforms.set(name, to: vector)
                case .vector3(let vector), .color(let vector): try uniforms.set(name, to: vector)
                case .vector4(let vector): try uniforms.set(name, to: vector)
                default: throw Failure.valueKindMismatch(property.description)
                }
            case .material(let live, let factor):
                let key = ObjectIdentifier(live)
                if materials[key] == nil {
                    materials[key] = (live, [])
                    materialOrder.append(key)
                }
                materials[key]!.properties.append(factor)
            default:
                continue
            }
        }
        for key in materialOrder {
            let (live, targets) = materials[key]!
            var properties = targets
            if properties.contains(.baseColorFactor), let rgb = properties.firstIndex(of: .baseColorRGB) {
                // An active RGB channel overlays the whole factor's RGB. A stopped
                // channel leaves the whole-factor binding in charge of those components.
                properties.remove(at: rgb)
                if blends[.material(live, .baseColorRGB)] != nil { properties.append(.baseColorRGB) }
            }
            var factors = live.current
            for factor in properties {
                guard let rest = live.rest.value(of: factor) else {
                    throw PhysicallyBasedMaterial.LiveFactors.Failure.missingFactor(factor)
                }
                try factors.set(factor, to: try blends[.material(live, factor)]?.resolve(rest: rest) ?? rest)
            }
            try live.set(factors)
        }
    }

    /// Returns a copy of a light rig with the blended values of every `rigLight`
    /// property that a playing action animates. Other lights and values keep the
    /// rig's. Throws `missingRigLight` for an unknown identifier,
    /// `unanimatableProperty` when a light lacks the property, such as `power` on a
    /// light with photometric emission, and the rig's validation errors.
    public func apply(to rig: LightRig) throws -> LightRig {
        let blends = try blendTracks { if case .rigLight = $0 { true } else { false } }
        guard !blends.isEmpty else { return rig }
        var lights = rig.lights
        for property in blends.order {
            guard case .rigLight(let id, let field) = property, let blend = blends[property] else { continue }
            guard let index = lights.firstIndex(where: { $0.id == id }) else { throw Failure.missingRigLight(id) }
            var light = lights[index]
            func number(_ rest: Float) throws -> Float {
                guard let value = try blend.resolve(rest: .number(rest)).floatValue else {
                    throw Failure.valueKindMismatch(property.description)
                }
                return value
            }
            func unanimatable(_ reason: String) -> Failure { .unanimatableProperty(property.description, reason) }
            switch field {
            case .position: light.position = try blend.resolve(rest: .vector3(light.position)).vector3Value!
            case .target: light.target = try blend.resolve(rest: .vector3(light.target)).vector3Value!
            case .color: light.color = try blend.resolve(rest: .color(light.color)).vector3Value!
            case .intensity:
                guard case .intensity(let value) = light.emission else { throw unanimatable("its emission is a power") }
                light.emission = .intensity(try number(value))
            case .power:
                guard case .power(let value) = light.emission else { throw unanimatable("its emission is an intensity") }
                light.emission = .power(try number(value))
            case .innerAngle, .outerAngle:
                guard case .spot(var inner, var outer, let range) = light.kind else { throw unanimatable("it is not a spot light") }
                if field == .innerAngle { inner = try number(inner) } else { outer = try number(outer) }
                light.kind = .spot(innerAngle: inner, outerAngle: outer, range: range)
            case .range:
                switch light.kind {
                case .point(let range?): light.kind = .point(range: try number(range))
                case .spot(let inner, let outer, let range?): light.kind = .spot(innerAngle: inner, outerAngle: outer, range: try number(range))
                default: throw unanimatable("it has no finite range")
                }
            case .width, .height:
                guard case .rectangle(var width, var height) = light.kind else { throw unanimatable("it is not a rectangle") }
                if field == .width { width = try number(width) } else { height = try number(height) }
                light.kind = .rectangle(width: width, height: height)
            }
            lights[index] = light
        }
        return try LightRig(lights: lights)
    }

    /// Writes blended node, camera and environment properties into a graph value.
    /// Translation, rotation and scale tracks are blended with the node tracks instead.
    func applyGraphProperties(_ blends: PropertyBlends, to graph: SceneGraph) throws -> (graph: SceneGraph, missing: [UUID]) {
        var nodeEntries: [UUID: [(AnimationProperty.Node, PropertyBlend)]] = [:]
        var nodeOrder: [UUID] = []
        for property in blends.order {
            guard case .node(let id, let field) = property, let blend = blends[property] else { continue }
            if nodeEntries[id] == nil { nodeOrder.append(id) }
            nodeEntries[id, default: []].append((field, blend))
        }
        var result = graph
        if !nodeEntries.isEmpty {
            var failure: (any Error)?
            result = graph.updatingNodes { node in
                guard let entries = nodeEntries.removeValue(forKey: node.id), failure == nil else { return }
                do { try Self.apply(entries, to: &node) } catch { failure = error }
            }
            if let failure { throw failure }
        }

        var cameraChanged = false
        var fieldOfView = result.camera.verticalFieldOfView
        var near = result.camera.near
        var far = result.camera.far
        for property in blends.order {
            guard let blend = blends[property] else { continue }
            func number(_ rest: Float) throws -> Float {
                guard let value = try blend.resolve(rest: .number(rest)).floatValue else {
                    throw Failure.valueKindMismatch(property.description)
                }
                return value
            }
            switch property {
            case .camera(.verticalFieldOfView): fieldOfView = try number(fieldOfView); cameraChanged = true
            case .camera(.near): near = try number(near); cameraChanged = true
            case .camera(.far): far = try number(far); cameraChanged = true
            case .camera(.translation):
                result.cameraTransform.translation = try blend.resolve(rest: .vector3(result.cameraTransform.translation)).vector3Value!
            case .camera(.rotation):
                result.cameraTransform.rotation = try blend.resolve(rest: .quaternion(result.cameraTransform.rotation)).quaternionValue!
            case .environment(.intensity):
                let intensity = try number(result.environmentIntensity)
                guard intensity >= 0 else { throw Failure.invalidPropertyValue(property.description) }
                result.environmentIntensity = intensity
            case .environment(.rotation):
                result.environmentRotation = try number(result.environmentRotation)
            default: continue
            }
        }
        if cameraChanged {
            result.camera = try PerspectiveCamera(verticalFieldOfView: fieldOfView, near: near, far: far)
        }
        return (result, nodeOrder.filter { nodeEntries[$0] != nil })
    }

    private static func apply(_ entries: [(AnimationProperty.Node, PropertyBlend)], to node: inout SceneNode) throws {
        func entry(_ field: AnimationProperty.Node) -> PropertyBlend? { entries.first { $0.0 == field }?.1 }
        func number(_ blend: PropertyBlend, _ rest: Float) throws -> Float {
            guard let value = try blend.resolve(rest: .number(rest)).floatValue else {
                throw Failure.valueKindMismatch(blend.property.description)
            }
            return value
        }
        func unanimatable(_ field: AnimationProperty.Node, _ reason: String) -> Failure {
            .unanimatableProperty(AnimationProperty.node(node.id, field).description, reason)
        }

        let morphIndices = entries.compactMap { entry -> (Int, PropertyBlend)? in
            if case .morphWeight(let index) = entry.0 { (index, entry.1) } else { nil }
        }.sorted { $0.0 < $1.0 }
        if entry(.morphWeights) != nil || !morphIndices.isEmpty {
            guard let count = node.mesh?.morphTargets?.targets.count ?? node.morphWeights?.count, count > 0 else {
                throw unanimatable(.morphWeights, "the node's mesh has no morph targets")
            }
            var weights = node.morphWeights ?? Array(repeating: 0, count: count)
            if let blend = entry(.morphWeights) {
                guard case .numbers(let values) = try blend.resolve(rest: .numbers(weights)) else {
                    throw Failure.valueKindMismatch(blend.property.description)
                }
                weights = values
            }
            for (index, blend) in morphIndices {
                guard weights.indices.contains(index) else { throw unanimatable(.morphWeight(index), "the mesh has \(count) morph targets") }
                weights[index] = try number(blend, weights[index])
            }
            node.morphWeights = weights
        }
        if let blend = entry(.materialParameters) {
            node.materialParameters = try blend.resolve(rest: .vector4(node.materialParameters)).vector4Value!
        }
        if let blend = entry(.castsShadow) {
            node.castsShadow = try blend.resolve(rest: .boolean(node.castsShadow)).booleanValue!
        }
        if let blend = entry(.isVisible) {
            node.isVisible = try blend.resolve(rest: .boolean(node.isVisible)).booleanValue!
        }

        let lightFields: [AnimationProperty.Node] = [.lightColor, .lightIntensity, .lightRange, .spotInnerConeAngle, .spotOuterConeAngle]
        guard lightFields.contains(where: { entry($0) != nil }) else { return }
        guard let light = node.light else {
            throw unanimatable(lightFields.first { entry($0) != nil }!, "the node has no light")
        }
        var color = light.color
        var intensity = light.intensity
        var kind = light.kind
        if let blend = entry(.lightColor) { color = try blend.resolve(rest: .color(color)).vector3Value! }
        if let blend = entry(.lightIntensity) { intensity = try number(blend, intensity) }
        if let blend = entry(.lightRange) {
            switch kind {
            case .point(let range?): kind = .point(range: try number(blend, range))
            case .spot(let range?, let inner, let outer): kind = .spot(range: try number(blend, range), innerConeAngle: inner, outerConeAngle: outer)
            default: throw unanimatable(.lightRange, "the light has no finite range")
            }
        }
        for field in [AnimationProperty.Node.spotInnerConeAngle, .spotOuterConeAngle] {
            guard let blend = entry(field) else { continue }
            guard case .spot(let range, var inner, var outer) = kind else { throw unanimatable(field, "the light is not a spot light") }
            if field == .spotInnerConeAngle { inner = try number(blend, inner) } else { outer = try number(blend, outer) }
            kind = .spot(range: range, innerConeAngle: inner, outerConeAngle: outer)
        }
        node.light = try PunctualLight(
            kind: kind, color: color, intensity: intensity, shadow: light.shadow,
            iesProfile: light.iesProfile, cookie: light.cookie
        )
    }
}
