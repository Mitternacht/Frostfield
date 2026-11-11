import Foundation
import simd

/// A value that an `AnimationTrack` keys and an animated property holds. The cases
/// cover the value types of Three.js `NumberKeyframeTrack`, `VectorKeyframeTrack`,
/// `QuaternionKeyframeTrack`, `ColorKeyframeTrack`, `BooleanKeyframeTrack` and
/// `StringKeyframeTrack`.
public enum AnimationValue: Equatable, Sendable {
    case number(Float)
    case vector2(SIMD2<Float>)
    case vector3(SIMD3<Float>)
    case vector4(SIMD4<Float>)
    /// A unit quaternion. Linear tracks interpolate it along the shortest arc.
    case quaternion(simd_quatf)
    /// Linear RGB, interpolated componentwise.
    case color(SIMD3<Float>)
    /// A discrete value. Tracks step between booleans and never interpolate them.
    case boolean(Bool)
    /// A discrete value. Tracks step between strings and never interpolate them.
    case string(String)
    /// A fixed-length list, such as every morph target weight of a mesh.
    case numbers([Float])

    public enum Kind: Hashable, Sendable, Codable {
        case number, vector2, vector3, vector4, quaternion, color, boolean, string
        case numbers(count: Int)

        /// Whether values of this kind interpolate. Booleans and strings only step.
        public var isContinuous: Bool { self != .boolean && self != .string }

        /// Floats per value; zero for booleans and strings.
        var componentCount: Int {
            switch self {
            case .number: 1
            case .vector2: 2
            case .vector3, .color: 3
            case .vector4, .quaternion: 4
            case .numbers(let count): count
            case .boolean, .string: 0
            }
        }

        /// Whether a value of `other` kind can drive a property of this kind. A color
        /// and a three-component vector are interchangeable.
        func accepts(_ other: Kind) -> Bool {
            if self == other { return true }
            switch (self, other) {
            case (.vector3, .color), (.color, .vector3): return true
            default: return false
            }
        }
    }

    public var kind: Kind {
        switch self {
        case .number: .number
        case .vector2: .vector2
        case .vector3: .vector3
        case .vector4: .vector4
        case .quaternion: .quaternion
        case .color: .color
        case .boolean: .boolean
        case .string: .string
        case .numbers(let values): .numbers(count: values.count)
        }
    }

    /// The interpolated components in double precision, or nil for a discrete value.
    /// Quaternions are `(x, y, z, w)`.
    var components: [Double]? {
        switch self {
        case .number(let value): [Double(value)]
        case .vector2(let value): [Double(value.x), Double(value.y)]
        case .vector3(let value), .color(let value): [Double(value.x), Double(value.y), Double(value.z)]
        case .vector4(let value): [Double(value.x), Double(value.y), Double(value.z), Double(value.w)]
        case .quaternion(let value):
            [Double(value.vector.x), Double(value.vector.y), Double(value.vector.z), Double(value.vector.w)]
        case .numbers(let values): values.map(Double.init)
        case .boolean, .string: nil
        }
    }

    /// Rebuilds a continuous value of a kind from its components.
    static func continuous(_ kind: Kind, _ components: some Collection<Double>) -> AnimationValue {
        let values = components.map { Float($0) }
        switch kind {
        case .number: return .number(values[0])
        case .vector2: return .vector2(SIMD2(values[0], values[1]))
        case .vector3: return .vector3(SIMD3(values[0], values[1], values[2]))
        case .color: return .color(SIMD3(values[0], values[1], values[2]))
        case .vector4: return .vector4(SIMD4(values[0], values[1], values[2], values[3]))
        case .quaternion: return .quaternion(simd_quatf(vector: SIMD4(values[0], values[1], values[2], values[3])))
        case .numbers: return .numbers(values)
        case .boolean, .string: preconditionFailure("Discrete kinds have no components.")
        }
    }

    /// Converts a color or three-component vector to the kind a property expects.
    func converted(to kind: Kind) -> AnimationValue {
        switch (self, kind) {
        case (.color(let value), .vector3): .vector3(value)
        case (.vector3(let value), .color): .color(value)
        default: self
        }
    }

    var isFinite: Bool { components?.allSatisfy(\.isFinite) ?? true }

    var floatValue: Float? { if case .number(let value) = self { value } else { nil } }
    var vector2Value: SIMD2<Float>? { if case .vector2(let value) = self { value } else { nil } }
    var vector4Value: SIMD4<Float>? { if case .vector4(let value) = self { value } else { nil } }
    var quaternionValue: simd_quatf? { if case .quaternion(let value) = self { value } else { nil } }
    var booleanValue: Bool? { if case .boolean(let value) = self { value } else { nil } }
    var vector3Value: SIMD3<Float>? {
        switch self {
        case .vector3(let value), .color(let value): value
        default: nil
        }
    }
}

/// What an animation track drives: a typed path to one property, like a Three.js
/// `PropertyBinding` path such as `.morphTargetInfluences[smile]` or
/// `.material.roughness`.
///
/// Graph properties change the graph that `AnimationMixer.apply(to:)` returns. Shader
/// uniforms and live material factors are shared objects that
/// `AnimationMixer.applyLiveProperties()` writes. Light rig lights change the rig
/// that `AnimationMixer.apply(to:)` returns for a `LightRig`. glTF camera and custom
/// properties are read with `AnimationMixer.value(of:rest:)`.
public enum AnimationProperty: Hashable, Sendable, CustomStringConvertible {
    /// A property of a scene graph node.
    case node(UUID, Node)
    /// The scene graph's camera projection or pose.
    case camera(Camera)
    /// The scene graph's image-based lighting.
    case environment(Environment)
    /// A declared value of live shader uniforms, such as a surface graph's.
    case uniform(ShaderUniforms, name: String)
    /// A factor of a material made with `PhysicallyBasedMaterial.withLiveFactors()`.
    case material(PhysicallyBasedMaterial.LiveFactors, Material)
    /// A light of a `LightRig`, by identifier.
    case rigLight(id: String, RigLight)
    /// A projection parameter of an imported glTF camera, by source camera index.
    /// `PreparedGLTFScene.cameraView(nodeIndex:in:viewportAspectRatio:animatedBy:)` reads it.
    case gltfCamera(Int, GLTFCamera)
    /// An application value with no built-in target, read with `AnimationMixer.value(of:rest:)`.
    case custom(String)

    public enum Node: Hashable, Sendable, Codable {
        /// Local transform components: vector3, quaternion, vector3.
        case translation, rotation, scale
        /// Every morph target weight of the node's mesh, like Three.js `morphTargetInfluences`.
        case morphWeights
        /// One morph target weight by target index.
        case morphWeight(Int)
        /// `SceneNode.materialParameters`, a vector4.
        case materialParameters
        /// `SceneNode.castsShadow`, a boolean.
        case castsShadow
        /// `SceneNode.isVisible`, a boolean inherited by the node's descendants.
        case isVisible
        /// The node's punctual light: linear color, intensity, range, and spot cone angles in radians.
        case lightColor, lightIntensity, lightRange, spotInnerConeAngle, spotOuterConeAngle
    }

    public enum Camera: String, Hashable, Sendable, Codable {
        /// Projection parameters, numbers in radians and world units.
        case verticalFieldOfView, near, far
        /// `SceneGraph.cameraTransform` components: vector3 and quaternion.
        case translation, rotation
    }

    public enum Environment: String, Hashable, Sendable, Codable {
        /// `SceneGraph.environmentIntensity` and `environmentRotation`, numbers.
        case intensity, rotation
    }

    /// The texture maps whose `TextureTransform` can be animated.
    public enum MaterialMap: String, Hashable, Sendable, CaseIterable, Codable {
        case baseColor, metallicRoughness, normal, occlusion, emissive
    }

    /// The animatable factors of `PhysicallyBasedMaterial.Factors`.
    public enum Material: Hashable, Sendable, Codable {
        /// Vector4.
        case baseColorFactor
        /// Linear RGB. Leaves the material's base-color alpha untouched.
        case baseColorRGB
        /// Numbers.
        case metallic, roughness
        /// Color.
        case emissiveFactor
        /// Numbers.
        case emissiveStrength, normalScale, occlusionStrength, alphaCutoff
        /// KHR_texture_transform offset (vector2), rotation in radians (number) and scale (vector2).
        case textureOffset(MaterialMap), textureRotation(MaterialMap), textureScale(MaterialMap)
        /// Numbers.
        case clearcoatFactor, clearcoatRoughness, transmissionFactor, specularFactor
        /// Color.
        case specularColor, sheenColor
        /// Numbers.
        case sheenRoughness, iridescenceFactor, anisotropyStrength, anisotropyRotation
    }

    public enum RigLight: String, Hashable, Sendable, Codable {
        /// Vector3 positions in meters.
        case position, target
        /// Linear color.
        case color
        /// The photometric value, for a light whose emission is `.intensity`.
        case intensity
        /// Total power, for a light whose emission is `.power`.
        case power
        /// Spot cone angles in radians.
        case innerAngle, outerAngle
        /// Point or spot range in meters.
        case range
        /// Rectangle size in meters.
        case width, height
    }

    public enum GLTFCamera: String, Hashable, Sendable, Codable {
        /// glTF `perspective.yfov`, `aspectRatio`, `znear` and `zfar`.
        case verticalFieldOfView, aspectRatio, near, far
        /// glTF `orthographic.xmag` and `ymag`.
        case halfWidth, halfHeight
    }

    /// The kind of value the property holds, or nil when it depends on the target:
    /// a mesh's morph target count or a custom value.
    public var valueKind: AnimationValue.Kind? {
        switch self {
        case .node(_, let property):
            switch property {
            case .translation, .scale: .vector3
            case .rotation: .quaternion
            case .morphWeights: nil
            case .morphWeight, .lightIntensity, .lightRange, .spotInnerConeAngle, .spotOuterConeAngle: .number
            case .materialParameters: .vector4
            case .castsShadow, .isVisible: .boolean
            case .lightColor: .color
            }
        case .camera(let property):
            switch property {
            case .verticalFieldOfView, .near, .far: .number
            case .translation: .vector3
            case .rotation: .quaternion
            }
        case .environment: .number
        case .uniform(let uniforms, let name):
            uniforms.declarations.first { $0.name == name }.flatMap { declaration -> AnimationValue.Kind? in
                switch declaration.type {
                case .scalar: AnimationValue.Kind.number
                case .vector2: .vector2
                case .vector3: .vector3
                case .vector4: .vector4
                case .matrix3, .matrix4, .int, .bool: nil
                }
            }
        case .material(_, let property):
            switch property {
            case .baseColorFactor: .vector4
            case .baseColorRGB, .emissiveFactor, .specularColor, .sheenColor: .color
            case .textureOffset, .textureScale: .vector2
            default: .number
            }
        case .rigLight(_, let property):
            switch property {
            case .position, .target: .vector3
            case .color: .color
            default: .number
            }
        case .gltfCamera: .number
        case .custom: nil
        }
    }

    /// Whether the property belongs to the scene graph.
    var isGraphProperty: Bool {
        switch self {
        case .node, .camera, .environment: true
        default: false
        }
    }

    /// Whether the property belongs to a shared object written by `applyLiveProperties()`.
    var isLiveProperty: Bool {
        switch self {
        case .uniform, .material: true
        default: false
        }
    }

    public var description: String {
        switch self {
        case .node(let id, let property): "node \(id).\(property)"
        case .camera(let property): "camera.\(property)"
        case .environment(let property): "environment.\(property)"
        case .uniform(_, let name): "uniforms.\(name)"
        case .material(_, let property): "material.\(property)"
        case .rigLight(let id, let property): "rig light \(id).\(property)"
        case .gltfCamera(let index, let property): "glTF camera \(index).\(property)"
        case .custom(let name): "custom \(name)"
        }
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs, rhs) {
        case (.node(let a, let p), .node(let b, let q)): a == b && p == q
        case (.camera(let p), .camera(let q)): p == q
        case (.environment(let p), .environment(let q)): p == q
        case (.uniform(let a, let p), .uniform(let b, let q)): a === b && p == q
        case (.material(let a, let p), .material(let b, let q)): a === b && p == q
        case (.rigLight(let a, let p), .rigLight(let b, let q)): a == b && p == q
        case (.gltfCamera(let a, let p), .gltfCamera(let b, let q)): a == b && p == q
        case (.custom(let a), .custom(let b)): a == b
        default: false
        }
    }

    public func hash(into hasher: inout Hasher) {
        switch self {
        case .node(let id, let property):
            hasher.combine(0)
            hasher.combine(id)
            hasher.combine(property)
        case .camera(let property):
            hasher.combine(1)
            hasher.combine(property)
        case .environment(let property):
            hasher.combine(2)
            hasher.combine(property)
        case .uniform(let uniforms, let name):
            hasher.combine(3)
            hasher.combine(ObjectIdentifier(uniforms))
            hasher.combine(name)
        case .material(let factors, let property):
            hasher.combine(4)
            hasher.combine(ObjectIdentifier(factors))
            hasher.combine(property)
        case .rigLight(let id, let property):
            hasher.combine(5)
            hasher.combine(id)
            hasher.combine(property)
        case .gltfCamera(let index, let property):
            hasher.combine(6)
            hasher.combine(index)
            hasher.combine(property)
        case .custom(let name):
            hasher.combine(7)
            hasher.combine(name)
        }
    }
}

extension AnimationProperty {
    /// Parses a Three.js-style binding path against named nodes, such as
    /// `"Arm.quaternion"`, `"Face.morphTargetInfluences[smile]"`,
    /// `"Face.morphTargetInfluences[2]"`, `"Lamp.intensity"` or `".position"` with a
    /// default node. Node properties are `position`, `quaternion`, `scale`,
    /// `morphTargetInfluences`, `castShadow`, `materialParameters`, and for lights
    /// `color`, `intensity`, `distance` (range), `angle` (outer cone) and
    /// `innerConeAngle`. As in Three.js, node names may contain dots, which split
    /// at the last dot before the property, and a bracketed key may contain any
    /// character, such as the Blender shape key `smile.L`. Morph target names
    /// resolve through `morphTargetNames`, keyed by node name; with the default node
    /// they resolve under `""` or under a name that `nodes` gives the default node.
    /// Throws `PathError` for an unknown node, property or morph target name.
    public init(
        path: String, nodes: [String: UUID], defaultNode: UUID? = nil, morphTargetNames: [String: [String]] = [:]
    ) throws {
        // Node names cannot contain brackets, so the first one opens the property's key.
        var head = Substring(path)
        var key: String?
        if let open = path.firstIndex(of: "[") {
            guard path.last == "]", path.index(after: open) < path.index(before: path.endIndex) else {
                throw PathError.malformed(path)
            }
            key = String(path[path.index(after: open)..<path.index(before: path.endIndex)])
            head = path[..<open]
        }
        guard let dot = head.lastIndex(of: ".") else { throw PathError.malformed(path) }
        let nodeName = String(head[..<dot])
        let member = String(head[head.index(after: dot)...])
        let id: UUID
        if nodeName.isEmpty {
            guard let defaultNode else { throw PathError.unknownNode(path) }
            id = defaultNode
        } else {
            guard let named = nodes[nodeName] else { throw PathError.unknownNode(nodeName) }
            id = named
        }

        let property: Node
        switch (member, key) {
        case ("position", nil): property = .translation
        case ("quaternion", nil): property = .rotation
        case ("scale", nil): property = .scale
        case ("morphTargetInfluences", nil): property = .morphWeights
        case ("morphTargetInfluences", let key?):
            let names = nodeName.isEmpty ? [""] + nodes.filter { $0.value == id }.map(\.key).sorted() : [nodeName]
            if let index = Int(key), index >= 0 {
                property = .morphWeight(index)
            } else if let index = names.lazy.compactMap({ morphTargetNames[$0]?.firstIndex(of: key) }).first {
                property = .morphWeight(index)
            } else {
                throw PathError.unknownMorphTarget(key)
            }
        case ("castShadow", nil): property = .castsShadow
        case ("visible", nil): property = .isVisible
        case ("materialParameters", nil): property = .materialParameters
        case ("color", nil): property = .lightColor
        case ("intensity", nil): property = .lightIntensity
        case ("distance", nil): property = .lightRange
        case ("angle", nil): property = .spotOuterConeAngle
        case ("innerConeAngle", nil): property = .spotInnerConeAngle
        default: throw PathError.unknownProperty(member)
        }
        self = .node(id, property)
    }

    public enum PathError: LocalizedError, Equatable {
        case malformed(String)
        case unknownNode(String)
        case unknownProperty(String)
        case unknownMorphTarget(String)

        public var errorDescription: String? {
            switch self {
            case .malformed(let path): "The animation path \"\(path)\" needs the form node.property or node.property[key]."
            case .unknownNode(let name): "The animation path names no known node \"\(name)\"."
            case .unknownProperty(let name): "The animation path property \"\(name)\" is not animatable."
            case .unknownMorphTarget(let name): "The animation path names no known morph target \"\(name)\"."
            }
        }
    }
}
