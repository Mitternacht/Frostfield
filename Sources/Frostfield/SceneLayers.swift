import Foundation

/// A set of up to 32 layers, like Three.js `Layers`. Nodes carry their own set and the
/// scene graph's camera carries another; a node takes part in a frame when the two
/// sets share a layer. New nodes and the default camera use layer 0 only.
public struct SceneLayers: OptionSet, Hashable, Sendable, Codable {
    /// The number of layers a set can hold.
    public static let count = 32

    /// Layer 0 only, the default for nodes and cameras.
    public static let `default` = SceneLayers(rawValue: 1)
    /// Every layer.
    public static let all = SceneLayers(rawValue: .max)

    /// Bit `i` is layer `i`.
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    /// The set containing only `layer`, which must be in `0..<32`.
    public init(layer: Int) {
        precondition((0..<Self.count).contains(layer), "Scene layers are numbered 0 through 31.")
        rawValue = 1 << UInt32(layer)
    }

    /// The set containing each listed layer.
    public init(layers: some Sequence<Int>) {
        self = layers.reduce(into: SceneLayers()) { $0.enable($1) }
    }

    /// The layer numbers in ascending order.
    public var layerNumbers: [Int] {
        (0..<Self.count).filter { rawValue & (1 << UInt32($0)) != 0 }
    }

    /// Whether `layer` (in `0..<32`) is in the set, like Three.js `isEnabled`.
    public func contains(layer: Int) -> Bool {
        contains(SceneLayers(layer: layer))
    }

    /// Whether the sets share a layer, like Three.js `Layers.test`.
    public func intersects(_ other: SceneLayers) -> Bool {
        rawValue & other.rawValue != 0
    }

    /// Adds a layer, like Three.js `enable`.
    public mutating func enable(_ layer: Int) {
        insert(SceneLayers(layer: layer))
    }

    /// Removes a layer, like Three.js `disable`.
    public mutating func disable(_ layer: Int) {
        remove(SceneLayers(layer: layer))
    }

    /// Flips a layer, like Three.js `toggle`.
    public mutating func toggle(_ layer: Int) {
        formSymmetricDifference(SceneLayers(layer: layer))
    }

    /// Replaces the set with one layer, like Three.js `set`.
    public mutating func set(only layer: Int) {
        self = SceneLayers(layer: layer)
    }
}

/// A JSON-compatible value for a scene node's application metadata, like the values
/// Three.js stores in `Object3D.userData`. Numbers are `Double`, so integers stay exact
/// up to 2⁵³. Archives reject nonfinite numbers, which JSON cannot represent.
public enum SceneMetadataValue: Hashable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([SceneMetadataValue])
    case object([String: SceneMetadataValue])

    public var boolValue: Bool? {
        if case .bool(let value) = self { value } else { nil }
    }

    public var numberValue: Double? {
        if case .number(let value) = self { value } else { nil }
    }

    public var stringValue: String? {
        if case .string(let value) = self { value } else { nil }
    }

    public var arrayValue: [SceneMetadataValue]? {
        if case .array(let value) = self { value } else { nil }
    }

    public var objectValue: [String: SceneMetadataValue]? {
        if case .object(let value) = self { value } else { nil }
    }

    /// Reads a member of an object value.
    public subscript(key: String) -> SceneMetadataValue? {
        objectValue?[key]
    }

    /// Whether every number in the value, including nested ones, is finite.
    public var isFinite: Bool {
        switch self {
        case .null, .bool, .string: true
        case .number(let value): value.isFinite
        case .array(let values): values.allSatisfy(\.isFinite)
        case .object(let members): members.values.allSatisfy(\.isFinite)
        }
    }
}

extension SceneMetadataValue: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([SceneMetadataValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: SceneMetadataValue].self))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

extension SceneMetadataValue: ExpressibleByNilLiteral, ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral, ExpressibleByStringLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral {
    public init(nilLiteral: ()) { self = .null }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(integerLiteral value: Int) { self = .number(Double(value)) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(stringLiteral value: String) { self = .string(value) }
    public init(arrayLiteral elements: SceneMetadataValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, SceneMetadataValue)...) {
        self = .object(Dictionary(elements, uniquingKeysWith: { _, last in last }))
    }
}
