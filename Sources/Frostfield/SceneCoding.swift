import Foundation
import simd

// Codable conformances for Frostfield's scene description values, so scenes, light
// rigs and shader graphs serialize like Three.js `toJSON` and node material JSON.
// Decoding runs each type's validating initializer, so invalid JSON throws instead
// of producing a value the public API could not create.

private func corrupted(_ container: some Any, _ message: String, codingPath: [any CodingKey]) -> DecodingError {
    DecodingError.dataCorrupted(.init(codingPath: codingPath, debugDescription: message))
}

/// Runs a validating initializer and reports its failure as corrupted data.
private func validated<T>(_ codingPath: [any CodingKey], _ make: () throws -> T) throws -> T {
    do {
        return try make()
    } catch let error as DecodingError {
        throw error
    } catch {
        throw DecodingError.dataCorrupted(.init(
            codingPath: codingPath, debugDescription: error.localizedDescription, underlyingError: error
        ))
    }
}

// MARK: Transforms, cameras and bounds

extension SceneTransform: Codable {
    private enum CodingKeys: String, CodingKey { case translation, rotation, scale }

    /// Rotation is stored as quaternion components `[x, y, z, w]`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            translation: try container.decode(SIMD3<Float>.self, forKey: .translation),
            rotation: simd_quatf(vector: try container.decode(SIMD4<Float>.self, forKey: .rotation)),
            scale: try container.decode(SIMD3<Float>.self, forKey: .scale)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(translation, forKey: .translation)
        try container.encode(rotation.vector, forKey: .rotation)
        try container.encode(scale, forKey: .scale)
    }
}

extension PerspectiveCamera: Codable {
    private enum CodingKeys: String, CodingKey { case verticalFieldOfView, near, far }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fieldOfView = try container.decode(Float.self, forKey: .verticalFieldOfView)
        let near = try container.decode(Float.self, forKey: .near)
        let far = try container.decode(Float.self, forKey: .far)
        self = try validated(decoder.codingPath) { try PerspectiveCamera(verticalFieldOfView: fieldOfView, near: near, far: far) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(verticalFieldOfView, forKey: .verticalFieldOfView)
        try container.encode(near, forKey: .near)
        try container.encode(far, forKey: .far)
    }
}

extension AxisAlignedBox: Codable {
    private enum CodingKeys: String, CodingKey { case minimum, maximum }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let minimum = try container.decode(SIMD3<Float>.self, forKey: .minimum)
        let maximum = try container.decode(SIMD3<Float>.self, forKey: .maximum)
        self = try validated(decoder.codingPath) { try AxisAlignedBox(minimum: minimum, maximum: maximum) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(minimum, forKey: .minimum)
        try container.encode(maximum, forKey: .maximum)
    }
}

extension PhysicallyBasedMaterial.TextureTransform: Codable {
    private enum CodingKeys: String, CodingKey { case offset, rotation, scale }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let offset = try container.decode(SIMD2<Float>.self, forKey: .offset)
        let rotation = try container.decode(Float.self, forKey: .rotation)
        let scale = try container.decode(SIMD2<Float>.self, forKey: .scale)
        self = try validated(decoder.codingPath) { try .init(offset: offset, rotation: rotation, scale: scale) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(offset, forKey: .offset)
        try container.encode(rotation, forKey: .rotation)
        try container.encode(scale, forKey: .scale)
    }
}

extension MeshTextureCoordinateSet: Codable {}

// MARK: Lights

extension AmbientLight: Codable {
    private enum CodingKeys: String, CodingKey { case color, intensity }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let color = try container.decode(SIMD3<Float>.self, forKey: .color)
        let intensity = try container.decode(Float.self, forKey: .intensity)
        self = try validated(decoder.codingPath) { try .init(color: color, intensity: intensity) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(color, forKey: .color)
        try container.encode(intensity, forKey: .intensity)
    }
}

extension PunctualLight.Shadow: Codable {
    private enum CodingKeys: String, CodingKey {
        case depthBias, normalBias, filterRadius, nearPlane, sampling, varianceBlurRadius, shadowStrength
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let depthBias = try container.decode(Float.self, forKey: .depthBias)
        let normalBias = try container.decode(Float.self, forKey: .normalBias)
        let filterRadius = try container.decode(Float.self, forKey: .filterRadius)
        let nearPlane = try container.decode(Float.self, forKey: .nearPlane)
        let sampling = try container.decodeIfPresent(ShadowMapSampling.self, forKey: .sampling) ?? .filtered
        let varianceBlurRadius = try container.decodeIfPresent(Int.self, forKey: .varianceBlurRadius) ?? 3
        let shadowStrength = try container.decodeIfPresent(Float.self, forKey: .shadowStrength) ?? 1
        self = try validated(decoder.codingPath) {
            try .init(
                depthBias: depthBias, normalBias: normalBias, filterRadius: filterRadius, nearPlane: nearPlane,
                sampling: sampling, varianceBlurRadius: varianceBlurRadius, shadowStrength: shadowStrength
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(depthBias, forKey: .depthBias)
        try container.encode(normalBias, forKey: .normalBias)
        try container.encode(filterRadius, forKey: .filterRadius)
        try container.encode(nearPlane, forKey: .nearPlane)
        if sampling != .filtered { try container.encode(sampling, forKey: .sampling) }
        if varianceBlurRadius != 3 { try container.encode(varianceBlurRadius, forKey: .varianceBlurRadius) }
        if shadowStrength != 1 { try container.encode(shadowStrength, forKey: .shadowStrength) }
    }
}

extension PunctualLight: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, range, innerConeAngle, outerConeAngle, width, height, color, intensity, shadow
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind: Kind
        switch try container.decode(String.self, forKey: .type) {
        case "directional": kind = .directional
        case "point": kind = .point(range: try container.decodeIfPresent(Float.self, forKey: .range))
        case "spot":
            kind = .spot(
                range: try container.decodeIfPresent(Float.self, forKey: .range),
                innerConeAngle: try container.decode(Float.self, forKey: .innerConeAngle),
                outerConeAngle: try container.decode(Float.self, forKey: .outerConeAngle)
            )
        case "rectangle":
            kind = .rectangle(
                width: try container.decode(Float.self, forKey: .width), height: try container.decode(Float.self, forKey: .height)
            )
        case let other:
            throw corrupted(container, "Unknown light type \(other).", codingPath: decoder.codingPath)
        }
        let color = try container.decode(SIMD3<Float>.self, forKey: .color)
        let intensity = try container.decode(Float.self, forKey: .intensity)
        let shadow = try container.decodeIfPresent(Shadow.self, forKey: .shadow)
        self = try validated(decoder.codingPath) { try PunctualLight(kind: kind, color: color, intensity: intensity, shadow: shadow) }
    }

    public func encode(to encoder: any Encoder) throws {
        guard iesProfile == nil, cookie == nil else {
            throw EncodingError.invalidValue(self, .init(
                codingPath: encoder.codingPath, debugDescription: "IES profiles and light cookies need an archive resource record."
            ))
        }
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch kind {
        case .directional:
            try container.encode("directional", forKey: .type)
        case .point(let range):
            try container.encode("point", forKey: .type)
            try container.encodeIfPresent(range, forKey: .range)
        case .spot(let range, let inner, let outer):
            try container.encode("spot", forKey: .type)
            try container.encodeIfPresent(range, forKey: .range)
            try container.encode(inner, forKey: .innerConeAngle)
            try container.encode(outer, forKey: .outerConeAngle)
        case .rectangle(let width, let height):
            try container.encode("rectangle", forKey: .type)
            try container.encode(width, forKey: .width)
            try container.encode(height, forKey: .height)
        }
        try container.encode(color, forKey: .color)
        try container.encode(intensity, forKey: .intensity)
        try container.encodeIfPresent(shadow, forKey: .shadow)
    }
}

extension LightRig.Light: Codable {
    private enum CodingKeys: String, CodingKey {
        case id, type, angularDiameter, range, innerAngle, outerAngle, width, height
        case position, target, color, power, intensity, castsShadow, isEnabled, isVisibleToCamera
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind: Kind
        switch try container.decode(String.self, forKey: .type) {
        case "directional": kind = .directional(angularDiameter: try container.decode(Float.self, forKey: .angularDiameter))
        case "point": kind = .point(range: try container.decodeIfPresent(Float.self, forKey: .range))
        case "spot":
            kind = .spot(
                innerAngle: try container.decode(Float.self, forKey: .innerAngle),
                outerAngle: try container.decode(Float.self, forKey: .outerAngle),
                range: try container.decodeIfPresent(Float.self, forKey: .range)
            )
        case "rectangle":
            kind = .rectangle(
                width: try container.decode(Float.self, forKey: .width), height: try container.decode(Float.self, forKey: .height)
            )
        case let other:
            throw corrupted(container, "Unknown rig light type \(other).", codingPath: decoder.codingPath)
        }
        let emission: Emission
        if let power = try container.decodeIfPresent(Float.self, forKey: .power) {
            emission = .power(power)
        } else {
            emission = .intensity(try container.decode(Float.self, forKey: .intensity))
        }
        self.init(
            id: try container.decode(String.self, forKey: .id), kind: kind,
            position: try container.decode(SIMD3<Float>.self, forKey: .position),
            target: try container.decode(SIMD3<Float>.self, forKey: .target),
            color: try container.decode(SIMD3<Float>.self, forKey: .color), emission: emission,
            castsShadow: try container.decode(Bool.self, forKey: .castsShadow),
            isEnabled: try container.decode(Bool.self, forKey: .isEnabled),
            isVisibleToCamera: try container.decode(Bool.self, forKey: .isVisibleToCamera)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        guard iesProfile == nil, cookie == nil else {
            throw EncodingError.invalidValue(self, .init(
                codingPath: encoder.codingPath, debugDescription: "IES profiles and light cookies need an archive resource record."
            ))
        }
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        switch kind {
        case .directional(let diameter):
            try container.encode("directional", forKey: .type)
            try container.encode(diameter, forKey: .angularDiameter)
        case .point(let range):
            try container.encode("point", forKey: .type)
            try container.encodeIfPresent(range, forKey: .range)
        case .spot(let inner, let outer, let range):
            try container.encode("spot", forKey: .type)
            try container.encode(inner, forKey: .innerAngle)
            try container.encode(outer, forKey: .outerAngle)
            try container.encodeIfPresent(range, forKey: .range)
        case .rectangle(let width, let height):
            try container.encode("rectangle", forKey: .type)
            try container.encode(width, forKey: .width)
            try container.encode(height, forKey: .height)
        }
        try container.encode(position, forKey: .position)
        try container.encode(target, forKey: .target)
        try container.encode(color, forKey: .color)
        switch emission {
        case .power(let value): try container.encode(value, forKey: .power)
        case .intensity(let value): try container.encode(value, forKey: .intensity)
        }
        try container.encode(castsShadow, forKey: .castsShadow)
        try container.encode(isEnabled, forKey: .isEnabled)
        try container.encode(isVisibleToCamera, forKey: .isVisibleToCamera)
    }
}

extension LightRig: Codable {
    private enum CodingKeys: String, CodingKey { case lights }

    public init(from decoder: any Decoder) throws {
        let lights = try decoder.container(keyedBy: CodingKeys.self).decode([Light].self, forKey: .lights)
        self = try validated(decoder.codingPath) { try LightRig(lights: lights) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(lights, forKey: .lights)
    }
}

// MARK: Shader graph vocabulary

extension MaterialGraph.ValueType: Codable {}
extension MaterialGraph.Parameter: Codable {}
extension MaterialGraph.Axis: Codable {}
extension MaterialGraph.Stage: Codable {}
extension MaterialGraph.UnaryOperation: Codable {}
extension MaterialGraph.BinaryOperation: Codable {}
extension MaterialGraph.NoiseKind: Codable {}
extension MaterialGraph.Oscillator: Codable {}
extension MaterialGraph.Derivative: Codable {}
extension MaterialGraph.BitOperation: Codable {}
extension MaterialGraph.MatrixOperation: Codable {}
extension ShaderTextureInput.Content: Codable {}
extension ShaderTextureInput.Addressing: Codable {}
extension ShaderTextureInput.Filtering: Codable {}
extension ShaderTextureInput.Shape: Codable {}
extension SurfaceShaderGraph.Channel: Codable {}

extension MaterialGraph.UV: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        switch try container.decode(String.self) {
        case "u": self = .u
        case "v": self = .v
        case let other: throw corrupted(container, "Unknown UV component \(other).", codingPath: decoder.codingPath)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(self == .u ? "u" : "v")
    }
}

extension MaterialGraph.Input: Codable {
    /// Encoded as the input's name, with `elementField(i)` as `"elementField:i"`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let name = try container.decode(String.self)
        let all: [MaterialGraph.Input] = [
            .uv0, .uv1, .worldPosition, .normal, .tangent, .viewDirection, .parameters, .baseColor, .metallic, .roughness,
            .emission, .objectPosition, .objectNormal, .screenUV, .texelSize, .gridUV, .elementIndex, .elementCount,
            .bitangent, .viewportSize, .screenCoordinate, .screenSize, .time, .deltaTime, .frame,
            .cameraPosition, .cameraNear, .cameraFar, .viewMatrix, .projectionMatrix, .modelMatrix, .normalMatrix,
            .directionalShadowVisibility,
        ]
        if let match = all.first(where: { $0.codingName == name }) {
            self = match
        } else if name.hasPrefix("elementField:"), let index = Int(name.dropFirst("elementField:".count)) {
            self = .elementField(index)
        } else {
            throw corrupted(container, "Unknown graph input \(name).", codingPath: decoder.codingPath)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(codingName)
    }

    private var codingName: String {
        switch self {
        case .uv0: "uv0"
        case .uv1: "uv1"
        case .worldPosition: "worldPosition"
        case .normal: "normal"
        case .tangent: "tangent"
        case .viewDirection: "viewDirection"
        case .parameters: "parameters"
        case .baseColor: "baseColor"
        case .metallic: "metallic"
        case .roughness: "roughness"
        case .emission: "emission"
        case .objectPosition: "objectPosition"
        case .objectNormal: "objectNormal"
        case .screenUV: "screenUV"
        case .texelSize: "texelSize"
        case .gridUV: "gridUV"
        case .elementIndex: "elementIndex"
        case .elementCount: "elementCount"
        case .bitangent: "bitangent"
        case .viewportSize: "viewportSize"
        case .screenCoordinate: "screenCoordinate"
        case .screenSize: "screenSize"
        case .time: "time"
        case .deltaTime: "deltaTime"
        case .frame: "frame"
        case .cameraPosition: "cameraPosition"
        case .cameraNear: "cameraNear"
        case .cameraFar: "cameraFar"
        case .directionalShadowVisibility: "directionalShadowVisibility"
        case .viewMatrix: "viewMatrix"
        case .projectionMatrix: "projectionMatrix"
        case .modelMatrix: "modelMatrix"
        case .normalMatrix: "normalMatrix"
        case .elementField(let index): "elementField:\(index)"
        }
    }
}

extension MaterialGraph.Node: Codable {
    private enum CodingKeys: String, CodingKey {
        case node, value, inputs, index, operation, components, noise, octaves, lacunarity, gain, count, exit, slots
        case iterations, skip
        case coordinateSet, transform, reversedDepth
    }

    /// Each node is an object with its `node` kind, its operand node IDs in `inputs`,
    /// and any constant, index or option it carries.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .node)
        let inputs = try container.decodeIfPresent([MaterialGraph.NodeID].self, forKey: .inputs) ?? []
        func operands(_ count: Int) throws -> [MaterialGraph.NodeID] {
            guard inputs.count == count else {
                throw corrupted(container, "Graph node \(kind) needs \(count) inputs.", codingPath: decoder.codingPath)
            }
            return inputs
        }
        switch kind {
        case "scalar": self = .scalar(try container.decode(Float.self, forKey: .value))
        case "integer": self = .integer(Int32(try container.decode(Int32.self, forKey: .value)))
        case "boolean": self = .boolean(try container.decode(Bool.self, forKey: .value))
        case "color": self = .color(try container.decode(SIMD3<Float>.self, forKey: .value))
        case "vector2": self = .vector2(try container.decode(SIMD2<Float>.self, forKey: .value))
        case "vector4": self = .vector4(try container.decode(SIMD4<Float>.self, forKey: .value))
        case "parameter": self = .parameter(try container.decode(MaterialGraph.Parameter.self, forKey: .value))
        case "uv": self = .uv(try container.decode(MaterialGraph.UV.self, forKey: .value))
        case "normal": self = .normal(try container.decode(MaterialGraph.Axis.self, forKey: .value))
        case "worldPosition": self = .worldPosition(try container.decode(MaterialGraph.Axis.self, forKey: .value))
        case "input": self = .input(try container.decode(MaterialGraph.Input.self, forKey: .value))
        case "uniform": self = .uniform(try container.decode(Int.self, forKey: .index))
        case "sampleTexture":
            self = .sampleTexture(slot: try container.decode(Int.self, forKey: .index), coordinates: try operands(1)[0])
        case "sampleTextureLevel":
            let levelInputs = try operands(2)
            self = .sampleTextureLevel(
                slot: try container.decode(Int.self, forKey: .index), coordinates: levelInputs[0], level: levelInputs[1]
            )
        case "sampleTextureGradient":
            let gradientInputs = try operands(3)
            self = .sampleTextureGradient(
                slot: try container.decode(Int.self, forKey: .index),
                coordinates: gradientInputs[0], ddx: gradientInputs[1], ddy: gradientInputs[2]
            )
        case "sampleCube":
            self = .sampleCube(slot: try container.decode(Int.self, forKey: .index), direction: try operands(1)[0])
        case "sampleTexture3D":
            self = .sampleTexture3D(slot: try container.decode(Int.self, forKey: .index), coordinates: try operands(1)[0])
        case "sampleScene": self = .sampleScene(coordinates: try operands(1)[0])
        case "sampleSceneLevel":
            let values = try operands(2)
            self = .sampleSceneLevel(coordinates: values[0], level: values[1])
        case "sampleDepth": self = .sampleDepth(coordinates: try operands(1)[0])
        case "viewportSafeUV":
            let values = try operands(2)
            self = .viewportSafeUV(coordinates: values[0], fallback: values[1])
        case "softParticleOpacity":
            let values = try operands(4)
            self = .softParticleOpacity(opacity: values[0], distance: values[1], contrast: values[2], depth: values[3])
        case "viewPositionFromDepth":
            let x = try operands(3)
            self = .viewPositionFromDepth(coordinates: x[0], depth: x[1], inverseProjection: x[2])
        case "normalFromDepth":
            let x = try operands(2)
            self = .normalFromDepth(
                coordinates: x[0], inverseProjection: x[1],
                reversedDepth: try container.decode(Bool.self, forKey: .reversedDepth)
            )
        case "denoise":
            let x = try operands(4)
            self = .denoise(
                coordinates: x[0], inverseProjection: x[1], settings: x[2], sampleIndex: x[3],
                normalSlot: try container.decodeIfPresent(Int.self, forKey: .index),
                reversedDepth: try container.decode(Bool.self, forKey: .reversedDepth)
            )
        case "linearDepth":
            let x = try operands(5)
            self = .linearDepth(coordinates: x[0], depth: x[1], inverseProjection: x[2], near: x[3], far: x[4])
        case "add": let x = try operands(2); self = .add(x[0], x[1])
        case "subtract": let x = try operands(2); self = .subtract(x[0], x[1])
        case "multiply": let x = try operands(2); self = .multiply(x[0], x[1])
        case "divide": let x = try operands(2); self = .divide(x[0], x[1])
        case "mix": let x = try operands(3); self = .mix(x[0], x[1], factor: x[2])
        case "sine": self = .sine(try operands(1)[0])
        case "cosine": self = .cosine(try operands(1)[0])
        case "floor": self = .floor(try operands(1)[0])
        case "fract": self = .fract(try operands(1)[0])
        case "smoothstep": let x = try operands(3); self = .smoothstep(edge0: x[0], edge1: x[1], value: x[2])
        case "noise": let x = try operands(3); self = .noise(x: x[0], y: x[1], z: x[2])
        case "clamp01": self = .clamp01(try operands(1)[0])
        case "rgb": let x = try operands(3); self = .rgb(red: x[0], green: x[1], blue: x[2])
        case "unary":
            self = .unary(try container.decode(MaterialGraph.UnaryOperation.self, forKey: .operation), try operands(1)[0])
        case "binary":
            let x = try operands(2)
            self = .binary(try container.decode(MaterialGraph.BinaryOperation.self, forKey: .operation), x[0], x[1])
        case "clamp": let x = try operands(3); self = .clamp(x[0], minimum: x[1], maximum: x[2])
        case "select": let x = try operands(3); self = .select(condition: x[0], ifTrue: x[1], ifFalse: x[2])
        case "remap":
            let x = try operands(5)
            self = .remap(x[0], fromMinimum: x[1], fromMaximum: x[2], toMinimum: x[3], toMaximum: x[4])
        case "swizzle": self = .swizzle(try operands(1)[0], try container.decode(String.self, forKey: .components))
        case "combine": self = .combine(inputs)
        case "latticeNoise":
            self = .latticeNoise(try container.decode(MaterialGraph.NoiseKind.self, forKey: .noise), position: try operands(1)[0])
        case "fractalNoise":
            self = .fractalNoise(
                try container.decode(MaterialGraph.NoiseKind.self, forKey: .noise), position: try operands(1)[0],
                octaves: try container.decode(Int.self, forKey: .octaves),
                lacunarity: try container.decode(Float.self, forKey: .lacunarity),
                gain: try container.decode(Float.self, forKey: .gain)
            )
        case "cellularNoise": self = .cellularNoise(position: try operands(1)[0])
        case "simplexNoise": self = .simplexNoise(position: try operands(1)[0])
        case "curlNoise": self = .curlNoise(position: try operands(1)[0])
        case "animatedVoronoi":
            let x = try operands(2)
            self = .animatedVoronoi(position: x[0], time: x[1])
        case "branch":
            let x = try operands(3)
            self = .branch(condition: x[0], ifTrue: x[1], ifFalse: x[2])
        case "loop":
            let x = try operands(2)
            self = .loop(
                count: try container.decode(Int.self, forKey: .count),
                iterations: try container.decodeIfPresent(Int.self, forKey: .iterations),
                initial: x[0], next: x[1],
                exit: try container.decodeIfPresent(Int.self, forKey: .exit),
                skip: try container.decodeIfPresent(Int.self, forKey: .skip)
            )
        case "switch":
            let count = try container.decode(Int.self, forKey: .count)
            let x = try operands(2 + count * 2)
            self = .switchSelect(
                selector: x[0], matches: Array(x[1..<(1 + count)]),
                values: Array(x[(1 + count)..<(1 + count * 2)]), fallback: x[x.count - 1]
            )
        case "discard":
            let x = try operands(2)
            self = .discard(condition: x[0], value: x[1])
        case "loopIndex": self = .loopIndex(loop: try operands(1)[0])
        case "loopValue": self = .loopValue(loop: try operands(1)[0])
        case "oscillator":
            self = .oscillator(try container.decode(MaterialGraph.Oscillator.self, forKey: .operation), try operands(1)[0])
        case "rotateUV":
            let x = try operands(3)
            self = .rotateUV(x[0], angle: x[1], center: x[2])
        case "spherizeUV":
            let x = try operands(3)
            self = .spherizeUV(x[0], strength: x[1], center: x[2])
        case "polar":
            let x = try operands(2)
            self = .polar(x[0], center: x[1])
        case "checker": self = .checker(try operands(1)[0])
        case "triplanarWeights":
            let x = try operands(2)
            self = .triplanarWeights(normal: x[0], sharpness: x[1])
        case "triplanarSample":
            let x = try operands(3)
            self = .triplanarSample(
                slots: try container.decode([Int].self, forKey: .slots), position: x[0], normal: x[1], sharpness: x[2]
            )
        case "parallaxOcclusion":
            let x = try operands(3)
            self = .parallaxOcclusion(
                slot: try container.decode(Int.self, forKey: .index), coordinates: x[0], view: x[1], depth: x[2],
                steps: try container.decode(Int.self, forKey: .count)
            )
        case "bumpMap":
            self = .bumpMap(
                slot: try container.decode(Int.self, forKey: .index),
                coordinateSet: try container.decode(MeshTextureCoordinateSet.self, forKey: .coordinateSet),
                transform: try container.decode(PhysicallyBasedMaterial.TextureTransform.self, forKey: .transform),
                scale: try operands(1)[0]
            )
        case "derivative":
            self = .derivative(try container.decode(MaterialGraph.Derivative.self, forKey: .operation), try operands(1)[0])
        case "bits":
            let x = try operands(2)
            self = .bits(try container.decode(MaterialGraph.BitOperation.self, forKey: .operation), x[0], x[1])
        case "hash": self = .hash(try operands(1)[0])
        case "interleavedGradientNoise": self = .interleavedGradientNoise(try operands(1)[0])
        case "matrix": self = .matrix(inputs)
        case "matrixMultiply":
            let x = try operands(2)
            self = .matrixMultiply(x[0], x[1])
        case "matrixUnary":
            self = .matrixUnary(try container.decode(MaterialGraph.MatrixOperation.self, forKey: .operation), try operands(1)[0])
        case "matrixColumn":
            self = .matrixColumn(try operands(1)[0], try container.decode(Int.self, forKey: .index))
        case "function":
            self = .function(try container.decode(Int.self, forKey: .index), arguments: inputs)
        case "bitcast": self = .bitcast(try operands(1)[0])
        case "countOneBits": self = .countOneBits(try operands(1)[0])
        case "numericCast": self = .numericCast(try operands(1)[0])
        case "constructStruct":
            self = .constructStruct(index: try container.decode(Int.self, forKey: .index), fields: inputs)
        case "structField":
            self = .structField(value: try operands(1)[0], field: try container.decode(Int.self, forKey: .index))
        default: throw corrupted(container, "Unknown graph node \(kind).", codingPath: decoder.codingPath)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        func write(_ kind: String, _ inputs: [MaterialGraph.NodeID] = []) throws {
            try container.encode(kind, forKey: .node)
            if !inputs.isEmpty { try container.encode(inputs, forKey: .inputs) }
        }
        switch self {
        case .scalar(let value): try write("scalar"); try container.encode(value, forKey: .value)
        case .integer(let value): try write("integer"); try container.encode(value, forKey: .value)
        case .boolean(let value): try write("boolean"); try container.encode(value, forKey: .value)
        case .color(let value): try write("color"); try container.encode(value, forKey: .value)
        case .vector2(let value): try write("vector2"); try container.encode(value, forKey: .value)
        case .vector4(let value): try write("vector4"); try container.encode(value, forKey: .value)
        case .parameter(let value): try write("parameter"); try container.encode(value, forKey: .value)
        case .uv(let value): try write("uv"); try container.encode(value, forKey: .value)
        case .normal(let value): try write("normal"); try container.encode(value, forKey: .value)
        case .worldPosition(let value): try write("worldPosition"); try container.encode(value, forKey: .value)
        case .input(let value): try write("input"); try container.encode(value, forKey: .value)
        case .uniform(let index): try write("uniform"); try container.encode(index, forKey: .index)
        case .sampleTexture(let slot, let coordinates):
            try write("sampleTexture", [coordinates]); try container.encode(slot, forKey: .index)
        case .sampleTextureLevel(let slot, let coordinates, let level):
            try write("sampleTextureLevel", [coordinates, level]); try container.encode(slot, forKey: .index)
        case .sampleTextureGradient(let slot, let coordinates, let ddx, let ddy):
            try write("sampleTextureGradient", [coordinates, ddx, ddy]); try container.encode(slot, forKey: .index)
        case .sampleCube(let slot, let direction):
            try write("sampleCube", [direction]); try container.encode(slot, forKey: .index)
        case .sampleTexture3D(let slot, let coordinates):
            try write("sampleTexture3D", [coordinates]); try container.encode(slot, forKey: .index)
        case .sampleScene(let coordinates): try write("sampleScene", [coordinates])
        case .sampleSceneLevel(let coordinates, let level): try write("sampleSceneLevel", [coordinates, level])
        case .sampleDepth(let coordinates): try write("sampleDepth", [coordinates])
        case .viewportSafeUV(let coordinates, let fallback):
            try write("viewportSafeUV", [coordinates, fallback])
        case .softParticleOpacity(let opacity, let distance, let contrast, let depth):
            try write("softParticleOpacity", [opacity, distance, contrast, depth])
        case .viewPositionFromDepth(let coordinates, let depth, let inverse):
            try write("viewPositionFromDepth", [coordinates, depth, inverse])
        case .normalFromDepth(let coordinates, let inverse, let reversed):
            try write("normalFromDepth", [coordinates, inverse])
            try container.encode(reversed, forKey: .reversedDepth)
        case .denoise(let coordinates, let inverse, let settings, let sampleIndex, let normalSlot, let reversed):
            try write("denoise", [coordinates, inverse, settings, sampleIndex])
            try container.encodeIfPresent(normalSlot, forKey: .index)
            try container.encode(reversed, forKey: .reversedDepth)
        case .linearDepth(let coordinates, let depth, let inverse, let near, let far):
            try write("linearDepth", [coordinates, depth, inverse, near, far])
        case .add(let a, let b): try write("add", [a, b])
        case .subtract(let a, let b): try write("subtract", [a, b])
        case .multiply(let a, let b): try write("multiply", [a, b])
        case .divide(let a, let b): try write("divide", [a, b])
        case .mix(let a, let b, let factor): try write("mix", [a, b, factor])
        case .sine(let a): try write("sine", [a])
        case .cosine(let a): try write("cosine", [a])
        case .floor(let a): try write("floor", [a])
        case .fract(let a): try write("fract", [a])
        case .smoothstep(let edge0, let edge1, let value): try write("smoothstep", [edge0, edge1, value])
        case .noise(let x, let y, let z): try write("noise", [x, y, z])
        case .clamp01(let a): try write("clamp01", [a])
        case .rgb(let red, let green, let blue): try write("rgb", [red, green, blue])
        case .unary(let operation, let a): try write("unary", [a]); try container.encode(operation, forKey: .operation)
        case .binary(let operation, let a, let b):
            try write("binary", [a, b]); try container.encode(operation, forKey: .operation)
        case .clamp(let a, let minimum, let maximum): try write("clamp", [a, minimum, maximum])
        case .select(let condition, let ifTrue, let ifFalse): try write("select", [condition, ifTrue, ifFalse])
        case .remap(let a, let fromMinimum, let fromMaximum, let toMinimum, let toMaximum):
            try write("remap", [a, fromMinimum, fromMaximum, toMinimum, toMaximum])
        case .swizzle(let a, let components):
            try write("swizzle", [a]); try container.encode(components, forKey: .components)
        case .combine(let inputs): try write("combine", inputs)
        case .latticeNoise(let kind, let position):
            try write("latticeNoise", [position]); try container.encode(kind, forKey: .noise)
        case .fractalNoise(let kind, let position, let octaves, let lacunarity, let gain):
            try write("fractalNoise", [position])
            try container.encode(kind, forKey: .noise)
            try container.encode(octaves, forKey: .octaves)
            try container.encode(lacunarity, forKey: .lacunarity)
            try container.encode(gain, forKey: .gain)
        case .cellularNoise(let position): try write("cellularNoise", [position])
        case .simplexNoise(let position): try write("simplexNoise", [position])
        case .curlNoise(let position): try write("curlNoise", [position])
        case .animatedVoronoi(let position, let time): try write("animatedVoronoi", [position, time])
        case .branch(let condition, let yes, let no): try write("branch", [condition, yes, no])
        case .switchSelect(let selector, let matches, let values, let fallback):
            try write("switch", [selector] + matches + values + [fallback])
            try container.encode(matches.count, forKey: .count)
        case .loop(let count, let iterations, let initial, let next, let exit, let skip):
            try write("loop", [initial, next])
            try container.encode(count, forKey: .count)
            try container.encodeIfPresent(exit, forKey: .exit)
            try container.encodeIfPresent(iterations, forKey: .iterations)
            try container.encodeIfPresent(skip, forKey: .skip)
        case .discard(let condition, let value): try write("discard", [condition, value])
        case .loopIndex(let loop): try write("loopIndex", [loop])
        case .loopValue(let loop): try write("loopValue", [loop])
        case .oscillator(let operation, let value):
            try write("oscillator", [value])
            try container.encode(operation, forKey: .operation)
        case .rotateUV(let uv, let angle, let center): try write("rotateUV", [uv, angle, center])
        case .spherizeUV(let uv, let strength, let center): try write("spherizeUV", [uv, strength, center])
        case .polar(let uv, let center): try write("polar", [uv, center])
        case .checker(let uv): try write("checker", [uv])
        case .triplanarWeights(let normal, let sharpness): try write("triplanarWeights", [normal, sharpness])
        case .triplanarSample(let slots, let position, let normal, let sharpness):
            try write("triplanarSample", [position, normal, sharpness])
            try container.encode(slots, forKey: .slots)
        case .parallaxOcclusion(let slot, let coordinates, let view, let depth, let steps):
            try write("parallaxOcclusion", [coordinates, view, depth])
            try container.encode(slot, forKey: .index)
            try container.encode(steps, forKey: .count)
        case .bumpMap(let slot, let set, let transform, let scale):
            try write("bumpMap", [scale])
            try container.encode(slot, forKey: .index)
            try container.encode(set, forKey: .coordinateSet)
            try container.encode(transform, forKey: .transform)
        case .derivative(let operation, let value):
            try write("derivative", [value])
            try container.encode(operation, forKey: .operation)
        case .bits(let operation, let a, let b):
            try write("bits", [a, b])
            try container.encode(operation, forKey: .operation)
        case .hash(let value): try write("hash", [value])
        case .interleavedGradientNoise(let value): try write("interleavedGradientNoise", [value])
        case .matrix(let columns): try write("matrix", columns)
        case .matrixMultiply(let a, let b): try write("matrixMultiply", [a, b])
        case .matrixUnary(let operation, let value):
            try write("matrixUnary", [value])
            try container.encode(operation, forKey: .operation)
        case .matrixColumn(let matrix, let index):
            try write("matrixColumn", [matrix])
            try container.encode(index, forKey: .index)
        case .function(let index, let arguments):
            try write("function", arguments); try container.encode(index, forKey: .index)
        case .bitcast(let value): try write("bitcast", [value])
        case .countOneBits(let value): try write("countOneBits", [value])
        case .numericCast(let value): try write("numericCast", [value])
        case .constructStruct(let index, let fields):
            try write("constructStruct", fields)
            try container.encode(index, forKey: .index)
        case .structField(let value, let field):
            try write("structField", [value])
            try container.encode(field, forKey: .index)
        }
    }
}

extension ShaderUniform: Codable {
    private enum CodingKeys: String, CodingKey { case name, type, defaultValue, columns, integerValue }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let name = try container.decode(String.self, forKey: .name)
        let type = try container.decode(MaterialGraph.ValueType.self, forKey: .type)
        switch type {
        case .int:
            self.init(name, try container.decode(Int32.self, forKey: .integerValue))
        case .structure:
            throw corrupted(container, "A struct is not a uniform type.", codingPath: decoder.codingPath)
        case .scalar, .bool, .vector2, .vector3, .vector4, .matrix3, .matrix4:
            let header = try container.decode(SIMD4<Float>.self, forKey: .defaultValue)
            switch type {
            case .scalar: self.init(name, header.x)
            case .bool: self.init(name, header.x != 0)
            case .vector2: self.init(name, SIMD2(header.x, header.y))
            case .vector3: self.init(name, SIMD3(header.x, header.y, header.z))
            case .vector4: self.init(name, header)
            case .matrix3, .matrix4:
                let columns = try container.decode([SIMD4<Float>].self, forKey: .columns)
                guard columns.count == type.uniformSlotCount else {
                    throw corrupted(
                        container, "A \(type.rawValue) uniform needs \(type.uniformSlotCount) columns.",
                        codingPath: decoder.codingPath
                    )
                }
                self.init(name: name, type: type, columns: columns)
                guard defaultValue == type.storedColumn(header) else {
                    throw corrupted(
                        container, "A matrix uniform's defaultValue must equal its first column.",
                        codingPath: decoder.codingPath
                    )
                }
            case .int:
                throw corrupted(container, "An integer uniform is stored as integerValue.", codingPath: decoder.codingPath)
            case .structure:
                throw corrupted(container, "A struct is not a uniform type.", codingPath: decoder.codingPath)
            }
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(type, forKey: .type)
        if type == .int {
            try container.encode(Int32(bitPattern: defaultValue.x.bitPattern), forKey: .integerValue)
            return
        }
        try container.encode(defaultValue, forKey: .defaultValue)
        if type.isMatrix { try container.encode(columns, forKey: .columns) }
    }
}

extension ShaderTextureInput: Codable {
    private enum CodingKeys: String, CodingKey { case name, content, addressing, filtering, shape }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            try container.decode(String.self, forKey: .name),
            content: try container.decode(Content.self, forKey: .content),
            addressing: try container.decode(Addressing.self, forKey: .addressing),
            filtering: try container.decode(Filtering.self, forKey: .filtering),
            shape: try container.decodeIfPresent(Shape.self, forKey: .shape) ?? .twoDimensional
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(content, forKey: .content)
        try container.encode(addressing, forKey: .addressing)
        try container.encode(filtering, forKey: .filtering)
        if shape != .twoDimensional { try container.encode(shape, forKey: .shape) }
    }
}

extension ShaderFunction.Parameter: Codable {
    private enum CodingKeys: String, CodingKey { case name, type }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            try container.decode(String.self, forKey: .name), try container.decode(MaterialGraph.ValueType.self, forKey: .type)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(type, forKey: .type)
    }
}

extension ShaderFunction: Codable {
    private enum CodingKeys: String, CodingKey { case name, parameters, returnType, body }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            name: try container.decode(String.self, forKey: .name),
            parameters: try container.decode([Parameter].self, forKey: .parameters),
            returns: try container.decode(MaterialGraph.ValueType.self, forKey: .returnType),
            body: try container.decode(String.self, forKey: .body)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(parameters, forKey: .parameters)
        try container.encode(returnType, forKey: .returnType)
        try container.encode(body, forKey: .body)
    }
}

/// The declarations every graph kind shares, encoded with the same keys.
private enum GraphKeys: String, CodingKey {
    case nodes, uniforms, textures, functions, structs, labels, output, outputs, fieldCount
}

private struct GraphDeclarations {
    var nodes: [MaterialGraph.Node]
    var uniforms: [ShaderUniform]
    var textures: [ShaderTextureInput]
    var functions: [ShaderFunction]
    var structs: [ShaderStruct]
    var labels: [MaterialGraph.NodeID: String]

    init(_ container: KeyedDecodingContainer<GraphKeys>) throws {
        nodes = try container.decode([MaterialGraph.Node].self, forKey: .nodes)
        uniforms = try container.decodeIfPresent([ShaderUniform].self, forKey: .uniforms) ?? []
        textures = try container.decodeIfPresent([ShaderTextureInput].self, forKey: .textures) ?? []
        functions = try container.decodeIfPresent([ShaderFunction].self, forKey: .functions) ?? []
        structs = try container.decodeIfPresent([ShaderStruct].self, forKey: .structs) ?? []
        structs = structs.enumerated().map { index, structure in
            ShaderStruct(index: index, name: structure.name, fields: structure.fields)
        }
        labels = try container.decodeIfPresent([MaterialGraph.NodeID: String].self, forKey: .labels) ?? [:]
    }

    static func encode(
        _ container: inout KeyedEncodingContainer<GraphKeys>, nodes: [MaterialGraph.Node], uniforms: [ShaderUniform],
        textures: [ShaderTextureInput], functions: [ShaderFunction], structs: [ShaderStruct],
        labels: [MaterialGraph.NodeID: String]
    ) throws {
        try container.encode(nodes, forKey: .nodes)
        if !uniforms.isEmpty { try container.encode(uniforms, forKey: .uniforms) }
        if !textures.isEmpty { try container.encode(textures, forKey: .textures) }
        if !functions.isEmpty { try container.encode(functions, forKey: .functions) }
        if !structs.isEmpty { try container.encode(structs, forKey: .structs) }
        if !labels.isEmpty { try container.encode(labels, forKey: .labels) }
    }
}

extension SurfaceShaderGraph.Outputs: Codable {
    private enum CodingKeys: String, CodingKey {
        case baseColor, metallic, roughness, emission, transmissionMask, normal, occlusion, positionOffset, pointSize, opacity, alphaTest, mask
        case castShadowColor, castShadowMask
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            baseColor: try container.decodeIfPresent(Int.self, forKey: .baseColor),
            metallic: try container.decodeIfPresent(Int.self, forKey: .metallic),
            roughness: try container.decodeIfPresent(Int.self, forKey: .roughness),
            emission: try container.decodeIfPresent(Int.self, forKey: .emission),
            transmissionMask: try container.decodeIfPresent(Int.self, forKey: .transmissionMask),
            normal: try container.decodeIfPresent(Int.self, forKey: .normal),
            occlusion: try container.decodeIfPresent(Int.self, forKey: .occlusion),
            positionOffset: try container.decodeIfPresent(Int.self, forKey: .positionOffset),
            pointSize: try container.decodeIfPresent(Int.self, forKey: .pointSize),
            opacity: try container.decodeIfPresent(Int.self, forKey: .opacity),
            alphaTest: try container.decodeIfPresent(Int.self, forKey: .alphaTest),
            mask: try container.decodeIfPresent(Int.self, forKey: .mask),
            castShadowColor: try container.decodeIfPresent(Int.self, forKey: .castShadowColor),
            castShadowMask: try container.decodeIfPresent(Int.self, forKey: .castShadowMask)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(baseColor, forKey: .baseColor)
        try container.encodeIfPresent(metallic, forKey: .metallic)
        try container.encodeIfPresent(roughness, forKey: .roughness)
        try container.encodeIfPresent(emission, forKey: .emission)
        try container.encodeIfPresent(transmissionMask, forKey: .transmissionMask)
        try container.encodeIfPresent(normal, forKey: .normal)
        try container.encodeIfPresent(occlusion, forKey: .occlusion)
        try container.encodeIfPresent(positionOffset, forKey: .positionOffset)
        try container.encodeIfPresent(pointSize, forKey: .pointSize)
        try container.encodeIfPresent(opacity, forKey: .opacity)
        try container.encodeIfPresent(alphaTest, forKey: .alphaTest)
        try container.encodeIfPresent(mask, forKey: .mask)
        try container.encodeIfPresent(castShadowColor, forKey: .castShadowColor)
        try container.encodeIfPresent(castShadowMask, forKey: .castShadowMask)
    }
}

extension SurfaceShaderGraph: Codable {
    /// Decoding validates the graph exactly as `init(nodes:uniforms:textures:functions:labels:outputs:)` does.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: GraphKeys.self)
        let parts = try GraphDeclarations(container)
        let outputs = try container.decode(Outputs.self, forKey: .outputs)
        self = try validated(decoder.codingPath) {
            try SurfaceShaderGraph(
                nodes: parts.nodes, uniforms: parts.uniforms, textures: parts.textures, functions: parts.functions,
                structs: parts.structs, labels: parts.labels, outputs: outputs
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: GraphKeys.self)
        try GraphDeclarations.encode(
            &container, nodes: nodes, uniforms: uniforms, textures: textures, functions: functions, structs: structs,
            labels: labels
        )
        try container.encode(outputs, forKey: .outputs)
    }
}

extension PostProcessGraph: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: GraphKeys.self)
        let parts = try GraphDeclarations(container)
        let output = try container.decode(Int.self, forKey: .output)
        self = try validated(decoder.codingPath) {
            try PostProcessGraph(
                nodes: parts.nodes, uniforms: parts.uniforms, textures: parts.textures, functions: parts.functions,
                structs: parts.structs, labels: parts.labels, output: output
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: GraphKeys.self)
        try GraphDeclarations.encode(
            &container, nodes: nodes, uniforms: uniforms, textures: textures, functions: functions, structs: structs,
            labels: labels
        )
        try container.encode(output, forKey: .output)
    }
}

extension TextureShaderGraph: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: GraphKeys.self)
        let parts = try GraphDeclarations(container)
        let output = try container.decode(Int.self, forKey: .output)
        self = try validated(decoder.codingPath) {
            try TextureShaderGraph(
                nodes: parts.nodes, uniforms: parts.uniforms, textures: parts.textures, functions: parts.functions,
                structs: parts.structs, labels: parts.labels, output: output
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: GraphKeys.self)
        try GraphDeclarations.encode(
            &container, nodes: nodes, uniforms: uniforms, textures: textures, functions: functions, structs: structs,
            labels: labels
        )
        try container.encode(output, forKey: .output)
    }
}

extension BufferShaderGraph: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: GraphKeys.self)
        let parts = try GraphDeclarations(container)
        let fieldCount = try container.decode(Int.self, forKey: .fieldCount)
        let outputs = try container.decode([Int: MaterialGraph.NodeID].self, forKey: .outputs)
        self = try validated(decoder.codingPath) {
            try BufferShaderGraph(
                nodes: parts.nodes, fieldCount: fieldCount, uniforms: parts.uniforms, textures: parts.textures,
                functions: parts.functions, structs: parts.structs, labels: parts.labels, outputs: outputs
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: GraphKeys.self)
        try GraphDeclarations.encode(
            &container, nodes: nodes, uniforms: uniforms, textures: textures, functions: functions, structs: structs,
            labels: labels
        )
        try container.encode(fieldCount, forKey: .fieldCount)
        try container.encode(outputs, forKey: .outputs)
    }
}

extension MaterialGraph: Codable {
    private enum CodingKeys: String, CodingKey { case nodes, output }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let nodes = try container.decode([Node].self, forKey: .nodes)
        let output = try container.decode(Int.self, forKey: .output)
        self = try validated(decoder.codingPath) { try MaterialGraph(nodes: nodes, output: output) }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(nodes, forKey: .nodes)
        try container.encode(output, forKey: .output)
    }
}
