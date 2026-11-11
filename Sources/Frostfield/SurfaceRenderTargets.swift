import Foundation
import Metal

/// A named layout for a PBR color pass and its secondary surface textures.
public struct SurfaceRenderTargets: Hashable, Sendable {
    public enum Value: Hashable, Sendable {
        /// Sampled linear base color before lighting, with straight alpha.
        case baseColor
        /// The final signed world-space shading normal. The fourth component is one.
        case worldNormal
        /// Sampled linear emission. The fourth component is one.
        case emission
        case metallic
        case roughness
        case occlusion

        private var isVector: Bool {
            switch self {
            case .baseColor, .worldNormal, .emission: true
            case .metallic, .roughness, .occlusion: false
            }
        }

        fileprivate func accepts(_ format: MTLPixelFormat) -> Bool {
            isVector ? [.rgba16Float, .rgba32Float].contains(format) : [.r16Float, .r32Float].contains(format)
        }

        fileprivate var metalType: String { isVector ? "float4" : "float" }

        fileprivate var metalExpression: String {
            switch self {
            case .baseColor: "(surface).base"
            case .worldNormal: "float4((surface).normal, 1.0f)"
            case .emission: "float4((surface).emission, 1.0f)"
            case .metallic: "(surface).metallic"
            case .roughness: "(surface).roughness"
            case .occlusion: "(surface).occlusion"
            }
        }
    }

    public struct Output: Hashable, Sendable {
        public let name: String
        public let value: Value
        public let pixelFormat: MTLPixelFormat

        public init(_ name: String, value: Value, pixelFormat: MTLPixelFormat) {
            self.name = name
            self.value = value
            self.pixelFormat = pixelFormat
        }
    }

    /// The premultiplied PBR color texture at slot zero is named `output`.
    public static let colorName = "output"
    public let colorPixelFormat: MTLPixelFormat
    /// Named secondary slots in attachment order, starting at slot one.
    public let outputs: [Output]

    /// Supply one to seven unique secondary names. Vector values use RGBA16Float
    /// or RGBA32Float, and scalar values use R16Float or R32Float.
    public init(colorPixelFormat: MTLPixelFormat = .rgba16Float, outputs: [Output]) throws {
        guard (1...7).contains(outputs.count), outputs.allSatisfy({ !$0.name.isEmpty && $0.name != Self.colorName }),
              Set(outputs.map(\.name)).count == outputs.count else {
            throw Failure.names
        }
        let colorFormats: [MTLPixelFormat] = [.rgba16Float, .rgba32Float, .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb]
        guard colorFormats.contains(colorPixelFormat) else { throw Failure.format(Self.colorName) }
        for output in outputs where !output.value.accepts(output.pixelFormat) { throw Failure.format(output.name) }
        self.colorPixelFormat = colorPixelFormat
        self.outputs = outputs
    }

    public struct Pass {
        public let targets: SurfaceRenderTargets
        let descriptor: MTLRenderPassDescriptor
    }

    /// Maps independently configured Metal attachments by name. Use equal-sized,
    /// persistent single-sample 2D textures and a matching Depth32Float attachment.
    /// Metal copies the descriptors, and their textures remain the caller's.
    public func makePass(
        colorAttachments: [String: MTLRenderPassColorAttachmentDescriptor],
        depthAttachment: MTLRenderPassDepthAttachmentDescriptor
    ) throws -> Pass {
        let names = [Self.colorName] + outputs.map(\.name)
        guard Set(colorAttachments.keys) == Set(names) else { throw Failure.attachments }
        let descriptor = MTLRenderPassDescriptor()
        for (index, pair) in zip(names, colorPixelFormats).enumerated() {
            let (name, format) = pair
            guard let attachment = colorAttachments[name], attachment.texture?.pixelFormat == format else {
                throw Failure.format(name)
            }
            descriptor.colorAttachments[index] = attachment
        }
        descriptor.depthAttachment = depthAttachment
        return Pass(targets: self, descriptor: descriptor)
    }

    var colorPixelFormats: [MTLPixelFormat] { [colorPixelFormat] + outputs.map(\.pixelFormat) }

    var metalSource: String {
        let fields = outputs.enumerated().map { index, output in
            "    \(output.value.metalType) value\(index) [[color(\(index + 1))]];"
        }.joined(separator: "\n")
        let values = outputs.map { $0.value.metalExpression }.joined(separator: ", ")
        return """
        struct FrostfieldPBRNamedOutput {
            float4 primary [[color(0)]];
        \(fields)
        };
        #define FROSTFIELD_PBR_COLOR_OUTPUT FrostfieldPBRNamedOutput
        #define FROSTFIELD_PBR_COLOR_RESULT(color, surface) FrostfieldPBRNamedOutput{(color), \(values)}
        """
    }

    public enum Failure: LocalizedError, Equatable {
        case names
        case format(String)
        case attachments
        case material
        case rendererLayout

        public var errorDescription: String? {
            switch self {
            case .names: "Use one to seven unique nonempty secondary output names, separate from output."
            case .format(let name): "The pixel format does not match the surface output named \(name)."
            case .attachments: "Supply one color attachment per surface output name."
            case .material: "Named surface outputs require an unblended PBR material without transmission."
            case .rendererLayout: "Prepare every mesh with this pass's named surface layout."
            }
        }
    }
}

public extension MeshRenderer {
    /// Encodes a prepared mesh against the same named layout used to create it.
    func encode(
        _ scene: MeshScene, pass: SurfaceRenderTargets.Pass, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard surfaceRenderTargets == pass.targets else { throw SurfaceRenderTargets.Failure.rendererLayout }
        try encode(scene, pass: pass.descriptor, commandBuffer: commandBuffer)
    }
}

public extension SceneGraph {
    /// Encodes PBR color and all named surface textures using the graph camera.
    /// Every visible mesh uses the pass's named layout and depth mode. A scene
    /// override supplies that pipeline for each selected mesh.
    /// Clear depth to 1 for forward depth or 0 for reversed depth.
    @discardableResult
    func encode(
        pass: SurfaceRenderTargets.Pass, commandBuffer: any MTLCommandBuffer,
        shadowMap: DirectionalShadowMap? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        depthMode: MeshDepthMode = .forward, scissor: MTLScissorRect? = nil, indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneDrawStatistics {
        try MeshRenderer.encode(
            self, pass: pass.descriptor, commandBuffer: commandBuffer, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps, depthMode: depthMode, scissor: scissor, indirectDraws: indirectDraws,
            surfaceRenderTargets: pass.targets
        )
    }
}
