import Foundation
import Metal
import simd

/// Fixed-function state for a mesh's color pass, like the base properties of a
/// Three.js `Material`: blending, depth and color writes, polygon offset, face
/// side, stencil, alpha hashing, alpha to coverage and transparent sorting.
///
/// Fragments write premultiplied linear RGBA, so the named blend modes use the
/// premultiplied factors. With source color `S`, source alpha `a` and destination `D`:
///
/// | Mode | RGB | Alpha |
/// |---|---|---|
/// | `normal` | `S + D (1 − a)` | `a + Dₐ (1 − a)` |
/// | `additive` | `S + D` | `a + Dₐ` |
/// | `subtractive` | `D (1 − S)` | `Dₐ` |
/// | `multiply` | `S D + D (1 − a)` | `Dₐ` |
///
/// Subtractive darkens the destination by the source color, as Three.js
/// `SubtractiveBlending` does; use `.custom` with `.reverseSubtract` for a
/// difference. Multiply equals `D × mix(1, color, a)` for premultiplied input.
/// Depth comparisons are stated for forward depth; a renderer prepared with
/// `.reversed` depth mirrors them, so `.less` still means nearer. A polygon offset
/// moves depth away from the camera for positive values in either depth mode.
public struct MeshRenderState: Equatable, Sendable {
    /// Which faces rasterize, like Three.js `FrontSide`, `BackSide` and `DoubleSide`.
    /// Lit materials shade back faces with the flipped normal. Scene picking
    /// respects the same authored side, including when a transform is mirrored.
    public enum Side: Equatable, Sendable {
        case front
        case back
        case double
    }

    /// Separate RGB and alpha blend equations, like Three.js `CustomBlending` with
    /// `blendSrc`, `blendDst`, `blendEquation` and their alpha counterparts.
    public struct BlendFunction: Equatable, Sendable {
        public var sourceRGB: MTLBlendFactor
        public var destinationRGB: MTLBlendFactor
        public var rgbOperation: MTLBlendOperation
        public var sourceAlpha: MTLBlendFactor
        public var destinationAlpha: MTLBlendFactor
        public var alphaOperation: MTLBlendOperation
        /// Linear RGBA for the blend-color factors, like Three.js `blendColor` and `blendAlpha`.
        public var constantColor: SIMD4<Float>

        public init(
            sourceRGB: MTLBlendFactor, destinationRGB: MTLBlendFactor, rgbOperation: MTLBlendOperation = .add,
            sourceAlpha: MTLBlendFactor? = nil, destinationAlpha: MTLBlendFactor? = nil,
            alphaOperation: MTLBlendOperation? = nil, constantColor: SIMD4<Float> = .zero
        ) {
            self.sourceRGB = sourceRGB
            self.destinationRGB = destinationRGB
            self.rgbOperation = rgbOperation
            self.sourceAlpha = sourceAlpha ?? sourceRGB
            self.destinationAlpha = destinationAlpha ?? destinationRGB
            self.alphaOperation = alphaOperation ?? rgbOperation
            self.constantColor = constantColor
        }
    }

    /// The color blend equation. See the table on ``MeshRenderState``.
    public enum Blending: Equatable, Sendable {
        /// Overwrites the destination.
        case none
        case normal
        case additive
        case subtractive
        case multiply
        case custom(BlendFunction)

        /// The equation, or nil for `.none`.
        public var function: BlendFunction? {
            switch self {
            case .none: nil
            case .normal:
                BlendFunction(sourceRGB: .one, destinationRGB: .oneMinusSourceAlpha)
            case .additive:
                BlendFunction(sourceRGB: .one, destinationRGB: .one)
            case .subtractive:
                BlendFunction(
                    sourceRGB: .zero, destinationRGB: .oneMinusSourceColor, sourceAlpha: .zero, destinationAlpha: .one
                )
            case .multiply:
                BlendFunction(
                    sourceRGB: .destinationColor, destinationRGB: .oneMinusSourceAlpha,
                    sourceAlpha: .zero, destinationAlpha: .one
                )
            case .custom(let function): function
            }
        }
    }

    /// Depth bias like OpenGL `glPolygonOffset` and Three.js `polygonOffsetFactor`
    /// and `polygonOffsetUnits`: `factor` scales the polygon's depth slope and `units`
    /// the smallest resolvable depth step. A nonzero `clamp` bounds the total bias.
    public struct DepthOffset: Equatable, Sendable {
        public var factor: Float
        public var units: Float
        public var clamp: Float

        public init(factor: Float, units: Float, clamp: Float = 0) {
            self.factor = factor
            self.units = units
            self.clamp = clamp
        }
    }

    /// An enabled 8-bit stencil test, like Three.js material stencil state.
    /// The same comparison and operations apply to both faces.
    public struct Stencil: Equatable, Sendable {
        public var reference: UInt8
        /// Compares the masked reference with the masked stored stencil value.
        public var compare: MTLCompareFunction
        public var readMask: UInt8
        public var writeMask: UInt8
        public var stencilFailure: MTLStencilOperation
        public var depthFailure: MTLStencilOperation
        public var pass: MTLStencilOperation

        public init(
            reference: UInt8 = 0, compare: MTLCompareFunction = .always,
            readMask: UInt8 = .max, writeMask: UInt8 = .max,
            stencilFailure: MTLStencilOperation = .keep, depthFailure: MTLStencilOperation = .keep,
            pass: MTLStencilOperation = .keep
        ) {
            self.reference = reference
            self.compare = compare
            self.readMask = readMask
            self.writeMask = writeMask
            self.stencilFailure = stencilFailure
            self.depthFailure = depthFailure
            self.pass = pass
        }

        var isValid: Bool {
            // Swift can construct imported Metal enum values outside their named cases.
            switch compare {
            case .never, .less, .equal, .lessEqual, .greater, .notEqual, .greaterEqual, .always:
                break
            @unknown default:
                return false
            }
            return [stencilFailure, depthFailure, pass].allSatisfy { operation in
                switch operation {
                case .keep, .zero, .replace, .incrementClamp, .decrementClamp, .invert, .incrementWrap, .decrementWrap:
                    true
                @unknown default:
                    false
                }
            }
        }
    }

    /// Nil disables stencil tests and writes. An enabled state prepares the
    /// renderer for a combined Depth32Float_Stencil8 color pass.
    public var stencil: Stencil?
    public var blending: Blending
    /// Draws after every non-transparent mesh, sorted by render order and then
    /// back to front by projected origin, like Three.js `transparent`. Transparent
    /// meshes do not cast shadows.
    public var transparent: Bool
    /// When false, every fragment passes and none writes depth, like Three.js
    /// `depthTest` under `WebGLRenderer`, which disables `GL_DEPTH_TEST` and with it
    /// depth writes. For fragments that all pass and still write depth (the
    /// `WebGPURenderer` behavior), keep this true and set `depthCompare` to `.always`.
    public var depthTest: Bool
    /// Writes passing fragments' depth, like Three.js `depthWrite`. Has no effect
    /// while `depthTest` is false.
    public var depthWrite: Bool
    /// The forward-depth comparison, like Three.js `depthFunc`. Frostfield's default is
    /// `.less`; Three.js defaults to `LessEqualDepth`.
    public var depthCompare: MTLCompareFunction
    /// Like Three.js `colorWrite`, per channel.
    public var colorWriteMask: MTLColorWriteMask
    public var depthOffset: DepthOffset?
    public var side: Side
    /// Which faces cast shadows. Nil keeps both sides, the shadow pass used before this
    /// setting existed. Three.js `shadowSide` null instead follows `side`.
    public var shadowSide: Side?
    /// Discards each fragment whose alpha is below a stable hashed threshold and
    /// writes the survivors opaque, like Three.js `alphaHash`. The threshold follows
    /// Wyman and McGuire's hashed alpha test on world position at pixel scale, so a
    /// still view keeps the same pattern. Supported by PBR and ``ClassicMaterial``.
    public var alphaHash: Bool
    /// Converts fragment alpha to MSAA sample coverage, like Three.js
    /// `alphaToCoverage`. Needs a multisampled renderer and `blending` `.none`: PBR and
    /// ``ClassicMaterial`` write straight (unpremultiplied) color so the resolve blends
    /// it, and blending as well would apply alpha twice. Covered samples then store
    /// alpha one, so a resolved pixel is premultiplied color with alpha equal to its
    /// coverage; Three.js under WebGL, which lacks alpha-to-one, resolves alpha × coverage. With an alpha cutoff (PBR
    /// `.mask` or a classic `alphaTest`), coverage is Three.js's
    /// `smoothstep(cutoff, cutoff + fwidth(a), a)`: solid inside, antialiased over
    /// about one pixel at the cutout edge, and discarded only at zero.
    public var alphaToCoverage: Bool

    public init(
        blending: Blending = .none, transparent: Bool = false, depthTest: Bool = true, depthWrite: Bool = true,
        depthCompare: MTLCompareFunction = .less, colorWriteMask: MTLColorWriteMask = .all,
        depthOffset: DepthOffset? = nil, side: Side = .front, shadowSide: Side? = nil,
        alphaHash: Bool = false, alphaToCoverage: Bool = false, stencil: Stencil? = nil
    ) {
        self.blending = blending
        self.transparent = transparent
        self.depthTest = depthTest
        self.depthWrite = depthWrite
        self.depthCompare = depthCompare
        self.colorWriteMask = colorWriteMask
        self.depthOffset = depthOffset
        self.side = side
        self.shadowSide = shadowSide
        self.alphaHash = alphaHash
        self.alphaToCoverage = alphaToCoverage
        self.stencil = stencil
    }

    /// Opaque front faces with depth test and writes.
    public static let opaque = MeshRenderState()
    /// Premultiplied normal blending, sorted back to front, depth tested without writes.
    /// PBR `.blend` materials use it by default.
    public static let transparent = MeshRenderState(blending: .normal, transparent: true, depthWrite: false)

    /// The state a renderer uses when none is supplied.
    static func defaultState(for material: MeshMaterial) -> MeshRenderState {
        switch material {
        case .physicallyBased(let material):
            var state: MeshRenderState = material.alphaMode == .blend ? .transparent : .opaque
            state.side = material.isDoubleSided ? .double : .front
            return state
        case .classic(let material):
            return material.defaultRenderState
        default:
            return .opaque
        }
    }

    func validate(sampleCount: Int, supportsAlphaControls: Bool) throws {
        guard stencil?.isValid != false else {
            throw MeshRenderer.Failure.renderState("Use known Metal stencil comparison and operation cases.")
        }
        if let offset = depthOffset {
            guard offset.factor.isFinite, offset.units.isFinite, offset.clamp.isFinite else {
                throw MeshRenderer.Failure.renderState("Polygon offset values must be finite.")
            }
        }
        if case .custom(let function) = blending {
            let color = function.constantColor
            guard color.x.isFinite, color.y.isFinite, color.z.isFinite, color.w.isFinite else {
                throw MeshRenderer.Failure.renderState("The blend constant color must be finite.")
            }
        }
        if alphaHash || alphaToCoverage {
            guard supportsAlphaControls else {
                throw MeshRenderer.Failure.renderState(
                    "Alpha hashing and alpha to coverage need a physically based or classic material."
                )
            }
            guard !(alphaHash && alphaToCoverage) else {
                throw MeshRenderer.Failure.renderState("Use either alpha hashing or alpha to coverage, not both.")
            }
        }
        if alphaToCoverage, sampleCount < 2 {
            throw MeshRenderer.Failure.renderState("Alpha to coverage needs a renderer with a sample count above one.")
        }
        if alphaToCoverage, blending != .none {
            throw MeshRenderer.Failure.renderState(
                "Alpha to coverage writes straight color for the multisample resolve; use it with blending .none."
            )
        }
    }

    /// Applies blending, the write mask and alpha to coverage to a pipeline descriptor.
    func configure(_ descriptor: MTLRenderPipelineDescriptor) {
        let color = descriptor.colorAttachments[0]!
        color.writeMask = colorWriteMask
        if let function = blending.function {
            color.isBlendingEnabled = true
            color.sourceRGBBlendFactor = function.sourceRGB
            color.destinationRGBBlendFactor = function.destinationRGB
            color.rgbBlendOperation = function.rgbOperation
            color.sourceAlphaBlendFactor = function.sourceAlpha
            color.destinationAlphaBlendFactor = function.destinationAlpha
            color.alphaBlendOperation = function.alphaOperation
        }
        descriptor.isAlphaToCoverageEnabled = alphaToCoverage
        // After coverage, covered samples store alpha one, so the resolve is premultiplied
        // with alpha equal to coverage. WebGL has no alpha-to-one, so Three.js resolves
        // alpha × coverage there.
        descriptor.isAlphaToOneEnabled = alphaToCoverage
    }

    /// The depth state for a renderer's depth mode.
    func depthDescriptor(depthMode: MeshDepthMode) -> MTLDepthStencilDescriptor {
        let descriptor = MTLDepthStencilDescriptor()
        descriptor.depthCompareFunction = depthTest ? Self.mirrored(depthCompare, for: depthMode) : .always
        // As in WebGL, turning the depth test off also turns depth writes off.
        descriptor.isDepthWriteEnabled = depthTest && depthWrite
        if let stencil {
            let face = MTLStencilDescriptor()
            face.stencilCompareFunction = stencil.compare
            face.readMask = UInt32(stencil.readMask)
            face.writeMask = UInt32(stencil.writeMask)
            face.stencilFailureOperation = stencil.stencilFailure
            face.depthFailureOperation = stencil.depthFailure
            face.depthStencilPassOperation = stencil.pass
            descriptor.frontFaceStencil = face
            descriptor.backFaceStencil = face
        }
        return descriptor
    }

    var cullMode: MTLCullMode {
        switch side {
        case .front: .back
        case .back: .front
        case .double: .none
        }
    }

    /// Nil casts with both sides. A set value uses the same winding rule as `cullMode`.
    var shadowCullMode: MTLCullMode {
        switch shadowSide {
        case nil, .double?: .none
        case .front?: .back
        case .back?: .front
        }
    }

    /// Whether the fragment should write the premultiplied result of a blend.
    var blends: Bool { blending != .none }

    private static func mirrored(_ function: MTLCompareFunction, for depthMode: MeshDepthMode) -> MTLCompareFunction {
        guard depthMode == .reversed else { return function }
        switch function {
        case .less: return .greater
        case .lessEqual: return .greaterEqual
        case .greater: return .less
        case .greaterEqual: return .lessEqual
        default: return function
        }
    }
}
