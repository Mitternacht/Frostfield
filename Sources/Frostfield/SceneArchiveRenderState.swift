import Metal
import simd

/// Archive representation of Metal's fixed-function state. Raw values are
/// checked before they are handed to Metal.
struct ArchiveMeshState: Codable {
    enum Side: String, Codable { case front, back, double }
    enum Blending: String, Codable { case none, normal, additive, subtractive, multiply, custom }

    struct Blend: Codable {
        var sourceRGB: UInt
        var destinationRGB: UInt
        var rgbOperation: UInt
        var sourceAlpha: UInt
        var destinationAlpha: UInt
        var alphaOperation: UInt
        var constantColor: SIMD4<Float>

        init(_ function: MeshRenderState.BlendFunction) {
            sourceRGB = function.sourceRGB.rawValue
            destinationRGB = function.destinationRGB.rawValue
            rgbOperation = function.rgbOperation.rawValue
            sourceAlpha = function.sourceAlpha.rawValue
            destinationAlpha = function.destinationAlpha.rawValue
            alphaOperation = function.alphaOperation.rawValue
            constantColor = function.constantColor
        }

        func makeFunction() throws -> MeshRenderState.BlendFunction {
            guard let sourceRGB = MTLBlendFactor(rawValue: sourceRGB),
                  let destinationRGB = MTLBlendFactor(rawValue: destinationRGB),
                  let rgbOperation = MTLBlendOperation(rawValue: rgbOperation),
                  let sourceAlpha = MTLBlendFactor(rawValue: sourceAlpha),
                  let destinationAlpha = MTLBlendFactor(rawValue: destinationAlpha),
                  let alphaOperation = MTLBlendOperation(rawValue: alphaOperation) else {
                throw SceneArchive.Failure.invalidReference("mesh blend equation")
            }
            return .init(
                sourceRGB: sourceRGB, destinationRGB: destinationRGB, rgbOperation: rgbOperation,
                sourceAlpha: sourceAlpha, destinationAlpha: destinationAlpha, alphaOperation: alphaOperation,
                constantColor: constantColor
            )
        }
    }

    struct Stencil: Codable {
        var reference: UInt8
        var compare: UInt
        var readMask: UInt8
        var writeMask: UInt8
        var stencilFailure: UInt
        var depthFailure: UInt
        var pass: UInt

        init(_ state: MeshRenderState.Stencil) {
            reference = state.reference
            compare = state.compare.rawValue
            readMask = state.readMask
            writeMask = state.writeMask
            stencilFailure = state.stencilFailure.rawValue
            depthFailure = state.depthFailure.rawValue
            pass = state.pass.rawValue
        }

        func makeState() throws -> MeshRenderState.Stencil {
            guard let compare = MTLCompareFunction(rawValue: compare),
                  let stencilFailure = MTLStencilOperation(rawValue: stencilFailure),
                  let depthFailure = MTLStencilOperation(rawValue: depthFailure),
                  let pass = MTLStencilOperation(rawValue: pass) else {
                throw SceneArchive.Failure.invalidReference("mesh stencil comparison or operation")
            }
            let state = MeshRenderState.Stencil(
                reference: reference, compare: compare, readMask: readMask, writeMask: writeMask,
                stencilFailure: stencilFailure, depthFailure: depthFailure, pass: pass
            )
            guard state.isValid else { throw SceneArchive.Failure.invalidReference("mesh stencil comparison or operation") }
            return state
        }
    }

    struct Offset: Codable {
        var factor: Float
        var units: Float
        var clamp: Float
    }

    var blending: Blending
    var customBlend: Blend?
    var transparent: Bool
    var depthTest: Bool
    var depthWrite: Bool
    var depthCompare: UInt
    var colorWriteMask: UInt
    var depthOffset: Offset?
    var side: Side
    /// Nil means both sides cast. Stored from archive version 37.
    var shadowSide: Side? = nil
    var alphaHash: Bool
    var alphaToCoverage: Bool
    var stencil: Stencil?

    init(_ state: MeshRenderState) {
        switch state.blending {
        case .none: blending = .none
        case .normal: blending = .normal
        case .additive: blending = .additive
        case .subtractive: blending = .subtractive
        case .multiply: blending = .multiply
        case .custom(let function):
            blending = .custom
            customBlend = Blend(function)
        }
        transparent = state.transparent
        depthTest = state.depthTest
        depthWrite = state.depthWrite
        depthCompare = state.depthCompare.rawValue
        colorWriteMask = state.colorWriteMask.rawValue
        depthOffset = state.depthOffset.map { Offset(factor: $0.factor, units: $0.units, clamp: $0.clamp) }
        side = switch state.side {
        case .front: .front
        case .back: .back
        case .double: .double
        }
        shadowSide = state.shadowSide.map { side in
            switch side {
            case .front: .front
            case .back: .back
            case .double: .double
            }
        }
        alphaHash = state.alphaHash
        alphaToCoverage = state.alphaToCoverage
        stencil = state.stencil.map(Stencil.init)
    }

    func makeState() throws -> MeshRenderState {
        let blend: MeshRenderState.Blending
        switch blending {
        case .none: blend = .none
        case .normal: blend = .normal
        case .additive: blend = .additive
        case .subtractive: blend = .subtractive
        case .multiply: blend = .multiply
        case .custom:
            guard let customBlend else { throw SceneArchive.Failure.invalidReference("mesh custom blend equation") }
            blend = .custom(try customBlend.makeFunction())
        }
        guard let compare = MTLCompareFunction(rawValue: depthCompare),
              colorWriteMask & ~MTLColorWriteMask.all.rawValue == 0 else {
            throw SceneArchive.Failure.invalidReference("mesh depth comparison or color write mask")
        }
        let faces: MeshRenderState.Side = switch side {
        case .front: .front
        case .back: .back
        case .double: .double
        }
        return MeshRenderState(
            blending: blend, transparent: transparent, depthTest: depthTest, depthWrite: depthWrite,
            depthCompare: compare, colorWriteMask: .init(rawValue: colorWriteMask),
            depthOffset: depthOffset.map { .init(factor: $0.factor, units: $0.units, clamp: $0.clamp) },
            side: faces, shadowSide: shadowSide.map { side in
                switch side {
                case .front: .front
                case .back: .back
                case .double: .double
                }
            }, alphaHash: alphaHash, alphaToCoverage: alphaToCoverage, stencil: try stencil?.makeState()
        )
    }
}
