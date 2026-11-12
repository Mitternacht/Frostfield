import Foundation
import simd

// Camera, model and screen inputs for surface graphs: what each renderer supplies
// and which inputs a renderer rejects.

extension SurfaceShaderGraph {
    /// The per-pixel channel outputs, which raster PBR and the path tracer evaluate.
    /// Colored shadow outputs run only in raster shadow passes and are not included.
    var surfaceOutputIDs: [MaterialGraph.NodeID] {
        outputs.surfaceChannels.map(\.1)
    }

    /// The first node the per-pixel outputs reach that only a rasterizer can evaluate:
    /// a derivative, screen input, or stored viewport sample.
    var screenSpaceNode: MaterialGraph.NodeID? { program.firstScreenSpaceNode(from: surfaceOutputIDs) }
    /// Screen coordinates are supplied for the path's camera pixel. Derivatives and viewport textures are not.
    var pathRejectedScreenNode: MaterialGraph.NodeID? { program.firstPathRejectedScreenNode(from: surfaceOutputIDs) }

    var viewportInputs: SurfaceViewportInputs {
        program.order(for: surfaceOutputIDs).reduce(into: []) { inputs, id in
            switch nodes[id] {
            case .sampleScene, .sampleSceneLevel: inputs.insert(.color)
            case .sampleDepth, .viewportSafeUV: inputs.insert(.depth)
            default: break
            }
        }
    }

    /// Shadow passes provide actual light-space screen coordinates and derivatives.
    /// They do not yet expose the light view's separate camera parameters.
    /// Covers the coverage outputs of every raster shadow pass, where `castShadowMask` replaces `mask`.
    var shadowCoverageCameraNode: MaterialGraph.NodeID? {
        Self.shadowUnsupportedNode(in: program, nodes: nodes, from: outputs.shadowCoverageChannels.map(\.1))
    }

    /// The first node under `roots` that a light-space shadow pass cannot evaluate. A colored
    /// shadow output additionally rejects the receiver's directional visibility.
    static func shadowUnsupportedNode(
        in program: ShaderProgram, nodes: [MaterialGraph.Node], from roots: [MaterialGraph.NodeID],
        rejectsShadowVisibility: Bool = false
    ) -> MaterialGraph.NodeID? {
        program.order(for: roots).first { id in
            switch nodes[id] {
            case .sampleScene, .sampleSceneLevel, .sampleDepth, .viewportSafeUV, .softParticleOpacity: return true
            default: break
            }
            guard case .input(let input) = nodes[id] else { return false }
            return switch input {
            case .cameraPosition, .cameraNear, .cameraFar, .viewDirection, .viewMatrix, .projectionMatrix: true
            case .directionalShadowVisibility: rejectsShadowVisibility
            default: false
            }
        }
    }

    var hasViewportMipLevelNode: Bool {
        nodes.contains { if case .sampleSceneLevel = $0 { true } else { false } }
    }

    var hasViewportSafeUVNode: Bool {
        nodes.contains { if case .viewportSafeUV = $0 { true } else { false } }
    }

    var hasSoftParticleNode: Bool {
        nodes.contains { if case .softParticleOpacity = $0 { true } else { false } }
    }

    var readsDirectionalShadow: Bool {
        program.uses({ $0 == .input(.directionalShadowVisibility) }, from: surfaceOutputIDs)
    }

    var readsSoftParticles: Bool {
        program.uses({ if case .softParticleOpacity = $0 { true } else { false } }, from: surfaceOutputIDs)
    }

    /// Whether the per-pixel outputs read the view or projection matrix or the clip distances.
    var readsCameraProjection: Bool { program.readsCameraProjection(from: surfaceOutputIDs) }

    var readsProjectionMatrix: Bool {
        program.uses({ node in
            if case .input(.projectionMatrix) = node { return true }
            return false
        }, from: surfaceOutputIDs)
    }

    var surfaceUniformMask: UInt32 { program.uniformMask(from: surfaceOutputIDs) }

    /// Whether the per-pixel outputs read the graph clock.
    var surfaceReadsClock: Bool { program.readsClock(from: surfaceOutputIDs) }

    /// Whether the vertex offset reads the graph clock.
    var vertexReadsClock: Bool { outputs.positionOffset.map { program.readsClock(from: [$0]) } ?? false }

    /// Whether the per-pixel outputs read `modelMatrix` or `normalMatrix`.
    var readsModelMatrices: Bool {
        program.uses({ node in
            guard case .input(let input) = node else { return false }
            return input == .modelMatrix || input == .normalMatrix
        }, from: surfaceOutputIDs)
    }
}

/// Fragment buffer 24 of a generated raster surface pipeline. Keep the layout in
/// lockstep with `FrostfieldGraphView` in `MeshRenderer.rasterGraphResources`.
struct PackedGraphView {
    var viewMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
    var modelMatrix: simd_float4x4
    /// The inverse transpose of the model matrix; graphs read the upper 3 × 3.
    var normalMatrix: simd_float4x4
    /// World-space camera position in xyz and the near distance in w.
    var camera: SIMD4<Float>
    /// Viewport origin in xy and size in zw, in pixels.
    var viewport: SIMD4<Float>
    /// The far distance in x and the render target size in pixels in yz.
    var clip: SIMD4<Float>
    /// Window depth range in xy and forward/reversed/logarithmic mode in z.
    var depthRange: SIMD4<Float>

    static let metalSource = """
    // Fragment buffer 24 of generated surface pipelines; see PackedGraphView.
    struct FrostfieldGraphView {
        float4x4 viewMatrix;
        float4x4 projectionMatrix;
        float4x4 modelMatrix;
        float4x4 normalMatrix;
        float4 camera;   // xyz position, w near
        float4 viewport; // xy origin, zw size in pixels
        float4 clip;     // x far, yz render target size in pixels
        float4 depthRange;
    };
    """
}

/// The camera and viewport values a raster pass supplies to every generated surface draw.
struct GraphViewContext {
    var viewMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
    var cameraPosition: SIMD3<Float>
    var near: Float
    var far: Float
    var viewportOrigin: SIMD2<Float>
    var viewportSize: SIMD2<Float>
    /// The color attachment's size in pixels.
    var targetSize: SIMD2<Float>
    /// False when the caller supplied only a combined clip matrix, so the view and
    /// projection matrices and clip distances are unknown.
    var hasProjection: Bool
    var windowDepth: SIMD2<Float>
    var depthMode: MeshDepthMode

    /// A pass that renders through a `PerspectiveCamera` at a transform.
    init(
        camera: PerspectiveCamera, cameraTransform: SceneTransform, width: Int, height: Int, depthMode: MeshDepthMode
    ) throws {
        viewMatrix = simd_inverse(cameraTransform.matrix)
        projectionMatrix = try camera.projection(aspectRatio: Float(width) / Float(height), depthMode: depthMode)
        cameraPosition = cameraTransform.translation
        near = camera.near
        far = camera.far
        viewportOrigin = .zero
        viewportSize = SIMD2(Float(width), Float(height))
        targetSize = viewportSize
        hasProjection = true
        windowDepth = SIMD2(0, 1)
        self.depthMode = depthMode
    }

    /// A scene pass. Uses ``SceneGraph/orthographicCamera`` when the graph has one.
    init(graph: SceneGraph, width: Int, height: Int, depthMode: MeshDepthMode) throws {
        let clips = graph.activeClipDistances
        viewMatrix = simd_inverse(graph.cameraTransform.matrix)
        projectionMatrix = try graph.activeProjection(aspectRatio: Float(width) / Float(height), depthMode: depthMode)
        cameraPosition = graph.cameraTransform.translation
        near = clips.near
        far = clips.far
        viewportOrigin = .zero
        viewportSize = SIMD2(Float(width), Float(height))
        targetSize = viewportSize
        hasProjection = true
        windowDepth = SIMD2(0, 1)
        self.depthMode = depthMode
    }

    /// A pass with a caller-supplied clip matrix and camera position, and the
    /// separate view and projection when the caller supplies them.
    init(
        cameraPosition: SIMD3<Float>, viewport: MTLViewportBox, targetSize: SIMD2<Float>, camera: ShaderGraphCamera?,
        windowDepth: SIMD2<Float>, depthMode: MeshDepthMode
    ) {
        viewMatrix = camera?.viewMatrix ?? simd_float4x4()
        projectionMatrix = camera?.projectionMatrix ?? simd_float4x4()
        self.cameraPosition = cameraPosition
        near = camera?.near ?? 0
        far = camera?.far ?? 0
        viewportOrigin = viewport.origin
        viewportSize = viewport.size
        self.targetSize = targetSize
        hasProjection = camera != nil
        self.windowDepth = windowDepth
        self.depthMode = depthMode
    }

    func packed(model: simd_float4x4, normal: simd_float4x4) -> PackedGraphView {
        let mode: Float = switch depthMode {
        case .forward: 0
        case .reversed: 1
        case .logarithmic: 2
        }
        return PackedGraphView(
            viewMatrix: viewMatrix, projectionMatrix: projectionMatrix, modelMatrix: model, normalMatrix: normal,
            camera: SIMD4(cameraPosition, near), viewport: SIMD4(lowHalf: viewportOrigin, highHalf: viewportSize),
            clip: SIMD4(far, targetSize.x, targetSize.y, 0),
            depthRange: SIMD4(windowDepth.x, windowDepth.y, mode, 0)
        )
    }

    func softParticleProjection() throws -> SIMD4<Float> {
        guard windowDepth.y > windowDepth.x else { throw SoftParticleDepth.Failure.projection }
        return try SoftParticleDepth.projectionCoefficients(
            projectionMatrix, depthMode: depthMode == .reversed ? .reversed : .forward
        )
    }

    func shadowViewport(resolution: Int) -> GraphViewContext {
        var context = self
        let size = SIMD2(repeating: Float(resolution))
        context.viewportOrigin = .zero
        context.viewportSize = size
        context.targetSize = size
        return context
    }
}

/// The camera that surface graphs' `viewMatrix`, `projectionMatrix`, `cameraNear`
/// and `cameraFar` inputs read when a raster pass supplies its own clip matrix:
/// a glTF camera view (`GLTFScene.Camera.View.graphCamera`), a reversed-depth view,
/// or one eye of a stereo pair. Pass it to `SceneGraph.encode(...clipFromWorld:...)`.
public struct ShaderGraphCamera: Equatable, Sendable {
    /// World to camera space, the inverse of the camera's pose.
    public let viewMatrix: simd_float4x4
    /// Camera to Metal clip space, with the depth mapping of the render pass.
    public let projectionMatrix: simd_float4x4
    /// Distance to the near clip plane. Orthographic cameras may use zero.
    public let near: Float
    /// Distance to the far clip plane, or infinity for an infinite perspective projection.
    public let far: Float

    /// Throws `Failure.invalid` unless both matrices are finite and invertible, `near`
    /// is finite and nonnegative, and `far` is greater than `near`.
    public init(viewMatrix: simd_float4x4, projectionMatrix: simd_float4x4, near: Float, far: Float) throws {
        func usable(_ matrix: simd_float4x4) -> Bool {
            let columns = [matrix.columns.0, matrix.columns.1, matrix.columns.2, matrix.columns.3]
            let determinant = simd_determinant(matrix)
            return columns.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }
                && determinant.isFinite && determinant != 0
        }
        guard usable(viewMatrix), usable(projectionMatrix), near.isFinite, near >= 0, !far.isNaN, far > near else {
            throw Failure.invalid
        }
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.near = near
        self.far = far
    }

    /// Whether `projectionMatrix × viewMatrix` equals `clipFromWorld` to within float
    /// rounding. A high-precision pass compares columns independently so a large
    /// world translation cannot mask a different authored projection or view axis.
    func matches(clipFromWorld: simd_float4x4, perColumn: Bool = false) -> Bool {
        let product = projectionMatrix * viewMatrix
        let pairs = [(product.columns.0, clipFromWorld.columns.0), (product.columns.1, clipFromWorld.columns.1),
                     (product.columns.2, clipFromWorld.columns.2), (product.columns.3, clipFromWorld.columns.3)]
        if perColumn {
            return pairs.allSatisfy { expected, actual in
                guard (0..<4).allSatisfy({ expected[$0].isFinite && actual[$0].isFinite }) else { return false }
                return simd_reduce_max(simd_abs(expected - actual)) <= 1e-4 * max(simd_reduce_max(simd_abs(actual)), 1)
            }
        }
        let scale = pairs.map { simd_reduce_max(simd_abs($0.1)) }.max() ?? 0
        return pairs.allSatisfy { simd_reduce_max(simd_abs($0.0 - $0.1)) <= 1e-4 * max(scale, 1) }
    }

    /// Builds the affine view after subtracting the authored camera origin in
    /// world space. The absolute translation was rounded to Float and must not
    /// be used for the cancellation at large coordinates.
    func relativeView(at origin: SIMD3<Double>) -> simd_double4x4? {
        let (x, y, z, w) = viewMatrix.columns
        guard x.w == 0, y.w == 0, z.w == 0, abs(w.w - 1) <= 8 * Float.ulpOfOne else { return nil }
        func xyz(_ column: SIMD4<Float>) -> SIMD3<Double> {
            SIMD3(Double(column.x), Double(column.y), Double(column.z))
        }
        let first = xyz(x) * origin.x
        let second = xyz(y) * origin.y
        let third = xyz(z) * origin.z
        let translation = xyz(w)
        // Bound both translation rounding and the input Float view-axis errors
        // independently per component, including sums that cancel.
        let magnitude = simd_abs(first) + simd_abs(second) + simd_abs(third) + simd_abs(translation)
        let tolerance = simd_max(magnitude, SIMD3(repeating: 1)) * (8 * Double(Float.ulpOfOne))
        guard all(simd_abs(first + second + third + translation) .<= tolerance) else { return nil }
        let inverseW = 1 / Double(w.w)
        return simd_double4x4(
            SIMD4(xyz(x) * inverseW, 0), SIMD4(xyz(y) * inverseW, 0), SIMD4(xyz(z) * inverseW, 0),
            SIMD4(0, 0, 0, 1)
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case invalid

        public var errorDescription: String? {
            "A shader graph camera needs finite, invertible view and projection matrices, a finite nonnegative near "
                + "distance and a greater far distance."
        }
    }
}

/// A viewport's origin and size in pixels.
struct MTLViewportBox {
    var origin: SIMD2<Float>
    var size: SIMD2<Float>
}

extension PathTracingScene {
    /// Leading float4 slots of every dispatch's graph uniform buffer: the camera
    /// position and near distance, the far distance, then the view and projection
    /// matrices by column.
    static let graphViewSlotCount = 10

    /// The camera values path-traced surface graphs read. They describe the tracing
    /// camera at every bounce, as raster graphs see the rendering camera.
    static func graphViewBlock(
        model: PathTracer.CameraModel, cameraTransform: SceneTransform, width: Int, height: Int,
        readsProjectionMatrix: Bool
    ) throws -> [SIMD4<Float>] {
        let view = pathViewMatrix(cameraTransform)
        var projection = simd_float4x4()
        if readsProjectionMatrix {
            projection = try pathProjection(model: model, width: width, height: height)
            if case .orthographic = model.projection {
                _ = try checkedPathCamera(model: model, view: view, projection: projection)
            }
        }
        return [
            SIMD4(cameraTransform.translation, model.near), SIMD4(model.far, 0, 0, 0),
            view.columns.0, view.columns.1, view.columns.2, view.columns.3,
            projection.columns.0, projection.columns.1, projection.columns.2, projection.columns.3,
        ]
    }

    /// Unjittered forward-depth matrices matching the path tracer's planar rays.
    static func planarPathCamera(
        model: PathTracer.CameraModel, cameraTransform: SceneTransform, width: Int, height: Int
    ) throws -> ShaderGraphCamera {
        try checkedPathCamera(
            model: model, view: pathViewMatrix(cameraTransform),
            projection: pathProjection(model: model, width: width, height: height)
        )
    }

    private static func checkedPathCamera(
        model: PathTracer.CameraModel, view: simd_float4x4, projection: simd_float4x4
    ) throws -> ShaderGraphCamera {
        guard let camera = try? ShaderGraphCamera(
            viewMatrix: view, projectionMatrix: projection, near: model.near, far: model.far
        ) else { throw PathTracer.Failure.camera }
        return camera
    }

    private static func pathProjection(
        model: PathTracer.CameraModel, width: Int, height: Int
    ) throws -> simd_float4x4 {
        switch model.projection {
        case .perspective(let fieldOfView, let shift):
            let camera = try PerspectiveCamera(verticalFieldOfView: fieldOfView, near: model.near, far: model.far)
            var projection = try camera.projection(aspectRatio: model.aspectRatio(width: width, height: height))
            projection.columns.2.x = shift.x
            projection.columns.2.y = shift.y
            return projection
        case .orthographic:
            let window = model.window(width: width, height: height)
            let depthScale = 1 / (Double(model.near) - Double(model.far))
            return simd_float4x4(columns: (
                SIMD4(1 / window.x, 0, 0, 0), SIMD4(0, 1 / window.y, 0, 0),
                SIMD4(0, 0, Float(depthScale), 0), SIMD4(0, 0, Float(Double(model.near) * depthScale), 1)
            ))
        case .equirectangular, .stereoEquirectangular:
            throw PathTracer.Failure.graphCameraProjection
        }
    }

    private static func pathViewMatrix(_ cameraTransform: SceneTransform) -> simd_float4x4 {
        var rigid = cameraTransform
        rigid.rotation = simd_normalize(rigid.rotation)
        return simd_inverse(rigid.matrix)
    }
}

extension PathTracingScene {
    /// Rejects surface graph inputs that the path tracer cannot evaluate: screen-space
    /// values have no meaning at secondary bounces, and an instanced node's traced
    /// instances carry per-instance matrices where raster graphs see the node's matrix.
    static func validatePathInputs(_ graph: SurfaceShaderGraph, node: ResolvedSceneNode) throws {
        if let screenSpace = graph.pathRejectedScreenNode {
            throw PathTracingScene.Failure.unsupported(
                node.id, "surface graph node \(screenSpace), which reads a screen-space derivative, "
                    + "stored viewport texture, soft-particle fade, or directional shadow visibility"
            )
        }
        if node.instances != nil, graph.readsModelMatrices {
            throw PathTracingScene.Failure.unsupported(
                node.id, "surface graphs that read modelMatrix or normalMatrix on an instanced node"
            )
        }
    }
}
