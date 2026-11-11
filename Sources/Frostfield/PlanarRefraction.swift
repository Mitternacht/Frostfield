import Foundation
import Metal
import simd

/// Captures the opposite side of a world-space plane from the original camera.
public struct PlanarRefraction: Sendable {
    public typealias MultisampleTargets = PlanarReflection.MultisampleTargets

    public let plane: Plane3D
    /// This node and its descendants are excluded from the capture.
    public let refractor: UUID
    /// Distance in world units to move the clip plane behind the authored surface.
    public let clipBias: Float

    public init(plane: Plane3D, refractor: UUID, clipBias: Float = 0) throws {
        guard clipBias.isFinite, clipBias >= 0 else { throw Failure.clipBias }
        self.plane = plane
        self.refractor = refractor
        self.clipBias = clipBias
    }

    /// The unchanged source view with an oblique projection that clips against
    /// the negative half-space. Nil means the camera is behind the authored plane
    /// or the kept region misses the view.
    public func refractedView(_ camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward) throws -> ShaderGraphCamera? {
        try captureView(camera, depthMode: depthMode)?.camera
    }

    /// Encodes color and depth into distinct tracked caller-owned targets as in
    /// `PlanarReflection.encode`. The source projection's XY window stays intact.
    /// Draw the refractor after capture using `samplePlanarRefraction` in the same
    /// viewport space. Exclude any other nodes that sample the capture targets.
    /// An inactive capture clears the targets. Use a retained command buffer and
    /// discard it if encoding throws.
    @discardableResult
    public func encode(
        _ graph: SceneGraph, camera: ShaderGraphCamera, color: any MTLTexture, depth: any MTLTexture,
        depthMode: MeshDepthMode = .forward, excluding: Set<UUID> = [],
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        generateMipmaps: Bool = false, shadowMap: DirectionalShadowMap? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, transmissionBackdrop: (any MTLTexture)? = nil,
        background: EnvironmentBackground? = nil, backgroundOptions: EnvironmentBackground.Options = .init(),
        multisample: MultisampleTargets? = nil, commandBuffer: any MTLCommandBuffer
    ) throws -> SceneDrawStatistics? {
        try PlanarReflection.encodeCapture(
            graph, view: { try captureView(camera, depthMode: depthMode) }, label: "refraction",
            color: color, depth: depth, depthMode: depthMode, excluding: excluding.union([refractor]), clearColor: clearColor,
            generateMipmaps: generateMipmaps, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps,
            transmissionBackdrop: transmissionBackdrop, background: background, backgroundOptions: backgroundOptions,
            multisample: multisample, commandBuffer: commandBuffer
        )
    }

    private func captureView(_ camera: ShaderGraphCamera, depthMode: MeshDepthMode) throws -> PlanarReflection.View? {
        try PlanarReflection.captureView(camera, plane: plane, operation: .refraction(clipBias: clipBias), depthMode: depthMode)
    }

    public enum Failure: LocalizedError, Equatable {
        case clipBias

        public var errorDescription: String? { "Use a finite, nonnegative planar refraction clip bias." }
    }
}

public extension ShaderTexture {
    /// Samples a same-view `PlanarRefraction` capture. The optional distortion
    /// is an offset in top-left viewport UV, such as a remapped DuDv texture.
    func samplePlanarRefraction(
        at coordinates: ShaderValue<SIMD2<Float>>? = nil, distortion: ShaderValue<SIMD2<Float>>? = nil
    ) -> ShaderValue<SIMD4<Float>> {
        let uv = coordinates ?? builder.screenUV
        return sample(distortion.map { uv + $0 } ?? uv)
    }
}
