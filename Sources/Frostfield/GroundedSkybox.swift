import Foundation
import Metal
import simd

/// A textured inward-facing sky with a curved transition to a projected floor.
public enum GroundedSkybox {
    /// Builds the projected sphere in capture-local coordinates. UVs stay
    /// attached to the original sphere directions, with +X at panorama center.
    /// The floor is at local Y = `-height` when the radius reaches the flat branch.
    public static func geometry(height: Float, radius: Float, resolution: Int = 128) throws -> MeshGeometry {
        guard height.isNormal, height > 0, radius.isNormal, radius > 0 else { throw Failure.dimensions }
        guard (1...128).contains(resolution) else { throw Failure.resolution }
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: max(3, 2 * resolution), rings: max(2, resolution))
        let threshold = -1.5 * Double(height)
        let vertices = sphere.vertices.map { vertex -> MeshVertex in
            // Match r187's mirrored sphere orientation in the native top-left UV convention.
            let direction = SIMD3<Double>(-Double(vertex.normal.x), Double(vertex.normal.y), -Double(vertex.normal.z))
            var position = Double(radius) * direction
            var inward = -direction
            if position.y <= threshold {
                position *= -Double(height) / position.y
                position.y = -Double(height)
                inward = SIMD3(0, 1, 0)
            } else if position.y < 0 {
                let t = position.y / threshold
                // Radial scale and its derivative give the smooth inward normal.
                let slope = -2 * t / (3 * threshold)
                inward = simd_normalize(-direction * (1 - t * t) + SIMD3(0, Double(radius) * slope, 0))
                position *= 1 - t * t / 3
            }
            return MeshVertex(position: SIMD3<Float>(position), normal: SIMD3<Float>(inward), textureCoordinate: vertex.textureCoordinate)
        }
        var inward = sphere.indices
        for start in stride(from: 0, to: inward.count, by: 3) {
            inward.swapAt(start + 1, start + 2)
        }
        return try MeshGeometry(vertices: vertices, indices: inward)
    }

    /// Prepares a scene node using an opaque, top-left 2:1 panorama with +X at
    /// its center. Move or rotate the returned node to place the capture origin
    /// and orient the panorama. The default placement puts the flat floor at Y = 0.
    /// It depth-tests after ordinary opaque meshes, leaves depth untouched, and
    /// does not cast shadows. Keep the camera inside the mesh and within its clip range.
    public static func makeNode(
        panorama: any MTLTexture, height: Float, radius: Float, resolution: Int = 128,
        colorPixelFormat: MTLPixelFormat = .rgba16Float, depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws -> SceneNode {
        try EnvironmentBackground.validate(panorama, device: panorama.device, type: .type2D)
        guard panorama.width == 2 * panorama.height else { throw Failure.panorama }
        let geometry = try geometry(height: height, radius: radius, resolution: resolution)
        let sampling = PhysicallyBasedMaterial.TextureSampling(addressU: .repeat, addressV: .clampToEdge)
        let sampler = try sampling.makeSamplerState(device: panorama.device)
        let material = try ClassicMaterial(model: .basic, map: .init(texture: panorama, sampler: sampler, sampling: sampling))
        let mesh = try MeshRenderer(
            device: panorama.device, colorPixelFormat: colorPixelFormat, geometry: geometry, material: .classic(material),
            depthMode: depthMode, renderState: .init(depthWrite: false), sampleCount: sampleCount
        )
        return SceneNode(
            transform: .init(translation: SIMD3(0, height, 0)), mesh: mesh, castsShadow: false,
            name: "Grounded skybox", renderOrder: Int.max
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case dimensions, resolution, panorama

        public var errorDescription: String? {
            switch self {
            case .dimensions: "Use positive normal ground height and skybox radius."
            case .resolution: "Use a skybox resolution in 1...128."
            case .panorama: "Use a 2:1 equirectangular panorama."
            }
        }
    }
}
