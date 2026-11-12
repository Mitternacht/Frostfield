import Metal
import simd

/// A shaped emissive mesh for an environment capture, like drei `<Lightformer>`.
///
/// Put the node in the scene passed to `EnvironmentLighting.capture`. The main view
/// simply does not include it. `color` is linear RGB in 0...1 and `intensity` scales it,
/// matching `MeshBasicMaterial.color.multiplyScalar(intensity)` with `toneMapped` false.
/// Flat forms are double-sided. A target turns the form toward that point the way
/// `Object3D.lookAt` does, unless `rotation` is set.
public enum Lightformer {
    public enum Form: Sendable, Equatable {
        case rectangle
        case circle
        case ring
        case box
    }

    public enum Failure: LocalizedError, Equatable {
        case color
        case intensity
        case scale
        case segments

        public var errorDescription: String? {
            switch self {
            case .color: "A lightformer color must be finite linear RGB from zero through one."
            case .intensity: "Lightformer intensity must be finite and nonnegative."
            case .scale: "Lightformer scale must be finite and positive on every axis."
            case .segments: "A circle or ring needs 3 to 256 segments."
            }
        }
    }

    /// A unit form centered on the origin. A rectangle, circle, or ring lies in XY and faces +Z.
    /// A ring's hole has radius 0.5. The box is the unit cube.
    public static func node(
        device: any MTLDevice, form: Form = .rectangle,
        color: SIMD3<Float> = SIMD3(repeating: 1), intensity: Float = 1,
        position: SIMD3<Float> = .zero, scale: SIMD3<Float> = SIMD3(repeating: 1),
        rotation: simd_quatf? = nil, target: SIMD3<Float>? = .zero, segments: Int = 64
    ) throws -> SceneNode {
        guard color.x.isFinite, color.y.isFinite, color.z.isFinite,
              (0...1).contains(color.x), (0...1).contains(color.y), (0...1).contains(color.z) else {
            throw Failure.color
        }
        guard intensity.isFinite, intensity >= 0 else { throw Failure.intensity }
        guard scale.x.isFinite, scale.y.isFinite, scale.z.isFinite, scale.x > 0, scale.y > 0, scale.z > 0 else {
            throw Failure.scale
        }
        let geometry: MeshGeometry
        let doubleSided: Bool
        switch form {
        case .rectangle:
            geometry = try rectangle()
            doubleSided = true
        case .circle:
            geometry = try annulus(inner: 0, outer: 1, segments: segments)
            doubleSided = true
        case .ring:
            geometry = try annulus(inner: 0.5, outer: 1, segments: segments)
            doubleSided = true
        case .box:
            geometry = MeshGeometry.cube()
            doubleSided = false
        }
        let material = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0, 0, 0, 1), emissiveFactor: color,
            alphaMode: .opaque, isDoubleSided: doubleSided, emissiveStrength: intensity,
            specular: .init(factor: 0), toneMapped: false
        )
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(material)
        )
        let orientation = try rotation ?? look(from: position, to: target)
        return SceneNode(
            transform: .init(translation: position, rotation: orientation, scale: scale),
            mesh: mesh, castsShadow: false, name: "Lightformer"
        )
    }

    private static func rectangle() throws -> MeshGeometry {
        let normal = SIMD3<Float>(0, 0, 1)
        let vertices = [
            MeshVertex(position: SIMD3(-0.5, -0.5, 0), normal: normal, textureCoordinate: SIMD2(0, 1)),
            MeshVertex(position: SIMD3(0.5, -0.5, 0), normal: normal, textureCoordinate: SIMD2(1, 1)),
            MeshVertex(position: SIMD3(0.5, 0.5, 0), normal: normal, textureCoordinate: SIMD2(1, 0)),
            MeshVertex(position: SIMD3(-0.5, 0.5, 0), normal: normal, textureCoordinate: SIMD2(0, 0)),
        ]
        return try MeshGeometry(vertices: vertices, indices: [0, 1, 2, 0, 2, 3])
    }

    private static func annulus(inner: Float, outer: Float, segments: Int) throws -> MeshGeometry {
        guard (3...256).contains(segments) else { throw Failure.segments }
        let normal = SIMD3<Float>(0, 0, 1)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        if inner == 0 {
            vertices.append(MeshVertex(position: .zero, normal: normal, textureCoordinate: SIMD2(0.5, 0.5)))
            for step in 0...segments {
                let angle = Float(step) / Float(segments) * 2 * .pi
                let point = SIMD2(cos(angle), sin(angle))
                vertices.append(MeshVertex(
                    position: SIMD3(point.x * outer, point.y * outer, 0), normal: normal,
                    textureCoordinate: point * 0.5 + 0.5
                ))
            }
            for step in 0..<segments {
                indices += [0, UInt32(step + 1), UInt32(step + 2)]
            }
        } else {
            for step in 0...segments {
                let angle = Float(step) / Float(segments) * 2 * .pi
                let direction = SIMD2(cos(angle), sin(angle))
                for radius in [inner, outer] {
                    vertices.append(MeshVertex(
                        position: SIMD3(direction.x * radius, direction.y * radius, 0), normal: normal,
                        textureCoordinate: direction * (radius * 0.5) + 0.5
                    ))
                }
            }
            for step in 0..<segments {
                let base = UInt32(step * 2)
                indices += [base, base + 1, base + 3, base, base + 3, base + 2]
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// `Object3D.lookAt`: local +Z points from the target back toward `position`.
    private static func look(from position: SIMD3<Float>, to target: SIMD3<Float>?) throws -> simd_quatf {
        guard let target else { return simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)) }
        let offset = position - target
        let distance = simd_length(offset)
        guard distance.isFinite, distance > 1e-5 else { return simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)) }
        let z = offset / distance
        let reference = abs(z.y) < 0.999 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
        let x = simd_normalize(simd_cross(reference, z))
        let y = simd_cross(z, x)
        return simd_quatf(simd_float3x3(columns: (x, y, z)))
    }
}
