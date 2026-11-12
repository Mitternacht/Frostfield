import simd

struct ArchiveClassicMaterial: Codable {
    enum Model: Codable {
        case basic
        case lambert
        case phong(specular: SIMD3<Float>, shininess: Float)
        case toon(gradient: ArchiveBinding?)
        case matcap(ArchiveBinding)
        case normal
        case depth
        case distance(reference: SIMD3<Float>, near: Float, far: Float)
        case shadow
    }

    struct Environment: Codable {
        enum Blend: String, Codable {
            case multiply, mix, add

            init(_ blend: ClassicMaterial.EnvironmentBlend) {
                self = switch blend {
                case .multiply: .multiply
                case .mix: .mix
                case .add: .add
                }
            }

            var runtime: ClassicMaterial.EnvironmentBlend {
                switch self {
                case .multiply: .multiply
                case .mix: .mix
                case .add: .add
                }
            }
        }

        var texture: Int
        var blend: Blend
        var reflectivity: Float
        var intensity: Float
        var yaw: Float
        var refractionRatio: Float?
    }

    var model: Model
    var color: SIMD3<Float>
    var opacity: Float
    var map: ArchiveBinding?
    var alphaMap: ArchiveBinding?
    var alphaTest: Float
    var aoMap: ArchiveBinding?
    var aoMapIntensity: Float
    var lightMap: ArchiveBinding?
    var lightMapIntensity: Float
    var emissive: SIMD3<Float>
    var emissiveMap: ArchiveBinding?
    var environmentMap: Environment?
    var ambient: SIMD3<Float>
    var receivesShadows: Bool
    var flatShading: Bool
    /// Nil means filled triangles. True is stored from archive version 35.
    var wireframe: Bool? = nil
    /// Nil means the material is drawn. False is stored from archive version 36.
    var isVisible: Bool? = nil
    /// Nil means the material accepts fog. False is stored from archive version 36.
    var fog: Bool? = nil
    /// Nil means no material dither. True is stored from archive version 38.
    var dithering: Bool? = nil
    var depthPacking: ClassicMaterial.DepthPacking?
    var distanceOutput: ClassicMaterial.DistanceOutput?
    var bumpMap: ArchivePBR.BumpMap?
    var normalMap: ArchiveBinding?
    var normalScale: SIMD2<Float>?
    var surfaceGraph: Int?
    var surfaceUniforms: Int?
    var surfaceTextures: [String: Int]?

    var hasExplicitDepthOutput: Bool { depthPacking != nil || distanceOutput == .red }
    var hasSurfaceDetail: Bool { bumpMap != nil || normalMap != nil || normalScale != nil }
}
