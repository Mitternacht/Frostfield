import Foundation
import Metal
import simd

extension PathTracer {
    /// First-hit output textures, written by every encode that requests them.
    ///
    /// Each texel describes one centered pinhole sample, even when the beauty
    /// image uses lens and subpixel samples. Stochastic surfaces are included
    /// where their alpha is at least one half. When camera motion is requested,
    /// the first hit and the primary beauty ray use its frame-wide jitter.
    /// Each texture must be distinct,
    /// tracked, shader-writable, and the size of the accumulation.
    // The caller orders GPU access to the textures on the queue.
    public struct AOVs: @unchecked Sendable {
        /// Diffuse albedo in RGB, with one in alpha for hits and zero for misses.
        /// Unlit surfaces use base color. RGBA16Float or RGBA32Float.
        public var albedo: (any MTLTexture)?
        /// Signed world-space shading normal, with the hit flag in alpha.
        /// RGBA16Float or RGBA32Float.
        public var normal: (any MTLTexture)?
        /// Distance along the camera forward axis for planar cameras and along
        /// the ray for panoramas. Misses are positive infinity. R32Float.
        public var depth: (any MTLTexture)?
        /// First-hit perceptual roughness, or zero for a light or miss. R16Float or R32Float.
        public var roughness: (any MTLTexture)?
        /// Emitted linear radiance in RGB, with the hit flag in alpha.
        /// RGBA16Float or RGBA32Float.
        public var emission: (any MTLTexture)?
        /// `PathTracer.objectID(for:)` of the hit node. R32Uint.
        public var objectID: (any MTLTexture)?
        /// One plus the material slot of the hit surface. R32Uint.
        /// Both IDs are zero for a miss or a camera-visible area light.
        public var materialID: (any MTLTexture)?
        /// View-dependent Fresnel reflectance in RGB, including base and clearcoat
        /// reflection. Unlit and camera-visible area light hits have zero reflectance.
        /// Alpha is one for hits and zero for misses. RGBA16Float or RGBA32Float.
        public var specularAlbedo: (any MTLTexture)?
        /// Normalized forward Metal depth for perspective and orthographic cameras.
        /// Zero is the near plane; one is the far plane or a miss. R32Float.
        /// Panorama projections do not have a planar device depth.
        public var projectedDepth: (any MTLTexture)?
        /// Requests an independent motion frame for the camera and moving or posed meshes.
        public var cameraMotion: CameraMotion?

        public init(
            albedo: (any MTLTexture)? = nil, normal: (any MTLTexture)? = nil,
            depth: (any MTLTexture)? = nil, roughness: (any MTLTexture)? = nil,
            emission: (any MTLTexture)? = nil, objectID: (any MTLTexture)? = nil,
            materialID: (any MTLTexture)? = nil, specularAlbedo: (any MTLTexture)? = nil,
            projectedDepth: (any MTLTexture)? = nil, cameraMotion: CameraMotion? = nil
        ) {
            self.albedo = albedo
            self.normal = normal
            self.depth = depth
            self.roughness = roughness
            self.emission = emission
            self.objectID = objectID
            self.materialID = materialID
            self.specularAlbedo = specularAlbedo
            self.projectedDepth = projectedDepth
            self.cameraMotion = cameraMotion
        }

        // Slot and mask order match frostfield_path_trace in PathTracingShading.h.
        static let formats: [[MTLPixelFormat]] = [
            [.rgba16Float, .rgba32Float], [.rgba16Float, .rgba32Float], [.r32Float],
            [.r16Float, .r32Float], [.rgba16Float, .rgba32Float], [.r32Uint], [.r32Uint],
            [.rgba16Float, .rgba32Float], [.r32Float], [.rg16Float, .rg32Float],
        ]

        var textures: [(any MTLTexture)?] {
            [albedo, normal, depth, roughness, emission, objectID, materialID, specularAlbedo, projectedDepth, cameraMotion?.vectors]
        }
    }

    /// Camera, instance, and pose motion for a full-image, planar pinhole render at
    /// one sample per pixel. Pass nil to the encoder's progressive `previous` parameter. Use
    /// `previousFrame` here for the previous completed motion frame.
    public struct CameraMotion: @unchecked Sendable {
        /// Unjittered previous-minus-current displacement in input pixels, X right
        /// and Y down. Pixels outside the preceding projection receive finite
        /// out-of-image displacement. RG16Float or RG32Float.
        public var vectors: any MTLTexture
        /// A successful preceding motion frame from this tracer, with the same
        /// dimensions, projection kind, stable vertex and instance identities,
        /// lighting, and surface uniforms. Instance transforms and native skin or
        /// morph poses can change. An explicit geometry or material replacement,
        /// incompatible geometry rebuild, or bulk instance replacement requires a fresh frame.
        /// Use nil on the first frame or after a camera cut. That frame writes
        /// zero vectors and can serve as the next frame's history.
        public var previousFrame: History?
        /// A shared subpixel sampling offset, in top-left pixel coordinates.
        /// Each component must lie in −0.5...0.5. This shifts sampling without
        /// adding jitter to the motion displacement.
        public var jitter: SIMD2<Float>

        public init(vectors: any MTLTexture, previousFrame: History? = nil, jitter: SIMD2<Float> = .zero) {
            self.vectors = vectors
            self.previousFrame = previousFrame
            self.jitter = jitter
        }
    }

    /// Stable 32-bit FNV-1a hash of the node UUID bytes, with zero mapped to one
    /// because it represents a miss. Multiple instances of a node share this
    /// value. Distinct nodes can have the same hash.
    public static func objectID(for nodeID: UUID) -> UInt32 {
        let hash = withUnsafeBytes(of: nodeID.uuid) { bytes in
            bytes.reduce(UInt32(2_166_136_261)) { ($0 ^ UInt32($1)) &* 16_777_619 }
        }
        return hash == 0 ? 1 : hash
    }
}
