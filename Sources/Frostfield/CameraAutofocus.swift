import Foundation
import Metal

extension CameraLens {
    /// Reads one stored depth texel and returns a lens focused at its view-axis distance.
    /// `screenUV` uses top-left normalized coordinates. Cleared depth returns nil.
    /// Commit the matching perspective render on `commandQueue` before reading, or
    /// synchronize a producer on another queue. The lens is returned after Metal
    /// completes the readback. Use it for a later `BokehDepthOfField` frame.
    public nonisolated(nonsending) func autofocused(
        from depth: any MTLTexture, at screenUV: SIMD2<Float> = SIMD2(repeating: 0.5),
        camera: PerspectiveCamera, depthMode: MeshDepthMode = .forward, commandQueue: any MTLCommandQueue
    ) async throws -> CameraLens? {
        guard (0...1).contains(screenUV.x), (0...1).contains(screenUV.y) else {
            throw AutofocusFailure.coordinates
        }
        try depthMode.requireProjectionDepth()
        guard depth.textureType == .type2D, depth.pixelFormat == .depth32Float, depth.sampleCount == 1 else {
            throw AutofocusFailure.depthTexture
        }
        let x = min(Int(Double(screenUV.x) * Double(depth.width)), depth.width - 1)
        let y = min(Int(Double(screenUV.y) * Double(depth.height)), depth.height - 1)
        let readback = try await TextureTransfer.readOnCaller(
            from: depth, region: MTLRegionMake2D(x, y, 1, 1), commandQueue: commandQueue
        )
        let stored = readback.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
        guard (0...1).contains(stored) else { throw AutofocusFailure.storedDepth }
        let reversed = depthMode == .reversed
        guard stored != (reversed ? 0 : 1) else { return nil }

        let near = Double(camera.near), far = Double(camera.far)
        let reversedDepth = reversed ? Double(stored) : 1 - Double(stored)
        var focused = self
        focused.focusDistance = Float(near * far / (near + reversedDepth * (far - near)))
        try focused.validate()
        return focused
    }

    public enum AutofocusFailure: LocalizedError, Equatable {
        case coordinates
        case depthTexture
        case storedDepth

        public var errorDescription: String? {
            switch self {
            case .coordinates: "Autofocus coordinates must be finite and inside the normalized image."
            case .depthTexture: "Autofocus requires a single-sample 2D Depth32Float texture."
            case .storedDepth: "Autofocus needs finite Metal projection depth from zero to one."
            }
        }
    }
}
