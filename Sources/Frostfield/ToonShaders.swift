import Foundation
import Metal
import simd

/// The example toon fragment shaders from Three.js `ToonShader.js`, before
/// `colorspace_fragment`. The light vector is used as stored, and the normal is
/// normalized. Hatching and dots use the fragment coordinate the caller supplies.
public enum ToonShaders {
    public struct Lighting: Equatable, Sendable {
        public var direction: SIMD3<Float>
        public var color: SIMD3<Float>
        public var ambient: SIMD3<Float>

        public init(direction: SIMD3<Float>, color: SIMD3<Float>, ambient: SIMD3<Float>) {
            self.direction = direction
            self.color = color
            self.ambient = ambient
        }
    }

    /// `ToonShader1`. `refraction` is the view-space refraction vector from the vertex shader.
    public static func directional(
        normal: SIMD3<Float>, refraction: SIMD3<Float>, base: SIMD3<Float>, lighting: Lighting
    ) -> SIMD3<Float> {
        let unitNormal = unit(normal)
        let weighting = lighting.ambient + lighting.color * max(simd_dot(unitNormal, lighting.direction), 0)
        let weight = simd_length(weighting)
        var intensity = smoothstep(edge0: -0.5, edge1: 1, value: pow(max(weight, 0), Float(20)))
        intensity += weight * 0.2
        let camera = simd_dot(unitNormal, refraction)
        intensity += pow(max(1 - abs(camera), 0), Float(6))
        intensity = intensity * 0.2 + 0.3
        if intensity < 0.5 {
            return 2 * intensity * base
        }
        return 1 - 2 * (1 - intensity) * (1 - base)
    }

    /// `ToonShaderHatching`. Later hatch tests overwrite earlier ones.
    public static func hatching(
        normal: SIMD3<Float>, fragment: SIMD2<Float>, base: SIMD3<Float>,
        lines: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>), lighting: Lighting
    ) -> SIMD3<Float> {
        let weight = simd_length(lightWeighting(normal: normal, lighting: lighting))
        var color = base
        if weight < 1, glslMod(fragment.x + fragment.y, 10) == 0 { color = lines.0 }
        if weight < 0.75, glslMod(fragment.x - fragment.y, 10) == 0 { color = lines.1 }
        if weight < 0.5, glslMod(fragment.x + fragment.y - 5, 10) == 0 { color = lines.2 }
        if weight < 0.3465, glslMod(fragment.x - fragment.y - 5, 10) == 0 { color = lines.3 }
        return color
    }

    /// `ToonShaderDotted`.
    public static func dotted(
        normal: SIMD3<Float>, fragment: SIMD2<Float>, base: SIMD3<Float>, line: SIMD3<Float>, lighting: Lighting
    ) -> SIMD3<Float> {
        let weight = simd_length(lightWeighting(normal: normal, lighting: lighting))
        var color = base
        if weight < 1, glslMod(fragment.x, 4.001) + glslMod(fragment.y, 4) > 6 { color = line }
        if weight < 0.5, glslMod(fragment.x + 2, 4.001) + glslMod(fragment.y + 2, 4) > 6 { color = line }
        return color
    }

    /// Evaluates the three shaders on the GPU and waits for the result.
    public static func evaluateOnGPU(_ samples: [GPUSample], device: any MTLDevice) throws -> [SIMD3<Float>] {
        let count = samples.count
        guard !samples.isEmpty else { return [] }
        guard count <= 65_536 else { throw Failure.count }
        guard let queue = device.makeCommandQueue() else { throw Failure.encoder }
        let library: any MTLLibrary
        do { library = try device.makeLibrary(source: metalSource, options: nil) } catch { throw Failure.pipeline(error.localizedDescription) }
        guard let function = library.makeFunction(name: "frostfield_toon_examples") else {
            throw Failure.pipeline("The toon kernel is missing.")
        }
        let pipeline: any MTLComputePipelineState
        do { pipeline = try device.makeComputePipelineState(function: function) } catch { throw Failure.pipeline(error.localizedDescription) }
        let source = try buffer(samples, device: device)
        guard let destination = device.makeBuffer(
            length: max(count, 1) * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared
        ) else { throw Failure.encoder }
        guard let buffer = queue.makeCommandBuffer() else { throw Failure.encoder }
        guard let encoder = buffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        var sampleCount = UInt32(count)
        encoder.setBytes(&sampleCount, length: MemoryLayout<UInt32>.stride, index: 2)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 64)
        let groups = max((count + width - 1) / width, 1)
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw Failure.command }
        let colors = UnsafeBufferPointer(start: destination.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count)
        return colors.map { SIMD3($0.x, $0.y, $0.z) }
    }

    public struct GPUSample: Equatable, Sendable {
        public var kind: Float
        public var normal: SIMD3<Float>
        public var lightDirection: SIMD3<Float>
        public var lightColor: SIMD3<Float>
        public var ambient: SIMD3<Float>
        public var base: SIMD3<Float>
        public var refraction: SIMD3<Float>
        public var fragment: SIMD2<Float>
        public var line1: SIMD3<Float>
        public var line2: SIMD3<Float>
        public var line3: SIMD3<Float>
        public var line4: SIMD3<Float>

        public init(
            kind: Kind, normal: SIMD3<Float>, lightDirection: SIMD3<Float>, lightColor: SIMD3<Float>,
            ambient: SIMD3<Float>, base: SIMD3<Float>, refraction: SIMD3<Float> = .zero, fragment: SIMD2<Float> = .zero,
            line1: SIMD3<Float> = .zero, line2: SIMD3<Float> = .zero, line3: SIMD3<Float> = .zero, line4: SIMD3<Float> = .zero
        ) {
            self.kind = Float(kind.rawValue)
            self.normal = normal
            self.lightDirection = lightDirection
            self.lightColor = lightColor
            self.ambient = ambient
            self.base = base
            self.refraction = refraction
            self.fragment = fragment
            self.line1 = line1
            self.line2 = line2
            self.line3 = line3
            self.line4 = line4
        }

        var packed: Packed {
            Packed(
                kind: SIMD4(kind, 0, 0, 0), normal: SIMD4(normal, 0), lightDirection: SIMD4(lightDirection, 0),
                lightColor: SIMD4(lightColor, 0), ambient: SIMD4(ambient, 0), base: SIMD4(base, 0),
                refraction: SIMD4(refraction, 0), pixel: SIMD4(fragment.x, fragment.y, 0, 0),
                line1: SIMD4(line1, 0), line2: SIMD4(line2, 0), line3: SIMD4(line3, 0), line4: SIMD4(line4, 0)
            )
        }

        public enum Kind: Int, Sendable {
            case directional = 0
            case hatching = 1
            case dotted = 2
        }
    }

    struct Packed {
        var kind, normal, lightDirection, lightColor, ambient, base, refraction, pixel, line1, line2, line3, line4: SIMD4<Float>
    }

    public enum Failure: LocalizedError, Equatable {
        case count
        case pipeline(String)
        case device
        case encoder
        case command

        public var errorDescription: String? {
            switch self {
            case .count: "A toon shader evaluation accepts at most 65,536 samples."
            case .pipeline(let detail): "Metal could not compile the toon example shaders. \(detail)"
            case .device: "Evaluate the toon shaders on the command buffer's device."
            case .encoder: "Metal could not create the toon shader encoder."
            case .command: "The toon shader command buffer did not complete."
            }
        }
    }

    private static func lightWeighting(normal: SIMD3<Float>, lighting: Lighting) -> SIMD3<Float> {
        lighting.ambient + lighting.color * max(simd_dot(unit(normal), lighting.direction), 0)
    }

    private static func unit(_ value: SIMD3<Float>) -> SIMD3<Float> {
        let length = simd_length(value)
        return length > 0 ? value / length : value
    }

    private static func smoothstep(edge0: Float, edge1: Float, value: Float) -> Float {
        let t = min(max((value - edge0) / (edge1 - edge0), 0), 1)
        return t * t * (3 - 2 * t)
    }

    private static func glslMod(_ value: Float, _ divisor: Float) -> Float {
        value - divisor * floor(value / divisor)
    }

    private static func buffer(_ samples: [GPUSample], device: any MTLDevice) throws -> any MTLBuffer {
        let packed = samples.map(\.packed)
        let buffer = packed.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: max(raw.count, 1), options: .storageModeShared)
        }
        guard let buffer else { throw Failure.encoder }
        return buffer
    }

    private static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct ToonSample {
        float4 kind, normal, lightDirection, lightColor, ambient, base, refraction, pixel, line1, line2, line3, line4;
    };

    static float frostfield_mod(float value, float divisor) {
        return value - divisor * floor(value / divisor);
    }

    static float3 frostfield_unit(float3 value) {
        float magnitude = length(value);
        return magnitude > 0.0f ? value / magnitude : value;
    }

    kernel void frostfield_toon_examples(
        device const ToonSample* samples [[buffer(0)]],
        device float4* colors [[buffer(1)]],
        constant uint& count [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= count) { return; }
        ToonSample sample = samples[id];
        float3 normal = frostfield_unit(sample.normal.xyz);
        float3 weighting = sample.ambient.xyz + sample.lightColor.xyz * max(dot(normal, sample.lightDirection.xyz), 0.0f);
        float weight = length(weighting);
        float3 color = sample.base.xyz;
        if (sample.kind.x < 0.5f) {
            float intensity = smoothstep(-0.5f, 1.0f, pow(max(weight, 0.0f), 20.0f));
            intensity += weight * 0.2f;
            float camera = dot(normal, sample.refraction.xyz);
            intensity += pow(max(1.0f - abs(camera), 0.0f), 6.0f);
            intensity = intensity * 0.2f + 0.3f;
            color = intensity < 0.5f ? 2.0f * intensity * sample.base.xyz
                : 1.0f - 2.0f * (1.0f - intensity) * (1.0f - sample.base.xyz);
        } else if (sample.kind.x < 1.5f) {
            float2 pixel = sample.pixel.xy;
            if (weight < 1.0f && frostfield_mod(pixel.x + pixel.y, 10.0f) == 0.0f) { color = sample.line1.xyz; }
            if (weight < 0.75f && frostfield_mod(pixel.x - pixel.y, 10.0f) == 0.0f) { color = sample.line2.xyz; }
            if (weight < 0.5f && frostfield_mod(pixel.x + pixel.y - 5.0f, 10.0f) == 0.0f) { color = sample.line3.xyz; }
            if (weight < 0.3465f && frostfield_mod(pixel.x - pixel.y - 5.0f, 10.0f) == 0.0f) { color = sample.line4.xyz; }
        } else {
            float2 pixel = sample.pixel.xy;
            if (weight < 1.0f && frostfield_mod(pixel.x, 4.001f) + frostfield_mod(pixel.y, 4.0f) > 6.0f) {
                color = sample.line1.xyz;
            }
            if (weight < 0.5f && frostfield_mod(pixel.x + 2.0f, 4.001f) + frostfield_mod(pixel.y + 2.0f, 4.0f) > 6.0f) {
                color = sample.line1.xyz;
            }
        }
        colors[id] = float4(color, 1);
    }
    """
}
