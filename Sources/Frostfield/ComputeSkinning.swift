import Foundation
import Metal
import simd

/// Linear blend skinning in a compute pass, for many skinned meshes at once, like
/// Three.js compute skinning on WebGPU.
///
/// Each prepared `Mesh` keeps its bind-pose vertices and four joint influences per
/// vertex on the GPU and owns an output buffer of skinned vertices in the
/// renderer's packed layout: float4 position, normal, both UV sets and tangent per
/// vertex. One `encode` call skins any number of meshes in one compute encoder. The
/// math is `MeshSkin.deform(_:using:)`'s: blended matrix, cofactor normals and
/// handedness-preserving tangents, compiled with safe math.
///
/// `PathTracingScene` can use it for skinned nodes in `updateGeometry(for:in:replacements:)`
/// through its `computeSkinning` property.
public final class ComputeSkinning {
    public let device: any MTLDevice
    private let pipeline: any MTLComputePipelineState

    /// Bind-pose inputs and the skinned output for one mesh.
    public final class Mesh {
        public let geometry: MeshGeometry
        public let skin: MeshSkin
        /// Skinned vertices, 64 bytes each, rewritten by every encode that includes the mesh.
        /// Readers must wait for that encode's command buffer.
        public let output: any MTLBuffer
        let source: any MTLBuffer
        let influences: any MTLBuffer
        let device: any MTLDevice

        init(device: any MTLDevice, geometry: MeshGeometry, skin: MeshSkin) throws {
            guard skin.influences.count == geometry.vertices.count else { throw SkinningError.influenceCount }
            var packed: [SIMD4<Float>] = []
            packed.reserveCapacity(geometry.vertices.count * 4)
            for vertex in geometry.vertices {
                let uv1 = vertex.secondaryTextureCoordinate ?? .zero
                packed += [
                    SIMD4(vertex.position, 1), SIMD4(vertex.normal, 0),
                    SIMD4(vertex.textureCoordinate.x, vertex.textureCoordinate.y, uv1.x, uv1.y), vertex.tangent ?? .zero,
                ]
            }
            let joints = skin.influences.map { influence in
                let j = influence.joints
                return PackedInfluence(joints: SIMD4(UInt32(j.x), UInt32(j.y), UInt32(j.z), UInt32(j.w)), weights: influence.weights)
            }
            source = try ComputeSkinning.buffer(packed, device: device, label: "compute skinning bind pose")
            influences = try ComputeSkinning.buffer(joints, device: device, label: "compute skinning influences")
            guard let output = device.makeBuffer(length: source.length, options: .storageModeShared) else {
                throw MeshRenderer.Failure.resource("compute skinning output")
            }
            output.label = "compute skinned vertices"
            self.output = output
            self.geometry = geometry
            self.skin = skin
            self.device = device
        }

        /// Reads the output with the bind pose's texture coordinates, colors, and indices.
        /// Call after the command buffer that skinned the mesh has completed.
        public func skinnedGeometry() throws -> MeshGeometry {
            let values = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: geometry.vertices.count * 4)
            let vertices = geometry.vertices.indices.map { index -> MeshVertex in
                let source = geometry.vertices[index]
                let base = index * 4
                let position = values[base]
                let normal = values[base + 1]
                let tangent = values[base + 3]
                return MeshVertex(
                    position: SIMD3(position.x, position.y, position.z), normal: SIMD3(normal.x, normal.y, normal.z),
                    textureCoordinate: source.textureCoordinate, secondaryTextureCoordinate: source.secondaryTextureCoordinate,
                    tangent: source.tangent == nil || tangent.w == 0 ? nil : tangent, color: source.color
                )
            }
            return try MeshGeometry(vertices: vertices, indices: geometry.indices)
        }
    }

    /// Matches `FrostfieldComputeInfluence`: 32 bytes.
    struct PackedInfluence {
        var joints: SIMD4<UInt32>
        var weights: SIMD4<Float>
    }

    public init(device: any MTLDevice) throws {
        let library: any MTLLibrary
        do {
            library = try ShaderLibraryCache.library(source: Self.source, device: device)
        } catch {
            throw MeshRenderer.Failure.library(ShaderProgram.diagnose(error, source: Self.source))
        }
        guard let function = library.makeFunction(name: "frostfield_compute_skinning") else {
            throw MeshRenderer.Failure.library("The compute skinning kernel is missing.")
        }
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw MeshRenderer.Failure.pipeline(error.localizedDescription)
        }
        self.device = device
    }

    /// Uploads a mesh's bind pose and influences once.
    public func makeMesh(geometry: MeshGeometry, skin: MeshSkin) throws -> Mesh {
        try Mesh(device: device, geometry: geometry, skin: skin)
    }

    /// Encodes skinning of every mesh with its palette into one compute encoder.
    /// Each palette needs one matrix per joint of its mesh's skin, and each mesh may
    /// appear once.
    public func encode(_ jobs: [(mesh: Mesh, palette: JointPalette)], commandBuffer: any MTLCommandBuffer) throws {
        guard !jobs.isEmpty else { return }
        guard Set(jobs.map { ObjectIdentifier($0.mesh) }).count == jobs.count else { throw Failure.duplicateMesh }
        guard commandBuffer.device === device else { throw MeshRenderer.Failure.wrongDevice }
        for job in jobs {
            guard job.mesh.device === device else { throw MeshRenderer.Failure.wrongDevice }
            guard job.palette.matrices.count == job.mesh.skin.jointCount else { throw SkinningError.paletteCount }
        }
        // A fresh palette buffer per encode keeps earlier command buffers' inputs intact.
        let palettes = try jobs.map { try Self.buffer($0.palette.matrices, device: device, label: "compute skinning palette") }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MeshRenderer.Failure.resource("compute skinning encoder")
        }
        encoder.label = "Frostfield compute skinning"
        encoder.setComputePipelineState(pipeline)
        let width = max(1, min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup))
        for (job, palette) in zip(jobs, palettes) {
            var count = UInt32(job.mesh.geometry.vertices.count)
            encoder.setBuffer(job.mesh.source, offset: 0, index: 0)
            encoder.setBuffer(job.mesh.influences, offset: 0, index: 1)
            encoder.setBuffer(palette, offset: 0, index: 2)
            encoder.setBuffer(job.mesh.output, offset: 0, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
            let groups = (Int(count) + width - 1) / width
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
        encoder.endEncoding()
    }

    /// Skins the meshes on a queue, waits, and returns their skinned geometry in order.
    public func skin(_ jobs: [(mesh: Mesh, palette: JointPalette)], queue: any MTLCommandQueue) throws -> [MeshGeometry] {
        guard let commandBuffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("compute skinning commands") }
        commandBuffer.label = "Frostfield compute skinning"
        try encode(jobs, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MeshRenderer.Failure.resource("compute skinning: \(commandBuffer.error?.localizedDescription ?? "the GPU work failed")")
        }
        return try jobs.map { try $0.mesh.skinnedGeometry() }
    }

    private static func buffer<T>(_ values: [T], device: any MTLDevice, label: String) throws -> any MTLBuffer {
        let length = max(MemoryLayout<T>.stride * values.count, 16)
        guard let buffer = values.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: min(bytes.count, length), options: .storageModeShared)
        }) else {
            throw MeshRenderer.Failure.resource(label)
        }
        buffer.label = label
        return buffer
    }

    public enum Failure: LocalizedError, Equatable {
        case duplicateMesh

        public var errorDescription: String? {
            switch self {
            case .duplicateMesh: "Each mesh may appear once in a compute skinning pass, because it has one output buffer."
            }
        }
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct FrostfieldComputeInfluence {
        uint4 joints;
        float4 weights;
    };

    static float3 frostfield_skin_normal(float3 normal, float4x4 skin) {
        float3 x = skin[0].xyz;
        float3 y = skin[1].xyz;
        float3 z = skin[2].xyz;
        float3 cofactorX = cross(y, z);
        float determinant = dot(x, cofactorX);
        float3x3 cofactors = float3x3(cofactorX, cross(z, x), cross(x, y));
        float3 transformed = (cofactors * normal) * (determinant < 0 ? -1.0f : 1.0f);
        float scale = max(abs(transformed.x), max(abs(transformed.y), abs(transformed.z)));
        if (!isfinite(determinant) || determinant == 0 || !isfinite(scale) || scale == 0) {
            return normal;
        }
        return normalize(transformed / scale);
    }

    static float4 frostfield_skin_tangent(float4 tangent, float4x4 skin) {
        if (tangent.w == 0) {
            return float4(0);
        }
        float determinant = dot(skin[0].xyz, cross(skin[1].xyz, skin[2].xyz));
        float3 direction = (skin * float4(tangent.xyz, 0)).xyz;
        float scale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
        if (!isfinite(determinant) || determinant == 0 || !isfinite(scale) || scale == 0) {
            return float4(0);
        }
        return float4(normalize(direction / scale), determinant < 0 ? -tangent.w : tangent.w);
    }

    kernel void frostfield_compute_skinning(
        const device float4* source [[buffer(0)]],
        const device FrostfieldComputeInfluence* influences [[buffer(1)]],
        const device float4x4* palette [[buffer(2)]],
        device float4* output [[buffer(3)]],
        constant uint& count [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= count) {
            return;
        }
        FrostfieldComputeInfluence influence = influences[id];
        float4 weights = influence.weights;
        float4x4 skin = palette[influence.joints.x] * weights.x + palette[influence.joints.y] * weights.y
            + palette[influence.joints.z] * weights.z + palette[influence.joints.w] * weights.w;
        uint base = id * 4;
        float4 position = source[base];
        output[base] = float4((skin * float4(position.xyz, 1)).xyz, 1);
        output[base + 1] = float4(frostfield_skin_normal(source[base + 1].xyz, skin), 0);
        output[base + 2] = source[base + 2];
        output[base + 3] = frostfield_skin_tangent(source[base + 3], skin);
    }
    """
}
