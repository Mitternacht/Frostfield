import Foundation
import Metal
import simd
import Synchronization

/// Applies a surface graph's per-vertex position offset on the GPU and rebuilds
/// normals from the deformed triangles.
///
/// The deformed vertices keep the renderer's packed layout, so color, shadow,
/// skinned and picking paths read them like the source mesh. Normals are the
/// area-weighted sum of incident face normals. Vertices that share a position
/// and an authored normal are smoothed together, which joins UV seams while
/// preserving authored hard edges. Tangents are re-orthogonalized.
final class MeshDeformation: @unchecked Sendable {
    private let source: any MTLBuffer
    private let indices: any MTLBuffer
    private let offsets: any MTLBuffer
    private let incident: any MTLBuffer
    private let signs: any MTLBuffer
    private let positions: any MTLComputePipelineState
    private let normals: any MTLComputePipelineState
    private let uniforms: ShaderUniforms?
    /// Whether the vertex offset reads the clock, so clock changes deform again.
    private let readsClock: Bool
    /// Texture writes have no revision the picking cache can observe.
    let supportsPickingAcceleration: Bool
    private let textures: [any MTLTexture]
    private let vertexCount: Int
    private let geometry: MeshGeometry
    private let state = Mutex(Progress())

    struct UniformSnapshot: Sendable {
        let values: [SIMD4<Float>]
        let revision: ShaderUniformRevision
    }

    var uniformRevision: ShaderUniformRevision {
        uniforms?.revision(readingClock: readsClock) ?? ShaderUniformRevision(values: 0)
    }

    func uniformSnapshot() -> UniformSnapshot {
        let values = uniforms?.bufferSnapshot()
        return UniformSnapshot(
            values: values?.values ?? [],
            revision: ShaderUniformRevision(values: values?.revision ?? 0, frame: readsClock ? values?.frameRevision : nil)
        )
    }

    private struct Key: Hashable, Sendable {
        let command: ObjectIdentifier
        let revision: ShaderUniformRevision
    }

    private final class Pose: @unchecked Sendable {
        let revision: ShaderUniformRevision
        let sequence: UInt64
        let vertices: any MTLBuffer

        init(revision: ShaderUniformRevision, sequence: UInt64, vertices: any MTLBuffer) {
            self.revision = revision
            self.sequence = sequence
            self.vertices = vertices
        }
    }

    // Weak metadata is only read or changed under `state`.
    private struct Pending: @unchecked Sendable {
        weak var command: (any MTLCommandBuffer)?
        weak var pose: Pose?
    }

    private struct Progress: Sendable {
        var completed: Pose?
        var pending: [Key: Pending] = [:]
        var sequence: UInt64 = 0
        var encodeCount = 0
    }

    var completedVertices: (any MTLBuffer)? { state.withLock { $0.completed?.vertices } }
    /// How many times the vertex deformation was encoded.
    var encodeCount: Int { state.withLock { $0.encodeCount } }

    init(
        device: any MTLDevice, graph: SurfaceShaderGraph, material: PhysicallyBasedMaterial,
        source: any MTLBuffer, geometry: MeshGeometry, fallbackTexture: () throws -> any MTLTexture
    ) throws {
        guard let function = try graph.metalVertexFunction(named: "frostfield_generated_vertex") else {
            throw MeshRenderer.Failure.unsupportedSurfaceGraph
        }
        let library: any MTLLibrary
        let metal = Self.source(function)
        do {
            library = try ShaderLibraryCache.library(source: metal, device: device)
        } catch {
            throw MeshRenderer.Failure.library(ShaderProgram.diagnose(error, source: metal))
        }
        do {
            positions = try device.makeComputePipelineState(
                function: try Self.function("frostfield_deform_positions", in: library)
            )
            normals = try device.makeComputePipelineState(function: try Self.function("frostfield_deform_normals", in: library))
        } catch let failure as MeshRenderer.Failure {
            throw failure
        } catch {
            throw MeshRenderer.Failure.pipeline(error.localizedDescription)
        }
        guard material.surfaceTextures.allSatisfy({ $0.device === device }) else { throw MeshRenderer.Failure.wrongDevice }
        let missing = MaterialGraph.maximumTextureCount - material.surfaceTextures.count
        textures = material.surfaceTextures + (missing > 0 ? Array(repeating: try fallbackTexture(), count: missing) : [])
        uniforms = material.surfaceUniforms
        readsClock = graph.vertexReadsClock
        supportsPickingAcceleration = !graph.program.uses({ node in
            switch node {
            case .sampleTexture, .triplanarSample, .parallaxOcclusion, .bumpMap: true
            default: false
            }
        }, from: graph.outputs.positionOffset.map { [$0] } ?? [])

        let adjacency = Self.adjacency(for: geometry)
        self.source = source
        self.indices = try Self.buffer(geometry.indices, device: device, label: "deformation indices")
        self.offsets = try Self.buffer(adjacency.offsets, device: device, label: "deformation adjacency offsets")
        self.incident = try Self.buffer(adjacency.triangles.isEmpty ? [0] : adjacency.triangles, device: device,
                                        label: "deformation adjacency")
        self.signs = try Self.buffer(adjacency.signs, device: device, label: "deformation normal signs")
        vertexCount = geometry.vertices.count
        self.geometry = geometry
    }

    /// Leases an immutable vertex result for this command and uniform revision.
    /// An unfinished result is visible only to the command that writes it.
    func acquireVertices(for commandBuffer: any MTLCommandBuffer) throws -> any MTLBuffer {
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw MeshRenderer.Failure.encoder
        }
        let uniforms = uniformSnapshot()
        let revision = uniforms.revision
        let key = Key(command: ObjectIdentifier(commandBuffer), revision: revision)
        let existing: (pose: Pose, isPending: Bool)? = state.withLock { progress in
            progress.pending = progress.pending.filter { $0.value.command != nil && $0.value.pose != nil }
            if let pending = progress.pending[key], pending.command === commandBuffer, let pose = pending.pose {
                return (pose, true)
            }
            if let completed = progress.completed, completed.revision == revision {
                return (completed, false)
            }
            return nil
        }
        if let existing {
            if !existing.isPending, !commandBuffer.retainedReferences {
                commandBuffer.addCompletedHandler { [pose = existing.pose] _ in withExtendedLifetime(pose) {} }
            }
            return existing.pose.vertices
        }

        guard let vertices = source.device.makeBuffer(length: source.length, options: .storageModePrivate) else {
            throw MeshRenderer.Failure.resource("deformed mesh vertices")
        }
        vertices.label = "Frostfield deformed mesh vertices"
        try encode(into: commandBuffer, destination: vertices, values: uniforms.values)
        let pose = state.withLock { progress in
            progress.sequence &+= 1
            let pose = Pose(revision: revision, sequence: progress.sequence, vertices: vertices)
            progress.pending[key] = Pending(command: commandBuffer, pose: pose)
            progress.encodeCount += 1
            return pose
        }
        // The command owns the pending result, even without retained Metal references.
        // Abandoning the command releases its result without publishing it.
        commandBuffer.addCompletedHandler { [weak self, pose] buffer in
            self?.state.withLock { progress in
                progress.pending.removeValue(forKey: key)
                if buffer.status == .completed, (progress.completed?.sequence ?? 0) < pose.sequence {
                    progress.completed = pose
                }
            }
        }
        return vertices
    }

    /// Reads a completed picking pose or evaluates the captured inputs. Without
    /// a snapshot, evaluate fresh inputs, including the current texture contents.
    func deformedGeometry(queue: any MTLCommandQueue, picking snapshot: UniformSnapshot? = nil) throws -> MeshGeometry {
        let (destination, command) = try prepareReadback(queue: queue, snapshot: snapshot)
        command.commit()
        command.waitUntilCompleted()
        return try readGeometry(destination, command: command)
    }

    nonisolated(nonsending) func readPickingGeometry(
        queue: any MTLCommandQueue, snapshot: UniformSnapshot
    ) async throws -> MeshGeometry {
        let (destination, command) = try prepareReadback(queue: queue, snapshot: snapshot)
        command.commit()
        await command.completed()
        try Task.checkCancellation()
        return try readGeometry(destination, command: command)
    }

    private func prepareReadback(
        queue: any MTLCommandQueue, snapshot: UniformSnapshot?
    ) throws -> (any MTLBuffer, any MTLCommandBuffer) {
        guard let destination = queue.device.makeBuffer(length: source.length, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer() else {
            throw MeshRenderer.Failure.resource("deformed geometry readback")
        }
        let completed = snapshot.flatMap { snapshot in
            state.withLock { $0.completed?.revision == snapshot.revision ? $0.completed : nil }
        }
        if let completed {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw MeshRenderer.Failure.encoder }
            blit.label = "Frostfield picking pose readback"
            blit.copy(from: completed.vertices, sourceOffset: 0, to: destination, destinationOffset: 0, size: source.length)
            blit.endEncoding()
        } else {
            try encode(into: commandBuffer, destination: destination, values: (snapshot ?? uniformSnapshot()).values)
        }
        return (destination, commandBuffer)
    }

    private func readGeometry(_ destination: any MTLBuffer, command: any MTLCommandBuffer) throws -> MeshGeometry {
        guard command.status == .completed else {
            throw MeshRenderer.Failure.resource("deformed geometry readback")
        }
        let packed = UnsafeBufferPointer(
            start: destination.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: vertexCount * 4
        )
        let vertices = geometry.vertices.enumerated().map { index, original in
            let position = packed[index * 4]
            let normal = packed[index * 4 + 1]
            let tangent = packed[index * 4 + 3]
            return MeshVertex(
                position: SIMD3(position.x, position.y, position.z), normal: SIMD3(normal.x, normal.y, normal.z),
                textureCoordinate: original.textureCoordinate,
                secondaryTextureCoordinate: original.secondaryTextureCoordinate,
                tangent: original.tangent == nil ? nil : tangent,
                color: original.color
            )
        }
        return try MeshGeometry(vertices: vertices, indices: geometry.indices)
    }

    private func encode(
        into commandBuffer: any MTLCommandBuffer, destination: any MTLBuffer, values: [SIMD4<Float>]
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw MeshRenderer.Failure.encoder }
        encoder.label = "Frostfield mesh deformation"
        var count = UInt32(vertexCount)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        if values.isEmpty {
            var unused = SIMD4<Float>.zero
            encoder.setBytes(&unused, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        } else {
            values.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        }
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBuffer(indices, offset: 0, index: 4)
        encoder.setBuffer(offsets, offset: 0, index: 5)
        encoder.setBuffer(incident, offset: 0, index: 6)
        encoder.setBuffer(signs, offset: 0, index: 7)
        encoder.setTextures(textures, range: 0..<textures.count)
        // Whole threadgroups work on GPUs without non-uniform dispatch, such as the Simulator;
        // both kernels skip indices past the vertex count. A serial compute encoder makes the
        // deformed positions visible to the normal pass.
        for pipeline in [positions, normals] {
            let width = pipeline.threadExecutionWidth
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                MTLSize(width: (vertexCount + width - 1) / width, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
        encoder.endEncoding()
    }

    // MARK: - Adjacency

    struct Adjacency {
        var offsets: [UInt32]
        var triangles: [UInt32]
        var signs: [Float]
    }

    /// Groups vertices by quantized position and matching authored normal, and lists
    /// each group's incident triangles. Signs orient face sums with the authored normals.
    static func adjacency(for geometry: MeshGeometry) -> Adjacency {
        let vertices = geometry.vertices
        var lower = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var upper = -lower
        for vertex in vertices {
            lower = simd_min(lower, vertex.position)
            upper = simd_max(upper, vertex.position)
        }
        let extent = vertices.isEmpty ? 1 : max(simd_length(upper - lower), 1e-6)
        let cell = extent * 1e-5

        struct Key: Hashable { let x: Int64, y: Int64, z: Int64 }
        var groupsByPosition: [Key: [Int]] = [:]
        var group = [Int](repeating: 0, count: vertices.count)
        var groupCount = 0
        for (index, vertex) in vertices.enumerated() {
            let q = (vertex.position / cell).rounded(.toNearestOrAwayFromZero)
            let key = Key(x: Int64(q.x), y: Int64(q.y), z: Int64(q.z))
            let candidates = groupsByPosition[key, default: []]
            if let match = candidates.first(where: { simd_dot(vertices[$0].normal, vertex.normal) > 0.999 * simd_length(vertices[$0].normal) * simd_length(vertex.normal) }) {
                group[index] = group[match]
            } else {
                group[index] = groupCount
                groupCount += 1
                groupsByPosition[key, default: []].append(index)
            }
        }

        var groupTriangles = [[UInt32]](repeating: [], count: groupCount)
        var faceSums = [SIMD3<Float>](repeating: .zero, count: groupCount)
        for triangle in 0..<(geometry.indices.count / 3) {
            let corners = (0..<3).map { Int(geometry.indices[triangle * 3 + $0]) }
            let a = vertices[corners[0]].position
            let face = simd_cross(vertices[corners[1]].position - a, vertices[corners[2]].position - a)
            for corner in Set(corners.map { group[$0] }) {
                groupTriangles[corner].append(UInt32(triangle))
                faceSums[corner] += face
            }
        }

        var offsets: [UInt32] = [0]
        var triangles: [UInt32] = []
        var signs: [Float] = []
        offsets.reserveCapacity(vertices.count + 1)
        signs.reserveCapacity(vertices.count)
        for (index, vertex) in vertices.enumerated() {
            triangles += groupTriangles[group[index]]
            offsets.append(UInt32(triangles.count))
            signs.append(simd_dot(faceSums[group[index]], vertex.normal) < 0 ? -1 : 1)
        }
        return Adjacency(offsets: offsets, triangles: triangles, signs: signs)
    }

    // MARK: - Metal

    private static func source(_ function: String) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;
        \(SurfaceShaderGraph.supportSource)

        struct FrostfieldGraphResources {
            constant float4* uniforms;
            texture2d<float> texture0, texture1, texture2, texture3, texture4, texture5, texture6, texture7;
        };

        static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& resources, uint index) {
            return resources.uniforms[index];
        }

        static inline texture2d<float> frostfield_graph_texture(thread const FrostfieldGraphResources& resources, uint slot) {
            switch (slot) {
                case 0u: return resources.texture0;
                case 1u: return resources.texture1;
                case 2u: return resources.texture2;
                case 3u: return resources.texture3;
                case 4u: return resources.texture4;
                case 5u: return resources.texture5;
                case 6u: return resources.texture6;
                default: return resources.texture7;
            }
        }

        // Vertex deformation samples the base mip level.
        static inline float4 frostfield_graph_sample(
            thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filtering, float2 uv
        ) {
            constexpr sampler repeatLinear(address::repeat, filter::linear);
            constexpr sampler clampLinear(address::clamp_to_edge, filter::linear);
            constexpr sampler mirrorLinear(address::mirrored_repeat, filter::linear);
            constexpr sampler repeatNearest(address::repeat, filter::nearest);
            constexpr sampler clampNearest(address::clamp_to_edge, filter::nearest);
            constexpr sampler mirrorNearest(address::mirrored_repeat, filter::nearest);
            texture2d<float> texture = frostfield_graph_texture(resources, slot);
            if (filtering == 0u) {
                return address == 1u ? texture.sample(clampLinear, uv, level(0.0f))
                    : address == 2u ? texture.sample(mirrorLinear, uv, level(0.0f)) : texture.sample(repeatLinear, uv, level(0.0f));
            }
            return address == 1u ? texture.sample(clampNearest, uv, level(0.0f))
                : address == 2u ? texture.sample(mirrorNearest, uv, level(0.0f)) : texture.sample(repeatNearest, uv, level(0.0f));
        }

        \(function)

        // Four float4 slots per vertex: position, normal, UV0/UV1, tangent.
        kernel void frostfield_deform_positions(
            device const float4* source [[buffer(0)]],
            device float4* destination [[buffer(1)]],
            constant float4* uniforms [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            array<texture2d<float>, 8> textures [[texture(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= count) { return; }
            uint base = id * 4u;
            float4 position = source[base];
            float4 normal = source[base + 1u];
            float4 uv = source[base + 2u];
            FrostfieldVertexInput input = { uv.xy, uv.zw, position.xyz, normal.xyz };
            FrostfieldGraphResources resources = {
                uniforms, textures[0], textures[1], textures[2], textures[3],
                textures[4], textures[5], textures[6], textures[7]
            };
            float3 offset = frostfield_generated_vertex(input, resources);
            destination[base] = float4(position.xyz + offset, 1.0f);
            destination[base + 1u] = normal;
            destination[base + 2u] = uv;
            destination[base + 3u] = source[base + 3u];
        }

        kernel void frostfield_deform_normals(
            device float4* vertices [[buffer(1)]],
            constant uint& count [[buffer(3)]],
            device const uint* indices [[buffer(4)]],
            device const uint* offsets [[buffer(5)]],
            device const uint* incident [[buffer(6)]],
            device const float* signs [[buffer(7)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= count) { return; }
            float3 sum = float3(0);
            for (uint entry = offsets[id]; entry < offsets[id + 1u]; ++entry) {
                uint triangle = incident[entry] * 3u;
                float3 a = vertices[indices[triangle] * 4u].xyz;
                float3 b = vertices[indices[triangle + 1u] * 4u].xyz;
                float3 c = vertices[indices[triangle + 2u] * 4u].xyz;
                sum += cross(b - a, c - a);
            }
            uint base = id * 4u;
            float3 normal = frostfield_graph_unit(sum * signs[id]);
            if (dot(normal, normal) == 0.0f) { return; }
            vertices[base + 1u] = float4(normal, 0.0f);
            float4 tangent = vertices[base + 3u];
            if (tangent.w != 0.0f) {
                float3 projected = frostfield_graph_unit(tangent.xyz - normal * dot(normal, tangent.xyz));
                if (dot(projected, projected) > 0.0f) { vertices[base + 3u] = float4(projected, tangent.w); }
            }
        }
        """
    }

    private static func function(_ name: String, in library: any MTLLibrary) throws -> any MTLFunction {
        guard let function = library.makeFunction(name: name) else { throw MeshRenderer.Failure.function(name) }
        return function
    }

    private static func buffer<T>(_ values: [T], device: any MTLDevice, label: String) throws -> any MTLBuffer {
        let buffer = values.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared) }
        }
        guard let buffer else { throw MeshRenderer.Failure.resource(label) }
        buffer.label = label
        return buffer
    }
}
