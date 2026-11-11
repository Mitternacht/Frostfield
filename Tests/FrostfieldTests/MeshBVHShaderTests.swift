import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshBVHShaderTests {
    @Test func `A caller compute shader returns bounded source hits, weights, normals and facing`() throws {
        let fixture = try GPUFixture()
        let hierarchy = try MeshBVH(geometry: mesh())
        #expect(hierarchy.nodeCount > 1)
        let origin = SIMD3<Float>(0.5, 0.5, 4)
        let direction = SIMD3<Float>(0, 0, -2)
        let results = try fixture.query(hierarchy, rays: [
            RayInput(origin, direction, maximum: .infinity),
            RayInput(origin, direction, minimum: 1.5, maximum: 1.5),
            RayInput(origin, direction, maximum: Float(1.5).nextDown),
            RayInput(origin, direction, minimum: Float(1.5).nextUp, maximum: 2),
            RayInput(origin, direction, maximum: 3, side: .back),
            RayInput(SIMD3(0.5, 0.5, -3), SIMD3(0, 0, 1), maximum: 1, side: .front),
            RayInput(SIMD3(-1, 0.5, 4), SIMD3(1, 0, -2), maximum: 1.5),
            RayInput(origin, SIMD3(0, 0, -0.5), maximum: 6),
            RayInput(SIMD3(1, -1, 4), direction),
            RayInput(origin, .zero),
            RayInput(SIMD3(.nan, 0, 4), direction),
            RayInput(origin, direction, minimum: -1),
            RayInput(origin, direction, maximum: .nan),
            RayInput(origin, direction, minimum: 2, maximum: 1),
        ])
        #expect(results.map(\.triangleIndex) == [
            3, 3, .max, 0, 4, 4, 3, 3, .max, .max, .max, .max, .max, .max,
        ])
        for index in [0, 1, 6, 7] {
            #expect(results[index].ids == SIMD4(9, 10, 11, 3))
            #expect(results[index].barycentric == SIMD3(0.5, 0.25, 0.25))
            #expect(results[index].normal == SIMD4(0, 0, 1, 1))
        }
        #expect(results[0].parameter == 1.5)
        #expect(results[3].ids == SIMD4(0, 1, 2, 0))
        #expect(results[3].parameter == 2)
        #expect(results[4].normal == SIMD4(0, 0, -1, 0))
        #expect(results[4].parameter == 3)
        #expect(results[5].normal == SIMD4(0, 0, -1, 1))
        #expect(results[5].parameter == 1)
        #expect(results[7].parameter == 6)
    }

    @Test func `Stackless traversal matches CPU source IDs on shared edges and a larger grid`() throws {
        let fixture = try GPUFixture()
        let hierarchy = try MeshBVH(geometry: .plane(width: 32, height: 32, columns: 32, rows: 32))
        #expect(hierarchy.depth > 5)
        var rays: [RayInput] = []
        for x in stride(from: -17, through: 17, by: 2) {
            for y in stride(from: -17, through: 17, by: 2) {
                rays.append(RayInput(SIMD3(Float(x), Float(y), 2), SIMD3(0, 0, -1), maximum: 2, side: .front))
                rays.append(RayInput(SIMD3(Float(x) + 0.5, Float(y) + 0.5, 2),
                                     SIMD3(0, 0, -1), maximum: 2, side: .front))
            }
        }
        let actual = try fixture.query(hierarchy, rays: rays)
        var hits = 0
        for (input, output) in zip(rays, actual) {
            let ray = try SceneRay(origin: input.origin, direction: input.direction, maximumDistance: 2)
            let cpu = try hierarchy.intersection(ray: ray)
            #expect(output.triangleIndex == cpu.map { UInt32($0.triangleIndex) } ?? .max)
            if let cpu {
                hits += 1
                #expect(output.parameter == cpu.distance)
                #expect(simd_length(output.barycentric - cpu.barycentric) < 1e-6)
                #expect(output.normal == SIMD4(0, 0, 1, 1))
            }
        }
        #expect(hits > 400)
        #expect(hits < rays.count)
    }

    @Test func `Refitting a mesh can publish a new shader snapshot while the old one stays queryable`() throws {
        let fixture = try GPUFixture()
        let hierarchy = try MeshBVH(geometry: mesh())
        let old = try hierarchy.makeShaderBuffer(device: fixture.device)
        let updated = try hierarchy.refitted(updatingPositions: [
            9: SIMD3(30, 0, 1), 10: SIMD3(32, 0, 1), 11: SIMD3(30, 2, 1),
        ]).makeShaderBuffer(device: fixture.device)
        #expect(old.triangleCount == updated.triangleCount)
        #expect(old.nodeCount == updated.nodeCount)
        let ray = RayInput(SIMD3(0.5, 0.5, 4), SIMD3(0, 0, -2))
        #expect(try fixture.query(old, rays: [ray]).first?.triangleIndex == 3)
        #expect(try fixture.query(updated, rays: [ray]).first?.triangleIndex == 0)
        #expect(try fixture.query(old, rays: [ray]).first?.triangleIndex == 3)
    }

    @Test func `Caller vertex and fragment stages can bind one snapshot and shade a pixel from the query`() async throws {
        let fixture = try GPUFixture()
        let hierarchy = try MeshBVH(geometry: mesh())
        let snapshot = try hierarchy.makeShaderBuffer(device: fixture.device)
        let descriptor = MTLRenderPipelineDescriptor()
        let vertex = try #require(fixture.library.makeFunction(name: "probeBVHVertex"))
        let fragment = try #require(fixture.library.makeFunction(name: "probeBVHFragment"))
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        let pipeline = try await fixture.device.makeRenderPipelineState(descriptor: descriptor)
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false
        )
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = .renderTarget
        let output = try #require(fixture.device.makeTexture(descriptor: textureDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = output
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let command = try #require(fixture.queue.makeCommandBuffer())
        let encoder = try #require(command.makeRenderCommandEncoder(descriptor: pass))
        encoder.setRenderPipelineState(pipeline)
        try snapshot.bind(to: encoder, stage: .vertex, index: 4)
        try snapshot.bind(to: encoder, stage: .fragment, index: 7)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        command.commit()
        let readback = try await TextureTransfer.read(from: output, commandQueue: fixture.queue)
        let texel = readback.bytes.withUnsafeBytes { $0.loadUnaligned(as: SIMD4<Float16>.self) }
        #expect(texel == SIMD4(0.5, 0.75, 1, 1))
    }

    private func mesh() throws -> MeshGeometry {
        let placements: [(x: Float, z: Float)] = [(0, 0), (20, 0), (-12, 0), (0, 1), (0, -2), (0, 0), (-20, 0), (12, 0)]
        let vertices = placements.flatMap { placement in
            [SIMD3(placement.x, 0, placement.z), SIMD3(placement.x + 2, 0, placement.z), SIMD3(placement.x, 2, placement.z)]
        }.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }
        var indices = vertices.indices.map(UInt32.init)
        indices.swapAt(13, 14)
        indices[22] = indices[21]
        indices[23] = indices[21]
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    private struct RayInput {
        enum Side: UInt32 { case front, back, both }

        let originMin: SIMD4<Float>
        let directionMax: SIMD4<Float>
        let options: SIMD4<UInt32>
        var origin: SIMD3<Float> { SIMD3(originMin.x, originMin.y, originMin.z) }
        var direction: SIMD3<Float> { SIMD3(directionMax.x, directionMax.y, directionMax.z) }

        init(_ origin: SIMD3<Float>, _ direction: SIMD3<Float>, minimum: Float = 0, maximum: Float = .infinity,
             side: Side = .both) {
            originMin = SIMD4(origin, minimum)
            directionMax = SIMD4(direction, maximum)
            options = SIMD4(side.rawValue, 0, 0, 0)
        }
    }

    private struct GPUHit {
        let ids: SIMD4<UInt32>
        let weights: SIMD4<Float>
        let normal: SIMD4<Float>
        var triangleIndex: UInt32 { ids.w }
        var parameter: Float { weights.w }
        var barycentric: SIMD3<Float> { SIMD3(weights.x, weights.y, weights.z) }
    }

    private struct GPUFixture {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let library: any MTLLibrary
        let compute: any MTLComputePipelineState

        init() throws {
            device = try #require(MTLCreateSystemDefaultDevice())
            queue = try #require(device.makeCommandQueue())
            let options = MTLCompileOptions()
            options.mathMode = .safe
            library = try device.makeLibrary(source: MeshBVHShaderBuffer.metalSource() + Self.program, options: options)
            compute = try device.makeComputePipelineState(function: #require(library.makeFunction(name: "probeBVHRays")))
        }

        func query(_ hierarchy: MeshBVH, rays: [RayInput]) throws -> [GPUHit] {
            try query(hierarchy.makeShaderBuffer(device: device), rays: rays)
        }

        func query(_ snapshot: MeshBVHShaderBuffer, rays: [RayInput]) throws -> [GPUHit] {
            let inputs = try #require(rays.withUnsafeBytes { bytes -> (any MTLBuffer)? in
                guard let address = bytes.baseAddress else { return nil }
                return device.makeBuffer(bytes: address, length: bytes.count, options: .storageModeShared)
            })
            let results = try #require(device.makeBuffer(
                length: rays.count * MemoryLayout<GPUHit>.stride, options: .storageModeShared
            ))
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(compute)
            #expect(throws: MeshBVHShaderBuffer.Failure.bindingIndex) { try snapshot.bind(to: encoder, index: 31) }
            try snapshot.bind(to: encoder, index: 3)
            encoder.setBuffer(inputs, offset: 0, index: 0)
            encoder.setBuffer(results, offset: 0, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: rays.count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(rays.count, compute.threadExecutionWidth), height: 1, depth: 1)
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            #expect(command.status == .completed)
            let pointer = results.contents().bindMemory(to: GPUHit.self, capacity: rays.count)
            return Array(UnsafeBufferPointer(start: pointer, count: rays.count))
        }

        private static let program = """

        struct ProbeRay { float4 originMin, directionMax; uint4 options; };
        struct ProbeHit { uint4 ids; float4 weights, normal; };
        kernel void probeBVHRays(
            device const ProbeRay* rays [[buffer(0)]], device ProbeHit* results [[buffer(1)]],
            device const uint4* bvh [[buffer(3)]], uint index [[thread_position_in_grid]]
        ) {
            ProbeRay input = rays[index];
            FrostfieldMeshBVHHit hit = frostfieldMeshBVHNearest(
                bvh, {input.originMin.xyz, input.directionMax.xyz, input.originMin.w, input.directionMax.w},
                static_cast<FrostfieldMeshBVHSide>(input.options.x)
            );
            results[index] = {uint4(hit.vertexIndices, hit.triangleIndex), float4(hit.barycentric, hit.parameter),
                              float4(hit.geometricNormal, hit.frontFacing ? 1.0f : 0.0f)};
        }

        struct ProbeVaryings { float4 position [[position]]; float weight; };
        vertex ProbeVaryings probeBVHVertex(uint vertexID [[vertex_id]], device const uint4* bvh [[buffer(4)]]) {
            float2 vertices[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            FrostfieldMeshBVHHit hit = frostfieldMeshBVHNearest(bvh, {float3(0.5,0.5,4), float3(0,0,-2), 0, 2});
            return {float4(vertices[vertexID], 0, 1), hit.barycentric.x};
        }

        fragment float4 probeBVHFragment(ProbeVaryings in [[stage_in]], device const uint4* bvh [[buffer(7)]]) {
            FrostfieldMeshBVHHit hit = frostfieldMeshBVHNearest(bvh, {float3(0.5,0.5,4), float3(0,0,-2), 0, 2});
            return float4(in.weight, hit.parameter * 0.5f, hit.frontFacing ? 1.0f : 0.0f, 1);
        }
        """
    }
}
