import Foundation
import Frostfield
import Metal
import Testing
import simd

private var supportsIndirectDraws: Bool {
    MTLCreateSystemDefaultDevice().map { MeshIndirectDraws.isSupported(on: $0) } ?? false
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshIndirectDrawLayoutTests {
    @Test("Indirect arguments validate the native record layout, offsets and device support")
    func layout() throws {
        typealias Native = MTLDrawIndexedPrimitivesIndirectArguments
        #expect(MeshIndirectDraws.recordByteCount == 20)
        #expect(MemoryLayout<Native>.offset(of: \.indexCount) == 0)
        #expect(MemoryLayout<Native>.offset(of: \.instanceCount) == 4)
        #expect(MemoryLayout<Native>.offset(of: \.indexStart) == 8)
        #expect(MemoryLayout<Native>.offset(of: \.baseVertex) == 12)
        #expect(MemoryLayout<Native>.offset(of: \.baseInstance) == 16)

        let fixture = try GeometryRenderFixture()
        let buffer = try #require(fixture.device.makeBuffer(length: 128, options: .storageModePrivate))
        let offsets = [64, 4, 64]
        #expect(try MeshIndirectDraws(buffer: buffer, byteOffsets: offsets).byteOffsets == offsets)
        #expect(throws: MeshIndirectDraws.Failure.count) { try MeshIndirectDraws(buffer: buffer, byteOffsets: []) }
        #expect(throws: MeshIndirectDraws.Failure.count) {
            try MeshIndirectDraws(buffer: buffer, byteOffsets: Array(repeating: 0, count: MeshIndirectDraws.maximumCount + 1))
        }
        for invalid in [-4, 2, 112, Int.max] {
            #expect(throws: MeshIndirectDraws.Failure.offset(invalid)) {
                try MeshIndirectDraws(buffer: buffer, byteOffsets: [invalid])
            }
        }
        let untracked = try #require(fixture.device.makeBuffer(
            length: 20, options: [.storageModePrivate, .hazardTrackingModeUntracked]
        ))
        #expect(throws: MeshIndirectDraws.Failure.untracked) { try MeshIndirectDraws(buffer: untracked) }

        if !MeshIndirectDraws.isSupported(on: fixture.device) {
            let mesh = try fixture.unlit(GeometryRenderFixture.square(0.25))
            let target = try fixture.makeTarget()
            let command = try #require(fixture.queue.makeCommandBuffer())
            let pass = fixture.pass(color: target.color, depth: target.depth)
            let arguments = try MeshIndirectDraws(buffer: buffer)
            let scene = MeshScene(camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2)))
            #expect(throws: MeshIndirectDraws.Failure.unsupported) {
                try mesh.encode(scene, pass: pass, commandBuffer: command, indirectDraws: arguments)
            }
            try mesh.encode(scene, pass: pass, commandBuffer: command)
            try fixture.finish(command)
            #expect(try fixture.pixel(fixture.pixels(of: target.color), x: 32, y: 32).w == 1)
        }
    }

    @Test("A graph validates indirect node ownership and instancing before encoding")
    func graphOwnership() throws {
        let fixture = try GeometryRenderFixture()
        let square = try GeometryRenderFixture.square(0.25)
        let mesh = try fixture.unlit(square)
        let buffer = try #require(fixture.device.makeBuffer(length: 20, options: .storageModePrivate))
        let source = try MeshIndirectDraws(buffer: buffer)
        let ordinary = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let group = SceneNode(children: [ordinary])
        let lod = SceneNode(levelsOfDetail: try LevelOfDetail(levels: [.init(mesh: mesh, distance: 0)]))
        let batch = SceneNode(batch: try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [square], capacity: 1
        ))
        let badInstances = SceneNode(
            mesh: mesh, instances: try InstanceSet(device: fixture.device, transforms: [matrix_identity_float4x4])
        )
        let graph = SceneGraph(roots: [group, lod, batch, badInstances], camera: fixture.camera)
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let missing = UUID()
        #expect(throws: MeshIndirectDraws.Failure.unknownNode(missing)) {
            try graph.encode(pass: pass, commandBuffer: command, indirectDraws: [missing: source])
        }
        for invalid in [group.id, lod.id, batch.id] {
            #expect(throws: MeshIndirectDraws.Failure.incompatibleNode(invalid)) {
                try graph.encode(pass: pass, commandBuffer: command, indirectDraws: [invalid: source])
            }
        }
        #expect(throws: MeshRenderer.Failure.instancingNotPrepared) {
            try graph.encode(pass: pass, commandBuffer: command, indirectDraws: [badInstances.id: source])
        }
        let valid = SceneGraph(roots: [group], camera: fixture.camera)
        if !MeshIndirectDraws.isSupported(on: fixture.device) {
            #expect(throws: MeshIndirectDraws.Failure.unsupported) {
                try valid.encode(pass: pass, commandBuffer: command, indirectDraws: [ordinary.id: source])
            }
        }
        #expect(try valid.encode(pass: pass, commandBuffer: command).drawnMeshCount == 1)
        try fixture.finish(command)
        #expect(try fixture.pixel(fixture.pixels(of: target.color), x: 32, y: 32).w == 1)
    }
}

@Suite(.enabled(if: supportsIndirectDraws, "Requires Metal indexed indirect arguments"))
struct MeshIndirectDrawTests {
    let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    @Test("Compute changes the drawn index range twice in one unretained command buffer")
    func computeDraws() throws {
        let leftTarget = try fixture.makeTarget()
        let rightTarget = try fixture.makeTarget()
        let command = try #require(fixture.queue.makeCommandBufferWithUnretainedReferences())
        let producer = try makeProducer()
        try autoreleasepool {
            let buffer = try makeSource()
            let arguments = try MeshIndirectDraws(buffer: buffer, byteOffsets: [64, 4])
            let mesh = try fixture.unlit(twoTriangles(), material: .textured(makeTexture()))
            let scene = MeshScene(
                camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2)),
                materialParameters: SIMD4(0.5, 0.75, 0.25, 1)
            )
            try write([.init(offset: 64, count: 3, start: 0), .init(offset: 4, count: 0, start: 3)],
                      into: buffer, producer: producer, command: command)
            try mesh.encode(
                scene, pass: fixture.pass(color: leftTarget.color, depth: leftTarget.depth),
                commandBuffer: command, indirectDraws: arguments
            )
            try write([.init(offset: 64, count: 0, start: 0), .init(offset: 4, count: 3, start: 3)],
                      into: buffer, producer: producer, command: command)
            try mesh.encode(
                scene, pass: fixture.pass(color: rightTarget.color, depth: rightTarget.depth),
                commandBuffer: command, indirectDraws: arguments
            )
        }
        try withExtendedLifetime(producer) { try fixture.finish(command) }
        let left = try fixture.pixels(of: leftTarget.color)
        let right = try fixture.pixels(of: rightTarget.color)
        let tint = SIMD4<Float>(32.0 / 255, 96.0 / 255, 48.0 / 255, 1)
        #expect(isNearly(fixture.pixel(left, x: 20, y: 32), tint))
        #expect(fixture.pixel(left, x: 44, y: 32) == .zero)
        #expect(fixture.pixel(right, x: 20, y: 32) == .zero)
        #expect(isNearly(fixture.pixel(right, x: 44, y: 32), tint))
    }

    @Test("Unsafe GPU records draw nothing and a valid nonzero base instance keeps its tint")
    func instanceValidation() throws {
        let mesh = try fixture.unlit(GeometryRenderFixture.square(0.26), supportsInstancing: true)
        let instances = try InstanceSet(
            device: fixture.device, transforms: [translationMatrix(-0.75), translationMatrix(0.75)],
            tints: [SIMD4(1, 0, 0, 1), SIMD4(0, 0, 1, 1)]
        )
        let source = try makeSource(length: 256)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let producer = try makeProducer()
        let records: [Record] = [
            .init(offset: 4, count: .max, start: 0),
            .init(offset: 36, count: 3, start: .max),
            .init(offset: 68, count: 3, start: 0, baseVertex: -1),
            .init(offset: 100, count: 3, start: 0, baseVertex: 2),
            .init(offset: 132, count: 6, start: 0, instanceCount: 2, baseInstance: 1),
            .init(offset: 164, count: 6, start: 0, baseInstance: .max),
            .init(offset: 196, count: 6, start: 0, baseInstance: 1),
        ]
        try write(records, into: source, producer: producer, command: command)
        let draws = try MeshIndirectDraws(buffer: source, byteOffsets: records.map { Int($0.offset) })
        let target = try fixture.makeTarget()
        try mesh.encode(
            MeshScene(camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2))),
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command,
            indirectDraws: draws, instances: instances
        )
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: target.color)
        #expect(fixture.pixel(pixels, x: 20, y: 32) == .zero)
        #expect(fixture.pixel(pixels, x: 44, y: 32) == SIMD4(0, 0, 1, 1))
    }

    @Test("Positive and negative base vertices use the selected range in graph and standalone draws")
    func signedBaseVertices() throws {
        let mesh = try fixture.unlit(twoTriangles())
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let node = SceneNode(transform: transform, mesh: mesh)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let source = try makeSource()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let producer = try makeProducer()
        try write([
            .init(offset: 4, count: 3, start: 0, baseVertex: 3),
            .init(offset: 64, count: 3, start: 3, baseVertex: -3),
        ], into: source, producer: producer, command: command)
        let positive = try MeshIndirectDraws(buffer: source, byteOffsets: [4])
        let negative = try MeshIndirectDraws(buffer: source, byteOffsets: [64])
        let right = try fixture.makeTarget()
        let left = try fixture.makeTarget()
        let stats = try graph.encode(
            pass: fixture.pass(color: right.color, depth: right.depth), commandBuffer: command,
            indirectDraws: [node.id: positive]
        )
        #expect(stats.indirectArgumentDrawCount == 1)
        #expect(stats.submittedTriangleCount == 0)
        try mesh.encode(
            MeshScene(camera: fixture.camera, transform: transform),
            pass: fixture.pass(color: left.color, depth: left.depth), commandBuffer: command, indirectDraws: negative
        )
        try fixture.finish(command)
        let rightPixels = try fixture.pixels(of: right.color)
        let leftPixels = try fixture.pixels(of: left.color)
        #expect(fixture.pixel(rightPixels, x: 20, y: 32) == .zero)
        #expect(fixture.pixel(rightPixels, x: 44, y: 32).w == 1)
        #expect(fixture.pixel(leftPixels, x: 20, y: 32).w == 1)
        #expect(fixture.pixel(leftPixels, x: 44, y: 32) == .zero)
    }

    @Test("The GPU checks both partial index blocks and the aligned interior of shifted ranges")
    func indexedRangeEdges() throws {
        let sourceGeometry = try twoTriangles()
        var indices = (0..<330).map { UInt32($0 % 3) }
        indices[195] = 3
        indices[240] = 3
        let mesh = try fixture.unlit(MeshGeometry(vertices: sourceGeometry.vertices, indices: indices))
        let scene = MeshScene(camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2)))
        let source = try makeSource(length: 256)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let producer = try makeProducer()
        try write([
            // Both high indices are outside this draw's selected interval.
            .init(offset: 4, count: 192, start: 3, baseVertex: 3),
            // The first offending index lies in the partial trailing block.
            .init(offset: 36, count: 195, start: 3, baseVertex: 3),
            // Both offending indices are in the first partial block.
            .init(offset: 68, count: 132, start: 195, baseVertex: 3),
            // The aligned interior contains both offending indices.
            .init(offset: 100, count: 96, start: 192, baseVertex: 3),
            .init(offset: 132, count: 192, start: 3, baseVertex: .max),
            .init(offset: 164, count: 192, start: 3, baseVertex: .min),
        ], into: source, producer: producer, command: command)
        let accepted = try fixture.makeTarget()
        let suppressed = try fixture.makeTarget()
        try mesh.encode(
            scene, pass: fixture.pass(color: accepted.color, depth: accepted.depth), commandBuffer: command,
            indirectDraws: MeshIndirectDraws(buffer: source, byteOffsets: [4])
        )
        try mesh.encode(
            scene, pass: fixture.pass(color: suppressed.color, depth: suppressed.depth), commandBuffer: command,
            indirectDraws: MeshIndirectDraws(buffer: source, byteOffsets: [36, 68, 100, 132, 164])
        )
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: accepted.color)
        #expect(fixture.pixel(pixels, x: 20, y: 32) == .zero)
        #expect(fixture.pixel(pixels, x: 44, y: 32).w == 1)
        #expect(try fixture.pixels(of: suppressed.color).allSatisfy { $0 == .zero })
    }

    @Test("Multiple argument offsets preserve their order and repeat a draw")
    func orderedOffsets() throws {
        let square = try GeometryRenderFixture.square(0.25)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: square,
            supportsInstancing: true, renderState: .transparent
        )
        let instances = try InstanceSet(
            device: fixture.device, transforms: [.init(diagonal: SIMD4(repeating: 1)), .init(diagonal: SIMD4(repeating: 1))],
            tints: [SIMD4(1, 0, 0, 0.5), SIMD4(0, 0, 1, 0.5)]
        )
        let source = try makeSource()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let producer = try makeProducer()
        try write([.init(offset: 4, count: 6, start: 0), .init(offset: 64, count: 6, start: 0, baseInstance: 1)],
                  into: source, producer: producer, command: command)
        let arguments = try MeshIndirectDraws(buffer: source, byteOffsets: [64, 4, 64])
        let target = try fixture.makeTarget()
        try mesh.encode(
            MeshScene(camera: fixture.camera, transform: .init(translation: SIMD3(0, 0, -2))),
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command,
            indirectDraws: arguments, instances: instances
        )
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: target.color)
        #expect(isNearly(fixture.pixel(pixels, x: 32, y: 32), SIMD4(0.055, 0, 0.625, 0.875)))
    }

    @Test("Graph arguments follow parent transforms, instance selection, visibility, and render order")
    func graphColor() throws {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(0.26),
            supportsInstancing: true, renderState: .transparent
        )
        let instances = try InstanceSet(
            device: fixture.device, transforms: [translationMatrix(-0.8), translationMatrix(0.4)],
            tints: [SIMD4(repeating: 1), SIMD4(repeating: 1)]
        )
        let indirect = SceneNode(
            transform: .init(translation: SIMD3(0.1, 0, 0)), mesh: mesh, materialParameters: SIMD4(1, 0, 0, 0.5),
            instances: instances, renderOrder: 2
        )
        let direct = SceneNode(
            transform: .init(translation: SIMD3(0.5, 0, 0)), mesh: mesh, materialParameters: SIMD4(0, 0, 1, 0.5),
            renderOrder: 1
        )
        let hiddenChild = SceneNode(transform: .init(translation: SIMD3(0, 0.65, 0)), mesh: mesh, instances: instances)
        let hidden = SceneNode(children: [hiddenChild], isVisible: false)
        let layered = SceneNode(
            transform: .init(translation: SIMD3(0, -0.65, 0)), mesh: mesh, instances: instances, layers: SceneLayers(layer: 2)
        )
        let culled = SceneNode(transform: .init(translation: SIMD3(10, 0, 0)), mesh: mesh, instances: instances)
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0.25, 0, -2)), children: [indirect, direct, hidden, layered, culled]
        )
        let graph = SceneGraph(roots: [parent], camera: fixture.camera)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let producer = try makeProducer()
        let source = try makeSource()
        try write([.init(offset: 4, count: 6, start: 0, baseInstance: 1), .init(offset: 64, count: 0, start: 0)],
                  into: source, producer: producer, command: command)
        let arguments = try MeshIndirectDraws(buffer: source, byteOffsets: [4, 64])
        let target = try fixture.makeTarget()
        let stats = try graph.encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: command,
            indirectDraws: [indirect.id: arguments, hiddenChild.id: arguments, layered.id: arguments, culled.id: arguments]
        )
        #expect(stats.drawnMeshCount == 3 && stats.drawnInstanceCount == 1)
        #expect(stats.indirectArgumentDrawCount == 2 && stats.indirectBatchExecutionCount == 0 && stats.culledMeshCount == 1)
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: target.color)
        #expect(isNearly(fixture.pixel(pixels, x: 44, y: 32), SIMD4(0.11, 0, 0.25, 0.75)))
        #expect(fixture.pixel(pixels, x: 25, y: 32) == .zero)
        #expect(fixture.pixel(pixels, x: 42, y: 22) == .zero)
        #expect(fixture.pixel(pixels, x: 42, y: 42) == .zero)
    }

    @Test("GPU-selected graph triangles write directional shadow, surface attributes and selection depth")
    func graphDepthPasses() throws {
        let geometry = try twoTriangles(normal: simd_normalize(SIMD3(0, 1, 1)))
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)), mesh: try fixture.unlit(geometry)
        )
        let receiver = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let graph = SceneGraph(roots: [receiver, blocker], camera: fixture.camera)
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        let source = try makeSource()
        let command = try #require(fixture.queue.makeCommandBuffer())
        let producer = try makeProducer()
        try write([.init(offset: 4, count: 3, start: 3)], into: source, producer: producer, command: command)
        let indirect = [blocker.id: try MeshIndirectDraws(buffer: source, byteOffsets: [4])]
        let color = try fixture.makeTarget()
        let stats = try graph.encode(
            pass: fixture.pass(color: color.color, depth: color.depth), commandBuffer: command,
            shadowMap: shadowMap, indirectDraws: indirect
        )
        #expect(stats.shadowDrawCount == 1 && stats.indirectArgumentDrawCount == 2)

        let attributeTarget = try fixture.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: fixture.side, height: fixture.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let attributes = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let attributeStats = try graph.encodeSurfaceAttributes(
            pass: fixture.pass(color: attributes, depth: attributeTarget.depth), commandBuffer: command,
            clipFromWorld: fixture.camera.projection(aspectRatio: 1), cameraPosition: .zero, indirectDraws: indirect
        )
        #expect(attributeStats.indirectArgumentDrawCount == 1)

        let outline = try SelectionOutline(device: fixture.device)
        #expect(try outline.encodeSelection(
            graph: graph, selection: [blocker.id], width: fixture.side, height: fixture.side,
            commandBuffer: command, indirectDraws: indirect
        ) == 1)
        let mask = try #require(outline.selectionMask)
        let bytesPerRow = fixture.side * MemoryLayout<Float>.stride
        let bytes = bytesPerRow * fixture.side
        let maskBuffer = try #require(fixture.device.makeBuffer(length: bytes, options: .storageModeShared))
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(
            from: mask, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(),
            sourceSize: MTLSize(width: mask.width, height: mask.height, depth: 1), to: maskBuffer,
            destinationOffset: 0, destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytes
        )
        blit.endEncoding()
        try fixture.finish(command)
        let pixels = try fixture.pixels(of: color.color)
        #expect(abs(fixture.pixel(pixels, x: 32, y: 32).x - 0.16) < 0.01)
        func packedNormal(x: Int) -> UInt32 {
            var texel = SIMD2<UInt32>.zero
            attributes.getBytes(&texel, bytesPerRow: MemoryLayout<SIMD2<UInt32>>.stride,
                                from: MTLRegionMake2D(x, 32, 1, 1), mipmapLevel: 0)
            return texel.x
        }
        #expect(packedNormal(x: 20) == 0 && packedNormal(x: 44) != 0)
        let depth = maskBuffer.contents().bindMemory(to: Float.self, capacity: fixture.side * fixture.side)
        #expect(depth[32 * fixture.side + 20] == 1 && depth[32 * fixture.side + 44] < 1)
    }

    private struct Record {
        let offset: UInt32
        let count: UInt32
        let start: UInt32
        var instanceCount: UInt32 = 1
        var baseVertex: Int32 = 0
        var baseInstance: UInt32 = 0
    }

    private func makeSource(length: Int = 128) throws -> any MTLBuffer {
        try #require(fixture.device.makeBuffer(length: length, options: [.storageModePrivate, .hazardTrackingModeTracked]))
    }

    private func makeProducer() throws -> any MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        struct Record { uint offset, count, start, instanceCount; int baseVertex; uint baseInstance; };
        kernel void write_indirect(
            device uint *buffer [[buffer(0)]], constant Record *records [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            const Record record = records[id];
            device uint *args = buffer + record.offset / 4;
            args[0] = record.count;
            args[1] = record.instanceCount;
            args[2] = record.start;
            args[3] = as_type<uint>(record.baseVertex);
            args[4] = record.baseInstance;
        }
        """
        let library = try fixture.device.makeLibrary(source: source, options: nil)
        return try fixture.device.makeComputePipelineState(function: #require(library.makeFunction(name: "write_indirect")))
    }

    private func write(
        _ records: [Record], into buffer: any MTLBuffer, producer: any MTLComputePipelineState, command: any MTLCommandBuffer
    ) throws {
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(producer)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        records.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        encoder.dispatchThreadgroups(
            MTLSize(width: records.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func twoTriangles(normal: SIMD3<Float> = SIMD3(0, 0, 1)) throws -> MeshGeometry {
        let points = [SIMD2<Float>(-0.25, -0.22), SIMD2(0.25, -0.22), SIMD2(0, 0.3)]
        let vertices = [Float(-0.75), 0.75].flatMap { x in
            points.map { MeshVertex(position: SIMD3(x + $0.x, $0.y, 0), normal: normal) }
        }
        return try MeshGeometry(vertices: vertices, indices: [0, 1, 2, 3, 4, 5])
    }

    private func makeTexture() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        var rgba = SIMD4<UInt8>(64, 128, 192, 255)
        texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &rgba, bytesPerRow: 4)
        return texture
    }
}
