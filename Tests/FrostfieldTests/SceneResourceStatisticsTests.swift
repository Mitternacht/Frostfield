@testable import Frostfield
import Metal
import Testing
import simd

struct SceneResourceStatisticsTests {
    @Test("Scene resource inventory counts shared drawables and texture storage once")
    func sharedStorage() throws {
        let fixture = try GeometryRenderFixture()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 8, height: 8, mipmapped: true
        )
        descriptor.usage = [.shaderRead, .pixelFormatView]
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let view = try #require(texture.makeTextureView(
            pixelFormat: .rgba8Unorm, textureType: .type2D, levels: 1..<2, slices: 0..<1
        ))
        let geometry = try GeometryRenderFixture.square(1)
        let near = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .textured(texture), supportsInstancing: true
        )
        let far = try fixture.unlit(geometry, material: .textured(view))
        let lod = try LevelOfDetail(levels: [.init(mesh: near, distance: 0), .init(mesh: far, distance: 10)])
        let instances = try InstanceSet(device: fixture.device, transforms: [matrix_identity_float4x4], capacity: 16)
        let points = try PointsRenderer(device: fixture.device, cloud: PointCloud(positions: [.zero]), material: PointsMaterial())
        let roots = [
            SceneNode(mesh: near, instances: instances),
            SceneNode(levelsOfDetail: lod, isVisible: false),
            SceneNode(children: [SceneNode(primitive: .points(points))]),
            SceneNode(mesh: near, instances: instances),
        ]
        var graph = SceneGraph(roots: roots, camera: fixture.camera)
        let stats = graph.resourceStatistics()
        #expect(stats.preparedGeometryCount == 3)
        #expect(stats.renderPipelineCount >= 3)
        #expect(stats.bufferCount == 5 + InstanceSet.bufferCount)
        #expect(stats.bufferLengthBytes >= InstanceSet.bufferCount * 16 * 64)
        let textureBytes = texture.allocatedSize > 0 ? texture.allocatedSize : nil
        #expect(stats.textureCount == 1 && stats.textureAllocatedBytes == textureBytes)
        #expect(stats.externalSurfaceTextureCount == 0)

        graph.roots.removeAll()
        let empty = graph.resourceStatistics()
        #expect(empty.preparedGeometryCount == 0 && empty.renderPipelineCount == 0)
        #expect(empty.bufferCount == 0 && empty.textureCount == 0)
        #expect(empty.bufferLengthBytes == 0)
        #expect(empty.bufferAllocatedBytes == 0 && empty.textureAllocatedBytes == 0)
    }

    @Test("Buffer lengths remain available when Metal omits the allocation size")
    func reportedBytes() throws {
        let fixture = try GeometryRenderFixture()
        let buffer = try #require(fixture.device.makeBuffer(length: 1024, options: .storageModeShared))
        var inventory = SceneResourceInventory()
        inventory.recordBuffer(buffer)
        inventory.recordBuffer(buffer)
        let stats = inventory.result
        let expected = buffer.allocatedSize > 0 ? buffer.allocatedSize : nil
        #expect(stats.bufferCount == 1 && stats.bufferLengthBytes == buffer.length)
        #expect(stats.bufferAllocatedBytes == expected)
    }

    @Test("Device totals follow the shader cache, attachment sizes and transient instance buffers")
    func deviceTotals() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let marker = UUID().uuidString.replacingOccurrences(of: "-", with: "_")
        let source = "kernel void frostfield_cache_probe_\(marker)() {}"
        let before = ShaderLibraryCache.snapshot(for: device)
        _ = try ShaderLibraryCache.library(source: source, device: device)
        let after = ShaderLibraryCache.snapshot(for: device)
        #expect(after.entries == before.entries + 1)
        #expect(after.sourceBytes == before.sourceBytes + source.utf8.count)
        _ = try ShaderLibraryCache.library(source: source, device: device)
        #expect(ShaderLibraryCache.snapshot(for: device).sourceBytes == after.sourceBytes)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 32, height: 16, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let totals = SceneGraph(roots: [], camera: camera).deviceAllocationTotals(device: device, attachments: [texture])
        #expect(totals.shaderCacheEntryCount == after.entries)
        #expect(totals.shaderCacheSourceBytes == after.sourceBytes)
        #expect(totals.deviceAllocatedBytes == device.currentAllocatedSize)
        #expect(totals.attachmentAllocatedBytes == texture.allocatedSize)
        // The simulator reports zero for this attachment. A device that reports a size must contribute it.
        if texture.allocatedSize > 0 {
            #expect(totals.attachmentAllocatedBytes > 0)
        }

        let set = try InstanceSet(device: device, transforms: [matrix_identity_float4x4])
        let transientBefore = TransientCommandAllocations.byteCount
        var held: [any MTLCommandBuffer] = []
        for index in 0..<4 {
            var transform = matrix_identity_float4x4
            transform.columns.3 = SIMD4(Float(index), 0, 0, 1)
            try set.setTransform(transform, at: 0)
            let command = try #require(queue.makeCommandBuffer())
            _ = try set.acquireBuffer(for: command)
            held.append(command)
        }
        let transient = SceneGraph(roots: [], camera: camera).deviceAllocationTotals(device: device)
        #expect(transient.transientBufferBytes > transientBefore)
        #expect(held.count == 4)
    }
}
