import Metal
import Synchronization

/// A snapshot of prepared geometry, persistent draw buffers, material textures and render pipelines.
public struct SceneResourceStatistics: Equatable, Sendable {
    public fileprivate(set) var preparedGeometryCount = 0
    public fileprivate(set) var renderPipelineCount = 0
    public fileprivate(set) var bufferCount = 0
    /// Sum of the distinct buffers' logical lengths.
    public fileprivate(set) var bufferLengthBytes = 0
    /// Nil when Metal omits an allocation size for any counted buffer.
    public fileprivate(set) var bufferAllocatedBytes: Int? = 0
    public fileprivate(set) var textureCount = 0
    /// Nil when Metal omits an allocation size for an independently allocated texture.
    public fileprivate(set) var textureAllocatedBytes: Int? = 0
    public fileprivate(set) var externalSurfaceTextureCount = 0
}

/// Device, cache, attachment and transient totals that ``SceneGraph/resourceStatistics()`` does not walk.
public struct DeviceAllocationTotals: Equatable, Sendable {
    /// ``MTLDevice/currentAllocatedSize`` for the device that rendered the scene.
    public var deviceAllocatedBytes: Int
    /// Runtime-compiled shader libraries still held for this device.
    public var shaderCacheEntryCount: Int
    /// UTF-8 source bytes of those cached libraries.
    public var shaderCacheSourceBytes: Int
    /// Sum of `allocatedSize` for the caller-owned attachments passed in.
    public var attachmentAllocatedBytes: Int
    /// Bytes of transient instance buffers allocated because every persistent slot was busy.
    public var transientBufferBytes: Int
}

/// Counts transient instance buffers created when an ``InstanceSet`` cannot reuse a persistent slot.
package enum TransientCommandAllocations {
    private static let bytes = Mutex(0)

    package static func record(_ length: Int) {
        bytes.withLock { $0 += length }
    }

    package static var byteCount: Int {
        bytes.withLock { $0 }
    }
}

public extension SceneGraph {
    /// Counts resources reachable from the current nodes, including hidden nodes
    /// and all prepared LODs. Shared resources and texture views count once.
    /// Allocated bytes are available only when Metal reports all contributing sizes.
    /// Texture bytes backed by a buffer
    /// count with that buffer. IOSurface storage belongs to the producer.
    /// Caller attachments, the shader-library cache and transient command resources are outside this snapshot.
    /// Use ``deviceAllocationTotals(device:attachments:)`` for those.
    func resourceStatistics() -> SceneResourceStatistics {
        var inventory = SceneResourceInventory()
        if let environment {
            for texture in [environment.skyCube, environment.diffuseCube, environment.specularCube, environment.brdfLUT] {
                inventory.recordTexture(texture)
            }
        }
        var nodes = roots
        while let node = nodes.popLast() {
            node.mesh?.recordResources(in: &inventory)
            node.batch?.renderer?.recordResources(in: &inventory)
            if let levels = node.levelsOfDetail?.levels {
                for level in levels { level.mesh.recordResources(in: &inventory) }
            }
            node.primitive?.recordResources(in: &inventory)
            node.instances?.recordResources(in: &inventory)
            nodes.append(contentsOf: node.children)
        }
        return inventory.result
    }

    /// Totals for the device, the shader cache, the supplied attachments and transient instance buffers.
    /// Attachment bytes are the sizes Metal reports. A texture that reports zero contributes zero.
    public func deviceAllocationTotals(
        device: any MTLDevice, attachments: [any MTLTexture] = []
    ) -> DeviceAllocationTotals {
        let cache = ShaderLibraryCache.snapshot(for: device)
        return DeviceAllocationTotals(
            deviceAllocatedBytes: device.currentAllocatedSize,
            shaderCacheEntryCount: cache.entries,
            shaderCacheSourceBytes: cache.sourceBytes,
            attachmentAllocatedBytes: attachments.reduce(0) { $0 + $1.allocatedSize },
            transientBufferBytes: TransientCommandAllocations.byteCount
        )
    }
}

struct SceneResourceInventory {
    private var geometries = Set<ObjectIdentifier>()
    private var buffers = Set<ObjectIdentifier>()
    private var textures = Set<ObjectIdentifier>()
    private var pipelines = Set<ObjectIdentifier>()
    private(set) var result = SceneResourceStatistics()
    private let forbiddenTextures: [any MTLTexture]
    private(set) var hasTextureFeedback = false

    init(forbiddenTextures: [any MTLTexture] = []) {
        self.forbiddenTextures = forbiddenTextures
    }

    mutating func recordGeometry(_ renderer: AnyObject) -> Bool {
        let inserted = geometries.insert(ObjectIdentifier(renderer)).inserted
        if inserted { result.preparedGeometryCount += 1 }
        return inserted
    }

    mutating func recordBuffer(_ buffer: (any MTLBuffer)?) {
        guard let buffer, buffers.insert(ObjectIdentifier(buffer)).inserted else { return }
        result.bufferCount += 1
        result.bufferLengthBytes += buffer.length
        if let total = result.bufferAllocatedBytes {
            let bytes = buffer.allocatedSize
            result.bufferAllocatedBytes = bytes > 0 ? total + bytes : nil
        }
    }

    mutating func recordTexture(_ texture: (any MTLTexture)?) {
        guard var root = texture else { return }
        while let parent = root.parent { root = parent }
        guard textures.insert(ObjectIdentifier(root)).inserted else { return }
        if forbiddenTextures.contains(where: { TextureTransfer.sharesStorage(root, $0) }) {
            hasTextureFeedback = true
        }
        result.textureCount += 1
        if let buffer = root.buffer {
            recordBuffer(buffer)
        } else if root.iosurface != nil {
            result.externalSurfaceTextureCount += 1
        } else if let total = result.textureAllocatedBytes {
            let bytes = root.allocatedSize
            result.textureAllocatedBytes = bytes > 0 ? total + bytes : nil
        }
    }

    mutating func recordPipeline(_ pipeline: (any MTLRenderPipelineState)?) {
        guard let pipeline, pipelines.insert(ObjectIdentifier(pipeline)).inserted else { return }
        result.renderPipelineCount += 1
    }
}
