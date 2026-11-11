import Foundation
import Metal
import simd

/// Completed visibility tests for selected nodes in one encoded scene view.
public struct SceneOcclusionFrame {
    public enum Visibility: Equatable, Sendable {
        /// At least one fragment passed the GPU visibility test.
        case visible
        /// The node encoded color draws, but none passed the visibility test.
        case occluded
        /// No color draw was encoded for this ID.
        case notDrawn
    }

    public let statistics: SceneDrawStatistics
    private let requested: Set<UUID>
    private let sections: [Section]
    private let commandBuffer: any MTLCommandBuffer

    fileprivate init(
        statistics: SceneDrawStatistics, requested: Set<UUID>, sections: [Section], commandBuffer: any MTLCommandBuffer
    ) {
        self.statistics = statistics
        self.requested = requested
        self.sections = sections
        self.commandBuffer = commandBuffer
    }

    /// Reads the result of each requested node once the command buffer has
    /// completed. Calls before completion throw rather than waiting on the GPU.
    public func results() throws -> [UUID: Visibility] {
        switch commandBuffer.status {
        case .completed: break
        case .error: throw SceneOcclusionError.commandFailed(commandBuffer.error?.localizedDescription)
        default: throw SceneOcclusionError.notReady
        }
        var result = Dictionary(uniqueKeysWithValues: requested.map { ($0, Visibility.notDrawn) })
        for section in sections {
            let values = section.buffer.contents().assumingMemoryBound(to: UInt64.self)
            for (index, id) in section.nodes.enumerated() where result[id] != .visible {
                result[id] = values[index] == 0 ? .occluded : .visible
            }
        }
        return result
    }

    fileprivate struct Section: @unchecked Sendable {
        let nodes: [UUID]
        let buffer: any MTLBuffer
    }
}

public enum SceneOcclusionError: LocalizedError, Equatable {
    case queryLimit
    case existingVisibilityBuffer
    case commandBufferState
    case resource
    case notReady
    case commandFailed(String?)

    public var errorDescription: String? {
        switch self {
        case .queryLimit: "A scene view can query visibility for at most 65,536 nodes."
        case .existingVisibilityBuffer: "Use a render pass without an existing visibility result buffer."
        case .commandBufferState: "Encode scene visibility queries before committing the command buffer."
        case .resource: "Metal could not allocate the scene visibility result buffer or descriptor."
        case .notReady: "The command buffer has not completed its scene visibility queries."
        case .commandFailed(let detail): detail ?? "Metal did not complete the scene visibility queries."
        }
    }
}

public extension SceneGraph {
    /// Encodes the normal scene color pass and queries visibility for selected
    /// node IDs. A node with both mesh and primitive draws is visible if either
    /// draw passes. The caller submits the command buffer and reads the frame
    /// after completion. The supplied pass must have no visibility result
    /// buffer. Prepare meshes and clear depth for the selected mode as in the
    /// normal color pass. Discard the command buffer on an encoding failure.
    func encodeWithOcclusionQueries(
        for nodeIDs: Set<UUID>, pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        shadowMap: DirectionalShadowMap? = nil, transmissionBackdrop: (any MTLTexture)? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, depthMode: MeshDepthMode = .forward,
        screenSpaceOcclusion: (any MTLTexture)? = nil,
        scissor: MTLScissorRect? = nil, indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneOcclusionFrame {
        let queries = try SceneOcclusionRecording(nodeIDs: nodeIDs, pass: pass, commandBuffer: commandBuffer)
        let statistics = try MeshRenderer.encode(
            self, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps,
            depthMode: depthMode, transmissionBackdrop: transmissionBackdrop,
            screenOcclusion: screenSpaceOcclusion, scissor: scissor,
            indirectDraws: indirectDraws, occlusionQueries: queries
        )
        return queries.frame(statistics: statistics)
    }

    /// Encodes visibility queries for an explicit raster view. Clip space,
    /// depth mode, attachment and viewport follow the matching `encode` overload.
    func encodeWithOcclusionQueries(
        for nodeIDs: Set<UUID>, pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        clipFromWorld: simd_float4x4, cameraPosition: SIMD3<Float>, directionToCamera: SIMD3<Float>? = nil,
        viewport: MTLViewport? = nil, depthMode: MeshDepthMode = .forward, shadowMap: DirectionalShadowMap? = nil,
        transmissionBackdrop: (any MTLTexture)? = nil, punctualShadowMaps: PunctualShadowMaps? = nil,
        graphCamera: ShaderGraphCamera? = nil, screenSpaceOcclusion: (any MTLTexture)? = nil,
        scissor: MTLScissorRect? = nil, indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> SceneOcclusionFrame {
        let queries = try SceneOcclusionRecording(nodeIDs: nodeIDs, pass: pass, commandBuffer: commandBuffer)
        let statistics = try MeshRenderer.encode(
            self, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap, punctualShadowMaps: punctualShadowMaps,
            view: (clipFromWorld, cameraPosition, directionToCamera), viewport: viewport, depthMode: depthMode,
            transmissionBackdrop: transmissionBackdrop, graphCamera: graphCamera,
            screenOcclusion: screenSpaceOcclusion, scissor: scissor, indirectDraws: indirectDraws, occlusionQueries: queries
        )
        return queries.frame(statistics: statistics)
    }
}

/// Allocates unique offsets per encoded drawable and a separate buffer for
/// each color pass. This keeps an opaque draw and a later transmission draw
/// from overwriting one another's results.
final class SceneOcclusionRecording {
    private let requested: Set<UUID>
    private let commandBuffer: any MTLCommandBuffer
    private var sections: [SceneOcclusionFrame.Section] = []

    init(nodeIDs: Set<UUID>, pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer) throws {
        guard nodeIDs.count <= 65_536 else { throw SceneOcclusionError.queryLimit }
        guard pass.visibilityResultBuffer == nil else { throw SceneOcclusionError.existingVisibilityBuffer }
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw SceneOcclusionError.commandBufferState
        }
        requested = nodeIDs
        self.commandBuffer = commandBuffer
    }

    struct Pass {
        let descriptor: MTLRenderPassDescriptor
        let meshes: [Int: Int]
        let primitives: [Int: Int]
    }

    func prepare(
        _ source: MTLRenderPassDescriptor, meshIndices: [Int], primitiveIndices: [Int], nodes: [ResolvedSceneNode]
    ) throws -> Pass? {
        guard !requested.isEmpty else { return nil }
        let stride = MemoryLayout<UInt64>.stride
        var slots: [UUID] = []
        func offsets(_ indices: [Int]) -> [Int: Int] {
            var result: [Int: Int] = [:]
            for index in indices {
                let id = nodes[index].id
                guard requested.contains(id) else { continue }
                result[index] = slots.count * stride
                slots.append(id)
            }
            return result
        }
        let meshes = offsets(meshIndices)
        let primitives = offsets(primitiveIndices)
        guard !slots.isEmpty else { return nil }
        guard let descriptor = source.copy() as? MTLRenderPassDescriptor,
              let buffer = commandBuffer.device.makeBuffer(length: slots.count * stride, options: .storageModeShared) else {
            throw SceneOcclusionError.resource
        }
        buffer.label = "Frostfield scene visibility"
        buffer.contents().initializeMemory(as: UInt64.self, repeating: 0, count: slots.count)
        descriptor.visibilityResultBuffer = buffer
        let section = SceneOcclusionFrame.Section(nodes: slots, buffer: buffer)
        sections.append(section)
        if !commandBuffer.retainedReferences {
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(section) {} }
        }
        return Pass(descriptor: descriptor, meshes: meshes, primitives: primitives)
    }

    func frame(statistics: SceneDrawStatistics) -> SceneOcclusionFrame {
        SceneOcclusionFrame(statistics: statistics, requested: requested, sections: sections, commandBuffer: commandBuffer)
    }
}
