import Foundation
import Metal
import simd

/// A read-only Metal snapshot of one mesh BVH, prepared for caller-owned shaders.
/// Bind one buffer and prepend the source returned by metalSource() to the shader.
public final class MeshBVHShaderBuffer {
    public let device: any MTLDevice
    public let nodeCount: Int
    public let triangleCount: Int
    let buffer: any MTLBuffer

    private struct PackedNode {
        let minimum: SIMD4<Float>
        let maximum: SIMD4<Float>
        let links: SIMD4<UInt32>
    }

    public enum RenderStage: Sendable {
        case vertex
        case fragment
    }

    init(device: any MTLDevice, geometry: MeshGeometry, nodes: [MeshBVH.Node], order: [UInt32]) throws {
        let slotSize = MemoryLayout<SIMD4<UInt32>>.stride
        let maximumSlots = min(Int(UInt32.max), device.maxBufferLength / slotSize)
        let nodeSlots = MemoryLayout<PackedNode>.stride / slotSize
        guard maximumSlots > 0, nodes.count <= (maximumSlots - 1) / nodeSlots else { throw Failure.bufferSize }
        let triangleStart = 1 + nodes.count * nodeSlots
        guard order.count <= maximumSlots - triangleStart else { throw Failure.bufferSize }
        let positionStart = triangleStart + order.count
        guard geometry.vertices.count <= maximumSlots - positionStart else { throw Failure.bufferSize }
        let length = (positionStart + geometry.vertices.count) * slotSize
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw Failure.allocation
        }
        buffer.label = "Frostfield mesh BVH shader snapshot"
        let data = buffer.contents()
        data.bindMemory(to: SIMD4<UInt32>.self, capacity: 1).initialize(to: SIMD4(
            UInt32(triangleStart), UInt32(positionStart), UInt32(nodes.count), UInt32(order.count)
        ))

        // A preorder node escapes to the next sibling or an ancestor's sibling.
        // A missed box skips its subtree without a per-thread traversal stack.
        let packedNodes = data.advanced(by: slotSize).bindMemory(to: PackedNode.self, capacity: nodes.count)
        var escapes = [UInt32](repeating: UInt32(nodes.count), count: nodes.count)
        for (index, node) in nodes.enumerated() {
            let escape = escapes[index]
            if node.count == 0 {
                escapes[index + 1] = node.start
                escapes[Int(node.start)] = escape
            }
            packedNodes.advanced(by: index).initialize(to: PackedNode(
                minimum: SIMD4(node.lower, 0), maximum: SIMD4(node.upper, 0),
                links: SIMD4(node.start, node.count, escape, 0)
            ))
        }

        let triangles = data.advanced(by: triangleStart * slotSize)
            .bindMemory(to: SIMD4<UInt32>.self, capacity: order.count)
        for (slot, triangle) in order.enumerated() {
            let start = Int(triangle) * 3
            triangles.advanced(by: slot).initialize(to: SIMD4(
                geometry.indices[start], geometry.indices[start + 1], geometry.indices[start + 2], triangle
            ))
        }

        let positions = data.advanced(by: positionStart * slotSize)
            .bindMemory(to: SIMD4<Float>.self, capacity: geometry.vertices.count)
        for (index, vertex) in geometry.vertices.enumerated() {
            positions.advanced(by: index).initialize(to: SIMD4(vertex.position, 0))
        }

        self.device = device
        self.buffer = buffer
        nodeCount = nodes.count
        triangleCount = order.count
    }

    /// Standalone Metal source for FrostfieldMeshBVHRay and
    /// frostfieldMeshBVHNearest. Compile with safe math for finite-ray
    /// validation, exact interval boundaries and infinite far limits.
    public static func metalSource() throws -> String {
        try BVHRayTriangleSource.splice(into: BVHRayTriangleSource.load("FrostfieldMeshBVH"))
    }

    /// Binds the read-only snapshot to one compute buffer slot.
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int) throws {
        try validateBinding(device: encoder.device, index: index)
        encoder.setBuffer(buffer, offset: 0, index: index)
    }

    /// Binds the read-only snapshot to one vertex or fragment buffer slot.
    public func bind(to encoder: any MTLRenderCommandEncoder, stage: RenderStage, index: Int) throws {
        try validateBinding(device: encoder.device, index: index)
        switch stage {
        case .vertex: encoder.setVertexBuffer(buffer, offset: 0, index: index)
        case .fragment: encoder.setFragmentBuffer(buffer, offset: 0, index: index)
        }
    }

    private func validateBinding(device: any MTLDevice, index: Int) throws {
        guard device === self.device else { throw Failure.wrongDevice }
        guard (0..<31).contains(index) else { throw Failure.bindingIndex }
    }

    public enum Failure: LocalizedError, Equatable {
        case bufferSize
        case allocation
        case shaderSource
        case wrongDevice
        case bindingIndex

        public var errorDescription: String? {
            switch self {
            case .bufferSize: "The shader BVH snapshot exceeds the device's buffer size."
            case .allocation: "Metal could not allocate a shader BVH snapshot."
            case .shaderSource: "The package is missing its BVH Metal source."
            case .wrongDevice: "The shader BVH snapshot and encoder must use the same device."
            case .bindingIndex: "The shader BVH buffer index must be between 0 and 30."
            }
        }
    }
}

enum BVHRayTriangleSource {
    /// Replaces the mesh-BVH include with the standalone header source, which carries
    /// the ray-triangle test. Throws when `source` has no such include, so a template
    /// that lost it cannot compile without the symbols its shaders call.
    static func spliceMeshBVH(into source: String) throws -> String {
        let include = "#include \"FrostfieldMeshBVH.h\""
        guard source.contains(include) else { throw MeshBVHShaderBuffer.Failure.shaderSource }
        return source.replacingOccurrences(of: include, with: try MeshBVHShaderBuffer.metalSource())
    }

    static func splice(into source: String) throws -> String {
        let include = "#include \"FrostfieldRayTriangle.h\""
        guard source.contains(include) else { throw MeshBVHShaderBuffer.Failure.shaderSource }
        return source.replacingOccurrences(of: include, with: try load("FrostfieldRayTriangle"))
    }

    static func load(_ name: String) throws -> String {
        guard let url = Bundle.module.url(forResource: name, withExtension: "h"),
              let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw MeshBVHShaderBuffer.Failure.shaderSource
        }
        return source
    }
}
