import Foundation
import Metal
import simd

/// Immutable source data and the prepared spines are resident across offset edits.
final class CurveFlowMetal: @unchecked Sendable {
    static let coordinateLimit: Double = 1e7
    private static let maximumIntervals = 131_072

    struct Interval {
        let centerStart: SIMD4<Float>
        let deltaEnd: SIMD4<Float>
        let tangentSpan: SIMD4<Float>
        let nextTangent: SIMD4<Float>
        let rotation: SIMD4<Float>
        let axisAngle: SIMD4<Float>
    }

    struct Placement {
        let first: UInt32
        let count: UInt32
        let closed: UInt32
        let offset: Float

        func shifted(to offset: Float) -> Self {
            Self(first: first, count: count, closed: closed, offset: offset)
        }
    }

    private struct Grid {
        let sourceCount: UInt32
        let outputCount: UInt32
    }

    private let device: any MTLDevice
    private let pipeline: any MTLComputePipelineState
    private let source: any MTLBuffer
    private let intervals: any MTLBuffer
    private let grid: Grid
    private let spines: [UUID: Placement]
    private let coordinateScale: Double

    init(
        device: any MTLDevice, mesh: MeshGeometry, placements: [CurveDeformer.Placement],
        sourceXOrigin: Double, distanceScale: Double
    ) throws {
        guard (1e-3...1e3).contains(distanceScale) else { throw CurveFlow.Failure.metalDomain }
        var localX = 0.0, radius = 0.0
        let vertices = try mesh.vertices.map { vertex in
            let position = SIMD3<Double>(vertex.position)
            let x = (position.x - sourceXOrigin) * distanceScale
            localX = max(localX, abs(x))
            radius = max(radius, simd_length(SIMD2(position.y, position.z)))
            guard localX <= Self.coordinateLimit, radius <= Self.coordinateLimit else {
                throw CurveFlow.Failure.metalDomain
            }
            // Prepare the immutable longitudinal term in Double so a distant
            // source origin does not lose precision in a Float subtraction.
            let normal = simd_normalize(SIMD3<Double>(vertex.normal))
            if let tangent = vertex.tangent {
                let direction = SIMD3<Double>(Double(tangent.x), Double(tangent.y), Double(tangent.z))
                guard abs(simd_dot(normal, direction)) < 0.01 else { throw CurveFlow.Failure.metalDomain }
            }
            return PackedMeshVertex(
                vertex, packedPosition: SIMD3(Float(x), vertex.position.y, vertex.position.z), packedNormal: SIMD3<Float>(normal)
            )
        }
        var table: [Interval] = []
        var spines: [UUID: Placement] = [:]
        var maxLength = 0.0
        for placement in placements where spines[placement.curve.frameTableID] == nil {
            let curve = placement.curve
            guard curve.boundary == .open || localX <= 4 * curve.length else { throw CurveFlow.Failure.metalDomain }
            let records = try curve.metalIntervals(crossSectionRadius: radius)
            guard records.count <= Self.maximumIntervals - table.count else { throw CurveFlow.Failure.metalDomain }
            spines[curve.frameTableID] = Placement(
                first: UInt32(table.count), count: UInt32(records.count), closed: curve.boundary == .closed ? 1 : 0,
                offset: 0
            )
            table += records
            maxLength = max(maxLength, curve.length)
        }
        let pipeline: any MTLComputePipelineState
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let function = library.makeFunction(name: "frostfield_curve_flow_write") else {
                throw CurveFlow.Failure.compute("The curve flow kernel is missing.")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw CurveFlow.Failure.compute(error.localizedDescription)
        }
        self.device = device
        self.pipeline = pipeline
        self.source = try Self.makeBuffer(device, vertices, label: "Frostfield curve flow source")
        self.intervals = try Self.makeBuffer(device, table, label: "Frostfield curve flow frame table")
        self.spines = spines
        self.coordinateScale = max(localX, radius, maxLength)
        self.grid = Grid(sourceCount: UInt32(vertices.count), outputCount: UInt32(vertices.count * placements.count))
    }

    func packed(_ placement: CurveDeformer.Placement) throws -> Placement {
        guard let spine = spines[placement.curve.frameTableID] else { throw CurveFlow.Failure.metalSpine }
        guard abs(placement.offset) <= Self.coordinateLimit,
              spine.closed == 0 || abs(placement.offset) <= 65_536 * placement.curve.length else {
            throw CurveFlow.Failure.metalDomain
        }
        let offset = spine.closed == 1
            ? placement.offset.truncatingRemainder(dividingBy: placement.curve.length) : placement.offset
        return spine.shifted(to: Float(offset))
    }

    func cullingBounds(_ bounds: AxisAlignedBox, placements: [Placement]) -> AxisAlignedBox {
        // Enclose the Float evaluator's position around the Double reference.
        let offsetScale = Double(placements.map { abs($0.offset) }.max() ?? 0)
        let positionScale = Double(simd_reduce_max(simd_max(simd_abs(bounds.minimum), simd_abs(bounds.maximum))))
        let margin = Float(max(coordinateScale, offsetScale, positionScale) * 0.000_03)
        return AxisAlignedBox(
            validatedMinimum: bounds.minimum - SIMD3(repeating: margin),
            maximum: bounds.maximum + SIMD3(repeating: margin)
        )
    }

    func encode(
        placements: [Placement], distanceScale: Float, output: any MTLBuffer, commandBuffer: any MTLCommandBuffer
    ) throws {
        let placementBuffer = try Self.makeBuffer(device, placements, label: "Frostfield curve flow offsets")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CurveFlow.Failure.compute("Metal could not start a curve flow compute encoder.")
        }
        encoder.label = "Frostfield curve flow"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(intervals, offset: 0, index: 1)
        encoder.setBuffer(placementBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var grid = self.grid, scale = distanceScale
        encoder.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 4)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
        let width = pipeline.threadExecutionWidth
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(grid.outputCount) + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        if !commandBuffer.retainedReferences {
            let retained = Retained(writer: self, placements: placementBuffer)
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        }
    }

    private struct Retained: @unchecked Sendable {
        let writer: CurveFlowMetal
        let placements: any MTLBuffer
    }

    private static func makeBuffer<T>(_ device: any MTLDevice, _ elements: [T], label: String) throws -> any MTLBuffer {
        try elements.withUnsafeBytes { bytes in
            guard bytes.count <= device.maxBufferLength, let address = bytes.baseAddress,
                  let buffer = device.makeBuffer(bytes: address, length: bytes.count, options: .storageModeShared) else {
                throw CurveFlow.Failure.resource
            }
            buffer.label = label
            return buffer
        }
    }
}
