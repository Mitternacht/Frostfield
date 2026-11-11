import Metal
import simd

extension PathTracingScene {
    struct EmissiveTriangleTable {
        let buffer: any MTLBuffer
        let count: Int
    }

    // In traversal order, including the software BLAS's reordered primitives. Binary
    // searches use the same rows by either (instance, primitive) or cumulative mass.
    private struct EmissiveTriangle {
        var instance: UInt32
        var primitive: UInt32
        var cumulative: Float
        var area: Float
    }

    static func makeEmissiveTriangles(
        instances: [Instance], bottomLevels: [GeometryKey: BottomLevel], materials: any MTLBuffer,
        device: any MTLDevice
    ) throws -> EmissiveTriangleTable? {
        let limit = 262_144
        // A hit on the current medium's boundary does not emit during a closed walk.
        let walkBoundaryFlags = lobeScattering | lobeDiffuseBoundary
        let records = materials.contents().assumingMemoryBound(to: Material.self)
        var weighted: [(triangle: EmissiveTriangle, cumulative: Double)] = []
        var total = 0.0
        for (instanceIndex, instance) in instances.enumerated() {
            if weighted.count == limit { break }
            let material = records[Int(instance.material)]
            guard instance.surfaceGraph == 0, material.emission.w == opaqueCoverage,
                  material.extensions.options.x <= 0.5,
                  material.lobes.flags.x & walkBoundaryFlags == 0,
                  let level = bottomLevels[instance.geometry] else { continue }
            let emission = SIMD3<Double>(
                Double(material.emission.x), Double(material.emission.y), Double(material.emission.z)
            )
            // Use authored intensity to choose triangles. Their sampled UV
            // radiance is evaluated separately with the material's actual map.
            let luminance = simd_dot(simd_max(emission, .zero), SIMD3(0.2126, 0.7152, 0.0722))
            guard luminance.isFinite, luminance > 0 else { continue }
            let world = instance.world
            let linear = simd_double3x3(
                position(world.columns.0), position(world.columns.1), position(world.columns.2)
            )
            let vertices = level.vertices.contents().assumingMemoryBound(to: Vertex.self)
            let indices = level.indices.contents().assumingMemoryBound(to: UInt32.self)
            for primitive in 0..<level.triangleCount {
                let corners = indices + 3 * primitive
                let a = position(vertices[Int(corners[0])].position)
                let ab = linear * (position(vertices[Int(corners[1])].position) - a)
                let ac = linear * (position(vertices[Int(corners[2])].position) - a)
                let area = Float(simd_length(simd_cross(ab, ac)) * 0.5)
                guard area.isFinite, area >= Float.leastNormalMagnitude else { continue }
                total += Double(area) * luminance
                weighted.append((EmissiveTriangle(
                    instance: UInt32(instanceIndex), primitive: UInt32(primitive), cumulative: 0, area: area
                ), total))
                if weighted.count == limit { break }
            }
        }
        guard !weighted.isEmpty else { return nil }
        var rows: [EmissiveTriangle] = []
        rows.reserveCapacity(weighted.count)
        // Both sampler modes choose on a 24-bit grid. Quantizing the thresholds
        // on that grid makes their differences the exact sampled probabilities.
        let lattice = Double(1 << 24)
        var previous: Float = 0
        for (var triangle, cumulative) in weighted {
            triangle.cumulative = Float((cumulative / total * lattice).rounded() / lattice)
            // A dim triangle can have zero mass after quantization. Its path hit
            // keeps full weight because the explicit strategy cannot select it.
            guard triangle.cumulative > previous else { continue }
            rows.append(triangle)
            previous = triangle.cumulative
        }
        let buffer = try makeBuffer(rows, device: device, label: "path emissive triangles")
        return EmissiveTriangleTable(buffer: buffer, count: rows.count)
    }

    private static func position(_ value: SIMD4<Float>) -> SIMD3<Double> {
        SIMD3(Double(value.x), Double(value.y), Double(value.z))
    }
}
