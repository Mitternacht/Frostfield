import Foundation
import simd

/// Raster-only camera-relative hierarchy composition. Keep this separate from
/// the normal graph traversal so a Float frame does not allocate these matrices.
struct HighPrecisionModelView {
    struct DrawMatrices {
        let clip: simd_float4x4
        let normal: simd_float3x3
    }

    struct LightPosition {
        let rootOrigin: SIMD3<Double>
        let offsetFromRoot: SIMD3<Double>

        func relative(to origin: SIMD3<Double>) -> SIMD3<Double> {
            rootOrigin - origin + offsetFromRoot
        }
    }

    let relativeWorldTransforms: [UUID: simd_double4x4]
    let lightPositions: [UUID: LightPosition]
    private let viewFromRelativeWorld: simd_double4x4
    private let clipFromView: simd_double4x4

    init(
        graph: SceneGraph, origin: SIMD3<Double>, viewFromRelativeWorld: simd_double4x4,
        projection: simd_float4x4, excluding: Set<UUID>
    ) {
        self.viewFromRelativeWorld = viewFromRelativeWorld
        clipFromView = simd_double4x4(
            SIMD4<Double>(projection[0]), SIMD4<Double>(projection[1]),
            SIMD4<Double>(projection[2]), SIMD4<Double>(projection[3])
        )
        var pending = graph.roots.reversed().map {
            (node: $0, parent: simd_double4x4?.none, root: SIMD3<Double>($0.transform.translation))
        }
        var worlds: [UUID: simd_double4x4] = [:]
        var lights: [UUID: LightPosition] = [:]
        while let (node, parent, root) = pending.popLast() {
            guard !excluding.contains(node.id), worlds[node.id] == nil else { continue }
            let rootRelative: simd_double4x4
            if node.isStatic, let frozen = node.staticWorldMatrix {
                // Keep the frozen world rotation and express its translation relative to this root.
                let columns = frozen.columns
                var relative = simd_double4x4(
                    SIMD4<Double>(columns.0), SIMD4<Double>(columns.1),
                    SIMD4<Double>(columns.2), SIMD4<Double>(columns.3)
                )
                relative[3] -= SIMD4(root, 0)
                rootRelative = relative
            } else {
                let local = node.transform.doubleMatrix(around: node.pivot, relativeTo: parent == nil ? root : .zero)
                rootRelative = node.resolvedDoubleWorldTransform(relativeTo: parent, local: local)
            }
            var frameRelative = rootRelative
            frameRelative[3] += SIMD4(root - origin, 0)
            worlds[node.id] = frameRelative
            switch node.light?.kind {
            case .point?, .spot?, .rectangle?:
                let offset = rootRelative[3]
                lights[node.id] = LightPosition(rootOrigin: root, offsetFromRoot: SIMD3(offset.x, offset.y, offset.z))
            default:
                break
            }
            for child in node.children.reversed() { pending.append((node: child, parent: rootRelative, root: root)) }
        }
        relativeWorldTransforms = worlds
        lightPositions = lights
    }

    func matrices(for id: UUID) -> DrawMatrices? {
        guard let world = relativeWorldTransforms[id] else { return nil }
        // Raster lighting uses world normals. Reconstruct their inverse transpose
        // in Double without the origin, and retain the existing world-space ABI.
        let normal = simd_transpose(simd_inverse(simd_double3x3(
            SIMD3(world[0].x, world[0].y, world[0].z),
            SIMD3(world[1].x, world[1].y, world[1].z),
            SIMD3(world[2].x, world[2].y, world[2].z)
        )))
        return DrawMatrices(clip: clipMatrix(for: world), normal: simd_float3x3(
            SIMD3<Float>(normal[0]), SIMD3<Float>(normal[1]), SIMD3<Float>(normal[2])
        ))
    }

    func clipMatrix(for relativeWorld: simd_double4x4) -> simd_float4x4 {
        Self.packed(clipFromView * (viewFromRelativeWorld * relativeWorld))
    }

    static func packed(_ matrix: simd_double4x4) -> simd_float4x4 {
        simd_float4x4(
            SIMD4<Float>(matrix[0]), SIMD4<Float>(matrix[1]),
            SIMD4<Float>(matrix[2]), SIMD4<Float>(matrix[3])
        )
    }
}

extension SceneTransform {
    func doubleMatrix(around pivot: SIMD3<Float>? = nil, relativeTo origin: SIMD3<Double> = .zero) -> simd_double4x4 {
        var result = simd_double4x4(simd_quatd(vector: SIMD4<Double>(rotation.vector)))
        result[0] *= Double(scale.x)
        result[1] *= Double(scale.y)
        result[2] *= Double(scale.z)
        // Subtract the common origin before adding a potentially much smaller pivot.
        result[3] = SIMD4(SIMD3<Double>(translation) - origin, 1)
        if let pivot {
            let point = SIMD4(SIMD3<Double>(pivot), 0)
            result[3] += point - result * point
        }
        return result
    }
}
