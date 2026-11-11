import Foundation
import Metal
import simd

public extension PathTracer {
    /// Controls the software BVH heatmap.
    struct BVHHeatmapOptions: Equatable, Sendable {
        public var displayTopLevel: Bool
        public var displayBottomLevel: Bool
        /// Count only boxes reached by the ray up to the first accepted mesh surface.
        /// Native face culling and stochastic or masked material coverage apply.
        public var stopAtSurface: Bool
        /// Count that maps to the hottest color, from 1 to 65,536.
        public var saturationCount: Int
        /// Maximum number of node tests per ray, including the surface search.
        /// Exhausted rays are magenta. The limit is from 1 to 65,536.
        public var maximumNodeVisits: Int

        public init(
            displayTopLevel: Bool = true, displayBottomLevel: Bool = true, stopAtSurface: Bool = false,
            saturationCount: Int = 64, maximumNodeVisits: Int = 16_384
        ) {
            self.displayTopLevel = displayTopLevel
            self.displayBottomLevel = displayBottomLevel
            self.stopAtSurface = stopAtSurface
            self.saturationCount = saturationCount
            self.maximumNodeVisits = maximumNodeVisits
        }
    }

    /// Exact completed sample counts, in top-left row-major image order.
    /// Unsupplied image pixels are zero. Each traced window uses uniform sampling.
    struct SampleDensity: Sendable {
        public let width: Int
        public let height: Int
        public let counts: [UInt32]

        /// Writes raw counts to an R32Float texture. A display range maps its
        /// endpoints to zero and one, clamping values outside the range.
        /// The red channel can be shown with `TextureDebugPass`.
        public func makeTexture(
            device: any MTLDevice, displayRange: ClosedRange<Int>? = nil
        ) throws -> any MTLTexture {
            if let displayRange {
                guard displayRange.lowerBound >= 0, displayRange.upperBound <= PathTracer.maximumSamples,
                      displayRange.lowerBound < displayRange.upperBound else { throw Failure.displayRange }
            }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r32Float, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = .shaderRead
            guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.textureAllocation }
            let values: [Float]
            if let displayRange {
                let lower = Float(displayRange.lowerBound)
                let span = Float(displayRange.upperBound - displayRange.lowerBound)
                values = counts.map { min(max((Float($0) - lower) / span, 0), 1) }
            } else {
                values = counts.map(Float.init)
            }
            values.withUnsafeBytes {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                    withBytes: $0.baseAddress!, bytesPerRow: width * MemoryLayout<Float>.stride
                )
            }
            texture.label = "Frostfield path sample density"
            return texture
        }

        public enum Failure: LocalizedError, Equatable {
            case imageDimensions
            case history
            case overlappingWindows
            case displayRange
            case textureAllocation

            public var errorDescription: String? {
                switch self {
                case .imageDimensions: "Sample density requires windows of one image within the accumulation size limit."
                case .history: "Sample density requires the latest completed history of each window from this tracer."
                case .overlappingWindows: "Sample density windows cannot overlap."
                case .displayRange: "Sample density display needs an increasing range within the tracer's sample limit."
                case .textureAllocation: "Metal could not allocate the sample density texture."
                }
            }
        }
    }
}

public extension PathTracingScene {
    /// A depth-limited snapshot of the actual nodes in one software path tracing BVH.
    struct BVHBounds: Sendable {
        public enum Level: Equatable, Sendable {
            /// World-space bounds over visible path-traced mesh instances.
            case topLevel
            /// Geometry-space bounds for one mesh occurrence. Use nil for a non-instanced node.
            case bottomLevel(nodeID: UUID, instanceIndex: Int? = nil)
        }

        public struct Node: Equatable, Sendable {
            /// Index within the selected tree.
            public let index: Int
            /// Distance in edges from the root.
            public let depth: Int
            public let bounds: AxisAlignedBox
            /// The count of instances or triangles in a leaf. Interior nodes have no count.
            public let leafPrimitiveCount: Int?
        }

        public let revision: UInt64
        public let level: Level
        /// Identity for the top level, or the selected instance's full affine transform.
        public let worldFromBounds: simd_float4x4
        /// Nodes in depth-first order, visiting the left child first.
        public let nodes: [Node]

        /// Edges in world space, suitable for a scene-root node from `DebugHelpers`.
        public func lineGeometry(color: SIMD4<Float> = SIMD4(1, 0.6, 0, 1)) throws -> LineGeometry {
            let positions = nodes.flatMap { node in
                LineGeometry.boxEdgePositions(node.bounds).map { point in
                    let world = worldFromBounds * SIMD4(point, 1)
                    return SIMD3(world.x, world.y, world.z)
                }
            }
            return try LineGeometry(
                positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
            )
        }

        public enum Failure: LocalizedError, Equatable {
            case hardwareAcceleration
            case depth
            case nodeLimit
            case missingInstance

            public var errorDescription: String? {
                switch self {
                case .hardwareAcceleration: "Metal ray tracing does not expose its BVH bounds."
                case .depth: "BVH debug depth must fall between zero and the last supported depth."
                case .nodeLimit: "The BVH debug depth includes more than 4,096 nodes."
                case .missingInstance: "The requested mesh occurrence is absent from the traced scene."
                }
            }
        }
    }

    /// Reads the same software tree used by path rays, without changing the scene.
    /// Root depth is zero. At most 4,096 nodes can be returned per snapshot.
    func softwareBVHBounds(level: BVHBounds.Level = .topLevel, maximumDepth: Int = 3) throws -> BVHBounds {
        guard accelerationKind == .software else { throw BVHBounds.Failure.hardwareAcceleration }
        guard (0..<Self.maximumBVHDepth).contains(maximumDepth) else { throw BVHBounds.Failure.depth }

        func snapshot(_ tree: UnsafeBufferPointer<BVHNode>, worldFromBounds: simd_float4x4) throws -> BVHBounds {
            var result: [BVHBounds.Node] = []
            var stack = [(index: 0, depth: 0)]
            while let entry = stack.popLast() {
                guard result.count < 4_096 else { throw BVHBounds.Failure.nodeLimit }
                let node = tree[entry.index]
                let count = Int(node.meta.w)
                result.append(BVHBounds.Node(
                    index: entry.index, depth: entry.depth,
                    bounds: AxisAlignedBox(
                        validatedMinimum: SIMD3(node.minimum.x, node.minimum.y, node.minimum.z),
                        maximum: SIMD3(node.maximum.x, node.maximum.y, node.maximum.z)
                    ),
                    leafPrimitiveCount: count > 0 ? count : nil
                ))
                if count == 0 && entry.depth < maximumDepth {
                    stack.append((Int(node.meta.y), entry.depth + 1))
                    stack.append((Int(node.meta.x), entry.depth + 1))
                }
            }
            return BVHBounds(revision: revision, level: level, worldFromBounds: worldFromBounds, nodes: result)
        }

        switch level {
        case .topLevel:
            return try acceleration.topLevelNodes.withUnsafeBufferPointer {
                try snapshot($0, worldFromBounds: matrix_identity_float4x4)
            }
        case .bottomLevel(let nodeID, let instanceIndex):
            guard let instance = acceleration.instances.first(where: {
                $0.nodeID == nodeID && $0.instanceIndex == instanceIndex
            }), let geometry = acceleration.bottomLevels[instance.geometry], let buffer = geometry.nodes else {
                throw BVHBounds.Failure.missingInstance
            }
            // Software builders create immutable, shared node buffers from BVHNode records.
            return try withExtendedLifetime(buffer) {
                try snapshot(
                    UnsafeBufferPointer(
                        start: buffer.contents().assumingMemoryBound(to: BVHNode.self), count: geometry.nodeCount
                    ),
                    worldFromBounds: instance.world
                )
            }
        }
    }
}
