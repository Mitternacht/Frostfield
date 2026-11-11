import Foundation
import Metal
import simd

extension PathTracer {
    /// Fog, haze or smoke in a box or a closed mesh region outside optical volumes. Camera and
    /// scattered rays sample free paths per color channel with Henyey-Greenstein scattering.
    /// They use shadow rays to sample area, analytic, and environment lights, so occluders
    /// cast visible light shafts. Fog attenuates both shadow and escaping rays. Each
    /// scattering event in the fog counts as one bounce.
    ///
    /// Inside a transmissive or subsurface volume the volume's own medium replaces the fog.
    public struct Fog: Equatable, Sendable {
        /// An immutable scalar field with one constant density per voxel. Values from
        /// zero to one multiply each channel's extinction. The first voxel touches the
        /// fog box's minimum corner. X varies fastest, then Y, then Z. A grid uses
        /// four bytes per voxel, up to 8 MiB, and can be shared by tracers on its device.
        public final class DensityGrid: Equatable, @unchecked Sendable {
            public static let maximumResolution = 128
            public let resolution: SIMD3<Int>
            public var device: any MTLDevice { buffer.device }
            // The buffer is written once during initialization and read only by the GPU.
            let buffer: any MTLBuffer

            public init(device: any MTLDevice, resolution: SIMD3<Int>, densities: [Float]) throws {
                guard (0..<3).allSatisfy({ (1...Self.maximumResolution).contains(resolution[$0]) }),
                      densities.count == resolution.x * resolution.y * resolution.z,
                      densities.allSatisfy({ (0...1).contains($0) }) else {
                    throw Failure.invalidFogDensityGrid
                }
                guard let buffer = densities.withUnsafeBytes({ bytes in
                    device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
                }) else { throw Failure.fogDensityAllocation }
                buffer.label = "Frostfield fog density"
                self.resolution = resolution
                self.buffer = buffer
            }

            public static func == (lhs: DensityGrid, rhs: DensityGrid) -> Bool { lhs === rhs }
        }
        /// An immutable, invisible boundary shared by fog snapshots on this Metal device.
        /// Use a watertight triangle mesh whose outer shells wind outward. Concave shapes
        /// and disjoint shells are supported. Interior cavity shells wind toward the empty
        /// cavity. Shells must not intersect or touch.
        ///
        /// Vertices closer than a millionth of the mesh's largest dimension weld together,
        /// so duplicated face corners and exporter rounding close the surface. Initialization
        /// rejects open or non-manifold surfaces, edges used with mixed orientation, degenerate
        /// faces, shells with no volume, inverted shells and shells nested with the same
        /// orientation. It checks every shell's winding number at one point and does not
        /// prove the absence of every self-intersection. It accepts up to 1,024 shells, and
        /// shell count times triangle count may not exceed 40 million.
        public final class MeshBoundary: Equatable, @unchecked Sendable {
            /// Vertices closer than this fraction of the largest dimension weld together.
            public static let weldFraction: Float = 1e-6
            public static let maximumShells = 1024
            /// Boundary crossings the shader tracks along one ray in one region. After that
            /// many, the region stops contributing along the ray and the rest of the ray
            /// counts as outside it. Faces that share an edge or vertex with the ray count once.
            public static let maximumRayCrossings = 256

            public var device: any MTLDevice { buffer.device }
            /// The local-space box enclosing the welded mesh.
            public let bounds: AxisAlignedBox
            /// Triangles after welding.
            public let triangleCount: Int
            /// Disconnected closed surfaces: outer shells plus cavity shells.
            public let shellCount: Int
            // Privately built once and accessed only by the GPU afterward.
            let buffer: any MTLBuffer

            public init(device: any MTLDevice, geometry: MeshGeometry) throws {
                let referenced = Set(geometry.indices)
                let positions = referenced.map { geometry.vertices[Int($0)].position }
                guard let box = try? AxisAlignedBox(enclosing: positions),
                      simd_reduce_max(simd_max(abs(box.minimum), abs(box.maximum))) <= 1e12 else {
                    throw Failure.invalidFogBoundary
                }
                let largest = simd_reduce_max(box.size)
                guard largest > 0, let welded = try? geometry.welded(
                    tolerance: .position(largest * Self.weldFraction)
                ) else { throw Failure.invalidFogBoundary }
                let bvh = try MeshBVH(geometry: welded)
                let bounds = bvh.bounds
                guard all(bounds.size .> .zero) else { throw Failure.invalidFogBoundary }
                let shells = try Self.validatedShellCount(geometry: welded, center: SIMD3<Double>(bounds.center))
                let snapshot = try bvh.makeShaderBuffer(device: device)
                self.bounds = bounds
                triangleCount = snapshot.triangleCount
                shellCount = shells
                buffer = snapshot.buffer
            }

            /// Checks closure and orientation, then classifies every shell with a winding probe.
            static func validatedShellCount(geometry: MeshGeometry, center: SIMD3<Double>) throws -> Int {
                let indices = geometry.indices
                let triangleCount = indices.count / 3
                let points = geometry.vertices.map { SIMD3<Double>($0.position) - center }
                guard triangleCount >= 4 else { throw Failure.invalidFogBoundary }

                // Every directed edge occurs once and has its reverse: a closed, orientable
                // surface in which neighboring faces agree on the winding.
                var edges = Set<UInt64>(minimumCapacity: indices.count)
                func key(_ from: UInt32, _ to: UInt32) -> UInt64 { UInt64(from) << 32 | UInt64(to) }
                var parent = Array(0..<geometry.vertices.count)
                func root(_ vertex: Int) -> Int {
                    var vertex = vertex
                    while parent[vertex] != vertex {
                        parent[vertex] = parent[parent[vertex]]
                        vertex = parent[vertex]
                    }
                    return vertex
                }
                var areas: [Double] = []
                areas.reserveCapacity(triangleCount)
                for start in stride(from: 0, to: indices.count, by: 3) {
                    let a = indices[start], b = indices[start + 1], c = indices[start + 2]
                    let area = simd_length(simd_cross(points[Int(b)] - points[Int(a)], points[Int(c)] - points[Int(a)])) / 2
                    guard area > 0,
                          edges.insert(key(a, b)).inserted, edges.insert(key(b, c)).inserted,
                          edges.insert(key(c, a)).inserted else { throw Failure.invalidFogBoundary }
                    areas.append(area)
                    let anchor = root(Int(a))
                    let second = root(Int(b))
                    parent[second] = anchor
                    let third = root(Int(c))
                    parent[third] = anchor
                }
                guard edges.allSatisfy({ edges.contains(key(UInt32(truncatingIfNeeded: $0), UInt32($0 >> 32))) }) else {
                    throw Failure.invalidFogBoundary
                }

                var shellOf: [Int: Int] = [:]
                var triangleShell = [Int](repeating: 0, count: triangleCount)
                for triangle in 0..<triangleCount {
                    let representative = root(Int(indices[triangle * 3]))
                    let next = shellOf.count
                    triangleShell[triangle] = shellOf[representative] ?? next
                    shellOf[representative] = triangleShell[triangle]
                }
                let shells = shellOf.count
                guard shells <= Self.maximumShells, shells * triangleCount <= 40_000_000 else {
                    throw Failure.invalidFogBoundary
                }

                var volumes = [Double](repeating: 0, count: shells)
                var probe = [Int](repeating: 0, count: shells)
                var probeArea = [Double](repeating: 0, count: shells)
                for triangle in 0..<triangleCount {
                    let shell = triangleShell[triangle]
                    let a = points[Int(indices[triangle * 3])]
                    let b = points[Int(indices[triangle * 3 + 1])]
                    let c = points[Int(indices[triangle * 3 + 2])]
                    volumes[shell] += simd_dot(a, simd_cross(b, c)) / 6
                    if areas[triangle] > probeArea[shell] {
                        probeArea[shell] = areas[triangle]
                        probe[shell] = triangle
                    }
                }
                // A flat, double-sided sheet encloses no volume. Compare each shell with its
                // largest face so a thin slab or a small shell in a large mesh still passes.
                for shell in 0..<shells {
                    let flat = 1e-9 * probeArea[shell] * probeArea[shell].squareRoot()
                    guard volumes[shell].isFinite, abs(volumes[shell]) > flat else { throw Failure.invalidFogBoundary }
                }
                if shells == 1 {
                    guard volumes[0] > 0 else { throw Failure.invalidFogBoundary }
                    return 1
                }

                // Just outside a shell, the other shells' winding must be zero for an outward
                // shell and one for a cavity. Anything else is an inverted shell, a same-orientation
                // nest, or a shell that intersects another.
                for shell in 0..<shells {
                    let first = probe[shell] * 3
                    let centroid = (points[Int(indices[first])] + points[Int(indices[first + 1])]
                        + points[Int(indices[first + 2])]) / 3
                    var winding = 0.0
                    for triangle in 0..<triangleCount where triangleShell[triangle] != shell {
                        winding += Self.solidAngle(
                            points[Int(indices[triangle * 3])] - centroid,
                            points[Int(indices[triangle * 3 + 1])] - centroid,
                            points[Int(indices[triangle * 3 + 2])] - centroid
                        )
                    }
                    winding /= 4 * .pi
                    guard winding.isFinite, abs(winding - (volumes[shell] > 0 ? 0 : 1)) < 0.1 else {
                        throw Failure.invalidFogBoundary
                    }
                }
                return shells
            }

            /// The signed solid angle a counterclockwise triangle subtends at the origin
            /// (Van Oosterom and Strackee 1983).
            private static func solidAngle(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>) -> Double {
                let la = simd_length(a), lb = simd_length(b), lc = simd_length(c)
                let denominator = la * lb * lc + simd_dot(a, b) * lc + simd_dot(a, c) * lb + simd_dot(b, c) * la
                return 2 * atan2(simd_dot(a, simd_cross(b, c)), denominator)
            }

            public static func == (lhs: MeshBoundary, rhs: MeshBoundary) -> Bool { lhs === rhs }
        }

        /// Extinction per world unit for each linear RGB channel: absorption plus out-scattering.
        public let extinction: SIMD3<Float>
        /// The single-scattering albedo per channel, σs / σt, from zero to one.
        public let albedo: SIMD3<Float>
        /// The Henyey-Greenstein g, greater than -1 and less than 1. Positive scatters forward.
        public let anisotropy: Float
        /// World-space bounds of the fog region.
        public let minimum: SIMD3<Float>
        public let maximum: SIMD3<Float>
        /// Optional per-voxel extinction multipliers. Nil is homogeneous fog.
        public let densityGrid: DensityGrid?
        /// A closed mesh shape, or nil for the axis-aligned world box.
        public let meshBoundary: MeshBoundary?
        /// Placement of the mesh and its optional grid, or nil for a world box.
        public let meshTransform: SceneTransform?
        /// The inverse placement, built in double precision. Identity for a world box.
        package let meshLocalFromWorld: simd_float4x4

        /// The smallest scale magnitude a mesh placement accepts.
        public static let minimumMeshScale: Float = 1e-6
        /// The largest ratio between the greatest and the smallest scale magnitude.
        public static let maximumMeshScaleRatio: Float = 1e6

        /// Extinction from 0 to 10⁶ per world unit, albedo components from zero to one, and a
        /// finite box with `minimum` below `maximum` on every axis. Throws `Failure.invalidFog`.
        public init(
            extinction: SIMD3<Float>, albedo: SIMD3<Float> = SIMD3(repeating: 1), anisotropy: Float = 0,
            minimum: SIMD3<Float>, maximum: SIMD3<Float>, densityGrid: DensityGrid? = nil
        ) throws {
            try self.init(
                extinction: extinction, albedo: albedo, anisotropy: anisotropy,
                minimum: minimum, maximum: maximum, densityGrid: densityGrid,
                meshBoundary: nil, meshTransform: nil, meshLocalFromWorld: matrix_identity_float4x4
            )
        }

        /// Fills the interior of a closed boundary. The optional grid spans the
        /// boundary's local bounds and moves with the mesh. Extinction stays per world unit,
        /// so non-uniform scale stretches the shape without changing the optical depth per
        /// world distance. A mirrored scale is allowed. Throws `Failure.invalidFogTransform`
        /// for a non-finite placement, a rotation that is not a unit quaternion, a scale
        /// magnitude below `minimumMeshScale` or a scale ratio above `maximumMeshScaleRatio`.
        public init(
            extinction: SIMD3<Float>, albedo: SIMD3<Float> = SIMD3(repeating: 1), anisotropy: Float = 0,
            meshBoundary: MeshBoundary, transform: SceneTransform = .init(), densityGrid: DensityGrid? = nil
        ) throws {
            guard let placement = Self.placement(of: transform),
                  let bounds = try? meshBoundary.bounds.transformed(by: placement.world) else {
                throw Failure.invalidFogTransform
            }
            try self.init(
                extinction: extinction, albedo: albedo, anisotropy: anisotropy,
                minimum: bounds.minimum, maximum: bounds.maximum, densityGrid: densityGrid,
                meshBoundary: meshBoundary, meshTransform: transform, meshLocalFromWorld: placement.local
            )
        }

        private init(
            extinction: SIMD3<Float>, albedo: SIMD3<Float>, anisotropy: Float,
            minimum: SIMD3<Float>, maximum: SIMD3<Float>, densityGrid: DensityGrid?,
            meshBoundary: MeshBoundary?, meshTransform: SceneTransform?, meshLocalFromWorld: simd_float4x4
        ) throws {
            guard (0..<3).allSatisfy({ (0...1e6).contains(extinction[$0]) && (0...1).contains(albedo[$0]) }),
                  anisotropy > -1, anisotropy < 1,
                  (0..<3).allSatisfy({ minimum[$0].isFinite && maximum[$0].isFinite && minimum[$0] < maximum[$0] }),
                  simd_reduce_max(simd_max(abs(minimum), abs(maximum))) <= 1e12 else {
                throw Failure.invalidFog
            }
            self.extinction = extinction
            self.albedo = albedo
            self.anisotropy = anisotropy
            self.minimum = minimum
            self.maximum = maximum
            self.densityGrid = densityGrid
            self.meshBoundary = meshBoundary
            self.meshTransform = meshTransform
            self.meshLocalFromWorld = meshLocalFromWorld
        }

        /// The world matrix and its inverse for a scale, rotation and translation, or nil when
        /// the placement is not finite or not safely invertible. The inverse comes from the
        /// factors in double precision rather than from inverting a rounded float matrix, so
        /// a near-singular scale is rejected instead of producing garbage local coordinates.
        static func placement(of transform: SceneTransform) -> (world: simd_double4x4, local: simd_float4x4)? {
            let q = SIMD4<Double>(transform.rotation.vector)
            let scale = SIMD3<Double>(transform.scale)
            let translation = SIMD3<Double>(transform.translation)
            guard (0..<4).allSatisfy({ q[$0].isFinite }),
                  (0..<3).allSatisfy({ scale[$0].isFinite && translation[$0].isFinite }),
                  abs(simd_length_squared(q) - 1) <= 1e-4 else { return nil }
            let magnitude = abs(scale)
            let low = simd_reduce_min(magnitude), high = simd_reduce_max(magnitude)
            guard low >= Double(minimumMeshScale), high <= low * Double(maximumMeshScaleRatio) else { return nil }
            let unit = q / simd_length(q)
            let (x, y, z, w) = (unit.x, unit.y, unit.z, unit.w)
            let r0 = SIMD3(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
            let r1 = SIMD3(2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w))
            let r2 = SIMD3(2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y))
            let world = simd_double4x4(columns: (
                SIMD4(r0 * scale.x, 0), SIMD4(r1 * scale.y, 0), SIMD4(r2 * scale.z, 0), SIMD4(translation, 1)
            ))
            // The inverse linear part is the transposed rotation with each row divided by its scale.
            let c0 = SIMD3(r0.x / scale.x, r1.x / scale.y, r2.x / scale.z)
            let c1 = SIMD3(r0.y / scale.x, r1.y / scale.y, r2.y / scale.z)
            let c2 = SIMD3(r0.z / scale.x, r1.z / scale.y, r2.z / scale.z)
            let shift = -(c0 * translation.x + c1 * translation.y + c2 * translation.z)
            let columns = [SIMD4(c0, 0), SIMD4(c1, 0), SIMD4(c2, 0), SIMD4(shift, 1)].map { SIMD4<Float>($0) }
            guard columns.allSatisfy({ column in (0..<4).allSatisfy { column[$0].isFinite } }) else { return nil }
            return (world, simd_float4x4(columns))
        }

        /// Gray fog: `density` is the extinction per world unit and `color` the single-scattering albedo.
        public init(
            density: Float, color: SIMD3<Float> = SIMD3(repeating: 1), anisotropy: Float = 0,
            minimum: SIMD3<Float>, maximum: SIMD3<Float>, densityGrid: DensityGrid? = nil
        ) throws {
            try self.init(
                extinction: SIMD3(repeating: density), albedo: color, anisotropy: anisotropy,
                minimum: minimum, maximum: maximum, densityGrid: densityGrid
            )
        }

        /// The unscattered, unabsorbed fraction over `distance` world units at unit density.
        /// For a grid, pass the integrated density along the segment in world units.
        public func transmittance(distance: Float) -> SIMD3<Float> {
            SIMD3((0..<3).map { channel in
                let coefficient = extinction[channel]
                return coefficient == 0 ? 1 : exp(-coefficient * max(distance, 0))
            })
        }
    }

    // Keep in lockstep with PTFogVolume in PathTracingShading.h.
    package struct MediaUniforms {
        /// Fog extinction (rgb).
        var fogExtinction: SIMD4<Float>
        /// Fog single-scattering albedo (rgb) and anisotropy (w).
        var fogAlbedo: SIMD4<Float>
        var fogMinimum: SIMD4<Float>
        var fogMaximum: SIMD4<Float>
        /// Voxel dimensions (xyz) and active record count (w in the first record).
        var fogResolution: SIMD4<UInt32> = .zero
        var fogDensityAddress: UInt64 = 0
        var fogBoundaryAddress: UInt64 = 0
        var fogLocalFromWorld = matrix_identity_float4x4

        static func pack(_ fogVolumes: [Fog]) -> [Self] {
            var active = fogVolumes.filter { simd_reduce_max($0.extinction) > 0 }.map { Self(fog: $0) }
            let count = active.count
            if active.isEmpty { active.append(Self(fog: nil)) }
            active[0].fogResolution.w = UInt32(count)
            return active
        }

        init(fog: Fog?) {
            if let fog, simd_reduce_max(fog.extinction) > 0 {
                fogExtinction = SIMD4(fog.extinction, 0)
                fogAlbedo = SIMD4(fog.albedo, fog.anisotropy)
                fogMinimum = SIMD4(fog.meshBoundary?.bounds.minimum ?? fog.minimum, 0)
                fogMaximum = SIMD4(fog.meshBoundary?.bounds.maximum ?? fog.maximum, 0)
                if let boundary = fog.meshBoundary {
                    fogBoundaryAddress = boundary.buffer.gpuAddress
                    fogLocalFromWorld = fog.meshLocalFromWorld
                }
                if let grid = fog.densityGrid {
                    let resolution = grid.resolution
                    fogResolution = SIMD4(UInt32(resolution.x), UInt32(resolution.y), UInt32(resolution.z), 0)
                    fogDensityAddress = grid.buffer.gpuAddress
                }
            } else {
                fogExtinction = .zero
                fogAlbedo = .zero
                fogMinimum = .zero
                fogMaximum = .zero
            }
        }
    }
}
