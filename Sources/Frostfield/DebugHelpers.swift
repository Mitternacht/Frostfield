import Foundation
import Metal
import simd

// Geometry for the Three.js helper objects. Each returns separate segments with
// linear vertex colors; colors that Three.js gives as sRGB hex values are
// converted to linear, as its color management does.
public extension LineGeometry {
    /// Linear RGBA for the five CameraHelper line groups.
    struct CameraFrustumColors: Equatable, Sendable {
        public var frustum: SIMD4<Float>
        public var cone: SIMD4<Float>
        public var up: SIMD4<Float>
        public var target: SIMD4<Float>
        public var cross: SIMD4<Float>

        public init(
            frustum: SIMD4<Float> = LineGeometry.helperColor(0xffaa00),
            cone: SIMD4<Float> = LineGeometry.helperColor(0xff0000),
            up: SIMD4<Float> = LineGeometry.helperColor(0x00aaff),
            target: SIMD4<Float> = LineGeometry.helperColor(0xffffff),
            cross: SIMD4<Float> = LineGeometry.helperColor(0x333333)
        ) {
            self.frustum = frustum
            self.cone = cone
            self.up = up
            self.target = target
            self.cross = cross
        }
    }

    enum SkeletonConnections: Equatable, Sendable {
        /// Connects joints only when their immediate parent is also a joint.
        case directJointParent
        /// Bridges intermediate transform nodes to the nearest joint ancestor.
        case nearestJointAncestor
    }

    struct SkeletonColors: Equatable, Sendable {
        public var joint: SIMD4<Float>
        public var parent: SIMD4<Float>

        public init(joint: SIMD4<Float> = SIMD4(0, 0, 1, 1), parent: SIMD4<Float> = SIMD4(0, 1, 0, 1)) {
            self.joint = joint
            self.parent = parent
        }
    }

    struct AxesColors: Equatable, Sendable {
        public var x: SIMD4<Float>
        public var y: SIMD4<Float>
        public var z: SIMD4<Float>

        public init(x: SIMD4<Float>, y: SIMD4<Float>, z: SIMD4<Float>) {
            self.x = x
            self.y = y
            self.z = z
        }
    }

    /// Three.js `AxesHelper`: X, Y and Z from the origin, each fading from its
    /// primary color toward a lighter tip with the same vertex colors.
    static func axes(size: Float = 1) throws -> LineGeometry {
        try axisSegments(size: size, colors: [
            SIMD4(1, 0, 0, 1), SIMD4(1, 0.6, 0, 1), SIMD4(0, 1, 0, 1),
            SIMD4(0.6, 1, 0, 1), SIMD4(0, 0, 1, 1), SIMD4(0, 0.6, 1, 1),
        ])
    }

    /// Uses each authored linear color at both ends of its axis, like `setColors`.
    static func axes(size: Float = 1, colors: AxesColors) throws -> LineGeometry {
        try axisSegments(size: size, colors: [colors.x, colors.x, colors.y, colors.y, colors.z, colors.z])
    }

    private static func axisSegments(size: Float, colors: [SIMD4<Float>]) throws -> LineGeometry {
        try requirePositive(size, "axes size")
        return try LineGeometry(
            positions: [.zero, SIMD3(size, 0, 0), .zero, SIMD3(0, size, 0), .zero, SIMD3(0, 0, size)],
            colors: colors, topology: .segments
        )
    }

    /// Three.js `GridHelper`: `divisions + 1` lines along each of X and Z across a
    /// square of `size` in the XZ plane. With an even division count the two
    /// center lines use `centerColor`.
    static func grid(
        size: Float = 10, divisions: Int = 10,
        centerColor: SIMD4<Float> = LineGeometry.helperColor(0x444444), color: SIMD4<Float> = LineGeometry.helperColor(0x888888)
    ) throws -> LineGeometry {
        try requirePositive(size, "grid size")
        guard (1...10_000).contains(divisions) else { throw PrimitiveGeometryError.invalidHelperParameter("1 to 10,000 grid divisions") }
        let half = Double(size) / 2
        let step = Double(size) / Double(divisions)
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        for index in 0...divisions {
            let k = Float(-half + step * Double(index))
            let h = Float(half)
            positions += [SIMD3(-h, 0, k), SIMD3(h, 0, k), SIMD3(k, 0, -h), SIMD3(k, 0, h)]
            let lineColor = divisions.isMultiple(of: 2) && index == divisions / 2 ? centerColor : color
            colors += Array(repeating: lineColor, count: 4)
        }
        return try LineGeometry(positions: positions, colors: colors, topology: .segments)
    }

    /// Three.js `PolarGridHelper`: `sectors` radial lines (when more than one)
    /// and `rings` circles of `divisions` segments in the XZ plane, alternating
    /// two colors. Angles start on +Z and turn toward +X.
    static func polarGrid(
        radius: Float = 10, sectors: Int = 16, rings: Int = 8, divisions: Int = 64,
        color1: SIMD4<Float> = LineGeometry.helperColor(0x444444), color2: SIMD4<Float> = LineGeometry.helperColor(0x888888)
    ) throws -> LineGeometry {
        try requirePositive(radius, "polar grid radius")
        guard (0...4_096).contains(sectors), (1...4_096).contains(rings), (3...4_096).contains(divisions) else {
            throw PrimitiveGeometryError.invalidHelperParameter("0 to 4,096 sectors, 1 to 4,096 rings and 3 to 4,096 divisions")
        }
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        if sectors > 1 {
            for index in 0..<sectors {
                let angle = Double(index) / Double(sectors) * 2 * .pi
                positions += [.zero, SIMD3(Float(sin(angle)) * radius, 0, Float(cos(angle)) * radius)]
                colors += Array(repeating: index & 1 == 1 ? color1 : color2, count: 2)
            }
        }
        for ring in 0..<rings {
            let r = Double(radius) - Double(radius) / Double(rings) * Double(ring)
            let ringColor = ring & 1 == 1 ? color1 : color2
            for step in 0..<divisions {
                for offset in 0...1 {
                    let angle = Double(step + offset) / Double(divisions) * 2 * .pi
                    positions.append(SIMD3(Float(sin(angle) * r), 0, Float(cos(angle) * r)))
                    colors.append(ringColor)
                }
            }
        }
        return try LineGeometry(positions: positions, colors: colors, topology: .segments)
    }

    /// Three.js `Box3Helper` and `BoxHelper`: the twelve edges of a box.
    static func box(_ box: AxisAlignedBox, color: SIMD4<Float> = SIMD4(1, 1, 0, 1)) throws -> LineGeometry {
        let positions = boxEdgePositions(box)
        return try LineGeometry(
            positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
        )
    }

    internal static func boxEdgePositions(_ box: AxisAlignedBox) -> [SIMD3<Float>] {
        let lower = box.minimum
        let upper = box.maximum
        let corners = [
            SIMD3(upper.x, upper.y, upper.z), SIMD3(lower.x, upper.y, upper.z),
            SIMD3(lower.x, lower.y, upper.z), SIMD3(upper.x, lower.y, upper.z),
            SIMD3(upper.x, upper.y, lower.z), SIMD3(lower.x, upper.y, lower.z),
            SIMD3(lower.x, lower.y, lower.z), SIMD3(upper.x, lower.y, lower.z),
        ]
        let edges = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
        return edges.map { corners[$0] }
    }

    /// Three.js `BoxHelper`: the world-space box around a node and all of its
    /// descendants' meshes and primitives, with mesh bounds widened by instances.
    /// Skinned meshes use their bind-pose bounds unless the node has culling
    /// bounds. Square world-space points use their current camera-facing extent.
    /// Skips subtrees marked `excludesFromDebugBounds`. Add the result at the
    /// scene root. Returns nil when the subtree has no bounds or its node is absent.
    static func boundingBox(
        of nodeID: UUID, in graph: SceneGraph, color: SIMD4<Float> = SIMD4(1, 1, 0, 1)
    ) throws -> LineGeometry? {
        guard let bounds = try graph.subtreeWorldBounds(of: nodeID) else { return nil }
        return try box(bounds, color: color)
    }

    /// Three.js `CameraHelper`: the frustum of a Metal projection as 25 segments
    /// in Three.js order and colors: the near and far rectangles and the four
    /// sides, the cone from the camera to the near corners, the up triangle above
    /// the near plane, the center line from near to far, the line from the camera
    /// to the near center, and the near and far crosses.
    ///
    /// Pass a camera's projection to get geometry in its local space (attach it
    /// to the camera's transform), or a clip-from-world matrix for world space.
    /// The camera is the perspective center of projection. An orthographic
    /// projection has none, so its cone starts at the local origin, as in
    /// Three.js; that is the camera only for a projection in its local space.
    /// The far plane must be finite.
    static func cameraFrustum(projection: simd_float4x4, depthMode: MeshDepthMode = .forward) throws -> LineGeometry {
        try cameraFrustum(projection: projection, depthMode: depthMode, colors: .init())
    }

    /// Uses custom colors in the same segment order as the default camera helper.
    static func cameraFrustum(
        projection: simd_float4x4, depthMode: MeshDepthMode = .forward, colors: CameraFrustumColors
    ) throws -> LineGeometry {
        try makeCameraFrustum(projection: projection, depthMode: depthMode, colors: colors, cameraOrigin: nil)
    }

    /// A world-space helper for an explicit camera with a finite far plane.
    /// Pass `graphCamera` and `depthMode` from a glTF camera view or a framed
    /// off-axis view. A separate view matrix locates an orthographic camera's origin.
    static func cameraFrustum(
        camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward, colors: CameraFrustumColors = .init()
    ) throws -> LineGeometry {
        guard camera.far.isFinite else {
            throw PrimitiveGeometryError.invalidHelperParameter("the projection needs a finite far plane")
        }
        let local = try makeCameraFrustum(
            projection: camera.projectionMatrix, depthMode: depthMode, colors: colors, cameraOrigin: .zero
        )
        let cameraToWorld = simd_inverse(helperDouble(camera.viewMatrix))
        let positions = try local.positions.map { position -> SIMD3<Float> in
            let world = cameraToWorld * SIMD4<Double>(SIMD4(position, 1))
            guard world.w.isFinite, world.w != 0 else {
                throw PrimitiveGeometryError.invalidHelperParameter("the view must map the frustum to finite world points")
            }
            return SIMD3<Float>(SIMD3(world.x, world.y, world.z) / world.w)
        }
        return try LineGeometry(positions: positions, colors: local.colors, topology: .segments)
    }

    private static func makeCameraFrustum(
        projection: simd_float4x4, depthMode: MeshDepthMode, colors style: CameraFrustumColors, cameraOrigin: SIMD3<Float>?
    ) throws -> LineGeometry {
        let determinant = simd_determinant(helperDouble(projection))
        guard primitiveFinite(projection), determinant.isFinite, determinant != 0 else {
            throw PrimitiveGeometryError.invalidHelperParameter("the projection must be finite and invertible")
        }
        let inverse = simd_inverse(helperDouble(projection))
        let nearDepth: Double = depthMode == .reversed ? 1 : 0
        let farDepth: Double = depthMode == .reversed ? 0 : 1
        func point(_ x: Double, _ y: Double, far: Bool) throws -> SIMD3<Float> {
            let unprojected = inverse * SIMD4(x, y, far ? farDepth : nearDepth, 1)
            guard unprojected.w.isFinite, unprojected.w != 0 else {
                throw PrimitiveGeometryError.invalidHelperParameter("the projection needs a finite far plane")
            }
            let position = SIMD3<Float>(SIMD3(unprojected.x, unprojected.y, unprojected.z) / unprojected.w)
            guard primitiveFinite(position) else {
                throw PrimitiveGeometryError.invalidHelperParameter("the projection needs a finite far plane")
            }
            return position
        }
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        func add(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ color: SIMD4<Float>) {
            positions += [a, b]
            colors += [color, color]
        }
        // Three.js n1…n4 and f1…f4: (−1, −1), (1, −1), (−1, 1), (1, 1).
        let n = try [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].map { try point($0.0, $0.1, far: false) }
        let f = try [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].map { try point($0.0, $0.1, far: true) }
        let rectangle = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for (a, b) in rectangle { add(n[a], n[b], style.frustum) }
        for (a, b) in rectangle { add(f[a], f[b], style.frustum) }
        for index in 0..<4 { add(n[index], f[index], style.frustum) }
        let camera = cameraOrigin ?? Self.projectionCenter(projection) ?? .zero
        for index in 0..<4 { add(camera, n[index], style.cone) }
        let u1 = try point(0.7, 1.1, far: false)
        let u2 = try point(-0.7, 1.1, far: false)
        let u3 = try point(0, 2, far: false)
        add(u1, u2, style.up)
        add(u2, u3, style.up)
        add(u3, u1, style.up)
        let nearCenter = try point(0, 0, far: false)
        add(nearCenter, try point(0, 0, far: true), style.target)
        add(camera, nearCenter, style.cross)
        for far in [false, true] {
            add(try point(-1, 0, far: far), try point(1, 0, far: far), style.cross)
            add(try point(0, -1, far: far), try point(0, 1, far: far), style.cross)
        }
        return try LineGeometry(positions: positions, colors: colors, topology: .segments)
    }

    /// The frustum of a perspective camera in its local space; see ``cameraFrustum(projection:depthMode:)``.
    static func cameraFrustum(camera: PerspectiveCamera, aspectRatio: Float) throws -> LineGeometry {
        try cameraFrustum(projection: camera.projection(aspectRatio: aspectRatio))
    }

    /// Three.js light helpers in the light's local space, colored with the light
    /// color; attach the result to the light node's transform. Lights point
    /// along local −Z.
    ///
    /// - Directional (`DirectionalLightHelper`): a square of half-extent `size`
    ///   and a line of length `size` along the light direction.
    /// - Point (`PointLightHelper`): three orthogonal circles of radius `size`.
    /// - Spot (`SpotLightHelper`): the axis, four generators and the outer rim
    ///   at the range, or at `size` without one, plus the inner rim when the
    ///   inner angle is nonzero.
    /// - Rectangle (`RectAreaLightHelper`): the outline and a line of length
    ///   half the shorter side along the emission direction.
    static func light(_ light: PunctualLight, size: Float = 1) throws -> LineGeometry {
        try requirePositive(size, "light helper size")
        let color = SIMD4(light.color, 1)
        var positions: [SIMD3<Float>] = []
        func add(_ a: SIMD3<Float>, _ b: SIMD3<Float>) { positions += [a, b] }
        func circle(radius: Float, center: SIMD3<Float>, u: SIMD3<Float>, v: SIMD3<Float>) {
            let segments = 32
            for index in 0..<segments {
                let a = Double(index) / Double(segments) * 2 * .pi
                let b = Double(index + 1) / Double(segments) * 2 * .pi
                add(
                    center + radius * (Float(cos(a)) * u + Float(sin(a)) * v),
                    center + radius * (Float(cos(b)) * u + Float(sin(b)) * v)
                )
            }
        }
        let x = SIMD3<Float>(1, 0, 0)
        let y = SIMD3<Float>(0, 1, 0)
        let z = SIMD3<Float>(0, 0, 1)
        switch light.kind {
        case .directional:
            let s = size
            let corners = [SIMD3(-s, s, 0), SIMD3(s, s, 0), SIMD3(s, -s, 0), SIMD3(-s, -s, 0)]
            for index in 0..<4 { add(corners[index], corners[(index + 1) % 4]) }
            add(.zero, SIMD3(0, 0, -s))
        case .point:
            circle(radius: size, center: .zero, u: x, v: y)
            circle(radius: size, center: .zero, u: y, v: z)
            circle(radius: size, center: .zero, u: z, v: x)
        case .spot(let range, let inner, let outer):
            let length = range ?? size
            let tip = SIMD3<Float>(0, 0, -length)
            let radius = length * tan(outer)
            add(.zero, tip)
            for direction in [x, -x, y, -y] { add(.zero, tip + direction * radius) }
            circle(radius: radius, center: tip, u: x, v: y)
            if inner > 0 { circle(radius: length * tan(inner), center: tip, u: x, v: y) }
        case .rectangle(let width, let height):
            let w = width / 2
            let h = height / 2
            let corners = [SIMD3(-w, h, 0), SIMD3(w, h, 0), SIMD3(w, -h, 0), SIMD3(-w, -h, 0)]
            for index in 0..<4 { add(corners[index], corners[(index + 1) % 4]) }
            add(.zero, SIMD3(0, 0, -min(w, h)))
        }
        return try LineGeometry(
            positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
        )
    }

    /// A two-tone octahedral wireframe for the same authored sky, ground and
    /// axis as `LightProbe.hemisphere`. The ground color draws the equator.
    static func hemisphereLight(
        skyColor: SIMD3<Float>, groundColor: SIMD3<Float>, direction: SIMD3<Float> = SIMD3(0, 1, 0),
        size: Float = 1
    ) throws -> LineGeometry {
        try requirePositive(size, "hemisphere helper size")
        guard primitiveFinite(direction), direction != .zero else {
            throw PrimitiveGeometryError.invalidHelperParameter("the hemisphere helper direction must be finite and nonzero")
        }
        let axis = simd_normalize(SIMD3<Double>(direction))
        let reference: SIMD3<Double> = abs(axis.y) < 0.99 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let tangent = simd_normalize(simd_cross(reference, axis))
        let bitangent = simd_cross(axis, tangent)
        let ring = [tangent, bitangent, -tangent, -bitangent].map { SIMD3<Float>($0 * Double(size)) }
        let sky = SIMD3<Float>(axis * Double(size))
        var positions = ring.flatMap { [sky, $0] }
        positions += ring.flatMap { [-sky, $0] }
        for index in ring.indices { positions += [ring[index], ring[(index + 1) % ring.count]] }
        let colors = Array(repeating: SIMD4(skyColor, 1), count: 8)
            + Array(repeating: SIMD4(groundColor, 1), count: 16)
        return try LineGeometry(positions: positions, colors: colors, topology: .segments)
    }

    /// Three.js `PlaneHelper` outline for the plane `dot(normal, p) + constant = 0`:
    /// a square of side `size` centered on the point nearest the origin, with
    /// both diagonals, in world space. The seven segments follow the Three.js
    /// line strip, which traces one side twice.
    static func plane(
        normal: SIMD3<Float>, constant: Float, size: Float = 1, color: SIMD4<Float> = SIMD4(1, 1, 0, 1)
    ) throws -> LineGeometry {
        try requirePositive(size, "plane helper size")
        let frame = try PlaneFrame(normal: normal, constant: constant)
        let h = size / 2
        let corners = [(1, -1), (-1, 1), (-1, -1), (1, 1)].map { frame.point(Float($0.0) * h, Float($0.1) * h) }
        // The Three.js line visits (1,-1), (-1,1), (-1,-1), (1,1), (-1,1), (-1,-1), (1,-1), (1,1).
        let path = [0, 1, 2, 3, 1, 2, 0, 3]
        var positions: [SIMD3<Float>] = []
        for index in 0..<(path.count - 1) { positions += [corners[path[index]], corners[path[index + 1]]] }
        return try LineGeometry(
            positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
        )
    }

    /// Three.js `SkeletonHelper`: one segment from each joint to its nearest
    /// joint ancestor, between world positions, blue at the joint and green at
    /// the ancestor, in Three.js vertex order. Three.js connects only a bone
    /// whose direct parent is a bone; this skips non-joint nodes in between.
    /// Pass the joint node identifiers, for example a glTF skin's joints mapped
    /// through `PreparedGLTFScene.nodeIDs`. Returns nil when no joint has a
    /// joint ancestor.
    static func skeleton(in graph: SceneGraph, joints: [UUID]) throws -> LineGeometry? {
        try skeleton(in: graph, joints: joints, connections: .nearestJointAncestor)
    }

    /// A world-space joint snapshot. Visibility filtering follows inherited
    /// node visibility. Camera layers affect the helper node independently.
    static func skeleton(
        in graph: SceneGraph, joints: [UUID], connections: SkeletonConnections, visibleOnly: Bool = false,
        colors style: SkeletonColors = .init()
    ) throws -> LineGeometry? {
        let jointSet = Set(joints)
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        var pending = graph.roots.reversed().map {
            (node: $0, parent: simd_float4x4?.none, parentJoint: SIMD3<Float>?.none, ancestor: SIMD3<Float>?.none)
        }
        while let (node, parent, parentJoint, ancestor) = pending.popLast() {
            if visibleOnly, !node.isVisible { continue }
            let world = node.resolvedWorldTransform(relativeTo: parent)
            var joint: SIMD3<Float>?
            if jointSet.contains(node.id) {
                let position = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
                guard primitiveFinite(position) else { throw SceneGraph.ValidationError.invalidWorldTransform(node.id) }
                let connection = connections == .directJointParent ? parentJoint : ancestor
                if let connection {
                    positions += [position, connection]
                    colors += [style.joint, style.parent]
                }
                joint = position
            }
            for child in node.children.reversed() { pending.append((child, world, joint, joint ?? ancestor)) }
        }
        guard !positions.isEmpty else { return nil }
        return try LineGeometry(positions: positions, colors: colors, topology: .segments)
    }

    /// The camera center for a perspective projection: the point whose clip x,
    /// y and w are all zero. Nil for an orthographic projection.
    private static func projectionCenter(_ projection: simd_float4x4) -> SIMD3<Float>? {
        let m = helperDouble(projection)
        func row(_ index: Int) -> SIMD4<Double> {
            SIMD4(m.columns.0[index], m.columns.1[index], m.columns.2[index], m.columns.3[index])
        }
        let rows = [row(0), row(1), row(3)]
        guard simd_length(SIMD3(rows[2].x, rows[2].y, rows[2].z)) > 1e-12 else { return nil }
        let matrix = simd_double3x3(rows: rows.map { SIMD3($0.x, $0.y, $0.z) })
        let determinant = simd_determinant(matrix)
        guard determinant.isFinite, abs(determinant) > 1e-18 else { return nil }
        let center = simd_inverse(matrix) * -SIMD3(rows[0].w, rows[1].w, rows[2].w)
        let result = SIMD3<Float>(center)
        return primitiveFinite(result) ? result : nil
    }
}

public extension LineGeometry {
    /// The linear color of an sRGB-encoded `0xRRGGBB` value with full opacity, as
    /// Three.js color management converts helper colors.
    static func helperColor(_ hex: UInt32) -> SIMD4<Float> {
        func linear(_ shift: UInt32) -> Float {
            let encoded = Double((hex >> shift) & 0xff) / 255
            return Float(encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4))
        }
        return SIMD4(linear(16), linear(8), linear(0), 1)
    }
}

/// Prepares Three.js-style helper nodes on one device, drawing their lines with
/// a shared width and blending.
public struct DebugHelpers {
    public let device: any MTLDevice
    public let colorPixelFormat: MTLPixelFormat
    /// The material for every helper line; its color multiplies the vertex colors.
    public let lineMaterial: LineMaterial
    /// The depth mode of the passes the helpers draw in. Lines work in either;
    /// the arrow head and plane fill are meshes prepared for this mode.
    public let depthMode: MeshDepthMode

    public init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat = .rgba16Float,
        lineMaterial: LineMaterial = LineMaterial(width: .pixels(1.5), join: .miter(), cap: .butt),
        depthMode: MeshDepthMode = .forward
    ) {
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.lineMaterial = lineMaterial
        self.depthMode = depthMode
    }

    /// A node that draws helper geometry with the shared line material.
    public func node(_ geometry: LineGeometry, transform: SceneTransform = SceneTransform()) throws -> SceneNode {
        var helper = SceneNode(
            transform: transform,
            primitive: .lines(try LineRenderer(
                device: device, colorPixelFormat: colorPixelFormat, geometry: geometry, material: lineMaterial
            ))
        )
        helper.excludesFromDebugBounds = true
        return helper
    }

    /// Prepares the hemisphere wireframe at `origin`, using the same color
    /// and axis inputs as the scene's diffuse hemisphere probe.
    public func hemisphereLight(
        skyColor: SIMD3<Float>, groundColor: SIMD3<Float>, direction: SIMD3<Float> = SIMD3(0, 1, 0),
        origin: SIMD3<Float> = .zero, size: Float = 1
    ) throws -> SceneNode {
        guard primitiveFinite(origin) else {
            throw PrimitiveGeometryError.invalidHelperParameter("the hemisphere helper origin must be finite")
        }
        let geometry = try LineGeometry.hemisphereLight(
            skyColor: skyColor, groundColor: groundColor, direction: direction, size: size
        )
        var helper = try node(geometry, transform: SceneTransform(translation: origin))
        helper.castsShadow = false
        helper.name = "Hemisphere light helper"
        return helper
    }

    /// Three.js `ArrowHelper`: a shaft line and an unlit cone head of `headLength`
    /// and base diameter `headWidth`, from `origin` along `direction`. The
    /// defaults follow Three.js: a head 20% of the length and 20% as wide as it is long.
    public func arrow(
        direction: SIMD3<Float>, origin: SIMD3<Float> = .zero, length: Float = 1,
        color: SIMD3<Float> = SIMD3(1, 1, 0), headLength: Float? = nil, headWidth: Float? = nil
    ) throws -> SceneNode {
        let layout = try ArrowLayout(
            direction: direction, origin: origin, length: length, color: color,
            headLength: headLength, headWidth: headWidth
        )
        var children: [SceneNode] = []
        if layout.shaftLength > 0 {
            children.append(try node(ArrowLayout.shaftGeometry(length: layout.shaftLength, color: color)))
        }
        children.append(try arrowHead(base: layout.shaftLength, tip: length, width: layout.headWidth, color: color))
        var helper = SceneNode(transform: layout.pose, children: children)
        helper.excludesFromDebugBounds = true
        return helper
    }

    func arrowHead(base: Float, tip: Float, width: Float, color: SIMD3<Float>) throws -> SceneNode {
        let cone = try MeshGeometry.lathe(
            profile: [LatheProfilePoint(radius: width / 2, y: base), LatheProfilePoint(radius: 0, y: tip)],
            segments: 16
        )
        let material = try PhysicallyBasedMaterial(baseColorFactor: SIMD4(color, 1), alphaMode: .opaque, shadingModel: .unlit)
        return SceneNode(
            mesh: try MeshRenderer(
                device: device, colorPixelFormat: colorPixelFormat, geometry: cone, material: .physicallyBased(material),
                depthMode: depthMode
            ),
            castsShadow: false
        )
    }

    /// Three.js `PlaneHelper`: the plane's outline from
    /// ``LineGeometry/plane(normal:constant:size:color:)`` and an unlit fill at
    /// 20% opacity. The fill is double-sided; Three.js shows it only from the
    /// side the normal faces.
    public func plane(
        normal: SIMD3<Float>, constant: Float, size: Float = 1, color: SIMD3<Float> = SIMD3(1, 1, 0)
    ) throws -> SceneNode {
        guard (0...1).contains(color.x), (0...1).contains(color.y), (0...1).contains(color.z) else {
            throw PrimitiveGeometryError.invalidHelperParameter("the plane color must be in 0 to 1")
        }
        // Build about the origin and place the node at the plane's center, so the
        // blended fill sorts by its own position.
        let outline = try LineGeometry.plane(normal: normal, constant: 0, size: size, color: SIMD4(color, 1))
        let center = try PlaneFrame(normal: normal, constant: constant).center
        let frame = try PlaneFrame(normal: normal, constant: 0)
        let h = size / 2
        let corners = [frame.point(-h, -h), frame.point(h, -h), frame.point(h, h), frame.point(-h, h)]
        let fill = try MeshGeometry(
            vertices: corners.map { MeshVertex(position: $0, normal: frame.normal) }, indices: [0, 1, 2, 0, 2, 3]
        )
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(color, 0.2), alphaMode: .blend, isDoubleSided: true, shadingModel: .unlit
        )
        let fillNode = SceneNode(
            mesh: try MeshRenderer(
                device: device, colorPixelFormat: colorPixelFormat, geometry: fill, material: .physicallyBased(material),
                depthMode: depthMode
            ),
            castsShadow: false
        )
        var outlineNode = try node(outline, transform: SceneTransform(translation: center))
        outlineNode.children = [fillNode]
        return outlineNode
    }
}

/// Validated dimensions and pose shared by static and attached arrows.
struct ArrowLayout {
    let pose: SceneTransform
    let headLength: Float
    let headWidth: Float
    let shaftLength: Float

    init(
        direction: SIMD3<Float>, origin: SIMD3<Float>, length: Float, color: SIMD3<Float>,
        headLength: Float?, headWidth: Float?
    ) throws {
        let head = headLength ?? 0.2 * length
        let width = headWidth ?? 0.2 * head
        try requirePositive(length, "arrow length")
        guard head.isFinite, head > 0, head <= length, width.isFinite, width > 0 else {
            throw PrimitiveGeometryError.invalidHelperParameter("the arrow head needs 0 < headLength ≤ length and a positive width")
        }
        guard primitiveFinite(direction), direction != .zero, primitiveFinite(origin),
              (0...1).contains(color.x), (0...1).contains(color.y), (0...1).contains(color.z) else {
            throw PrimitiveGeometryError.invalidHelperParameter("the arrow needs a finite origin, a nonzero direction and color in 0 to 1")
        }
        let directionLength = simd_length(direction)
        let unit = directionLength.isNormal ? direction / directionLength
            : SIMD3<Float>(simd_normalize(SIMD3<Double>(direction)))
        // The arrow is built along +Y, as in Three.js, then turned to the direction.
        let rotation = unit.y < -0.999_999 ? simd_quatf(angle: .pi, axis: SIMD3(1, 0, 0))
            : simd_quatf(from: SIMD3(0, 1, 0), to: unit)
        pose = SceneTransform(translation: origin, rotation: rotation)
        self.headLength = head
        self.headWidth = width
        shaftLength = length - head
    }

    var shaftTransform: SceneTransform {
        SceneTransform(scale: SIMD3(1, shaftLength > 0 ? shaftLength : 1, 1))
    }

    var headTransform: SceneTransform {
        SceneTransform(
            translation: SIMD3(0, shaftLength, 0), scale: SIMD3(headWidth, headLength, headWidth)
        )
    }

    static func shaftGeometry(length: Float, color: SIMD3<Float>) throws -> LineGeometry {
        try LineGeometry(
            positions: [.zero, SIMD3(0, length, 0)], colors: Array(repeating: SIMD4(color, 1), count: 2)
        )
    }
}

/// An orthonormal frame on the plane `dot(normal, p) + constant = 0`.
struct PlaneFrame {
    let normal: SIMD3<Float>
    let center: SIMD3<Float>
    let u: SIMD3<Float>
    let v: SIMD3<Float>

    init(normal: SIMD3<Float>, constant: Float) throws {
        let length = simd_length(normal)
        guard primitiveFinite(normal), length.isNormal, constant.isFinite else {
            throw PrimitiveGeometryError.invalidHelperParameter("the plane needs a finite nonzero normal and finite constant")
        }
        let unit = normal / length
        self.normal = unit
        center = -unit * (constant / length)
        let reference = abs(unit.y) < 0.99 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
        u = simd_normalize(simd_cross(reference, unit))
        v = simd_cross(unit, u)
    }

    func point(_ a: Float, _ b: Float) -> SIMD3<Float> {
        center + u * a + v * b
    }
}

private func helperDouble(_ matrix: simd_float4x4) -> simd_double4x4 {
    let (x, y, z, w) = matrix.columns
    return simd_double4x4(columns: (SIMD4<Double>(x), SIMD4<Double>(y), SIMD4<Double>(z), SIMD4<Double>(w)))
}

private func requirePositive(_ value: Float, _ name: String) throws {
    guard value.isFinite, value > 0 else {
        throw PrimitiveGeometryError.invalidHelperParameter("\(name) must be finite and positive")
    }
}

extension SceneGraph {
    /// The world box around a node's subtree, enclosing meshes (with instances)
    /// and primitives (with world-unit padding), or nil when nothing draws.
    func subtreeWorldBounds(of nodeID: UUID) throws -> AxisAlignedBox? {
        var pending = roots.reversed().map { (node: $0, parent: simd_float4x4?.none, inside: false) }
        var lower = SIMD3<Float>(repeating: .infinity)
        var upper = SIMD3<Float>(repeating: -.infinity)
        var found = false
        while let (node, parent, ancestorInside) = pending.popLast() {
            guard !node.excludesFromDebugBounds else { continue }
            let world = node.resolvedWorldTransform(relativeTo: parent)
            let inside = ancestorInside || node.id == nodeID
            found = found || node.id == nodeID
            if inside {
                var boxes: [AxisAlignedBox] = []
                let mesh = node.mesh ?? node.levelsOfDetail?.levels.first?.mesh
                if let local = node.cullingBounds {
                    if mesh != nil || node.batch != nil || node.primitive != nil { boxes.append(try local.transformed(by: world)) }
                } else {
                    if let batch = node.batch, let box = try batch.bounds(world: world, visibleOnly: false) {
                        boxes.append(box)
                    }
                    if let mesh {
                        let local = try node.instances.flatMap { try $0.bounds(enclosing: mesh.bounds) } ?? mesh.bounds
                        if node.instances?.count != 0 { boxes.append(try local.transformed(by: world)) }
                    }
                    if let primitive = node.primitive {
                        let anchors = try primitive.localBounds.transformed(by: world)
                        let padding: SIMD3<Float>
                        if case .points(let renderer) = primitive, renderer.material.shape == .square,
                           case .world(let size) = renderer.material.size, cameraTransform.scale == SIMD3(repeating: 1),
                           abs(simd_length_squared(cameraTransform.rotation.vector) - 1) < 1e-4 {
                            let right = simd_abs(cameraTransform.rotation.act(SIMD3(1, 0, 0)))
                            let up = simd_abs(cameraTransform.rotation.act(SIMD3(0, 1, 0)))
                            padding = (right + up) * (size * renderer.maximumSizeScale / 2)
                        } else {
                            padding = SIMD3(repeating: primitive.worldPadding)
                        }
                        boxes.append(try AxisAlignedBox(minimum: anchors.minimum - padding, maximum: anchors.maximum + padding))
                    }
                }
                for box in boxes {
                    lower = simd_min(lower, box.minimum)
                    upper = simd_max(upper, box.maximum)
                }
            }
            for child in node.children.reversed() { pending.append((child, world, inside)) }
        }
        guard found, primitiveFinite(lower), primitiveFinite(upper) else { return nil }
        return try AxisAlignedBox(minimum: lower, maximum: upper)
    }
}
