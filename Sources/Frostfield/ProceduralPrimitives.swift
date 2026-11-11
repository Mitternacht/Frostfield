import Foundation
import simd

// Three.js-compatible primitive generators, checked against Three.js r187. Vertex
// order, positions, normals and the order of triangles and their corners follow the
// Three.js geometry of the same name, so ported scenes, groups and index ranges line
// up. Texture coordinates use Frostfield's top-left origin, so V is one minus the
// Three.js value. Every vertex carries an analytic normal and a tangent along
// increasing U whose handedness makes cross(normal, tangent) × w point along
// increasing V. Deliberate differences: zero-area triangles that Three.js emits are
// left out (the capsule's 2 × radialSegments pole triangles and, at zero height, its
// middle band; a ring's center slivers at zero inner radius), and capsule pole
// vertices take the U of the wedge they close, where Three.js is one wedge off. The
// vertex arrays keep Three.js's counts, so the remaining indices match its values.
extension MeshGeometry {
    /// A box centered at the origin, like Three.js `BoxGeometry`. Faces follow +X, −X,
    /// +Y, −Y, +Z, −Z, each a grid with its own top-left UV chart and one material
    /// group (indices 0 to 5). Supply positive sizes and 1 to 256 segments per axis.
    public static func box(
        width: Float = 1, height: Float = 1, depth: Float = 1,
        widthSegments: Int = 1, heightSegments: Int = 1, depthSegments: Int = 1
    ) throws -> MeshGeometry {
        guard [width, height, depth].allSatisfy({ $0.isNormal && $0 > 0 }) else { throw PrimitiveError.size }
        guard [widthSegments, heightSegments, depthSegments].allSatisfy({ (1...256).contains($0) }) else {
            throw PrimitiveError.segments
        }
        let half = SIMD3<Double>(Double(width), Double(height), Double(depth)) / 2
        let segments = SIMD3(widthSegments, heightSegments, depthSegments)
        // (normal axis, right axis and sign, up axis and sign), matching MeshGeometry.cube().
        let faces: [(normal: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>)] = [
            (SIMD3(1, 0, 0), SIMD3(0, 0, -1), SIMD3(0, 1, 0)),
            (SIMD3(-1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 0)),
            (SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, -1)),
            (SIMD3(0, -1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)),
            (SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0)),
            (SIMD3(0, 0, -1), SIMD3(-1, 0, 0), SIMD3(0, 1, 0)),
        ]
        var builder = PrimitiveBuilder()
        var groups: [MeshGeometryGroup] = []
        for (faceIndex, face) in faces.enumerated() {
            let rightAxis = Self.axisIndex(face.right)
            let upAxis = Self.axisIndex(face.up)
            let columns = segments[rightAxis]
            let rows = segments[upAxis]
            let rightExtent = half[rightAxis]
            let upExtent = half[upAxis]
            let center = face.normal * half[Self.axisIndex(face.normal)]
            let first = builder.indices.count
            let base = builder.vertexCount
            for row in 0...rows {
                let v = Double(row) / Double(rows)
                for column in 0...columns {
                    let u = Double(column) / Double(columns)
                    let position = center + face.right * ((u - 0.5) * 2 * rightExtent) + face.up * ((0.5 - v) * 2 * upExtent)
                    builder.add(
                        position: position, normal: face.normal, uv: SIMD2(u, v),
                        dPdu: face.right, dPdv: -face.up
                    )
                }
            }
            builder.addGrid(base: base, columns: columns, rows: rows)
            groups.append(MeshGeometryGroup(start: first, count: builder.indices.count - first, materialIndex: faceIndex))
        }
        return try builder.geometry().withGroups(groups)
    }

    /// A cylinder or truncated cone around Y, like Three.js `CylinderGeometry`. The
    /// side starts at `thetaStart` radians from +Z toward +X and sweeps `thetaLength`.
    /// Groups are 0 for the side, 1 for the top cap and 2 for the bottom cap. One
    /// radius may be zero, which closes that end in a point; an open-ended or
    /// zero-radius end has no cap. Supply 3 to 512 radial and 1 to 512 height segments.
    public static func cylinder(
        radiusTop: Float = 1, radiusBottom: Float = 1, height: Float = 1,
        radialSegments: Int = 32, heightSegments: Int = 1, openEnded: Bool = false,
        thetaStart: Float = 0, thetaLength: Float = 2 * .pi
    ) throws -> MeshGeometry {
        guard radiusTop.isFinite, radiusBottom.isFinite, radiusTop >= 0, radiusBottom >= 0,
              radiusTop > 0 || radiusBottom > 0, height.isNormal, height > 0 else {
            throw PrimitiveError.size
        }
        guard (3...512).contains(radialSegments), (1...512).contains(heightSegments) else { throw PrimitiveError.segments }
        try validateArc(start: thetaStart, length: thetaLength, maximum: 2 * .pi)
        let top = Double(radiusTop)
        let bottom = Double(radiusBottom)
        let fullHeight = Double(height)
        let halfHeight = fullHeight / 2
        let slope = (bottom - top) / fullHeight
        let start = Double(thetaStart)
        let length = Double(thetaLength)

        var builder = PrimitiveBuilder()
        var groups: [MeshGeometryGroup] = []
        var grid: [[UInt32]] = []
        for row in 0...heightSegments {
            let v = Double(row) / Double(heightSegments)
            let radius = v * (bottom - top) + top
            var line: [UInt32] = []
            for column in 0...radialSegments {
                let u = Double(column) / Double(radialSegments)
                let theta = u * length + start
                let sine = sin(theta)
                let cosine = cos(theta)
                let position = SIMD3(radius * sine, -v * fullHeight + halfHeight, radius * cosine)
                let normal = simd_normalize(SIMD3(sine, slope, cosine))
                line.append(builder.add(
                    position: position, normal: normal, uv: SIMD2(u, v),
                    dPdu: SIMD3(cosine, 0, -sine) * length,
                    dPdv: SIMD3(sine * (bottom - top), -fullHeight, cosine * (bottom - top))
                ))
            }
            grid.append(line)
        }
        // Column-major like Three.js generateTorso, which loops x outside y.
        for column in 0..<radialSegments {
            for row in 0..<heightSegments {
                let a = grid[row][column]
                let b = grid[row + 1][column]
                let c = grid[row + 1][column + 1]
                let d = grid[row][column + 1]
                if top > 0 || row != 0 { builder.indices += [a, b, d] }
                if bottom > 0 || row != heightSegments - 1 { builder.indices += [b, c, d] }
            }
        }
        groups.append(MeshGeometryGroup(start: 0, count: builder.indices.count, materialIndex: 0))
        if !openEnded {
            for (isTop, radius) in [(true, top), (false, bottom)] where radius > 0 {
                let first = builder.indices.count
                let sign: Double = isTop ? 1 : -1
                let y = halfHeight * sign
                let normal = SIMD3<Double>(0, sign, 0)
                var centers: [UInt32] = []
                for _ in 1...radialSegments {
                    centers.append(builder.add(
                        position: SIMD3(0, y, 0), normal: normal, uv: SIMD2(0.5, 0.5),
                        dPdu: SIMD3(0, 0, 1), dPdv: SIMD3(-sign, 0, 0)
                    ))
                }
                var rim: [UInt32] = []
                for column in 0...radialSegments {
                    let theta = Double(column) / Double(radialSegments) * length + start
                    let sine = sin(theta)
                    let cosine = cos(theta)
                    rim.append(builder.add(
                        position: SIMD3(radius * sine, y, radius * cosine), normal: normal,
                        uv: SIMD2(cosine * 0.5 + 0.5, 0.5 - sine * 0.5 * sign),
                        dPdu: SIMD3(0, 0, 1), dPdv: SIMD3(-sign, 0, 0)
                    ))
                }
                for column in 0..<radialSegments {
                    let center = centers[column]
                    let i = rim[column]
                    let next = rim[column + 1]
                    builder.indices += isTop ? [i, next, center] : [next, i, center]
                }
                groups.append(MeshGeometryGroup(
                    start: first, count: builder.indices.count - first, materialIndex: isTop ? 1 : 2
                ))
            }
        }
        return try builder.geometry().withGroups(groups)
    }

    /// A cone around Y with its apex at +height/2, like Three.js `ConeGeometry`.
    /// Groups are 0 for the side and 2 for the base.
    public static func cone(
        radius: Float = 1, height: Float = 1, radialSegments: Int = 32, heightSegments: Int = 1,
        openEnded: Bool = false, thetaStart: Float = 0, thetaLength: Float = 2 * .pi
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0 else { throw PrimitiveError.size }
        return try cylinder(
            radiusTop: 0, radiusBottom: radius, height: height, radialSegments: radialSegments,
            heightSegments: heightSegments, openEnded: openEnded, thetaStart: thetaStart, thetaLength: thetaLength
        )
    }

    /// A capsule around Y, like Three.js `CapsuleGeometry`: a cylinder of `height`
    /// between two hemispheres of `radius`, so the total height is `height + 2 radius`.
    /// Rows run from the bottom pole to the top, as in Three.js, with `capSegments` rows
    /// per cap and `heightSegments` along the cylinder. Around each row, U = θ / 2π puts
    /// vertices at (−r cos θ, y, r sin θ), so the seam faces −X and U grows toward +Z;
    /// V runs from the top pole to the bottom by arc length. Supply 1 to 128 cap
    /// segments, 3 to 512 radial and 1 to 512 height segments. The geometry has the
    /// Three.js vertex count, (radialSegments + 1)(2 capSegments + heightSegments + 1),
    /// and its triangles without the zero-area ones: 2 × radialSegments at the poles,
    /// plus the 2 × radialSegments × heightSegments of the middle band when `height` is 0.
    public static func capsule(
        radius: Float = 1, height: Float = 1, capSegments: Int = 4, radialSegments: Int = 8,
        heightSegments: Int = 1
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0, height.isFinite, height >= 0 else { throw PrimitiveError.size }
        guard (1...128).contains(capSegments), (3...512).contains(radialSegments), (1...512).contains(heightSegments) else {
            throw PrimitiveError.segments
        }
        let r = Double(radius)
        let length = Double(height)
        let halfLength = length / 2
        let capArc = Double.pi / 2 * r
        let totalArc = 2 * capArc + length
        let rowCount = 2 * capSegments + heightSegments
        var builder = PrimitiveBuilder()
        for row in 0...rowCount {
            // The profile's radial distance and height, its unit normal in the meridian
            // plane (outward, up), and the arc length from the bottom pole.
            let profileRadius: Double
            let y: Double
            let meridian: SIMD2<Double>
            let arc: Double
            if row <= capSegments {
                let progress = Double(row) / Double(capSegments)
                let angle = progress * .pi / 2
                profileRadius = r * sin(angle)
                y = -halfLength - r * cos(angle)
                meridian = SIMD2(sin(angle), -cos(angle))
                arc = progress * capArc
            } else if row <= capSegments + heightSegments {
                let progress = Double(row - capSegments) / Double(heightSegments)
                profileRadius = r
                y = -halfLength + progress * length
                meridian = SIMD2(1, 0)
                arc = capArc + progress * length
            } else {
                let progress = Double(row - capSegments - heightSegments) / Double(capSegments)
                let angle = progress * .pi / 2
                profileRadius = r * cos(angle)
                y = halfLength + r * sin(angle)
                meridian = SIMD2(cos(angle), sin(angle))
                arc = capArc + length + progress * capArc
            }
            // Each pole vertex closes the wedge to its left at the bottom and to its right
            // at the top, so it sits at that wedge's center in U.
            let poleOffset = row == 0 ? -0.5 : row == rowCount ? 0.5 : 0
            let pole = row == 0 || row == rowCount
            for column in 0...radialSegments {
                let u = Double(column) / Double(radialSegments)
                let theta = u * 2 * .pi
                let radial = SIMD3(-cos(theta), 0, sin(theta))
                let normal = pole ? SIMD3<Double>(0, meridian.y > 0 ? 1 : -1, 0) : radial * meridian.x + SIMD3(0, meridian.y, 0)
                // dP/dV runs down the profile, perpendicular to the normal in the meridian plane.
                let down = radial * meridian.y + SIMD3(0, -meridian.x, 0)
                builder.add(
                    position: radial * profileRadius + SIMD3(0, y, 0), normal: normal,
                    uv: SIMD2(u + poleOffset / Double(radialSegments), 1 - min(max(arc / totalArc, 0), 1)),
                    dPdu: SIMD3(sin(theta), 0, cos(theta)), dPdv: down
                )
            }
        }
        let stride = UInt32(radialSegments + 1)
        // At zero height the middle rows coincide with the equator, so their band has no area.
        let flatBand = length == 0 ? (capSegments + 1)..<(capSegments + heightSegments + 1) : 0..<0
        for row in 1...UInt32(rowCount) where !flatBand.contains(Int(row)) {
            for column in 0..<UInt32(radialSegments) {
                // Three.js order: (i1, i2, i3) and (i2, i4, i3) between the previous row and this one.
                let i1 = (row - 1) * stride + column
                let i2 = i1 + 1
                let i3 = row * stride + column
                let i4 = i3 + 1
                if row != 1 { builder.indices += [i1, i2, i3] }
                if row != UInt32(rowCount) { builder.indices += [i2, i4, i3] }
            }
        }
        return try builder.geometry()
    }

    /// A torus around Z, like Three.js `TorusGeometry`: the tube center follows a
    /// circle of `radius` in the XY plane from +X through `arc` radians, and the
    /// tube has radius `tube`. The tube's cross-section sweeps `thetaLength` radians
    /// from `thetaStart`, measured from the outer equator toward +Z, so a partial sweep
    /// leaves an open trough. Use `tube < radius`. Supply 3 to 512 segments each way.
    public static func torus(
        radius: Float = 1, tube: Float = 0.4, radialSegments: Int = 12, tubularSegments: Int = 48,
        arc: Float = 2 * .pi, thetaStart: Float = 0, thetaLength: Float = 2 * .pi
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0, tube.isNormal, tube > 0, tube < radius else { throw PrimitiveError.size }
        guard (3...512).contains(radialSegments), (3...512).contains(tubularSegments) else { throw PrimitiveError.segments }
        try validateArc(start: 0, length: arc, maximum: 2 * .pi)
        try validateArc(start: thetaStart, length: thetaLength, maximum: 2 * .pi)
        let major = Double(radius)
        let minor = Double(tube)
        let sweep = Double(arc)
        let tubeStart = Double(thetaStart)
        let tubeSweep = Double(thetaLength)
        var builder = PrimitiveBuilder()
        for j in 0...radialSegments {
            for i in 0...tubularSegments {
                let u = Double(i) / Double(tubularSegments) * sweep
                let v = tubeStart + Double(j) / Double(radialSegments) * tubeSweep
                let ring = major + minor * cos(v)
                let position = SIMD3(ring * cos(u), ring * sin(u), minor * sin(v))
                let center = SIMD3(major * cos(u), major * sin(u), 0)
                // Frostfield V is one minus the Three.js V, so dP/dV reverses the tube angle.
                let dPdv = SIMD3(-minor * sin(v) * cos(u), -minor * sin(v) * sin(u), minor * cos(v)) * -tubeSweep
                builder.add(
                    position: position, normal: simd_normalize(position - center),
                    uv: SIMD2(Double(i) / Double(tubularSegments), 1 - Double(j) / Double(radialSegments)),
                    dPdu: SIMD3(-sin(u), cos(u), 0) * (ring * sweep), dPdv: dPdv
                )
            }
        }
        let stride = UInt32(tubularSegments + 1)
        for j in 1...UInt32(radialSegments) {
            for i in 1...UInt32(tubularSegments) {
                let a = stride * j + i - 1
                let b = stride * (j - 1) + i - 1
                let c = stride * (j - 1) + i
                let d = stride * j + i
                builder.indices += [a, b, d, b, c, d]
            }
        }
        return try builder.geometry()
    }

    /// A (p, q) torus knot tube, like Three.js `TorusKnotGeometry`. The curve winds
    /// `p` times around the Z axis and `q` times through the hole. Its frame uses the
    /// exact curve derivative, so the seam rings coincide. Use coprime positive `p`
    /// and `q` for one closed knot, 3 to 2,048 tubular and 3 to 256 radial segments.
    public static func torusKnot(
        radius: Float = 1, tube: Float = 0.4, tubularSegments: Int = 64, radialSegments: Int = 8,
        p: Int = 2, q: Int = 3
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0, tube.isNormal, tube > 0 else { throw PrimitiveError.size }
        guard (3...2_048).contains(tubularSegments), (3...256).contains(radialSegments) else { throw PrimitiveError.segments }
        guard (1...64).contains(p), (1...64).contains(q), Self.greatestCommonDivisor(p, q) == 1 else {
            throw PrimitiveError.knotWinding
        }
        let r = Double(radius)
        let tubeRadius = Double(tube)
        let ratio = Double(q) / Double(p)
        let span = Double(p) * 2 * .pi
        func curve(_ u: Double) -> SIMD3<Double> {
            let c = cos(ratio * u)
            return SIMD3(r * (2 + c) * 0.5 * cos(u), r * (2 + c) * 0.5 * sin(u), r * sin(ratio * u) * 0.5)
        }
        func derivative(_ u: Double) -> SIMD3<Double> {
            let c = cos(ratio * u)
            let s = sin(ratio * u)
            return SIMD3(
                r * 0.5 * (-ratio * s * cos(u) - (2 + c) * sin(u)),
                r * 0.5 * (-ratio * s * sin(u) + (2 + c) * cos(u)),
                r * 0.5 * ratio * c
            )
        }
        var builder = PrimitiveBuilder()
        for i in 0...tubularSegments {
            // The last ring reuses the first ring's parameter so the seam closes exactly.
            let u = i == tubularSegments ? 0 : Double(i) / Double(tubularSegments) * span
            let point = curve(u)
            let tangent = simd_normalize(derivative(u))
            let crossed = simd_cross(tangent, point + curve(u))
            guard simd_length(crossed) > 1e-9 else { throw PrimitiveError.degenerateFrame }
            let binormal = simd_normalize(crossed)
            let normalAxis = simd_normalize(simd_cross(binormal, tangent))
            for j in 0...radialSegments {
                let v = Double(j) / Double(radialSegments) * 2 * .pi
                let cx = -tubeRadius * cos(v)
                let cy = tubeRadius * sin(v)
                let offset = normalAxis * cx + binormal * cy
                let tubeDerivative = (normalAxis * sin(v) + binormal * cos(v)) * (tubeRadius * 2 * .pi)
                builder.add(
                    position: point + offset, normal: simd_normalize(offset),
                    uv: SIMD2(Double(i) / Double(tubularSegments), 1 - Double(j) / Double(radialSegments)),
                    dPdu: tangent, dPdv: -tubeDerivative
                )
            }
        }
        let stride = UInt32(radialSegments + 1)
        for j in 1...UInt32(tubularSegments) {
            for i in 1...UInt32(radialSegments) {
                let a = stride * (j - 1) + (i - 1)
                let b = stride * j + (i - 1)
                let c = stride * j + i
                let d = stride * (j - 1) + i
                builder.indices += [a, b, d, b, c, d]
            }
        }
        return try builder.geometry()
    }

    /// A flat annulus facing +Z, like Three.js `RingGeometry`. The sector starts at
    /// `thetaStart` radians from +X toward +Y. UVs map the outer circle's square
    /// bounds to [0, 1]. An inner radius of zero makes a fan: the innermost row keeps
    /// only its (b, c, d) triangles, leaving out the zero-area slivers Three.js emits.
    public static func ring(
        innerRadius: Float = 0.5, outerRadius: Float = 1, thetaSegments: Int = 32, phiSegments: Int = 1,
        thetaStart: Float = 0, thetaLength: Float = 2 * .pi
    ) throws -> MeshGeometry {
        guard innerRadius.isFinite, innerRadius >= 0, outerRadius.isNormal, outerRadius > innerRadius else {
            throw PrimitiveError.size
        }
        guard (3...1_024).contains(thetaSegments), (1...256).contains(phiSegments) else { throw PrimitiveError.segments }
        try validateArc(start: thetaStart, length: thetaLength, maximum: 2 * .pi)
        let inner = Double(innerRadius)
        let outer = Double(outerRadius)
        let start = Double(thetaStart)
        let length = Double(thetaLength)
        var builder = PrimitiveBuilder()
        for j in 0...phiSegments {
            let radius = inner + (outer - inner) * Double(j) / Double(phiSegments)
            for i in 0...thetaSegments {
                let angle = start + Double(i) / Double(thetaSegments) * length
                let position = SIMD3(radius * cos(angle), radius * sin(angle), 0)
                builder.add(
                    position: position, normal: SIMD3(0, 0, 1),
                    uv: SIMD2((position.x / outer + 1) / 2, (1 - position.y / outer) / 2),
                    dPdu: SIMD3(1, 0, 0), dPdv: SIMD3(0, -1, 0)
                )
            }
        }
        let stride = UInt32(thetaSegments + 1)
        for j in 0..<UInt32(phiSegments) {
            for i in 0..<UInt32(thetaSegments) {
                let a = j * stride + i
                let b = a + stride
                let c = b + 1
                let d = a + 1
                if j != 0 || inner > 0 { builder.indices += [a, b, d] }
                builder.indices += [b, c, d]
            }
        }
        return try builder.geometry()
    }

    /// A flat disk or sector facing +Z, like Three.js `CircleGeometry`, starting at
    /// `thetaStart` radians from +X toward +Y. UVs map the disk's square bounds to [0, 1].
    public static func circle(
        radius: Float = 1, segments: Int = 32, thetaStart: Float = 0, thetaLength: Float = 2 * .pi
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0 else { throw PrimitiveError.size }
        guard (3...1_024).contains(segments) else { throw PrimitiveError.segments }
        try validateArc(start: thetaStart, length: thetaLength, maximum: 2 * .pi)
        let r = Double(radius)
        var builder = PrimitiveBuilder()
        builder.add(
            position: .zero, normal: SIMD3(0, 0, 1), uv: SIMD2(0.5, 0.5),
            dPdu: SIMD3(1, 0, 0), dPdv: SIMD3(0, -1, 0)
        )
        for index in 0...segments {
            let angle = Double(thetaStart) + Double(index) / Double(segments) * Double(thetaLength)
            let position = SIMD3(r * cos(angle), r * sin(angle), 0)
            builder.add(
                position: position, normal: SIMD3(0, 0, 1), uv: SIMD2((position.x / r + 1) / 2, (1 - position.y / r) / 2),
                dPdu: SIMD3(1, 0, 0), dPdv: SIMD3(0, -1, 0)
            )
        }
        for index in 1...UInt32(segments) {
            builder.indices += [index, index + 1, 0]
        }
        return try builder.geometry()
    }

    /// A sphere or spherical patch, like Three.js `SphereGeometry`. Longitude `phi`
    /// starts at −X and turns toward +Z; colatitude `theta` runs from the +Y pole.
    /// Pole vertices use the center of their wedge in U. Supply 3 to 1,024 width and
    /// 2 to 1,024 height segments and angles within one turn and one half turn.
    public static func sphere(
        radius: Float = 1, widthSegments: Int = 32, heightSegments: Int = 16,
        phiStart: Float = 0, phiLength: Float = 2 * .pi, thetaStart: Float = 0, thetaLength: Float = .pi
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0 else { throw PrimitiveError.size }
        guard (3...1_024).contains(widthSegments), (2...1_024).contains(heightSegments) else { throw PrimitiveError.segments }
        try validateArc(start: phiStart, length: phiLength, maximum: 2 * .pi)
        guard thetaStart.isFinite, thetaLength.isFinite, thetaStart >= 0, thetaLength > 0,
              Double(thetaStart) + Double(thetaLength) <= Double.pi + 1e-6 else {
            throw PrimitiveError.angle
        }
        let r = Double(radius)
        let phi0 = Double(phiStart)
        let phiSpan = Double(phiLength)
        let theta0 = Double(thetaStart)
        let thetaSpan = Double(thetaLength)
        let thetaEnd = min(theta0 + thetaSpan, .pi)
        let reachesNorth = thetaStart == 0
        let reachesSouth = Float(thetaEnd) >= Float.pi
        var builder = PrimitiveBuilder()
        var grid: [[UInt32]] = []
        for iy in 0...heightSegments {
            let v = Double(iy) / Double(heightSegments)
            var uOffset = 0.0
            if iy == 0, reachesNorth {
                uOffset = 0.5 / Double(widthSegments)
            } else if iy == heightSegments, reachesSouth {
                uOffset = -0.5 / Double(widthSegments)
            }
            let theta = iy == heightSegments ? thetaEnd : theta0 + v * thetaSpan
            var line: [UInt32] = []
            for ix in 0...widthSegments {
                let u = Double(ix) / Double(widthSegments)
                let phi = phi0 + u * phiSpan
                let direction = SIMD3(-cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta))
                line.append(builder.add(
                    position: direction * r, normal: direction, uv: SIMD2(u + uOffset, v),
                    dPdu: SIMD3(sin(phi), 0, cos(phi)) * phiSpan,
                    dPdv: SIMD3(-cos(phi) * cos(theta), -sin(theta), sin(phi) * cos(theta)) * thetaSpan
                ))
            }
            grid.append(line)
        }
        for iy in 0..<heightSegments {
            for ix in 0..<widthSegments {
                let a = grid[iy][ix + 1]
                let b = grid[iy][ix]
                let c = grid[iy + 1][ix]
                let d = grid[iy + 1][ix + 1]
                if iy != 0 || !reachesNorth { builder.indices += [a, b, d] }
                if iy != heightSegments - 1 || !reachesSouth { builder.indices += [b, c, d] }
            }
        }
        return try builder.geometry()
    }

    /// The four-faced polyhedron of Three.js `TetrahedronGeometry`, inscribed in a
    /// sphere of `radius`. See ``polyhedron(vertices:faces:radius:detail:)``.
    public static func tetrahedron(radius: Float = 1, detail: Int = 0) throws -> MeshGeometry {
        try polyhedron(
            vertices: [SIMD3(1, 1, 1), SIMD3(-1, -1, 1), SIMD3(-1, 1, -1), SIMD3(1, -1, -1)],
            faces: [SIMD3(2, 1, 0), SIMD3(0, 3, 2), SIMD3(1, 3, 0), SIMD3(2, 3, 1)],
            radius: radius, detail: detail
        )
    }

    /// The eight-faced polyhedron of Three.js `OctahedronGeometry`.
    public static func octahedron(radius: Float = 1, detail: Int = 0) throws -> MeshGeometry {
        try polyhedron(
            vertices: [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)],
            faces: [
                SIMD3(0, 2, 4), SIMD3(0, 4, 3), SIMD3(0, 3, 5), SIMD3(0, 5, 2),
                SIMD3(1, 2, 5), SIMD3(1, 5, 3), SIMD3(1, 3, 4), SIMD3(1, 4, 2),
            ],
            radius: radius, detail: detail
        )
    }

    /// The twenty-faced polyhedron of Three.js `IcosahedronGeometry`. With a high
    /// detail it is a geodesic sphere of nearly uniform triangles.
    public static func icosahedron(radius: Float = 1, detail: Int = 0) throws -> MeshGeometry {
        let t = Float((1 + 5.0.squareRoot()) / 2)
        return try polyhedron(
            vertices: [
                SIMD3(-1, t, 0), SIMD3(1, t, 0), SIMD3(-1, -t, 0), SIMD3(1, -t, 0),
                SIMD3(0, -1, t), SIMD3(0, 1, t), SIMD3(0, -1, -t), SIMD3(0, 1, -t),
                SIMD3(t, 0, -1), SIMD3(t, 0, 1), SIMD3(-t, 0, -1), SIMD3(-t, 0, 1),
            ],
            faces: [
                SIMD3(0, 11, 5), SIMD3(0, 5, 1), SIMD3(0, 1, 7), SIMD3(0, 7, 10), SIMD3(0, 10, 11),
                SIMD3(1, 5, 9), SIMD3(5, 11, 4), SIMD3(11, 10, 2), SIMD3(10, 7, 6), SIMD3(7, 1, 8),
                SIMD3(3, 9, 4), SIMD3(3, 4, 2), SIMD3(3, 2, 6), SIMD3(3, 6, 8), SIMD3(3, 8, 9),
                SIMD3(4, 9, 5), SIMD3(2, 4, 11), SIMD3(6, 2, 10), SIMD3(8, 6, 7), SIMD3(9, 8, 1),
            ],
            radius: radius, detail: detail
        )
    }

    /// The twelve-faced polyhedron of Three.js `DodecahedronGeometry`; each
    /// pentagon is three triangles.
    public static func dodecahedron(radius: Float = 1, detail: Int = 0) throws -> MeshGeometry {
        let t = Float((1 + 5.0.squareRoot()) / 2)
        let r = 1 / t
        return try polyhedron(
            vertices: [
                SIMD3(-1, -1, -1), SIMD3(-1, -1, 1), SIMD3(-1, 1, -1), SIMD3(-1, 1, 1),
                SIMD3(1, -1, -1), SIMD3(1, -1, 1), SIMD3(1, 1, -1), SIMD3(1, 1, 1),
                SIMD3(0, -r, -t), SIMD3(0, -r, t), SIMD3(0, r, -t), SIMD3(0, r, t),
                SIMD3(-r, -t, 0), SIMD3(-r, t, 0), SIMD3(r, -t, 0), SIMD3(r, t, 0),
                SIMD3(-t, 0, -r), SIMD3(t, 0, -r), SIMD3(-t, 0, r), SIMD3(t, 0, r),
            ],
            faces: [
                SIMD3(3, 11, 7), SIMD3(3, 7, 15), SIMD3(3, 15, 13), SIMD3(7, 19, 17), SIMD3(7, 17, 6), SIMD3(7, 6, 15),
                SIMD3(17, 4, 8), SIMD3(17, 8, 10), SIMD3(17, 10, 6), SIMD3(8, 0, 16), SIMD3(8, 16, 2), SIMD3(8, 2, 10),
                SIMD3(0, 12, 1), SIMD3(0, 1, 18), SIMD3(0, 18, 16), SIMD3(6, 10, 2), SIMD3(6, 2, 13), SIMD3(6, 13, 15),
                SIMD3(2, 16, 18), SIMD3(2, 18, 3), SIMD3(2, 3, 13), SIMD3(18, 1, 9), SIMD3(18, 9, 11), SIMD3(18, 11, 3),
                SIMD3(4, 14, 12), SIMD3(4, 12, 0), SIMD3(4, 0, 8), SIMD3(11, 9, 5), SIMD3(11, 5, 19), SIMD3(11, 19, 7),
                SIMD3(19, 5, 14), SIMD3(19, 14, 4), SIMD3(19, 4, 17), SIMD3(1, 12, 14), SIMD3(1, 14, 5), SIMD3(1, 5, 9),
            ],
            radius: radius, detail: detail
        )
    }

    /// A polyhedron projected onto a sphere, like Three.js `PolyhedronGeometry`. Each
    /// counterclockwise triangular face splits into `(detail + 1)²` triangles whose
    /// vertices move onto the sphere. Detail 0 keeps flat face normals; higher detail
    /// uses smooth radial normals. Every triangle owns its three vertices. UVs are
    /// equirectangular, with U from the azimuth `atan2(z, −x)` and V from the +Y pole,
    /// corrected so faces do not straddle the seam. Supply detail 0 to 64.
    public static func polyhedron(
        vertices source: [SIMD3<Float>], faces: [SIMD3<Int>], radius: Float = 1, detail: Int = 0
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0 else { throw PrimitiveError.size }
        guard (0...64).contains(detail) else { throw PrimitiveError.segments }
        guard !faces.isEmpty, source.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0 != .zero }),
              faces.allSatisfy({ face in (0..<3).allSatisfy { source.indices.contains(face[$0]) } }) else {
            throw PrimitiveError.polyhedronFaces
        }
        let r = Double(radius)
        var positions: [SIMD3<Double>] = []
        let columns = detail + 1
        positions.reserveCapacity(faces.count * columns * columns * 3)
        for face in faces {
            let a = SIMD3<Double>(source[face.x])
            let b = SIMD3<Double>(source[face.y])
            let c = SIMD3<Double>(source[face.z])
            var lattice: [[SIMD3<Double>]] = []
            for i in 0...columns {
                let aj = simd_mix(a, c, SIMD3(repeating: Double(i) / Double(columns)))
                let bj = simd_mix(b, c, SIMD3(repeating: Double(i) / Double(columns)))
                let rows = columns - i
                lattice.append((0...rows).map { j in
                    rows == 0 ? aj : simd_mix(aj, bj, SIMD3(repeating: Double(j) / Double(rows)))
                })
            }
            for i in 0..<columns {
                for j in 0..<(2 * (columns - i) - 1) {
                    let k = j / 2
                    if j.isMultiple(of: 2) {
                        positions += [lattice[i][k + 1], lattice[i + 1][k], lattice[i][k]]
                    } else {
                        positions += [lattice[i][k + 1], lattice[i + 1][k + 1], lattice[i + 1][k]]
                    }
                }
            }
        }
        positions = positions.map { simd_normalize($0) * r }

        func azimuth(_ p: SIMD3<Double>) -> Double { atan2(p.z, -p.x) }
        func inclination(_ p: SIMD3<Double>) -> Double { atan2(-p.y, (p.x * p.x + p.z * p.z).squareRoot()) }
        var uvs = positions.map { SIMD2(azimuth($0) / (2 * .pi) + 0.5, inclination($0) / .pi + 0.5) }
        for start in stride(from: 0, to: positions.count, by: 3) {
            let centroid = (positions[start] + positions[start + 1] + positions[start + 2]) / 3
            let centerAzimuth = azimuth(centroid)
            for offset in 0..<3 {
                let index = start + offset
                if centerAzimuth < 0, uvs[index].x == 1 { uvs[index].x -= 1 }
                if positions[index].x == 0, positions[index].z == 0 {
                    uvs[index].x = centerAzimuth / (2 * .pi) + 0.5
                }
            }
            let us = [uvs[start].x, uvs[start + 1].x, uvs[start + 2].x]
            if us.max()! > 0.9, us.min()! < 0.1 {
                for offset in 0..<3 where uvs[start + offset].x < 0.2 { uvs[start + offset].x += 1 }
            }
        }

        var builder = PrimitiveBuilder()
        for start in stride(from: 0, to: positions.count, by: 3) {
            let p = [positions[start], positions[start + 1], positions[start + 2]]
            let t = [uvs[start], uvs[start + 1], uvs[start + 2]]
            let crossed = simd_cross(p[1] - p[0], p[2] - p[0])
            guard simd_length(crossed) > 0 else { throw PrimitiveError.polyhedronFaces }
            let faceNormal = simd_normalize(crossed)
            // Per-face UV derivatives give the tangent frame of these unshared vertices.
            let e1 = p[1] - p[0]
            let e2 = p[2] - p[0]
            let d1 = t[1] - t[0]
            let d2 = t[2] - t[0]
            let determinant = d1.x * d2.y - d1.y * d2.x
            var dPdu = SIMD3<Double>.zero
            var dPdv = SIMD3<Double>.zero
            if abs(determinant) > 1e-14 {
                dPdu = (e1 * d2.y - e2 * d1.y) / determinant
                dPdv = (e2 * d1.x - e1 * d2.x) / determinant
            }
            for offset in 0..<3 {
                let normal = detail == 0 ? faceNormal : simd_normalize(p[offset])
                var u = dPdu
                var v = dPdv
                if simd_length_squared(u - normal * simd_dot(normal, u)) < 1e-20 {
                    // Pole triangles: follow the azimuth and the meridian toward the south pole.
                    u = simd_cross(SIMD3(0, 1, 0), normal)
                    if simd_length_squared(u) < 1e-20 { u = SIMD3(0, 0, 1) }
                    v = simd_cross(normal, u) * -1
                }
                builder.add(position: p[offset], normal: normal, uv: t[offset], dPdu: u, dPdv: v)
            }
            let base = UInt32(builder.vertexCount - 3)
            builder.indices += [base, base + 1, base + 2]
        }
        return try builder.geometry()
    }

    private static func axisIndex(_ axis: SIMD3<Double>) -> Int {
        abs(axis.x) > 0.5 ? 0 : (abs(axis.y) > 0.5 ? 1 : 2)
    }

    private static func validateArc(start: Float, length: Float, maximum: Float) throws {
        guard start.isFinite, length.isFinite, length > 0, length <= maximum * (1 + 1e-6) else {
            throw PrimitiveError.angle
        }
    }

    private static func greatestCommonDivisor(_ a: Int, _ b: Int) -> Int {
        b == 0 ? a : greatestCommonDivisor(b, a % b)
    }

    public enum PrimitiveError: LocalizedError, Equatable {
        case size
        case segments
        case angle
        case knotWinding
        case degenerateFrame
        case polyhedronFaces

        public var errorDescription: String? {
            switch self {
            case .size: "Primitive sizes and radii must be finite and positive, within the documented relations."
            case .segments: "A primitive segment or detail count is outside its documented range."
            case .angle: "Primitive angles must be finite, with a positive sweep of at most one turn (half a turn for sphere colatitude)."
            case .knotWinding: "Torus knot windings p and q must be coprime integers from 1 to 64."
            case .degenerateFrame: "The torus knot curve has no stable frame at a sample."
            case .polyhedronFaces: "Polyhedron vertices must be finite and nonzero, and faces must index them and enclose area."
            }
        }
    }
}

/// Accumulates primitive vertices with analytic frames in double precision.
struct PrimitiveBuilder {
    private(set) var vertices: [MeshVertex] = []
    var indices: [UInt32] = []
    var vertexCount: Int { vertices.count }

    /// Adds a vertex. The tangent is dP/dU made perpendicular to the normal; its
    /// handedness makes cross(normal, tangent) × w point along dP/dV.
    @discardableResult
    mutating func add(
        position: SIMD3<Double>, normal: SIMD3<Double>, uv: SIMD2<Double>,
        dPdu: SIMD3<Double>, dPdv: SIMD3<Double>
    ) -> UInt32 {
        let n = simd_normalize(normal)
        var tangent = dPdu - n * simd_dot(n, dPdu)
        if simd_length_squared(tangent) < 1e-24 {
            // dP/dU vanishes at a pole; use the direction perpendicular to dP/dV.
            tangent = simd_cross(dPdv, n)
        }
        if simd_length_squared(tangent) < 1e-24 {
            tangent = abs(n.x) < 0.9 ? simd_cross(n, SIMD3(1, 0, 0)) : simd_cross(n, SIMD3(0, 1, 0))
        }
        tangent = simd_normalize(tangent)
        let handedness: Float = simd_dot(simd_cross(n, tangent), dPdv) < 0 ? -1 : 1
        vertices.append(MeshVertex(
            position: SIMD3<Float>(position), normal: SIMD3<Float>(n),
            textureCoordinate: SIMD2<Float>(uv), tangent: SIMD4(SIMD3<Float>(tangent), handedness)
        ))
        return UInt32(vertices.count - 1)
    }

    /// Two counterclockwise triangles per cell of a row-major grid whose rows run
    /// down the face (increasing V) and columns along increasing U.
    mutating func addGrid(base: Int, columns: Int, rows: Int) {
        let stride = UInt32(columns + 1)
        for row in 0..<UInt32(rows) {
            for column in 0..<UInt32(columns) {
                let a = UInt32(base) + row * stride + column
                let b = a + stride
                // Three.js buildPlane: (a, b, d) and (b, c, d) with d = a + 1 and c = b + 1.
                indices += [a, b, a + 1, b, b + 1, a + 1]
            }
        }
    }

    func geometry() throws -> MeshGeometry {
        try MeshGeometry(vertices: vertices, indices: indices)
    }
}
