import Foundation
import simd

/// Line-by-line ports of the Three.js r187 geometry generators that Frostfield's
/// primitives follow (`src/geometries/*Geometry.js`), in Three.js conventions: UV
/// origin at the bottom left, so V points up. Tests compare the engine's output with
/// them vertex by vertex and index by index.
/// Copyright © 2010-2026 three.js authors. See the repository's MIT license.
enum ThreeGeometryReference {
    struct Mesh {
        var positions: [SIMD3<Double>] = []
        var normals: [SIMD3<Double>] = []
        var uvs: [SIMD2<Double>] = []
        var indices: [Int] = []
        var groups: [(start: Int, count: Int, materialIndex: Int)] = []

        mutating func add(_ position: SIMD3<Double>, _ normal: SIMD3<Double>, _ uv: SIMD2<Double>) {
            positions.append(position)
            normals.append(normal)
            uvs.append(uv)
        }
    }

    static func box(
        width: Double, height: Double, depth: Double, widthSegments: Int, heightSegments: Int, depthSegments: Int
    ) -> Mesh {
        var mesh = Mesh()
        var groupStart = 0
        // u, v, w are axis indices; udir and vdir flip them.
        func buildPlane(
            _ u: Int, _ v: Int, _ w: Int, _ udir: Double, _ vdir: Double, _ width: Double, _ height: Double,
            _ depth: Double, _ gridX: Int, _ gridY: Int, _ materialIndex: Int
        ) {
            let numberOfVertices = mesh.positions.count
            let segmentWidth = width / Double(gridX)
            let segmentHeight = height / Double(gridY)
            let gridX1 = gridX + 1
            for iy in 0...gridY {
                let y = Double(iy) * segmentHeight - height / 2
                for ix in 0...gridX {
                    let x = Double(ix) * segmentWidth - width / 2
                    var vector = SIMD3<Double>.zero
                    vector[u] = x * udir
                    vector[v] = y * vdir
                    vector[w] = depth / 2
                    var normal = SIMD3<Double>.zero
                    normal[w] = depth > 0 ? 1 : -1
                    mesh.add(vector, normal, SIMD2(Double(ix) / Double(gridX), 1 - Double(iy) / Double(gridY)))
                }
            }
            var groupCount = 0
            for iy in 0..<gridY {
                for ix in 0..<gridX {
                    let a = numberOfVertices + ix + gridX1 * iy
                    let b = numberOfVertices + ix + gridX1 * (iy + 1)
                    let c = numberOfVertices + (ix + 1) + gridX1 * (iy + 1)
                    let d = numberOfVertices + (ix + 1) + gridX1 * iy
                    mesh.indices += [a, b, d, b, c, d]
                    groupCount += 6
                }
            }
            mesh.groups.append((groupStart, groupCount, materialIndex))
            groupStart += groupCount
        }
        buildPlane(2, 1, 0, -1, -1, depth, height, width, depthSegments, heightSegments, 0)
        buildPlane(2, 1, 0, 1, -1, depth, height, -width, depthSegments, heightSegments, 1)
        buildPlane(0, 2, 1, 1, 1, width, depth, height, widthSegments, depthSegments, 2)
        buildPlane(0, 2, 1, 1, -1, width, depth, -height, widthSegments, depthSegments, 3)
        buildPlane(0, 1, 2, 1, -1, width, height, depth, widthSegments, heightSegments, 4)
        buildPlane(0, 1, 2, -1, -1, width, height, -depth, widthSegments, heightSegments, 5)
        return mesh
    }

    static func cylinder(
        radiusTop: Double, radiusBottom: Double, height: Double, radialSegments: Int, heightSegments: Int,
        openEnded: Bool, thetaStart: Double, thetaLength: Double
    ) -> Mesh {
        var mesh = Mesh()
        let halfHeight = height / 2
        var groupStart = 0
        var indexArray: [[Int]] = []
        let slope = (radiusBottom - radiusTop) / height
        for y in 0...heightSegments {
            var row: [Int] = []
            let v = Double(y) / Double(heightSegments)
            let radius = v * (radiusBottom - radiusTop) + radiusTop
            for x in 0...radialSegments {
                let u = Double(x) / Double(radialSegments)
                let theta = u * thetaLength + thetaStart
                row.append(mesh.positions.count)
                mesh.add(
                    SIMD3(radius * sin(theta), -v * height + halfHeight, radius * cos(theta)),
                    simd_normalize(SIMD3(sin(theta), slope, cos(theta))), SIMD2(u, 1 - v)
                )
            }
            indexArray.append(row)
        }
        var groupCount = 0
        for x in 0..<radialSegments {
            for y in 0..<heightSegments {
                let a = indexArray[y][x]
                let b = indexArray[y + 1][x]
                let c = indexArray[y + 1][x + 1]
                let d = indexArray[y][x + 1]
                if radiusTop > 0 || y != 0 { mesh.indices += [a, b, d]; groupCount += 3 }
                if radiusBottom > 0 || y != heightSegments - 1 { mesh.indices += [b, c, d]; groupCount += 3 }
            }
        }
        mesh.groups.append((groupStart, groupCount, 0))
        groupStart += groupCount
        func cap(_ top: Bool) {
            let centerStart = mesh.positions.count
            let radius = top ? radiusTop : radiusBottom
            let sign: Double = top ? 1 : -1
            for _ in 1...radialSegments {
                mesh.add(SIMD3(0, halfHeight * sign, 0), SIMD3(0, sign, 0), SIMD2(0.5, 0.5))
            }
            let centerEnd = mesh.positions.count
            for x in 0...radialSegments {
                let theta = Double(x) / Double(radialSegments) * thetaLength + thetaStart
                mesh.add(
                    SIMD3(radius * sin(theta), halfHeight * sign, radius * cos(theta)), SIMD3(0, sign, 0),
                    SIMD2(cos(theta) * 0.5 + 0.5, sin(theta) * 0.5 * sign + 0.5)
                )
            }
            for x in 0..<radialSegments {
                let c = centerStart + x
                let i = centerEnd + x
                mesh.indices += top ? [i, i + 1, c] : [i + 1, i, c]
            }
            mesh.groups.append((groupStart, 3 * radialSegments, top ? 1 : 2))
            groupStart += 3 * radialSegments
        }
        if !openEnded {
            if radiusTop > 0 { cap(true) }
            if radiusBottom > 0 { cap(false) }
        }
        return mesh
    }

    static func capsule(radius: Double, height: Double, capSegments: Int, radialSegments: Int, heightSegments: Int) -> Mesh {
        var mesh = Mesh()
        let halfHeight = height / 2
        let capArcLength = Double.pi / 2 * radius
        let totalArcLength = 2 * capArcLength + height
        let numVerticalSegments = capSegments * 2 + heightSegments
        let verticesPerRow = radialSegments + 1
        for iy in 0...numVerticalSegments {
            let currentArcLength: Double
            let profileY: Double
            let profileRadius: Double
            let normalYComponent: Double
            if iy <= capSegments {
                let progress = Double(iy) / Double(capSegments)
                let angle = progress * .pi / 2
                profileY = -halfHeight - radius * cos(angle)
                profileRadius = radius * sin(angle)
                normalYComponent = -radius * cos(angle)
                currentArcLength = progress * capArcLength
            } else if iy <= capSegments + heightSegments {
                let progress = Double(iy - capSegments) / Double(heightSegments)
                profileY = -halfHeight + progress * height
                profileRadius = radius
                normalYComponent = 0
                currentArcLength = capArcLength + progress * height
            } else {
                let progress = Double(iy - capSegments - heightSegments) / Double(capSegments)
                let angle = progress * .pi / 2
                profileY = halfHeight + radius * sin(angle)
                profileRadius = radius * cos(angle)
                normalYComponent = radius * sin(angle)
                currentArcLength = capArcLength + height + progress * capArcLength
            }
            let v = max(0, min(1, currentArcLength / totalArcLength))
            var uOffset = 0.0
            if iy == 0 { uOffset = 0.5 / Double(radialSegments) } else if iy == numVerticalSegments { uOffset = -0.5 / Double(radialSegments) }
            for ix in 0...radialSegments {
                let u = Double(ix) / Double(radialSegments)
                let theta = u * .pi * 2
                mesh.add(
                    SIMD3(-profileRadius * cos(theta), profileY, profileRadius * sin(theta)),
                    simd_normalize(SIMD3(-profileRadius * cos(theta), normalYComponent, profileRadius * sin(theta))),
                    SIMD2(u + uOffset, v)
                )
            }
            if iy > 0 {
                let previous = (iy - 1) * verticesPerRow
                for ix in 0..<radialSegments {
                    let i1 = previous + ix
                    let i2 = previous + ix + 1
                    let i3 = iy * verticesPerRow + ix
                    let i4 = iy * verticesPerRow + ix + 1
                    mesh.indices += [i1, i2, i3, i2, i4, i3]
                }
            }
        }
        return mesh
    }

    static func torus(
        radius: Double, tube: Double, radialSegments: Int, tubularSegments: Int, arc: Double,
        thetaStart: Double, thetaLength: Double
    ) -> Mesh {
        var mesh = Mesh()
        for j in 0...radialSegments {
            let v = thetaStart + Double(j) / Double(radialSegments) * thetaLength
            for i in 0...tubularSegments {
                let u = Double(i) / Double(tubularSegments) * arc
                let vertex = SIMD3((radius + tube * cos(v)) * cos(u), (radius + tube * cos(v)) * sin(u), tube * sin(v))
                let center = SIMD3(radius * cos(u), radius * sin(u), 0)
                mesh.add(
                    vertex, simd_normalize(vertex - center),
                    SIMD2(Double(i) / Double(tubularSegments), Double(j) / Double(radialSegments))
                )
            }
        }
        for j in 1...radialSegments {
            for i in 1...tubularSegments {
                let a = (tubularSegments + 1) * j + i - 1
                let b = (tubularSegments + 1) * (j - 1) + i - 1
                let c = (tubularSegments + 1) * (j - 1) + i
                let d = (tubularSegments + 1) * j + i
                mesh.indices += [a, b, d, b, c, d]
            }
        }
        return mesh
    }

    static func ring(
        innerRadius: Double, outerRadius: Double, thetaSegments: Int, phiSegments: Int, thetaStart: Double, thetaLength: Double
    ) -> Mesh {
        var mesh = Mesh()
        var radius = innerRadius
        let radiusStep = (outerRadius - innerRadius) / Double(phiSegments)
        for _ in 0...phiSegments {
            for i in 0...thetaSegments {
                let segment = thetaStart + Double(i) / Double(thetaSegments) * thetaLength
                let vertex = SIMD3(radius * cos(segment), radius * sin(segment), 0)
                mesh.add(vertex, SIMD3(0, 0, 1), SIMD2((vertex.x / outerRadius + 1) / 2, (vertex.y / outerRadius + 1) / 2))
            }
            radius += radiusStep
        }
        for j in 0..<phiSegments {
            let level = j * (thetaSegments + 1)
            for i in 0..<thetaSegments {
                let segment = i + level
                let a = segment
                let b = segment + thetaSegments + 1
                let c = segment + thetaSegments + 2
                let d = segment + 1
                mesh.indices += [a, b, d, b, c, d]
            }
        }
        return mesh
    }
}
