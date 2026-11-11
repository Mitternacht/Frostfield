import Metal
import simd

extension VOXLoader.Model {
    /// Uploads occupancy as R8Unorm in the original X-fastest Z-up grid.
    /// Occupied cells sample as one and empty cells as zero.
    public func makeOccupancyTexture(device: any MTLDevice) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .r8Unorm
        descriptor.width = size.x
        descriptor.height = size.y
        descriptor.depth = size.z
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw VOXLoader.Failure.textureAllocation }

        texture.label = "Frostfield VOX occupancy"
        let values = voxelGrid(occupancyOnly: true)
        values.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake3D(0, 0, 0, size.x, size.y, size.z), mipmapLevel: 0, slice: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: size.x, bytesPerImage: size.x * size.y
            )
        }
        return texture
    }

    private func voxelGrid(occupancyOnly: Bool) -> [UInt8] {
        var values = [UInt8](repeating: 0, count: size.x * size.y * size.z)
        for voxel in voxels {
            let point = voxel.position
            values[Int(point.x) + size.x * (Int(point.y) + size.y * Int(point.z))] = occupancyOnly ? 255 : voxel.paletteIndex
        }
        return values
    }

    /// Culls faces between any two occupied cells and greedily merges only
    /// coplanar, same-palette faces with the same outward normal.
    func makeMesh(palette: [SIMD4<UInt8>], maximumVertices: Int) throws -> MeshGeometry {
        let grid = voxelGrid(occupancyOnly: false)
        let steps = SIMD3(1, size.x, size.x * size.y)
        let halfSize = SIMD3<Float>(size) * 0.5
        let colors = palette.map { rgba in
            SIMD4(
                GLTFColorTables.sRGBToLinear(Float(rgba.x) / 255),
                GLTFColorTables.sRGBToLinear(Float(rgba.y) / 255),
                GLTFColorTables.sRGBToLinear(Float(rgba.z) / 255),
                1
            )
        }
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []

        for axis in 0..<3 {
            let u = (axis + 1) % 3, v = (axis + 2) % 3
            let width = size[u], height = size[v]
            var mask = [Int16](repeating: 0, count: width * height)
            for slice in 0...size[axis] {
                for y in 0..<height {
                    for x in 0..<width {
                        let cell = slice * steps[axis] + x * steps[u] + y * steps[v]
                        let behind = slice > 0 ? grid[cell - steps[axis]] : 0
                        let ahead = slice < size[axis] ? grid[cell] : 0
                        mask[x + width * y] = behind == 0 ? -Int16(ahead) : (ahead == 0 ? Int16(behind) : 0)
                    }
                }

                for y in 0..<height {
                    var x = 0
                    while x < width {
                        let origin = x + width * y
                        let face = mask[origin]
                        guard face != 0 else {
                            x += 1
                            continue
                        }
                        var run = 1
                        while x + run < width, mask[origin + run] == face { run += 1 }
                        var rows = 1
                        while y + rows < height {
                            let start = origin + width * rows
                            guard mask[start..<(start + run)].allSatisfy({ $0 == face }) else { break }
                            rows += 1
                        }

                        guard vertices.count + 4 <= maximumVertices else { throw VOXLoader.Failure.meshLimit }
                        let base = UInt32(vertices.count)
                        var corner = SIMD3<Int>.zero, edgeU = SIMD3<Int>.zero, edgeV = SIMD3<Int>.zero
                        corner[axis] = slice
                        corner[u] = x
                        corner[v] = y
                        edgeU[u] = run
                        edgeV[v] = rows
                        let second = face > 0 ? edgeU : edgeV
                        let fourth = face > 0 ? edgeV : edgeU
                        var normal = SIMD3<Int>.zero
                        normal[axis] = face > 0 ? 1 : -1
                        let orientedNormal = SIMD3(Float(normal.x), Float(normal.z), Float(-normal.y))
                        let color = colors[Int(abs(face))]
                        for offset in [SIMD3<Int>.zero, second, edgeU &+ edgeV, fourth] {
                            let point = SIMD3<Float>(corner &+ offset)
                            let position = SIMD3(point.x - halfSize.x, point.z - halfSize.z, halfSize.y - point.y)
                            vertices.append(MeshVertex(position: position, normal: orientedNormal, color: color))
                        }
                        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])

                        for row in 0..<rows {
                            let start = origin + width * row
                            for index in start..<(start + run) { mask[index] = 0 }
                        }
                        x += run
                    }
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }
}
