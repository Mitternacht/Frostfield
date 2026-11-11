import Foundation
import Frostfield
import Metal
import simd

/// Generated meshes, textures and sky for the Scale Lab. Everything is computed
/// from closed-form functions, so every run produces the same scene.
enum LabScaleGeometry {
    /// The three field object kinds, each prepared at three levels of detail.
    enum Kind: Int, CaseIterable {
        case rock
        case crystal
        case tree
    }

    /// A deterministic value in `0..<1` for an index and a salt (SplitMix64).
    static func random(_ index: Int, _ salt: UInt64) -> Float {
        var z = UInt64(truncatingIfNeeded: index) &* 0x9E37_79B9_7F4A_7C15 &+ salt &* 0xD1B5_4A32_D192_ED03
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 24)
    }

    // MARK: - Field meshes

    /// The mesh for one kind at one level; level 0 is the finest.
    static func field(_ kind: Kind, level: Int) throws -> MeshGeometry {
        switch (kind, level) {
        case (.rock, 0): try rock(segments: 24, rings: 14)
        case (.rock, 1): try rock(segments: 10, rings: 6)
        case (.rock, _): try rock(segments: 5, rings: 3)
        case (.crystal, 0): try crystals(count: 6, sides: 7, bands: 2)
        case (.crystal, 1): try crystals(count: 3, sides: 5, bands: 1)
        case (.crystal, _): try crystals(count: 2, sides: 3, bands: 1)
        case (.tree, 0): try tree(segments: 14, tiers: 3, trunkSegments: 8)
        case (.tree, 1): try tree(segments: 7, tiers: 1, trunkSegments: 4)
        case (.tree, _): try tree(segments: 4, tiers: 1, trunkSegments: 3)
        }
    }

    /// A flattened, lumpy sphere. The displacement depends only on direction, so
    /// every level approximates the same shape.
    static func rock(segments: Int, rings: Int) throws -> MeshGeometry {
        let sphere = try MeshGeometry.uvSphere(radius: 0.5, segments: segments, rings: rings)
        let positions = sphere.vertices.map { vertex -> SIMD3<Float> in
            let d = simd_normalize(vertex.position)
            let bump = 0.16 * sin(d.x * 5.1 + d.y * 2.3 + 1.7) + 0.11 * sin(d.z * 6.7 - d.x * 3.1 + 0.4)
                + 0.07 * sin((d.y + d.z) * 9.3 + 2.2)
            var p = d * 0.5 * (1 + bump)
            p.y = p.y * 0.62 + 0.1
            return p
        }
        return try flatShaded(positions: positions, coordinates: sphere.vertices.map(\.textureCoordinate), indices: sphere.indices)
    }

    /// A cluster of tapered prisms with pointed tips, open at the buried base.
    static func crystals(count: Int, sides: Int, bands: Int) throws -> MeshGeometry {
        var positions: [SIMD3<Float>] = []
        var coordinates: [SIMD2<Float>] = []
        var indices: [UInt32] = []
        for crystal in 0..<count {
            // The first crystal stands upright; the others lean outward around it.
            let around = Float(crystal) * 2.4
            let lean: Float = crystal == 0 ? 0 : 0.28 + 0.1 * Float(crystal % 2)
            let scale: Float = crystal == 0 ? 1 : 0.62 - 0.05 * Float(crystal % 3)
            let offset = crystal == 0 ? SIMD3<Float>.zero : SIMD3(cos(around), 0, sin(around)) * 0.16
            let tilt = simd_quatf(angle: lean, axis: SIMD3(sin(around), 0, -cos(around)))
            let radius: Float = 0.13 * scale
            let height: Float = 0.75 * scale
            let tip: Float = 0.28 * scale
            let base = UInt32(positions.count)
            func place(_ p: SIMD3<Float>) -> SIMD3<Float> { tilt.act(p) + offset + SIMD3(0, -0.12, 0) }
            for band in 0...bands {
                let t = Float(band) / Float(bands)
                let ring = radius * (1 - 0.18 * t)
                for side in 0..<sides {
                    let angle = Float(side) / Float(sides) * 2 * .pi
                    positions.append(place(SIMD3(ring * cos(angle), height * t, ring * sin(angle))))
                    coordinates.append(SIMD2(Float(side) / Float(sides), 1 - t))
                }
            }
            positions.append(place(SIMD3(0, height + tip, 0)))
            coordinates.append(SIMD2(0.5, 0))
            let apex = UInt32(positions.count - 1)
            func vertex(_ band: Int, _ side: Int) -> UInt32 { base + UInt32(band * sides + side % sides) }
            for band in 0..<bands {
                for side in 0..<sides {
                    let a = vertex(band, side), b = vertex(band, side + 1)
                    let c = vertex(band + 1, side + 1), d = vertex(band + 1, side)
                    indices += [a, c, b, a, d, c]
                }
            }
            for side in 0..<sides {
                indices += [vertex(bands, side), apex, vertex(bands, side + 1)]
            }
        }
        return try flatShaded(positions: positions, coordinates: coordinates, indices: indices)
    }

    /// A conifer: a trunk and tiered foliage. Both parts sample a two-color atlas.
    static func tree(segments: Int, tiers: Int, trunkSegments: Int) throws -> MeshGeometry {
        let trunk = try MeshGeometry.lathe(profile: [
            LatheProfilePoint(radius: 0.12, y: -0.05), LatheProfilePoint(radius: 0.08, y: 0.9),
        ], segments: trunkSegments, capBottom: false, capTop: false)
        let profile: [LatheProfilePoint]
        if tiers >= 3 {
            profile = [
                LatheProfilePoint(radius: 0.95, y: 0.7, hard: true), LatheProfilePoint(radius: 0.42, y: 1.45, hard: true),
                LatheProfilePoint(radius: 0.78, y: 1.45, hard: true), LatheProfilePoint(radius: 0.3, y: 2.2, hard: true),
                LatheProfilePoint(radius: 0.58, y: 2.2, hard: true), LatheProfilePoint(radius: 0, y: 3.1),
            ]
        } else {
            profile = [LatheProfilePoint(radius: 0.9, y: 0.6, hard: true), LatheProfilePoint(radius: 0, y: 3.1)]
        }
        let foliage = try MeshGeometry.lathe(profile: profile, segments: segments, capBottom: true, capTop: false)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for (part, coordinate) in [(trunk, SIMD2<Float>(0.25, 0.5)), (foliage, SIMD2<Float>(0.75, 0.5))] {
            let base = UInt32(vertices.count)
            vertices += part.vertices.map { MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: coordinate) }
            indices += part.indices.map { $0 + base }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// Splits every triangle into its own three vertices with the face normal.
    static func flatShaded(positions: [SIMD3<Float>], coordinates: [SIMD2<Float>], indices: [UInt32]) throws -> MeshGeometry {
        var vertices: [MeshVertex] = []
        var faces: [UInt32] = []
        vertices.reserveCapacity(indices.count)
        faces.reserveCapacity(indices.count)
        for start in stride(from: 0, to: indices.count, by: 3) {
            let corners = (0..<3).map { Int(indices[start + $0]) }
            let a = positions[corners[0]], b = positions[corners[1]], c = positions[corners[2]]
            let cross = simd_cross(b - a, c - a)
            let length = simd_length(cross)
            guard length > 1e-8 else { continue }
            let base = UInt32(vertices.count)
            for corner in corners {
                vertices.append(MeshVertex(position: positions[corner], normal: cross / length, textureCoordinate: coordinates[corner]))
            }
            faces += [base, base + 1, base + 2]
        }
        return try MeshGeometry(vertices: vertices, indices: faces)
    }

    // MARK: - Hero meshes

    /// A disc in the XY plane facing +Z, with planar UVs on both faces centered at (0.5, 0.5).
    static func disc(radius: Float, thickness: Float, segments: Int) throws -> MeshGeometry {
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        let half = thickness / 2
        for face: Float in [1, -1] {
            let center = UInt32(vertices.count)
            vertices.append(MeshVertex(position: SIMD3(0, 0, half * face), normal: SIMD3(0, 0, face), textureCoordinate: SIMD2(0.5, 0.5)))
            for segment in 0...segments {
                let angle = Float(segment) / Float(segments) * 2 * .pi
                let x = cos(angle), y = sin(angle)
                vertices.append(MeshVertex(
                    position: SIMD3(x * radius, y * radius, half * face), normal: SIMD3(0, 0, face),
                    textureCoordinate: SIMD2(0.5 + 0.5 * x * face, 0.5 - 0.5 * y)
                ))
            }
            for segment in 0..<UInt32(segments) {
                let a = center + 1 + segment
                indices += face > 0 ? [center, a, a + 1] : [center, a + 1, a]
            }
        }
        let rim = UInt32(vertices.count)
        for segment in 0...segments {
            let angle = Float(segment) / Float(segments) * 2 * .pi
            let normal = SIMD3(cos(angle), sin(angle), 0)
            let u = Float(segment) / Float(segments)
            vertices.append(MeshVertex(position: normal * radius + SIMD3(0, 0, half), normal: normal, textureCoordinate: SIMD2(u, 0)))
            vertices.append(MeshVertex(position: normal * radius - SIMD3(0, 0, half), normal: normal, textureCoordinate: SIMD2(u, 1)))
        }
        for segment in 0..<UInt32(segments) {
            let a = rim + 2 * segment, d = a + 1, b = a + 2, c = a + 3
            indices += [a, d, b, b, d, c]
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// An upright triangular prism: an equilateral cross-section in XZ and height along Y.
    static func prism(side: Float, height: Float) throws -> MeshGeometry {
        let r = side / sqrt(3)
        let contour = (0..<3).map { index -> SIMD2<Float> in
            let angle = Float.pi / 2 + Float(index) * 2 * .pi / 3
            return SIMD2(r * cos(angle), r * sin(angle))
        }
        let extruded = try MeshGeometry.extrude(contour: contour, depth: height)
        // Turn the extrusion axis (Z) to Y: (x, y, z) -> (x, z, -y).
        let turn = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        return try MeshGeometry(vertices: extruded.vertices.map {
            MeshVertex(position: turn.act($0.position), normal: turn.act($0.normal), textureCoordinate: $0.textureCoordinate)
        }, indices: extruded.indices)
    }

    // MARK: - Textures

    /// An RGBA8 texture from encoded channel values in 0...1, with a mip chain when requested.
    static func texture(
        queue: any MTLCommandQueue, size: SIMD2<Int>, sRGB: Bool, mipmapped: Bool, label: String,
        texel: (_ u: Float, _ v: Float) -> SIMD4<Float>
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: sRGB ? .rgba8Unorm_srgb : .rgba8Unorm, width: size.x, height: size.y, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = queue.device.makeTexture(descriptor: descriptor) else { throw MeshRenderer.Failure.resource(label) }
        texture.label = label
        var bytes = [UInt8](repeating: 0, count: size.x * size.y * 4)
        for y in 0..<size.y {
            for x in 0..<size.x {
                let value = simd_clamp(texel((Float(x) + 0.5) / Float(size.x), (Float(y) + 0.5) / Float(size.y)), .zero, SIMD4(repeating: 1))
                for channel in 0..<4 {
                    bytes[(y * size.x + x) * 4 + channel] = UInt8((value[channel] * 255).rounded())
                }
            }
        }
        texture.replace(region: MTLRegionMake2D(0, 0, size.x, size.y), mipmapLevel: 0, withBytes: bytes, bytesPerRow: size.x * 4)
        if mipmapped {
            guard let buffer = queue.makeCommandBuffer(), let blit = buffer.makeBlitCommandEncoder() else {
                throw MeshRenderer.Failure.resource("\(label) mipmaps")
            }
            blit.generateMipmaps(for: texture)
            blit.endEncoding()
            buffer.commit()
            buffer.waitUntilCompleted()
        }
        return texture
    }

    /// A periodic sum of sines, so the pattern tiles across the texture edges.
    static func ripple(_ u: Float, _ v: Float, _ terms: [(Float, Float, Float)]) -> Float {
        terms.reduce(0) { $0 + sin(2 * .pi * ($1.0 * u + $1.1 * v) + $1.2) } / Float(terms.count)
    }

    /// A woven plaid in sRGB: navy ground, berry and gold stripes, a fine twill.
    static func plaid(queue: any MTLCommandQueue) throws -> any MTLTexture {
        try texture(queue: queue, size: SIMD2(256, 256), sRGB: true, mipmapped: true, label: "Scale Lab plaid") { u, v in
            func stripes(_ t: Float) -> (Float, Float) {
                let wide = abs((t * 4).truncatingRemainder(dividingBy: 1) - 0.5) < 0.14 ? Float(1) : 0
                let thin = abs((t * 4 + 0.5).truncatingRemainder(dividingBy: 1) - 0.5) < 0.025 ? Float(1) : 0
                return (wide, thin)
            }
            let (wideU, thinU) = stripes(u)
            let (wideV, thinV) = stripes(v)
            var color = SIMD3<Float>(0.16, 0.18, 0.36)
            color = simd_mix(color, SIMD3(0.55, 0.13, 0.25), SIMD3(repeating: 0.5 * (wideU + wideV)))
            color = simd_mix(color, SIMD3(0.86, 0.68, 0.25), SIMD3(repeating: max(thinU, thinV)))
            let twill = ((Int(u * 256) + Int(v * 256)) / 2) % 2 == 0 ? Float(1.06) : 0.94
            return SIMD4(color * twill, 1)
        }
    }

    /// A checkerboard in sRGB with an orange cross, so tiling and rotation are easy to read.
    static func checker(queue: any MTLCommandQueue) throws -> any MTLTexture {
        try texture(queue: queue, size: SIMD2(128, 128), sRGB: true, mipmapped: true, label: "Scale Lab checker") { u, v in
            if abs(u - 0.5) < 0.035 || abs(v - 0.5) < 0.035 { return SIMD4(0.98, 0.52, 0.12, 1) }
            let even = (Int(u * 4) + Int(v * 4)) % 2 == 0
            return even ? SIMD4(0.93, 0.9, 0.82, 1) : SIMD4(0.08, 0.45, 0.5, 1)
        }
    }

    /// Mottled grass and soil in sRGB that tiles.
    static func ground(queue: any MTLCommandQueue) throws -> any MTLTexture {
        try texture(queue: queue, size: SIMD2(256, 256), sRGB: true, mipmapped: true, label: "Scale Lab ground") { u, v in
            let broad = ripple(u, v, [(1, 2, 0.3), (3, -1, 1.9), (-2, 3, 4.1), (5, 4, 2.6)])
            let fine = ripple(u, v, [(17, 9, 0.7), (-13, 21, 2.2), (29, -23, 5.0)])
            let mix = simd_clamp(0.5 + broad * 0.6 + fine * 0.2, 0, 1)
            let grass = SIMD3<Float>(0.36, 0.42, 0.22)
            let soil = SIMD3<Float>(0.45, 0.38, 0.27)
            return SIMD4(simd_mix(grass, soil, SIMD3(repeating: mix)) * (0.95 + 0.1 * fine), 1)
        }
    }

    /// Bark on the left half, needles on the right, sampled at constant UVs.
    static func treeAtlas(queue: any MTLCommandQueue) throws -> any MTLTexture {
        try texture(queue: queue, size: SIMD2(8, 1), sRGB: true, mipmapped: false, label: "Scale Lab tree atlas") { u, _ in
            u < 0.5 ? SIMD4(0.36, 0.25, 0.17, 1) : SIMD4(0.2, 0.42, 0.24, 1)
        }
    }

    /// Linear green swirls for the iridescence thickness map.
    static func filmThickness(queue: any MTLCommandQueue) throws -> any MTLTexture {
        try texture(queue: queue, size: SIMD2(256, 128), sRGB: false, mipmapped: true, label: "Scale Lab film thickness") { u, v in
            let swirl = sin(2 * .pi * (3 * u + 0.6 * sin(2 * .pi * (2 * v + u)))) * 0.5 + 0.5
            return SIMD4(1, simd_clamp(0.15 + 0.85 * swirl, 0, 1), 0, 1)
        }
    }

    /// Linear tangent-space directions around the UV center: circular brushing.
    static func brushDirections(queue: any MTLCommandQueue) throws -> any MTLTexture {
        try texture(queue: queue, size: SIMD2(128, 128), sRGB: false, mipmapped: false, label: "Scale Lab brush directions") { u, v in
            let offset = SIMD2(u - 0.5, v - 0.5)
            let length = simd_length(offset)
            let direction = length > 1e-4 ? SIMD2(-offset.y, offset.x) / length : SIMD2(1, 0)
            return SIMD4(direction.x * 0.5 + 0.5, direction.y * 0.5 + 0.5, 1, 1)
        }
    }

    // MARK: - Sky

    static let skyWidth = 512
    static let skyHeight = 256

    /// A clear-afternoon sky in linear RGB, row-major from the top-left in the
    /// raster equirectangular orientation, with soft cloud bands and a dim ground.
    static func sky() -> [SIMD3<Float>] {
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(skyWidth * skyHeight)
        for y in 0..<skyHeight {
            let latitude = (0.5 - (Float(y) + 0.5) / Float(skyHeight)) * .pi
            let up = sin(latitude)
            for x in 0..<skyWidth {
                let azimuth = ((Float(x) + 0.5) / Float(skyWidth) - 0.5) * (2 * .pi)
                if up < 0 {
                    let fade = min(1, -up * 6)
                    pixels.append(simd_mix(SIMD3(0.62, 0.66, 0.7), SIMD3(0.16, 0.15, 0.13), SIMD3(repeating: fade)))
                    continue
                }
                let height = pow(up, 0.5)
                var color = simd_mix(SIMD3<Float>(0.78, 0.84, 0.92), SIMD3(0.18, 0.34, 0.72), SIMD3(repeating: height))
                let cloud = ripple(azimuth / (2 * .pi), up, [(3, 4, 0.5), (5, -7, 2.1), (8, 11, 4.4), (13, -5, 1.2)])
                let band = max(0, cloud * 1.8 - 0.25) * max(0, 1 - abs(up - 0.25) * 3.2)
                color = simd_mix(color, SIMD3(1.05, 1.05, 1.08), SIMD3(repeating: min(band, 0.8)))
                pixels.append(color)
            }
        }
        return pixels
    }

    static func skyLighting(queue: any MTLCommandQueue) throws -> EnvironmentLighting {
        let pixels = sky()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: skyWidth, height: skyHeight, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = queue.device.makeTexture(descriptor: descriptor) else { throw MeshRenderer.Failure.resource("Scale Lab sky") }
        let halves = pixels.map { SIMD4<Float16>(SIMD3<Float16>($0), 1) }
        halves.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, skyWidth, skyHeight), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: skyWidth * MemoryLayout<SIMD4<Float16>>.stride
            )
        }
        return try EnvironmentLighting(queue: queue, equirectangular: texture)
    }

    static func pathSky(device: any MTLDevice) throws -> PathTracer.EnvironmentMap {
        try PathTracer.EnvironmentMap(device: device, pixels: sky(), width: skyWidth, height: skyHeight)
    }
}
