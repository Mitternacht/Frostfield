import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite("Color lookup tables")
struct ColorLookupTableParsingTests {
    @Test("A .cube file parses its title, domain and red-fastest data")
    func cube() throws {
        let text = """
        # Created by a test
        TITLE "Warm look"
        LUT_3D_SIZE 2
        DOMAIN_MIN 0 0 0
        DOMAIN_MAX 1 2 1

        0.0 0.0 0.0
        1.0 0.0 0.0   # red index 1
        0.0 1.0 0.0
        1.0 1.0 0.0
        0.0 0.0 1.0
        1.0 0.0 1.0
        0.0 1.0 1.0
        1.0 1.0 1.0
        """
        let table = try ColorLookupTable(cube: text)
        #expect(table.size == 2)
        #expect(table.title == "Warm look")
        #expect(table.domainMaximum == SIMD3(1, 2, 1))
        #expect(table.values[1] == SIMD3(1, 0, 0))
        #expect(table.values[2] == SIMD3(0, 1, 0))
        #expect(table.values[4] == SIMD3(0, 0, 1))
        // The domain maps green 2 to the top lattice point.
        #expect(table.lookup(SIMD3(0, 2, 0)) == SIMD3(0, 1, 0))
        #expect(simd_reduce_max(abs(table.lookup(SIMD3(0.5, 1, 0.5)) - SIMD3(0.5, 0.5, 0.5))) < 1e-6)
    }

    @Test("Malformed .cube files are rejected with the reason")
    func cubeErrors() {
        #expect(throws: ColorLookupTable.Failure.oneDimensional) { try ColorLookupTable(cube: "LUT_1D_SIZE 4\n0 0 0") }
        #expect(throws: ColorLookupTable.Failure.size(66)) { try ColorLookupTable(cube: "LUT_3D_SIZE 66") }
        #expect(throws: ColorLookupTable.Failure.valueCount(expected: 8, actual: 1)) {
            try ColorLookupTable(cube: "LUT_3D_SIZE 2\n0 0 0")
        }
        #expect(throws: ColorLookupTable.Failure.syntax(line: 3, "Keywords must precede the table data.")) {
            try ColorLookupTable(cube: "LUT_3D_SIZE 2\n0 0 0\nTITLE \"late\"")
        }
        #expect(throws: ColorLookupTable.Failure.syntax(line: 2, "Expected three finite numbers.")) {
            try ColorLookupTable(cube: "LUT_3D_SIZE 2\n0 nan 0")
        }
    }

    @Test("A .3dl table reorders blue-fastest integers and normalizes by its bit depth")
    func threeDL() throws {
        // 2 × 2 × 2 lattice, 10-bit output: value index = blue + 2 · (green + 2 · red).
        var lines = ["0 1023"]
        for red in 0..<2 {
            for green in 0..<2 {
                for blue in 0..<2 {
                    lines.append("\(red * 1023) \(green * 512) \(blue * 1000)")
                }
            }
        }
        let table = try ColorLookupTable(threeDL: lines.joined(separator: "\n"))
        #expect(table.size == 2)
        for red in 0..<2 {
            for green in 0..<2 {
                for blue in 0..<2 {
                    let expected = SIMD3(Float(red * 1023), Float(green * 512), Float(blue * 1000)) / 1023
                    #expect(table.values[red + 2 * (green + 2 * blue)] == expected)
                }
            }
        }
        // A Mesh header fixes 12-bit output even when values are small.
        let header = try ColorLookupTable(threeDL: "Mesh 1 12\n" + lines.joined(separator: "\n"))
        #expect(header.values[1] == SIMD3<Float>(1023, 0, 0) / 4095)
    }

    @Test(".3dl lattices must be evenly spaced from zero, with the last position at most one code short")
    func threeDLLattice() throws {
        func table(lattice: [Int]) throws -> ColorLookupTable {
            let size = lattice.count
            var lines = [lattice.map(String.init).joined(separator: " ")]
            for red in 0..<size {
                for green in 0..<size {
                    for blue in 0..<size { lines.append("\(lattice[red]) \(lattice[green]) \(lattice[blue])") }
                }
            }
            return try ColorLookupTable(threeDL: lines.joined(separator: "\n"))
        }
        // The common Lustre lattices: 17 points to 1023 and 33 points to 4095, plus exact steps.
        let seventeen = try table(lattice: (0..<16).map { $0 * 64 } + [1023])
        #expect(seventeen.size == 17)
        #expect(try table(lattice: (0..<32).map { $0 * 128 } + [4095]).size == 33)
        #expect(try table(lattice: [0, 512, 1024]).size == 3)
        // An identity-valued table looks up its own lattice inputs.
        #expect(simd_reduce_max(abs(seventeen.lookup(SIMD3(0.25, 0.5, 0.75)) - SIMD3(256, 512, 768) / 1023)) < 1e-6)
        for lattice in [[0, 100, 1023], [10, 20, 30], [0, 64, 128, 1000], [0, 0, 0], [0, 1, 1]] {
            #expect(throws: ColorLookupTable.Failure.nonuniformLattice, "\(lattice)") { try table(lattice: lattice) }
        }
    }

    @Test("Tetrahedral and trilinear interpolation agree at lattice points and on separable tables")
    func interpolationReference() throws {
        let curve = try ColorLookupTable(size: 9) { c in SIMD3(c.x * c.x, c.y.squareRoot(), 1 - c.z) }
        for point in [SIMD3<Float>(0.25, 0.5, 0.75), SIMD3(0.125, 0.875, 0.375)] {
            let difference = curve.lookup(point, interpolation: .tetrahedral) - curve.lookup(point, interpolation: .trilinear)
            #expect(simd_reduce_max(abs(difference)) < 1e-6)
        }
        // Between lattice points a cross term separates the two schemes.
        let mixed = try ColorLookupTable(size: 2) { c in SIMD3(repeating: c.x * c.y * c.z) }
        let center = SIMD3<Float>(repeating: 0.5)
        #expect(abs(mixed.lookup(center, interpolation: .trilinear).x - 0.125) < 1e-6)
        #expect(abs(mixed.lookup(center, interpolation: .tetrahedral).x - 0.5) < 1e-6)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ColorLookupPassTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    private static func frame(_ x: Int, _ y: Int) -> SIMD4<Float> {
        let alpha: Float = (x + y) % 3 == 0 ? 0.5 : 1
        let straight = SIMD3(Float(x) / 7, Float(y) / 5, Float((x * 3 + y * 5) % 11) / 10)
        return SIMD4(straight * alpha, alpha)
    }

    private func apply(
        _ table: ColorLookupTable, settings: ColorLookupPass.Settings, width: Int, height: Int,
        pixel: @escaping (Int, Int) -> SIMD4<Float>
    ) throws -> [SIMD4<Float>] {
        let pass = try ColorLookupPass(device: gpu.device, table: table, settings: settings)
        let source = try gpu.color(width: width, height: height, pixel: pixel)
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        return try gpu.read(destination)
    }

    @Test("An identity table leaves the frame unchanged in every mode", arguments: [
        ColorLookupPass.Settings(interpolation: .tetrahedral, inputEncoding: .sRGB),
        ColorLookupPass.Settings(interpolation: .trilinear, inputEncoding: .sRGB),
        ColorLookupPass.Settings(interpolation: .tetrahedral, inputEncoding: .linear),
        ColorLookupPass.Settings(interpolation: .trilinear, inputEncoding: .linear),
    ])
    func identity(settings: ColorLookupPass.Settings) throws {
        let table = try ColorLookupTable.identity(size: 33)
        let result = try apply(table, settings: settings, width: 8, height: 6, pixel: Self.frame)
        for (index, pixel) in result.enumerated() {
            let input = Self.frame(index % 8, index / 8)
            #expect(PostProcessFixture.close(pixel, input, tolerance: 1.5e-3), "pixel \(index): \(pixel) ≠ \(input)")
        }
    }

    @Test("A known table maps lattice inputs exactly to its values", arguments: [
        ColorLookupTable.Interpolation.tetrahedral, .trilinear,
    ])
    func latticePoints(interpolation: ColorLookupTable.Interpolation) throws {
        let size = 5
        let table = try ColorLookupTable(size: size) { c in SIMD3(c.y * c.y, 1 - c.x, 0.25 + 0.5 * c.z * c.x) }
        let width = size * size
        let result = try apply(
            table, settings: .init(interpolation: interpolation, inputEncoding: .linear), width: width, height: size
        ) { x, y in
            SIMD4(Float(x % size) / 4, Float(x / size) / 4, Float(y) / 4, 1)
        }
        for (index, pixel) in result.enumerated() {
            let (x, y) = (index % width, index / width)
            let lattice = table.values[(x % size) + size * ((x / size) + size * y)]
            #expect(PostProcessFixture.close(pixel, SIMD4(lattice, 1), tolerance: 1e-3), "(\(x), \(y)): \(pixel) ≠ \(lattice)")
        }
    }

    @Test("Off-lattice colors follow the CPU reference, and intensity mixes with the original")
    func offLattice() throws {
        let table = try ColorLookupTable(size: 3) { c in SIMD3(c.x * c.y, c.y * c.z + 0.1, c.z * c.x) }
        let inputs: [SIMD3<Float>] = [SIMD3(0.3, 0.6, 0.9), SIMD3(0.8, 0.2, 0.55), SIMD3(0.45, 0.7, 0.15)]
        for interpolation in [ColorLookupTable.Interpolation.tetrahedral, .trilinear] {
            let result = try apply(
                table, settings: .init(interpolation: interpolation, inputEncoding: .linear), width: 3, height: 1
            ) { x, _ in SIMD4(inputs[x], 1) }
            for (index, pixel) in result.enumerated() {
                let expected = table.lookup(inputs[index], interpolation: interpolation)
                #expect(PostProcessFixture.close(pixel, SIMD4(expected, 1), tolerance: 2e-3), "\(interpolation) \(index)")
            }
        }
        let tetra = table.lookup(inputs[0], interpolation: .tetrahedral)
        let tri = table.lookup(inputs[0], interpolation: .trilinear)
        #expect(abs(tetra.x - tri.x) + abs(tetra.y - tri.y) + abs(tetra.z - tri.z) > 0.01)

        let half = try apply(
            table, settings: .init(intensity: 0.5, interpolation: .tetrahedral, inputEncoding: .linear), width: 1, height: 1
        ) { _, _ in SIMD4(inputs[0], 1) }
        #expect(PostProcessFixture.close(half[0], SIMD4((inputs[0] + tetra) / 2, 1), tolerance: 2e-3))
    }

    @Test("Invalid intensity is rejected before encoding")
    func invalidIntensity() throws {
        let pass = try ColorLookupPass(device: gpu.device, table: try .identity(size: 2), settings: .init(intensity: 2))
        let source = try gpu.color(width: 2, height: 2)
        let destination = try gpu.color(width: 2, height: 2)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.invalidSettings("Lookup intensity must be from 0 to 1.")) {
            try pass.encode(source: source, destination: destination, commandBuffer: buffer)
        }
    }
}
