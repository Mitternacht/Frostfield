import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PDBLoaderTests {
    @Test("PDB preserves selected-model atom metadata and deduplicates serial-resolved bonds")
    func structure() throws {
        let first = Self.atom(
            serial: 10, name: " CA ", position: SIMD3(-2, 0, -4), residue: "GLY",
            chain: "A", sequence: -2, insertion: "B", alternate: "A", occupancy: 0.5, temperature: 12.75
        )
        let second = Self.atom(
            kind: "HETATM", serial: 42, name: " O  ", position: SIMD3(2, 0, -4), element: "O"
        )
        let next = Self.atom(serial: 10, name: " C  ", position: SIMD3(9, 9, 9), element: "C")
        let input = [
            "MODEL        3", first, second, Self.connect(42, [10, 10]), "ENDMDL",
            "MODEL        4", next, Self.connect(10, [333]), "ENDMDL",
            Self.connect(10, [42, 9_999, 10, 0]), "END",
        ].joined(separator: "\r\n")
        let molecule = try PDBLoader.load(input)
        #expect(molecule.modelNumber == 3 && molecule.atoms.count == 2)
        let atom = try #require(molecule.atoms.first)
        #expect(atom.kind == .atom && atom.serial == 10 && atom.name == "CA" && atom.element == "C")
        #expect(atom.residueName == "GLY" && atom.chain == "A" && atom.residueSequence == -2)
        #expect(atom.insertionCode == "B" && atom.alternateLocation == "A")
        #expect(atom.occupancy == 0.5 && atom.temperatureFactor == 12.75)
        #expect(molecule.atoms[1].kind == .heteroatom && molecule.atoms[1].element == "O")
        #expect(molecule.pointCloud.positions == [SIMD3(-2, 0, -4), SIMD3(2, 0, -4)])
        let colors = try #require(molecule.pointCloud.colors)
        #expect(isNearly(colors[0], SIMD4(0.27889, 0.27889, 0.27889, 1), tolerance: 0.0001))
        #expect(isNearly(colors[1], SIMD4(1, 0.00402, 0.00402, 1), tolerance: 0.0001))
        let bond = try #require(molecule.bonds.first)
        #expect(molecule.bonds.count == 1 && bond.first == 1 && bond.second == 0)
        let lines = try #require(molecule.bondGeometry)
        #expect(lines.topology == .segments && lines.positions == [SIMD3(2, 0, -4), SIMD3(-2, 0, -4)])
    }

    @Test("PDB accepts sparse serials and models without bonds")
    func standaloneAtoms() throws {
        let input = Self.atom(serial: 18, name: "CL  ", position: SIMD3(2, 1, 0))
            + "\n" + Self.atom(serial: 31, name: " X  ", position: SIMD3(4, 1, 0), element: "X")
        let molecule = try PDBLoader.load(input)
        #expect(molecule.modelNumber == nil && molecule.bondGeometry == nil)
        #expect(molecule.bonds.isEmpty && molecule.atoms.map(\.element) == ["Cl", "X"])
        let colors = try #require(molecule.pointCloud.colors)
        #expect(colors[0].y > 0.85 && colors[0].x < 0.02)
        #expect(colors[1].x == colors[1].y && colors[1].y == colors[1].z)
    }

    @Test("PDB rejects truncated coordinates, duplicate serials, malformed connectivity, and open models")
    func malformed() {
        let atom = Self.atom(serial: 5, name: " O  ", position: SIMD3(0, 0, -4))
        #expect(throws: PDBLoader.Failure.encoding) { try PDBLoader.load(Data([0xFF])) }
        #expect(throws: PDBLoader.Failure.empty) { try PDBLoader.load("HEADER EMPTY") }
        #expect(throws: PDBLoader.Failure.invalidAtom(1)) { try PDBLoader.load(String(atom.prefix(50))) }
        #expect(throws: PDBLoader.Failure.duplicateSerial(5)) { try PDBLoader.load(atom + "\n" + atom) }
        #expect(throws: PDBLoader.Failure.invalidConnectivity(2)) { try PDBLoader.load(atom + "\nCONECT    5    X") }
        #expect(throws: PDBLoader.Failure.invalidModel(1)) { try PDBLoader.load("MODEL        1\n" + atom) }
    }

    @Test("PDB atoms and CONECT segments reach Metal point and line renderers")
    func render() throws {
        let input = [
            Self.atom(serial: 2, name: " O  ", position: SIMD3(-2, 0, -4), element: "O"),
            Self.atom(serial: 9, name: " N  ", position: SIMD3(2, 0, -4), element: "N"),
            Self.connect(2, [9]),
        ].joined(separator: "\n")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("pdb")
        try Data(input.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let molecule = try PDBLoader.load(contentsOf: url)
        let fixture = try PrimitiveRenderFixture(side: 64)
        let line = try LineRenderer(
            device: fixture.device, geometry: #require(molecule.bondGeometry),
            material: LineMaterial(width: .pixels(4))
        )
        let points = try PointsRenderer(
            device: fixture.device, cloud: molecule.pointCloud, material: PointsMaterial(size: .pixels(6))
        )
        let pixels = try fixture.render(fixture.graph([
            SceneNode(primitive: .lines(line)), SceneNode(primitive: .points(points)),
        ])).pixels
        let colors = try #require(molecule.pointCloud.colors)
        #expect(isNearly(pixels[29 * 64 + 16], colors[0]))
        #expect(isNearly(pixels[29 * 64 + 48], colors[1]))
        #expect(isNearly(pixels[32 * 64 + 32], SIMD4(1, 1, 1, 1)))
        #expect(pixels[24 * 64 + 32] == .zero)
    }

    private static func atom(
        kind: String = "ATOM", serial: Int, name: String, position: SIMD3<Float>, element: String = "",
        residue: String = "", chain: String = "", sequence: Int? = nil,
        insertion: String = "", alternate: String = "", occupancy: Float? = nil, temperature: Float? = nil
    ) -> String {
        var bytes = Array(repeating: UInt8(32), count: 80)
        func put(_ value: String, in columns: Range<Int>, right: Bool = false) {
            let text = Array(value.utf8)
            let start = right ? columns.upperBound - text.count : columns.lowerBound
            bytes.replaceSubrange(start..<(start + text.count), with: text)
        }
        func decimal(_ value: Float, digits: Int) -> String {
            String(format: "%.*f", locale: Locale(identifier: "en_US_POSIX"), digits, value)
        }
        put(kind, in: 0..<6)
        put(String(serial), in: 6..<11, right: true)
        put(name, in: 12..<16)
        put(alternate, in: 16..<17)
        put(residue, in: 17..<20)
        put(chain, in: 21..<22)
        if let sequence { put(String(sequence), in: 22..<26, right: true) }
        put(insertion, in: 26..<27)
        for axis in 0..<3 {
            let start = 30 + axis * 8
            put(decimal(position[axis], digits: 3), in: start..<(start + 8), right: true)
        }
        if let occupancy { put(decimal(occupancy, digits: 2), in: 54..<60, right: true) }
        if let temperature { put(decimal(temperature, digits: 2), in: 60..<66, right: true) }
        put(element, in: 76..<78, right: true)
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func connect(_ source: Int, _ targets: [Int]) -> String {
        "CONECT" + ([source] + targets).map {
            let value = String($0)
            return String(repeating: " ", count: 5 - value.count) + value
        }.joined()
    }
}
