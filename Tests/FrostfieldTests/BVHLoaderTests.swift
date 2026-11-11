import Foundation
import Frostfield
import Testing
import simd

struct BVHLoaderTests {
    static let orderedChannels = """
    HIERARCHY
    ROOT Hips
    {
        OFFSET 1 2 -0.5
        CHANNELS 6 Zrotation Xposition Xrotation Zposition Yrotation Yposition
        JOINT Knee
        {
            OFFSET 0 -1 0.2
            CHANNELS 3 Yrotation Zrotation Xrotation
            JOINT Foot
            {
                OFFSET 0 -0.8 0.4
                CHANNELS 4 Xrotation Zposition Yrotation Zrotation
                End Site
                {
                    OFFSET 0.1 0 0.3
                }
            }
        }
        JOINT Arm
        {
            OFFSET 0.6 0.3 -0.2
            CHANNELS 3 Xrotation Yrotation Zrotation
            End Site
            {
                OFFSET 0.5 0 0
            }
        }
    }
    MOTION
    Frames: 3
    Frame Time: 0.125
    30 0.2 -10 1.5 40 -0.3    15 -20 8    -5 0.1 25 12    5 -12 25
    -20 1.3 22 -0.5 65 0.4    30 25 -12    15 -0.3 -10 30    45 25 -30
    50 -0.6 -45 0.8 -35 1.0    -25 55 13    60 0.7 15 -35    -30 20 55
    """
}

extension BVHLoaderTests {
    @Test("Ordered joint channels reproduce r187 motion at and between keys")
    func referencePlayback() throws {
        let asset = try BVHLoader.parse(Self.orderedChannels)
        #expect(asset.clip.name == "animation")
        #expect(asset.frameCount == 3 && asset.frameTime == 0.125 && asset.clip.duration == 0.25)
        #expect(asset.clip.tracks.count == 8)
        #expect(asset.clip.tracks.allSatisfy { $0.times == [0, 0.125, 0.25] })
        let graph = try skeleton(asset)
        let rest = try graph.resolve()
        #expect(rest.map(\.name) == ["Hips", "Knee", "Foot", "ENDSITE", "Arm", "ENDSITE"])
        #expect(asset.root.transform.translation == SIMD3(1, 2, -0.5))
        #expect(asset.root.children[0].children[0].children[0].transform.translation == SIMD3(0.1, 0, 0.3))
        #expect(asset.clip.nodeIDs == [rest[0].id, rest[1].id, rest[2].id, rest[4].id])

        // Sampled with Three.js r187 BVHLoader and AnimationMixer using LoopOnce.
        let samples: [(time: Float, hips: SIMD4<Float>, foot: SIMD3<Float>, footEnd: SIMD3<Float>, armEnd: SIMD3<Float>)] = [
            (0, SIMD4(-0.1672934, 0.3079118, 0.2134916, 0.9119346), SIMD3(2.2749754, 0.3858499, 1.919726),
             SIMD3(2.5764649, 0.4782829, 1.8960575), SIMD3(1.66, 2.3888327, 0.1861955)),
            (0.0625, SIMD4(0.0429157, 0.4144482, 0.0884976, 0.9047426), SIMD3(2.4252909, 0.3149346, -0.0731837),
             SIMD3(2.7344584, 0.3801749, -0.0858006), SIMD3(2.1883462, 2.6036069, -0.9443737)),
            (0.125, SIMD4(0.2500687, 0.4914705, -0.0427986, 0.8331211), SIMD3(2.1884042, 0.8760866, -1.873539),
             SIMD3(2.4994468, 0.8239946, -1.8967535), SIMD3(2.4858045, 3.0254792, -1.9376719)),
            (0.1875, SIMD4(0.0223240, 0.0519687, 0.2639020, 0.9628897), SIMD3(2.7188333, 1.6556350, 0.2886882),
             SIMD3(2.9374848, 1.7695124, 0.4867374), SIMD3(1.9200085, 3.6012732, -0.8723782)),
            (0.25, SIMD4(-0.2133666, -0.4060304, 0.4766707, 0.7499327), SIMD3(0.2359903, 2.5286926, 2.3699158),
             SIMD3(0.1854603, 2.7493102, 2.5907654), SIMD3(0.6574691, 4.0804188, -0.059285)),
        ]
        var mixer = AnimationMixer()
        let action = try mixer.addAction(for: asset.clip, loop: .once, clampWhenFinished: true)
        try mixer.play(action)
        var previous: Float = 0
        for sample in samples {
            try mixer.update(deltaTime: Double(sample.time - previous))
            previous = sample.time
            let posed = try mixer.apply(to: graph)
            let orientation = posed.roots[0].transform.rotation.vector
            #expect(min(simd_length(orientation - sample.hips), simd_length(orientation + sample.hips)) < 0.000_02)
            #expect(simd_distance(try posed.worldPosition(of: rest[2].id), sample.foot) < 0.000_02)
            #expect(simd_distance(try posed.worldPosition(of: rest[3].id), sample.footEnd) < 0.000_02)
            #expect(simd_distance(try posed.worldPosition(of: rest[5].id), sample.armEnd) < 0.000_02)
        }
        #expect(graph.roots[0].transform.translation == SIMD3(1, 2, -0.5))
        #expect(try asset.clip.sample(at: 0.125)[rest[2].id]?.translation == SIMD3(0, -0.8, 0.1))
    }

    @Test("Independent track flags and repeated joint names preserve each node's bindings")
    func trackOptions() throws {
        let source = Self.orderedChannels.replacingOccurrences(of: "JOINT Arm", with: "JOINT Knee")
        let position = try BVHLoader.parse(source, options: .init(animateBoneRotations: false))
        let ids = position.root.children.map(\.id)
        #expect(position.root.children.map(\.name) == ["Knee", "Knee"])
        #expect(ids[0] != ids[1])
        let poses = try position.clip.sample(at: 0.125)
        #expect(poses[ids[0]]?.translation == SIMD3(0, -1, 0.2))
        #expect(poses[ids[1]]?.translation == SIMD3(0.6, 0.3, -0.2))
        #expect(poses[ids[0]]?.rotation == nil && poses[ids[1]]?.rotation == nil)

        let rotation = try BVHLoader.parse(source, options: .init(animateBonePositions: false))
        let turns = try rotation.clip.sample(at: 0.125)
        #expect(turns[rotation.root.id]?.translation == nil)
        #expect(turns[rotation.root.id]?.rotation != nil)

        let rest = try BVHLoader.parse(source, options: .init(animateBonePositions: false, animateBoneRotations: false))
        #expect(rest.clip.tracks.isEmpty && rest.clip.duration == 0)
        #expect(try rest.clip.sample(at: 0).isEmpty)
    }

    @Test("Local files and UTF-8 bytes accept compact joints, blank lines, BOM, CRLF and colon names")
    func inputForms() throws {
        let text = """
        HIERARCHY
        root Rig:Hips{ OFFSET 1 2 3 CHANNELS 2 Zrotation Xposition end SITE{OFFSET 0 1 0} }
        MOTION
        Frames : 1
        Frame Time : 0.04

        90\t-2
        """
        let bytes = Data(("\u{FEFF}" + text.replacingOccurrences(of: "\n", with: "\r\n")).utf8)
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "short.bvh")
        try bytes.write(to: url)
        let asset = try BVHLoader.load(contentsOf: url)
        #expect(asset.root.name == "Rig:Hips")
        #expect(asset.root.children.first?.name == "ENDSITE")
        #expect(asset.frameCount == 1 && asset.clip.duration == 0)
        let pose = try #require(asset.clip.sample(at: 12)[asset.root.id])
        #expect(pose.translation == SIMD3(-1, 2, 3))
        let turn = try #require(pose.rotation)
        #expect(simd_distance(turn.act(SIMD3(1, 0, 0)), SIMD3(0, 1, 0)) < 0.000_01)
        #expect(throws: BVHLoader.Failure.fileURL) { try BVHLoader.load(contentsOf: directory) }
        #expect(throws: BVHLoader.Failure.encoding) { try BVHLoader.load(Data([0xFF])) }

        let noChannels = """
        HIERARCHY ROOT Rest { OFFSET 0 0 0 CHANNELS 0 }
        MOTION
        Frames: 2
        Frame Time: 0.5
        """
        let still = try BVHLoader.parse(noChannels)
        #expect(still.clip.duration == 0.5)
        #expect(try still.clip.sample(at: 0.4)[still.root.id]?.translation == .zero)
    }

    @Test("Invalid declarations, rows, times and resource budgets fail at import")
    func malformed() throws {
        func document(_ channels: String = "1 Xposition", count: String = "2", interval: String = "0.5",
                      data: String = "1\n2") -> String {
            """
            HIERARCHY
            ROOT Hips { OFFSET 0 0 0 CHANNELS \(channels) }
            MOTION
            Frames: \(count)
            Frame Time: \(interval)
            \(data)
            """
        }
        #expect(throws: BVHLoader.Failure.channel(line: 2, name: "Scale")) {
            try BVHLoader.parse(document("1 Scale"))
        }
        #expect(throws: BVHLoader.Failure.channel(line: 2, name: "Xposition")) {
            try BVHLoader.parse(document("2 Xposition Xposition"))
        }
        #expect(throws: BVHLoader.Failure.frameCount(line: 4)) {
            try BVHLoader.parse(document(count: "0"))
        }
        #expect(throws: BVHLoader.Failure.frameTime(line: 5)) {
            try BVHLoader.parse(document(interval: "1e-99"))
        }
        #expect(throws: BVHLoader.Failure.frameTime(line: 5)) {
            try BVHLoader.parse(document(count: "3", interval: "3e38", data: "0\n0\n0"))
        }
        #expect(throws: BVHLoader.Failure.frame(index: 0, line: 6)) {
            try BVHLoader.parse(document(data: "1 2\n3"))
        }
        #expect(throws: BVHLoader.Failure.frame(index: 1, line: 7)) {
            try BVHLoader.parse(document(data: "1\nNaN"))
        }
        #expect(throws: BVHLoader.Failure.frame(index: 0, line: 6)) {
            try BVHLoader.parse(document("2 Xposition Yposition", data: "1\n2 3 4"))
        }
        #expect(throws: BVHLoader.Failure.frame(index: 0, line: 6)) {
            try BVHLoader.parse(document(data: "1e40\n2"))
        }
        #expect(throws: BVHLoader.Failure.syntax(line: 8, expected: "the end of the motion data")) {
            try BVHLoader.parse(document(data: "1\n2\n3"))
        }
        #expect(throws: BVHLoader.Failure.size) {
            try BVHLoader.parse(document(count: "2000001"))
        }
        let tooDeep = "HIERARCHY ROOT Hips { OFFSET 0 0 0 CHANNELS 0 " +
            String(repeating: "JOINT Child { OFFSET 0 0 0 CHANNELS 0 ", count: 256)
        #expect(throws: BVHLoader.Failure.size) { try BVHLoader.parse(tooDeep) }

        let deepest = "HIERARCHY ROOT Hips { OFFSET 0 0 0 CHANNELS 0 " +
            String(repeating: "JOINT Child { OFFSET 0 0 0 CHANNELS 0 ", count: 255) +
            String(repeating: "} ", count: 256) + "MOTION\nFrames: 1\nFrame Time: 0.5"
        let atLimit = try BVHLoader.parse(deepest)
        #expect(try skeleton(atLimit).resolve().count == 256)
    }

    private func skeleton(_ asset: BVHLoader.Asset) throws -> SceneGraph {
        SceneGraph(
            roots: [asset.root],
            camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
    }
}
