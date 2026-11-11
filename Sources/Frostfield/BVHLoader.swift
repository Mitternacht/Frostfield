import Foundation
import simd

/// Loads BioVision Hierarchy motion capture as a scene node hierarchy and a
/// playable clip. The authored offsets and translations keep their source units.
public enum BVHLoader {
    public struct Options: Hashable, Sendable {
        public var animateBonePositions: Bool
        public var animateBoneRotations: Bool

        public init(animateBonePositions: Bool = true, animateBoneRotations: Bool = true) {
            self.animateBonePositions = animateBonePositions
            self.animateBoneRotations = animateBoneRotations
        }
    }

    public struct Asset {
        /// The rest skeleton. End sites are named `ENDSITE` and have no tracks.
        public let root: SceneNode
        /// Position and rotation tracks targeting node IDs in `root`.
        public let clip: AnimationClip
        public let frameCount: Int
        /// Seconds between the authored samples. The first sample is at zero.
        public let frameTime: Double
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case encoding
        case size
        case syntax(line: Int, expected: String)
        case channel(line: Int, name: String)
        case frameCount(line: Int)
        case frameTime(line: Int)
        /// A zero-based frame index and one-based line number.
        case frame(index: Int, line: Int)

        public var errorDescription: String? {
            switch self {
            case .fileURL: "BVH import requires a regular local file."
            case .encoding: "BVH import requires UTF-8 text."
            case .size: "The BVH input exceeds the supported size or hierarchy limits."
            case .syntax(let line, let expected): "Expected \(expected) on BVH line \(line)."
            case .channel(let line, let name): "BVH line \(line) has an unknown or duplicate channel '\(name)'."
            case .frameCount(let line): "BVH line \(line) needs a positive frame count."
            case .frameTime(let line): "BVH line \(line) needs a positive frame interval with representable key times."
            case .frame(let index, let line): "BVH frame \(index) on line \(line) needs one finite value per declared channel."
            }
        }
    }

    private static let maximumFileBytes = 64 * 1_024 * 1_024

    public static func load(contentsOf url: URL, options: Options = .init()) throws -> Asset {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        guard let size = info.fileSize, size <= maximumFileBytes else { throw Failure.size }
        return try load(Data(contentsOf: url, options: .mappedIfSafe), options: options)
    }

    public static func load(_ data: Data, options: Options = .init()) throws -> Asset {
        guard data.count <= maximumFileBytes else { throw Failure.size }
        guard let text = String(data: data, encoding: .utf8) else { throw Failure.encoding }
        return try parse(text, options: options)
    }

    /// Parses a single-root hierarchy, up to 4,096 nodes, 256 hierarchy levels,
    /// and 2,000,000 joint-frame pairs. A motion row holds all channels for one
    /// frame. Keys use linear translation and spherical quaternion interpolation.
    public static func parse(_ text: String, options: Options = .init()) throws -> Asset {
        guard text.utf8.count <= maximumFileBytes else { throw Failure.size }
        var parser = BVHParser(text)
        return try parser.parse(options: options)
    }
}

private struct BVHParser {
    private typealias Failure = BVHLoader.Failure
    private static let maximumNodes = 4_096
    private static let maximumDepth = 256
    private static let maximumJointFrames = 2_000_000

    private enum Channel: String, Hashable {
        case xPosition = "Xposition", yPosition = "Yposition", zPosition = "Zposition"
        case xRotation = "Xrotation", yRotation = "Yrotation", zRotation = "Zrotation"
    }

    private struct Joint {
        let id: UUID
        let offset: SIMD3<Double>
        let channels: [Channel]
    }

    private struct PendingNode {
        let id: UUID
        let name: String
        let position: SIMD3<Float>
        var children: [SceneNode] = []

        func build() -> SceneNode {
            SceneNode(id: id, transform: .init(translation: position), children: children, name: name)
        }
    }

    private var scanner: Scanner
    private var lookahead: Token?
    private var joints: [Joint] = []
    private var nodeCount = 0

    init(_ text: String) { scanner = Scanner(text) }

    mutating func parse(options: BVHLoader.Options) throws -> BVHLoader.Asset {
        try expect("HIERARCHY")
        try expect("ROOT", ignoringCase: true)
        let rootName = try name()
        let root = try hierarchy(rootName: rootName)
        try expect("MOTION")
        try expectHeader("Frames")
        let count = try take(expected: "a frame count")
        guard let frameCount = Int(count.text), frameCount > 0 else { throw Failure.frameCount(line: count.line) }
        guard frameCount <= Self.maximumJointFrames / joints.count else { throw Failure.size }
        try expect("Frame")
        try expectHeader("Time")
        let interval = try take(expected: "a frame interval")
        guard let frameTime = Double(interval.text), frameTime.isFinite, frameTime > 0,
              Float(frameTime).isFinite, Float(frameTime) > 0 else { throw Failure.frameTime(line: interval.line) }
        var times: [Float] = []
        times.reserveCapacity(frameCount)
        for frame in 0..<frameCount {
            let time = Float(Double(frame) * frameTime)
            guard time.isFinite, times.last.map({ $0 < time }) != false else {
                throw Failure.frameTime(line: interval.line)
            }
            times.append(time)
        }

        let tracks = try motion(times: times, after: interval.line, options: options)
        let clip = tracks.isEmpty
            ? AnimationClip(name: "animation", duration: 0, blendMode: .normal, targets: [], reference: [:],
                            tracks: [], trackReference: [:])
            : try AnimationClip(name: "animation", tracks: tracks)
        return BVHLoader.Asset(root: root, clip: clip, frameCount: frameCount, frameTime: frameTime)
    }

    private mutating func hierarchy(rootName: String) throws -> SceneNode {
        var pending = [try beginNode(name: rootName, endSite: false)]
        while true {
            let next = try take(expected: "JOINT, End Site, or }")
            switch next.text.uppercased() {
            case "}":
                let node = pending.removeLast().build()
                if pending.isEmpty { return node }
                pending[pending.count - 1].children.append(node)
            case "JOINT":
                guard pending.count < Self.maximumDepth else { throw Failure.size }
                let childName = try name()
                pending.append(try beginNode(name: childName, endSite: false))
            case "END":
                guard pending.count < Self.maximumDepth else { throw Failure.size }
                try expect("SITE", ignoringCase: true)
                let end = try beginNode(name: "ENDSITE", endSite: true)
                try expect("}")
                pending[pending.count - 1].children.append(end.build())
            default: throw Failure.syntax(line: next.line, expected: "JOINT, End Site, or }")
            }
        }
    }

    private mutating func beginNode(name: String, endSite: Bool) throws -> PendingNode {
        guard nodeCount < Self.maximumNodes else { throw Failure.size }
        nodeCount += 1
        try expect("{")
        try expect("OFFSET")
        let offsetLine = try peek()?.line ?? scanner.line
        let offset = try SIMD3(number(), number(), number())
        let position = SIMD3<Float>(offset)
        guard position.isFiniteVector else { throw Failure.syntax(line: offsetLine, expected: "a representable OFFSET") }
        let node = PendingNode(id: UUID(), name: name, position: position)
        if endSite { return node }

        try expect("CHANNELS")
        let token = try take(expected: "a channel count")
        guard let count = Int(token.text), (0...6).contains(count) else {
            throw Failure.syntax(line: token.line, expected: "a channel count from zero through six")
        }
        var channels: [Channel] = []
        for _ in 0..<count {
            let token = try take(expected: "a channel name")
            guard let channel = Channel(rawValue: token.text), !channels.contains(channel) else {
                throw Failure.channel(line: token.line, name: token.text)
            }
            channels.append(channel)
        }
        joints.append(Joint(id: node.id, offset: offset, channels: channels))
        return node
    }

    private mutating func motion(
        times: [Float], after headerLine: Int, options: BVHLoader.Options
    ) throws -> [AnimationTrack] {
        var positions = options.animateBonePositions ? [[SIMD3<Float>]](repeating: [], count: joints.count) : []
        var rotations = options.animateBoneRotations ? [[simd_quatf]](repeating: [], count: joints.count) : []
        let hasChannels = joints.contains { !$0.channels.isEmpty }
        var precedingLine = headerLine
        for frame in times.indices {
            let line: Int
            if hasChannels {
                let next = try peek()
                guard let token = next, token.line > precedingLine else {
                    throw Failure.frame(index: frame, line: next?.line ?? scanner.line)
                }
                line = token.line
            } else {
                line = headerLine
            }

            for index in joints.indices {
                let joint = joints[index]
                var translation = joint.offset
                var rotation = simd_quatd(real: 1, imag: .zero)
                for channel in joint.channels {
                    guard let token = try peek(), token.line == line, let value = Double(token.text), value.isFinite else {
                        throw Failure.frame(index: frame, line: line)
                    }
                    lookahead = nil
                    switch channel {
                    case .xPosition: translation.x += value
                    case .yPosition: translation.y += value
                    case .zPosition: translation.z += value
                    case .xRotation: rotation *= Self.rotation(value, axis: SIMD3(1, 0, 0))
                    case .yRotation: rotation *= Self.rotation(value, axis: SIMD3(0, 1, 0))
                    case .zRotation: rotation *= Self.rotation(value, axis: SIMD3(0, 0, 1))
                    }
                }
                let position = SIMD3<Float>(translation)
                guard position.isFiniteVector else { throw Failure.frame(index: frame, line: line) }
                if options.animateBonePositions { positions[index].append(position) }
                if options.animateBoneRotations {
                    rotations[index].append(simd_quatf(vector: SIMD4<Float>(rotation.vector)))
                }
            }
            if hasChannels, try peek()?.line == line { throw Failure.frame(index: frame, line: line) }
            precedingLine = line
        }
        if let trailing = try peek() { throw Failure.syntax(line: trailing.line, expected: "the end of the motion data") }

        var tracks: [AnimationTrack] = []
        for (index, joint) in joints.enumerated() {
            if options.animateBonePositions {
                tracks.append(try .vector(.node(joint.id, .translation), times: times, values: positions[index]))
            }
            if options.animateBoneRotations {
                tracks.append(try .quaternion(.node(joint.id, .rotation), times: times, values: rotations[index]))
            }
        }
        return tracks
    }

    private static func rotation(_ degrees: Double, axis: SIMD3<Double>) -> simd_quatd {
        // A full quaternion period keeps huge finite angles representable.
        simd_quatd(angle: degrees.truncatingRemainder(dividingBy: 720) * (.pi / 180), axis: axis)
    }

    private mutating func name() throws -> String {
        let token = try take(expected: "a joint name")
        guard token.text != "{", token.text != "}" else {
            throw Failure.syntax(line: token.line, expected: "a joint name")
        }
        return token.text
    }

    private mutating func number() throws -> Double {
        let token = try take(expected: "a finite number")
        guard let value = Double(token.text), value.isFinite else {
            throw Failure.syntax(line: token.line, expected: "a finite number")
        }
        return value
    }

    private mutating func expectHeader(_ word: String) throws {
        let token = try take(expected: word + ":")
        if token.text == word { try expect(":") }
        else if token.text != word + ":" { throw Failure.syntax(line: token.line, expected: word + ":") }
    }

    private mutating func expect(_ text: String, ignoringCase: Bool = false) throws {
        let token = try take(expected: text)
        let actual = ignoringCase ? token.text.uppercased() : token.text
        guard actual == text else { throw Failure.syntax(line: token.line, expected: text) }
    }

    private mutating func take(expected: String) throws -> Token {
        guard let token = try peek() else { throw Failure.syntax(line: scanner.line, expected: expected) }
        lookahead = nil
        return token
    }

    private mutating func peek() throws -> Token? {
        if lookahead == nil { lookahead = try scanner.next() }
        return lookahead
    }

    private struct Token {
        let text: String
        let line: Int
    }

    /// Streams tokens from validated UTF-8 without allocating an array of all words.
    private struct Scanner {
        let bytes: String.UTF8View
        var index: String.Index
        var line = 1

        init(_ text: String) {
            bytes = text.utf8
            index = bytes.starts(with: [0xEF, 0xBB, 0xBF]) ? bytes.index(bytes.startIndex, offsetBy: 3) : bytes.startIndex
        }

        mutating func next() throws -> Token? {
            while index != bytes.endIndex, Self.whitespace(bytes[index]) {
                let byte = bytes[index]
                bytes.formIndex(after: &index)
                if byte == 10 || byte == 13 { line += 1 }
                if byte == 13, index != bytes.endIndex, bytes[index] == 10 { bytes.formIndex(after: &index) }
            }
            guard index != bytes.endIndex else { return nil }
            let start = index
            bytes.formIndex(after: &index)
            if bytes[start] != 123, bytes[start] != 125 {
                var length = 1
                while index != bytes.endIndex {
                    let byte = bytes[index]
                    if Self.whitespace(byte) || byte == 123 || byte == 125 { break }
                    guard length < 1_024 else { throw Failure.size }
                    length += 1
                    bytes.formIndex(after: &index)
                }
            }
            return Token(text: String(decoding: bytes[start..<index], as: UTF8.self), line: line)
        }

        private static func whitespace(_ byte: UInt8) -> Bool { byte == 32 || (9...13).contains(byte) }
    }
}
