import Foundation
import simd

/// Prepared meshes that a scene node switches between by camera distance, like
/// Three.js `LOD`.
///
/// Distance is measured in world units from the camera position to the node's
/// world origin; the node's scale does not change it. Level `i` is selected from
/// `levels[i].distance` until the next level's distance. Hysteresis delays
/// switching back toward a finer level: while level `i` is current, the camera
/// must come closer than `distance × (1 − hysteresis)` to leave it. Graph
/// resolution, and therefore encoding, picking and path-tracing snapshots,
/// selects with `currentLevel` as the reference, and every pass of one encode
/// (color and shadows) uses that one level. Selection itself does not change the
/// graph; call `SceneGraph.updateLevelsOfDetail(cameraPosition:)` once a frame to
/// store the selection so hysteresis carries over.
public struct LevelOfDetail {
    public static let maximumLevelCount = 16

    public struct Level {
        public let mesh: MeshRenderer
        /// The camera distance in world units at which this level starts. Finite and nonnegative.
        public let distance: Float
        /// The fraction of `distance` the camera must move back inside before
        /// leaving this level for a finer one, in `0..<1`. Zero switches exactly at `distance`.
        public let hysteresis: Float

        public init(mesh: MeshRenderer, distance: Float, hysteresis: Float = 0) {
            self.mesh = mesh
            self.distance = distance
            self.hysteresis = hysteresis
        }
    }

    public let levels: [Level]
    /// The level selected most recently, used as the hysteresis reference.
    /// `SceneGraph.updateLevelsOfDetail(cameraPosition:)` stores each selection here.
    public package(set) var currentLevel: Int

    /// Takes 1 to 16 levels with strictly increasing distances. The first level
    /// shows at any distance below the second level's.
    public init(levels: [Level], currentLevel: Int = 0) throws {
        guard (1...Self.maximumLevelCount).contains(levels.count) else { throw Failure.levelCount }
        for (index, level) in levels.enumerated() {
            guard level.distance.isFinite, level.distance >= 0,
                  index == 0 || level.distance > levels[index - 1].distance else {
                throw Failure.distance(index)
            }
            guard level.hysteresis.isFinite, (0..<1).contains(level.hysteresis) else { throw Failure.hysteresis(index) }
        }
        guard levels.indices.contains(currentLevel) else { throw Failure.currentLevel }
        self.levels = levels
        self.currentLevel = currentLevel
    }

    /// The level for a camera at this distance, with hysteresis relative to `currentLevel`.
    /// Matches Three.js `LOD.update`. A nonfinite distance selects the current level.
    public func level(forDistance distance: Float) -> Int {
        guard distance.isFinite else { return currentLevel }
        var selected = 0
        for index in levels.indices.dropFirst() {
            var threshold = levels[index].distance
            if index == currentLevel { threshold -= threshold * levels[index].hysteresis }
            guard distance >= threshold else { break }
            selected = index
        }
        return selected
    }

    public enum Failure: LocalizedError, Equatable {
        case levelCount
        case distance(Int)
        case hysteresis(Int)
        case currentLevel

        public var errorDescription: String? {
            switch self {
            case .levelCount: "A level of detail needs 1 to \(LevelOfDetail.maximumLevelCount) levels."
            case .distance(let index): "Level \(index) needs a finite, nonnegative distance beyond the previous level's."
            case .hysteresis(let index): "Level \(index) needs a hysteresis fraction from zero up to one."
            case .currentLevel: "The current level must be one of the levels."
            }
        }
    }
}

public extension SceneGraph {
    /// Stores each level-of-detail node's selection for a camera position as its
    /// `currentLevel`, so the next frame's hysteresis starts from it. Uses the
    /// graph's camera translation when the position is nil. Rethrows resolution errors.
    mutating func updateLevelsOfDetail(cameraPosition: SIMD3<Float>? = nil) throws {
        let resolved = try resolve(cameraPosition: cameraPosition ?? cameraTransform.translation)
        var selections: [UUID: Int] = [:]
        for node in resolved {
            if let level = node.levelOfDetail { selections[node.id] = level }
        }
        guard !selections.isEmpty else { return }
        self = updatingNodes { node in
            if let level = selections[node.id] { node.levelsOfDetail?.currentLevel = level }
        }
    }
}
