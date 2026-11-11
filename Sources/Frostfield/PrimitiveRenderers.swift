import Foundation
import Metal
import simd

/// How a point, line or sprite draw combines with the color already in the pass.
/// Every mode outputs premultiplied linear RGBA.
public enum PrimitiveBlending: Equatable, Sendable {
    /// Writes color without blending and writes depth. Edges are hard: a pixel
    /// is drawn when its center lies inside the shape. Draws with opaque meshes.
    case opaque
    /// Premultiplied source-over. Tests depth without writing it and is sorted
    /// back to front with blended meshes. Wide lines and point discs antialias
    /// their edges; sprite and square-point edges follow the quad.
    case alpha
    /// Adds premultiplied color, independent of order, with the same edges as
    /// `.alpha`. Tests depth without writing it.
    case additive
}

/// A point, line or sprite renderer attached to a scene node, like a Three.js
/// `Points`, `Line`/`LineSegments2` or `Sprite` object.
///
/// Scene graph encoding draws it in the color pass with the node's world
/// transform, and its `materialParameters` multiply the material color. Primitives
/// share the opaque and blended mesh draw ordering. Blended primitives sort by
/// the projected center of their world bounds. Primitives without depth
/// testing draw last, in node order. Primitives do not cast or receive shadows.
public enum ScenePrimitive {
    /// A point cloud, like `Points`.
    case points(PointsRenderer)
    /// A line strip, loop or separate segments, like `Line`, `LineLoop`,
    /// `LineSegments`, `Line2` or `LineSegments2`.
    case lines(LineRenderer)
    /// A batch of sprites, like a group of `Sprite` objects sharing a material.
    case sprites(SpriteRenderer)

    var rendererIdentity: AnyObject {
        switch self {
        case .points(let renderer): renderer
        case .lines(let renderer): renderer
        case .sprites(let renderer): renderer
        }
    }

    var colorMap: (any MTLTexture)? {
        switch self {
        case .points(let renderer): renderer.material.map
        case .lines: nil
        case .sprites(let renderer): renderer.material.map
        }
    }

    var drawElements: PrimitiveDrawElements? {
        switch self {
        case .points(let renderer):
            let range = renderer.activePoints
            return range.isEmpty ? nil : .points(range)
        case .lines(let renderer): return renderer.activeLines.map(PrimitiveDrawElements.lines)
        case .sprites(let renderer): return .sprites(renderer.sprites.count)
        }
    }

    func recordResources(in inventory: inout SceneResourceInventory) {
        switch self {
        case .points(let renderer): renderer.recordResources(in: &inventory)
        case .lines(let renderer): renderer.recordResources(in: &inventory)
        case .sprites(let renderer): renderer.recordResources(in: &inventory)
        }
    }

    var device: any MTLDevice {
        switch self {
        case .points(let renderer): renderer.device
        case .lines(let renderer): renderer.device
        case .sprites(let renderer): renderer.device
        }
    }

    var colorPixelFormat: MTLPixelFormat {
        switch self {
        case .points(let renderer): renderer.colorPixelFormat
        case .lines(let renderer): renderer.colorPixelFormat
        case .sprites(let renderer): renderer.colorPixelFormat
        }
    }

    var sampleCount: Int {
        switch self {
        case .points(let renderer): renderer.sampleCount
        case .lines(let renderer): renderer.sampleCount
        case .sprites(let renderer): renderer.sampleCount
        }
    }

    var blending: PrimitiveBlending {
        switch self {
        case .points(let renderer): renderer.material.blending
        case .lines(let renderer): renderer.material.blending
        case .sprites(let renderer): renderer.material.blending
        }
    }

    var depthTest: Bool {
        switch self {
        case .points(let renderer): renderer.material.depthTest
        case .lines(let renderer): renderer.material.depthTest
        case .sprites(let renderer): renderer.material.depthTest
        }
    }

    var softParticleFade: SoftParticleDepth.Fade? {
        switch self {
        case .points(let renderer): renderer.material.softParticleFade
        case .lines: nil
        case .sprites(let renderer): renderer.material.softParticleFade
        }
    }

    /// The local box around the primitive's anchor points.
    var localBounds: AxisAlignedBox {
        switch self {
        case .points(let renderer): renderer.cloud.bounds
        case .lines(let renderer): renderer.geometry.bounds
        case .sprites(let renderer): renderer.bounds
        }
    }

    /// How far the drawn shape reaches past its anchor points in world units.
    /// Pixel-sized shapes report zero, so they cull by their anchors, as Three.js does.
    var worldPadding: Float {
        switch self {
        case .points(let renderer):
            guard case .world(let size) = renderer.material.size else { return 0 }
            let cornerScale: Float = renderer.material.shape == .square ? Float(2).squareRoot() : 1
            return size * (renderer.maximumSizeScale / 2) * cornerScale
        case .lines(let renderer):
            guard case .world(let width) = renderer.material.width else { return 0 }
            var reach: Float = renderer.material.cap == .square ? Float(2).squareRoot() : 1
            if renderer.geometry.topology != .segments, case .miter(let limit) = renderer.material.join {
                reach = max(reach, limit)
            }
            return (width / 2) * reach
        case .sprites(let renderer):
            return renderer.material.sizeAttenuation ? renderer.maximumReach : 0
        }
    }

    func encode(
        _ draw: PrimitiveDraw, into encoder: any MTLRenderCommandEncoder, depthMode: MeshDepthMode,
        softParticles: PrimitiveSoftParticleFrame?
    ) {
        switch self {
        case .points(let renderer): renderer.encode(draw, into: encoder, depthMode: depthMode, softParticles: softParticles)
        case .lines(let renderer): renderer.encode(draw, into: encoder, depthMode: depthMode)
        case .sprites(let renderer): renderer.encode(draw, into: encoder, depthMode: depthMode, softParticles: softParticles)
        }
    }
}

public extension SceneNode {
    /// A node that draws a point, line or sprite renderer. Its material parameters
    /// multiply the material color as a linear RGBA tint.
    init(
        id: UUID = UUID(),
        transform: SceneTransform = SceneTransform(),
        primitive: ScenePrimitive,
        materialParameters: SIMD4<Float> = SIMD4(repeating: 1),
        children: [SceneNode] = [],
        frustumCulled: Bool = true,
        cullingBounds: AxisAlignedBox? = nil
    ) {
        self.init(
            id: id, transform: transform, materialParameters: materialParameters, children: children,
            castsShadow: false, frustumCulled: frustumCulled, cullingBounds: cullingBounds
        )
        self.primitive = primitive
    }
}

// MARK: - Points

/// The appearance of a point cloud, like Three.js `PointsMaterial`.
public struct PointsMaterial {
    public enum Size: Equatable, Sendable {
        /// A diameter in pixels at every distance (`sizeAttenuation = false`).
        case pixels(Float)
        /// A diameter in world units, so the drawn size falls as 1/depth like a
        /// real object of that width (`sizeAttenuation = true`). Three.js omits the
        /// camera's focal scale; Frostfield matches a mesh of the same width.
        case world(Float)
    }

    public enum Shape: Equatable, Sendable {
        /// A screen-aligned square, the Three.js default.
        case square
        /// A disc, antialiased when the material blends.
        case circle
    }

    public var size: Size
    public var shape: Shape
    /// Linear RGB and opacity, with straight alpha.
    public var color: SIMD4<Float>
    /// An optional straight-alpha color texture across each point's square, top-left origin.
    public var map: (any MTLTexture)?
    /// Pixels whose material alpha falls below this value are discarded.
    public var alphaTest: Float
    public var blending: PrimitiveBlending
    /// When false the points draw over everything, after every depth-tested draw.
    public var depthTest: Bool
    /// Optional intersection fade against `SceneGraph.viewportSnapshot` depth.
    /// Use alpha or additive blending.
    public var softParticleFade: SoftParticleDepth.Fade?

    public init(
        size: Size = .pixels(2), shape: Shape = .square, color: SIMD4<Float> = SIMD4(repeating: 1),
        map: (any MTLTexture)? = nil, alphaTest: Float = 0, blending: PrimitiveBlending = .opaque,
        depthTest: Bool = true, softParticleFade: SoftParticleDepth.Fade? = nil
    ) {
        self.size = size
        self.shape = shape
        self.color = color
        self.map = map
        self.alphaTest = alphaTest
        self.blending = blending
        self.depthTest = depthTest
        self.softParticleFade = softParticleFade
    }
}

/// A point cloud prepared on one Metal device. Each point draws as a
/// camera-facing quad in one instanced draw, so sizes have no hardware limit.
public final class PointsRenderer {
    public let device: any MTLDevice
    public let colorPixelFormat: MTLPixelFormat
    /// The sample count required of the color and depth attachments.
    public let sampleCount: Int
    public let cloud: PointCloud
    public let material: PointsMaterial
    /// The optional acceleration used by world-ray picking.
    public let pickingBVH: PointCloudBVH?
    /// The source vertices drawn and picked by this renderer. Edits reuse the
    /// prepared point and color buffers. Set the range between scene encodes.
    public var drawRange: GeometryDrawRange = .all
    var activePoints: Range<Int> { drawRange.resolve(elementCount: cloud.positions.count) }
    /// The largest per-point size multiplier, or one without per-point sizes.
    let maximumSizeScale: Float
    private let pipeline: any MTLRenderPipelineState
    private let depthStates: PrimitiveDepthStates
    private let points: any MTLBuffer
    private let colors: (any MTLBuffer)?

    fileprivate func recordResources(in inventory: inout SceneResourceInventory) {
        guard inventory.recordGeometry(self) else { return }
        inventory.recordPipeline(pipeline)
        inventory.recordBuffer(points)
        inventory.recordBuffer(colors)
        inventory.recordTexture(material.map)
    }

    /// Sizes must be finite and positive, color components finite and
    /// nonnegative with opacity in 0…1, and the alpha test in 0…1. A map must be
    /// a shader-readable 2D texture on this device.
    public convenience init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat = .rgba16Float, sampleCount: Int = 1,
        cloud: PointCloud, material: PointsMaterial
    ) throws {
        try self.init(device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                      cloud: cloud, material: material, sharingGeometryWith: nil)
    }

    /// Prepares the BVH's cloud for drawing and uses the same immutable
    /// hierarchy for scene world-ray picking. Draw range changes affect both.
    public convenience init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat = .rgba16Float, sampleCount: Int = 1,
        pickingBVH: PointCloudBVH, material: PointsMaterial
    ) throws {
        try self.init(device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                      cloud: pickingBVH.cloud, material: material, sharingGeometryWith: nil, pickingBVH: pickingBVH)
    }

    private init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, sampleCount: Int,
        cloud: PointCloud, material: PointsMaterial, sharingGeometryWith source: PointsRenderer?,
        pickingBVH: PointCloudBVH? = nil
    ) throws {
        switch material.size {
        case .pixels(let value), .world(let value):
            guard value.isFinite, value > 0 else { throw PrimitiveRendererError.size }
        }
        try PrimitivePipeline.validate(color: material.color, alphaTest: material.alphaTest)
        try PrimitivePipeline.validate(map: material.map, device: device)
        try PrimitivePipeline.validate(softParticleFade: material.softParticleFade, blending: material.blending)
        let library = try PrimitivePipeline.library(device: device)
        pipeline = try PrimitivePipeline.make(
            device: device, library: library, vertex: "frostfield_points_vertex", fragment: "frostfield_points_fragment",
            hasColors: cloud.colors != nil, hasMap: material.map != nil, hasSoftParticles: material.softParticleFade != nil,
            colorPixelFormat: colorPixelFormat, sampleCount: sampleCount, blending: material.blending, label: "Frostfield points"
        )
        depthStates = try PrimitiveDepthStates(device: device, blending: material.blending, depthTest: material.depthTest)
        if let source {
            maximumSizeScale = source.maximumSizeScale
            points = source.points
            colors = source.colors
        } else {
            let sizes = cloud.sizes
            let records = cloud.positions.enumerated().map { index, position in
                SIMD4(position, sizes?[index] ?? 1)
            }
            maximumSizeScale = sizes?.max() ?? 1
            points = try PrimitivePipeline.buffer(records, device: device, label: "Frostfield point positions")
            colors = try cloud.colors.map {
                try PrimitivePipeline.buffer($0.map(SIMD4<Float16>.init), device: device, label: "Frostfield point colors")
            }
        }
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.sampleCount = sampleCount
        self.cloud = cloud
        self.material = material
        self.pickingBVH = pickingBVH ?? source?.pickingBVH
    }

    func replacingMaterial(_ material: PointsMaterial, format: MTLPixelFormat, samples: Int) throws -> PointsRenderer {
        try PointsRenderer(device: device, colorPixelFormat: format, sampleCount: samples,
                           cloud: cloud, material: material, sharingGeometryWith: self)
    }

    fileprivate func encode(
        _ draw: PrimitiveDraw, into encoder: any MTLRenderCommandEncoder, depthMode: MeshDepthMode,
        softParticles: PrimitiveSoftParticleFrame?
    ) {
        guard case .points(let range) = draw.elements else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthStates.state(for: depthMode))
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(points, offset: range.lowerBound * MemoryLayout<SIMD4<Float>>.stride, index: 0)
        if let colors {
            encoder.setVertexBuffer(colors, offset: range.lowerBound * MemoryLayout<SIMD4<Float16>>.stride, index: 1)
        }
        var view = draw.view
        encoder.setVertexBytes(&view, length: MemoryLayout<PackedPrimitiveView>.stride, index: 2)
        var style = PackedPointStyle(material: material, tint: draw.tint)
        encoder.setVertexBytes(&style, length: MemoryLayout<PackedPointStyle>.stride, index: 3)
        encoder.setFragmentBytes(&style, length: MemoryLayout<PackedPointStyle>.stride, index: 0)
        if let map = material.map { encoder.setFragmentTexture(map, index: 0) }
        if let fade = material.softParticleFade { softParticles?.bind(fade, into: encoder) }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: range.count)
    }
}

// MARK: - Lines

/// The appearance of a line, covering Three.js `LineBasicMaterial`,
/// `LineDashedMaterial` and the fat-line `LineMaterial`.
public struct LineMaterial: Sendable {
    public enum Width: Equatable, Sendable {
        /// One-pixel lines rasterized by Metal without joins, caps or
        /// antialiasing, like `LineBasicMaterial` on WebGL. The cheapest mode.
        case hairline
        /// A width in pixels at every distance (`LineMaterial.worldUnits = false`).
        case pixels(Float)
        /// A width in world units that narrows with distance (`worldUnits = true`).
        case world(Float)
    }

    public enum Join: Equatable, Sendable {
        /// Sharp corners that become bevels where the miter length exceeds
        /// `limit` times the width, as in SVG. The limit must be at least 1.
        case miter(limit: Float = 4)
        case round
        case bevel
    }

    public enum Cap: Equatable, Sendable {
        case butt
        case round
        /// Extends each open end by half the width.
        case square
    }

    /// A repeating pattern measured in ``LineSegment`` distances: world units
    /// along the line, as Three.js `LineDashedMaterial` uses after
    /// `computeLineDistances()`.
    public struct Dash: Equatable, Sendable {
        public let length: Float
        public let gap: Float
        /// Shifts the pattern back along the line.
        public let offset: Float

        /// Length and gap must be finite and positive; the offset finite.
        public init(length: Float, gap: Float, offset: Float = 0) throws {
            guard length.isFinite, gap.isFinite, offset.isFinite, length > 0, gap > 0 else {
                throw PrimitiveRendererError.dash
            }
            self.length = length
            self.gap = gap
            self.offset = offset
        }
    }

    public var width: Width
    /// The corner shape between consecutive segments of a strip or loop.
    public var join: Join
    /// The shape at each open end.
    public var cap: Cap
    /// Linear RGB and opacity, with straight alpha.
    public var color: SIMD4<Float>
    public var dash: Dash?
    /// `.alpha` antialiases every edge. `.opaque` writes depth with hard edges.
    public var blending: PrimitiveBlending
    /// When false the line draws over everything, after every depth-tested draw.
    public var depthTest: Bool

    /// Round joins and caps match the Three.js fat-line appearance.
    public init(
        width: Width = .pixels(1), join: Join = .round, cap: Cap = .round,
        color: SIMD4<Float> = SIMD4(repeating: 1), dash: Dash? = nil,
        blending: PrimitiveBlending = .alpha, depthTest: Bool = true
    ) {
        self.width = width
        self.join = join
        self.cap = cap
        self.color = color
        self.dash = dash
        self.blending = blending
        self.depthTest = depthTest
    }
}

/// A line prepared on one Metal device.
///
/// Wide lines draw each segment as a screen-space strip of three quads: the
/// start region, the body and the end region. Its fragment computes the exact
/// distance to the stroke, so pixel and world widths, miter, round and bevel
/// joins, caps and dashes are analytic and antialiased. Each join is split along
/// its bisector between the two segments that meet there; where a stroke reaches
/// past the bisector over a short neighbour, it adds only what the neighbour
/// leaves uncovered, so two consecutive segments composite to the union of their
/// strokes without gaps or double blending. Caps and joins keep their vertex's
/// depth, and depth and dash distances are perspective-correct along each
/// segment. Segments are trimmed at the near plane in clip space for forward and
/// reversed depth.
public final class LineRenderer {
    public let device: any MTLDevice
    public let colorPixelFormat: MTLPixelFormat
    /// The sample count required of the color and depth attachments.
    public let sampleCount: Int
    public let geometry: LineGeometry
    public let material: LineMaterial
    /// The active window in source-vertex order.
    public private(set) var drawRange: GeometryDrawRange = .all
    private let pipeline: any MTLRenderPipelineState
    private let depthStates: PrimitiveDepthStates
    private let points: any MTLBuffer
    private let colors: (any MTLBuffer)?
    private let fullLines: PreparedLineSegments
    fileprivate private(set) var activeLines: PreparedLineSegments?
    /// The selected segments in submission order, with indices into the original positions.
    public var drawnSegments: [LineSegment] { activeLines?.segments ?? [] }

    fileprivate func recordResources(in inventory: inout SceneResourceInventory) {
        guard inventory.recordGeometry(self) else { return }
        inventory.recordPipeline(pipeline)
        inventory.recordBuffer(points)
        inventory.recordBuffer(colors)
        inventory.recordBuffer(fullLines.buffer)
        inventory.recordBuffer(activeLines?.buffer)
    }

    /// Widths must be finite and positive and a miter limit at least 1. Color
    /// components must be finite and nonnegative with opacity in 0…1.
    public convenience init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat = .rgba16Float, sampleCount: Int = 1,
        geometry: LineGeometry, material: LineMaterial = LineMaterial()
    ) throws {
        try self.init(device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                      geometry: geometry, material: material, sharingGeometryWith: nil)
    }

    private convenience init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, sampleCount: Int,
        geometry: LineGeometry, material: LineMaterial, sharingGeometryWith source: LineRenderer?
    ) throws {
        switch material.width {
        case .hairline: break
        case .pixels(let value), .world(let value):
            guard value.isFinite, value > 0 else { throw PrimitiveRendererError.size }
        }
        if case .miter(let limit) = material.join, !(limit.isFinite && limit >= 1) {
            throw PrimitiveRendererError.miterLimit
        }
        try PrimitivePipeline.validate(color: material.color, alphaTest: 0)
        let library = try PrimitivePipeline.library(device: device)
        let hairline = material.width == .hairline
        let pipeline = try PrimitivePipeline.make(
            device: device, library: library,
            vertex: hairline ? "frostfield_hairline_vertex" : "frostfield_line_vertex",
            fragment: hairline ? "frostfield_hairline_fragment" : "frostfield_line_fragment",
            hasColors: geometry.colors != nil, hasMap: false,
            colorPixelFormat: colorPixelFormat, sampleCount: sampleCount, blending: material.blending, label: "Frostfield lines"
        )
        let depthStates = try PrimitiveDepthStates(device: device, blending: material.blending, depthTest: material.depthTest)
        try self.init(device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                      geometry: geometry, material: material, pipeline: pipeline, depthStates: depthStates,
                      sharingGeometryWith: source)
    }

    private init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, sampleCount: Int,
        geometry: LineGeometry, material: LineMaterial, pipeline: any MTLRenderPipelineState,
        depthStates: PrimitiveDepthStates, sharingGeometryWith source: LineRenderer? = nil
    ) throws {
        self.pipeline = pipeline
        self.depthStates = depthStates
        if let source {
            points = source.points
            colors = source.colors
            fullLines = source.fullLines
        } else {
            points = try PrimitivePipeline.buffer(
                geometry.positions.map { SIMD4($0, 1) }, device: device, label: "Frostfield line points"
            )
            colors = try geometry.colors.map {
                try PrimitivePipeline.buffer($0.map(SIMD4<Float16>.init), device: device, label: "Frostfield line colors")
            }
            fullLines = try Self.makeSegments(geometry.segments, device: device, dynamic: false)
        }
        activeLines = fullLines
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.sampleCount = sampleCount
        self.geometry = geometry
        self.material = material
    }

    func replacingMaterial(_ material: LineMaterial, format: MTLPixelFormat, samples: Int) throws -> LineRenderer {
        try LineRenderer(device: device, colorPixelFormat: format, sampleCount: samples,
                         geometry: geometry, material: material, sharingGeometryWith: self)
    }

    /// Retains the prepared pipeline and immutable buffers of earlier draws.
    /// Debug helpers publish a separate renderer for each geometry snapshot.
    func replacingGeometry(_ geometry: LineGeometry) throws -> LineRenderer {
        let replacement: LineRenderer
        if (self.geometry.colors == nil) == (geometry.colors == nil) {
            replacement = try LineRenderer(
                device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                geometry: geometry, material: material, pipeline: pipeline, depthStates: depthStates
            )
        } else {
            replacement = try LineRenderer(
                device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                geometry: geometry, material: material
            )
        }
        try replacement.setDrawRange(drawRange)
        return replacement
    }

    /// Selects source vertices without replacing the point, color or pipeline
    /// resources. Strip endpoints and loops use the selected vertices for caps
    /// and joins. Segment pairs are formed from the first selected vertex.
    public func setDrawRange(_ range: GeometryDrawRange) throws {
        guard range != drawRange else { return }
        let indices = range.resolve(elementCount: geometry.positions.count,
                                    elementsPerPrimitive: geometry.topology == .segments ? 2 : 1)
        let selected: PreparedLineSegments?
        if indices.count == geometry.positions.count {
            selected = fullLines
        } else {
            let segments = try geometry.segments(in: indices)
            selected = segments.isEmpty ? nil : try Self.makeSegments(segments, device: device, dynamic: true)
        }
        activeLines = selected
        drawRange = range
    }

    private static func makeSegments(
        _ lineSegments: [LineSegment], device: any MTLDevice, dynamic: Bool
    ) throws -> PreparedLineSegments {
        let packed = lineSegments.indices.map { index in
            let segment = lineSegments[index]
            // Strips and loops list their segments in order, so a joined
            // neighbour is the adjacent segment, wrapping around a loop.
            let before = index > 0 ? index - 1 : lineSegments.count - 1
            let after = index + 1 < lineSegments.count ? index + 1 : 0
            let previous = segment.previous.flatMap { lineSegments[before].start == $0 ? before : nil }
            let next = segment.next.flatMap { lineSegments[after].end == $0 ? after : nil }
            return PackedLineSegment(
                points: SIMD4(
                    Int32(segment.previous ?? -1), Int32(segment.start), Int32(segment.end), Int32(segment.next ?? -1)
                ),
                distances: SIMD2(segment.startDistance, segment.endDistance),
                neighbors: SIMD2(Int32(previous ?? -1), Int32(next ?? -1))
            )
        }
        let buffer = try PrimitivePipeline.buffer(packed, device: device, label: "Frostfield line segments")
        return PreparedLineSegments(segments: lineSegments, buffer: buffer, dynamic: dynamic)
    }

    fileprivate func encode(_ draw: PrimitiveDraw, into encoder: any MTLRenderCommandEncoder, depthMode: MeshDepthMode) {
        guard case .lines(let lines) = draw.elements else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthStates.state(for: depthMode))
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(points, offset: 0, index: 0)
        if let colors { encoder.setVertexBuffer(colors, offset: 0, index: 1) }
        encoder.setVertexBuffer(lines.buffer, offset: 0, index: 2)
        var view = draw.view
        encoder.setVertexBytes(&view, length: MemoryLayout<PackedPrimitiveView>.stride, index: 3)
        var style = PackedLineStyle(material: material, tint: draw.tint, sampleCount: sampleCount)
        encoder.setVertexBytes(&style, length: MemoryLayout<PackedLineStyle>.stride, index: 4)
        encoder.setFragmentBytes(&style, length: MemoryLayout<PackedLineStyle>.stride, index: 0)
        let count = lines.segments.count
        if material.width == .hairline {
            encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: count * 2)
        } else {
            // Three quads per segment in one strip: start region, body and end region.
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 8, instanceCount: count)
        }
    }
}

struct PreparedLineSegments: @unchecked Sendable {
    let segments: [LineSegment]
    let buffer: any MTLBuffer
    let dynamic: Bool
}

// MARK: - Sprites

/// The appearance shared by a batch of sprites, like Three.js `SpriteMaterial`.
public struct SpriteMaterial {
    /// An optional straight-alpha color texture, sampled with each sprite's region.
    public var map: (any MTLTexture)?
    /// Linear RGB and opacity, with straight alpha.
    public var color: SIMD4<Float>
    /// True sizes sprites in world units, so they shrink with distance; false
    /// sizes them in pixels. Three.js keeps a non-attenuated sprite at the size it
    /// would have one unit from the camera; Frostfield uses pixels instead.
    public var sizeAttenuation: Bool
    /// Pixels whose material alpha falls below this value are discarded.
    public var alphaTest: Float
    /// Sprites in one renderer draw in array order; blending sorts the batch as a whole.
    public var blending: PrimitiveBlending
    /// When false the sprites draw over everything, after every depth-tested draw.
    public var depthTest: Bool
    /// Optional intersection fade against `SceneGraph.viewportSnapshot` depth.
    /// Use alpha or additive blending.
    public var softParticleFade: SoftParticleDepth.Fade?

    public init(
        map: (any MTLTexture)? = nil, color: SIMD4<Float> = SIMD4(repeating: 1), sizeAttenuation: Bool = true,
        alphaTest: Float = 0, blending: PrimitiveBlending = .alpha, depthTest: Bool = true,
        softParticleFade: SoftParticleDepth.Fade? = nil
    ) {
        self.map = map
        self.color = color
        self.sizeAttenuation = sizeAttenuation
        self.alphaTest = alphaTest
        self.blending = blending
        self.depthTest = depthTest
        self.softParticleFade = softParticleFade
    }
}

/// Sprites prepared on one Metal device and drawn in one instanced draw. Each
/// sprite lies in a plane parallel to the image plane at its anchor's depth, as
/// Three.js sprites do, and rotates on screen about its anchor.
public final class SpriteRenderer {
    public static let maximumSpriteCount = 1 << 24

    public let device: any MTLDevice
    public let colorPixelFormat: MTLPixelFormat
    /// The sample count required of the color and depth attachments.
    public let sampleCount: Int
    public let sprites: [Sprite]
    public let material: SpriteMaterial
    /// The box around every sprite anchor.
    public let bounds: AxisAlignedBox
    /// The largest anchor-to-corner distance, in the sprites' size units.
    let maximumReach: Float
    private let pipeline: any MTLRenderPipelineState
    private let depthStates: PrimitiveDepthStates
    private let records: any MTLBuffer

    fileprivate func recordResources(in inventory: inout SceneResourceInventory) {
        guard inventory.recordGeometry(self) else { return }
        inventory.recordPipeline(pipeline)
        inventory.recordBuffer(records)
        inventory.recordTexture(material.map)
    }

    /// Supply 1 to `maximumSpriteCount` sprites with finite fields, nonnegative
    /// sizes and colors that follow ``PointCloud`` rules.
    public convenience init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat = .rgba16Float, sampleCount: Int = 1,
        sprites: [Sprite], material: SpriteMaterial = SpriteMaterial()
    ) throws {
        try self.init(device: device, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount,
                      sprites: sprites, material: material, sharingGeometryWith: nil)
    }

    private init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, sampleCount: Int,
        sprites: [Sprite], material: SpriteMaterial, sharingGeometryWith source: SpriteRenderer?
    ) throws {
        if source == nil {
            guard (1...Self.maximumSpriteCount).contains(sprites.count) else { throw PrimitiveRendererError.spriteCount }
            if let index = sprites.firstIndex(where: { !$0.isValid }) { throw PrimitiveRendererError.invalidSprite(index) }
        }
        try PrimitivePipeline.validate(color: material.color, alphaTest: material.alphaTest)
        try PrimitivePipeline.validate(map: material.map, device: device)
        try PrimitivePipeline.validate(softParticleFade: material.softParticleFade, blending: material.blending)
        let library = try PrimitivePipeline.library(device: device)
        pipeline = try PrimitivePipeline.make(
            device: device, library: library, vertex: "frostfield_sprites_vertex", fragment: "frostfield_sprites_fragment",
            hasColors: false, hasMap: material.map != nil, hasSoftParticles: material.softParticleFade != nil,
            colorPixelFormat: colorPixelFormat, sampleCount: sampleCount, blending: material.blending, label: "Frostfield sprites"
        )
        depthStates = try PrimitiveDepthStates(device: device, blending: material.blending, depthTest: material.depthTest)
        records = try source?.records ?? PrimitivePipeline.buffer(
            sprites.map(PackedSpriteRecord.init), device: device, label: "Frostfield sprites"
        )
        bounds = try source?.bounds ?? AxisAlignedBox(enclosing: sprites.map(\.position))
        maximumReach = source?.maximumReach ?? sprites.map(\.reach).max() ?? 0
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.sampleCount = sampleCount
        self.sprites = sprites
        self.material = material
    }

    func replacingMaterial(_ material: SpriteMaterial, format: MTLPixelFormat, samples: Int) throws -> SpriteRenderer {
        try SpriteRenderer(device: device, colorPixelFormat: format, sampleCount: samples,
                           sprites: sprites, material: material, sharingGeometryWith: self)
    }

    fileprivate func encode(
        _ draw: PrimitiveDraw, into encoder: any MTLRenderCommandEncoder, depthMode: MeshDepthMode,
        softParticles: PrimitiveSoftParticleFrame?
    ) {
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthStates.state(for: depthMode))
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(records, offset: 0, index: 0)
        var view = draw.view
        encoder.setVertexBytes(&view, length: MemoryLayout<PackedPrimitiveView>.stride, index: 1)
        var style = PackedSpriteStyle(
            color: material.color * draw.tint,
            parameters: SIMD4(material.sizeAttenuation ? 1 : 0, material.alphaTest, 0, 0)
        )
        encoder.setVertexBytes(&style, length: MemoryLayout<PackedSpriteStyle>.stride, index: 2)
        encoder.setFragmentBytes(&style, length: MemoryLayout<PackedSpriteStyle>.stride, index: 0)
        if let map = material.map { encoder.setFragmentTexture(map, index: 0) }
        if let fade = material.softParticleFade { softParticles?.bind(fade, into: encoder) }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: sprites.count)
    }
}

/// Failures preparing or encoding point, line and sprite renderers.
public enum PrimitiveRendererError: LocalizedError, Equatable {
    case size
    case color
    case alphaTest
    case map
    case mapAliasing
    case softParticleBlending
    case dash
    case miterLimit
    case spriteCount
    case invalidSprite(Int)
    case library(String)
    case function(String)
    case pipeline(String)
    case resource(String)
    case projection

    public var errorDescription: String? {
        switch self {
        case .size: "Point sizes and line widths must be finite and positive."
        case .color: "Primitive colors need finite, nonnegative RGB up to 65,504 and opacity in 0 to 1."
        case .alphaTest: "The alpha test must be in 0 to 1."
        case .map: "A primitive map must be a shader-readable 2D color texture on the renderer's device."
        case .mapAliasing: "An override map cannot share storage with the color attachment or its resolve texture."
        case .softParticleBlending: "Soft points and sprites require alpha or additive blending."
        case .dash: "A dash needs a finite, positive length and gap and a finite offset."
        case .miterLimit: "A miter limit must be finite and at least 1."
        case .spriteCount: "Supply 1 to \(SpriteRenderer.maximumSpriteCount) sprites."
        case .invalidSprite(let index):
            "Sprite \(index) needs finite fields, a nonnegative size and a valid color."
        case .library(let message): "Could not load Frostfield’s shader library: \(message)"
        case .function(let name): "The shader library does not contain \(name)."
        case .pipeline(let message): "Could not create the primitive pipeline: \(message)"
        case .resource(let name): "Metal could not allocate \(name)."
        case .projection: "The view's projection has no usable vertical scale for pixel-sized primitives."
        }
    }
}

// MARK: - Scene graph encoding

/// One view's constants shared by every primitive draw in a pass.
struct PrimitiveView {
    private struct RelativeView {
        let transforms: HighPrecisionModelView
        let projection: simd_float4x4
        let frustum: ViewFrustum?
    }

    let clipFromWorld: simd_float4x4
    /// Origin and size in framebuffer pixels.
    let viewport: SIMD4<Float>
    /// Pixels per world unit at clip w = 1.
    let pixelsPerUnit: Float
    let depthMode: MeshDepthMode
    private let relativeView: RelativeView?

    init(
        clipFromWorld: simd_float4x4, viewport: MTLViewport?, attachment: any MTLTexture, depthMode: MeshDepthMode,
        precision: HighPrecisionModelView?, orthographicBack: SIMD3<Float>? = nil
    ) throws {
        let region = viewport.map {
            SIMD4(Float($0.originX), Float($0.originY), Float($0.width), Float($0.height))
        } ?? SIMD4(0, 0, Float(attachment.width), Float(attachment.height))
        guard let scale = Self.verticalScale(of: clipFromWorld, orthographicBack: orthographicBack) else { throw PrimitiveRendererError.projection }
        self.clipFromWorld = clipFromWorld
        self.viewport = region
        pixelsPerUnit = Float(scale * Double(region.w) / 2)
        self.depthMode = depthMode
        relativeView = precision.map {
            let projection = $0.clipMatrix(for: matrix_identity_double4x4)
            return RelativeView(transforms: $0, projection: projection, frustum: try? ViewFrustum(clipFromWorld: projection))
        }
    }

    /// The projection's vertical focal scale (P₁₁) for a rigid camera: the clip y
    /// row with any depth coupling removed, relative to the clip w row. An
    /// orthographic matrix has no w row; its depth row stands in for the coupling.
    static func verticalScale(
        of clipFromWorld: simd_float4x4, orthographicBack: SIMD3<Float>? = nil
    ) -> Double? {
        func row(_ index: Int) -> SIMD3<Double> {
            let (x, y, z, _) = clipFromWorld.columns
            return SIMD3(Double(x[index]), Double(y[index]), Double(z[index]))
        }
        let y = row(1)
        let w = row(3)
        let perspective = simd_length_squared(w) > 1e-24
        let reference = perspective ? w : orthographicBack.map(SIMD3<Double>.init) ?? row(2)
        let referenceLength = simd_length_squared(reference)
        let perpendicular = referenceLength > 0 ? y - reference * (simd_dot(y, reference) / referenceLength) : y
        let scale = simd_length(perpendicular) / (perspective ? simd_length(w) : 1)
        return scale.isFinite && scale > 0 ? scale : nil
    }

    /// The draw for one node, or nil when its world bounds are outside the frustum.
    func draw(
        _ primitive: ScenePrimitive, elements: PrimitiveDrawElements, node: ResolvedSceneNode, nodeIndex: Int,
        frustum: ViewFrustum?, device: any MTLDevice, color: any MTLTexture, recordsOpaqueDepth: Bool,
        overrideTint: SIMD4<Float>?
    ) throws -> PrimitiveDraw? {
        guard primitive.device === device else { throw MeshRenderer.Failure.wrongDevice }
        guard primitive.colorPixelFormat == color.pixelFormat else { throw MeshRenderer.Failure.attachments }
        guard primitive.sampleCount == color.sampleCount else { throw MeshRenderer.Failure.sampleCount }
        let localBounds = node.cullingBounds ?? primitive.localBounds
        let clipFromLocal: simd_float4x4
        let anchors: AxisAlignedBox
        let boundsProjection: simd_float4x4
        let cullingFrustum: ViewFrustum?
        if let relativeView, let world = relativeView.transforms.relativeWorldTransforms[node.id] {
            clipFromLocal = relativeView.transforms.clipMatrix(for: world)
            anchors = try localBounds.transformed(by: world)
            boundsProjection = relativeView.projection
            cullingFrustum = relativeView.frustum
        } else {
            clipFromLocal = clipFromWorld * node.worldTransform
            anchors = try localBounds.transformed(by: node.worldTransform)
            boundsProjection = clipFromWorld
            cullingFrustum = frustum
        }
        let tint = overrideTint ?? node.materialParameters
        guard primitiveFinite(clipFromLocal), tint.x.isFinite, tint.y.isFinite, tint.z.isFinite, tint.w.isFinite else {
            throw MeshRenderer.Failure.transform
        }
        let padding = SIMD3<Float>(repeating: node.cullingBounds == nil ? primitive.worldPadding : 0)
        let bounds = try AxisAlignedBox(minimum: anchors.minimum - padding, maximum: anchors.maximum + padding)
        if node.frustumCulled, let cullingFrustum, cullingFrustum.classify(box: bounds) == .outside { return nil }
        let center = boundsProjection * SIMD4(bounds.center, 1)
        let phase: SceneColorOrder.Phase = !primitive.depthTest ? .overlay : (primitive.blending == .opaque ? .opaque : .blended)
        let kind: SceneRenderOrdering.Kind = switch primitive {
        case .points: .points
        case .lines: .lines
        case .sprites: .sprites
        }
        let order = SceneColorOrder(
            phase: phase, kind: kind, node: node, nodeIndex: nodeIndex, clipPosition: center, depthMode: depthMode,
            recordsOpaqueDepth: recordsOpaqueDepth
        )
        return PrimitiveDraw(
            primitive: primitive, elements: elements,
            view: PackedPrimitiveView(
                clipFromLocal: clipFromLocal, viewport: viewport,
                projection: SIMD4(pixelsPerUnit, depthMode == .reversed ? 1 : 0, 0, 0)
            ),
            tint: tint, order: order, isMaterialOverride: overrideTint != nil
        )
    }
}

/// One primitive node's draw in a scene graph color pass.
struct PrimitiveDraw {
    let primitive: ScenePrimitive
    let elements: PrimitiveDrawElements
    let view: PackedPrimitiveView
    let tint: SIMD4<Float>
    let order: SceneColorOrder
    let isMaterialOverride: Bool

    var isBlended: Bool { order.phase == .blended }
    var isOverlay: Bool { order.phase == .overlay }

    func encode(
        into encoder: any MTLRenderCommandEncoder, depthMode: MeshDepthMode, softParticles: PrimitiveSoftParticleFrame?
    ) {
        primitive.encode(self, into: encoder, depthMode: depthMode, softParticles: softParticles)
    }

    func retainBuffersIfNeeded(on commandBuffer: any MTLCommandBuffer) {
        guard !commandBuffer.retainedReferences else { return }
        if isMaterialOverride {
            let retained = RetainedMaterial(primitive: primitive)
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        }
        if case .lines(let lines) = elements, lines.dynamic {
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(lines.buffer) {} }
        }
    }

    private struct RetainedMaterial: @unchecked Sendable {
        let primitive: ScenePrimitive
    }
}

enum PrimitiveDrawElements {
    case points(Range<Int>)
    case lines(PreparedLineSegments)
    case sprites(Int)
}

// MARK: - Pipelines and packed records

/// View constants shared by soft point and sprite draws.
struct PrimitiveSoftParticleFrame {
    let depth: any MTLTexture
    let projection: SIMD4<Float>
    let target: SIMD4<Float>
    let clear: Float

    init(depth: any MTLTexture, view: GraphViewContext) throws {
        self.depth = depth
        projection = try view.softParticleProjection()
        target = SIMD4(lowHalf: view.targetSize, highHalf: view.windowDepth)
        clear = view.depthMode == .reversed ? 0 : 1
    }

    func bind(_ fade: SoftParticleDepth.Fade, into encoder: any MTLRenderCommandEncoder) {
        // Three float4 slots, matching FrostfieldPrimitiveSoftParticle in MSL.
        var parameters = (projection, target, SIMD4(fade.distance, fade.contrast, clear, 0))
        encoder.setFragmentBytes(&parameters, length: MemoryLayout.size(ofValue: parameters), index: 1)
        encoder.setFragmentTexture(depth, index: 1)
    }
}

/// Matches `FrostfieldPrimitiveView` in Primitives.metal: 96 bytes.
struct PackedPrimitiveView {
    var clipFromLocal: simd_float4x4
    var viewport: SIMD4<Float>
    var projection: SIMD4<Float>
}

/// Matches `FrostfieldPointStyle`: 48 bytes.
struct PackedPointStyle {
    var color: SIMD4<Float>
    var size: SIMD4<Float>
    var flags: SIMD4<Float>

    init(material: PointsMaterial, tint: SIMD4<Float>) {
        color = material.color * tint
        let value: Float
        let world: Float
        switch material.size {
        case .pixels(let pixels): (value, world) = (pixels, 0)
        case .world(let units): (value, world) = (units, 1)
        }
        size = SIMD4(value, world, material.shape == .circle ? 1 : 0, material.alphaTest)
        flags = SIMD4(material.blending == .opaque ? 0 : 1, 0, 0, 0)
    }
}

/// Matches `FrostfieldSpriteRecord`: 64 bytes.
struct PackedSpriteRecord {
    var positionRotation: SIMD4<Float>
    var sizeCenter: SIMD4<Float>
    var color: SIMD4<Float>
    var region: SIMD4<Float>

    init(_ sprite: Sprite) {
        positionRotation = SIMD4(sprite.position, sprite.rotation)
        sizeCenter = SIMD4(sprite.size.x, sprite.size.y, sprite.center.x, sprite.center.y)
        color = sprite.color
        region = sprite.textureRegion
    }
}

/// Matches `FrostfieldSpriteStyle`: 32 bytes.
struct PackedSpriteStyle {
    var color: SIMD4<Float>
    var parameters: SIMD4<Float>
}

/// Matches `FrostfieldLineSegment`: 32 bytes.
struct PackedLineSegment {
    var points: SIMD4<Int32>
    var distances: SIMD2<Float>
    /// The segments joined at the start and end, or −1.
    var neighbors: SIMD2<Int32>
}

/// Matches `FrostfieldLineStyle`: 64 bytes.
struct PackedLineStyle {
    var color: SIMD4<Float>
    var width: SIMD4<Float>
    var shape: SIMD4<Float>
    var dash: SIMD4<Float>

    init(material: LineMaterial, tint: SIMD4<Float>, sampleCount: Int) {
        color = material.color * tint
        var halfWidth: Float = 0.5
        var world: Float = 0
        switch material.width {
        case .hairline: break
        case .pixels(let pixels): halfWidth = pixels / 2
        case .world(let units): (halfWidth, world) = (units / 2, 1)
        }
        var limit: Float = 1
        let join: Float
        switch material.join {
        case .miter(let value): (join, limit) = (0, value)
        case .round: join = 1
        case .bevel: join = 2
        }
        let cap: Float = switch material.cap {
        case .butt: 0
        case .round: 1
        case .square: 2
        }
        width = SIMD4(halfWidth, world, material.blending == .opaque ? 0 : 1, limit)
        // Multisample positions may extend past the fragment's pixel center.
        let rasterPadding: Float = sampleCount > 1 ? 1 : (material.blending == .opaque ? 0.01 : 0.51)
        shape = SIMD4(join, cap, material.blending == .additive ? 1 : 0, rasterPadding)
        dash = material.dash.map { SIMD4($0.length, $0.gap, $0.offset, 1) } ?? .zero
    }
}

/// Forward and reversed depth states for one blending mode.
struct PrimitiveDepthStates {
    private let forward: any MTLDepthStencilState
    private let reversed: any MTLDepthStencilState

    init(device: any MTLDevice, blending: PrimitiveBlending, depthTest: Bool) throws {
        func make(_ compare: MTLCompareFunction) throws -> any MTLDepthStencilState {
            let descriptor = MTLDepthStencilDescriptor()
            descriptor.depthCompareFunction = depthTest ? compare : .always
            descriptor.isDepthWriteEnabled = depthTest && blending == .opaque
            guard let state = device.makeDepthStencilState(descriptor: descriptor) else {
                throw PrimitiveRendererError.resource("primitive depth state")
            }
            return state
        }
        // Equal depth passes so lines drawn over coincident mesh edges stay visible.
        forward = try make(.lessEqual)
        reversed = try make(.greaterEqual)
    }

    func state(for mode: MeshDepthMode) -> any MTLDepthStencilState {
        mode == .forward ? forward : reversed
    }
}

enum PrimitivePipeline {
    static func library(device: any MTLDevice) throws -> any MTLLibrary {
        do {
            return try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw PrimitiveRendererError.library(error.localizedDescription)
        }
    }

    static func make(
        device: any MTLDevice, library: any MTLLibrary, vertex: String, fragment: String,
        hasColors: Bool, hasMap: Bool, hasSoftParticles: Bool = false, colorPixelFormat: MTLPixelFormat, sampleCount: Int,
        blending: PrimitiveBlending, label: String
    ) throws -> any MTLRenderPipelineState {
        guard RendererCapabilities.supportsRasterSampleCount(sampleCount, on: device) else {
            throw MeshRenderer.Failure.sampleCount
        }
        let constants = MTLFunctionConstantValues()
        var colors = hasColors
        var map = hasMap
        var softParticles = hasSoftParticles
        constants.setConstantValue(&colors, type: .bool, index: 20)
        constants.setConstantValue(&map, type: .bool, index: 21)
        constants.setConstantValue(&softParticles, type: .bool, index: 22)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = label
        do {
            descriptor.vertexFunction = try library.makeFunction(name: vertex, constantValues: constants)
            descriptor.fragmentFunction = try library.makeFunction(name: fragment, constantValues: constants)
        } catch {
            throw PrimitiveRendererError.function("\(vertex) or \(fragment): \(error.localizedDescription)")
        }
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        descriptor.rasterSampleCount = sampleCount
        descriptor.depthAttachmentPixelFormat = MeshRenderer.depthPixelFormat
        if blending != .opaque {
            let color = descriptor.colorAttachments[0]!
            color.isBlendingEnabled = true
            color.sourceRGBBlendFactor = .one
            color.sourceAlphaBlendFactor = .one
            color.destinationRGBBlendFactor = blending == .additive ? .one : .oneMinusSourceAlpha
            color.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        }
        do {
            return try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw PrimitiveRendererError.pipeline(error.localizedDescription)
        }
    }

    static func validate(color: SIMD4<Float>, alphaTest: Float) throws {
        guard isValidPrimitiveColor(color) else { throw PrimitiveRendererError.color }
        guard alphaTest.isFinite, (0...1).contains(alphaTest) else { throw PrimitiveRendererError.alphaTest }
    }

    static func validate(softParticleFade: SoftParticleDepth.Fade?, blending: PrimitiveBlending) throws {
        if softParticleFade != nil, blending == .opaque { throw PrimitiveRendererError.softParticleBlending }
    }

    static func validate(map: (any MTLTexture)?, device: any MTLDevice) throws {
        guard let map else { return }
        guard map.device === device else { throw MeshRenderer.Failure.wrongDevice }
        guard map.textureType == .type2D, !map.isFramebufferOnly,
              map.usage.isEmpty || map.usage.contains(.shaderRead) else { throw PrimitiveRendererError.map }
    }

    static func buffer<T>(_ values: [T], device: any MTLDevice, label: String) throws -> any MTLBuffer {
        let buffer = values.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared) }
        }
        guard let buffer else { throw PrimitiveRendererError.resource(label) }
        buffer.label = label
        return buffer
    }
}

func primitiveFinite(_ matrix: simd_float4x4) -> Bool {
    let (x, y, z, w) = matrix.columns
    return [x, y, z, w].allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }
}
