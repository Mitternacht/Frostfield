import CoreText
import Foundation
import Metal
import simd

/// A camera-orientation overlay for an existing Metal scene color target.
public final class AxisViewHelper {
    public enum Axis: Int, CaseIterable, Sendable {
        case positiveX, positiveY, positiveZ, negativeX, negativeY, negativeZ

        /// The direction from the orbit target toward a camera aligned to this axis.
        public var direction: SIMD3<Float> {
            switch self {
            case .positiveX: SIMD3(1, 0, 0)
            case .positiveY: SIMD3(0, 1, 0)
            case .positiveZ: SIMD3(0, 0, 1)
            case .negativeX: SIMD3(-1, 0, 0)
            case .negativeY: SIMD3(0, -1, 0)
            case .negativeZ: SIMD3(0, 0, -1)
            }
        }

        var isPositive: Bool { rawValue < Self.negativeX.rawValue }

        var orbitAngles: (yaw: Double, pitch: Float) {
            switch self {
            case .positiveX: (.pi / 2, 0)
            case .positiveY: (0, .pi / 2)
            case .positiveZ: (0, 0)
            case .negativeX: (-.pi / 2, 0)
            case .negativeY: (0, -.pi / 2)
            case .negativeZ: (.pi, 0)
            }
        }

        var color: SIMD4<Float> {
            switch self {
            case .positiveX: LineGeometry.helperColor(0xff4466)
            case .positiveY: LineGeometry.helperColor(0x88ff44)
            case .positiveZ: LineGeometry.helperColor(0x4488ff)
            default: SIMD4(0, 0, 0, 0.2)
            }
        }
    }

    public struct Endpoint: Equatable, Sendable {
        public let axis: Axis
        /// The center of the marker in top-left destination pixels.
        public let position: SIMD2<Float>
        /// The marker's camera-space +Z coordinate. Larger is closer.
        public let depth: Float
    }

    /// A one-line glyph mask prepared once for reuse on a signed endpoint.
    public struct Label {
        fileprivate let glyphs: TextureDebugPass.Label
        fileprivate let color: SIMD4<Float>

        /// The font is in pixels for a 128-pixel overlay. Long labels and larger
        /// fonts scale to fit the marker. Supply straight linear RGBA.
        public init(
            _ text: String, device: any MTLDevice, font: CTFont? = nil,
            color: SIMD4<Float> = SIMD4(0, 0, 0, 1)
        ) throws {
            let limit = Float(Float16.greatestFiniteMagnitude)
            guard (0...limit).contains(color.x), (0...limit).contains(color.y), (0...limit).contains(color.z),
                  (0...1).contains(color.w) else { throw Failure.labelColor }
            self.glyphs = try TextureDebugPass.Label(text, device: device, font: font)
            self.color = color
        }
    }

    /// One camera orientation and overlay region shared by drawing and hit testing.
    public struct Layout: Equatable, Sendable {
        public let origin: SIMD2<Int>
        public let size: Int
        /// Endpoints in back-to-front paint order. Ties follow axis declaration order.
        public let endpoints: [Endpoint]

        /// Radius of both the rendered discs and their hit regions, in pixels.
        public let markerRadius: Float

        /// The square region uses top-left destination pixels. The rotation is the
        /// camera's local-to-world orientation, including any roll.
        public init(cameraRotation: simd_quatf, origin: SIMD2<Int>, size: Int = 128) throws {
            try self.init(cameraRotation: cameraRotation, origin: origin, size: size,
                          markerRadius: Float(size) * (7.0 / 128))
        }

        /// Uses a custom disc and hit radius, up to a quarter of the square width.
        public init(
            cameraRotation: simd_quatf, origin: SIMD2<Int>, size: Int = 128, markerRadius: Float
        ) throws {
            let pixelLimit = 1 << 24
            guard (1...pixelLimit).contains(size), origin.x >= 0, origin.y >= 0,
                  origin.x <= pixelLimit - size, origin.y <= pixelLimit - size else { throw Failure.region }
            let vector = SIMD4<Double>(cameraRotation.vector)
            guard vector.x.isFinite, vector.y.isFinite, vector.z.isFinite, vector.w.isFinite, vector != .zero else {
                throw Failure.rotation
            }
            guard markerRadius > 0, markerRadius <= Float(size) * 0.25 else { throw Failure.markerRadius }
            let inverse = simd_quatd(vector: simd_normalize(vector)).inverse
            let center = SIMD2<Float>(Float(origin.x), Float(origin.y)) + SIMD2(repeating: Float(size) * 0.5)
            self.origin = origin
            self.size = size
            self.markerRadius = markerRadius
            endpoints = Axis.allCases.map { axis in
                let view = inverse.act(SIMD3<Double>(axis.direction))
                let offset = SIMD2(Float(view.x), -Float(view.y)) * (Float(size) * 0.25)
                return Endpoint(axis: axis, position: center + offset, depth: Float(view.z))
            }.sorted {
                $0.depth == $1.depth ? $0.axis.rawValue < $1.axis.rawValue : $0.depth < $1.depth
            }
        }

        /// Returns the closest visible marker containing the point. Stems and
        /// the antialiasing fringe are outside the hit region.
        public func hitTest(at pixel: SIMD2<Float>) -> Axis? {
            let radiusSquared = markerRadius * markerRadius
            return endpoints.reversed().first {
                simd_length_squared(pixel - $0.position) <= radiusSquared
            }?.axis
        }

        fileprivate var pixelOrigin: SIMD2<Float> { SIMD2(Float(origin.x), Float(origin.y)) }
    }

    private struct Mark {
        // Three float4 values match AxisViewMark in TextureDebugPass.metal.
        let points: SIMD4<Float>
        let color: SIMD4<Float>
        let dimensions: SIMD4<Float>

        init(
            from a: SIMD2<Float>, to b: SIMD2<Float>, color: SIMD4<Float>, radius: Float, size: Float,
            glyphSize: SIMD2<Float> = .zero
        ) {
            points = SIMD4(a.x, a.y, b.x, b.y)
            self.color = color
            dimensions = SIMD4(radius, size, glyphSize.x, glyphSize.y)
        }
    }

    public let device: any MTLDevice
    public let pixelFormat: MTLPixelFormat
    private let pipeline: any MTLRenderPipelineState
    private let labeledPipeline: any MTLRenderPipelineState

    public init(device: any MTLDevice, pixelFormat: MTLPixelFormat = .rgba16Float) throws {
        guard PostProcessPass.supportedPixelFormats.contains(pixelFormat) else { throw Failure.pixelFormat }
        self.device = device
        self.pixelFormat = pixelFormat
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_axis_view_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_axis_view_fragment"),
                  let labeled = library.makeFunction(name: "frostfield_axis_view_labeled") else { throw Failure.missingShader }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield axis view helper"
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = pixelFormat
            MeshRenderState.transparent.configure(descriptor)
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.label = "Frostfield axis view labels"
            descriptor.fragmentFunction = labeled
            labeledPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let error as Failure {
            throw error
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Blends the overlay into an existing single-sample color target after its
    /// scene or display pass. The destination is loaded and stored with no depth
    /// attachment. Use an uncommitted command buffer with retained references.
    public func encode(layout: Layout, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws {
        try encode(layout: layout, labels: [:], destination: destination, commandBuffer: commandBuffer)
    }

    /// Draws optional endpoint labels in the same paint order as the markers.
    /// The dictionary may contain either sign of any axis.
    public func encode(
        layout: Layout, labels: [Axis: Label], destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else { throw Failure.commandBuffer }
        guard commandBuffer.device === device, destination.device === device,
              labels.values.allSatisfy({ $0.glyphs.mask.device === device }) else { throw Failure.wrongDevice }
        guard destination.pixelFormat == pixelFormat, destination.textureType == .type2D, destination.sampleCount == 1,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              destination.hazardTrackingMode != .untracked else { throw Failure.destination }
        let origin = layout.origin
        let size = layout.size
        guard size <= destination.width - origin.x, size <= destination.height - origin.y else { throw Failure.region }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield axis view helper"
        encoder.setViewport(MTLViewport(
            originX: Double(origin.x), originY: Double(origin.y),
            width: Double(size), height: Double(size), znear: 0, zfar: 1
        ))
        encoder.setScissorRect(MTLScissorRect(x: origin.x, y: origin.y, width: size, height: size))
        encoder.setRenderPipelineState(pipeline)

        let width = Float(size)
        let center = SIMD2<Float>(repeating: width * 0.5)
        let stems = layout.endpoints.filter { $0.axis.isPositive }.map { endpoint in
            let end = endpoint.position - layout.pixelOrigin
            return Mark(from: center, to: center + (end - center) * 0.8,
                        color: endpoint.axis.color, radius: width * 0.01, size: width)
        }
        func mark(_ endpoint: Endpoint, glyphSize: SIMD2<Float> = .zero) -> Mark {
            let pixel = endpoint.position - layout.pixelOrigin
            return Mark(from: pixel, to: pixel, color: endpoint.axis.color, radius: layout.markerRadius,
                        size: width, glyphSize: glyphSize)
        }
        if labels.isEmpty {
            Self.draw(stems + layout.endpoints.map { mark($0) }, into: encoder)
        } else {
            Self.draw(stems, into: encoder)
            for endpoint in layout.endpoints {
                if let label = labels[endpoint.axis] {
                    let glyphs = label.glyphs
                    let natural = SIMD2(Float(glyphs.width), Float(glyphs.height))
                    let maximum = layout.markerRadius * Float(2).squareRoot()
                    let scale = min(width / 128, maximum / simd_reduce_max(natural))
                    var item = mark(endpoint, glyphSize: natural * scale)
                    var ink = label.color
                    encoder.setRenderPipelineState(labeledPipeline)
                    encoder.setFragmentTexture(glyphs.mask, index: 0)
                    encoder.setFragmentBytes(&ink, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                    encoder.setVertexBytes(&item, length: MemoryLayout<Mark>.stride, index: 0)
                } else {
                    var item = mark(endpoint)
                    encoder.setRenderPipelineState(pipeline)
                    encoder.setVertexBytes(&item, length: MemoryLayout<Mark>.stride, index: 0)
                }
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
        }
        encoder.endEncoding()
    }

    private static func draw(_ marks: [Mark], into encoder: any MTLRenderCommandEncoder) {
        marks.withUnsafeBytes {
            encoder.setVertexBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: marks.count)
    }

    public enum Failure: LocalizedError, Equatable {
        case rotation
        case region
        case markerRadius
        case labelColor
        case pixelFormat
        case missingShader
        case pipeline(String)
        case commandBuffer
        case wrongDevice
        case destination
        case encoder

        public var errorDescription: String? {
            switch self {
            case .rotation: "The axis view needs a finite nonzero camera quaternion."
            case .region: "The axis view needs a positive square region inside the destination."
            case .markerRadius: "The marker radius must be positive and at most one quarter of the overlay width."
            case .labelColor: "An axis label needs finite linear RGB and alpha between zero and one."
            case .pixelFormat: "The axis view needs a linear RGBA16Float or an 8-bit sRGB target."
            case .missingShader: "The axis view Metal functions were missing."
            case .pipeline(let reason): "Could not prepare the axis view pipeline. \(reason)"
            case .commandBuffer: "The axis view needs an uncommitted command buffer with retained references."
            case .wrongDevice: "The axis view, labels and destination must use the command buffer's Metal device."
            case .destination: "The axis view needs a matching single-sample, hazard-tracked 2D render target."
            case .encoder: "Could not make an axis view render encoder."
            }
        }
    }
}
