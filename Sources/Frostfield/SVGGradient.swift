import CoreGraphics
import Foundation
import Metal
import simd

public extension SVGLoader {
    struct Gradient: Equatable, Sendable {
        public enum Kind: Equatable, Sendable {
            case linear(start: SIMD2<Double>, end: SIMD2<Double>)
            case radial(center: SIMD2<Double>, focal: SIMD2<Double>, radius: Double)
        }

        public enum Units: Equatable, Sendable { case objectBoundingBox, userSpaceOnUse }
        public enum Spread: Equatable, Sendable { case pad, `repeat`, reflect }

        public struct Stop: Equatable, Sendable {
            public let offset: Float
            /// Straight-alpha linear sRGB, including stop opacity.
            public let color: SIMD4<Float>
        }

        public let kind: Kind
        public let units: Units
        public let spread: Spread
        public let transform: CGAffineTransform
        public let stops: [Stop]
        public let interpolatesInLinearRGB: Bool
    }

    struct PreparedGradientFill {
        public let geometry: MeshGeometry
        /// Shader-readable straight-alpha linear RGBA16Float, including fill opacity.
        public let texture: any MTLTexture
    }
}

public extension SVGLoader.Element {
    /// Evaluates a gradient fill in authored SVG document coordinates.
    func gradientColor(at point: CGPoint) throws -> SIMD4<Float>? {
        guard let sampler = try gradientSampler() else { return nil }
        guard point.x.isFinite, point.y.isFinite else { throw SVGLoader.Failure.gradient("sample point") }
        return sampler.color(at: point)
    }

    /// Prepares a gradient texture in the normalized cap UVs of the matching mesh.
    /// Resolution is the width and height of the sampled field, in 1...1024.
    /// Use a textured mesh material and alpha blending to draw the result.
    func prepareGradientFill(
        device: any MTLDevice, resolution: Int = 256, tolerance: Float = 0.25, yUp: Bool = false
    ) throws -> SVGLoader.PreparedGradientFill? {
        guard (1...1_024).contains(resolution) else { throw SVGLoader.Failure.gradient("texture resolution") }
        guard case .gradient = fill, let geometry = try fillGeometry(tolerance: tolerance, yUp: yUp),
              let sampler = try gradientSampler() else { return nil }
        let origin = geometry.vertices[0].position
        var lower = origin, upper = origin
        for vertex in geometry.vertices {
            lower = simd_min(lower, vertex.position)
            upper = simd_max(upper, vertex.position)
        }
        let width = Double(upper.x) - Double(lower.x)
        let height = Double(upper.y) - Double(lower.y)
        var pixels: [SIMD4<Float16>] = []
        pixels.reserveCapacity(resolution * resolution)
        for row in 0..<resolution {
            let y = Double(upper.y) - height * (Double(row) + 0.5) / Double(resolution)
            for column in 0..<resolution {
                let x = Double(lower.x) + width * (Double(column) + 0.5) / Double(resolution)
                pixels.append(SIMD4<Float16>(sampler.color(at: CGPoint(x: x, y: yUp ? -y : y))))
            }
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: resolution, height: resolution, mipmapped: false
        )
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw SVGLoader.Failure.gradient("Metal texture allocation")
        }
        texture.label = "Frostfield SVG gradient"
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, resolution, resolution), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: resolution * MemoryLayout<SIMD4<Float16>>.stride
            )
        }
        return SVGLoader.PreparedGradientFill(geometry: geometry, texture: texture)
    }

    private func gradientSampler() throws -> SVGGradientSampler? {
        guard isVisible, fillOpacity > 0, case .gradient(let gradient) = fill, let gradientSource else { return nil }
        return try SVGGradientSampler(
            gradient: gradient, localBounds: gradientSource.bounds, documentTransform: gradientSource.transform,
            opacity: fillOpacity
        )
    }
}

private struct SVGGradientSampler {
    typealias Gradient = SVGLoader.Gradient

    let kind: Gradient.Kind
    let spread: Gradient.Spread
    let inverse: CGAffineTransform
    let stops: [Gradient.Stop]
    let linearRGB: Bool
    let opacity: Float

    init(gradient: Gradient, localBounds: CGRect, documentTransform: CGAffineTransform, opacity: Float) throws {
        var placement = gradient.transform
        if gradient.units == .objectBoundingBox {
            guard localBounds.width > 0, localBounds.height > 0 else {
                throw SVGLoader.Failure.gradient("empty object bounds")
            }
            let box = CGAffineTransform(
                a: localBounds.width, b: 0, c: 0, d: localBounds.height, tx: localBounds.minX, ty: localBounds.minY
            )
            placement = placement.concatenating(box)
        }
        placement = placement.concatenating(documentTransform)
        let determinant = placement.a * placement.d - placement.b * placement.c
        guard determinant.isFinite, determinant != 0 else { throw SVGLoader.Failure.gradient("singular transform") }
        let inverted = placement.inverted()
        guard [inverted.a, inverted.b, inverted.c, inverted.d, inverted.tx, inverted.ty].allSatisfy(\.isFinite) else {
            throw SVGLoader.Failure.gradient("unrepresentable transform")
        }
        inverse = inverted
        spread = gradient.spread
        linearRGB = gradient.interpolatesInLinearRGB
        self.opacity = opacity
        if case .radial(let center, let focal, let radius) = gradient.kind {
            let difference = focal - center
            let distance = simd_length(difference)
            let safeFocal = radius > 0 && distance >= radius ? center + difference * (radius * 0.999_999 / distance) : focal
            kind = .radial(center: center, focal: safeFocal, radius: radius)
        } else {
            kind = gradient.kind
        }

        stops = gradient.stops.map { stop in
            let c = stop.color
            if gradient.interpolatesInLinearRGB { return stop }
            return Gradient.Stop(offset: stop.offset, color: SIMD4(
                GLTFColorTables.linearToSRGB(c.x), GLTFColorTables.linearToSRGB(c.y),
                GLTFColorTables.linearToSRGB(c.z), c.w
            ))
        }
    }

    func color(at point: CGPoint) -> SIMD4<Float> {
        let local = point.applying(inverse)
        let position = SIMD2<Double>(local.x, local.y)
        let t: Double
        switch kind {
        case .linear(let start, let end):
            let axis = end - start
            let lengthSquared = simd_length_squared(axis)
            if lengthSquared == 0 { return output(stops.last!.color) }
            t = simd_dot(position - start, axis) / lengthSquared
        case .radial(let center, let focal, let radius):
            if radius == 0 { return output(stops.last!.color) }
            let axis = center - focal
            let p = position - focal
            let a = radius * radius - simd_length_squared(axis)
            let b = simd_dot(p, axis)
            let c = simd_length_squared(p)
            let root = sqrt(max(0, b * b + a * c))
            // Algebraically equivalent form avoids cancellation when b is positive.
            t = b > 0 ? c / (root + b) : (root - b) / a
        }

        guard t.isFinite else { return output(stops.last!.color) }
        let value: Float
        switch spread {
        case .pad: value = Float(min(max(t, 0), 1))
        case .repeat: value = Float(t - floor(t))
        case .reflect:
            let period = abs(t).truncatingRemainder(dividingBy: 2)
            value = Float(period <= 1 ? period : 2 - period)
        }
        return output(interpolated(value))
    }

    private func interpolated(_ offset: Float) -> SIMD4<Float> {
        var low = 0, high = stops.count
        while low < high {
            let mid = low + (high - low) / 2
            if stops[mid].offset <= offset { low = mid + 1 } else { high = mid }
        }
        if low == 0 { return stops[0].color }
        if low == stops.count { return stops[low - 1].color }
        let a = stops[low - 1], b = stops[low]
        let t = (offset - a.offset) / (b.offset - a.offset)
        return a.color + (b.color - a.color) * t
    }

    private func output(_ color: SIMD4<Float>) -> SIMD4<Float> {
        let alpha = color.w * opacity
        if linearRGB { return SIMD4(color.x, color.y, color.z, alpha) }
        return SIMD4(
            GLTFColorTables.sRGBToLinear(color.x), GLTFColorTables.sRGBToLinear(color.y),
            GLTFColorTables.sRGBToLinear(color.z), alpha
        )
    }
}

struct SVGGradientDefinition {
    let radial: Bool
    var properties: [String: String]
    var stops: [SVGLoader.Gradient.Stop]
    let reference: String?

    static func identifier(in paint: String) -> String? {
        let value = paint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.lowercased().hasPrefix("url("), value.last == ")" else { return nil }
        var body = value.dropFirst(4).dropLast().trimmingCharacters(in: .whitespacesAndNewlines)
        if let quote = body.first, (quote == "\"" || quote == "'"), body.last == quote {
            body = body.dropFirst().dropLast().trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard body.first == "#", body.count > 1, !body.contains(where: \.isWhitespace) else { return nil }
        return String(body.dropFirst())
    }

    func gradient(
        viewport: CGRect?, dotsPerInch: Double, parseTransform: (String) throws -> CGAffineTransform
    ) throws -> SVGLoader.Gradient? {
        guard !stops.isEmpty else { return nil }
        let userSpace = properties["gradientUnits"] == "userSpaceOnUse"
        func coordinate(_ name: String, _ fallback: String, axis: Int) throws -> Double {
            let text = (properties[name] ?? fallback).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.hasSuffix("%") {
                return try SVGLength.parse(text, dotsPerInch: dotsPerInch, name: name)
            }
            guard let percent = Double(text.dropLast()), percent.isFinite, abs(percent) <= SVGLoader.coordinateLimit else {
                throw SVGLoader.Failure.gradient(name)
            }
            let fraction = percent / 100
            if !userSpace { return fraction }
            guard let viewport else {
                if fraction == 0 { return 0 }
                throw SVGLoader.Failure.gradient("percentage without a viewport")
            }
            switch axis {
            case 0: return viewport.minX + viewport.width * fraction
            case 1: return viewport.minY + viewport.height * fraction
            default: return hypot(viewport.width, viewport.height) / sqrt(2) * fraction
            }
        }

        let kind: SVGLoader.Gradient.Kind
        if radial {
            let center = try SIMD2(coordinate("cx", "50%", axis: 0), coordinate("cy", "50%", axis: 1))
            let x = try properties["fx"] == nil ? center.x : coordinate("fx", "50%", axis: 0)
            let y = try properties["fy"] == nil ? center.y : coordinate("fy", "50%", axis: 1)
            let radius = try coordinate("r", "50%", axis: 2)
            guard radius >= 0 else { throw SVGLoader.Failure.gradient("negative radius") }
            kind = .radial(center: center, focal: SIMD2(x, y), radius: radius)
        } else {
            kind = try .linear(
                start: SIMD2(coordinate("x1", "0%", axis: 0), coordinate("y1", "0%", axis: 1)),
                end: SIMD2(coordinate("x2", "100%", axis: 0), coordinate("y2", "0%", axis: 1))
            )
        }
        let spread: SVGLoader.Gradient.Spread
        switch properties["spreadMethod"] {
        case "repeat": spread = .repeat
        case "reflect": spread = .reflect
        default: spread = .pad
        }
        return SVGLoader.Gradient(
            kind: kind, units: userSpace ? .userSpaceOnUse : .objectBoundingBox, spread: spread,
            transform: try properties["gradientTransform"].map(parseTransform) ?? .identity,
            stops: stops, interpolatesInLinearRGB: properties["color-interpolation"]?.lowercased() == "linearrgb"
        )
    }
}
