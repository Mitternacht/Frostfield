import CoreGraphics
import Foundation

/// SVG numbers allow sign and fractional-point adjacency, plus individual arc flags.
struct SVGNumbers {
    private let bytes: [UInt8]
    private(set) var index = 0

    init(_ text: String) { bytes = Array(text.utf8) }

    var next: UInt8? { index < bytes.count ? bytes[index] : nil }

    mutating func skipSpace() {
        while let next, Self.isSpace(next) { index += 1 }
    }

    mutating func skipSeparator() {
        skipSpace()
        if take(44) { skipSpace() }
    }

    mutating func take(_ value: UInt8) -> Bool {
        guard next == value else { return false }
        index += 1
        return true
    }

    mutating func letter() -> UInt8? {
        guard let next, Self.isLetter(next) else { return nil }
        index += 1
        return next
    }

    mutating func word() -> String {
        let start = index
        while letter() != nil {}
        return String(decoding: bytes[start..<index], as: UTF8.self)
    }

    mutating func number() throws -> Double {
        skipSeparator()
        let start = index
        if next == 43 || next == 45 { index += 1 }
        let whole = digits()
        let fractional = take(46) ? digits() : 0
        guard whole + fractional > 0 else { throw SVGLoader.Failure.pathData(index) }
        if next == 69 || next == 101 {
            index += 1
            if next == 43 || next == 45 { index += 1 }
            guard digits() > 0 else { throw SVGLoader.Failure.pathData(index) }
        }
        guard index - start <= 128,
              let value = Double(String(decoding: bytes[start..<index], as: UTF8.self)),
              value.isFinite, abs(value) <= SVGLoader.coordinateLimit else { throw SVGLoader.Failure.pathData(start) }
        return value
    }

    mutating func flag() throws -> Bool {
        skipSeparator()
        if take(48) { return false }
        if take(49) { return true }
        throw SVGLoader.Failure.pathData(index)
    }

    private mutating func digits() -> Int {
        let start = index
        while let next, (48...57).contains(next) { index += 1 }
        return index - start
    }

    static func isSpace(_ value: UInt8) -> Bool { value == 32 || (9...13).contains(value) }
    private static func isLetter(_ value: UInt8) -> Bool { (65...90).contains(value) || (97...122).contains(value) }
}

enum SVGLength {
    static func parse(_ text: String, dotsPerInch: Double, name: String) throws -> CGFloat {
        var reader = SVGNumbers(text)
        let number = try reader.number()
        let unit = reader.word()
        reader.skipSpace()
        guard reader.next == nil else { throw SVGLoader.Failure.attribute(name) }
        let scale: Double
        switch unit {
        case "", "px": scale = 1
        case "in": scale = dotsPerInch
        case "cm": scale = dotsPerInch / 2.54
        case "mm": scale = dotsPerInch / 25.4
        case "pt": scale = dotsPerInch / 72
        case "pc": scale = dotsPerInch / 6
        default: throw SVGLoader.Failure.attribute(name)
        }
        let value = number * scale
        guard value.isFinite, abs(value) <= SVGLoader.coordinateLimit else { throw SVGLoader.Failure.attribute(name) }
        return value
    }
}

enum SVGPathData {
    static func parse(_ text: String) throws -> CGPath {
        var reader = SVGNumbers(text)
        let path = CGMutablePath()
        var command: UInt8?
        var current = SIMD2<Double>.zero
        var origin: SIMD2<Double>?
        var cubicControl: SIMD2<Double>?
        var quadraticControl: SIMD2<Double>?
        var afterClose = false
        var commandsLeft = SVGLoader.maximumPathCommands

        while true {
            reader.skipSpace()
            guard reader.next != nil else { break }
            if let next = reader.letter() { command = next }
            guard let active = command else { throw SVGLoader.Failure.pathData(reader.index) }
            let uppercase = active >= 97 ? active - 32 : active
            guard origin != nil || uppercase == 77 else { throw SVGLoader.Failure.pathData(reader.index) }
            guard commandsLeft > 0 else { throw SVGLoader.Failure.size }
            commandsLeft -= 1

            let relative = active != uppercase
            let base = relative ? current : .zero
            func point() throws -> SIMD2<Double> {
                try SIMD2(reader.number(), reader.number()) + base
            }
            if afterClose, uppercase != 77, uppercase != 90 { path.move(to: cg(current)) }
            afterClose = false
            var nextCubic: SIMD2<Double>?
            var nextQuadratic: SIMD2<Double>?
            switch uppercase {
            case 77: // M, followed by implicit L
                current = try point()
                origin = current
                path.move(to: cg(current))
                command = relative ? 108 : 76
            case 76: // L
                current = try point()
                path.addLine(to: cg(current))
            case 72: // H
                current.x = try reader.number() + base.x
                path.addLine(to: cg(current))
            case 86: // V
                current.y = try reader.number() + base.y
                path.addLine(to: cg(current))
            case 67: // C
                let first = try point(), second = try point()
                current = try point()
                path.addCurve(to: cg(current), control1: cg(first), control2: cg(second))
                nextCubic = second
            case 83: // S
                let first = cubicControl.map { 2 * current - $0 } ?? current
                let second = try point()
                current = try point()
                path.addCurve(to: cg(current), control1: cg(first), control2: cg(second))
                nextCubic = second
            case 81: // Q
                let control = try point()
                current = try point()
                path.addQuadCurve(to: cg(current), control: cg(control))
                nextQuadratic = control
            case 84: // T
                let control = quadraticControl.map { 2 * current - $0 } ?? current
                current = try point()
                path.addQuadCurve(to: cg(current), control: cg(control))
                nextQuadratic = control
            case 65: // A
                let radii = try SIMD2(reader.number(), reader.number())
                let angle = try reader.number() * .pi / 180
                let large = try reader.flag(), sweep = try reader.flag()
                let end = try point()
                try arc(path, from: current, to: end, radii: radii, angle: angle, large: large, sweep: sweep)
                current = end
            case 90: // Z
                path.closeSubpath()
                current = origin ?? current
                command = nil
                afterClose = true
            default: throw SVGLoader.Failure.pathData(reader.index)
            }
            guard abs(current.x) <= SVGLoader.coordinateLimit, abs(current.y) <= SVGLoader.coordinateLimit else {
                throw SVGLoader.Failure.pathData(reader.index)
            }
            cubicControl = nextCubic
            quadraticControl = nextQuadratic
        }
        return path
    }

    private static func cg(_ point: SIMD2<Double>) -> CGPoint { CGPoint(x: point.x, y: point.y) }

    /// SVG endpoint-to-center conversion. A short cubic approximation keeps the
    /// native path usable for drawing, flattening and later affine transforms.
    private static func arc(
        _ path: CGMutablePath, from start: SIMD2<Double>, to end: SIMD2<Double>,
        radii: SIMD2<Double>, angle: Double, large: Bool, sweep: Bool
    ) throws {
        guard start != end else { return }
        var radius = SIMD2(abs(radii.x), abs(radii.y))
        if radius.x == 0 || radius.y == 0 {
            path.addLine(to: cg(end))
            return
        }
        let cosine = cos(angle), sine = sin(angle)
        let half = (start - end) / 2
        let rotated = SIMD2(cosine * half.x + sine * half.y, -sine * half.x + cosine * half.y)
        var unit = rotated / radius
        let fit = hypot(unit.x, unit.y)
        guard fit.isFinite, fit > 0 else { throw SVGLoader.Failure.geometry }
        if fit > 1 {
            radius *= fit
            unit /= fit
        }
        guard radius.x.isFinite, radius.y.isFinite, radius.x <= SVGLoader.coordinateLimit,
              radius.y <= SVGLoader.coordinateLimit else { throw SVGLoader.Failure.geometry }
        let norm = hypot(unit.x, unit.y)
        let sign: Double = large == sweep ? -1 : 1
        let offset = sign * sqrt(max(0, 1 - norm * norm)) * SIMD2(unit.y / norm, -unit.x / norm)
        let first = unit - offset, last = -unit - offset
        let centerLocal = offset * radius
        func rotate(_ p: SIMD2<Double>) -> SIMD2<Double> {
            SIMD2(cosine * p.x - sine * p.y, sine * p.x + cosine * p.y)
        }
        let center = (start + end) / 2 + rotate(centerLocal)
        let startAngle = atan2(first.y, first.x)
        var span = atan2(first.x * last.y - first.y * last.x, first.x * last.x + first.y * last.y)
        if sweep, span < 0 { span += 2 * .pi }
        if !sweep, span > 0 { span -= 2 * .pi }
        let pieces = max(1, Int(ceil(abs(span) / (.pi / 4))))
        let step = span / Double(pieces)
        let factor = 4 / 3 * tan(step / 4)
        for index in 0..<pieces {
            let a = startAngle + Double(index) * step, b = a + step
            let p = SIMD2(cos(a), sin(a)), q = SIMD2(cos(b), sin(b))
            let c1 = center + rotate((p + factor * SIMD2(-p.y, p.x)) * radius)
            let c2 = center + rotate((q - factor * SIMD2(-q.y, q.x)) * radius)
            let endpoint = index == pieces - 1 ? end : center + rotate(q * radius)
            path.addCurve(to: cg(endpoint), control1: cg(c1), control2: cg(c2))
        }
    }
}
