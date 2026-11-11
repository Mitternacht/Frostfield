import Foundation
import Frostfield
import Metal
import simd

/// Procedural 992.2 GT3 sheet metal and greenhouse. The front points along +Z.
enum LabPorscheBody {
    private static let halfLength: Float = 2.285
    private static let sillHeight: Float = 0.132
    private static let axles: [(z: Float, hub: Float, opening: Float)] = [
        (1.2275, 0.332, 0.356), (-1.2275, 0.348, 0.374),
    ]

    // Longitudinal centerline, shoulder width, rocker width, and side crown, in meters.
    private static let bodySections: [(z: Float, profile: SIMD4<Float>)] = [
        (-2.285, SIMD4(0.716, 0.806, 0.804, 0.047)),
        (-2.110, SIMD4(0.733, 0.845, 0.807, 0.075)),
        (-1.860, SIMD4(0.764, 0.859, 0.796, 0.091)),
        (-1.580, SIMD4(0.797, 0.863, 0.787, 0.107)),
        (-1.230, SIMD4(0.799, 0.861, 0.777, 0.111)),
        (-0.900, SIMD4(0.800, 0.846, 0.763, 0.102)),
        (-0.590, SIMD4(0.796, 0.804, 0.748, 0.093)),
        (-0.160, SIMD4(0.792, 0.788, 0.742, 0.089)),
        ( 0.300, SIMD4(0.809, 0.779, 0.749, 0.087)),
        ( 0.645, SIMD4(0.810, 0.795, 0.760, 0.088)),
        ( 0.920, SIMD4(0.770, 0.820, 0.782, 0.089)),
        ( 1.235, SIMD4(0.732, 0.851, 0.798, 0.096)),
        ( 1.545, SIMD4(0.682, 0.855, 0.805, 0.093)),
        ( 1.830, SIMD4(0.629, 0.858, 0.812, 0.084)),
        ( 2.090, SIMD4(0.569, 0.844, 0.829, 0.059)),
        ( 2.285, SIMD4(0.548, 0.828, 0.831, 0.036)),
    ]

    // Center roof height, edge width, crown and belt width.
    private static let cabinSections: [(z: Float, profile: SIMD4<Float>)] = [
        (-1.615, SIMD4(0.844, 0.509, 0.033, 0.605)),
        (-1.475, SIMD4(0.897, 0.520, 0.050, 0.663)),
        (-1.260, SIMD4(0.988, 0.536, 0.057, 0.714)),
        (-1.020, SIMD4(1.100, 0.552, 0.063, 0.735)),
        (-0.765, SIMD4(1.196, 0.566, 0.063, 0.748)),
        (-0.530, SIMD4(1.253, 0.576, 0.058, 0.749)),
        (-0.258, SIMD4(1.279, 0.584, 0.052, 0.745)),
        (-0.030, SIMD4(1.269, 0.586, 0.049, 0.736)),
        ( 0.200, SIMD4(1.226, 0.596, 0.049, 0.724)),
        ( 0.355, SIMD4(1.152, 0.624, 0.057, 0.713)),
        ( 0.515, SIMD4(1.035, 0.649, 0.057, 0.706)),
        ( 0.667, SIMD4(0.920, 0.677, 0.056, 0.704)),
        ( 0.780, SIMD4(0.837, 0.695, 0.040, 0.696)),
    ]

    static func frontDeckY(x: Float, z: Float) -> Float {
        deckY(x: x, z: z)
    }

    static func rearDeckY(x: Float, z: Float) -> Float {
        deckY(x: x, z: z)
    }

    static func frontFasciaZ(x: Float, y: Float) -> Float {
        fasciaZ(x: x, y: y, end: 1)
    }

    static func rearFasciaZ(x: Float, y: Float) -> Float {
        fasciaZ(x: x, y: y, end: -1)
    }

    static func nodes(device: any MTLDevice, paint: PhysicallyBasedMaterial) throws -> [SceneNode] {
        let gasket = try PhysicallyBasedMaterial(
            roughness: 0.58, baseColorFactor: SIMD4(0.016, 0.017, 0.019, 1), isDoubleSided: true
        )
        let glass = try PhysicallyBasedMaterial(
            roughness: 0.048, baseColorFactor: SIMD4(0.69, 0.76, 0.80, 1), isDoubleSided: true,
            transmission: .init(factor: 0.99, indexOfRefraction: 1.49)
        )
        let lining = try PhysicallyBasedMaterial(
            roughness: 0.77, baseColorFactor: SIMD4(0.018, 0.019, 0.021, 1), isDoubleSided: true
        )

        var painted = Patch()
        var rubber = Patch()
        var windows = Patch()
        var interior = Patch()
        painted.add(columns: 104, rows: 128, reversed: true, surface: deckPoint) { _, p in
            let cabinOpening = abs(p.x) < 0.594 && p.z > -1.39 && p.z < 0.625
            return !cabinOpening && !LabPorscheTrim.headlightContains(x: p.x, z: p.z)
                && !LabPorscheTrim.hoodDuctContains(x: p.x, z: p.z)
        }
        for end: Float in [-1, 1] {
            painted.add(columns: 104, rows: 32, reversed: end < 0,
                        surface: { fasciaPoint($0, end: end) }) { _, p in
                end < 0 || !LabPorscheTrim.frontIntakeContains(x: p.x, y: p.y)
            }
        }
        for side: Float in [-1, 1] {
            painted.add(columns: 128, rows: 32, reversed: side > 0,
                        surface: { bodySide($0, side: side) }) { _, p in
                outsideWheel(y: p.y, z: p.z)
            }
            for axle in axles {
                addWheelhouse(axle, side: side, paint: &painted, lining: &interior)
            }

            let flank: (SIMD2<Float>) -> SIMD3<Float> = { cabinSide($0, side: side) }
            painted.add(columns: 62, rows: 24, reversed: side > 0, surface: flank) { uv, _ in
                sideWindow(uv) <= 0
            }
            let pillar: Patch.Mask = { uv, _ in abs(cabinStation(uv.x) + 0.633) < 0.021 }
            rubber.add(columns: 62, rows: 24, reversed: side > 0, surface: flank,
                       additionalClips: [{ uv, _ in sideWindow(uv) < 0.018 }]) { uv, _ in
                sideWindow(uv) > 0
            }
            rubber.add(columns: 62, rows: 24, reversed: side > 0, surface: flank,
                       additionalClips: [pillar]) { uv, _ in sideWindow(uv) >= 0.018 }
            windows.add(columns: 62, rows: 24, reversed: side > 0, surface: flank,
                        additionalClips: [{ uv, p in !pillar(uv, p) }]) { uv, _ in sideWindow(uv) >= 0.018 }
            addDoorSeams(side: side, to: &rubber)
        }

        painted.add(columns: 56, rows: 62, reversed: true, surface: cabinTop) { uv, _ in
            windscreen(uv) <= 0
        }
        rubber.add(columns: 56, rows: 62, reversed: true, surface: cabinTop,
                   additionalClips: [{ uv, _ in windscreen(uv) < 0.022 }]) { uv, _ in
            windscreen(uv) > 0
        }
        windows.add(columns: 56, rows: 62, reversed: true, surface: cabinTop) { uv, _ in
            windscreen(uv) >= 0.022
        }
        addDeckSeams(to: &rubber)

        interior.add(columns: 24, rows: 128, reversed: false, surface: undertrayPoint) { _, p in
            abs(p.x) <= 0.61 || outsideWheel(y: p.y, z: p.z)
        }
        return [
            try node(device, patch: painted, material: paint),
            try node(device, patch: rubber, material: gasket),
            try node(device, patch: windows, material: glass),
            try node(device, patch: interior, material: lining),
        ]
    }

    private static func node(
        _ device: any MTLDevice, patch: Patch, material: PhysicallyBasedMaterial
    ) throws -> SceneNode {
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float,
            geometry: MeshGeometry(vertices: patch.vertices, indices: patch.indices),
            material: .physicallyBased(material)
        )
        return SceneNode(mesh: mesh)
    }

    private static func deckY(x: Float, z: Float) -> Float {
        let center = profile(bodySections, at: z).x
        let absX = abs(x)
        let fender = 0.172 * exp(-square((absX - 0.665) / 0.199)) * exp(-square((z - 1.535) / 0.700))
        let rearHaunch = 0.063 * exp(-square((absX - 0.747) / 0.224)) * exp(-square((z + 1.194) / 0.538))
        let shoulder = 0.032 * pow(absX / 0.85, 8)
        let hoodFlute = 0.018 * exp(-square((absX - 0.428) / 0.093)) * exp(-square((z - 1.475) / 0.624))
        return center + fender + rearHaunch + hoodFlute - shoulder
    }

    private static func gridStation(_ t: Float) -> Float {
        halfLength * sin((t - 0.5) * .pi)
    }

    private static func deckPoint(_ uv: SIMD2<Float>) -> SIMD3<Float> {
        let station = gridStation(uv.y)
        let x = (2 * uv.x - 1) * profile(bodySections, at: station).y
        let z = endWrappedZ(x: x, station: station)
        return roundedEnd(SIMD3(x, deckY(x: x, z: z), z))
    }

    private static func endWrappedZ(x: Float, station: Float) -> Float {
        let end: Float = station >= 0 ? 1 : -1
        let wrap = pow(abs(x) / 0.85, 3.6) * (end > 0 ? 0.214 : 0.186)
        return station - end * wrap * smooth(1.55, halfLength, abs(station))
    }

    private static func fasciaZ(x: Float, y: Float, end: Float) -> Float {
        let lowerRoll = (end > 0 ? 0.033 : 0.046) * (1 - smooth(0.145, 0.45, y))
        return endWrappedZ(x: x, station: end * halfLength) - end * lowerRoll
    }

    private static func fasciaPoint(_ uv: SIMD2<Float>, end: Float) -> SIMD3<Float> {
        let across = 2 * uv.x - 1
        let station = end * halfLength
        let edge = rawSide(station: station, height: uv.y, side: 1)
        let x = across * edge.x
        let edgeWidth = profile(bodySections, at: station).y
        let upperX = across * edgeWidth
        let upperZ = endWrappedZ(x: upperX, station: station)
        let edgeTop = deckY(x: edgeWidth, z: endWrappedZ(x: edgeWidth, station: station))
        let shoulder = edge.y - mix(sillHeight, edgeTop, uv.y)
        let y = mix(sillHeight, deckY(x: upperX, z: upperZ), uv.y) + shoulder * pow(abs(across), 3)
        return roundedEnd(SIMD3(x, y, fasciaZ(x: x, y: y, end: end)))
    }

    // The deck, side and fascia all meet on the same curved return.
    private static func roundedEnd(_ p: SIMD3<Float>) -> SIMD3<Float> {
        guard abs(p.z) > 1.87 else { return p }
        let end: Float = p.z >= 0 ? 1 : -1
        let radius: Float = end > 0 ? 0.091 : 0.077
        let edgeZ = endWrappedZ(x: p.x, station: end * halfLength)
        let setback = max((edgeZ - p.z) * end, 0)
        let crown = deckY(x: p.x, z: p.z)
        let drop = max(crown - p.y, 0)
        let minimum = min(setback, drop)
        let diagonal = setback - drop
        guard abs(diagonal) < radius, minimum < radius * 0.48 else { return p }

        let angle = (.pi / 4) * (1 + diagonal / radius)
        let curveSetback = radius * (1 - cos(angle))
        let curveDrop = radius * (1 - sin(angle))
        let blend = 1 - smooth(0, radius * 0.48, minimum)
        let z = p.z - end * blend * (curveSetback - max(diagonal, 0))
        let y = p.y + deckY(x: p.x, z: z) - crown - blend * (curveDrop - max(-diagonal, 0))
        return SIMD3(p.x, y, z)
    }

    private static func bodySide(_ uv: SIMD2<Float>, side: Float) -> SIMD3<Float> {
        roundedEnd(rawSide(station: gridStation(uv.x), height: uv.y, side: side))
    }

    private static func rawSide(station: Float, height: Float, side: Float) -> SIMD3<Float> {
        let section = profile(bodySections, at: station)
        let crown = pow(max(sin(.pi * height), 0), 0.78)
        let inwardSculpt = 0.035 * exp(-square((station - 0.08) / 0.78))
            * exp(-square((height - 0.56) / 0.25)) * crown
        let halfWidth = mix(section.z, section.y, height) + section.w * crown - inwardSculpt
        let upperZ = endWrappedZ(x: section.y, station: station)
        let panelY = mix(sillHeight, deckY(x: section.y, z: upperZ), height)
        let end: Float = station >= 0 ? 1 : -1
        let lowerRoll = (end > 0 ? 0.033 : 0.046) * (1 - smooth(0.145, 0.45, panelY))
        let z = endWrappedZ(x: halfWidth, station: station) - end * lowerRoll * smooth(1.80, halfLength, abs(station))
        let shoulderY = deckY(x: halfWidth, z: z) - 0.28 * square(1 - height)
        let y = mix(panelY, shoulderY, smooth(0.82, 1, height))
        return SIMD3(side * halfWidth, y, z)
    }

    private static func undertrayPoint(_ uv: SIMD2<Float>) -> SIMD3<Float> {
        let station = gridStation(uv.y)
        let end: Float = station >= 0 ? 1 : -1
        let x = (2 * uv.x - 1) * profile(bodySections, at: station).z
        let lowerRoll: Float = end > 0 ? 0.033 : 0.046
        let z = endWrappedZ(x: x, station: station) - end * lowerRoll * smooth(1.80, halfLength, abs(station))
        return SIMD3(x, sillHeight, z)
    }

    private static func sideAt(y: Float, z: Float, side: Float) -> SIMD3<Float> {
        var station = z
        var point = SIMD3<Float>.zero
        for _ in 0..<4 {
            var low: Float = 0
            var high: Float = 1
            for _ in 0..<13 {
                let height = (low + high) * 0.5
                point = roundedEnd(rawSide(station: station, height: height, side: side))
                if point.y < y {
                    low = height
                } else {
                    high = height
                }
            }
            station += z - point.z
        }
        return point
    }

    private static func outsideWheel(y: Float, z: Float) -> Bool {
        axles.allSatisfy { square(y - $0.hub) + square(z - $0.z) >= square($0.opening) }
    }

    private static func addWheelhouse(
        _ axle: (z: Float, hub: Float, opening: Float), side: Float, paint: inout Patch, lining: inout Patch
    ) {
        let start = asin((sillHeight - axle.hub) / axle.opening)
        func ring(_ uv: SIMD2<Float>, inner: Bool) -> SIMD3<Float> {
            let angle = mix(start, .pi - start, uv.x)
            let y = axle.hub + axle.opening * sin(angle)
            let z = axle.z + axle.opening * cos(angle)
            var outer = sideAt(y: y, z: z, side: side)
            let inset = inner ? mix(0.021, 0.176, uv.y) : uv.y * 0.021
            outer.x -= side * inset
            let lipRoll = min(inset * 0.29, 0.005)
            outer.y -= lipRoll * sin(angle)
            outer.z -= lipRoll * cos(angle)
            return outer
        }
        paint.add(columns: 88, rows: 3, reversed: side > 0, surface: { ring($0, inner: false) })
        lining.add(columns: 66, rows: 4, reversed: side > 0, surface: { ring($0, inner: true) })
    }

    private static func cabinStation(_ u: Float) -> Float {
        mix(-1.615, 0.780, u)
    }

    private static func cabinTop(_ uv: SIMD2<Float>) -> SIMD3<Float> {
        let across = 2 * uv.x - 1
        let station = cabinStation(uv.y)
        let roof = profile(cabinSections, at: station)
        let frontWrap = 0.125 * smooth(0.14, 0.78, station)
        let rearWrap = 0.090 * (1 - smooth(-1.615, -0.70, station))
        let z = station + (rearWrap - frontWrap) * square(across)
        let x = across * roof.y
        let closing = smooth(-1.615, -1.49, station) * (1 - smooth(0.695, 0.780, station))
        let top = roof.x - roof.z * square(across)
        let y = mix(deckY(x: x, z: z) + 0.001, top, closing)
        return SIMD3(x, y, z)
    }

    private static func cabinSide(_ uv: SIMD2<Float>, side: Float) -> SIMD3<Float> {
        let station = cabinStation(uv.x)
        let roof = profile(cabinSections, at: station)
        let rail = cabinTop(SIMD2(side > 0 ? 1 : 0, uv.x))
        let x = side * roof.w
        let belt = SIMD3(x, deckY(x: x, z: rail.z) + 0.0006, rail.z)
        var p = belt + uv.y * (rail - belt)
        let crown = 0.022 * sin(.pi * uv.y) * smooth(0.015, 0.24, abs(rail.y - belt.y))
        p.x += side * crown
        return p
    }

    private static func sideWindow(_ uv: SIMD2<Float>) -> Float {
        let s = cabinStation(uv.x)
        let rise = uv.y
        let fore = 0.66 - 0.145 * square(rise)
        let aft = -1.34 + 0.196 * square(rise)
        let bottom = 0.085 + 0.25 * (1 - smooth(-1.36, -0.80, s))
        let top = 0.88 - 0.20 * (1 - smooth(-1.34, -1.12, s))
        let endRound = 0.075 * square((rise - 0.53) / 0.47)
        return min(fore - s - endRound, s - aft - endRound, rise - bottom, top - rise)
    }

    private static func windscreen(_ uv: SIMD2<Float>) -> Float {
        let s = cabinStation(uv.y)
        let across = abs(2 * uv.x - 1)
        let front = min(s - (0.224 + 0.095 * square(across)), 0.728 - s,
                        0.926 - across - 0.12 * square((s - 0.51) / 0.29))
        let back = min(s + 1.52, (-0.685 - 0.14 * square(across)) - s,
                       0.878 - across - 0.115 * square((s + 1.09) / 0.44))
        return max(front, back)
    }

    private static func addDoorSeams(side: Float, to patch: inout Patch) {
        let path: [SIMD2<Float>] = [
            SIMD2(0.623, 0.789), SIMD2(0.71, 0.692), SIMD2(0.77, 0.477), SIMD2(0.739, 0.257),
            SIMD2(0.691, 0.226), SIMD2(0.16, 0.214), SIMD2(-0.39, 0.214), SIMD2(-0.638, 0.235),
            SIMD2(-0.697, 0.284), SIMD2(-0.74, 0.527), SIMD2(-0.749, 0.722), SIMD2(-0.695, 0.791),
        ]
        addSeam(path, width: 0.0030, to: &patch) { p in
            var point = sideAt(y: p.y, z: p.x, side: side)
            point.x += side * 0.0011
            return point
        }
        let handle: [SIMD2<Float>] = [SIMD2(-0.603, 0.698), SIMD2(-0.543, 0.707), SIMD2(-0.455, 0.710)]
        addSeam(handle, width: 0.012, to: &patch) { p in
            var point = sideAt(y: p.y, z: p.x, side: side)
            point.x += side * 0.0018
            return point
        }
    }

    private static func addDeckSeams(to patch: inout Patch) {
        let hood: [SIMD2<Float>] = [
            SIMD2(-0.566, 0.792), SIMD2(-0.527, 1.128), SIMD2(-0.476, 1.566),
            SIMD2(-0.440, 1.956), SIMD2(-0.432, 2.067), SIMD2(-0.390, 2.109),
            SIMD2(0.390, 2.109), SIMD2(0.432, 2.067), SIMD2(0.440, 1.956),
            SIMD2(0.476, 1.566), SIMD2(0.527, 1.128), SIMD2(0.566, 0.792),
        ]
        addSeam(hood, width: 0.0028, to: &patch) { p in
            SIMD3(p.x, deckY(x: p.x, z: p.y) + 0.0012, p.y)
        }

        let engineCover: [SIMD2<Float>] = [
            SIMD2(-0.462, -1.545), SIMD2(-0.536, -1.722), SIMD2(-0.549, -2.055),
            SIMD2(-0.480, -2.122), SIMD2(0.480, -2.122), SIMD2(0.549, -2.055),
            SIMD2(0.536, -1.722), SIMD2(0.462, -1.545),
        ]
        addSeam(engineCover, width: 0.0026, to: &patch) { p in
            SIMD3(p.x, deckY(x: p.x, z: p.y) + 0.0012, p.y)
        }
    }

    private static func addSeam(
        _ points: [SIMD2<Float>], width: Float, to patch: inout Patch,
        project: (SIMD2<Float>) -> SIMD3<Float>
    ) {
        func point(_ s: Float) -> SIMD2<Float> {
            let index = s * Float(points.count - 1)
            let i = min(max(Int(index), 0), points.count - 2)
            let t = index - Float(i)
            let b = points[i]
            let c = points[i + 1]
            let a = points[max(0, i - 1)]
            let d = points[min(points.count - 1, i + 2)]
            return cubic(b, c, (c - a) * 0.5, (d - b) * 0.5, t)
        }
        patch.add(columns: (points.count - 1) * 12, rows: 1, reversed: false, surface: { uv in
            let p = point(uv.x)
            let tangent = point(min(1, uv.x + 0.002)) - point(max(0, uv.x - 0.002))
            let perp = simd_normalize(SIMD2(-tangent.y, tangent.x))
            return project(p + width * (uv.y - 0.5) * perp)
        })
    }

    private static func profile(_ sections: [(z: Float, profile: SIMD4<Float>)], at z: Float) -> SIMD4<Float> {
        let index = sections.firstIndex { $0.z > z }.map { max($0 - 1, 0) } ?? sections.count - 2
        let a = sections[max(0, index - 1)]
        let b = sections[index]
        let c = sections[index + 1]
        let d = sections[min(sections.count - 1, index + 2)]
        let step = c.z - b.z
        let t = min(max((z - b.z) / step, 0), 1)
        let startSlope = (c.profile - a.profile) * (step / (c.z - a.z))
        let endSlope = (d.profile - b.profile) * (step / (d.z - b.z))
        let tt = t * t
        let ttt = tt * t
        return b.profile * (2 * ttt - 3 * tt + 1) + startSlope * (ttt - 2 * tt + t)
            + c.profile * (-2 * ttt + 3 * tt) + endSlope * (ttt - tt)
    }

    private static func cubic(
        _ a: SIMD2<Float>, _ b: SIMD2<Float>, _ tangentA: SIMD2<Float>, _ tangentB: SIMD2<Float>, _ t: Float
    ) -> SIMD2<Float> {
        let tt = t * t
        let ttt = tt * t
        return a * (2 * ttt - 3 * tt + 1) + tangentA * (ttt - 2 * tt + t)
            + b * (-2 * ttt + 3 * tt) + tangentB * (ttt - tt)
    }

    private static func square(_ value: Float) -> Float { value * value }

    private static func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
        a + (b - a) * t
    }

    private static func smooth(_ a: Float, _ b: Float, _ value: Float) -> Float {
        let t = min(max((value - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// An indexed patch clips cells to curved apertures and evaluates exact intersection vertices.
    private struct Patch {
        typealias Mask = (SIMD2<Float>, SIMD3<Float>) -> Bool
        private struct Sample {
            let uv: SIMD2<Float>
            let position: SIMD3<Float>
            let index: UInt32
        }

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []

        mutating func add(
            columns: Int, rows: Int, reversed: Bool, surface: (SIMD2<Float>) -> SIMD3<Float>,
            additionalClips: [Mask] = [], mask: @escaping Mask = { _, _ in true }
        ) {
            func normal(_ uv: SIMD2<Float>) -> SIMD3<Float> {
                let h: Float = 0.0005
                let du = surface(SIMD2(min(1, uv.x + h), uv.y)) - surface(SIMD2(max(0, uv.x - h), uv.y))
                let dv = surface(SIMD2(uv.x, min(1, uv.y + h))) - surface(SIMD2(uv.x, max(0, uv.y - h)))
                let cross = reversed ? simd_cross(dv, du) : simd_cross(du, dv)
                let length = simd_length(cross)
                return length > 0 ? cross / length : SIMD3(0, 1, 0)
            }

            var samples: [Sample] = []
            let stride = columns + 1
            samples.reserveCapacity(stride * (rows + 1))
            for row in 0...rows {
                for column in 0...columns {
                    let uv = SIMD2(Float(column) / Float(columns), Float(row) / Float(rows))
                    let p = surface(uv)
                    let index = UInt32(vertices.count)
                    vertices.append(MeshVertex(position: p, normal: normal(uv), textureCoordinate: uv))
                    samples.append(Sample(uv: uv, position: p, index: index))
                }
            }
            let clips = [mask] + additionalClips
            let primaryInside = samples.map { mask($0.uv, $0.position) }

            for row in 0..<rows {
                for column in 0..<columns {
                    let a = row * stride + column
                    let b = a + 1
                    let c = a + stride
                    let d = c + 1
                    if additionalClips.isEmpty && primaryInside[a] && primaryInside[b] &&
                        primaryInside[c] && primaryInside[d] {
                        let ids = [samples[a].index, samples[b].index, samples[c].index, samples[d].index]
                        indices += reversed ? [ids[0], ids[2], ids[1], ids[1], ids[2], ids[3]]
                            : [ids[0], ids[1], ids[2], ids[1], ids[3], ids[2]]
                    } else if !additionalClips.isEmpty || primaryInside[a] || primaryInside[b] ||
                                primaryInside[c] || primaryInside[d] {
                        emit([samples[a], samples[b], samples[c]])
                        emit([samples[b], samples[d], samples[c]])
                    }
                }
            }

            func emit(_ triangle: [Sample]) {
                var polygon = triangle
                for keep in clips {
                    var clipped: [Sample] = []
                    for i in polygon.indices {
                        let a = polygon[i]
                        let b = polygon[(i + 1) % polygon.count]
                        let includeA = keep(a.uv, a.position)
                        let includeB = keep(b.uv, b.position)
                        if includeA { clipped.append(a) }
                        if includeA != includeB {
                            var low = a.uv
                            var high = b.uv
                            for _ in 0..<10 {
                                let middle = (low + high) * 0.5
                                if keep(middle, surface(middle)) == includeA {
                                    low = middle
                                } else {
                                    high = middle
                                }
                            }
                            let uv = (low + high) * 0.5
                            let p = surface(uv)
                            let index = UInt32(vertices.count)
                            vertices.append(MeshVertex(position: p, normal: normal(uv), textureCoordinate: uv))
                            clipped.append(Sample(uv: uv, position: p, index: index))
                        }
                    }
                    polygon = clipped
                    if polygon.count < 3 { return }
                }
                for i in 1..<(polygon.count - 1) {
                    let a = polygon[0].index
                    let b = polygon[i].index
                    let c = polygon[i + 1].index
                    indices += reversed ? [a, c, b] : [a, b, c]
                }
            }
        }

    }
}
