import Foundation

/// Photometric resources shared by node lights and the optional rig.
struct ArchivePhotometry: Codable {
    struct References: Codable {
        var profile: Int?
        var cookie: Int?
    }

    struct Cookie: Codable {
        var width: Int
        var height: Int
        var texels: Int
    }

    struct RigLight: Codable {
        var id: String
        var resources: References
    }

    /// Binary views containing the UTF-8 source of each profile.
    var profiles: [Int] = []
    var cookies: [Cookie] = []
    var rigLights: [RigLight] = []
}

struct ArchivePhotometryCapture {
    private var resources = ArchivePhotometry()
    private var profiles: [ObjectIdentifier: Int] = [:]
    private var cookies: [ObjectIdentifier: Int] = [:]

    var record: ArchivePhotometry? {
        resources.profiles.isEmpty && resources.cookies.isEmpty ? nil : resources
    }

    mutating func node(
        _ light: PunctualLight?, writer: inout ArchiveBinaryWriter
    ) throws -> (light: PunctualLight?, resources: ArchivePhotometry.References?) {
        guard let light, let references = capture(light.iesProfile, light.cookie, writer: &writer) else {
            return (light, nil)
        }
        let base = try PunctualLight(kind: light.kind, color: light.color, intensity: light.intensity, shadow: light.shadow)
        return (base, references)
    }

    mutating func rig(_ rig: LightRig?, writer: inout ArchiveBinaryWriter) -> LightRig? {
        guard var rig else { return nil }
        for index in rig.lights.indices {
            let light = rig.lights[index]
            guard let references = capture(light.iesProfile, light.cookie, writer: &writer) else { continue }
            resources.rigLights.append(.init(id: light.id, resources: references))
            rig.lights[index].iesProfile = nil
            rig.lights[index].cookie = nil
        }
        return rig
    }

    private mutating func capture(
        _ profile: IESProfile?, _ cookie: LightCookie?, writer: inout ArchiveBinaryWriter
    ) -> ArchivePhotometry.References? {
        guard profile != nil || cookie != nil else { return nil }
        let profileIndex = profile.map { profile in
            let identity = ObjectIdentifier(profile.storage)
            if let index = profiles[identity] { return index }
            let index = resources.profiles.count
            resources.profiles.append(writer.append(Data(profile.storage.source.utf8)))
            profiles[identity] = index
            return index
        }
        let cookieIndex = cookie.map { cookie in
            let identity = ObjectIdentifier(cookie.storage)
            if let index = cookies[identity] { return index }
            let index = resources.cookies.count
            let texels = writer.append(cookie.transmittance.flatMap { [$0.x, $0.y, $0.z] })
            resources.cookies.append(.init(width: cookie.width, height: cookie.height, texels: texels))
            cookies[identity] = index
            return index
        }
        return .init(profile: profileIndex, cookie: cookieIndex)
    }
}

struct ArchivePhotometryLoader {
    let record: ArchivePhotometry?
    let reader: ArchiveBinaryReader
    private var profiles: [Int: IESProfile] = [:]
    private var cookies: [Int: LightCookie] = [:]

    init(record: ArchivePhotometry?, reader: ArchiveBinaryReader) {
        self.record = record
        self.reader = reader
    }

    mutating func node(
        _ light: PunctualLight?, resources: ArchivePhotometry.References?, id: UUID
    ) throws -> PunctualLight? {
        guard let resources else { return light }
        guard let light else { throw SceneArchive.Failure.invalidReference("node light \(id)") }
        let (profile, cookie) = try restore(resources)
        do {
            return try PunctualLight(
                kind: light.kind, color: light.color, intensity: light.intensity, shadow: light.shadow,
                iesProfile: profile, cookie: cookie
            )
        } catch {
            throw SceneArchive.Failure.invalidPhotometry("node \(id): \(error.localizedDescription)")
        }
    }

    mutating func rig(_ rig: LightRig?) throws -> LightRig? {
        guard let bindings = record?.rigLights, !bindings.isEmpty else { return rig }
        guard var rig else { throw SceneArchive.Failure.invalidReference("light rig for photometric resources") }
        let indices = Dictionary(uniqueKeysWithValues: rig.lights.enumerated().map { ($0.element.id, $0.offset) })
        var seen = Set<String>()
        for binding in bindings {
            guard let index = indices[binding.id] else {
                throw SceneArchive.Failure.invalidReference("rig light \(binding.id)")
            }
            guard seen.insert(binding.id).inserted else {
                throw SceneArchive.Failure.invalidPhotometry("duplicate resources for rig light \(binding.id)")
            }
            let (profile, cookie) = try restore(binding.resources)
            rig.lights[index].iesProfile = profile
            rig.lights[index].cookie = cookie
        }
        do {
            return try LightRig(lights: rig.lights)
        } catch {
            throw SceneArchive.Failure.invalidPhotometry(error.localizedDescription)
        }
    }

    private mutating func restore(_ references: ArchivePhotometry.References) throws -> (IESProfile?, LightCookie?) {
        guard references.profile != nil || references.cookie != nil else {
            throw SceneArchive.Failure.invalidPhotometry("empty light resource reference")
        }
        let profile = try references.profile.map { try self.profile($0) }
        let cookie = try references.cookie.map { try self.cookie($0) }
        return (profile, cookie)
    }

    private mutating func profile(_ index: Int) throws -> IESProfile {
        if let profile = profiles[index] { return profile }
        guard let record, record.profiles.indices.contains(index) else {
            throw SceneArchive.Failure.invalidReference("IES profile \(index)")
        }
        let bytes = try reader.bytes(record.profiles[index])
        guard let source = String(data: bytes, encoding: .utf8) else {
            throw SceneArchive.Failure.invalidPhotometry("IES profile \(index) is not UTF-8")
        }
        let profile: IESProfile
        do {
            profile = try IESProfile(string: source)
        } catch {
            throw SceneArchive.Failure.invalidPhotometry("IES profile \(index): \(error.localizedDescription)")
        }
        profiles[index] = profile
        return profile
    }

    private mutating func cookie(_ index: Int) throws -> LightCookie {
        if let cookie = cookies[index] { return cookie }
        guard let record, record.cookies.indices.contains(index) else {
            throw SceneArchive.Failure.invalidReference("light cookie \(index)")
        }
        let cookie = record.cookies[index]
        let dimensions = 1...LightCookie.maximumDimension
        guard dimensions.contains(cookie.width), dimensions.contains(cookie.height) else {
            throw SceneArchive.Failure.invalidPhotometry("light cookie \(index) dimensions")
        }
        let count = cookie.width * cookie.height * 3
        let components = try reader.array(cookie.texels, of: Float.self, count: count)
        let pixels = stride(from: 0, to: count, by: 3).map {
            SIMD3(components[$0], components[$0 + 1], components[$0 + 2])
        }
        let restored: LightCookie
        do {
            restored = try LightCookie(width: cookie.width, height: cookie.height, transmittance: pixels)
        } catch {
            throw SceneArchive.Failure.invalidPhotometry("light cookie \(index): \(error.localizedDescription)")
        }
        cookies[index] = restored
        return restored
    }
}
