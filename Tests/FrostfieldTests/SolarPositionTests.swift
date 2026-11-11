import Foundation
import Frostfield
import Testing
import simd

struct SolarPositionTests {
    /// Reda and Andreas, "Solar Position Algorithm for Solar Radiation Applications"
    /// (NREL/TP-560-34302, 2008), Table A5.1: Golden, Colorado, 17 October 2003,
    /// 12:30:30 at UTC−7. SPA reports a topocentric zenith of 50.11162° (with
    /// refraction at 820 mbar and 11 °C), azimuth 194.34024° and an equation of
    /// time of 14.641503 minutes. NOAA's lower-order series, standard-atmosphere
    /// refraction and geocentric position stay within 0.01° of it.
    @Test("NOAA position matches NREL's SPA reference within 0.01°")
    func solarPositionAlgorithmReference() throws {
        let local = DateComponents(year: 2003, month: 10, day: 17, hour: 12, minute: 30, second: 30)
        let zone = try #require(TimeZone(secondsFromGMT: -7 * 3_600))
        let position = try SolarPosition(localTime: local, timeZone: zone, latitude: 39.742476, longitude: -105.1786)
        #expect(abs(position.julianDay - 2_452_930.312847) < 1e-6)
        #expect(abs(position.zenith - 50.11162) < 0.01, "zenith \(position.zenith)")
        #expect(abs(position.azimuth - 194.34024) < 0.01, "azimuth \(position.azimuth)")
        #expect(abs(position.equationOfTime - 14.641503) < 0.02, "equation of time \(position.equationOfTime)")
        #expect(abs(position.declination - -9.316179) < 0.01)

        // The same instant as an absolute date, and the world direction: south-southwest,
        // north is −Z and east +X.
        let date = Date(timeIntervalSince1970: 1_066_419_030) // 2003-10-17T19:30:30Z
        let same = try SolarPosition(date: date, latitude: 39.742476, longitude: -105.1786)
        #expect(abs(same.azimuth - position.azimuth) < 1e-4 && abs(same.elevation - position.elevation) < 1e-4)
        let direction = position.direction()
        let elevation = position.elevation * .pi / 180
        #expect(abs(Double(direction.y) - sin(elevation)) < 1e-6)
        #expect(direction.x < 0 && direction.z > 0 && direction.z > -direction.x)
        #expect(abs(simd_length(direction) - 1) < 1e-6)
        // Refraction lifts the apparent Sun by about 0.019° at this elevation.
        #expect(abs((position.elevation - position.geometricElevation) - 0.0193) < 0.001)
    }

    /// Meeus, *Astronomical Algorithms* (2nd ed.), Example 25.a: 1992 October 13.0
    /// TD with the low-accuracy method that NOAA uses: apparent longitude 199.90895°,
    /// right ascension 198.38083°, declination −7.78507° and distance 0.99766 AU.
    /// Example 28.b gives an equation of time of 13.71 minutes for the same instant.
    @Test("Solar coordinates reproduce Meeus's worked examples")
    func meeusExamples() throws {
        let position = try SolarPosition(julianDay: 2_448_908.5, latitude: 0, longitude: 0)
        #expect(abs(position.apparentLongitude - 199.90895) < 0.0005)
        #expect(abs(position.rightAscension - 198.38083) < 0.0005)
        #expect(abs(position.declination - -7.78507) < 0.0005)
        #expect(abs(position.earthSunDistance - 0.99766) < 0.00001)
        #expect(abs(position.equationOfTime - 13.71) < 0.01)
    }

    @Test("Noon and night geometry, refraction and validation")
    func geometryAndValidation() throws {
        // Near the June solstice the Sun stands near the zenith on the Tropic of
        // Cancer at local apparent noon, and below the horizon twelve hours later.
        let zone = try #require(TimeZone(identifier: "UTC"))
        let noon = try SolarPosition(
            localTime: DateComponents(year: 2024, month: 6, day: 20, hour: 12, minute: 2), timeZone: zone,
            latitude: 23.44, longitude: 0
        )
        #expect(noon.elevation > 89.5 && abs(noon.hourAngle) < 0.6)
        let midnight = try SolarPosition(
            localTime: DateComponents(year: 2024, month: 6, day: 20, hour: 23, minute: 59), timeZone: zone,
            latitude: 23.44, longitude: 0
        )
        #expect(midnight.elevation < -40 && midnight.direction().y < 0)
        // A summer afternoon in a daylight-saving zone uses that zone's offset.
        let berlin = try #require(TimeZone(identifier: "Europe/Berlin"))
        let summer = try SolarPosition(
            localTime: DateComponents(year: 2024, month: 7, day: 1, hour: 13, minute: 30), timeZone: berlin,
            latitude: 52.52, longitude: 13.405
        )
        let utc = try SolarPosition(
            localTime: DateComponents(year: 2024, month: 7, day: 1, hour: 11, minute: 30), timeZone: zone,
            latitude: 52.52, longitude: 13.405
        )
        #expect(summer == utc)
        #expect(summer.elevation > 55 && summer.azimuth > 180 && summer.azimuth < 230)

        #expect(SolarPosition.refraction(elevation: 90) == 0)
        #expect(abs(SolarPosition.refraction(elevation: 0) - 1_735.0 / 3_600) < 1e-12)
        #expect(SolarPosition.refraction(elevation: 10) > SolarPosition.refraction(elevation: 30))

        #expect(throws: SolarPosition.Failure.latitude) {
            try SolarPosition(julianDay: 2_451_545, latitude: 91, longitude: 0)
        }
        #expect(throws: SolarPosition.Failure.longitude) {
            try SolarPosition(julianDay: 2_451_545, latitude: 0, longitude: .nan)
        }
        #expect(throws: SolarPosition.Failure.date) {
            try SolarPosition(julianDay: .infinity, latitude: 0, longitude: 0)
        }
        #expect(throws: SolarPosition.Failure.date) {
            try SolarPosition(localTime: DateComponents(hour: 12), timeZone: zone, latitude: 0, longitude: 0)
        }
    }

    @Test("At the poles the Sun circles the horizon with the hour angle")
    func poles() throws {
        // 21 June 2026: the Sun stays about 23.4° above the north pole's horizon and
        // turns 90° every six hours. The pole takes the limit along the given meridian,
        // which a latitude 11 m from the pole matches.
        for hour in [0.0, 6, 12, 18] {
            let julianDay = 2_461_212.5 + hour / 24
            let north = try SolarPosition(julianDay: julianDay, latitude: 90, longitude: 0)
            let near = try SolarPosition(julianDay: julianDay, latitude: 89.9999, longitude: 0)
            #expect(Self.angle(north.azimuth, north.hourAngle + 180) < 1e-9, "\(north.azimuth) at \(hour) h")
            #expect(Self.angle(north.azimuth, near.azimuth) < 1e-3, "\(north.azimuth) vs \(near.azimuth)")
            #expect(Self.angle(north.azimuth, 180 + 15 * (hour - 12)) < 1.5)
            #expect(abs(north.elevation - 23.44) < 0.1)

            let south = try SolarPosition(julianDay: julianDay + 182, latitude: -90, longitude: 0)
            let nearSouth = try SolarPosition(julianDay: julianDay + 182, latitude: -89.9999, longitude: 0)
            #expect(Self.angle(south.azimuth, -south.hourAngle) < 1e-9, "\(south.azimuth) at \(hour) h")
            #expect(Self.angle(south.azimuth, nearSouth.azimuth) < 1e-3, "\(south.azimuth) vs \(nearSouth.azimuth)")
            #expect(south.elevation > 20)
        }
        // A longitude shifts the meridian the azimuth is measured from.
        let east = try SolarPosition(julianDay: 2_461_212.5, latitude: 90, longitude: 90)
        let west = try SolarPosition(julianDay: 2_461_212.5, latitude: 90, longitude: 0)
        #expect(Self.angle(east.azimuth, west.azimuth + 90) < 1e-6)
    }

    @Test("Local times skipped by daylight saving resolve to the pre-change offset")
    func daylightSavingGap() throws {
        let losAngeles = try #require(TimeZone(identifier: "America/Los_Angeles"))
        // Clocks jump from 02:00 PST to 03:00 PDT on 8 March 2026: 02:30 PST is 10:30 UTC.
        let skipped = try SolarPosition(
            localTime: DateComponents(year: 2026, month: 3, day: 8, hour: 2, minute: 30), timeZone: losAngeles,
            latitude: 34.05, longitude: -118.25
        )
        let daylight = try SolarPosition(
            localTime: DateComponents(year: 2026, month: 3, day: 8, hour: 3, minute: 30), timeZone: losAngeles,
            latitude: 34.05, longitude: -118.25
        )
        let utc = try SolarPosition(date: Date(timeIntervalSince1970: 1_772_965_800), latitude: 34.05, longitude: -118.25)
        #expect(skipped == utc && daylight == utc)
        // Half-hour steps through the whole day all resolve; 02:00 and 03:00 name one instant.
        var instants = [Double]()
        for step in 0..<48 {
            let position = try SolarPosition(
                localTime: DateComponents(year: 2026, month: 3, day: 8, hour: step / 2, minute: 30 * (step % 2)),
                timeZone: losAngeles, latitude: 34.05, longitude: -118.25
            )
            instants.append(position.julianDay)
        }
        #expect(instants[4] == instants[6] && instants[5] == instants[7])
        #expect(abs((instants[47] - instants[0]) * 24 - 22.5) < 1e-6)
        // Invalid fields still throw.
        for components in [
            DateComponents(year: 2026, month: 2, day: 30, hour: 12), DateComponents(year: 2026, month: 13, day: 1),
            DateComponents(year: 2026, month: 3, day: 8, hour: 24),
        ] {
            #expect(throws: SolarPosition.Failure.date) {
                try SolarPosition(localTime: components, timeZone: losAngeles, latitude: 0, longitude: 0)
            }
        }
    }

    /// The absolute difference between two angles in degrees, from 0 to 180.
    private static func angle(_ a: Double, _ b: Double) -> Double {
        let difference = (a - b).truncatingRemainder(dividingBy: 360)
        return min(abs(difference), 360 - abs(difference))
    }
}
