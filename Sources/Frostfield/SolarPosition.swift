import Foundation
import simd

/// The Sun's apparent position for an observer on Earth, computed with NOAA's
/// solar position equations.
///
/// NOAA's spreadsheet and web calculator use the low-accuracy solar coordinates
/// from Jean Meeus, *Astronomical Algorithms*: the geometric mean longitude and
/// anomaly, the equation of center, nutation and aberration in longitude, the
/// corrected obliquity, the equation of time from Smart's series (Meeus, formula
/// 28.3), and NOAA's piecewise refraction approximation for a standard atmosphere. NOAA describes the results
/// as valid from 1800 to 2100 and usable from −1000 to 3000. Universal time stands
/// in for terrestrial time, as in NOAA's tools, which moves the Sun by under 0.001°.
///
/// Angles are degrees. Azimuth is measured clockwise from true north.
/// [NOAA solar calculation details](https://gml.noaa.gov/grad/solcalc/calcdetails.html)
public struct SolarPosition: Equatable, Sendable {
    /// Julian day of the instant in universal time.
    public let julianDay: Double
    /// Apparent ecliptic longitude of the Sun in degrees.
    public let apparentLongitude: Double
    /// Apparent right ascension in degrees, from 0 to 360.
    public let rightAscension: Double
    /// Apparent declination in degrees.
    public let declination: Double
    /// Distance from Earth to the Sun in astronomical units.
    public let earthSunDistance: Double
    /// Apparent solar time minus mean solar time, in minutes.
    public let equationOfTime: Double
    /// Hour angle in degrees from −180 to 180, negative before solar noon.
    public let hourAngle: Double
    /// Angle between the Sun's center and the horizon without atmospheric refraction.
    public let geometricElevation: Double
    /// Elevation including NOAA's approximate atmospheric refraction, as an observer sees it.
    public let elevation: Double
    /// Azimuth clockwise from true north, from 0 up to 360.
    public let azimuth: Double

    /// The apparent zenith angle, `90 − elevation`.
    public var zenith: Double { 90 - elevation }

    /// The position at an instant for a latitude from −90 to 90 degrees (north
    /// positive) and a longitude from −180 to 180 degrees (east positive).
    public init(date: Date, latitude: Double, longitude: Double) throws {
        let seconds = date.timeIntervalSince1970
        guard seconds.isFinite else { throw Failure.date }
        try self.init(julianDay: seconds / 86_400 + 2_440_587.5, latitude: latitude, longitude: longitude)
    }

    /// The position at a wall-clock time in a time zone, such as 12:30:30 on
    /// 17 October 2003 in UTC−7. The components need a year, month and day; hour,
    /// minute, second and nanosecond default to zero. Uses the Gregorian calendar
    /// and the zone's offset at that date, including daylight saving time.
    ///
    /// A time that a spring-forward change skips, such as 02:30 when clocks jump
    /// from 02:00 to 03:00, is read with the offset in effect before the change, so
    /// it names the same instant as 03:30 daylight time. A time that an autumn
    /// change repeats resolves to its first occurrence. Both follow Foundation's
    /// `Calendar.date(from:)`.
    public init(localTime: DateComponents, timeZone: TimeZone, latitude: Double, longitude: Double) throws {
        guard localTime.year != nil, localTime.month != nil, localTime.day != nil else { throw Failure.date }
        var components = localTime
        components.hour = localTime.hour ?? 0
        components.minute = localTime.minute ?? 0
        components.second = localTime.second ?? 0
        // Check the fields in a zone without daylight saving, so a skipped local
        // time is resolved below instead of being rejected as an invalid date.
        var fields = Calendar(identifier: .gregorian)
        guard let utc = TimeZone(secondsFromGMT: 0) else { throw Failure.date }
        fields.timeZone = utc
        components.calendar = nil
        components.timeZone = nil
        guard components.isValidDate(in: fields) else { throw Failure.date }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        guard let date = calendar.date(from: components) else { throw Failure.date }
        try self.init(date: date, latitude: latitude, longitude: longitude)
    }

    /// The position at a Julian day in universal time.
    public init(julianDay: Double, latitude: Double, longitude: Double) throws {
        guard julianDay.isFinite, (0...5_373_484).contains(julianDay) else { throw Failure.date }
        guard latitude.isFinite, (-90...90).contains(latitude) else { throw Failure.latitude }
        guard longitude.isFinite, (-180...180).contains(longitude) else { throw Failure.longitude }
        let radians = Double.pi / 180
        let century = (julianDay - 2_451_545) / 36_525

        // Geometric mean longitude and anomaly, orbital eccentricity and equation of center.
        let meanLongitude = Self.wrap(280.46646 + century * (36_000.76983 + century * 0.0003032))
        let meanAnomaly = 357.52911 + century * (35_999.05029 - 0.0001537 * century)
        let eccentricity = 0.016708634 - century * (0.000042037 + 0.0000001267 * century)
        let anomaly = meanAnomaly * radians
        let center = sin(anomaly) * (1.914602 - century * (0.004817 + 0.000014 * century))
            + sin(2 * anomaly) * (0.019993 - 0.000101 * century) + sin(3 * anomaly) * 0.000289
        let trueLongitude = meanLongitude + center
        let trueAnomaly = meanAnomaly + center
        let distance = 1.000001018 * (1 - eccentricity * eccentricity)
            / (1 + eccentricity * cos(trueAnomaly * radians))

        // Nutation and aberration in longitude, then the corrected obliquity of the ecliptic.
        let omega = (125.04 - 1_934.136 * century) * radians
        let apparent = trueLongitude - 0.00569 - 0.00478 * sin(omega)
        let meanObliquity = 23 + (26 + (21.448 - century * (46.815 + century * (0.00059 - century * 0.001813))) / 60) / 60
        let obliquity = (meanObliquity + 0.00256 * cos(omega)) * radians
        let lambda = apparent * radians
        let ascension = Self.wrap(atan2(cos(obliquity) * sin(lambda), cos(lambda)) / radians)
        let declination = asin(sin(obliquity) * sin(lambda))

        let y = pow(tan(obliquity / 2), 2)
        let l0 = meanLongitude * radians
        let equation = 4 / radians * (
            y * sin(2 * l0) - 2 * eccentricity * sin(anomaly)
                + 4 * eccentricity * y * sin(anomaly) * cos(2 * l0)
                - 0.5 * y * y * sin(4 * l0) - 1.25 * eccentricity * eccentricity * sin(2 * anomaly)
        )

        // True solar time from universal time; NOAA's local time minus zone offset is the same instant.
        let universalMinutes = ((julianDay + 0.5) - (julianDay + 0.5).rounded(.down)) * 1_440
        let solarTime = Self.wrap(universalMinutes + equation + 4 * longitude, period: 1_440)
        let hourAngle = solarTime / 4 - 180

        let phi = latitude * radians
        let cosZenith = simd_clamp(
            sin(phi) * sin(declination) + cos(phi) * cos(declination) * cos(hourAngle * radians), -1, 1
        )
        let zenith = acos(cosZenith)
        let elevation = 90 - zenith / radians

        // Azimuth from Meeus's formula 13.5, measured westward from south and then
        // turned to clockwise from north. It equals NOAA's arccosine form, and stays
        // defined at the poles, where it is the limit along the given longitude's
        // meridian: hour angle + 180° at the north pole and −hour angle at the south pole.
        let hour = hourAngle * radians
        let east = sin(hour)
        let south = cos(hour) * sin(phi) - tan(declination) * cos(phi)
        let azimuth: Double
        if abs(east) < 1e-12, abs(south) < 1e-12 {
            // The Sun at the zenith has no azimuth; face south in the north, north in the south.
            azimuth = latitude > 0 ? 180 : 0
        } else {
            azimuth = Self.wrap(atan2(east, south) / radians + 180)
        }

        self.julianDay = julianDay
        apparentLongitude = Self.wrap(apparent)
        rightAscension = ascension
        self.declination = declination / radians
        earthSunDistance = distance
        equationOfTime = equation
        self.hourAngle = hourAngle
        geometricElevation = elevation
        self.elevation = elevation + Self.refraction(elevation: elevation)
        self.azimuth = azimuth
    }

    /// Wraps an angle or a time into [0, period).
    private static func wrap(_ value: Double, period: Double = 360) -> Double {
        let wrapped = value - period * (value / period).rounded(.down)
        return wrapped < period ? wrapped : 0
    }

    /// NOAA's approximate atmospheric refraction in degrees for a geometric elevation in degrees.
    public static func refraction(elevation: Double) -> Double {
        guard elevation.isFinite, elevation <= 85 else { return 0 }
        let tangent = tan(elevation * .pi / 180)
        let arcseconds: Double
        if elevation > 5 {
            arcseconds = 58.1 / tangent - 0.07 / pow(tangent, 3) + 0.000086 / pow(tangent, 5)
        } else if elevation > -0.575 {
            arcseconds = 1_735 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)))
        } else {
            arcseconds = -20.772 / tangent
        }
        return arcseconds / 3_600
    }

    /// The unit direction from the observer toward the Sun in Frostfield's right-handed
    /// world: +Y up, +X east and −Z north, so a camera with an identity rotation faces
    /// north. Pass `refracted: false` for the geometric (airless) direction.
    public func direction(refracted: Bool = true) -> SIMD3<Float> {
        let radians = Double.pi / 180
        let elevation = (refracted ? self.elevation : geometricElevation) * radians
        let azimuth = self.azimuth * radians
        return SIMD3<Float>(SIMD3(cos(elevation) * sin(azimuth), sin(elevation), -cos(elevation) * cos(azimuth)))
    }

    public enum Failure: LocalizedError, Equatable {
        case date
        case latitude
        case longitude

        public var errorDescription: String? {
            switch self {
            case .date: "Use a finite date between 4713 BCE and 10000 CE with a year, month and day."
            case .latitude: "Latitude must be between −90 and 90 degrees, north positive."
            case .longitude: "Longitude must be between −180 and 180 degrees, east positive."
            }
        }
    }
}
