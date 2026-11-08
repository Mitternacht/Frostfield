import Metal
import Testing
@testable import Frostfield

/// The survey must report exactly what a device's family probe answers, and
/// must find the highest Apple family even when the SDK lacks a case name.
struct RenderCapabilitiesTests {
    @Test func reportsHighestAppleFamilyFromProbe() {
        let survey = RenderCapabilities(deviceName: "probe") { family in
            (MTLGPUFamily.apple1.rawValue...MTLGPUFamily.apple7.rawValue).contains(family.rawValue)
        }
        #expect(survey.highestAppleFamily == 7)
        #expect(survey.supportsMetal3 == false)
        #expect(survey.supportsMetal4 == false)
    }

    @Test func reportsNoAppleFamilyForNonAppleProbe() {
        let survey = RenderCapabilities(deviceName: "probe") { $0 == .metal3 }
        #expect(survey.highestAppleFamily == nil)
        #expect(survey.supportsMetal3)
    }

    @Test func findsFamiliesNewerThanNamedCases() {
        // Raw value 1015 has no case name in any current SDK; the survey must
        // still report it because it probes by raw value.
        let survey = RenderCapabilities(deviceName: "probe") { $0.rawValue == 1015 }
        #expect(survey.highestAppleFamily == 15)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func surveysTheSystemDevice() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let survey = RenderCapabilities(device: device)
        #expect(survey.deviceName == device.name)
        #expect(survey.supportsMetal3 == device.supportsFamily(.metal3))
    }
}
