import CoreGraphics
import Testing
@testable import Frostfield

/// Walks the schedule through the lifecycle, invalidation, and backpressure
/// transitions a UIKit host feeds it, and checks what the host is told to do.
struct RenderScheduleTests {
    /// An active, visible, sized host that has not drawn yet.
    static func ready(_ mode: RenderSchedule.Mode = .requested) -> RenderSchedule {
        var schedule = RenderSchedule(mode: mode)
        schedule.setActivity(.active)
        schedule.setVisible(true)
        schedule.updateDrawableSize(boundsSize: CGSize(width: 100, height: 200), displayScale: 2)
        return schedule
    }

    /// An active, visible, sized host that presented its first frame at time 0.
    static func settled(_ mode: RenderSchedule.Mode = .requested) -> RenderSchedule {
        var schedule = ready(mode)
        _ = schedule.frameWillBegin(at: 0)
        schedule.frameDidEnd(.submitted(frame: 1), framesInFlight: 1)
        return schedule
    }

    // MARK: Drawable size

    @Test(arguments: [
        (CGSize(width: 402, height: 874), 3.0, CGSize(width: 1206, height: 2622)),
        (CGSize(width: 820, height: 1180), 2.0, CGSize(width: 1640, height: 2360)),
        (CGSize(width: 512.5, height: 300.25), 2.0, CGSize(width: 1025, height: 601)),
        (CGSize(width: 0.4, height: 0.4), 3.0, CGSize(width: 1, height: 1)),
    ])
    func `Drawable size is bounds times display scale in whole pixels`(bounds: CGSize, scale: CGFloat, expected: CGSize) {
        #expect(RenderSchedule.drawableSize(boundsSize: bounds, displayScale: scale) == expected)
    }

    @Test(arguments: [
        (CGSize.zero, 2.0),
        (CGSize(width: 100, height: 0), 3.0),
        (CGSize(width: 100, height: 100), 0.0),
        (CGSize(width: 0.1, height: 100), 3.0),
        (CGSize(width: CGFloat.nan, height: 10), 2.0),
        (CGSize(width: CGFloat.infinity, height: 10), 2.0),
    ])
    func `Empty or non-finite sizes have no drawable`(bounds: CGSize, scale: CGFloat) {
        #expect(RenderSchedule.drawableSize(boundsSize: bounds, displayScale: scale) == nil)
    }

    @Test func `A size or scale change invalidates; an identical layout does not`() {
        var schedule = Self.settled()
        #expect(!schedule.needsDisplay)

        let resized1 = schedule.updateDrawableSize(boundsSize: CGSize(width: 100, height: 200), displayScale: 2)
        #expect(!resized1)
        #expect(!schedule.needsDisplay)

        // Rotation: same scale, swapped bounds.
        let resized2 = schedule.updateDrawableSize(boundsSize: CGSize(width: 200, height: 100), displayScale: 2)
        #expect(resized2)
        #expect(schedule.drawableSize == CGSize(width: 400, height: 200))
        #expect(schedule.shouldRequestDisplay)
        _ = schedule.frameWillBegin(at: 1)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)

        // Moving to a display with another scale: same bounds, new scale.
        let resized3 = schedule.updateDrawableSize(boundsSize: CGSize(width: 200, height: 100), displayScale: 3)
        #expect(resized3)
        #expect(schedule.drawableSize == CGSize(width: 600, height: 300))
        #expect(schedule.shouldRequestDisplay)
    }

    // MARK: Pacing

    @Test func `Drawing starts only when active or inactive, visible, and sized`() {
        var schedule = RenderSchedule(mode: .continuous)
        #expect(schedule.pacing == .stopped, "a new schedule waits for its host")

        schedule.setActivity(.active)
        #expect(schedule.pacing == .stopped, "not visible yet")
        schedule.setVisible(true)
        #expect(schedule.pacing == .stopped, "no drawable area yet")
        schedule.updateDrawableSize(boundsSize: CGSize(width: 10, height: 10), displayScale: 2)
        #expect(schedule.pacing == .continuous)

        schedule.updateDrawableSize(boundsSize: .zero, displayScale: 2)
        #expect(schedule.pacing == .stopped, "an empty view has nothing to draw")
    }

    @Test(arguments: [
        (RenderSchedule.Mode.continuous, RenderSchedule.Activity.active, true, RenderSchedule.Pacing.continuous),
        (.continuous, .inactive, true, .onRequest),
        (.continuous, .background, true, .stopped),
        (.continuous, .active, false, .stopped),
        (.requested, .active, true, .onRequest),
        (.requested, .inactive, true, .onRequest),
        (.requested, .background, true, .stopped),
        (.requested, .active, false, .stopped),
    ])
    func `Pacing follows mode, scene activity, and visibility`(
        mode: RenderSchedule.Mode, activity: RenderSchedule.Activity, visible: Bool, expected: RenderSchedule.Pacing
    ) {
        var schedule = RenderSchedule(mode: mode)
        schedule.updateDrawableSize(boundsSize: CGSize(width: 10, height: 10), displayScale: 2)
        schedule.setActivity(activity)
        schedule.setVisible(visible)
        #expect(schedule.pacing == expected)
    }

    @Test func `The draw loop runs only for continuous pacing`() {
        var schedule = Self.settled(.continuous)
        #expect(schedule.drawLoop == .init(isPaused: false, enableSetNeedsDisplay: false))
        schedule.setActivity(.inactive)
        #expect(schedule.drawLoop == .init(isPaused: true, enableSetNeedsDisplay: true))
        schedule.setVisible(false)
        #expect(schedule.drawLoop == .init(isPaused: true, enableSetNeedsDisplay: false))
    }

    // MARK: Pause and resume

    @Test func `Pausing a continuous scene stops recurring frames`() throws {
        var schedule = Self.settled(.continuous)
        let began4 = schedule.frameWillBegin(at: 1)
        #expect(began4 != nil)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)

        schedule.setActivity(.inactive)
        #expect(!schedule.shouldRequestDisplay, "nothing changed, so nothing is requested")
        let began5 = schedule.frameWillBegin(at: 2)
        #expect(began5 == nil, "a stray callback submits nothing")

        schedule.setActivity(.background)
        let began6 = schedule.frameWillBegin(at: 3)
        #expect(began6 == nil)
    }

    @Test func `An inactive scene still redraws once after invalidation`() {
        var schedule = Self.settled(.continuous)
        schedule.setActivity(.inactive)
        schedule.updateDrawableSize(boundsSize: CGSize(width: 300, height: 200), displayScale: 2)
        #expect(schedule.shouldRequestDisplay)
        let began7 = schedule.frameWillBegin(at: 1)
        #expect(began7 != nil)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        #expect(!schedule.shouldRequestDisplay)
        let began8 = schedule.frameWillBegin(at: 2)
        #expect(began8 == nil)
    }

    @Test func `Hidden content keeps an invalidation until it is visible again`() {
        var schedule = Self.settled()
        schedule.setVisible(false)
        schedule.invalidate()
        #expect(!schedule.shouldRequestDisplay)
        let began9 = schedule.frameWillBegin(at: 1)
        #expect(began9 == nil)

        schedule.setVisible(true)
        #expect(schedule.shouldRequestDisplay)
    }

    @Test func `Showing hidden, unchanged content does not redraw`() {
        var schedule = Self.settled()
        schedule.setVisible(false)
        schedule.setVisible(true)
        #expect(!schedule.shouldRequestDisplay)
    }

    @Test func `Returning from the background redraws once`() {
        var schedule = Self.settled()
        schedule.setActivity(.inactive)
        schedule.setActivity(.background)
        #expect(schedule.pacing == .stopped)
        #expect(!schedule.shouldRequestDisplay)

        schedule.setActivity(.inactive)
        #expect(schedule.shouldRequestDisplay)
        _ = schedule.frameWillBegin(at: 1)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        schedule.setActivity(.active)
        #expect(!schedule.shouldRequestDisplay, "becoming active again does not redraw")
    }

    // MARK: Invalidation

    @Test func `A still scene draws only after invalidation`() {
        var schedule = Self.settled()
        #expect(!schedule.shouldRequestDisplay)
        let began10 = schedule.frameWillBegin(at: 1)
        #expect(began10 == nil, "an uninvalidated callback is refused")

        schedule.invalidate()
        #expect(schedule.shouldRequestDisplay)
        let began11 = schedule.frameWillBegin(at: 2)
        #expect(began11 != nil)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        #expect(!schedule.shouldRequestDisplay)
        let began12 = schedule.frameWillBegin(at: 3)
        #expect(began12 == nil)
    }

    // MARK: Backpressure

    @Test func `A frame skipped for lack of a slot waits for a completion, then retries`() {
        var schedule = Self.settled()
        schedule.invalidate()
        let began13 = schedule.frameWillBegin(at: 1)
        #expect(began13 != nil)
        schedule.frameDidEnd(.skippedNoSlot, framesInFlight: 2)

        #expect(schedule.needsDisplay)
        #expect(schedule.deferral == .frameSlot)
        #expect(!schedule.shouldRequestDisplay, "no request until a slot frees")
        schedule.invalidate()
        #expect(!schedule.shouldRequestDisplay, "a new invalidation does not free a slot")

        schedule.frameDidComplete()
        #expect(schedule.shouldRequestDisplay)
        let began14 = schedule.frameWillBegin(at: 2)
        #expect(began14 != nil)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        schedule.frameDidComplete()
        #expect(!schedule.shouldRequestDisplay, "nothing owed after the retry")
    }

    @Test(arguments: [FrameOutcome.skippedNoTarget, .encodingFailed])
    func `A frame without a target waits for a completion or an invalidation`(outcome: FrameOutcome) {
        var schedule = Self.settled()
        schedule.invalidate()
        _ = schedule.frameWillBegin(at: 1)
        schedule.frameDidEnd(outcome, framesInFlight: 1)
        #expect(schedule.needsDisplay)
        #expect(schedule.deferral == .target)
        #expect(!schedule.shouldRequestDisplay, "no immediate re-request loop")

        var afterCompletion = schedule
        afterCompletion.frameDidComplete()
        #expect(afterCompletion.shouldRequestDisplay)

        var afterInvalidation = schedule
        afterInvalidation.invalidate()
        #expect(afterInvalidation.shouldRequestDisplay)
    }

    /// With no frame in flight no completion will arrive, so waiting for one
    /// would leave the owed frame undrawn: a requested scene resized after
    /// its last frame completed, whose drawable then fails, must still be
    /// asked to draw again. It is asked once; a second failure waits for the
    /// next invalidation instead of re-requesting in a loop.
    @Test(arguments: [FrameOutcome.skippedNoTarget, .encodingFailed])
    func `A frame without a target and nothing in flight is retried once`(outcome: FrameOutcome) throws {
        var schedule = Self.settled()
        schedule.frameDidComplete()
        schedule.updateDrawableSize(boundsSize: CGSize(width: 200, height: 100), displayScale: 2)
        #expect(schedule.shouldRequestDisplay)
        let began = schedule.frameWillBegin(at: 1)
        #expect(began != nil)
        schedule.frameDidEnd(outcome, framesInFlight: 0)

        #expect(schedule.needsDisplay)
        #expect(schedule.deferral == nil)
        #expect(schedule.shouldRequestDisplay, "the retry is requested although no completion will come")
        let retry = schedule.frameWillBegin(at: 1.016)
        #expect(retry != nil)
        schedule.frameDidEnd(outcome, framesInFlight: 0)

        #expect(schedule.needsDisplay)
        #expect(schedule.deferral == .target)
        #expect(!schedule.shouldRequestDisplay, "one retry, then wait for an invalidation")
        let stray = schedule.frameWillBegin(at: 1.032)
        #expect(stray != nil, "a draw callback from elsewhere may still draw the owed frame")
        schedule.frameDidEnd(outcome, framesInFlight: 0)
        #expect(!schedule.shouldRequestDisplay)

        schedule.invalidate()
        #expect(schedule.shouldRequestDisplay)
        _ = schedule.frameWillBegin(at: 2)
        schedule.frameDidEnd(outcome, framesInFlight: 0)
        #expect(schedule.shouldRequestDisplay, "an invalidation restores the retry")
        _ = schedule.frameWillBegin(at: 2.016)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        #expect(!schedule.needsDisplay)
        #expect(!schedule.shouldRequestDisplay)
    }

    @Test func `A submitted frame restores the retry for a later missing target`() {
        var schedule = Self.settled()
        schedule.frameDidComplete()
        schedule.invalidate()
        _ = schedule.frameWillBegin(at: 1)
        schedule.frameDidEnd(.skippedNoTarget, framesInFlight: 0)
        _ = schedule.frameWillBegin(at: 1.016)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        schedule.frameDidComplete()

        schedule.invalidate()
        _ = schedule.frameWillBegin(at: 2)
        schedule.frameDidEnd(.skippedNoTarget, framesInFlight: 0)
        #expect(schedule.deferral == nil)
        #expect(schedule.shouldRequestDisplay)
    }

    @Test func `Continuous drawing keeps drawing while a slot is deferred`() {
        var schedule = Self.settled(.continuous)
        _ = schedule.frameWillBegin(at: 1)
        schedule.frameDidEnd(.skippedNoSlot, framesInFlight: 2)
        #expect(schedule.pacing == .continuous)
        let began15 = schedule.frameWillBegin(at: 1.016)
        #expect(began15 != nil, "the next refresh tries again")
    }

    // MARK: Animation clock

    @Test func `Continuous frames advance the clock by the elapsed time`() throws {
        var schedule = Self.ready(.continuous)
        let began16 = schedule.frameWillBegin(at: 10.0)
        #expect(began16 == 0, "the first frame starts an interval")
        let began17 = schedule.frameWillBegin(at: 10.05)
        let next = try #require(began17)
        #expect(abs(next - 0.05) < 1e-9)
    }

    @Test func `A stalled frame advances at most one maximum step`() throws {
        var schedule = Self.settled(.continuous)
        _ = schedule.frameWillBegin(at: 0)
        let began18 = schedule.frameWillBegin(at: 5)
        let next = try #require(began18)
        #expect(next == RenderSchedule.maximumFrameStep)
    }

    @Test func `A timestamp that goes backwards does not reverse the clock`() throws {
        var schedule = Self.ready(.continuous)
        _ = schedule.frameWillBegin(at: 10)
        let began19 = schedule.frameWillBegin(at: 9)
        let next = try #require(began19)
        #expect(next == 0)
        let began19b = schedule.frameWillBegin(at: 9.02)
        let later = try #require(began19b)
        #expect(abs(later - 0.02) < 1e-9, "the clock continues from the earlier timestamp")
    }

    enum Pause: CaseIterable {
        case background, inactive, hidden, requestedMode
    }

    @Test(arguments: Pause.allCases)
    func `Resuming after a pause does not jump by the paused interval`(pause: Pause) throws {
        var schedule = Self.settled(.continuous)
        _ = schedule.frameWillBegin(at: 0)
        let began20 = schedule.frameWillBegin(at: 0.05)
        let beforePause = try #require(began20)

        switch pause {
        case .background: schedule.setActivity(.background)
        case .inactive: schedule.setActivity(.inactive)
        case .hidden: schedule.setVisible(false)
        case .requestedMode: schedule.setMode(.requested)
        }
        #expect(schedule.pacing != .continuous)
        let began21 = schedule.frameWillBegin(at: 10)
        #expect(began21 == nil, "nothing changed, so nothing is drawn while paused")

        switch pause {
        case .background, .inactive: schedule.setActivity(.active)
        case .hidden: schedule.setVisible(true)
        case .requestedMode: schedule.setMode(.continuous)
        }
        #expect(schedule.pacing == .continuous)

        // Thirty seconds later the clock resumes where it stopped.
        let began22 = schedule.frameWillBegin(at: 30)
        #expect(began22 == beforePause)
        let began23 = schedule.frameWillBegin(at: 30.02)
        let after = try #require(began23)
        #expect(abs(after - beforePause - 0.02) < 1e-9)
    }

    @Test func `Requested frames draw the paused animation time`() throws {
        var schedule = Self.settled(.continuous)
        _ = schedule.frameWillBegin(at: 0)
        _ = schedule.frameWillBegin(at: 0.04)
        schedule.frameDidEnd(.submitted(frame: 2), framesInFlight: 1)
        schedule.setMode(.requested)
        schedule.invalidate()
        let began24 = schedule.frameWillBegin(at: 9)
        let time = try #require(began24)
        #expect(abs(time - 0.04) < 1e-9)
        #expect(schedule.animationTime == time)
    }
}
