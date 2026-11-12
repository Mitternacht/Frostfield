import Frostfield
import Testing

struct RenderScheduleTests {
    @Test
    func `A missing drawable can retry once per redraw request`() {
        var schedule = RenderSchedule()
        schedule.setActive(true)

        let firstRetry = schedule.reserveDrawableRetry()
        #expect(firstRetry)
        let repeatedRetry = schedule.reserveDrawableRetry()
        #expect(!repeatedRetry)
        #expect(schedule.canDraw)

        schedule.invalidate()
        let afterInvalidation = schedule.reserveDrawableRetry()
        #expect(afterInvalidation)

        schedule.setActive(false)
        let hiddenRetry = schedule.reserveDrawableRetry()
        #expect(!hiddenRetry)
    }

    @Test
    func `On-demand rendering draws once per invalidation`() throws {
        var schedule = RenderSchedule()

        #expect(!schedule.shouldRequestDisplay)
        let startedInactive = schedule.beginFrame()
        #expect(!startedInactive)

        schedule.setActive(true)
        #expect(schedule.shouldRequestDisplay)

        let initialFrame = schedule.beginFrame()
        try #require(initialFrame)
        schedule.finishFrame()

        schedule.setActive(true)
        schedule.setMode(.onDemand)
        #expect(!schedule.shouldRequestDisplay)

        schedule.invalidate()
        schedule.invalidate()
        #expect(schedule.shouldRequestDisplay)
        let invalidatedFrame = schedule.beginFrame()
        try #require(invalidatedFrame)
        schedule.finishFrame()
        #expect(!schedule.canDraw)
    }

    @Test
    func `Invalidation during a frame waits for its completion`() throws {
        var schedule = RenderSchedule()
        schedule.setActive(true)

        let initialFrame = schedule.beginFrame()
        try #require(initialFrame)

        schedule.invalidate()
        #expect(!schedule.shouldRequestDisplay)
        let overlappingFrame = schedule.beginFrame()
        #expect(!overlappingFrame)

        schedule.finishFrame()
        #expect(schedule.shouldRequestDisplay)
        let followUpFrame = schedule.beginFrame()
        try #require(followUpFrame)
        schedule.finishFrame()
        #expect(!schedule.canDraw)
    }

    @Test
    func `Resuming while a frame is in flight waits for its completion`() throws {
        var schedule = RenderSchedule()
        schedule.setActive(true)

        let initialFrame = schedule.beginFrame()
        try #require(initialFrame)

        schedule.setActive(false)
        schedule.setActive(true)
        #expect(!schedule.canDraw)
        #expect(!schedule.shouldRequestDisplay)

        schedule.finishFrame()
        #expect(schedule.shouldRequestDisplay)
    }

    @Test
    func `Continuous mode pauses and can return to on-demand rendering`() throws {
        var schedule = RenderSchedule()
        schedule.setContinuous(true)
        #expect(!schedule.shouldRunContinuously)
        schedule.setActive(true)
        #expect(schedule.shouldRunContinuously)
        schedule.setActive(false)
        schedule.setMode(.continuous)
        #expect(!schedule.shouldRunContinuously)

        schedule.setActive(true)
        #expect(schedule.shouldRunContinuously)
        #expect(!schedule.shouldRequestDisplay)

        let firstFrame = schedule.beginFrame()
        try #require(firstFrame)
        #expect(!schedule.canDraw)
        schedule.finishFrame()
        let secondFrame = schedule.beginFrame()
        try #require(secondFrame)

        schedule.setActive(false)
        #expect(!schedule.shouldRunContinuously)
        schedule.invalidate()
        schedule.finishFrame()
        #expect(!schedule.canDraw)

        schedule.setActive(true)
        #expect(schedule.shouldRunContinuously)
        let resumedFrame = schedule.beginFrame()
        try #require(resumedFrame)
        schedule.finishFrame()

        schedule.setMode(.onDemand)
        #expect(!schedule.shouldRunContinuously)
        #expect(!schedule.shouldRequestDisplay)
        #expect(!schedule.canDraw)
    }
}
