/// Frame scheduling for a single host. Its owner serializes access.
package struct RenderSchedule {
    package enum Mode {
        case onDemand
        case continuous
    }

    package private(set) var isActive = false
    private var mode: Mode = .onDemand
    private var needsDisplay = true
    private var isFrameInFlight = false
    private var canRetryDrawable = true

    package init() {}

    package var canDraw: Bool {
        isActive && !isFrameInFlight && (needsDisplay || mode == .continuous)
    }

    package var shouldRunContinuously: Bool {
        isActive && mode == .continuous
    }

    package var shouldRequestDisplay: Bool {
        mode == .onDemand && canDraw
    }

    package mutating func setActive(_ active: Bool) {
        guard active != isActive else { return }
        isActive = active
        if active {
            invalidate()
        }
    }

    package mutating func setMode(_ mode: Mode) {
        self.mode = mode
    }

    package mutating func setContinuous(_ continuous: Bool) {
        setMode(continuous ? .continuous : .onDemand)
    }

    package mutating func invalidate() {
        needsDisplay = true
        canRetryDrawable = true
    }

    package mutating func reserveDrawableRetry() -> Bool {
        guard shouldRequestDisplay, canRetryDrawable else { return false }
        canRetryDrawable = false
        return true
    }

    package mutating func beginFrame() -> Bool {
        guard canDraw else { return false }
        isFrameInFlight = true
        // Consume the request now so invalidations during GPU work can queue another frame.
        needsDisplay = false
        return true
    }

    package mutating func finishFrame() {
        isFrameInFlight = false
    }
}
