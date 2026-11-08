import CoreGraphics

/// Decides when one render host draws, from facts the host knows about its own
/// view and scene.
///
/// A host keeps one schedule per rendered view and reports, from its own
/// context: the drawing ``Mode``, its scene's ``Activity``, whether the view
/// is visible, its drawable size, content invalidations, and the outcome of
/// every frame. The schedule answers how the view's draw loop should run
/// (``drawLoop``), whether to ask for a display update
/// (``shouldRequestDisplay``), and whether a draw callback may submit a frame
/// (``frameWillBegin(at:)``).
///
/// The rules:
///
/// - A backgrounded scene, a hidden view, or an empty drawable stops drawing
///   entirely (``Pacing/stopped``). Invalidations are kept and drawn on resume.
/// - An active scene in ``Mode/continuous`` mode runs the draw loop.
/// - Every other visible state draws only after ``invalidate()``
///   (``Pacing/onRequest``). An inactive scene therefore submits no recurring
///   frames, but still redraws once after, for example, a resize.
/// - A requested frame that could not get a frame slot or a drawable stays
///   owed and is retried after a submitted frame completes (a missing drawable
///   also after the next invalidation), so the host never waits and never
///   re-requests in a tight loop.
/// - Animation time advances only while drawing continuously, by at most
///   ``maximumFrameStep`` per frame, so a resumed scene continues where it
///   stopped instead of jumping by the paused interval.
///
/// The schedule is a value with no clock, thread, or UI dependency. A host
/// mutates it from the thread that drives its view (the main actor for UIKit).
public struct RenderSchedule: Sendable, Equatable {
    /// How a visible, active host draws.
    public enum Mode: Sendable, Equatable {
        /// Draw only after ``RenderSchedule/invalidate()``. A still scene
        /// submits nothing.
        case requested
        /// Draw on every display refresh while the scene is active.
        case continuous
    }

    /// The host scene's lifecycle state, mapped from its own window scene.
    public enum Activity: Sendable, Equatable {
        /// Foreground and receiving events.
        case active
        /// Foreground but not receiving events, for example during a system
        /// overlay or an app-switcher transition.
        case inactive
        /// Not on screen.
        case background
    }

    /// How the host's draw loop runs in the current state.
    public enum Pacing: Sendable, Equatable {
        /// No draw callbacks and no submissions.
        case stopped
        /// Draw callbacks only after a display request; no recurring frames.
        case onRequest
        /// Draw on every display refresh.
        case continuous
    }

    /// Draw-loop settings in the terms `MTKView` uses.
    public struct DrawLoop: Sendable, Equatable {
        /// Whether the view's internal display-refresh loop is stopped.
        public var isPaused: Bool
        /// Whether a display request (`setNeedsDisplay()`) triggers a draw.
        public var enableSetNeedsDisplay: Bool
    }

    /// Why an owed frame is waiting instead of being requested again.
    public enum Deferral: Sendable, Equatable {
        /// Every frame slot was in flight. Retried after a frame completes.
        case frameSlot
        /// The host supplied no drawable, or Metal could not start encoding.
        /// Retried after a frame completes or the content is invalidated.
        case target
    }

    /// The largest animation-time advance between two continuous frames, in
    /// seconds. A longer gap, such as a stalled main thread, advances
    /// animation time by this much only.
    public static let maximumFrameStep: Double = 0.1

    /// The drawing mode for a visible, active host.
    public private(set) var mode: Mode
    /// The host scene's lifecycle state. Starts as ``Activity/background``
    /// until the host reports its scene.
    public private(set) var activity: Activity = .background
    /// Whether the rendered view is in a window and not hidden.
    public private(set) var isVisible = false
    /// The drawable size in pixels, or `nil` while the view has no area.
    public private(set) var drawableSize: CGSize?
    /// Whether presented content is stale and a frame is owed.
    public private(set) var needsDisplay = true
    /// Why an owed frame is waiting, if it is.
    public private(set) var deferral: Deferral?
    /// Animation time in seconds. Advances only during continuous pacing.
    public private(set) var animationTime: Double = 0

    /// The timestamp of the last continuous frame, or `nil` when the next
    /// continuous frame starts a new interval.
    private var lastFrameTimestamp: Double?

    /// Creates a schedule that is stopped until the host reports an active or
    /// inactive scene, a visible view, and a non-empty drawable.
    public init(mode: Mode) {
        self.mode = mode
    }

    /// How the draw loop runs, from activity, visibility, drawable, and mode.
    public var pacing: Pacing {
        guard activity != .background, isVisible, drawableSize != nil else { return .stopped }
        return activity == .active && mode == .continuous ? .continuous : .onRequest
    }

    /// The `MTKView` loop settings for ``pacing``.
    public var drawLoop: DrawLoop {
        switch pacing {
        case .stopped: DrawLoop(isPaused: true, enableSetNeedsDisplay: false)
        case .onRequest: DrawLoop(isPaused: true, enableSetNeedsDisplay: true)
        case .continuous: DrawLoop(isPaused: false, enableSetNeedsDisplay: false)
        }
    }

    /// Whether the host should ask its view for one display update now: a
    /// frame is owed, drawing is on request, and the frame is not deferred.
    public var shouldRequestDisplay: Bool {
        pacing == .onRequest && needsDisplay && deferral == nil
    }

    // MARK: Host state

    public mutating func setMode(_ mode: Mode) {
        self.mode = mode
        pacingMayHaveChanged()
    }

    /// Reports the host scene's lifecycle state. Returning from the background
    /// invalidates once rather than relying on the layer keeping its contents
    /// across suspension.
    public mutating func setActivity(_ activity: Activity) {
        if self.activity == .background, activity != .background {
            needsDisplay = true
        }
        self.activity = activity
        pacingMayHaveChanged()
    }

    public mutating func setVisible(_ isVisible: Bool) {
        self.isVisible = isVisible
        pacingMayHaveChanged()
    }

    /// Records the view's size and the display scale of its own trait
    /// environment, and invalidates when the resulting pixel size changes.
    ///
    /// - Returns: Whether ``drawableSize`` changed, so the host resizes its
    ///   drawable.
    @discardableResult
    public mutating func updateDrawableSize(boundsSize: CGSize, displayScale: CGFloat) -> Bool {
        let size = Self.drawableSize(boundsSize: boundsSize, displayScale: displayScale)
        guard size != drawableSize else { return false }
        drawableSize = size
        invalidate()
        pacingMayHaveChanged()
        return true
    }

    /// The pixel size of a drawable covering `boundsSize` points at
    /// `displayScale`, rounded to whole pixels, or `nil` when either
    /// dimension is empty or not finite.
    public static func drawableSize(boundsSize: CGSize, displayScale: CGFloat) -> CGSize? {
        let width = (boundsSize.width * displayScale).rounded()
        let height = (boundsSize.height * displayScale).rounded()
        guard width.isFinite, height.isFinite, width >= 1, height >= 1 else { return nil }
        return CGSize(width: width, height: height)
    }

    /// Marks presented content as stale, for example after a content or
    /// drawable change. A frame is then owed; a stopped host keeps the debt
    /// until it resumes. Clears a ``Deferral/target`` deferral, because the
    /// change may have made a drawable available.
    public mutating func invalidate() {
        needsDisplay = true
        if deferral == .target { deferral = nil }
    }

    // MARK: Frames

    /// Called at the start of a draw callback.
    ///
    /// - Parameter timestamp: A monotonic time in seconds.
    /// - Returns: The animation time to render, or `nil` when this callback
    ///   must not submit a frame: drawing is stopped, or drawing is on request
    ///   and nothing is owed.
    public mutating func frameWillBegin(at timestamp: Double) -> Double? {
        switch pacing {
        case .stopped:
            return nil
        case .onRequest:
            return needsDisplay ? animationTime : nil
        case .continuous:
            if let last = lastFrameTimestamp {
                animationTime += min(max(timestamp - last, 0), Self.maximumFrameStep)
            }
            lastFrameTimestamp = timestamp
            return animationTime
        }
    }

    /// Records the renderer's outcome for a frame that ``frameWillBegin(at:)``
    /// allowed.
    public mutating func frameDidEnd(_ outcome: FrameOutcome) {
        switch outcome {
        case .submitted:
            needsDisplay = false
            deferral = nil
        case .skippedNoSlot:
            deferral = .frameSlot
        case .skippedNoTarget, .encodingFailed:
            deferral = .target
        }
    }

    /// Records that a submitted frame completed on the GPU, which frees its
    /// frame slot and drawable. Ends any deferral, so an owed requested frame
    /// is requested again.
    public mutating func frameDidComplete() {
        deferral = nil
    }

    private mutating func pacingMayHaveChanged() {
        if pacing != .continuous {
            lastFrameTimestamp = nil
        }
    }
}
