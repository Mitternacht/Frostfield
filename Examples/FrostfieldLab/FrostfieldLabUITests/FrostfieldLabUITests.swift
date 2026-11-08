import XCTest

/// Drives the lab's render host through rendering, failure, invalidation,
/// visibility, background, and rotation, and checks the counters the host
/// publishes in its status label's accessibility value.
final class FrostfieldLabUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// The host renders through the package renderer: a frame is submitted and
    /// completed, and no failure message is shown.
    @MainActor
    func testRenderHostSubmitsAFrame() throws {
        let app = XCUIApplication()
        app.launch()

        XCTAssertTrue(app.otherElements["renderView"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts["renderFailure"].exists)
        let counters = try waitForCounters(in: app) { $0["submitted"] >= 1 && $0["completed"] >= 1 }
        XCTAssertEqual(counters.text("pacing"), "onRequest")
    }

    /// Without a Metal device the lab shows the renderer's typed error instead
    /// of an empty view.
    @MainActor
    func testMissingDeviceShowsFailureState() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-FrostfieldLabSimulateMissingDevice"]
        app.launch()

        let failure = app.staticTexts["renderFailure"]
        XCTAssertTrue(failure.waitForExistence(timeout: 10))
        XCTAssertTrue(failure.label.contains("No Metal device is available."), failure.label)
        XCTAssertFalse(app.otherElements["renderView"].exists)
    }

    /// In requested mode a still scene submits nothing and receives no draw
    /// callbacks; one invalidation produces exactly one frame.
    @MainActor
    func testStillSceneRedrawsOnlyAfterInvalidation() throws {
        let app = XCUIApplication()
        app.launch()
        let settled = try waitForSettledCounters(in: app)
        XCTAssertEqual(settled.text("mode"), "requested")

        Thread.sleep(forTimeInterval: 2)
        let still = try readCounters(in: app)
        XCTAssertEqual(still["submitted"], settled["submitted"], "a still scene submitted frames")
        XCTAssertEqual(still["callbacks"], settled["callbacks"], "a still scene received draw callbacks")
        record("still for 2 s: \(settled) -> \(still)")

        app.buttons["redraw"].tap()
        let redrawn = try waitForCounters(in: app) { $0["completed"] == settled["submitted"] + 1 }
        XCTAssertEqual(redrawn["submitted"], settled["submitted"] + 1)
        Thread.sleep(forTimeInterval: 1)
        XCTAssertEqual(try readCounters(in: app)["submitted"], settled["submitted"] + 1, "one invalidation, one frame")
    }

    /// Continuous drawing submits recurring frames; hiding the view stops both
    /// submissions and draw callbacks until it is shown again.
    @MainActor
    func testHiddenContinuousSceneStopsSubmitting() throws {
        let app = XCUIApplication()
        app.launch()
        let start = try waitForSettledCounters(in: app)

        app.segmentedControls["drawingMode"].buttons["Continuous"].tap()
        _ = try waitForCounters(in: app) { $0["submitted"] >= start["submitted"] + 30 }

        app.buttons["toggleVisibility"].tap()
        let hidden = try waitForSettledCounters(in: app)
        XCTAssertEqual(hidden.text("pacing"), "stopped")
        Thread.sleep(forTimeInterval: 2)
        let stillHidden = try readCounters(in: app)
        XCTAssertEqual(stillHidden["submitted"], hidden["submitted"], "hidden content submitted frames")
        XCTAssertEqual(stillHidden["callbacks"], hidden["callbacks"], "hidden content received draw callbacks")
        record("hidden for 2 s: \(hidden) -> \(stillHidden)")

        app.buttons["toggleVisibility"].tap()
        let shown = try waitForCounters(in: app) { $0["submitted"] >= hidden["submitted"] + 30 }
        XCTAssertEqual(shown.text("pacing"), "continuous")
    }

    /// A continuous scene sent to the background stops its loop: while the
    /// scene is not active it receives at most the single redraw it owes on
    /// returning, then resumes continuous drawing when active again.
    @MainActor
    func testBackgroundedContinuousSceneSubmitsNoRecurringWork() throws {
        let app = XCUIApplication()
        app.launch()
        _ = try waitForSettledCounters(in: app)
        app.segmentedControls["drawingMode"].buttons["Continuous"].tap()
        let running = try waitForCounters(in: app) { $0["submitted"] >= 30 }

        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 3)
        app.activate()

        let resumed = try waitForCounters(in: app) {
            $0.text("activity") == "active" && $0["submitted"] >= running["submitted"] + 30
        }
        let callbacksWhileAway = resumed["callbacksWhileNotActive"] - running["callbacksWhileNotActive"]
        let submittedWhileAway = resumed["submittedWhileNotActive"] - running["submittedWhileNotActive"]
        record("while not active: \(callbacksWhileAway) callbacks, \(submittedWhileAway) frames; \(running) -> \(resumed)")
        XCTAssertLessThanOrEqual(callbacksWhileAway, 1, "draw callbacks while not active: \(callbacksWhileAway)")
        XCTAssertLessThanOrEqual(submittedWhileAway, 1, "frames submitted while not active: \(submittedWhileAway)")
        XCTAssertEqual(resumed.text("pacing"), "continuous")
    }

    /// A still scene redraws exactly once after returning from the background
    /// and then stays still.
    @MainActor
    func testRequestedSceneRedrawsOnceAfterBackground() throws {
        let app = XCUIApplication()
        app.launch()
        let settled = try waitForSettledCounters(in: app)

        XCUIDevice.shared.press(.home)
        Thread.sleep(forTimeInterval: 2)
        app.activate()

        _ = try waitForCounters(in: app) { $0.text("activity") == "active" }
        let returned = try waitForSettledCounters(in: app)
        record("background and return: \(settled) -> \(returned)")
        XCTAssertEqual(returned["submitted"], settled["submitted"] + 1, "one redraw after the background")
        Thread.sleep(forTimeInterval: 1)
        XCTAssertEqual(try readCounters(in: app)["submitted"], settled["submitted"] + 1)
    }

    /// Rotation changes the view's bounds; the drawable follows the bounds
    /// times the display scale, and the new size is drawn once.
    @MainActor
    func testRotationResizesTheDrawable() throws {
        XCUIDevice.shared.orientation = .portrait
        addTeardownBlock { @MainActor in XCUIDevice.shared.orientation = .portrait }
        let app = XCUIApplication()
        app.launch()
        let portrait = try waitForSettledCounters(in: app)
        try assertDrawableCoversWindow(portrait, in: app)

        XCUIDevice.shared.orientation = .landscapeLeft
        let landscape = try waitForCounters(in: app) {
            $0.text("drawable") != portrait.text("drawable") && $0["completed"] == $0["submitted"]
        }
        try assertDrawableCoversWindow(landscape, in: app)
        XCTAssertGreaterThan(landscape["submitted"], portrait["submitted"], "the resized drawable was drawn")
        record("rotation: \(portrait) -> \(landscape)")

        let portraitSize = try XCTUnwrap(portrait.size("drawable"))
        let landscapeSize = try XCTUnwrap(landscape.size("drawable"))
        XCTAssertEqual(landscapeSize.width, portraitSize.height)
        XCTAssertEqual(landscapeSize.height, portraitSize.width)
    }

    /// FF-06: the camera fixture's pose and field of view are app state. Each
    /// change draws one frame, and neither the changes, the fixture switch,
    /// nor continuous animation creates a pipeline or a buffer.
    @MainActor
    func testCameraFixtureUpdatesCreateNoGPUResources() throws {
        let app = XCUIApplication()
        app.launch()
        let triangle = try waitForSettledCounters(in: app)
        XCTAssertEqual(triangle.text("fixture"), "triangle")

        app.segmentedControls["fixture"].buttons["Camera"].tap()
        let start = try waitForCounters(in: app) {
            $0.text("fixture") == "camera" && $0["completed"] == triangle["submitted"] + 1
        }
        XCTAssertEqual(start.text("pose"), "rest")
        XCTAssertEqual(start.text("fieldOfView"), "60")

        var previous = start
        for pose in ["Moved", "Turned", "Scaled", "Rest"] {
            app.segmentedControls["cameraPose"].buttons[pose].tap()
            let drawn = try waitForCounters(in: app) {
                $0.text("pose") == pose.lowercased() && $0["completed"] == previous["submitted"] + 1
            }
            XCTAssertEqual(drawn["submitted"], previous["submitted"] + 1, "one pose change, one frame")
            previous = drawn
        }

        app.sliders["fieldOfView"].adjust(toNormalizedSliderPosition: 0.9)
        let widened = try waitForSettledCounters(in: app)
        let fieldOfView = try XCTUnwrap(widened.text("fieldOfView").flatMap(Int.init))
        XCTAssertGreaterThan(fieldOfView, 80)
        XCTAssertGreaterThan(widened["submitted"], previous["submitted"], "the new field of view was drawn")

        app.segmentedControls["drawingMode"].buttons["Continuous"].tap()
        let animated = try waitForCounters(in: app) { $0["submitted"] >= widened["submitted"] + 60 }
        record("camera fixture: \(triangle) -> \(animated)")

        for counters in [start, previous, widened, animated] {
            XCTAssertEqual(counters["pipelinesCreated"], triangle["pipelinesCreated"])
            XCTAssertEqual(counters["buffersAllocated"], triangle["buffersAllocated"])
            XCTAssertEqual(counters["bufferBytesAllocated"], triangle["bufferBytesAllocated"])
        }
        XCTAssertEqual(triangle["pipelinesCreated"], 2)
        XCTAssertEqual(triangle["buffersAllocated"], 2)
    }

    // MARK: Helpers

    /// Writes measured counters into the test log as an activity.
    @MainActor
    private func record(_ message: String) {
        XCTContext.runActivity(named: "Measured \(message)") { _ in }
    }

    /// Checks that the drawable equals the window's size in points times the
    /// reported display scale.
    @MainActor
    private func assertDrawableCoversWindow(_ counters: FrameCounters, in app: XCUIApplication) throws {
        let window = app.windows.firstMatch.frame.size
        let scale = try XCTUnwrap(counters.double("scale"))
        let drawable = try XCTUnwrap(counters.size("drawable"))
        XCTAssertEqual(drawable.width, (window.width * scale).rounded(), "drawable \(drawable), window \(window) @\(scale)x")
        XCTAssertEqual(drawable.height, (window.height * scale).rounded(), "drawable \(drawable), window \(window) @\(scale)x")
    }

    @MainActor
    private func readCounters(in app: XCUIApplication) throws -> FrameCounters {
        let status = app.staticTexts["frameStatus"]
        XCTAssertTrue(status.waitForExistence(timeout: 10))
        return FrameCounters(status.value as? String)
    }

    /// Waits until every submitted frame has completed and two readings half a
    /// second apart agree.
    @MainActor
    private func waitForSettledCounters(in app: XCUIApplication) throws -> FrameCounters {
        var previous = try waitForCounters(in: app) { $0["submitted"] >= 1 && $0["completed"] == $0["submitted"] }
        for _ in 0..<20 {
            Thread.sleep(forTimeInterval: 0.5)
            let current = try readCounters(in: app)
            if current == previous, current["completed"] == current["submitted"] {
                return current
            }
            previous = current
        }
        XCTFail("counters did not settle: \(previous)")
        return previous
    }

    @MainActor
    private func waitForCounters(
        in app: XCUIApplication,
        timeout: TimeInterval = 10,
        until condition: @escaping (FrameCounters) -> Bool
    ) throws -> FrameCounters {
        let status = app.staticTexts["frameStatus"]
        XCTAssertTrue(status.waitForExistence(timeout: timeout))
        let predicate = NSPredicate { _, _ in condition(FrameCounters(status.value as? String)) }
        let expectation = XCTNSPredicateExpectation(predicate: predicate, object: nil)
        let result = XCTWaiter.wait(for: [expectation], timeout: timeout)
        let counters = FrameCounters(status.value as? String)
        XCTAssertEqual(result, .completed, "status: \(counters)")
        return counters
    }
}

/// Parses the status label's `key=value;key=value` accessibility value.
struct FrameCounters: Equatable, CustomStringConvertible {
    private var values: [String: String] = [:]

    init(_ text: String?) {
        for pair in (text ?? "").split(separator: ";") {
            let parts = pair.split(separator: "=", maxSplits: 1)
            if parts.count == 2 {
                values[String(parts[0])] = String(parts[1])
            }
        }
    }

    /// An integer counter, or -1 when absent so comparisons fail visibly.
    subscript(key: String) -> Int { values[key].flatMap { Int($0) } ?? -1 }

    func text(_ key: String) -> String? { values[key] }

    func double(_ key: String) -> Double? { values[key].flatMap { Double($0) } }

    /// A `WIDTHxHEIGHT` value.
    func size(_ key: String) -> CGSize? {
        guard let parts = values[key]?.split(separator: "x"), parts.count == 2,
              let width = Double(parts[0]), let height = Double(parts[1]) else { return nil }
        return CGSize(width: width, height: height)
    }

    var description: String {
        values.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ";")
    }
}
