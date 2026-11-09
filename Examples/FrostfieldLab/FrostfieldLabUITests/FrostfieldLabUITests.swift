import XCTest

/// Drives the lab's render host through rendering, failure, invalidation,
/// visibility, background, and rotation, and checks the counters the host
/// publishes in its status label's accessibility value. Every launch passes
/// `-FrostfieldLabUITesting`, without which the lab leaves that value to
/// VoiceOver's reading of the visible text.
final class FrostfieldLabUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// The host renders through the package renderer: a frame is submitted and
    /// completed, and no failure message is shown.
    @MainActor
    func testRenderHostSubmitsAFrame() throws {
        let app = makeApp()
        app.launch()

        XCTAssertTrue(app.otherElements["renderView"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts["renderFailure"].exists)
        let counters = try waitForCounters(in: app) { $0["submitted"] >= 1 && $0["completed"] >= 1 }
        XCTAssertEqual(counters.text("pacing"), "onRequest")
    }

    /// Without `-FrostfieldLabUITesting` the status label exposes only its
    /// visible text, so VoiceOver reads what the panel shows rather than the
    /// tests' `key=value` protocol.
    @MainActor
    func testStatusIsReadAsItsVisibleText() throws {
        let app = XCUIApplication()
        app.launch()
        let status = app.staticTexts["frameStatus"]
        XCTAssertTrue(status.waitForExistence(timeout: 10))
        let label = status.label
        XCTAssertTrue(label.contains("submitted"), label)
        XCTAssertTrue(label.contains("requested"), label)
        XCTAssertFalse(label.contains("="), label)
        let value = (status.value as? String) ?? ""
        XCTAssertFalse(value.contains("="), "accessibility value: \(value)")
    }

    /// Without a Metal device the lab shows the renderer's typed error instead
    /// of an empty view.
    @MainActor
    func testMissingDeviceShowsFailureState() throws {
        let app = makeApp(["-FrostfieldLabSimulateMissingDevice"])
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
        let app = makeApp()
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

    /// FF-03: a requested frame whose drawable is missing while no frame is
    /// in flight would wait for a completion that never comes. The host asks
    /// for it once more and draws it without any interaction.
    @MainActor
    func testMissingDrawableWithNothingInFlightIsRetried() throws {
        let app = makeApp(["-FrostfieldLabSimulateMissingDrawables", "1"])
        app.launch()
        let settled = try waitForSettledCounters(in: app)
        record("one missing drawable: \(settled)")
        XCTAssertEqual(settled["skippedNoTarget"], 1)
        XCTAssertGreaterThanOrEqual(settled["submitted"], 1, "the owed frame was drawn on the retry")
        XCTAssertEqual(settled.text("pacing"), "onRequest")
    }

    /// FF-03: the retry is bounded. When the retry also finds no drawable,
    /// the host stops asking (no callbacks in a loop) and draws after the
    /// next invalidation.
    @MainActor
    func testRepeatedMissingDrawableWaitsForInvalidation() throws {
        let app = makeApp(["-FrostfieldLabSimulateMissingDrawables", "2"])
        app.launch()
        let failed = try waitForCounters(in: app) { $0["skippedNoTarget"] == 2 }
        Thread.sleep(forTimeInterval: 2)
        let still = try readCounters(in: app)
        record("two missing drawables, then 2 s: \(failed) -> \(still)")
        XCTAssertEqual(still["submitted"], 0)
        XCTAssertEqual(still["skippedNoTarget"], 2)
        XCTAssertEqual(still["callbacks"], failed["callbacks"], "no draw callbacks after the bounded retry")

        app.buttons["redraw"].tap()
        let drawn = try waitForCounters(in: app) { $0["submitted"] == 1 && $0["completed"] == 1 }
        XCTAssertEqual(drawn["skippedNoTarget"], 2)
    }

    /// Continuous drawing submits recurring frames; hiding the view stops both
    /// submissions and draw callbacks until it is shown again.
    @MainActor
    func testHiddenContinuousSceneStopsSubmitting() throws {
        let app = makeApp()
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
        let app = makeApp()
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
        let app = makeApp()
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
        let app = makeApp()
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
        let app = makeApp()
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

    /// FF-07: the cube fixture uploads its mesh once when the view loads.
    /// Pose changes and continuous animation change only the transform, so
    /// the mesh renderer's pipeline, depth-state, and buffer counters stay
    /// constant. The app runs with Metal API validation, and switching to a
    /// fixture without a depth buffer and back must not trip it.
    @MainActor
    func testCubeFixtureReusesMeshBuffers() throws {
        let app = makeApp(["-FrostfieldLabFixture", "geometry"])
        app.launchEnvironment["MTL_DEBUG_LAYER"] = "1"
        app.launch()
        let start = try waitForSettledCounters(in: app)
        XCTAssertEqual(start.text("fixture"), "geometry")
        XCTAssertEqual(start.text("pose"), "front")
        XCTAssertEqual(start["meshVertices"], 24)
        XCTAssertEqual(start["meshIndices"], 36)
        // The normal diagnostic, plus the unlit and pulse pipelines the
        // material fixture prepares when the scene loads (FF-08).
        XCTAssertEqual(start["pipelinesCreated"], 3)
        XCTAssertEqual(start["depthStatesCreated"], 1)
        // Two per-frame uniform buffers plus the cube's position, normal, and
        // index buffers: 2 × 64 + 24 × 16 × 2 + 36 × 4 bytes.
        XCTAssertEqual(start["buffersAllocated"], 5)
        XCTAssertEqual(start["bufferBytesAllocated"], 1040)

        var previous = start
        for pose in ["Corner", "Stretched", "Front"] {
            app.segmentedControls["cubePose"].buttons[pose].tap()
            let drawn = try waitForCounters(in: app) {
                $0.text("pose") == pose.lowercased() && $0["completed"] == previous["submitted"] + 1
            }
            XCTAssertEqual(drawn["submitted"], previous["submitted"] + 1, "one pose change, one frame")
            previous = drawn
        }

        app.segmentedControls["drawingMode"].buttons["Continuous"].tap()
        let animated = try waitForCounters(in: app) { $0["submitted"] >= previous["submitted"] + 120 }
        record("cube fixture: \(start) -> \(animated)")
        for counters in [previous, animated] {
            for key in ["pipelinesCreated", "depthStatesCreated", "buffersAllocated", "bufferBytesAllocated"] {
                XCTAssertEqual(counters[key], start[key], key)
            }
        }

        // The triangle fixture's pipeline has no depth attachment; the view
        // drops its depth buffer for it and restores it for the cube.
        app.segmentedControls["drawingMode"].buttons["Requested"].tap()
        app.segmentedControls["fixture"].buttons["Triangle"].tap()
        let triangle = try waitForCounters(in: app) { $0.text("fixture") == "triangle" && $0["completed"] >= 1 }
        app.segmentedControls["fixture"].buttons["Cube"].tap()
        let cubeAgain = try waitForCounters(in: app) {
            $0.text("fixture") == "geometry" && $0["completed"] == $0["submitted"] && $0["submitted"] > animated["submitted"]
        }
        XCTAssertEqual(cubeAgain["buffersAllocated"], start["buffersAllocated"])
        XCTAssertEqual(triangle["buffersAllocated"], 2)
    }

    /// FF-08: the material fixture shades the cube's mesh with the built-in
    /// unlit material and with the app's own `lab_pulse_fragment`. Choosing a
    /// color or material draws one frame, continuous drawing feeds the
    /// animation time into the pulse material's parameters, and the pixels on
    /// screen change with it, while the renderer's pipeline, depth-state, and
    /// buffer counters stay at their launch values. The app runs with Metal
    /// API validation.
    @MainActor
    func testMaterialFixtureAnimatesParametersWithoutCreatingPipelines() throws {
        let app = makeApp(["-FrostfieldLabFixture", "material"])
        app.launchEnvironment["MTL_DEBUG_LAYER"] = "1"
        app.launch()
        let start = try waitForSettledCounters(in: app)
        XCTAssertEqual(start.text("fixture"), "material")
        XCTAssertEqual(start.text("material"), "unlit")
        XCTAssertEqual(start.text("materialFunction"), "ff_unlit_fragment")
        XCTAssertEqual(start["meshVertices"], 24)
        // Normal diagnostic, unlit, and pulse: all prepared at launch.
        XCTAssertEqual(start["pipelinesCreated"], 3)
        // Two per-frame uniform buffers and the cube's three mesh buffers;
        // the material fixture shares the cube fixture's mesh.
        XCTAssertEqual(start["buffersAllocated"], 5)
        let amber = try cubePixel(in: app)
        XCTAssertGreaterThan(amber.red, 0.8, "amber: \(amber)")
        XCTAssertLessThan(amber.blue, 0.3, "amber: \(amber)")

        app.segmentedControls["unlitColor"].buttons["Teal"].tap()
        let teal = try waitForCounters(in: app) {
            $0.text("unlitColor") == "teal" && $0["completed"] == start["submitted"] + 1
        }
        XCTAssertEqual(teal["submitted"], start["submitted"] + 1, "one color change, one frame")
        let tealPixel = try cubePixel(in: app)
        XCTAssertLessThan(tealPixel.red, 0.3, "teal: \(tealPixel)")
        XCTAssertGreaterThan(tealPixel.green, 0.5, "teal: \(tealPixel)")

        app.segmentedControls["material"].buttons["Pulse"].tap()
        let pulse = try waitForCounters(in: app) {
            $0.text("material") == "pulse" && $0["completed"] == teal["submitted"] + 1
        }
        XCTAssertEqual(pulse.text("materialFunction"), "lab_pulse_fragment")

        app.segmentedControls["drawingMode"].buttons["Continuous"].tap()
        var samples: [RGB] = []
        for _ in 0..<4 {
            Thread.sleep(forTimeInterval: 0.4)
            samples.append(try cubePixel(in: app))
        }
        let animated = try waitForCounters(in: app) { $0["submitted"] >= pulse["submitted"] + 120 }
        record("material fixture: \(start) -> \(animated); pulse samples \(samples)")
        let times = [pulse, animated].compactMap { $0.text("materialTime").flatMap(Double.init) }
        XCTAssertEqual(times.count, 2)
        XCTAssertGreaterThan(times[1], times[0] + 1, "the app advanced the material's time parameter")
        let distinct = Set(samples.map { "\(Int($0.red * 255))-\(Int($0.green * 255))-\(Int($0.blue * 255))" })
        XCTAssertGreaterThan(distinct.count, 1, "the pulse material's pixels did not change: \(samples)")

        for counters in [teal, pulse, animated] {
            for key in ["pipelinesCreated", "depthStatesCreated", "buffersAllocated", "bufferBytesAllocated"] {
                XCTAssertEqual(counters[key], start[key], key)
            }
        }
    }

    // MARK: Helpers

    /// The lab, launched so that it publishes its counters for these tests.
    @MainActor
    private func makeApp(_ arguments: [String] = []) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-FrostfieldLabUITesting"] + arguments
        return app
    }

    struct RGB: CustomStringConvertible {
        var red: Double, green: Double, blue: Double
        var description: String { String(format: "(%.3f, %.3f, %.3f)", red, green, blue) }
    }

    /// The color of the cube a little below and right of its centre in the
    /// material fixture's corner pose, read from a screenshot of the render
    /// view alone, in the view's own coordinates. The fixture's camera sits
    /// 0.5 below the cube's centre at distance 5 with a 50° field of view, so
    /// the centre projects to 39.3 % of the view's height.
    @MainActor
    private func cubePixel(in app: XCUIApplication) throws -> RGB {
        let renderView = app.otherElements["renderView"]
        let size = renderView.frame.size
        let point = CGPoint(x: size.width / 2 + 20, y: size.height * 0.393 + 30)
        let image = renderView.screenshot().image
        let cgImage = try XCTUnwrap(image.cgImage)
        let pixelX = Int(point.x * image.scale), pixelY = Int(point.y * image.scale)
        var bytes = [UInt8](repeating: 0, count: 4)
        let context = try XCTUnwrap(CGContext(
            data: &bytes, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        // Shift the image so the sampled pixel covers the context's only
        // pixel; Core Graphics counts y from the bottom.
        context.draw(cgImage, in: CGRect(
            x: -pixelX, y: pixelY - cgImage.height + 1, width: cgImage.width, height: cgImage.height
        ))
        let rgb = RGB(red: Double(bytes[0]) / 255, green: Double(bytes[1]) / 255, blue: Double(bytes[2]) / 255)
        // The view's clear color is (0.08, 0.09, 0.11); the sample must be on the cube.
        XCTAssertGreaterThan(max(rgb.red, rgb.green, rgb.blue), 0.2, "sampled the background: \(rgb)")
        return rgb
    }

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
