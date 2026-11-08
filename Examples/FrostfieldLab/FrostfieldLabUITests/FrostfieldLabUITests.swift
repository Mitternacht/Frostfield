import XCTest

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

        let status = app.staticTexts["frameStatus"]
        XCTAssertTrue(status.waitForExistence(timeout: 5))
        let completedOne = NSPredicate { _, _ in
            let counters = FrameCounters(status.value as? String)
            return (counters["submitted"] ?? 0) >= 1 && (counters["completed"] ?? 0) >= 1
        }
        let expectation = XCTNSPredicateExpectation(predicate: completedOne, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [expectation], timeout: 10), .completed, "status: \(String(describing: status.value))")
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
}

/// Parses the status label's `key=value;key=value` accessibility value.
struct FrameCounters {
    private var values: [String: String] = [:]

    init(_ text: String?) {
        for pair in (text ?? "").split(separator: ";") {
            let parts = pair.split(separator: "=", maxSplits: 1)
            if parts.count == 2 {
                values[String(parts[0])] = String(parts[1])
            }
        }
    }

    subscript(key: String) -> Int? { values[key].flatMap { Int($0) } }

    func text(_ key: String) -> String? { values[key] }
}
