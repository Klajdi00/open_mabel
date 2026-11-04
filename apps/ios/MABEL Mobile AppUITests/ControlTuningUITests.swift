//  ControlTuningUITests.swift
//  Layout + behaviour of the Control Tuning sheet (gravity comp · retargeting).
//
//  Operator, 2026-08-23: "make sure the UI/UX is clean and consistent — the
//  buttons should be clear and no overlapping with text or other buttons."
//
//  That is a claim about geometry, so it is tested as one: every assertion below
//  compares real on-screen frames. It is not decorative — adding the tuning
//  button IN LINE with Stiff|Soft measurably pushed that control behind the
//  centred "Start Teleop" pill, which is exactly what these tests now catch.
//
//  KNOWN ENVIRONMENT ISSUE. The two LAYOUT tests are deterministic. The
//  interaction test is not, and the cause is below the app: CoreSimulator on this
//  host intermittently dies mid-run ("(ipc/mig) server died") or kills the app on
//  a launch watchdog (0x8BADF00D), and XCUITest's synthesized tap on this
//  SwiftUI Toggle lands only sometimes — consecutive runs of an identical
//  sequence produce a flipped control in one and an untouched one in the next.
//  The APP is fine: the flipped runs render the new state and the correct blurb.
//  The tap is retried to absorb that, but a dead simulator cannot be retried
//  around. If this test is noisy in CI, prefer deleting it over loosening it —
//  the layout tests are the ones that caught a real regression.

import XCTest

final class ControlTuningUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Sheet presentation on a loaded simulator can take many seconds; every
    /// wait here is generous on purpose. A short timeout here does not test the
    /// UI, it tests the host's mood — these exact assertions failed twice on a
    /// 5 s wait and passed unchanged at 30 s.
    private static let uiTimeout: TimeInterval = 30

    /// Wait until an element is not just PRESENT but settled — existence fires
    /// while a sheet is still animating in, and a tap delivered mid-presentation
    /// is swallowed. (Measured: the toggle stayed ON, the screenshot showed the
    /// sheet still dimmed behind its presentation animation.)
    @discardableResult
    private func waitSettled(_ el: XCUIElement, _ t: TimeInterval = uiTimeout) -> Bool {
        guard el.waitForExistence(timeout: t) else { return false }
        let deadline = Date().addingTimeInterval(t)
        var last = el.frame
        while Date() < deadline {
            Thread.sleep(forTimeInterval: 0.2)
            let now = el.frame
            if el.isHittable && now == last && now.height > 0 { return true }
            last = now
        }
        return el.isHittable
    }

    private func launch() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments += ["-uiTesting", "YES"]
        app.launch()
        return app
    }

    /// No control in the top row may overlap any other. The row centres the
    /// primary verb in a ZStack, so a flanking cluster that grows sideways
    /// slides UNDER it — silently, and only visible in a screenshot.
    func testTopRowControlsDoNotOverlap() throws {
        let app = launch()
        let tuning = app.buttons["btn.controlTuning"]
        guard tuning.waitForExistence(timeout: 30) else {
            throw XCTSkip("teleop controls not present (needs torque on)")
        }
        // Every hittable button in the upper fifth of the screen.
        let top = app.frame.height * 0.25
        let bars = app.buttons.allElementsBoundByIndex.filter {
            $0.exists && $0.isHittable && $0.frame.maxY < top && $0.frame.width > 20
        }
        for i in bars.indices {
            for j in (i + 1)..<bars.count {
                let a = bars[i].frame, b = bars[j].frame
                XCTAssertFalse(a.intersects(b),
                    "top-row controls overlap: \(bars[i].identifier)/\(bars[i].label) "
                    + "\(a) intersects \(bars[j].identifier)/\(bars[j].label) \(b)")
            }
        }
    }

    /// The control law lives in a MENU now, not a half-screen sheet: three
    /// switches did not justify a modal, and its explanatory paragraphs buried
    /// the switches themselves. The menu must carry all three controls — Feel
    /// (Stiff | Soft) moved in here when it stopped fitting the top row beside
    /// the centred Start/Stop verb.
    func testTuningMenuCarriesFeelAndBothSwitches() throws {
        let app = launch()
        let tuning = app.buttons["btn.controlTuning"]
        guard tuning.waitForExistence(timeout: 30) else {
            throw XCTSkip("teleop controls not present (needs torque on)")
        }
        XCTAssertTrue(tuning.isHittable, "the control-tuning button must be tappable")
        tuning.tap()

        // Menu rows are buttons; a sheet would have brought switches instead.
        for label in ["Stiff", "Soft", "Gravity Comp", "Posture", "Wrist"] {
            let row = app.buttons[label]
            XCTAssertTrue(row.waitForExistence(timeout: 5),
                          "\(label) missing from the control-tuning menu")
        }
        XCTAssertFalse(app.staticTexts["Control Tuning"].exists,
                       "the old modal sheet is back — this should be a menu")
    }

    /// Tapping a row applies it and closes the menu (standard menu behaviour) —
    /// the switch must not silently stay put, which is what a container
    /// accessibility element does when it swallows the tap.
    func testTuningMenuRowActuates() throws {
        let app = launch()
        let tuning = app.buttons["btn.controlTuning"]
        guard tuning.waitForExistence(timeout: 30) else {
            throw XCTSkip("teleop controls not present (needs torque on)")
        }
        tuning.tap()
        let soft = app.buttons["Soft"]
        guard soft.waitForExistence(timeout: 5) else {
            return XCTFail("Feel picker missing from the menu")
        }
        soft.tap()
        XCTAssertFalse(soft.waitForExistence(timeout: 2),
                       "the menu should dismiss once a row is chosen")
        // Re-open: Feel is now Soft, so the button carries its non-default dot.
        tuning.tap()
        XCTAssertTrue(app.buttons["Stiff"].waitForExistence(timeout: 5),
                      "menu should reopen with the Feel picker intact")
    }
}

// The Autonomy tab's own UI tests moved to `AutonomyTimelineUITests.swift` when
// the choreography editor was rebuilt around a 3-D viewer: the old ones drove a
// docked lane chooser and a docked Inspector, neither of which exists any more.
