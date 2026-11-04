//
//  TransportChipUITests.swift
//  MABEL Mobile AppUITests
//
//  The connection chip in the global status bar. Wi-Fi and Bluetooth are
//  genuinely different machines to drive — BLE carries the 20 Hz joint stream
//  but no cameras, no map, no recording — so WHICH one is live is status, not a
//  detail in Settings. These assert the two things that make it usable: it sits
//  between "which robot" and "how is it doing", and it opens both ways in.
//

import XCTest

final class TransportChipUITests: XCTestCase {
    private static let uiTimeout: TimeInterval = 25

    override func setUp() { continueAfterFailure = false }

    private func launch(nearby: String? = nil) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments += ["-uiTesting", "YES", "-startTab", "robot"]
        if let nearby { app.launchArguments += ["-debugNearbyRobot", nearby] }
        app.launch()
        return app
    }

    func testTheTransportChipSitsBetweenTheRobotAndTheVitals() {
        let app = launch()
        let robot = app.buttons["status.robot"]
        let transport = app.buttons["status.transport"]
        let vitals = app.buttons["status.vitals"]
        for e in [robot, transport, vitals] {
            XCTAssertTrue(e.waitForExistence(timeout: Self.uiTimeout),
                          "\(e.identifier) missing from the status bar")
        }
        XCTAssertLessThanOrEqual(robot.frame.maxX, transport.frame.minX + 1,
            "the transport chip must follow the robot switcher")
        XCTAssertLessThanOrEqual(transport.frame.maxX, vitals.frame.minX + 1,
            "the transport chip must precede the vitals summary")
        // Same row, not stacked under it.
        XCTAssertEqual(transport.frame.midY, vitals.frame.midY, accuracy: 6)
    }

    func testTheChipNamesTheTransportAndOpensBothWaysIn() {
        let app = launch(nearby: "MABEL-01")
        let transport = app.buttons["status.transport"]
        XCTAssertTrue(transport.waitForExistence(timeout: Self.uiTimeout))
        // The label says which transport, in words, not just a glyph.
        XCTAssertTrue(transport.label.contains("Connection:"),
                      "the chip should announce the transport, got \(transport.label)")

        // Planting a robot on the air also raises the PROXIMITY CARD, which
        // covers the bar. Dismiss it first: otherwise the tap below lands on the
        // card and the test fails only when it happens to have appeared in time,
        // which is the worst kind of flake.
        let notNow = app.buttons["Not Now"]
        if notNow.waitForExistence(timeout: 6) { notNow.tap() }

        transport.tap()
        // Both routes are offered from one list — switching is a tap, not a trip
        // through Settings.
        XCTAssertTrue(app.staticTexts["NETWORK"].waitForExistence(timeout: Self.uiTimeout),
                      "the picker should offer the network route")
        XCTAssertTrue(app.staticTexts["BLUETOOTH"].exists,
                      "the picker should offer Bluetooth")
        // The robot itself, in the picker (the proximity card is gone by now, so
        // this cannot be matching that).
        XCTAssertTrue(app.staticTexts["MABEL-01"].exists,
                      "a robot on the air should be one tap away")
    }
}
