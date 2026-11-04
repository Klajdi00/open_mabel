//
//  DriveControlsTests.swift
//  The split Drive cockpit (2026-10-16): TRANSLATE stick · LIFT slider ·
//  ROTATE stick. These pin the SIGN conventions of the 1-D controls — a lift
//  slider that lowers when pushed up, or a rotate stick that leaks the other
//  axis, is exactly the kind of bug that looks fine in a screenshot.
//  Same text in both apps bar the import line.
//

import XCTest
@testable import MABEL_Mobile_App

final class DriveControlsTests: XCTestCase {

    private let travel: CGFloat = 50

    /// LIFT: pushing UP (screen y decreasing) is +, which `setLiftRate` sends
    /// as nav.ry > 0 → the engine's drone lift RAISES the column.
    func test_vertical_upIsPositive() {
        let up = AxisStick.deflection(translation: CGSize(width: 0, height: -25), travel: travel, axis: .vertical)
        let down = AxisStick.deflection(translation: CGSize(width: 0, height: 25), travel: travel, axis: .vertical)
        XCTAssertEqual(up, 0.5, accuracy: 1e-9)
        XCTAssertEqual(down, -0.5, accuracy: 1e-9)
    }

    /// ROTATE: right is +, matching the old right stick's dx → `yaw`.
    func test_horizontal_rightIsPositive() {
        let r = AxisStick.deflection(translation: CGSize(width: 40, height: 0), travel: travel, axis: .horizontal)
        XCTAssertEqual(r, 0.8, accuracy: 1e-9)
    }

    /// The whole point of the split: the cross axis NEVER leaks. A sloppy
    /// diagonal drag on the rotate stick commands no lift, and vice versa.
    func test_crossAxisIsIgnored() {
        let diag = CGSize(width: 30, height: -45)
        XCTAssertEqual(AxisStick.deflection(translation: diag, travel: travel, axis: .horizontal), 0.6, accuracy: 1e-9)
        XCTAssertEqual(AxisStick.deflection(translation: diag, travel: travel, axis: .vertical), 0.9, accuracy: 1e-9)
        XCTAssertEqual(AxisStick.deflection(translation: CGSize(width: 0, height: 80), travel: travel, axis: .horizontal), 0)
        XCTAssertEqual(AxisStick.deflection(translation: CGSize(width: 80, height: 0), travel: travel, axis: .vertical), 0)
    }

    /// Clamped to the track: an overshooting drag is full scale, never more.
    func test_clampedToUnit() {
        for t in [CGSize(width: 900, height: 0), CGSize(width: -900, height: 0)] {
            XCTAssertEqual(abs(AxisStick.deflection(translation: t, travel: travel, axis: .horizontal)), 1)
        }
        XCTAssertEqual(AxisStick.deflection(translation: CGSize(width: 0, height: -900), travel: travel, axis: .vertical), 1)
        XCTAssertEqual(AxisStick.deflection(translation: CGSize(width: 0, height: 900), travel: travel, axis: .vertical), -1)
    }

    /// A degenerate (zero-length) track commands nothing rather than ±inf/NaN.
    func test_zeroTravel_isZero() {
        XCTAssertEqual(AxisStick.deflection(translation: CGSize(width: 10, height: 10), travel: 0, axis: .vertical), 0)
    }
}
