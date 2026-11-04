//
//  DeckLayoutTests.swift
//  MabelTeleopTests
//
//  Full immersion: the status bar sits at the TOP of the head view and the control
//  deck at the BOTTOM (operator, 2026-10-17), and BOTH must be inside the headset's
//  field of view (operator, 2026-10-23: "make sure the bottom bar can be viewed … at
//  one point it was so low that I cannot even see — when I move the head down the bar
//  moves away … do not have that again"). The old rule put them OUTSIDE a true-scale
//  window, 36°+ off-axis, at/over the edge of a ~±45° view, and the deck followed head
//  pitch so it could never be looked at. The pitch half is pinned in DeckFollowerTests.
//

import XCTest
@testable import MabelTeleop

final class DeckLayoutTests: XCTestCase {

    private func window(scale: Float) -> StereoWindowGeometry {
        StereoWindowGeometry(fovyDeg: 83.3, aspect: 4.0 / 3.0, distance: StereoWindowGeometry.defaultDistance,
                             ipd: StereoWindowGeometry.defaultIPD, scale: scale)
    }

    private func half(_ h: Float) -> Float { atan(h / 2 / DeckLayout.distance) }

    private var scales: [Float] {
        [StereoWindowGeometry.scaleRange.lowerBound, StereoWindowGeometry.defaultScale,
         StereoWindowGeometry.scaleRange.upperBound]
    }

    func test_deckBelowAndStatusAboveEyeLevel() {
        for s in scales {
            XCTAssertGreaterThan(DeckLayout.deckCentreAngle(height: 0.3), 0, "deck below eye level")
            XCTAssertLessThan(DeckLayout.statusCentreAngle(height: 0.05, window: window(scale: s)), 0,
                              "status above eye level")
        }
    }

    /// Looking straight ahead, the deck's top edge and the whole status bar are in view.
    func test_bothBarsInsideTheFieldOfViewLookingAhead() {
        XCTAssertLessThan(DeckLayout.deckTop, DeckLayout.viewHalfV - 5 * .pi / 180,
                          "the deck's top must show in the lower view, not at its edge")
        for s in scales {
            for h: Float in [0.03, 0.06, 0.1] {
                let c = DeckLayout.statusCentreAngle(height: h, window: window(scale: s))
                XCTAssertLessThanOrEqual(-c + half(h), DeckLayout.viewHalfV, "status top in view (scale \(s))")
                XCTAssertLessThanOrEqual(-c + half(h), DeckLayout.statusTopMax + 1e-5)
            }
        }
    }

    /// The whole deck fits in view once the operator looks down at it — even a tall one.
    func test_wholeDeckReachableByLookingDown() {
        for h: Float in [0.2, 0.35, 0.5] {
            let c = DeckLayout.deckCentreAngle(height: h)
            XCTAssertLessThan(half(h), DeckLayout.viewHalfV, "deck fits the view when centred")
            XCTAssertLessThan(c, 60 * .pi / 180, "centring it needs a comfortable nod")
        }
    }

    /// The deck's placement does NOT depend on the window: a true-scale window must not push it down.
    func test_deckIndependentOfWindowSize() {
        XCTAssertEqual(DeckLayout.deckCentreAngle(height: 0.3) - half(0.3), DeckLayout.deckTop, accuracy: 1e-5)
    }

    /// The status bar sits INSIDE the window's top edge — framed with the head view.
    func test_statusInsideTheWindowTop() {
        for s in scales {
            let g = window(scale: s)
            let top = -DeckLayout.statusCentreAngle(height: 0.05, window: g) + half(0.05)
            XCTAssertLessThan(top, g.halfHeightAngle, "scale \(s)")
        }
    }

    /// Operator, 2026-10-24: "the top bar is way too high and the bottom control panel
    /// is too high too — bring the control panel way lower". Neither may creep back.
    func test_statusLowAndDeckLow() {
        XCTAssertLessThanOrEqual(DeckLayout.statusTopMax, 20 * .pi / 180, "status bar a glance up, not a look up")
        XCTAssertGreaterThanOrEqual(DeckLayout.deckTop, 32 * .pi / 180, "deck well down in the view")
    }
}
