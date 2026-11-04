//
//  WholeBodyStopTests.swift
//  MabelTeleopTests
//
//  STOP MUST STOP, AND HOME MUST WIN.
//
//  Two operator reports, 2026-07-30, with one root cause between them: the app
//  had no single "end the mirroring" operation. `bodyTracking` gated the HANDS,
//  and everything treated clearing it as ending the session — but the outbound
//  frame source streams a LIVE HEAD POSE whenever the immersive space is open,
//  independent of that flag. So:
//
//    · Stop Teleop in full immersion left the robot's neck following the
//      operator's head. From inside the headset that is indistinguishable from
//      "teleop is still running", because you cannot see which limb the wire
//      thinks it is driving.
//    · Home mid-teleop wrote the rest pose into joint targets while the
//      retargeter re-asserted the operator's arms on the very next frame. The
//      robot fought itself and never arrived.
//
//  These pin the invariant rather than either symptom: after any explicit stop,
//  NOTHING the operator's body does reaches the robot — and engaging again
//  restores it, because a fix that permanently welded head tracking to hand
//  tracking would quietly kill active perception (looking around without the
//  arms), which is a real mode.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class WholeBodyStopTests: XCTestCase {

    private func engaged() -> RobotController {
        let c = RobotController()
        c.immersiveActive = true
        c.engageWholeBody()
        XCTAssertTrue(c.bodyTracking, "precondition: whole body engaged")
        XCTAssertTrue(c.headEnabled, "precondition: head mirroring live")
        return c
    }

    func testStopEndsHandsANDHead() {
        let c = engaged()
        c.stopWholeBodyTeleop()
        XCTAssertFalse(c.bodyTracking, "hands still streaming after Stop")
        XCTAssertFalse(c.headEnabled,
                       "the neck still follows the operator's head after Stop — "
                       + "this is the bug that read as 'teleop never stopped'")
    }

    func testHomeStopsTheMirroringFirst() {
        // Otherwise the rest-pose glide and the retargeter write the same joints
        // every frame and the robot never arrives.
        let c = engaged()
        c.goRestPose()
        XCTAssertFalse(c.bodyTracking, "Home left the retargeter driving the arms")
        XCTAssertFalse(c.headEnabled, "Home left the head driving the neck")
    }

    func testEngagingAgainRestoresHeadMirroring() {
        // The other half of the fix, and the one a careless version breaks: if
        // Stop merely froze the head and nothing put it back, the SECOND engage
        // would drive the arms with a neck that never moved again.
        let c = engaged()
        c.stopWholeBodyTeleop()
        c.engageWholeBody()
        XCTAssertTrue(c.bodyTracking)
        XCTAssertTrue(c.headEnabled, "head mirroring did not come back with the session")
    }

    func testHeadCanStillBeToggledIndependently() {
        // Active perception — the neck follows the head with the arms idle — is a
        // real mode. The fix must not weld the two flags together permanently.
        let c = engaged()
        c.toggleHeadTracking()
        XCTAssertFalse(c.headEnabled)
        XCTAssertTrue(c.bodyTracking, "toggling the head must not stop the hands")
    }
}
