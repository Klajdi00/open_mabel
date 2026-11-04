//
//  HandStalenessTests.swift
//  MabelTeleopTests
//
//  A hand that leaves the field of view must EXPIRE, not freeze.
//
//  ARKit delivers one chirality per anchor update, so the broadcast loop carries
//  `left`/`right` across iterations to emit both hands every time — correct, and
//  load-bearing. But those locals were only ever ASSIGNED. ARKit may simply stop
//  emitting updates for a hand that leaves the field of view rather than sending
//  one with `isTracked == false`, so that hand kept streaming its last good pose
//  — `isTracked: true`, full skeleton — at the full frame rate, forever.
//
//  The server only drops a hand whose `isTracked` is false (mabel/sources/wire.py
//  `_hand_world`), so it kept retargeting that arm to a FROZEN target while the
//  other hand moved normally. That is the operator's 2026-07-31 report — "either
//  the left or the right hand is working, both cannot work together" — and the
//  stuck arm is whichever hand went out of view first, with no way back.
//
//  The fix is per-side refresh stamps, expired at BOTH ends: in the producer
//  (so a mixed one-hand-gone case clears immediately) and at READ time on the
//  wire tick (because when BOTH hands leave, no anchor update arrives at all and
//  the producer never runs again).
//
//  The simulator has no hand tracking, so these drive the expiry rule directly.
//

import XCTest
@testable import MabelTeleop

final class HandStalenessTests: XCTestCase {

    private func pose(_ c: Chirality) -> HandPose {
        HandPose(chirality: c, anchorTransform: Transform4x4(matrix: []),
                 joints: [], isTracked: true)
    }

    private func snap(l: TimeInterval?, r: TimeInterval?,
                      now: TimeInterval = 100) -> HandTrackingService.Snapshot {
        HandTrackingService.Snapshot(
            left: l != nil ? pose(.left) : nil,
            right: r != nil ? pose(.right) : nil,
            timestamp: now,
            leftStamp: l ?? 0,
            rightStamp: r ?? 0)
    }

    // MARK: - The expiry rule

    func testBothFreshSurvive() {
        let s = snap(l: 100, r: 100)
        let (l, r) = s.fresh(after: 100 - HandTrackingService.sideStaleTimeout)
        XCTAssertNotNil(l)
        XCTAssertNotNil(r)
    }

    func testStaleSideIsDroppedWhileTheOtherSurvives() {
        // The exact reported shape: one hand left the view 1 s ago and has not
        // been refreshed since; the other is live.
        let s = snap(l: 99.0, r: 100.0)
        let (l, r) = s.fresh(after: 100 - HandTrackingService.sideStaleTimeout)
        XCTAssertNil(l, "a hand ARKit stopped refreshing must not keep streaming")
        XCTAssertNotNil(r, "the live hand must be unaffected — that is the bug")
    }

    func testMirrorImageIsSymmetric() {
        let s = snap(l: 100.0, r: 99.0)
        let (l, r) = s.fresh(after: 100 - HandTrackingService.sideStaleTimeout)
        XCTAssertNotNil(l)
        XCTAssertNil(r, "the rule must not favour a side")
    }

    func testBothStaleAreBothDropped() {
        // No anchor update arrives at all in this case, which is why the wire
        // tick — not the producer — has to be the one applying the cutoff.
        let s = snap(l: 98.0, r: 98.5)
        let (l, r) = s.fresh(after: 100 - HandTrackingService.sideStaleTimeout)
        XCTAssertNil(l)
        XCTAssertNil(r)
    }

    func testNeverSeenSideIsNotResurrected() {
        // stamp 0 = never seen; it must never pass the cutoff.
        let s = snap(l: nil, r: 100.0)
        let (l, _) = s.fresh(after: 100 - HandTrackingService.sideStaleTimeout)
        XCTAssertNil(l)
    }

    func testTimeoutIsWiderThanArkitJitterButWellInsideReactionTime() {
        // ARKit refreshes a tracked hand at ~90 Hz (11 ms). The window must
        // swallow ordinary jitter without letting a genuinely gone hand drive
        // the robot for a noticeable time.
        let t = HandTrackingService.sideStaleTimeout
        XCTAssertGreaterThan(t, 0.05, "would expire on ordinary frame jitter")
        XCTAssertLessThan(t, 0.5, "a frozen arm would persist too long")
    }

    func testAFreshlyRefreshedSideRecovers() {
        // Bringing the hand back into view must re-arm it — the failure mode
        // was that it never came back.
        let gone = snap(l: 99.0, r: 100.0)
        XCTAssertNil(gone.fresh(after: 100 - HandTrackingService.sideStaleTimeout).0)
        let back = snap(l: 100.5, r: 100.5, now: 100.5)
        XCTAssertNotNil(back.fresh(after: 100.5 - HandTrackingService.sideStaleTimeout).0)
    }

    // MARK: - An UNTRACKED sample is the ABSENCE of a pose, not a pose
    //
    // The 2026-08-01 report ("super super laggy — only one hand can be tracked
    // at a time"). Everything above tests the expiry rule with poses that are
    // `isTracked: true`, which is precisely the case that was already right.
    // ARKit reports a momentarily-lost hand as an anchor with `isTracked ==
    // false` and an EMPTY skeleton, and folding those in did two things at
    // once: it overwrote the last good skeleton (the server drops an untracked
    // hand outright) AND refreshed the freshness stamp, so neither expiry could
    // fire and the side sat "fresh" while carrying nothing.

    private func untracked(_ c: Chirality) -> HandPose {
        HandPose(chirality: c, anchorTransform: Transform4x4(matrix: []),
                 joints: [], isTracked: false)
    }

    func testUntrackedSampleDoesNotOverwriteTheLastGoodPose() {
        var m = HandTrackingService.Merger()
        m.apply(pose(.left), at: 100)
        m.apply(untracked(.left), at: 100.05)      // a one-frame blip
        m.expire(at: 100.05)
        XCTAssertNotNil(m.left, "a blink must hold the last good hand, not erase it")
        XCTAssertTrue(m.left?.isTracked ?? false,
                      "and it must still be the TRACKED pose — the server drops untracked")
    }

    func testUntrackedSampleDoesNotRefreshTheStamp() {
        // The compounding half: if an untracked sample stamps the side, it is
        // "fresh" forever and can never expire NOR fall back to anything.
        var m = HandTrackingService.Merger()
        m.apply(pose(.left), at: 100)
        for i in 1...30 { m.apply(untracked(.left), at: 100 + Double(i) * 0.02) }
        m.expire(at: 100.6)
        XCTAssertNil(m.left, "a hand untracked for 0.6 s must expire, not sit fresh")
    }

    func testOneHandBlinkingLeavesTheOtherAlone() {
        // The reported shape, exactly: both hands up, one flickers.
        var m = HandTrackingService.Merger()
        m.apply(pose(.left), at: 100)
        m.apply(pose(.right), at: 100)
        for i in 1...4 {
            let t = 100 + Double(i) * 0.03
            m.apply(untracked(.right), at: t)      // right blinks
            m.apply(pose(.left), at: t)            // left stays solid
            m.expire(at: t)
        }
        XCTAssertNotNil(m.left)
        XCTAssertNotNil(m.right, "a 0.12 s blink is inside the hold window — both hands live")
    }

    func testASustainedLossStillDropsThatSideOnly() {
        var m = HandTrackingService.Merger()
        m.apply(pose(.left), at: 100)
        m.apply(pose(.right), at: 100)
        for i in 1...20 {                          // 0.6 s of right-hand loss
            let t = 100 + Double(i) * 0.03
            m.apply(untracked(.right), at: t)
            m.apply(pose(.left), at: t)
            m.expire(at: t)
        }
        XCTAssertNotNil(m.left, "the live hand is untouched")
        XCTAssertNil(m.right, "a hand genuinely gone must still be dropped")
    }

    func testRecoveryAfterASustainedLoss() {
        var m = HandTrackingService.Merger()
        m.apply(pose(.right), at: 100)
        m.expire(at: 100.6)
        XCTAssertNil(m.right)
        m.apply(pose(.right), at: 100.7)           // back in view
        m.expire(at: 100.7)
        XCTAssertNotNil(m.right, "re-acquisition must re-arm the side")
    }

    func testNeverSeenSideStaysNil() {
        var m = HandTrackingService.Merger()
        m.apply(pose(.left), at: 100)
        m.expire(at: 100)
        XCTAssertNotNil(m.left)
        XCTAssertNil(m.right, "stamp 0 must never pass the cutoff")
    }

    func testSnapshotCarriesThePerSideStamps() {
        // The wire tick expires at READ time off these stamps (both hands gone
        // = no anchor update at all = this loop never runs again).
        var m = HandTrackingService.Merger()
        m.apply(pose(.left), at: 100)
        m.apply(pose(.right), at: 100.1)
        let s = m.snapshot(at: 100.1)
        XCTAssertEqual(s.leftStamp, 100, accuracy: 1e-9)
        XCTAssertEqual(s.rightStamp, 100.1, accuracy: 1e-9)
        let (l, r) = s.fresh(after: 100.1 - HandTrackingService.sideStaleTimeout)
        XCTAssertNotNil(l)
        XCTAssertNotNil(r)
    }
}
