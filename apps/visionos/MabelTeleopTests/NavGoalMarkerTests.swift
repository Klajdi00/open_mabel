//
//  NavGoalMarkerTests.swift
//  MabelTeleopTests
//
//  The goal marker is placed by a floor TAP, which a simulator run cannot perform —
//  so its geometry and its new cancel path are pinned here instead of by eye.
//
//  Covers the 2026-07-16 operator asks: the marker was "way too big" (it was a 0.8 m
//  ring around a 0.7 m tower that hid the floor it pointed at), and a placed goal had
//  no way to be discarded — the only exit from a mis-tap was to dispatch it and then
//  stop the robot.
//
//  The same constants ship in the iOS app's LiveMapRig (LiveMapScene.swift); keep the two in step.
//

import XCTest
import SceneKit
@testable import MabelTeleop

@MainActor
final class NavGoalMarkerTests: XCTestCase {

    private func placedMarker() -> SCNNode? {
        let scene = LiveMapRig()
        scene.apply(goal: SCNVector3(1, 0, 2), yaw: 0)
        return scene.goalMarker
    }

    private func geometry<T: SCNGeometry>(_ marker: SCNNode, _ name: String, as: T.Type) -> T? {
        marker.childNodes.first { $0.name == name }?.geometry as? T
    }

    func testMarkerIsPlacedAtTheRequestedGroundPoint() {
        let scene = LiveMapRig()
        scene.apply(goal: SCNVector3(1.25, 0, -2.5), yaw: 0)
        let m = scene.goalMarker
        XCTAssertNotNil(m, "a tapped goal must show a marker immediately, before any route")
        XCTAssertEqual(m!.simdPosition.x, 1.25, accuracy: 1e-4)
        XCTAssertEqual(m!.simdPosition.y, 0, accuracy: 1e-4, "the marker stands on the floor")
        XCTAssertEqual(m!.simdPosition.z, -2.5, accuracy: 1e-4)
    }

    /// The size cut the operator asked for. These are exact so a future tweak has to
    /// be deliberate rather than a drift back toward the old tower.
    func testMarkerIsTheSmallerSize() {
        guard let m = placedMarker() else { return XCTFail("no marker") }

        guard let pillar = geometry(m, "goal_pillar", as: SCNCylinder.self) else {
            return XCTFail("no pillar")
        }
        XCTAssertEqual(pillar.height, 0.38, accuracy: 1e-6)
        XCTAssertEqual(pillar.radius, 0.025, accuracy: 1e-6)

        guard let ring = geometry(m, "goal_ring", as: SCNTube.self) else {
            return XCTFail("no ring")
        }
        XCTAssertEqual(ring.outerRadius, 0.22, accuracy: 1e-6)

        guard let arrow = geometry(m, "goal_arrow", as: SCNCone.self) else {
            return XCTFail("no arrow")
        }
        XCTAssertEqual(arrow.height, 0.14, accuracy: 1e-6)
    }

    /// Regression guard on the actual complaint: the marker must stay well under the
    /// old footprint (0.80 m ring, 0.70 m pillar) rather than creep back up.
    func testMarkerIsMuchSmallerThanTheOldTower() {
        guard let m = placedMarker() else { return XCTFail("no marker") }
        let ring = geometry(m, "goal_ring", as: SCNTube.self)!
        let pillar = geometry(m, "goal_pillar", as: SCNCylinder.self)!
        XCTAssertLessThan(ring.outerRadius * 2, 0.55, "ring must be well under the old 0.80 m")
        XCTAssertLessThan(pillar.height, 0.45, "pillar must be well under the old 0.70 m")
    }

    /// …but still a comfortable gaze/drag target — shrinking it must not make the yaw
    /// ring unusable.
    func testYawRingStaysAGrabbableTarget() {
        guard let m = placedMarker() else { return XCTFail("no marker") }
        let ring = geometry(m, "goal_ring", as: SCNTube.self)!
        XCTAssertGreaterThan(ring.outerRadius * 2, 0.30, "ring must stay easy to grab")
        XCTAssertGreaterThan(ring.outerRadius - ring.innerRadius, 0.05, "ring needs real width")
    }

    /// The ring must remain the WIDEST part, so it is what the operator reaches for.
    func testRingIsTheOutermostFeature() {
        guard let m = placedMarker() else { return XCTFail("no marker") }
        let ring = geometry(m, "goal_ring", as: SCNTube.self)!
        let pillar = geometry(m, "goal_pillar", as: SCNCylinder.self)!
        XCTAssertGreaterThan(ring.outerRadius, pillar.radius)
    }

    // MARK: - Cancel (the new ✕ beside GO)

    func testClearGoalRemovesTheMarker() {
        let scene = LiveMapRig()
        scene.apply(goal: SCNVector3(1, 0, 2), yaw: 0)
        XCTAssertNotNil(scene.goalMarker)
        scene.clearGoal()
        XCTAssertNil(scene.goalMarker, "cancelling a placed goal must drop the marker")
    }

    /// Cancel then re-place: the second goal must appear (a stale container or a
    /// non-nil goalMarker would make the next tap look dead).
    func testGoalCanBeRePlacedAfterCancelling() {
        let scene = LiveMapRig()
        scene.apply(goal: SCNVector3(1, 0, 2), yaw: 0)
        scene.clearGoal()
        scene.apply(goal: SCNVector3(-3, 0, 0.5), yaw: 1.2)
        XCTAssertNotNil(scene.goalMarker, "a goal placed after a cancel must still show")
        XCTAssertEqual(scene.goalMarker!.simdPosition.x, -3, accuracy: 1e-4)
    }

    /// Placing twice must not stack markers — the old one is replaced, not buried.
    func testPlacingTwiceLeavesOneMarker() {
        let scene = LiveMapRig()
        scene.apply(goal: SCNVector3(1, 0, 2), yaw: 0)
        scene.apply(goal: SCNVector3(2, 0, 1), yaw: 0)
        let markers = scene.goalContainer.childNodes.filter { $0.name == "goal_marker" }
        XCTAssertEqual(markers.count, 1)
        XCTAssertEqual(scene.goalMarker!.simdPosition.x, 2, accuracy: 1e-4)
    }

    /// The ring/pillar/arrow keep their `goal_` names — the drag recognizer begins
    /// only on a hit whose node name has that prefix, so a rename silently kills
    /// yaw dragging.
    func testMarkerPartsKeepTheirHitTestNames() {
        guard let m = placedMarker() else { return XCTFail("no marker") }
        let names = Set(m.childNodes.compactMap(\.name))
        XCTAssertTrue(names.contains("goal_pillar"))
        XCTAssertTrue(names.contains("goal_ring"))
        XCTAssertTrue(names.contains("goal_arrow"))
        XCTAssertEqual(m.name, "goal_marker")
        for n in names { XCTAssertTrue(n.hasPrefix("goal_"), "\(n) would not hit-test as a goal") }
    }
}
