//
//  StagePlannerTests.swift
//  MabelTeleopTests
//
//  Locks the RealityKit-stage green-ball planner (StagePlanner) — the port
//  that keeps Task space on the TRUE-3D stage (operator 2026-07-18). The
//  planner is tick-driven precisely so these tests can step preview/execute
//  deterministically with synthetic timestamps, and the rig is the real
//  Resources/MabelRig bundle, so the IK runs against the real kinematics.
//

import XCTest
import RealityKit
import simd
@testable import MabelTeleop

@MainActor
final class StagePlannerTests: XCTestCase {

    private func makePlanner() async -> (StagePlanner, RobotRigEntity) {
        let rig = RobotRigEntity()
        await rig.load(interactive: true)
        XCTAssertFalse(rig.joints.isEmpty, "rig.json + PLYs must load in the test host")
        let planner = StagePlanner(rig: rig)
        planner.joints = { RobotModel.homePose }
        return (planner, rig)
    }

    /// setActive builds the palm balls + orientation rings under the wrist
    /// mounts and toggles them with the mode.
    func testHandlesInstallAndToggle() async {
        let (planner, rig) = await makePlanner()
        planner.setActive(true)
        for side in ["left", "right"] {
            let mount = rig.joints["\(side)_arm_7"]?.node
            let ball = mount?.children.first { $0.name == "handle_\(side)" }
            XCTAssertNotNil(ball, "\(side) palm ball installed")
            XCTAssertEqual(ball?.isEnabled, true)
            for axis in ["roll", "pitch", "yaw"] {
                XCTAssertNotNil(ball?.children.first { $0.name == "ring_\(axis)_\(side)" },
                                "\(side) \(axis) ring installed")
            }
        }
        planner.setActive(false)
        let ball = rig.joints["left_arm_7"]?.node.children.first { $0.name == "handle_left" }
        XCTAssertEqual(ball?.isEnabled, false, "handles hide when the planner disarms")
    }

    /// Staging freezes the arm, emits the preview summary, and the ghost
    /// preview actually moves the planned joint on the rig.
    func testStagePreviewAndCancel() async {
        let (planner, rig) = await makePlanner()
        planner.setActive(true)
        var previews: [ArmPlanPreview?] = []
        planner.onPlanChange = { previews.append($0) }

        planner.stagePlanForTest(side: "left", deltas: ["left_arm_2": 0.3])
        XCTAssertTrue(planner.hasPlan)
        guard let p = previews.last ?? nil else { return XCTFail("no preview emitted") }
        XCTAssertEqual(p.side, "left")
        XCTAssertEqual(p.deltaRad, 0.3, accuracy: 0.05)
        XCTAssertTrue(rig.frozenJoints.contains("left_arm_2"),
                      "telemetry must not fight the ghost preview")

        // Mid-ramp preview tick (phase 0.7 of the 1.4 s ramp) → the joint node
        // sits between start and target.
        guard let j = rig.joints["left_arm_2"] else { return XCTFail("no left_arm_2") }
        let start = Float(RobotModel.homePose["left_arm_2"] ?? 0)
        planner.tick(now: 100.0)          // seeds the preview clock
        planner.tick(now: 100.7)
        let mid = j.node.orientation.angle * (simd_dot(j.node.orientation.axis, simd_normalize(j.axis)) < 0 ? -1 : 1)
        XCTAssertNotEqual(mid, start, accuracy: 0.02, "preview must move the ghosted joint")

        planner.cancelPlan()
        XCTAssertFalse(planner.hasPlan)
        XCTAssertTrue(rig.frozenJoints.isEmpty, "cancel releases telemetry")
        XCTAssertNil(previews.last ?? nil, "cancel announces the cleared plan")
    }

    /// A micro-drag (below the 0.02 rad floor) never stages — and never leaves
    /// joints frozen behind.
    func testMicroDragDoesNotStage() async {
        let (planner, rig) = await makePlanner()
        planner.setActive(true)
        var fired = false
        planner.onPlanChange = { _ in fired = true }
        planner.stagePlanForTest(side: "left", deltas: ["left_arm_2": 0.005])
        XCTAssertFalse(planner.hasPlan)
        XCTAssertFalse(fired, "no pill for a micro-drag")
        XCTAssertTrue(rig.frozenJoints.isEmpty, "micro-drag must unfreeze")
    }

    /// Execute streams the SAME eased trajectory the ghost previewed — the
    /// only joint send in the whole flow — and finishes clean.
    func testExecuteStreamsTrajectory() async {
        let (planner, rig) = await makePlanner()
        _ = rig     // keep the rig alive for the planner's lifetime
        planner.setActive(true)
        var sends: [[String: Double]] = []
        var lastPreview: ArmPlanPreview? = ArmPlanPreview(side: "x", deltaRad: 0, collision: false, reason: "")
        planner.onJointsChange = { sends.append($0) }
        planner.onPlanChange = { lastPreview = $0 }

        planner.stagePlanForTest(side: "left", deltas: ["left_arm_2": 0.3])
        planner.executePlan()
        // Playback is sized to the MOTION (ArmPlan.duration: a 0.3 rad plan runs
        // ~minDuration 0.4 s), so mid-trajectory ticks must land INSIDE that —
        // the old 200.7/201.5 stamps assumed the fixed 1.4 s playback and the
        // run was already over by the second tick.
        planner.tick(now: 200.0)          // seeds the execute clock (t = 0)
        planner.tick(now: 200.1)          // mid-trajectory
        planner.tick(now: 200.2)          // mid-trajectory
        planner.tick(now: 210.0)          // t ≥ 1 → final config + cleanup

        XCTAssertGreaterThanOrEqual(sends.count, 4, "execute streams every tick")
        let start = RobotModel.homePose["left_arm_2"] ?? 0
        let target = ArmPlan.clamp("left_arm_2", start + 0.3)
        XCTAssertEqual(sends.last?["left_arm_2"] ?? .nan, target, accuracy: 1e-6,
                       "the streamed trajectory must land exactly on the staged target")
        XCTAssertFalse(planner.hasPlan, "plan resolves after the run")
        XCTAssertNil(lastPreview, "the pill clears after the run")
    }

    /// Real-geometry IK smoke test: pinch the left palm ball, drag it, and the
    /// release stages a plan (side + freeze + preview all wired).
    func testBallDragStagesViaIK() async {
        let (planner, rig) = await makePlanner()
        planner.setActive(true)
        var preview: ArmPlanPreview? = nil
        planner.onPlanChange = { preview = $0 }

        guard let ball = rig.joints["left_arm_7"]?.node.children.first(where: { $0.name == "handle_left" })
        else { return XCTFail("no left handle") }
        XCTAssertTrue(planner.claims(ball), "the ball's drag belongs to the planner")

        let ballWorld = ball.position(relativeTo: nil)
        planner.beginDrag(entity: ball, scenePoint: ballWorld)
        planner.updateDrag(scenePoint: ballWorld + SIMD3<Float>(0, -0.15, 0.25))
        planner.endDrag()

        XCTAssertTrue(planner.hasPlan, "a real drag stages a plan")
        XCTAssertEqual(preview?.side, "left")
        XCTAssertGreaterThan(preview?.deltaRad ?? 0, 0.02, "the IK must have moved the arm")
        XCTAssertTrue(rig.frozenJoints.contains("left_arm_1"))
    }
}
