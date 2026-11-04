//
//  SoftPassiveTests.swift
//  MabelTeleopTests
//
//  SOFT (compliance) is a PASSIVE mode: the robot floats under gravity compensation
//  and the OPERATOR moves it with their hands. The app must send NO command data —
//  a joint/base/grip target in Soft is the app fighting the person holding the arm.
//
//  This is a safety contract, not a preference, so it is pinned here rather than
//  left to whichever views happen to hide their controls. Before this the Soft
//  toggle was independent of `space`, so Joint space kept its sliders live and they
//  still streamed joint_command while the robot was meant to be limp.
//
//  The server assumes the same thing: sim_teleop_bridge.py holds the compliance spec
//  sticky *because* a Soft session "sends no app input → no driver token".
//

import XCTest
@testable import MabelTeleop

/// Records everything the app puts on the wire.
private final class SpyLink: RobotLink {
    var status: ConnectionStatus = .connected
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((String, String?, String) -> Void)?
    var onMapList: (([MapInfo], String?) -> Void)?
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
    var sent: [RobotCommand] = []

    func connect() {}
    func disconnect() {}
    func send(_ c: RobotCommand) { sent.append(c) }

    /// Commands that MOVE the robot (as opposed to declaring intent / stopping).
    var motionCommands: [RobotCommand] {
        sent.filter {
            switch $0 {
            case .jointTarget, .navGoal, .liftRate: return true
            case let .twist(f, s, y): return f != 0 || s != 0 || y != 0   // a zero twist is a STOP
            default: return false
            }
        }
    }
}

@MainActor
final class SoftPassiveTests: XCTestCase {

    private func softController() -> (RobotController, SpyLink) {
        let link = SpyLink()
        let c = RobotController(link: link)
        // The app BOOTS E-STOPPED (2026-07-18); these tests probe the SOFT
        // gate specifically, so run them torque-on like the operator would.
        c.torqueOnForTest(link)
        c.setControl(method: .joint, type: .compliance, region: .arm)
        link.sent.removeAll()          // drop the control-spec declaration itself
        return (c, link)
    }

    private func stiffController() -> (RobotController, SpyLink) {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.torqueOnForTest(link)
        c.setControl(method: .joint, type: .impedance, region: .arm)
        link.sent.removeAll()
        return (c, link)
    }

    // MARK: - Soft sends nothing that moves the robot

    func testSoftIsPassive() {
        let (c, _) = softController()
        XCTAssertTrue(c.passiveCompliance)
    }

    func testSoftDropsJointCommands() {
        let (c, link) = softController()
        c.setJoint("right_arm_1", 0.5)
        XCTAssertTrue(link.motionCommands.isEmpty,
                      "a joint target in Soft fights the human holding the arm")
    }

    func testSoftDropsBatchedJointCommands() {
        let (c, link) = softController()
        c.setJoints(["right_arm_1": 0.4, "right_arm_2": 0.2])
        XCTAssertTrue(link.motionCommands.isEmpty)
    }

    func testSoftDropsGrip() {
        let (c, link) = softController()
        c.setGrip("right", 1.0)
        XCTAssertTrue(link.motionCommands.isEmpty)
    }

    func testSoftDropsBaseDriving() {
        let (c, link) = softController()
        c.drive(translate: CGVector(dx: 1, dy: 1), yaw: 1)
        c.driveTwist(forward: 0.5, strafe: 0, yaw: 0.2)
        XCTAssertTrue(link.motionCommands.isEmpty)
    }

    func testSoftDropsNavGoals() {
        let (c, link) = softController()
        c.sendNavGoal(x: 2, y: 1, yaw: 0)
        XCTAssertTrue(link.motionCommands.isEmpty)
    }

    func testSoftDropsLiftJogging() {
        let (c, link) = softController()
        c.setLiftRate(0.8)
        XCTAssertTrue(link.motionCommands.isEmpty)
    }

    /// A dropped joint command must not desync the on-screen model either: if the
    /// target moved locally the slider would jump and snap back.
    func testSoftLeavesJointTargetsUntouched() {
        let (c, _) = softController()
        let before = c.jointTargets["right_arm_1"]
        c.setJoint("right_arm_1", 1.23)
        XCTAssertEqual(c.jointTargets["right_arm_1"], before)
    }

    // MARK: - …but safety and intent still get through

    /// Zeroing the base is how the robot is made SAFE — entering Soft calls it.
    func testSoftStillAllowsStop() {
        let (c, link) = softController()
        c.stopBase()
        XCTAssertFalse(link.sent.isEmpty, "a stop must never be suppressed")
    }

    /// Parking the lift (rate 0) is a stop, not a command.
    func testSoftStillAllowsLiftStop() {
        let (c, link) = softController()
        c.setLiftRate(0)
        XCTAssertFalse(link.sent.isEmpty)
    }

    /// In the sim the external force IS the operator's hand — the stand-in for
    /// physically grabbing the real robot. Suppressing it would make Soft inert.
    func testSoftStillAllowsTheOperatorsHand() {
        let (c, link) = softController()
        c.applyExternalForce(["l_hand_mount": [1, 0, 0]])
        XCTAssertFalse(link.sent.isEmpty, "the drag is how Soft is driven in sim")
    }

    /// Declaring the mode must always reach the robot, or it never goes limp.
    func testSoftSpecItselfIsSent() {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.setControl(method: .joint, type: .compliance, region: .arm)
        XCTAssertFalse(link.sent.isEmpty)
        XCTAssertTrue(c.passiveCompliance)
    }

    // MARK: - Stiff is unaffected

    func testStiffStillCommands() {
        let (c, link) = stiffController()
        XCTAssertFalse(c.passiveCompliance)
        c.setJoint("right_arm_1", 0.5)
        XCTAssertFalse(link.motionCommands.isEmpty, "Stiff must still drive the robot")
    }

    /// Leaving Soft must restore commanding — the gate is the live mode, not a latch.
    func testLeavingSoftRestoresCommanding() {
        let (c, link) = softController()
        c.setJoint("right_arm_1", 0.5)
        XCTAssertTrue(link.motionCommands.isEmpty)

        c.setControl(method: .joint, type: .impedance, region: .arm)
        link.sent.removeAll()
        c.setJoint("right_arm_1", 0.5)
        XCTAssertFalse(link.motionCommands.isEmpty, "Stiff after Soft must command again")
    }

    /// Navigation (the Drive home spec) is impedance, so it must never be passive.
    func testNavigationIsNotPassive() {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.setNavigation()
        XCTAssertFalse(c.passiveCompliance)
    }
}
