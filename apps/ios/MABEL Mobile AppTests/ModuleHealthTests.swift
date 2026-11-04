//
//  ModuleHealthTests.swift
//  MABEL Mobile AppTests
//
//  THE STATUS PANEL, OVER BLUETOOTH.
//
//  Connected over BLE the panel showed every module on "WAIT / no data yet" — including a
//  lift that was plugged in and streaming — because the packed frame carried pose and
//  safety flags and nothing else. It now carries a HEALTH frame once a second
//  (server/ble_protocol.py encode_health): counts, flags, and ONE BIT PER MOTOR.
//
//  What these tests hold down is the part that is easy to get wrong and impossible to see:
//  that the bit which comes back set belongs to the motor that is actually alive. An
//  off-by-one in the bitmap would report a healthy arm with the WRONG joint flagged, and
//  send someone to the wrong connector.
//

import XCTest
import CoreBluetooth
@testable import MABEL_Mobile_App

final class ModuleHealthTests: XCTestCase {

    /// Build a HEALTH frame the way the robot does, so the test exercises the real layout
    /// rather than a Swift-shaped idea of it.
    private func frame(_ mods: [(id: String, live: [Bool], flags: UInt8)]) -> Data {
        var b: [UInt8] = [0x82, UInt8(mods.count)]
        for m in mods {
            let idx = UInt8(BLEProtocol.moduleIds.firstIndex(of: m.id)!)
            let n = m.live.count
            var bm = [UInt8](repeating: 0, count: (n + 7) / 8)
            for (k, on) in m.live.enumerated() where on { bm[k / 8] |= 1 << (k % 8) }
            b += [idx, UInt8(n), m.flags] + bm
        }
        return Data(b)
    }

    // 0x01 has-driver, 0x02 torque, 0x04 ok
    func testAHealthyArmReadsSevenOfSeven() {
        let rows = BLEProtocol.decodeHealth(frame([("rightArm", Array(repeating: true, count: 7), 0x05)]))
        XCTAssertEqual(rows?.count, 1)
        let arm = rows![0]
        XCTAssertEqual(arm.id, "rightArm")
        XCTAssertEqual(arm.label, "Right Arm")
        XCTAssertEqual(arm.up, 7)
        XCTAssertEqual(arm.n, 7)
        XCTAssertEqual(arm.countText, "7/7")
        XCTAssertTrue(arm.ok)
        XCTAssertEqual(arm.detail, "7/7")
        XCTAssertEqual(arm.actuators.map(\.label), ["J1", "J2", "J3", "J4", "J5", "J6", "J7"])
    }

    /// THE ONE THAT MATTERS: two motors silent, and the panel has to name THOSE two.
    func testTheSilentMotorsAreTheOnesNamed() {
        let live = [true, true, false, true, true, false, true]      // J3 and J6 are dead
        let rows = BLEProtocol.decodeHealth(frame([("rightArm", live, 0x01)]))!
        let arm = rows[0]
        XCTAssertEqual(arm.up, 5)
        XCTAssertEqual(arm.n, 7)
        XCTAssertFalse(arm.ok)
        XCTAssertEqual(arm.detail, "J3, J6")
        XCTAssertEqual(arm.actuators.filter { !$0.ok }.map(\.label), ["J3", "J6"])
        XCTAssertEqual(arm.actuators[2].detail, "no reply")
        XCTAssertTrue(arm.actuators[0].ok, "J1 is alive and must not be flagged")
    }

    /// A 17-motor hand crosses the bitmap's byte boundaries twice — the place an
    /// off-by-one would hide.
    func testASeventeenMotorHandSurvivesTheByteBoundaries() {
        var live = Array(repeating: true, count: 17)
        live[7] = false; live[8] = false; live[16] = false          // either side of byte 1, and the last
        let hand = BLEProtocol.decodeHealth(frame([("leftHand", live, 0x01)]))![0]
        XCTAssertEqual(hand.n, 17)
        XCTAssertEqual(hand.up, 14)
        XCTAssertEqual(hand.actuators.filter { !$0.ok }.map(\.label), ["M8", "M9", "M17"])
    }

    /// A module with NO DRIVER is not a broken one. n == 0 has to stay distinguishable, or
    /// the panel paints a limb that was never fitted bright red.
    func testAModuleWithNoDriverIsNotReportedAsBroken() {
        let rows = BLEProtocol.decodeHealth(frame([("leftHand", [], 0x00)]))!
        XCTAssertEqual(rows[0].n, 0)
        XCTAssertFalse(rows[0].hasDriver)
        XCTAssertEqual(rows[0].detail, "no driver")
        XCTAssertEqual(rows[0].countText, "", "no motors means no count to show")
    }

    /// Bus open, nothing answering: different cause, different words.
    func testAnEntireBusDownSaysSo() {
        let rows = BLEProtocol.decodeHealth(frame([("leftArm", Array(repeating: false, count: 7), 0x01)]))!
        XCTAssertEqual(rows[0].up, 0)
        XCTAssertEqual(rows[0].detail, "link down")
    }

    func testTheSwerveBaseIsLabelledByCorner() {
        let base = BLEProtocol.decodeHealth(frame([("base", Array(repeating: true, count: 8), 0x05)]))![0]
        XCTAssertEqual(base.actuators.map(\.label),
                       ["FL Steer", "FL Drive", "FR Steer", "FR Drive",
                        "RL Steer", "RL Drive", "RR Steer", "RR Drive"])
    }

    func testTheColumnAndTheCameraAreNamedNotNumbered() {
        let rows = BLEProtocol.decodeHealth(frame([("lift", [true], 0x05), ("camHead", [true], 0x05)]))!
        XCTAssertEqual(rows[0].actuators.map(\.label), ["Column"])
        XCTAssertEqual(rows[1].actuators.map(\.label), ["Stereo"])
        XCTAssertEqual(rows[1].group, "sensors", "the camera belongs under SENSORS, not with the limbs")
    }

    /// A short read on a radio link is normal. Half a roster would report live motors as
    /// dead, so a truncated frame must be dropped whole.
    func testATruncatedFrameIsDroppedRatherThanHalfBelieved() {
        var d = frame([("rightArm", Array(repeating: true, count: 7), 0x05)])
        d.removeLast()
        XCTAssertNil(BLEProtocol.decodeHealth(d))
        XCTAssertNil(BLEProtocol.decodeHealth(Data([0x82])))
        XCTAssertNil(BLEProtocol.decodeHealth(Data()))
    }

    /// The pose frame and the health frame share a characteristic and are told apart by
    /// their first byte. Neither decoder may accept the other's frame.
    func testTheTwoFrameTypesDoNotDecodeEachOther() throws {
        let health = frame([("lift", [true], 0x05)])
        XCTAssertNil(BLEProtocol.decodeState(health))
        let pose = try XCTUnwrap(BLEProtocol.encodeJointCommand(["lift_lower": 0.1]))  // 0x01
        XCTAssertNil(BLEProtocol.decodeHealth(pose))
    }

    /// A server that reports NO motor counts at all (an older bridge) must keep its rows
    /// readable — silence is not the same claim as "no motors", and painting those rows
    /// grey would report a healthy robot as unfitted.
    func testARosterWithoutCountsStillReadsAsFittedAndHealthy() {
        let legacy = RobotState.Module(id: "leftArm", label: "Left Arm", group: nil,
                                       ok: true, detail: "7 joints")
        XCTAssertTrue(legacy.hasDriver, "no count reported is not a claim of no motors")
        XCTAssertEqual(legacy.countText, "", "nothing to count means no count is shown")
        let unfitted = RobotState.Module(id: "leftHand", label: "Left Hand", group: nil,
                                         ok: false, detail: "no driver", up: 0, n: 0)
        XCTAssertFalse(unfitted.hasDriver)
        XCTAssertEqual(unfitted.countText, "")
    }

    /// Bluetooth carries no battery gauge, no thermal and no round-trip. The link must
    /// declare that, or the panel renders RobotState's defaults as if they were readings.
    func testTheLinkDeclaresTheVitalsItCannotCarry() {
        let l = BLELink(id: UUID(), name: "NOVA",
                        central: CBCentralManager(delegate: nil, queue: .main))
        XCTAssertFalse(l.state.vitalsAvailable)
        XCTAssertFalse(l.state.batteryAvailable)
    }
}
