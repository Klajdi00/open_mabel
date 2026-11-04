//
//  LiftUnitsOverBluetoothTests.swift
//  MabelTeleopTests
//
//  THE LIFT STICK OVER BLUETOOTH — now a RATE, like every other path.
//
//  History, because the contract in this file has moved twice and the reasons are worth
//  keeping. The packed BLE format carried positions only, so the stick was integrated HERE
//  into a metres-per-stage target. First it integrated the wrong units — a 0…1 fraction
//  written into a metres field, starting at 0.5, which asked for a metre of column against
//  0.635 m of travel and ran the lift to its ceiling. That was fixed by integrating in
//  metres, from the measured height, at the column's own speed.
//
//  What no amount of integrating could fix is the servo. The column is BANG-BANG: full duty
//  or nothing. Give it a slowly-advancing position target and it drives full duty,
//  overshoots its 1.5 mm stop band, cuts, and repeats — which an operator feels as "when
//  the joystick is not pushed all the way it jitters instead of going slowly". Measured in
//  NOVA's lift simulation: at 0.10 stick the old integrator stuttered 14 times in
//  a 3 s hold and delivered 3.3 mm/s; sending the stick as a rate and letting the robot map
//  it to open-loop duty delivers 7.6 mm/s with zero stutters.
//
//  So the stick now rides the teleop frame as `lift`, the robot turns it into duty
//  (server/plant.py jog_effort), and this suite pins THAT: the app must send a stick
//  position and must not send a column height.
//

import XCTest
import CoreBluetooth
@testable import MabelTeleop

@MainActor
final class LiftUnitsOverBluetoothTests: XCTestCase {

    private static let stageMax = 0.3175          // half of the 0.635 m column travel

    private func link() -> BLELink {
        BLELink(id: UUID(), name: "NOVA",
                central: CBCentralManager(delegate: nil, queue: .main))
    }

    private func telemetry(_ l: BLELink, perStage: Double) {
        l.applyStateFrame(positions: ["lift_lower": perStage, "lift_upper": perStage],
                          estop: false, torque: true)
    }

    /// THE HEADLINE: a stick touch must not put a column HEIGHT on the wire. A height is
    /// the one thing this servo cannot be asked for slowly.
    func testTheStickSendsARateAndNeverAHeight() {
        let l = link()
        telemetry(l, perStage: 0.10)
        l.send(.liftRate(0.25))                     // a gentle push

        XCTAssertEqual(l.liftStick, 0.25, accuracy: 1e-12)
        XCTAssertNil(l.joints["lift_lower"],
            "a stick must not command a stage height — that is what made it stutter")
        XCTAssertNil(l.joints["lift_upper"])
    }

    /// Small stick, small number: the app passes the deflection through untouched and the
    /// ROBOT decides what duty that is. One speed map, on the robot, for every transport.
    func testTheDeflectionIsPassedThroughUnscaled() {
        let l = link()
        telemetry(l, perStage: 0.10)
        for r in [0.05, 0.25, 0.5, 0.75, 1.0, -0.33] {
            l.send(.liftRate(r))
            XCTAssertEqual(l.liftStick, r, accuracy: 1e-12,
                           "the app must not reinterpret the stick — \(r)")
        }
    }

    /// …and it is CLAMPED, so a bad caller cannot ask for more than full deflection.
    func testAnOutOfRangeStickIsClamped() {
        let l = link()
        telemetry(l, perStage: 0.10)
        l.send(.liftRate(4.0));  XCTAssertEqual(l.liftStick, 1.0, accuracy: 1e-12)
        l.send(.liftRate(-9.0)); XCTAssertEqual(l.liftStick, -1.0, accuracy: 1e-12)
    }

    /// Letting go must SAY zero, not merely go quiet. The robot's 0.5 s deadman would stop
    /// the column either way, but half a second of unasked-for travel is 18 mm.
    func testAReleasedStickSaysZero() {
        let l = link()
        telemetry(l, perStage: 0.10)
        l.send(.liftRate(1.0))
        XCTAssertEqual(l.liftStick, 1.0, accuracy: 1e-12)
        l.send(.liftRate(0.0))
        XCTAssertEqual(l.liftStick, 0.0, accuracy: 1e-12, "release must be commanded, not implied")
    }

    /// E-stop zeroes the stick, and says so, rather than leaving a held deflection latched
    /// for whenever the link resumes.
    func testEstopZeroesTheStick() {
        let l = link()
        telemetry(l, perStage: 0.10)
        l.send(.liftRate(1.0))
        l.send(.estop(true))
        XCTAssertEqual(l.liftStick, 0.0, accuracy: 1e-12)
        l.send(.liftRate(1.0))
        XCTAssertEqual(l.liftStick, 0.0, accuracy: 1e-12, "and it stays refused while estopped")
    }

    /// The SLIDER is a different control and keeps its own units: it names a height, which
    /// is exactly right for an absolute move — the firmware profiles it and arrives.
    func testTheSliderStillSendsAHeightInStageMetres() {
        let l = link()
        telemetry(l, perStage: 0.10)
        l.send(.liftFraction(0.5))
        let lower = l.joints["lift_lower"] ?? -1
        XCTAssertEqual(lower, Self.stageMax * 0.5, accuracy: 1e-9,
                       "the slider's fraction scales onto the stage travel")
        XCTAssertLessThanOrEqual(lower, Self.stageMax,
                                 "a stage target may never exceed half the travel")
        XCTAssertEqual(l.joints["lift_upper"] ?? -1, lower, accuracy: 1e-12, "stages move together")
    }

    /// The frame that carries it must actually carry it — 16 bytes, lift in the 4th slot,
    /// and the robot still accepts the 12-byte frame an older build sends.
    func testTheTeleopFrameCarriesTheLift() {
        let f = BLEProtocol.encodeTeleopFrame(vx: 0, vy: 0, wz: 0, lift: -0.4)
        XCTAssertEqual(f.count, 17, "type byte + four floats")
        let bytes = [UInt8](f)
        var bits: UInt32 = 0
        for (i, b) in bytes[13...16].enumerated() { bits |= UInt32(b) << (8 * UInt32(i)) }
        let lift = Double(Float(bitPattern: bits))
        XCTAssertEqual(lift, -0.4, accuracy: 1e-6)
        let plain = BLEProtocol.encodeTeleopFrame(vx: 1, vy: 0, wz: 0)
        XCTAssertEqual(plain.count, 17, "the default keeps the frame one shape")
    }
}
