//
//  RetargetScopeTests.swift
//  MabelTeleopTests
//
//  LOCAL | GLOBAL retargeting scope — is the mobile BASE in the follow loop?
//
//  Re-added 2026-07-31 on operator request. This is NOT the retired `frame`
//  axis: that one was about which MEASUREMENTS the estimator was allowed to
//  believe. `scope` is about which DoFs are in the optimisation —
//
//    LOCAL  (default) arms + hands + torso + lift + neck follow the operator;
//                     the base stays where the joystick left it. Deliberately
//                     the easier problem: the arm IK solves against a root that
//                     cannot move under it.
//    GLOBAL           the base joins in, past its translation and yaw boxes.
//
//  The safety-shaped half of this is the DEFAULT. A base that moves on its own
//  is the consequential state, so it must be asked for and never inherited —
//  the server applies the same rule to an absent or unrecognised value, so an
//  older client build cannot be handed base authority it never requested.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class RetargetScopeTests: XCTestCase {

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

        /// The `scope` of every control spec put on the wire, in order.
        var scopes: [String?] {
            sent.compactMap { (c: RobotCommand) -> String?? in
                if case let .controlSpec(_, _, _, _, _, scope) = c { return .some(scope) }
                return nil
            }
        }
        var twists: [(Double, Double, Double)] {
            sent.compactMap { (c: RobotCommand) -> (Double, Double, Double)? in
                if case let .twist(f, s, y) = c { return (f, s, y) }
                return nil
            }
        }
    }

    private func make() -> (RobotController, SpyLink) {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.torqueOnForTest(link)
        return (c, link)
    }

    // MARK: - The default

    func testDefaultScopeIsLocal() {
        let (c, _) = make()
        XCTAssertEqual(c.retargetScope, .local,
                       "base authority must be asked for, never the default")
    }

    func testEveryControlSpecCarriesTheScope() {
        let (c, link) = make()
        c.setControl(method: .retarget, type: .impedance, region: .wholeBody)
        XCTAssertEqual(link.scopes.last ?? nil, "local",
                       "the server must never have to guess the scope")
    }

    // MARK: - Switching

    func testSwitchingToGlobalIsSentOnTheWire() {
        let (c, link) = make()
        c.retargetScope = .global
        XCTAssertEqual(link.scopes.last ?? nil, "global")
    }

    func testLeavingGlobalParksTheBase() {
        // The engine only zeroes the base when something overwrites it, so a
        // scope change that means "stop following" has to say so — otherwise
        // the base coasts on the last follow command it was given.
        let (c, link) = make()
        c.retargetScope = .global
        link.sent.removeAll()
        c.retargetScope = .local
        XCTAssertTrue(link.twists.contains { $0 == (0, 0, 0) },
                      "leaving GLOBAL must park the base")
    }

    func testRedundantSetIsNotResent() {
        let (c, link) = make()
        c.setControl(method: .retarget, type: .impedance, region: .wholeBody)
        let n = link.scopes.count
        c.retargetScope = .local          // already local
        XCTAssertEqual(link.scopes.count, n, "a no-op switch must not spam the wire")
    }

    // MARK: - The wire value itself

    func testScopeRawValuesMatchTheServerContract() {
        XCTAssertEqual(RetargetScope.local.rawValue, "local")
        XCTAssertEqual(RetargetScope.global.rawValue, "global")
        XCTAssertEqual(RetargetScope.allCases.count, 2)
    }
}
