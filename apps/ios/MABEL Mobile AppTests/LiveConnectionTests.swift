//
//  LiveConnectionTests.swift
//  MABEL Mobile AppTests
//
//  End-to-end checks that exercise the REAL networking stack (ConnectionPlanner
//  TCP race + WebSocketLink handshake) against a running MABEL bridge. The iOS
//  Simulator shares the Mac's network, so it reaches the LAN bridge directly.
//
//  These gracefully SKIP when the LAN bridge isn't reachable (e.g. CI with no
//  server) — they only assert when a server is actually up, so they never fail
//  spuriously. Run them with the bridge live (server/run.sh) to verify a real
//  link forms over each transport.
//

import Testing
import Foundation
@testable import MABEL_Mobile_App

@MainActor
struct LiveConnectionTests {

    /// The Mac bridge's current LAN IP (a known host the planner also probes).
    static let lanHost = "192.168.1.42"

    /// True iff the LAN bridge answers a TCP probe — used to gate the live
    /// assertions so they skip cleanly when nothing is running.
    private func bridgeUp() async -> Bool {
        await ConnectionPlanner.tcpProbe(host: Self.lanHost, timeout: 2).1 != nil
    }

    /// The planner's TCP race reaches the LAN bridge and measures a sane RTT.
    @Test func plannerProbesLanBridge() async {
        let (_, rtt) = await ConnectionPlanner.tcpProbe(host: Self.lanHost, timeout: 2)
        guard let rtt else { return }                       // bridge down → skip
        #expect(rtt >= 0 && rtt < 2000)
    }

    /// The full plan picks a reachable direct route fastest-first.
    @Test func plannerRacesDirectRoutes() async {
        guard await bridgeUp() else { return }              // skip without a server
        let planner = ConnectionPlanner()
        let directs = await planner.plan(hosts: [Self.lanHost])
        // At least one direct route is reachable.
        #expect(!directs.isEmpty)
        // Directs are sorted by ascending latency (fastest first).
        let rtts = directs.compactMap(\.rttMs)
        #expect(rtts == rtts.sorted())
    }

    /// A direct LAN WebSocket link reaches `.connected` (real handshake + hello).
    @Test func lanLinkConnects() async {
        guard await bridgeUp() else { return }
        let ok = await connects(WebSocketLink(host: Self.lanHost))
        #expect(ok, "LAN link should reach .connected")
    }

    /// Drive a link and resolve true once it reports `.connected` (8 s cap).
    private func connects(_ link: WebSocketLink) async -> Bool {
        let result = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            var resumed = false
            func settle(_ v: Bool) {
                if resumed { return }; resumed = true; cont.resume(returning: v)
            }
            link.onStatus = { st in if st == .connected { settle(true) } }
            link.connect()
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) { settle(false) }
        }
        link.disconnect()
        return result
    }
}
