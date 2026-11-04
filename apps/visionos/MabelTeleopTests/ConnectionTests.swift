//
//  ConnectionTests.swift
//  MabelTeleopTests
//
//  Verifies the network connection stack — Wi-Fi LAN and Tailscale — that the
//  headset shares with the iOS app and web console.
//
//  Pure-logic checks (route classification, link wiring) always run.
//  The LIVE checks exercise the real ConnectionPlanner TCP race + WebSocketLink
//  handshake against a running bridge and MEASURE per-method latency;
//  they skip cleanly when no bridge is reachable, so they never fail spuriously.
//

import XCTest
@testable import MabelTeleop

final class ConnectionTests: XCTestCase {

    /// The Mac bridge's current LAN IP (a known host the planner also probes).
    static let lanHost = "192.168.1.188"

    // MARK: - Pure logic

    func test_classify_routeKind() {
        XCTAssertEqual(NetworkConfig.classify("100.64.0.5"), .tailscale)
        XCTAssertEqual(NetworkConfig.classify("100.127.0.1"), .tailscale)
        XCTAssertEqual(NetworkConfig.classify("192.168.1.188"), .lan)
        XCTAssertEqual(NetworkConfig.classify("10.0.0.5"), .lan)
        XCTAssertEqual(NetworkConfig.classify("172.20.10.2"), .lan)
        XCTAssertEqual(NetworkConfig.classify("100.200.1.1"), .lan)   // outside CGNAT band
    }

    func test_directLink_videoBase() throws {
        let lan = WebSocketLink(host: Self.lanHost)
        XCTAssertEqual(lan.videoBase?.absoluteString, "http://\(Self.lanHost):8080")
    }

    // MARK: - Live: each transport connects + latency

    private func bridgeUp() async -> Bool {
        await ConnectionPlanner.tcpProbe(host: Self.lanHost, timeout: 2).1 != nil
    }

    func test_live_planner_racesFastest() async throws {
        guard await bridgeUp() else { throw XCTSkip("No LAN bridge — skipping live test") }
        let planner = await ConnectionPlanner()
        let routes = await planner.plan(hosts: [Self.lanHost])
        XCTAssertFalse(routes.isEmpty, "a direct route should be reachable")
        if let direct = routes.first {
            print(String(format: "  [race] fastest direct = %@ %@ (%.0f ms TCP)",
                         direct.kind.rawValue, direct.host, direct.rttMs ?? -1))
        }
    }

    func test_live_lan_connects_andLatency() async throws {
        guard await bridgeUp() else { throw XCTSkip("No LAN bridge") }
        await measureLink(WebSocketLink(host: Self.lanHost), label: "Wi-Fi LAN")
    }

    func test_live_tailscale_connects_andLatency() async throws {
        // Only meaningful if Tailscale is up; skip if that host isn't reachable.
        guard let ts = NetworkConfig.tailscaleHosts.first,
              await ConnectionPlanner.tcpProbe(host: ts, timeout: 2).1 != nil else {
            throw XCTSkip("Tailscale host unreachable")
        }
        await measureLink(WebSocketLink(host: ts), label: "Tailscale")
    }

    /// Drive a real link to `.connected` + first telemetry, asserting success and
    /// printing the handshake / round-trip / bridge-reported latency.
    private func measureLink(_ link: WebSocketLink, label: String) async {
        let start = Date()
        let result: (connectedMs: Double?, firstStateMs: Double?, bridgeMs: Double?) =
        await withCheckedContinuation { (cont: CheckedContinuation<(Double?, Double?, Double?), Never>) in
            let lock = NSLock()
            var connectedMs: Double?, firstStateMs: Double?, bridgeMs: Double?
            var resumed = false
            func maybeFinish() {
                // Resolve once we've seen the first telemetry (full round-trip).
                if firstStateMs != nil, !resumed { resumed = true; cont.resume(returning: (connectedMs, firstStateMs, bridgeMs)) }
            }
            link.onStatus = { st in
                if st == .connected {
                    lock.lock(); if connectedMs == nil { connectedMs = Date().timeIntervalSince(start) * 1000 }; lock.unlock()
                }
            }
            link.onState = { s in
                lock.lock()
                if firstStateMs == nil { firstStateMs = Date().timeIntervalSince(start) * 1000; bridgeMs = s.latencyMs }
                lock.unlock()
                maybeFinish()
            }
            link.connect()
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
                if !resumed { resumed = true; cont.resume(returning: (connectedMs, firstStateMs, bridgeMs)) }
            }
        }
        link.disconnect()
        XCTAssertNotNil(result.connectedMs, "\(label) should reach .connected")
        XCTAssertNotNil(result.firstStateMs, "\(label) should stream telemetry")
        print(String(format: "  [%@]  handshake %.0f ms · first state %.0f ms · bridge %.0f ms",
                     label, result.connectedMs ?? -1, result.firstStateMs ?? -1, result.bridgeMs ?? -1))
    }
}
