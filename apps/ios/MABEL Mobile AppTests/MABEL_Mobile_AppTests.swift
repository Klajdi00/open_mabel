//
//  MABEL_Mobile_AppTests.swift
//  MABEL Mobile AppTests
//
//

import Testing
import Foundation
@testable import MABEL_Mobile_App

struct MABEL_Mobile_AppTests {

    @Test func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
        // Swift Testing Documentation
        // https://developer.apple.com/documentation/testing
    }

    // MARK: - ServerDiscovery IPv4 selection

    /// The exact multi-interface address set the Mac advertised after the
    /// networking work (loopback + two self-assigned link-locals + the real
    /// Wi-Fi LAN, in nondeterministic order). Must pick the LAN IP.
    @Test func picksLanIpFromRealAdvertisedSet() {
        let observed = ["127.0.0.1", "169.254.24.161", "192.168.1.34", "169.254.22.181"]
        #expect(ServerDiscovery.bestReachable(observed) == "192.168.1.34")
        // Order must not matter.
        #expect(ServerDiscovery.bestReachable(observed.reversed()) == "192.168.1.34")
        #expect(ServerDiscovery.bestReachable(observed.shuffled()) == "192.168.1.34")
    }

    /// Real LAN beats a Tailscale CGNAT (100.64/10) address.
    @Test func prefersLanOverTailscale() {
        #expect(ServerDiscovery.bestReachable(["100.64.0.5", "192.168.1.50"]) == "192.168.1.50")
        #expect(ServerDiscovery.bestReachable(["192.168.1.50", "100.64.0.5"]) == "192.168.1.50")
    }

    /// A 10.x home LAN still beats Tailscale CGNAT.
    @Test func prefers10NetOverTailscale() {
        #expect(ServerDiscovery.bestReachable(["100.64.0.5", "10.0.0.42"]) == "10.0.0.42")
    }

    /// Loopback and link-local are never usable.
    @Test func rejectsUnreachable() {
        #expect(ServerDiscovery.reachability("127.0.0.1") < 0)
        #expect(ServerDiscovery.reachability("169.254.5.5") < 0)
        #expect(ServerDiscovery.reachability("0.0.0.0") < 0)
        #expect(ServerDiscovery.bestReachable(["127.0.0.1", "169.254.1.1"]) == nil)
        #expect(ServerDiscovery.bestReachable([]) == nil)
    }

    /// If only Tailscale is reachable (e.g. remote, not on the same Wi-Fi),
    /// still return it rather than nothing.
    @Test func fallsBackToTailscaleWhenOnlyOption() {
        #expect(ServerDiscovery.bestReachable(["100.64.0.5"]) == "100.64.0.5")
    }

    // MARK: - NetworkConfig route classification

    /// IPs are tagged by the network the operator sees: Tailscale CGNAT
    /// (100.64/10) vs ordinary LAN.
    @Test func classifiesRouteKind() {
        #expect(NetworkConfig.classify("100.64.0.5") == .tailscale)
        #expect(NetworkConfig.classify("100.127.255.1") == .tailscale)
        #expect(NetworkConfig.classify("192.168.1.42") == .lan)
        #expect(NetworkConfig.classify("10.0.0.5") == .lan)
        #expect(NetworkConfig.classify("172.20.10.2") == .lan)
        // 100.x OUTSIDE the CGNAT band is a public 100-net, not Tailscale.
        #expect(NetworkConfig.classify("100.200.1.1") == .lan)
    }

    /// A direct link derives its MJPEG video base from the host.
    @Test func directLinkVideoBase() throws {
        let lan = WebSocketLink(host: "192.168.1.42")
        #expect(lan.videoBase?.absoluteString == "http://192.168.1.42:8080")
    }
}
