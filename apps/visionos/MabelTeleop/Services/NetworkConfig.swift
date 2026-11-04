//
//  NetworkConfig.swift
//  MabelTeleop
//
//  The connection map shared with the iOS app and the web console
//  (web_gui/control_center): ways to reach the same bridge, tried fastest first.
//
//    · Wi-Fi LAN  — the Mac's LAN IP, lowest latency on the same network. Found
//                   automatically by Bonjour (ServerDiscovery); these KNOWN_HOSTS
//                   are a fallback for networks that block mDNS (client isolation).
//    · Tailscale  — the Mac's 100.x CGNAT IP, reachable from ANY network when
//                   Tailscale is up on both ends. Network-independent.
//    · Bluetooth  — BLE GATT straight to the robot when no network answers.
//

import Foundation

/// Which transport a live link is using — drives the Settings network readout.
enum RouteKind: String, Equatable {
    case lan       = "Wi-Fi LAN"
    case tailscale = "Tailscale"
    case demo      = "On-device"
    /// No network at all: BLE GATT straight to the robot (see BLELink).
    case bluetooth = "Bluetooth"

    var icon: String {
        switch self {
        case .lan:       return "wifi"
        case .tailscale: return "lock.shield"
        case .demo:      return "cpu"
        case .bluetooth: return "dot.radiowaves.left.and.right"
        }
    }
    /// Short tag for the status pill / chips.
    var tag: String {
        switch self {
        case .lan:       return "WI-FI"
        case .tailscale: return "TAILSCALE"
        case .demo:      return "DEMO"
        case .bluetooth: return "BLUETOOTH"
        }
    }

    /// Map the server's authoritative transport label (from `_classify_network`:
    /// "LAN" / "Tailscale" / "Localhost") to our kind. nil for values we
    /// don't display as a route (e.g. "Internet" / "Unknown") so the caller can
    /// fall back to the client-side guess.
    init?(serverTransport: String) {
        switch serverTransport.lowercased() {
        case "lan":               self = .lan
        case "tailscale":         self = .tailscale
        case "localhost", "demo": self = .demo
        case "bluetooth":         self = .bluetooth
        default:                  return nil
        }
    }
}

/// One way to reach the bridge, with its measured probe latency.
struct ConnectionRoute: Equatable, Identifiable {
    let id = UUID()
    let kind: RouteKind
    let host: String            // numeric IP or hostname
    var rttMs: Double? = nil    // TCP-connect latency from the race

    static func == (a: ConnectionRoute, b: ConnectionRoute) -> Bool {
        a.kind == b.kind && a.host == b.host
    }
}

enum NetworkConfig {
    /// Bridge ports (sim_teleop_bridge.py defaults).
    static let wsPort   = 9090
    static let httpPort = 8080
    static let mapPort  = 9092      // dedicated live-SLAM map stream (/map)

    /// The live-SLAM map WebSocket for a direct (LAN / Tailscale) host.
    static func directMapURL(host: String, accessCode: String? = nil) -> URL? {
        guard var c = URLComponents(string: "ws://\(host):\(mapPort)/map") else { return nil }
        if let code = accessCode, !code.isEmpty {
            c.queryItems = [URLQueryItem(name: "key", value: code)]
        }
        return c.url
    }

    /// The duplex voice plane's port (`server/docs/data_streams.md`).
    static let audioPort = 9610

    /// Tailscale CGNAT addresses — work on ANY network when Tailscale is up.
    static let tailscaleHosts: [String] = [
        "100.64.0.5",
    ]

    /// Known LAN IPs to probe in parallel when Bonjour is blocked (client
    /// isolation). All private ranges, never routable from the public internet.
    static let knownLANHosts: [String] = [
        "192.168.1.42",                               // example: robot computer on the LAN
        "172.20.10.2", "172.20.10.3", "172.20.10.4",  // phone hotspot range
    ]

    /// Classify a host into the route kind we show the operator.
    /// Tailscale uses the 100.64.0.0/10 CGNAT block (everything else that is a
    /// dotted-quad / .local is LAN).
    static func classify(_ host: String) -> RouteKind {
        let h = host.trimmingCharacters(in: .whitespaces).lowercased()
        let o = h.split(separator: ".").compactMap { Int($0) }
        if o.count == 4, o[0] == 100, (64...127).contains(o[1]) { return .tailscale }
        return .lan
    }

    /// Addresses of the second robot profile, one per network it joins.
    static let novaHosts = [
        "10.42.0.1",                                        // its own Wi-Fi hotspot (no internet needed)
        "192.168.1.43",                                     // example: LAN
        "mabel-orin.local",                                 // any shared LAN, by mDNS
        "100.64.0.3",                                       // example: Tailscale
    ]

    /// Named robots offered before discovery — the simulation and the physical
    /// rigs, each with its LAN + Tailscale routes. The store seeds these
    /// once; the server-owned name then overrides on connect. 1:1 with the iOS app.
    static func seedRobots() -> [Robot] {
        [
            Robot(id: "mabel-sim", name: "Simulation", kind: .sim,
                  routes: knownLANHosts + tailscaleHosts, saved: true),
            Robot(id: "thor-swerve", name: "MABEL Real (thor)", kind: .real,
                  routes: ["192.168.1.42", "100.64.0.2"], ssh: "thor", saved: true),
            // NOVA on the Orin: every network it is known to sit on, raced at once (the first
            // to take a TCP connect on :9090 wins), then Bluetooth (BLECentral, beacon "NOVA")
            // when nothing on the network answers.
            Robot(id: "nova-orin", name: "NOVA (orin)", kind: .real,
                  routes: novaHosts, ssh: "orin", saved: true),
        ]
    }
}
