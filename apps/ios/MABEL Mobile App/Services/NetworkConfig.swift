//
//  NetworkConfig.swift
//  MABEL Mobile App
//
//  The connection map shared with the Vision Pro app and the web console
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
    /// once; the server-owned name then overrides on connect. 1:1 with the Vision Pro app.
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

/// Per-robot stick→twist sign mapping. The simulation and the physical rig have
/// INDEPENDENT base-frame conventions, so each robot kind gets its own signs:
///   · Sim  — matches `controller/mabel/teleop_controller._navigate`
///            (tvx = ly, tvy = lx): forward = +ly, strafe-right = −lx.
///   · Real — the real rig now routes through ROS (app → /mabel_cmd →
///            base_bridge_client → hardware_bridge), which applies the sim→real
///            180° base-frame flip (vx,vy → −vx,−vy). That makes the real rig match
///            the SIM stick signs (forward = +ly, strafe-right = −lx). (The old −ly/+lx
///            was for the retired direct mabel_swerve_bridge path; re-verified
///            2026-06-28 after the bridge migration.)
/// Yaw is the same on both (right stick right = turn right).
struct TwistSigns {
    let fwd: Double, strafe: Double, yaw: Double
    static func forKind(_ k: RobotKind) -> TwistSigns {
        switch k {
        case .real: return TwistSigns(fwd: +1, strafe: -1, yaw: -1)   // physical rig (via ROS bridge)
        default:    return TwistSigns(fwd: +1, strafe: -1, yaw: -1)   // sim / demo / unknown
        }
    }
}

/// The real robot's safety state, carried in every `robot_state.server` (and
/// `hello`): is a driver lock held, by whom, and has a LOCAL operator armed it.
/// `you` is filled in by the link from `hello.you` so the UI knows if WE hold it.
struct DriverState: Equatable {
    var locked = false
    var holder = ""           // display name of the lock holder
    var holderId: String?     // connection id of the holder (compared to our `you`)
    var armed = true          // real robot: false until armed locally on thor

    init() {}
    init(from p: [String: Any]) {
        locked = (p["locked"] as? Bool) ?? false
        holder = (p["holder"] as? String) ?? ""
        holderId = p["holderId"] as? String
        armed = (p["armed"] as? Bool) ?? true
    }
}

/// Server identity + live speed caps, carried in `hello` and every `robot_state`
/// so a rename or a cap change on any client converges on every client.
struct ServerMeta: Equatable {
    var server: String?
    var kind: RobotKind = .unknown
    var name: String?
    var id: String?
    /// WHICH robot ("MABEL" / "NOVA"), as the gateway says it (hello.robot). nil from an older
    /// gateway — then `RobotRig` falls back to the `server` string.
    var robot: String?
    /// The end effector mounted per side ("left"/"right" → "v1" MABEL's ORCA hand / "v2" NOVA's / "gripper"
    /// the OpenArm v1 parallel gripper), from hello.hands.
    /// The hands swap between the robots, so this — not the robot — decides the hand drawn.
    var hands: [String: String]?
    var vmax: Double?
    var wmax: Double?
    var vmaxCeil: Double?
    var wmaxCeil: Double?
    /// Security posture (real robot): is an access code / local arm required, and
    /// the live driver-lock + arm state.
    var authRequired = false
    var armRequired = false
    var driver = DriverState()
    /// OUR connection id on this robot (from `hello.you`), stamped by the link onto
    /// every meta so the UI can tell whether WE hold the lock (driver.holderId == you).
    var you: String?
    /// Networking detail for every client connected to this robot (the tap-to-
    /// inspect popup), classified server-side by monitor_taps.
    var clients: [ClientInfo] = []
    /// The transport THIS client reached the robot over, classified server-side
    /// ("LAN" / "Tailscale" / "Localhost"). Authoritative — preferred
    /// over the app's own host-IP guess. nil when talking to an older server.
    var transport: String?
    /// Whether the server considers this client's link encrypted end-to-end
    /// (e.g. Tailscale). nil when the server doesn't report it.
    var secure: Bool?

    /// An EMPTY meta, for a transport that has no `hello` payload to parse — the
    /// BLE link builds one field by field. Declaring `init(from:)` suppresses the
    /// memberwise initialiser, so this has to be spelled out; the visionOS copy of
    /// this struct always had it, which is why `BLELink` compiled there and not
    /// here.
    init() {}

    init(from payload: [String: Any]) {
        server = payload["server"] as? String
        if let k = payload["kind"] as? String { kind = RobotKind(rawValue: k) ?? .from(serverString: server) }
        else { kind = .from(serverString: server) }
        name = payload["name"] as? String
        id = payload["id"] as? String
        robot = payload["robot"] as? String
        if let h = payload["hands"] as? [String: String] {
            hands = h
        } else if let h = payload["hands"] as? String {          // NOVA's first gateway: "orca-v2" for both
            let g = h.lowercased().hasSuffix("v1") ? "v1" : "v2"
            hands = ["left": g, "right": g]
        }
        vmax = payload["vmax"] as? Double
        wmax = payload["wmax"] as? Double
        vmaxCeil = payload["vmaxCeil"] as? Double
        wmaxCeil = payload["wmaxCeil"] as? Double
        authRequired = (payload["authRequired"] as? Bool) ?? false
        armRequired = (payload["armRequired"] as? Bool) ?? false
        if let d = payload["driver"] as? [String: Any] { driver = DriverState(from: d) }
        if let arr = payload["clients"] as? [[String: Any]] {
            clients = arr.map(ClientInfo.init(from:))
        }
        transport = payload["transport"] as? String
        secure = payload["secure"] as? Bool
        if let pm = payload["ports"] as? [String: Any] {
            var m: [String: Int] = [:]
            for (k, v) in pm { if let i = v as? Int { m[k] = i } }
            ports = m
        }
    }

    /// Full endpoint map from `hello.ports` (2026-07 data-plane update): the
    /// legacy plane (teleop/camera/udpVideo/map) plus the 96xx streams
    /// (telemetry, per-camera clouds, slam, hi-res video). Empty when talking
    /// to an older server — callers fall back to the hardcoded defaults.
    var ports: [String: Int] = [:]
}

/// One client connected to a robot — shown in the host inspector popup.
struct ClientInfo: Equatable, Identifiable {
    var id: String
    var ip: String
    var device: String
    var os: String
    var transport: String      // LAN / Tailscale / Localhost
    var app: String
    var driver: Bool
    var secure: Bool

    init(from p: [String: Any]) {
        id = (p["id"] as? String) ?? UUID().uuidString
        ip = (p["ip"] as? String) ?? ""
        device = (p["device"] as? String) ?? "Client"
        os = (p["os"] as? String) ?? ""
        transport = (p["transport"] as? String) ?? ""
        app = (p["app"] as? String) ?? ""
        driver = (p["driver"] as? Bool) ?? false
        secure = (p["secure"] as? Bool) ?? false
    }

    /// Network-location label derived from the transport (no geo-IP needed).
    var location: String {
        switch transport.lowercased() {
        case "lan":        return "Local network"
        case "tailscale":  return "VPN · Tailscale"
        case "localhost":  return "This machine"
        default:           return transport.isEmpty ? "—" : transport
        }
    }
}
