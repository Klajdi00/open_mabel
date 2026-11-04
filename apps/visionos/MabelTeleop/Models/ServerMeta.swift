//
//  ServerMeta.swift
//  MabelTeleop
//
//  Server identity + live speed caps + safety posture, carried in `hello` and
//  every `robot_state.server`, so a rename / cap change / driver-lock change on
//  ANY client (iOS / web / headset) converges on every client. The bridge
//  (`sim_teleop_bridge.py` / `mabel_swerve_bridge.py`) owns this state, persists
//  it, and broadcasts it. Ported 1:1 from the MABEL iOS app so both clients read
//  the same fields.
//

import Foundation

/// The real robot's safety state, carried in every `robot_state.server` (and
/// `hello`): is a driver lock held, by whom, and has a LOCAL operator armed it.
/// `you` (on ServerMeta) is filled in by the link from `hello.you` so the UI
/// knows if WE hold the lock (driver.holderId == you).
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

/// Networking detail for every client connected to this robot (the tap-to-
/// inspect popup), classified server-side by monitor_taps.
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

struct ServerMeta: Equatable {
    var server: String?          // identity string, e.g. "mabel_sim_bridge"
    var kind: RobotKind = .unknown
    var name: String?            // server-owned display name (the synced one)
    var id: String?              // stable robot id, if the server reports one
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
    /// (Tailscale). nil when the server doesn't report it.
    var secure: Bool?
    /// Capabilities the gateway asks of the client (hello.caps). "stream_always" = the
    /// Mac Retarget Lab: stream tracking continuously while connected. Robot gateways
    /// send none.
    var caps: [String] = []

    /// An EMPTY meta, for a transport that has no `hello` payload to parse — the
    /// BLE link builds one field by field. Declaring `init(from:)` suppresses the
    /// memberwise initialiser, so this has to be spelled out.
    init() {}

    init(from payload: [String: Any]) {
        server = payload["server"] as? String
        if let k = payload["kind"] as? String {
            kind = RobotKind(rawValue: k) ?? .from(serverString: server)
        } else {
            kind = .from(serverString: server)
        }
        name = payload["name"] as? String
        id = payload["id"] as? String
        robot = payload["robot"] as? String
        // Only a PER-SIDE report counts. NOVA's first gateway sent a constant "orca-v2" whatever was
        // mounted — while MABEL's v1 hands were on it, so trusting it drew v2 fingers driven by v1
        // angles (2026-10-14). A bare string says nothing about the hands: ignore it (→ v1, see handGen).
        if let h = payload["hands"] as? [String: String] {
            hands = h
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
        caps = payload["caps"] as? [String] ?? []
    }
}
