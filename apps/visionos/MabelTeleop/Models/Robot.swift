//
//  Robot.swift
//  MabelTeleop
//
//  A robot the app can connect to — a simulation or a physical rig. Robots are remembered (persisted) with their host/IP, SSH alias, kind,
//  and a display name, so the same robot reads the same everywhere and is easy
//  to find. Ported 1:1 from the MABEL iOS app so both clients share the model.
//

import SwiftUI

enum RobotKind: String, Codable, Equatable {
    case sim, real, unknown

    var label: String {
        switch self {
        case .sim:     return "Simulation"
        case .real:    return "Real robot"
        case .unknown: return "Robot"
        }
    }
    /// Short badge for the row.
    var badge: String {
        switch self {
        case .sim:     return "SIM"
        case .real:    return "REAL"
        case .unknown: return "—"
        }
    }
    var icon: String {
        switch self {
        case .sim:     return "cube.transparent"          // virtual / simulated twin
        case .real:    return "gearshape.2.fill"          // physical machine (system symbol)
        case .unknown: return "questionmark.circle"
        }
    }
    var tint: Color {
        switch self {
        case .sim:     return Theme.Palette.accent
        case .real:    return Theme.Palette.live
        case .unknown: return Theme.Palette.textTertiary
        }
    }
    /// Map the server's `hello.kind` (or a server identity string) to a kind.
    static func from(serverString s: String?) -> RobotKind {
        guard let s = s?.lowercased() else { return .unknown }
        if s == "sim" || s.contains("sim") { return .sim }
        if s == "real" || s.contains("real") || s.contains("hardware") { return .real }
        return .unknown
    }
}

struct Robot: Identifiable, Codable, Equatable {
    /// Stable identity — the SAME robot reached via LAN, Tailscale, or Bluetooth is
    /// ONE entry keyed by this id (the server reports it in `hello`/`robot_state`).
    var id: String
    /// Display name — server-owned and synced to all clients; user-editable.
    var name: String
    var kind: RobotKind
    /// Direct routes to this robot (LAN IPs + Tailscale), fastest-first raced by
    /// the planner. A robot can be reached on any of them.
    var routes: [String] = []
    /// SSH alias or "user@ip" for physical robots.
    var ssh: String? = nil
    /// Remembered across launches (so we never have to re-detect from scratch).
    var saved: Bool = true

    // Transient (recomputed each session; not the identity).
    var online: Bool = false

    enum CodingKeys: String, CodingKey { case id, name, kind, routes, ssh, saved }

    /// A short, friendly summary of how this robot can be reached.
    var routeSummary: String {
        var parts: [String] = []
        if routes.contains(where: { NetworkConfig.classify($0) == .lan }) { parts.append("Wi-Fi") }
        if routes.contains(where: { NetworkConfig.classify($0) == .tailscale }) { parts.append("Tailscale") }
        return parts.isEmpty ? "—" : parts.joined(separator: " · ")
    }

    /// The primary host shown in the row (first LAN route, else first route).
    var displayHost: String {
        routes.first { NetworkConfig.classify($0) == .lan } ?? routes.first ?? "—"
    }
}
