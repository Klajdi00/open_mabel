//
//  MabelWidgetSnapshot.swift
//  MABEL Mobile App  +  MabelWidgetExtension   (shared, compiled into BOTH targets)
//
//  The contract between the app and the Home/Lock-Screen widget. A WidgetKit
//  extension can't open a WebSocket or render live SceneKit, so the app writes
//  a compact, Codable snapshot of the latest robot state — plus a rendered PNG
//  of the current 3D pose — into a shared App Group container. The widget's
//  timeline provider reads it back. Keep this struct small and stable: every
//  field crosses a process boundary and is persisted between launches.
//

import Foundation
import SwiftUI
import ActivityKit

// MARK: - Snapshot

struct MabelWidgetSnapshot: Codable, Equatable {
    /// When the app captured this snapshot (drives the "updated 3m ago" staleness).
    var date: Date = .now

    // Identity
    var robotName: String = "MABEL"
    var target: String = "—"            // "SIM" · "REAL" · "—"
    var targetColor: ColorHex = .textTertiary

    // Connection
    var connectionLabel: String = "Offline"      // "Connected" · "Demo Mode" · "Connecting…" · "Offline"
    var connectionColor: ColorHex = .danger
    var isLive: Bool = false
    var transport: String = "—"          // "Wi-Fi LAN" · "Tailscale" · "Bluetooth" · "On-device"
    var transportIcon: String = "wifi.slash"
    var latencyMs: Double = 0

    // Power
    var batteryLevel: Double = 0         // 0…1
    var batteryVoltage: Double = 0
    var charging: Bool = false

    // Control posture
    var controlSummary: String = "—"     // e.g. "Guide · Hold · Arm" / "Navigation"
    var driver: String = "—"             // "You are driving" · "Observing" · "—"
    var isDriver: Bool = false
    var estop: Bool = false

    // Motion / health
    var linearSpeed: Double = 0          // m/s
    var liftFraction: Double = 0         // 0…1
    var cpuTemp: Double = 0              // °C
    var wifiRSSI: Int = 0               // dBm

    var diagnostics: [Diag] = []

    /// File name (in the App Group container) of the rendered pose PNG, if any.
    var poseImage: String? = nil

    struct Diag: Codable, Equatable, Identifiable {
        enum Level: Int, Codable { case ok = 0, warn = 1, fault = 2 }
        var id: String { subsystem }
        var subsystem: String
        var message: String
        var level: Level
    }

    // MARK: Derived helpers (used by the widget views)

    var batteryPercent: Int { Int((batteryLevel * 100).rounded()) }

    var batteryColor: ColorHex {
        if charging { return .live }
        return batteryLevel > 0.4 ? .live : (batteryLevel > 0.18 ? .warn : .danger)
    }

    var worstLevel: Diag.Level {
        diagnostics.map(\.level).max(by: { $0.rawValue < $1.rawValue }) ?? .ok
    }

    /// Shown before the app has ever written data (fresh install / widget added
    /// first). Honest "offline" state so the widget always shows *something*.
    static let coldStart = MabelWidgetSnapshot(
        robotName: "MABEL",
        target: "—", targetColor: .textTertiary,
        connectionLabel: "Open app to connect", connectionColor: .textTertiary, isLive: false,
        transport: "—", transportIcon: "wifi.slash", latencyMs: 0,
        batteryLevel: 0, batteryVoltage: 0,
        controlSummary: "Standby", driver: "—", isDriver: false,
        linearSpeed: 0, liftFraction: 0, cpuTemp: 0, wifiRSSI: 0,
        diagnostics: []
    )

    /// A neutral, plausible snapshot for the widget gallery / placeholder.
    static let placeholder = MabelWidgetSnapshot(
        robotName: "MABEL",
        target: "SIM", targetColor: .accent,
        connectionLabel: "Connected", connectionColor: .live, isLive: true,
        transport: "Wi-Fi LAN", transportIcon: "wifi", latencyMs: 18,
        batteryLevel: 0.78, batteryVoltage: 48.4,
        controlSummary: "Guide · Hold · Arm", driver: "You are driving", isDriver: true,
        linearSpeed: 0.0, liftFraction: 0.35, cpuTemp: 46, wifiRSSI: -52,
        diagnostics: [
            .init(subsystem: "Swerve Drive", message: "3/3 nominal", level: .ok),
            .init(subsystem: "Right Arm", message: "Joint 3 near limit", level: .warn),
            .init(subsystem: "Vision", message: "RGB-D 30 fps", level: .ok),
        ]
    )
}

// MARK: - Portable colour tokens
//
// The widget process can't reach the app's `Theme`, so the snapshot carries
// semantic colour *tokens* (not raw values that could drift). Both targets
// resolve them through the same table, so the widget matches the app exactly.

enum ColorHex: Int, Codable {
    case accent, live, warn, danger, textPrimary, textSecondary, textTertiary, ink

    var hex: UInt32 {
        switch self {
        case .accent:        return 0xC25B2A   // brand rust
        case .live:          return 0x2E9E52
        case .warn:          return 0xCC8A1E
        case .danger:        return 0xD23B2A
        case .textPrimary:   return 0xF2EEE6
        case .textSecondary: return 0xB8B2A8
        case .textTertiary:  return 0x8A847B
        case .ink:           return 0x161412
        }
    }

    var color: Color {
        Color(.sRGB,
              red: Double((hex >> 16) & 0xFF) / 255,
              green: Double((hex >> 8) & 0xFF) / 255,
              blue: Double(hex & 0xFF) / 255)
    }
}

// MARK: - App Group bridge

enum MabelWidgetShared {
    /// MUST match the App Group enabled on both the app and the widget target.
    static let appGroup = "group.org.example.MABEL-Mobile-App"
    /// WidgetKit kind id (used by the app to reload and by the widget definition).
    static let kind = "MabelStatusWidget"

    private static let snapshotFile = "snapshot.json"

    static func container() -> URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroup)
    }

    /// Absolute URL for a file in the shared container (e.g. the pose PNG).
    static func fileURL(_ name: String) -> URL? {
        container()?.appendingPathComponent(name)
    }

    static func load() -> MabelWidgetSnapshot? {
        guard let url = fileURL(snapshotFile),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder.snapshot.decode(MabelWidgetSnapshot.self, from: data)
    }

    @discardableResult
    static func save(_ snapshot: MabelWidgetSnapshot) -> Bool {
        guard let url = fileURL(snapshotFile),
              let data = try? JSONEncoder.snapshot.encode(snapshot) else { return false }
        do { try data.write(to: url, options: .atomic); return true } catch { return false }
    }
}

// MARK: - Live Activity (Dynamic Island)
//
// Shared between the app (which starts / updates / ends the activity while
// connected to a host) and the widget extension (which renders the Lock-Screen
// banner + every Dynamic Island presentation). The ContentState is what changes
// each update; it must stay small (ActivityKit caps the encoded state at ~4 KB).

struct MabelLiveActivityAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var date: Date

        // Connection
        var connectionLabel: String
        var connectionColor: ColorHex
        var isLive: Bool
        var target: String
        var targetColor: ColorHex
        var transport: String
        var transportIcon: String
        var latencyMs: Double

        // Power
        var batteryLevel: Double
        var charging: Bool

        // Control posture
        var controlSummary: String
        var driver: String
        var isDriver: Bool
        var estop: Bool
        var linearSpeed: Double
        var liftFraction: Double

        // Current high-level activity (recording / policy / navigation / control)
        var activityIcon: String
        /// Emoji that represents the current mode (shown in the compact island).
        var activityEmoji: String
        var activityTitle: String
        var activityDetail: String
        /// 0…1 when a real progress is known, or < 0 for indeterminate/none.
        var progress: Double
        /// Estimated seconds remaining for the running task, or < 0 if unknown.
        var etaSeconds: Double

        /// Pose PNG name in the shared container (rendered by the widget publisher).
        var poseImage: String?

        var batteryPercent: Int { Int((batteryLevel * 100).rounded()) }
        var batteryColor: ColorHex {
            if charging { return .live }
            return batteryLevel > 0.4 ? .live : (batteryLevel > 0.18 ? .warn : .danger)
        }
        var hasProgress: Bool { progress >= 0 }
        var hasETA: Bool { etaSeconds >= 0 }

        /// "2m 13s" / "45s" / "1h 04m" — human time remaining.
        var etaText: String {
            guard etaSeconds >= 0 else { return "" }
            let s = Int(etaSeconds.rounded())
            if s >= 3600 { return "\(s / 3600)h \(String(format: "%02dm", (s % 3600) / 60))" }
            if s >= 60 { return "\(s / 60)m \(String(format: "%02ds", s % 60))" }
            return "\(s)s"
        }

        /// Sample state for previews / the widget gallery.
        static let preview = ContentState(
            date: .now,
            connectionLabel: "Connected", connectionColor: .live, isLive: true,
            target: "SIM", targetColor: .accent, transport: "Wi-Fi LAN",
            transportIcon: "wifi", latencyMs: 18,
            batteryLevel: 0.78, charging: false,
            controlSummary: "Guide · Hold · Arm", driver: "You are driving",
            isDriver: true, estop: false, linearSpeed: 0.2, liftFraction: 0.35,
            activityIcon: "brain.head.profile", activityEmoji: "🤖",
            activityTitle: "Pick & Place", activityDetail: "Executing policy",
            progress: 0.62, etaSeconds: 27, poseImage: nil)
    }

    /// Stable for the life of the activity.
    var robotName: String
}

private extension JSONEncoder {
    static let snapshot: JSONEncoder = {
        let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; return e
    }()
}
private extension JSONDecoder {
    static let snapshot: JSONDecoder = {
        let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d
    }()
}
