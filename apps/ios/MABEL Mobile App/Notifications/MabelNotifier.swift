//
//  MabelNotifier.swift
//  MABEL Mobile App
//
//  The notification engine. It owns one taxonomy of robot events used by BOTH
//  paths that can reach the operator:
//
//    • LOCAL  — fired by RobotEventWatcher while the app is alive (foreground or
//               its short background window). Immediate, no server needed.
//    • REMOTE — Apple Push (APNs) sent by the teleop server so the operator is
//               alerted even when the phone is asleep or the app is killed (the
//               app can't observe the robot when it isn't running). This class
//               registers for push + hands the device token to the server.
//
//  The whole point is to be USEFUL without being NOISY: events are edge-triggered,
//  per-category opt-out, throttled, and ranked by interruption level so only
//  genuine safety/faults break through Focus.
//

import Foundation
import Combine
import UserNotifications
import UIKit

// MARK: - Taxonomy

enum RobotEventKind: String, CaseIterable, Codable {
    case connectionLost          // link to the robot dropped
    case connectionRestored      // …and came back (opt-in, quiet)
    case tipOver                 // chassis tilted past a safe angle — possible fall
    case estop                   // emergency stop engaged
    case overheat                // compute / motor temperature high
    case armDisarmed             // real robot lost its motion arm
    case batteryLow              // ≤ 20%
    case batteryCritical         // ≤ 10%
    case driverTaken             // another operator took the single-driver lock
    case policyComplete          // a dispatched policy finished (server-sourced)
    case navigationComplete      // autonomous navigation reached its goal
    case episodeSaved            // a teleop episode was recorded (opt-in)

    /// User-facing on/off group this event belongs to.
    var category: NotifCategory {
        switch self {
        case .connectionLost, .connectionRestored:           return .connection
        case .tipOver, .estop, .overheat, .armDisarmed:      return .safety
        case .batteryLow, .batteryCritical:                  return .battery
        case .policyComplete, .navigationComplete, .driverTaken: return .tasks
        case .episodeSaved:                                  return .episodes
        }
    }

    /// Some events are quiet by default even when their category is on.
    var defaultOptIn: Bool {
        switch self {
        case .connectionRestored, .episodeSaved: return false
        default: return true
        }
    }

    /// How hard it should try to reach the user. Safety + critical break Focus.
    var level: UNNotificationInterruptionLevel {
        switch self {
        case .tipOver, .estop, .connectionLost, .batteryCritical, .armDisarmed:
            return .timeSensitive
        case .episodeSaved, .connectionRestored:
            return .passive
        default:
            return .active
        }
    }

    /// Minimum spacing between two of the SAME event (anti-spam safety net).
    var minInterval: TimeInterval {
        switch self {
        case .connectionLost, .connectionRestored: return 60
        case .batteryLow, .batteryCritical:        return 600
        case .overheat:                            return 300
        default:                                   return 20
        }
    }

    /// Notifications of the same kind collapse into one thread.
    var threadID: String { "mabel.\(category.rawValue)" }
}

enum NotifCategory: String, CaseIterable {
    case safety, connection, battery, tasks, episodes

    var title: String {
        switch self {
        case .safety:     return "Safety & Faults"
        case .connection: return "Connection"
        case .battery:    return "Battery"
        case .tasks:      return "Tasks & Policies"
        case .episodes:   return "Episode Recording"
        }
    }
    var blurb: String {
        switch self {
        case .safety:     return "Tip-over, E-stop, overheating, lost motor arm"
        case .connection: return "The link to the robot drops"
        case .battery:    return "Battery gets low or critical"
        case .tasks:      return "A policy or navigation run finishes; driver lock taken"
        case .episodes:   return "Each teleop episode is saved (off by default)"
        }
    }
    var icon: String {
        switch self {
        case .safety:     return "exclamationmark.shield.fill"
        case .connection: return "antenna.radiowaves.left.and.right"
        case .battery:    return "battery.25"
        case .tasks:      return "checkmark.circle.fill"
        case .episodes:   return "film.stack"
        }
    }
    /// UserDefaults key for this category's toggle.
    var key: String { "notif.\(rawValue)" }
    /// Default on/off (Episodes off — it would fire constantly).
    var defaultOn: Bool { self != .episodes }
}

/// A concrete event ready to present. Built by the watcher or from a push payload.
struct RobotEvent {
    var kind: RobotEventKind
    var title: String
    var body: String
}

// MARK: - Engine

@MainActor
final class MabelNotifier: ObservableObject {
    static let shared = MabelNotifier()

    static let masterKey = "notif.enabled"

    @Published private(set) var authorization: UNAuthorizationStatus = .notDetermined
    @Published private(set) var apnsToken: String? = nil

    private var lastFired: [RobotEventKind: Date] = [:]
    private let center = UNUserNotificationCenter.current()

    private init() {
        UserDefaults.standard.register(defaults: [
            Self.masterKey: true,
            NotifCategory.safety.key: true,
            NotifCategory.connection.key: true,
            NotifCategory.battery.key: true,
            NotifCategory.tasks.key: true,
            NotifCategory.episodes.key: false,
            "notif.connectionRestored": false,
        ])
    }

    // MARK: Setup

    /// Refresh status and, if we've been granted (or can ask provisionally),
    /// register for remote push so the server can reach a sleeping phone.
    func start() {
        center.getNotificationSettings { [weak self] s in
            Task { @MainActor in
                self?.authorization = s.authorizationStatus
                if s.authorizationStatus == .authorized || s.authorizationStatus == .provisional {
                    UIApplication.shared.registerForRemoteNotifications()
                }
            }
        }
    }

    /// Ask the user. We request provisionally-quiet + alert so the first prompt is
    /// gentle; safety events still use time-sensitive delivery.
    func requestAuthorization() async {
        #if DEBUG
        // Screenshot/UI harness: `defaults write … debugSkipNotifPrompt -bool YES`
        // keeps the system permission dialog from covering the scene.
        if UserDefaults.standard.bool(forKey: "debugSkipNotifPrompt") { return }
        #endif
        do {
            let granted = try await center.requestAuthorization(options: [.alert, .sound, .badge])
            let s = await center.notificationSettings()
            authorization = s.authorizationStatus
            if granted { UIApplication.shared.registerForRemoteNotifications() }
        } catch {
            authorization = .denied
        }
    }

    var masterEnabled: Bool { UserDefaults.standard.bool(forKey: Self.masterKey) }

    func isEnabled(_ kind: RobotEventKind) -> Bool {
        guard masterEnabled else { return false }
        guard UserDefaults.standard.bool(forKey: kind.category.key) else { return false }
        if !kind.defaultOptIn {
            // Quiet-by-default events need their own explicit opt-in.
            switch kind {
            case .connectionRestored: return UserDefaults.standard.bool(forKey: "notif.connectionRestored")
            case .episodeSaved:       return true   // gated by the Episodes category itself
            default:                  return true
            }
        }
        return true
    }

    // MARK: Token (from AppDelegate) → server

    func setAPNSToken(_ token: String) { apnsToken = token }

    // MARK: Present

    /// Present a local notification for `event`, honouring toggles + throttling.
    func post(_ event: RobotEvent) {
        guard isEnabled(event.kind) else { return }
        // If the operator is actively in the app, the live UI already shows this —
        // a banner would just be noise. Local alerts are for when you're away;
        // true offline delivery is the server's APNs push.
        if UIApplication.shared.applicationState == .active { return }
        let now = Date()
        if let last = lastFired[event.kind], now.timeIntervalSince(last) < event.kind.minInterval {
            return   // too soon for this kind — drop it
        }
        lastFired[event.kind] = now

        let content = UNMutableNotificationContent()
        content.title = event.title
        content.body = event.body
        content.sound = event.kind.level == .passive ? nil : .default
        content.interruptionLevel = event.kind.level
        content.threadIdentifier = event.kind.threadID
        content.categoryIdentifier = "MABEL_EVENT"
        content.userInfo = ["kind": event.kind.rawValue, "deeplink": "mabelv2://island"]

        let req = UNNotificationRequest(identifier: "\(event.kind.rawValue)-\(Int(now.timeIntervalSince1970))",
                                        content: content, trigger: nil)   // deliver now
        center.add(req)
    }

    /// Build an event from a remote-push payload (so server pushes are formatted
    /// identically to local ones). Returns nil if the payload isn't ours.
    func event(fromPush userInfo: [AnyHashable: Any]) -> RobotEvent? {
        guard let raw = userInfo["kind"] as? String, let kind = RobotEventKind(rawValue: raw) else { return nil }
        let aps = userInfo["aps"] as? [String: Any]
        let alert = aps?["alert"] as? [String: Any]
        let title = (alert?["title"] as? String) ?? userInfo["title"] as? String ?? "MABEL"
        let body = (alert?["body"] as? String) ?? userInfo["body"] as? String ?? ""
        return RobotEvent(kind: kind, title: title, body: body)
    }

    /// Register the actionable category (a single "Open" maps to a tap-through).
    func registerCategories() {
        let open = UNNotificationAction(identifier: "OPEN_MABEL", title: "Open MABEL",
                                        options: [.foreground])
        let cat = UNNotificationCategory(identifier: "MABEL_EVENT", actions: [open],
                                         intentIdentifiers: [], options: [])
        center.setNotificationCategories([cat])
    }
}
