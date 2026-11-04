//
//  MabelLiveActivity.swift
//  MABEL Mobile App
//
//  Owns the Dynamic Island / Lock-Screen Live Activity. It starts the activity
//  ONLY while the app is connected to a real host (a MuJoCo sim bridge or the
//  physical robot — not the on-device demo), keeps it updated with throttled
//  telemetry + the current high-level activity (recording / policy / navigation /
//  teleop), and ends it the moment the link drops. The Dynamic Island UI itself
//  lives in the widget extension (MabelLiveActivityWidget).
//

import Foundation
import Combine
import ActivityKit

@MainActor
final class MabelLiveActivity {

    private weak var controller: RobotController?
    private var bag = Set<AnyCancellable>()
    private var activity: Activity<MabelLiveActivityAttributes>?

    private var lastPush = Date.distantPast
    private var lastKey = ""

    func start(_ controller: RobotController) {
        guard self.controller == nil else { return }
        self.controller = controller
        controller.objectWillChange
            .throttle(for: .seconds(3), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] _ in self?.sync() }
            .store(in: &bag)
        sync()
    }

    // MARK: - Lifecycle driven by connection

    private func sync() {
        guard let c = controller else { return }
        // Show the island ONLY while connected to a real host (sim bridge or
        // robot). The on-device demo (.demo) and disconnected both count as "not
        // connected" → end the activity so a stale "SIM connected" never lingers.
        guard c.connection == .connected else { stopActivity(); return }

        let state = buildState(c)
        if let activity {
            let key = changeKey(state)
            let now = Date()
            // Push on a meaningful change, otherwise at most every ~8s (battery,
            // latency drift) to respect ActivityKit's update budget.
            guard key != lastKey || now.timeIntervalSince(lastPush) > 8 else { return }
            lastKey = key; lastPush = now
            let content = ActivityContent(state: state, staleDate: now.addingTimeInterval(900))
            Task { await activity.update(content) }
        } else {
            startActivity(c, state: state)
        }
    }

    private func startActivity(_ c: RobotController, state: MabelLiveActivityAttributes.ContentState) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else { return }
        let attributes = MabelLiveActivityAttributes(robotName: c.robotStore.active?.name ?? "MABEL")
        let content = ActivityContent(state: state, staleDate: Date().addingTimeInterval(900))
        do {
            activity = try Activity.request(attributes: attributes, content: content, pushType: nil)
            lastKey = changeKey(state); lastPush = Date()
        } catch {
            // Activities disabled / over budget — fail quietly; we retry on next sync.
        }
    }

    private func stopActivity() {
        guard let a = activity else { return }
        activity = nil
        Task { await a.end(nil, dismissalPolicy: .immediate) }
    }

    /// End all of our activities (call on app teardown if needed).
    func endAll() {
        stopActivity()
        Task { for a in Activity<MabelLiveActivityAttributes>.activities { await a.end(nil, dismissalPolicy: .immediate) } }
    }

    // MARK: - State assembly

    private func buildState(_ c: RobotController) -> MabelLiveActivityAttributes.ContentState {
        let s = c.state
        let act = derivedActivity(c)
        return .init(
            date: .now,
            connectionLabel: c.connection.label,
            connectionColor: connectionColor(c.connection),
            isLive: c.connection.isLive,
            target: c.target.label,
            targetColor: targetColor(c.target),
            transport: c.connectionMethod?.rawValue ?? "—",
            transportIcon: c.connectionMethod?.icon ?? "wifi.slash",
            latencyMs: s.latencyMs,
            batteryLevel: s.batteryLevel,
            charging: s.charging,
            controlSummary: c.controlSummary,
            driver: driverString(c),
            isDriver: c.isDriver,
            estop: c.estopEngaged,
            linearSpeed: s.linearSpeed,
            liftFraction: c.liftFraction,
            activityIcon: act.icon,
            activityEmoji: act.emoji,
            activityTitle: act.title,
            activityDetail: act.detail,
            progress: act.progress,
            etaSeconds: act.eta,
            poseImage: MabelWidgetShared.load()?.poseImage
        )
    }

    private typealias DerivedActivity = (icon: String, emoji: String, title: String, detail: String, progress: Double, eta: Double)

    /// The single most relevant thing the robot is doing right now (real sources
    /// only; no fabricated progress). Priority: E-stop > recording > policy >
    /// navigation > teleop control mode. Each mode carries an emoji + (where it's
    /// genuinely known) a progress fraction and an ETA in seconds.
    private func derivedActivity(_ c: RobotController) -> DerivedActivity {
        if c.estopEngaged {
            return ("hand.raised.fill", "🛑", "Emergency Stop", "Motion halted", -1, -1)
        }
        if c.recording {
            let task = c.recordTask.isEmpty ? "Recording" : c.recordTask
            return ("record.circle.fill", "🔴", task, "\(c.episodeCount) episodes", -1, -1)
        }
        if let p = c.runningPolicy {
            // Estimate progress + ETA against the policy's time horizon.
            var progress = -1.0, eta = -1.0
            if p.horizonSec > 0, let since = c.runningPolicySince {
                let elapsed = Date().timeIntervalSince(since)
                let horizon = Double(p.horizonSec)
                progress = min(0.99, max(0, elapsed / horizon))
                eta = max(0, horizon - elapsed)
            }
            return (p.icon, "🤖", p.name, "Executing policy", progress, eta)
        }
        if let np = c.navProgress {
            let goal = c.navGoal.isEmpty ? "Navigating" : c.navGoal
            return ("location.north.line.fill", "🧭", goal, "En route", np, -1)
        }
        // Teleoperation — show the active control mode.
        if c.teleopMethod == .navigation {
            return ("gamecontroller.fill", "🕹️", "Base Driving", "Teleoperation", -1, -1)
        }
        return ("slider.horizontal.3", "🦾", c.controlSummary, "Teleoperation", -1, -1)
    }

    /// What forces an immediate push (vs. waiting for the throttle window).
    private func changeKey(_ s: MabelLiveActivityAttributes.ContentState) -> String {
        "\(s.connectionLabel)|\(s.target)|\(s.estop)|\(s.activityIcon)|\(s.activityTitle)|\(s.isDriver)|\(Int((max(0, s.progress) * 20).rounded()))"
    }

    // MARK: - Mapping

    private func connectionColor(_ s: ConnectionStatus) -> ColorHex {
        switch s {
        case .disconnected: return .danger
        case .connecting:   return .warn
        case .connected:    return .live
        case .demo:         return .accent
        }
    }
    private func targetColor(_ t: ConnectionTarget) -> ColorHex {
        switch t {
        case .simulation: return .accent
        case .realRobot:  return .live
        case .unknown:    return .textTertiary
        }
    }
    private func driverString(_ c: RobotController) -> String {
        guard c.connection.isLive, c.target == .realRobot else { return "—" }
        if c.isDriver { return "You are driving" }
        return c.driverHolder.isEmpty ? "Observing" : "Locked by \(c.driverHolder)"
    }
}
