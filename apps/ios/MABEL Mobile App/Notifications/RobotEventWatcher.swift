//
//  RobotEventWatcher.swift
//  MABEL Mobile App
//
//  Turns RobotController state transitions into notification events. It is the
//  LOCAL half of the notification system (the server's APNs pushes are the other
//  half, for when the app isn't running). Everything here is EDGE-triggered and
//  debounced so a steady state never re-alerts:
//
//    • connection lost only after a grace period (longer than auto-reconnect),
//    • battery thresholds fire once per crossing,
//    • tip-over uses hysteresis,
//    • events while the app is in the FOREGROUND are suppressed by MabelNotifier
//      (you're already looking at the live UI — no need to buzz you).
//
//  It also hands our APNs token to the connected server so it can push us later.
//

import Foundation
import Combine
import simd

@MainActor
final class RobotEventWatcher {

    private weak var controller: RobotController?
    private let notifier = MabelNotifier.shared
    private var bag = Set<AnyCancellable>()

    // Edge state
    private var wasConnected = false
    private var everConnected = false
    private var lostGrace: Timer?
    private var didReportLost = false
    private var prevEstop = false
    private var prevArmed = true
    private var prevHolder = ""
    private var tipped = false
    private var firedLow = false
    private var firedCritical = false
    private var prevEpisodes = -1
    private var prevNav: Double? = nil
    private var sentToken: String?

    func start(_ controller: RobotController) {
        guard self.controller == nil else { return }
        self.controller = controller
        notifier.registerCategories()

        controller.objectWillChange
            .throttle(for: .seconds(1), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] _ in self?.evaluate() }
            .store(in: &bag)
        // Push our token to the server when it (or the connection) becomes available.
        notifier.$apnsToken
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.pushTokenIfPossible() }
            .store(in: &bag)
    }

    // MARK: - Edge evaluation

    private func evaluate() {
        guard let c = controller else { return }
        let connected = (c.connection == .connected)   // a real host (sim bridge / real robot)
        let s = c.state

        // 1) Connection lost / restored ----------------------------------------
        if connected {
            everConnected = true
            lostGrace?.invalidate(); lostGrace = nil
            if didReportLost {
                didReportLost = false
                notifier.post(.init(kind: .connectionRestored, title: "\(name(c)) reconnected",
                                    body: "The link to your robot is back online."))
            }
            pushTokenIfPossible()
        } else if wasConnected, everConnected, lostGrace == nil, !didReportLost {
            // Was live, now isn't — wait out auto-reconnect before alarming.
            lostGrace = Timer.scheduledTimer(withTimeInterval: 12, repeats: false) { [weak self] _ in
                Task { @MainActor in self?.confirmLost() }
            }
        }
        wasConnected = connected

        // The rest only make sense against a live robot.
        guard connected else { resetTransients(); return }

        // 2) Tip-over (chassis tilt past a safe angle) -------------------------
        let tilt = tiltDegrees(s.baseQuat)
        if tilt > 35, !tipped {
            tipped = true
            notifier.post(.init(kind: .tipOver, title: "⚠️ \(name(c)) may have tipped",
                                body: String(format: "Chassis tilted %.0f° from level. Check the robot.", tilt)))
        } else if tilt < 20 { tipped = false }

        // 3) E-stop engaged ----------------------------------------------------
        if c.estopEngaged, !prevEstop {
            notifier.post(.init(kind: .estop, title: "🛑 Emergency stop — \(name(c))",
                                body: "The robot's emergency stop is engaged. Motion is halted."))
        }
        prevEstop = c.estopEngaged

        // 4) Overheat ----------------------------------------------------------
        if s.cpuTemp > 80 {
            notifier.post(.init(kind: .overheat, title: "🌡️ \(name(c)) running hot",
                                body: String(format: "Compute temperature is %.0f °C.", s.cpuTemp)))
        }

        // 5) Arm disarmed (real robot lost motion enable) ----------------------
        if c.target == .realRobot {
            if prevArmed, !c.armed {
                notifier.post(.init(kind: .armDisarmed, title: "\(name(c)) disarmed",
                                    body: "The robot is no longer armed — motion is disabled."))
            }
            prevArmed = c.armed

            // 6) Driver lock taken by someone else -----------------------------
            if !c.isDriver, !c.driverHolder.isEmpty, c.driverHolder != prevHolder {
                notifier.post(.init(kind: .driverTaken, title: "Control taken on \(name(c))",
                                    body: "\(c.driverHolder) is now driving the robot."))
            }
            prevHolder = c.driverHolder
        }

        // 7) Battery thresholds (once per crossing) ----------------------------
        let b = s.batteryLevel
        if b > 0.25 { firedLow = false; firedCritical = false }
        if b <= 0.10, !firedCritical {
            firedCritical = true; firedLow = true
            notifier.post(.init(kind: .batteryCritical, title: "🪫 \(name(c)) battery critical",
                                body: "Battery at \(pct(b)). Dock or swap the pack now."))
        } else if b <= 0.20, !firedLow {
            firedLow = true
            notifier.post(.init(kind: .batteryLow, title: "\(name(c)) battery low",
                                body: "Battery at \(pct(b))."))
        }

        // 8) Episode saved (opt-in) -------------------------------------------
        if c.recording {
            if prevEpisodes >= 0, c.episodeCount > prevEpisodes {
                notifier.post(.init(kind: .episodeSaved, title: "Episode saved · \(c.recordTask)",
                                    body: "\(c.episodeCount) episodes recorded in this task."))
            }
            prevEpisodes = c.episodeCount
        } else { prevEpisodes = -1 }

        // 9) Navigation complete (when a follower reports progress → 1) --------
        if let np = c.navProgress {
            if let prev = prevNav, prev < 0.999, np >= 0.999 {
                let goal = c.navGoal.isEmpty ? "its goal" : c.navGoal
                notifier.post(.init(kind: .navigationComplete, title: "Arrived · \(name(c))",
                                    body: "Navigation to \(goal) is complete."))
            }
            prevNav = np
        } else { prevNav = nil }
    }

    private func confirmLost() {
        guard let c = controller, c.connection != .connected else { return }
        didReportLost = true
        lostGrace = nil
        notifier.post(.init(kind: .connectionLost, title: "Lost connection to \(name(c))",
                            body: "The robot link dropped and hasn't recovered."))
    }

    private func resetTransients() {
        prevEstop = false; tipped = false; firedLow = false; firedCritical = false
        prevEpisodes = -1; prevNav = nil
    }

    private func pushTokenIfPossible() {
        guard let c = controller, c.connection == .connected,
              let token = notifier.apnsToken, token != sentToken else { return }
        sentToken = token
        c.registerPushToken(token)
    }

    // MARK: - Helpers

    private func name(_ c: RobotController) -> String { c.robotStore.active?.name ?? "MABEL" }
    private func pct(_ b: Double) -> String { "\(Int((b * 100).rounded()))%" }

    /// Angle (degrees) between the chassis "up" axis and world up, from the
    /// MuJoCo world quaternion [w,x,y,z] (Z-up). 0 = perfectly level.
    private func tiltDegrees(_ q: [Double]) -> Double {
        guard q.count == 4 else { return 0 }
        let w = q[0], x = q[1], y = q[2], z = q[3]
        // World-up component of the body's local +Z after rotation: R[2][2].
        let zz = 1 - 2 * (x * x + y * y)
        let clamped = max(-1, min(1, zz))
        return acos(clamped) * 180 / .pi
    }
}
