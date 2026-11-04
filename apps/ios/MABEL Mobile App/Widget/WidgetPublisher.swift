//
//  WidgetPublisher.swift
//  MABEL Mobile App
//
//  Bridges the live RobotController to the Home/Lock-Screen widget. It snapshots
//  the telemetry, renders the current 3D pose into the shared App Group
//  container, and asks WidgetKit to reload — but cheaply:
//
//    • Telemetry (the JSON) is written immediately on every capture; it's tiny.
//    • The pose PNG is rendered OFF the main thread and only when it changed
//      (WidgetPoseRenderer hashes the pose), so the app never janks.
//    • reloadAllTimelines() is rate-limited (~20s) except on important changes
//      (connect/disconnect, E-stop) so we stay inside WidgetKit's reload budget
//      and updates don't get silently dropped.
//
//  We always write a snapshot — even disconnected/demo — so the widget shows the
//  robot (last/home pose + an OFFLINE state) instead of going blank.
//

import Foundation
import Combine
import WidgetKit

@MainActor
final class MabelWidgetPublisher {

    private weak var controller: RobotController?
    private var bag = Set<AnyCancellable>()

    private var lastReload = Date.distantPast
    private var lastConnection = ""
    private var lastEstop = false

    func start(_ controller: RobotController) {
        guard self.controller == nil else { return }
        self.controller = controller

        controller.objectWillChange
            .throttle(for: .seconds(5), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] _ in self?.capture() }
            .store(in: &bag)

        capture()   // seed immediately so the widget has real data + a pose ASAP
    }

    /// Force an immediate, fresh capture (call on background / disconnect).
    func flush() { capture(force: true) }

    // MARK: - Capture

    private func capture(force: Bool = false) {
        guard let c = controller else { return }

        var snap = buildSnapshot(c)
        // Keep showing the last rendered pose until the new one is ready.
        let previous = MabelWidgetShared.load()
        snap.poseImage = previous?.poseImage ?? WidgetPoseRenderer.fileName
        MabelWidgetShared.save(snap)

        // Important transitions refresh the widget right away; routine ticks wait.
        let important = force
            || snap.connectionLabel != lastConnection
            || snap.estop != lastEstop
        lastConnection = snap.connectionLabel
        lastEstop = snap.estop
        reloadIfNeeded(force: important)

        // Render the pose off-main; only re-renders when the pose actually moved.
        let joints = c.modelJoints
        let heading = c.state.heading
        let quat = c.state.baseQuat
        let needFirstRender = (previous?.poseImage == nil)
        WidgetPoseRenderer.shared.render(joints: joints, heading: heading, baseQuat: quat,
                                         force: force || needFirstRender) { [weak self] name in
            guard let name else { return }
            DispatchQueue.main.async {
                guard let self else { return }
                var latest = MabelWidgetShared.load() ?? snap
                if latest.poseImage != name {
                    latest.poseImage = name
                    MabelWidgetShared.save(latest)
                }
                self.reloadIfNeeded(force: needFirstRender)
            }
        }
    }

    private func reloadIfNeeded(force: Bool) {
        let now = Date()
        guard force || now.timeIntervalSince(lastReload) > 20 else { return }
        lastReload = now
        WidgetCenter.shared.reloadAllTimelines()
    }

    // MARK: - Snapshot assembly

    private func buildSnapshot(_ c: RobotController) -> MabelWidgetSnapshot {
        let s = c.state
        var snap = MabelWidgetSnapshot()
        snap.date = .now
        snap.robotName = c.robotStore.active?.name ?? "MABEL"
        snap.target = c.target.label
        snap.targetColor = targetColor(c.target)

        snap.connectionLabel = c.connection.label
        snap.connectionColor = connectionColor(c.connection)
        snap.isLive = c.connection.isLive
        snap.transport = c.connectionMethod?.rawValue ?? "—"
        snap.transportIcon = c.connectionMethod?.icon ?? "wifi.slash"
        snap.latencyMs = s.latencyMs

        snap.batteryLevel = s.batteryLevel
        snap.batteryVoltage = s.batteryVoltage
        snap.charging = s.charging

        snap.controlSummary = c.controlSummary
        snap.estop = c.estopEngaged
        snap.isDriver = c.isDriver
        snap.driver = driverString(c)

        snap.linearSpeed = s.linearSpeed
        snap.liftFraction = c.liftFraction
        snap.cpuTemp = s.cpuTemp
        snap.wifiRSSI = s.wifiRSSI

        snap.diagnostics = s.diagnostics.prefix(6).map {
            .init(subsystem: $0.subsystem, message: $0.message, level: level($0.level))
        }
        return snap
    }

    // MARK: - Mapping (live types → portable tokens)

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
        return c.driverHolder.isEmpty ? "Observing (lock free)" : "Locked by \(c.driverHolder)"
    }

    private func level(_ l: RobotState.Diagnostic.Level) -> MabelWidgetSnapshot.Diag.Level {
        switch l {
        case .ok:    return .ok
        case .warn:  return .warn
        case .fault: return .fault
        }
    }
}
