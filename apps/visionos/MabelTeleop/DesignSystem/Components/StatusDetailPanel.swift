//
//  StatusDetailPanel.swift
//  MabelTeleop
//
//  Control-Center-style dropdown revealed by tapping the status summary —
//  every telemetry reading in a tidy grid, same as the iOS app.
//

import SwiftUI

struct StatusDetailPanel: View {
    /// Which modules are showing their individual motors. Collapsed by default: the
    /// count on the row is the answer 95 % of the time, and the motor list is what you
    /// open when the count is not n/n.
    @State private var openModules: Set<String> = []
    @EnvironmentObject private var controller: RobotController

    private let cols = [GridItem(.adaptive(minimum: 120), spacing: 8)]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StatusPill(title: controller.connection.label, color: controller.connection.color, pulses: true)

            // A SIM has no battery / voltage / CPU-thermal / Wi-Fi to report (they'd
            // be fake constants) — those real-robot-only tiles drop out in sim.
            let isSim = controller.target.isSimulation
            // A reading the LINK cannot carry is shown as "—". Over Bluetooth there is no
            // round-trip to measure, no battery gauge and no thermal to read, and printing
            // RobotState's defaults made a robot on BLE look exactly like one on Wi-Fi.
            let vitals = controller.state.vitalsAvailable
            let dim = Theme.Palette.textTertiary
            LazyVGrid(columns: cols, spacing: 8) {
                if !isSim {
                    cell("bolt.fill", vitals ? String(format: "%.0f", controller.state.batteryLevel * 100) : "—",
                         vitals ? "%" : "", vitals ? batteryTint : dim, "Battery")
                    cell("battery.100", vitals ? String(format: "%.1f", controller.state.batteryVoltage) : "—",
                         vitals ? "V" : "", vitals ? Theme.Palette.textSecondary : dim, "Voltage")
                }
                cell("speedometer", String(format: "%.2f", controller.state.linearSpeed), "m/s", Theme.Palette.accent, "Speed")
                cell("location.north.line", String(format: "%.0f", controller.state.heading * 180 / .pi), "°", Theme.Palette.textSecondary, "Heading")
                if !isSim {
                    cell("wifi", vitals ? "\(controller.state.wifiRSSI)" : "—", vitals ? "dBm" : "",
                         vitals ? Theme.Palette.live : dim, "Wi-Fi")
                }
                cell("bolt.horizontal", vitals ? String(format: "%.0f", controller.state.latencyMs) : "—",
                     vitals ? "ms" : "", vitals ? Theme.Palette.warn : dim, "Latency")
                if !isSim {
                    cell("thermometer.medium", vitals ? String(format: "%.0f", controller.state.cpuTemp) : "—",
                         vitals ? "°C" : "", vitals ? tempTint : dim, "CPU")
                }
                cell(controller.target.icon, controller.target.label, "", controller.target.color, "Driving")
            }

            modulesSection
        }
        .padding(18)
        .frame(width: 560)
    }

    // MARK: - Per-module connection roster (ported from iOS v2)

    private var modulesSection: some View {
        // No roster yet (just connected / older bridge / demo) → show the robot's
        // shape with every module WAITING rather than an empty box or a wall of red.
        let waiting = controller.state.modules.isEmpty
        let mods = waiting ? RobotState.Module.placeholder : controller.state.modules
        let limbs = mods.filter { $0.group == nil }
        // SECTIONS COME FROM THE SERVER, in the order it first mentions each group.
        // This used to hardcode `group == "sensors"`, so the moment the roster grew an
        // "audio" group its rows VANISHED — a module the server is reporting must never
        // silently disappear because the app did not recognise its section.
        var groupOrder: [String] = []
        for m in mods { if let g = m.group, !groupOrder.contains(g) { groupOrder.append(g) } }
        // Modules with NO DRIVER are left out of the headline: "1/9 up" on a robot whose
        // hands and base are simply not fitted reads like a broken robot.
        let fitted = mods.filter { waiting || $0.hasDriver }
        let up = fitted.filter(\.ok).count

        return VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 5) {
                Text("MODULES")
                    .font(.system(size: 8.5, weight: .bold)).tracking(0.5)
                    .foregroundStyle(Theme.Palette.textTertiary)
                Spacer(minLength: 0)
                Text(waiting ? "waiting…" : "\(up)/\(fitted.count) up")
                    .font(.system(size: 8.5, weight: .bold, design: .monospaced))
                    .foregroundStyle(waiting ? Theme.Palette.warn
                                     : (up == fitted.count ? Theme.Palette.live : Theme.Palette.danger))
            }
            // One row per module, each stating its OWN status.
            VStack(spacing: 3) { ForEach(limbs) { moduleRow($0, waiting: waiting) } }

            ForEach(groupOrder, id: \.self) { g in
                let rows = mods.filter { $0.group == g }
                let upN = rows.filter(\.ok).count
                HStack(spacing: 5) {
                    Text(g.uppercased())
                        .font(.system(size: 8, weight: .bold)).tracking(0.5)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Text(waiting ? "—" : "\(upN)/\(rows.count)")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(waiting ? Theme.Palette.warn
                                         : (upN == rows.count
                                            ? Theme.Palette.live : Theme.Palette.danger))
                    Spacer(minLength: 0)
                }
                .padding(.top, 2)
                VStack(spacing: 3) { ForEach(rows) { moduleRow($0, waiting: waiting) } }
            }
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }

    /// One module, one line: dot · name · COUNT · evidence. The count is the headline —
    /// "7/7" says every motor on that bus is answering, "5/7" says two are not and the
    /// limb must not be trusted. Tapping opens the motor list, so a fault is actionable
    /// ("J3, J6") instead of merely alarming ("DOWN").
    private func moduleRow(_ m: RobotState.Module, waiting: Bool) -> some View {
        // FOUR states, four colours: GREEN n/n · RED degraded or bus down · AMBER no
        // roster yet · GREY NO DRIVER in this build (NOVA's hands / swerve base), which
        // is not a fault and must not be painted as one.
        let health: RobotState.Module.Health = waiting ? .waiting : (m.ok ? .up : .down)
        let noDriver = !waiting && !m.hasDriver
        let tint: Color = {
            if noDriver { return Theme.Palette.textTertiary }
            switch health {
            case .up: return Theme.Palette.live
            case .down: return Theme.Palette.danger
            case .waiting: return Theme.Palette.warn
            }
        }()
        let open = openModules.contains(m.id)
        let canOpen = !m.actuators.isEmpty
        // The COUNT where there is one, the old word where there is not — a server that
        // reports no motor counts still gets a readable row.
        let verdict: String = {
            if waiting { return "WAIT" }
            if !m.countText.isEmpty { return m.countText }
            return m.hasDriver ? (m.ok ? "OK" : "DOWN") : "—"
        }()
        return VStack(alignment: .leading, spacing: 3) {
            Button {
                guard canOpen else { return }
                Haptics.select()
                withAnimation(Theme.Motion.snappy) {
                    if open { openModules.remove(m.id) } else { openModules.insert(m.id) }
                }
            } label: {
                HStack(spacing: 6) {
                    Circle().fill(tint).frame(width: 6, height: 6)
                    Text(m.label)
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(health == .up ? Theme.Palette.textPrimary
                                                       : Theme.Palette.textSecondary)
                        .lineLimit(1).minimumScaleFactor(0.8)
                    Spacer(minLength: 4)
                    Text(verdict)
                        .font(.system(size: 9, weight: .heavy, design: .monospaced)).tracking(0.3)
                        .foregroundStyle(tint)
                    Text(waiting ? "no data yet" : m.detail)
                        .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(health == .up ? Theme.Palette.textTertiary : tint.opacity(0.9))
                        .lineLimit(1).minimumScaleFactor(0.7)
                        .frame(width: 74, alignment: .trailing)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 7, weight: .black))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .rotationEffect(.degrees(open ? 90 : 0))
                        .opacity(canOpen ? 1 : 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!canOpen)

            if open { actuatorGrid(m) }
        }
        .accessibilityElement(children: .combine)
        // VoiceOver gets the same three facts as the row, in words: the count where there
        // is one, "no driver" where the module is not fitted, and the evidence either way.
        .accessibilityLabel({
            if waiting { return "\(m.label): waiting, no data yet" }
            if let up = m.up, let n = m.n, n > 0 {
                return "\(m.label): \(up) of \(n) motors online, \(m.detail)"
            }
            return "\(m.label): \(m.hasDriver ? (m.ok ? "ok" : "down") : "no driver"), \(m.detail)"
        }())
    }

    /// EVERY MOTOR IN THE MODULE, one pill each: name, colour, and its own evidence
    /// where that fits (a live motor shows its temperature). A hand is seventeen
    /// Dynamixels, so they wrap into a grid instead of a list — the point is to spot the
    /// red one at a glance, not to read a table.
    private func actuatorGrid(_ m: RobotState.Module) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 56), spacing: 4)],
                  alignment: .leading, spacing: 4) {
            ForEach(m.actuators) { a in
                HStack(spacing: 3) {
                    Circle().fill(a.ok ? Theme.Palette.live : Theme.Palette.danger)
                        .frame(width: 4, height: 4)
                    Text(a.label)
                        .font(.system(size: 8.5, weight: .semibold, design: .monospaced))
                        .foregroundStyle(a.ok ? Theme.Palette.textSecondary : Theme.Palette.danger)
                    if a.ok, !a.detail.isEmpty, a.detail.count <= 6 {
                        Text(a.detail)
                            .font(.system(size: 8, weight: .medium, design: .monospaced))
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                }
                .lineLimit(1).minimumScaleFactor(0.7)
                .padding(.horizontal, 5).padding(.vertical, 3)
                .background(a.ok ? Theme.Palette.surfaceHi : Theme.Palette.danger.opacity(0.16),
                            in: Capsule())
                .accessibilityLabel("\(a.label): \(a.ok ? "online" : "offline"), \(a.detail)")
            }
        }
        .padding(.leading, 12)
        .padding(.bottom, 2)
    }

    private func cell(_ icon: String, _ value: String, _ unit: String, _ tint: Color, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 5) {
                Image(icon: icon).font(.system(size: 12, weight: .semibold)).foregroundStyle(tint)
                Text(label.uppercased()).font(.system(size: 9, weight: .bold)).tracking(0.5)
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.mabel(18, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
                    .contentTransition(.numericText())
                if !unit.isEmpty {
                    Text(unit).font(.system(size: 11, weight: .medium, design: .monospaced)).foregroundStyle(Theme.Palette.textTertiary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(11)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }

    private var batteryTint: Color {
        let l = controller.state.batteryLevel
        return l > 0.4 ? Theme.Palette.live : (l > 0.18 ? Theme.Palette.warn : Theme.Palette.danger)
    }
    private var tempTint: Color {
        controller.state.cpuTemp > 70 ? Theme.Palette.danger : (controller.state.cpuTemp > 58 ? Theme.Palette.warn : Theme.Palette.textSecondary)
    }
}
