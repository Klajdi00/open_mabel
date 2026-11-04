//
//  StatusDetailPanel.swift
//  MABEL Mobile App
//
//  The Control-Center-style dropdown revealed by tapping the status summary.
//  Shows every telemetry reading in a tidy grid plus the speed governor, so
//  the top bar itself stays slim and never has to scroll.
//

import SwiftUI

struct StatusDetailPanel: View {
    @EnvironmentObject private var controller: RobotController
    @Binding var expanded: Bool

    private let cols = [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)]
    /// Which modules are showing their individual motors. Collapsed by default: the
    /// count on the row is the answer 95 % of the time, and the motor list is what you
    /// open when the count is not n/n.
    @State private var openModules: Set<String> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            Label(controller.stopStatusLabel, systemImage: "exclamationmark.octagon")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(controller.stopState.confirmed ? Theme.Palette.live : Theme.Palette.warn)
            if let notice = controller.safetyNotice {
                Text(notice).font(.system(size: 11)).foregroundStyle(Theme.Palette.danger)
            }
            if let pending = controller.state.stopUnconfirmed, !pending.isEmpty {
                Text("Motor stop not verified: " + pending.joined(separator: ", "))
                    .font(.system(size: 11)).foregroundStyle(Theme.Palette.warn)
            }

            // The essentials only — concise enough for a floating popover. A SIM has
            // no battery or CPU-thermal to report (they'd be fake constants), so those
            // tiles drop out there — the sim shows Speed + Latency, the two live sim
            // numbers.
            let isSim = controller.target.isSimulation
            LazyVGrid(columns: cols, spacing: 8) {
                if !isSim {
                    if controller.state.batteryAvailable {
                        cell("bolt.fill", String(format: "%.0f", controller.state.batteryLevel * 100), "%", batteryTint, "Battery")
                    } else {
                        // Server says no battery monitor exists — honest n/a.
                        cell("bolt.slash", "—", "", Theme.Palette.textTertiary, "Battery")
                    }
                }
                cell("speedometer", String(format: "%.2f", controller.state.linearSpeed), "m/s", Theme.Palette.accent, "Speed")
                // A reading the link cannot carry is shown as "—". Over Bluetooth there is
                // no round-trip to measure and no thermal to read, and printing the
                // struct's defaults made a robot on BLE look exactly like one on Wi-Fi.
                let vitals = controller.state.vitalsAvailable
                cell("bolt.horizontal", vitals ? String(format: "%.0f", controller.state.latencyMs) : "—",
                     vitals ? "ms" : "", vitals ? Theme.Palette.warn : Theme.Palette.textTertiary, "Latency")
                if !isSim {
                    cell("thermometer.medium", vitals ? String(format: "%.0f", controller.state.cpuTemp) : "—",
                         vitals ? "°C" : "", vitals ? tempTint : Theme.Palette.textTertiary, "CPU")
                }
            }
            if !isSim, !controller.state.batteryAvailable, !controller.state.batteryReason.isEmpty {
                Text(controller.state.batteryReason)
                    .font(.system(size: 9.5, weight: .medium))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }

            // WHICH PART OF THE ROBOT IS ACTUALLY ATTACHED (robot_state.modules,
            // server/module_status.py). ALWAYS shown: the roster rides a 0.5 Hz
            // heartbeat, so there is a beat after connecting before it lands, and a
            // section that pops into existence late reads as a glitch. Until it
            // arrives every module shows WAITING (amber) — unknown, not broken.
            modulesSection

            // Server-authoritative echoes (2026-07): what the ROBOT reports it
            // is doing — e-stop, per-subsystem torque, active control method,
            // data collection. Only shown when the server sends them.
            if controller.state.estop != nil || !controller.state.torqueState.isEmpty
                || !controller.state.controlMethod.isEmpty || controller.state.dataCollection != nil {
                serverEchoes
            }

            // Speed governor — reachable here in any orientation.
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Label("Speed Governor", systemImage: "gauge.with.dots.needle.50percent")
                        .font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.Palette.textSecondary)
                    Spacer()
                    Text("\(Int(controller.speedScale * 100))%").font(.mabelMonoSmall).foregroundStyle(Theme.Palette.accent)
                }
                Slider(value: $controller.speedScale, in: 0.1...1).tint(Theme.Palette.accent)
            }
            .padding(9)
            .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        }
        .padding(13)
        .frame(width: 290)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }

    // MARK: - Modules (which limb / sensor is actually connected)

    /// The roster the server measured, in the server's order. Two columns for the
    /// eight actuator modules, then the sensor group (4 cams + lidar) under its own
    /// count — the panel is a 290 pt popover, so this has to stay dense: one line
    /// per module, dot + name, with the detail only where it earns its space.
    private var modulesSection: some View {
        // No roster yet (just connected / older bridge / demo) → show the robot's
        // shape with every module WAITING rather than an empty box or a wall of red.
        let waiting = controller.state.modules.isEmpty
        let mods = waiting ? RobotState.Module.placeholder : controller.state.modules
        let limbs = mods.filter { $0.group == nil }
        // SECTIONS COME FROM THE SERVER, in the order it first mentions each group.
        // This used to hardcode `group == "sensors"`, so the moment the roster grew
        // an "audio" group its rows VANISHED — a module the server is reporting must
        // never silently disappear because the app didn't recognise its section.
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
            // ONE ROW PER MODULE, each stating its OWN status — a dot alone makes
            // you hunt a caption to learn what is wrong with which part. Single
            // column (not two) so every row has room for its verdict on the right.
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
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }

    /// One module, one line: dot · name · COUNT · evidence. The count is the headline —
    /// "7/7" says every motor on that bus is answering, "5/7" says two are not and the
    /// limb must not be trusted. Tapping opens the motor list, so a fault is actionable
    /// ("J3, J6") instead of merely alarming ("DOWN").
    private func moduleRow(_ m: RobotState.Module, waiting: Bool) -> some View {
        // FOUR states, four colours — never conflate them:
        //   GREEN  n/n      every motor answering
        //   RED    up/n     the bus is up and named motors are silent, or it is down
        //   AMBER  WAIT     no roster yet — unknown, not broken
        //   GREY   —        NO DRIVER in this build (NOVA's hands / swerve base). Not a
        //                   fault: painting it red sends someone hunting a cable that
        //                   was never fitted.
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
    /// Dynamixels and this is a 290 pt popover, so they wrap into a grid instead of a
    /// list — the point is to spot the red one at a glance, not to read a table.
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
                    // Only a SHORT reading rides along ("38 °C"). "no reply" is already
                    // said by the colour, and repeating it would push the name out.
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

    private var header: some View {
        HStack(spacing: 8) {
            StatusPill(title: controller.connection.label, color: controller.connection.color, pulses: true)
            Spacer()
            Button {
                Haptics.select()
                withAnimation(Theme.Motion.snappy) { expanded = false }
            } label: {
                Image(systemName: "chevron.up")
                    .font(.system(size: 12, weight: .bold)).foregroundStyle(Theme.Palette.textTertiary)
                    .padding(8).background(Theme.Palette.surface, in: Circle())
            }
            .buttonStyle(PressableStyle())
        }
    }

    private func cell(_ icon: String, _ value: String, _ unit: String, _ tint: Color, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 5) {
                Image(icon: icon).font(.system(size: 11, weight: .semibold)).foregroundStyle(tint)
                Text(label.uppercased()).font(.system(size: 8.5, weight: .bold)).tracking(0.5)
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.mabel(17, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
                    .contentTransition(.numericText())
                if !unit.isEmpty {
                    Text(unit).font(.system(size: 10, weight: .medium, design: .monospaced)).foregroundStyle(Theme.Palette.textTertiary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }

    /// ROBOT-reported status rows: e-stop, torque enables, control method,
    /// data collection — server-authoritative, so they reflect every client.
    private var serverEchoes: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("ROBOT REPORTS").font(.system(size: 8.5, weight: .heavy)).tracking(1)
                .foregroundStyle(Theme.Palette.textTertiary)
            if let es = controller.state.estop {
                echoRow("exclamationmark.octagon.fill",
                        "E-stop", es ? "ENGAGED" : "clear",
                        es ? Theme.Palette.danger : Theme.Palette.live)
            }
            if !controller.state.torqueState.isEmpty {
                let t = controller.state.torqueState
                let order = ["left_hand", "right_hand", "arms", "base"]
                let parts = order.compactMap { k -> String? in
                    guard let v = t[k] else { return nil }
                    let name = k.replacingOccurrences(of: "_", with: " ")
                    return "\(name) \(v ? "ON" : "off")"
                }
                echoRow("bolt.circle", "Torque", parts.joined(separator: " · "),
                        t.values.contains(true) ? Theme.Palette.live : Theme.Palette.textTertiary)
            }
            if !controller.state.controlMethod.isEmpty {
                echoRow("slider.horizontal.3", "Control", controller.state.controlMethod,
                        Theme.Palette.accent)
            }
            if let dc = controller.state.dataCollection {
                let detail = dc.active
                    ? "\(dc.label ?? "session") · \(Int(dc.elapsedSec)) s"
                    : (dc.error ?? "idle")
                echoRow("record.circle", "Collection", detail,
                        dc.active ? Theme.Palette.danger
                            : (dc.error != nil ? Theme.Palette.warn : Theme.Palette.textTertiary))
            }
        }
        .padding(9)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }

    private func echoRow(_ icon: String, _ label: String, _ value: String, _ tint: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon).font(.system(size: 10, weight: .semibold)).foregroundStyle(tint)
            Text(label.uppercased()).font(.system(size: 8.5, weight: .bold)).tracking(0.5)
                .foregroundStyle(Theme.Palette.textTertiary)
                .frame(width: 68, alignment: .leading)
            Text(value).font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.Palette.textSecondary)
                .lineLimit(1).minimumScaleFactor(0.7)
            Spacer(minLength: 0)
        }
    }

    private var batteryTint: Color {
        let l = controller.state.batteryLevel
        return l > 0.4 ? Theme.Palette.live : (l > 0.18 ? Theme.Palette.warn : Theme.Palette.danger)
    }
    private var tempTint: Color {
        controller.state.cpuTemp > 70 ? Theme.Palette.danger : (controller.state.cpuTemp > 58 ? Theme.Palette.warn : Theme.Palette.textSecondary)
    }
}
