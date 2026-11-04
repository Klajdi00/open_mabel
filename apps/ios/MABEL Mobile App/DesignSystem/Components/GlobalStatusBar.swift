//
//  GlobalStatusBar.swift
//  MABEL Mobile App
//
//  A slim, always-visible top bar — like the iOS status bar. It shows only
//  the essentials (connection, what we're driving, a compact vitals summary)
//  plus the always-reachable actions (speed governor, sim Reset, E-stop).
//  Tapping the summary opens a Control-Center-style dropdown (StatusDetailPanel)
//  with every reading, so nothing has to scroll and the action buttons always
//  fit and stay tappable.
//

import SwiftUI

struct GlobalStatusBar: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.isLandscape) private var isLandscape
    @Binding var expanded: Bool
    /// Floating "liquid glass" capsule (overlaid on the full-screen camera) vs the
    /// normal full-width docked bar.
    var floating: Bool = false

    @State private var showPicker = false
    @State private var showScenes = false
    @State private var showTransport = false
    @ObservedObject private var ble = BLECentral.shared

    /// How much the bar has to squeeze. The SIM adds WORLD + RESET to the
    /// row, and on a portrait iPhone the full row measured ~445 pt against a
    /// 402 pt screen; every chip is fixedSize, so the HStack could not shrink
    /// and instead widened its parent — the whole app slid right and the
    /// E-STOP went off the edge (2026-10-13). ViewThatFits now picks the first
    /// density that fits; the e-stop is never dropped, only its label.
    private enum Density { case full, compact, tight }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            barRow(.full)
            barRow(.compact)
            barRow(.tight)
        }
        .lineLimit(1)
        .padding(.horizontal, floating ? 2 : 11)
        .padding(.vertical, 3)
        // Fill the available width and never exceed it: the bar must not drive
        // the page width. The action buttons hold layoutPriority(2) so the info
        // chips + spacer yield first and the actions never get pushed off-screen.
        .frame(maxWidth: .infinity)
        // Floating = bare (just the status pills + buttons over the camera, no
        // glass bar). Docked = the full-width material bar with a hairline.
        .background(alignment: .bottom) {
            if !floating {
                Rectangle().fill(.regularMaterial)
                    .overlay(alignment: .bottom) { Rectangle().fill(Theme.Palette.hairlineSoft).frame(height: 1) }
            }
        }
        // The scene-switch loading overlay covers EVERYTHING while the sim
        // stack relaunches; the status bar is in every tab's hierarchy, so it
        // is the one stable host.
        .fullScreenCover(isPresented: Binding(
            get: { controller.simStudio.switching != nil },
            set: { if !$0 { controller.simStudio.dismissSwitch() } })) {
            SceneLoadingOverlay(studio: controller.simStudio)
                .presentationBackground(.clear)
                .interactiveDismissDisabled(controller.simStudio.switching?.failed == nil)
        }
        .sheet(isPresented: $showScenes) { SceneLibrarySheet(studio: controller.simStudio) }
        .onAppear {
            controller.syncSimStudio()
            #if DEBUG
            // Screenshot harness (synthetic taps don't land on the Simulator):
            // boot with the scene sheet open, or fire a scene switch by name.
            if UserDefaults.standard.bool(forKey: "debugScenes") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) { showScenes = true }
            }
            if let n = UserDefaults.standard.string(forKey: "debugSwitchScene") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 4.0) {
                    if let sc = controller.simStudio.scenes.first(where: { $0.name == n }) {
                        controller.simStudio.switchScene(sc)
                    }
                }
            }
            // Screenshot harness for the Home glide (synthetic taps don't land): fire
            // goHome a few seconds after connect so the arms-down glide is capturable.
            if UserDefaults.standard.bool(forKey: "debugGoHome") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { controller.goHome() }
            }
            // Reset harness: fire the sim reset repeatedly so the snap-to-home window
            // (where the hand-neutral pose must render, not a contorted zero pose) is
            // easy to catch in a screenshot.
            if UserDefaults.standard.bool(forKey: "debugResetSim") {
                for t in stride(from: 3.0, through: 9.0, by: 0.6) {
                    DispatchQueue.main.asyncAfter(deadline: .now() + t) { controller.resetSimRandom() }
                }
            }
            #endif
        }
        .onChange(of: controller.target) { _, _ in controller.syncSimStudio() }
        .onChange(of: controller.videoBase) { _, _ in controller.syncSimStudio() }
    }

    @ViewBuilder private func barRow(_ d: Density) -> some View {
        HStack(spacing: d == .full ? 8 : 6) {
            robotSwitcher                     // which robot (sim / real) + dropdown

            transportChip                     // HOW we reach it: Wi-Fi vs Bluetooth

            summaryButton(compact: d != .full) // robot status (taps to expand)

            Spacer(minLength: d == .full ? 6 : 2)

            if controller.canReset {
                if d == .full {
                    sceneButton               // jump to another sim world (sim only)
                    resetButton               // re-home / random respawn (sim only)
                } else {
                    simMenu                   // WORLD + RESET folded into one chip
                }
            }
            // Home GLIDES EVERY JOINT to the rest pose — it is a motion command, so
            // it only exists once the motors can actually take one. While the e-stop
            // holds (torque off) the robot cannot move, and a live-looking Home
            // button there is a control that silently does nothing. Press TORQUE ON
            // first; then Home appears. (2026-07-17)
            if controller.connection.isLive, !controller.estopEngaged {
                homeButton                    // glide to the rest pose (sim + real)
            }
            stopButton(iconOnly: d == .tight) // always reachable
        }
    }

    // MARK: - Robot switcher (icon-only; a compact glass dropdown on every page)

    private var robotSwitcher: some View {
        Button {
            Haptics.select(); showPicker.toggle()
        } label: {
            HStack(spacing: 4) {
                // The robot mark, drawn (RobotGlyph.swift). This used to be
                // `gearshape.2.fill`, which reads as "settings" everywhere else in
                // iOS — and sat inches from an actual settings gear — so the control
                // that says WHICH ROBOT YOU ARE DRIVING looked like preferences.
                // Non-robot states keep their symbols: a question mark means unknown, and those are the right glyphs.
                if isRobotState {
                    // SOLID head with the eyes cut out when live; OUTLINED head with
                    // solid eyes when not. The silhouette is identical either way, so
                    // the state reads from weight and colour rather than from the mark
                    // changing shape — and the eyes are visible in both, which they
                    // were not while they were punched through to transparency.
                    RobotGlyph(size: 12, filled: controller.connection.isLive)
                        .foregroundStyle(controller.connection.color)
                        .shadow(color: controller.connection.isLive
                                ? controller.connection.color.opacity(0.55) : .clear,
                                radius: 3)                       // a quiet glow only when live
                        .opacity(controller.connection == .connecting ? 0.7 : 1)
                        .scaleEffect(controller.connection == .connecting ? 0.94 : 1)
                        .animation(controller.connection == .connecting
                                   ? Theme.Motion.smooth.repeatForever(autoreverses: true)
                                   : Theme.Motion.snappy,
                                   value: controller.connection)
                } else {
                    Image(icon: statusIcon)
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(controller.connection.color)
                        .symbolEffect(.pulse, isActive: controller.connection == .connecting)
                }
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .heavy))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(.horizontal, 10).padding(.vertical, 7)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay { Capsule().strokeBorder(controller.connection.color.opacity(0.45), lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("status.robot")
        .popover(isPresented: $showPicker, arrowEdge: .top) {
            robotPicker.presentationCompactAdaptation(.popover)
        }
    }

    /// Connection state as a single glanceable icon, coloured by `connection.color`
    /// (green live · orange connecting · rust demo · red offline):
    ///   • connected → the SIM / REAL icon (what we're actually driving)
    ///   • connecting → antenna
    ///   • demo → SIM icon (on-device)
    ///   • disconnected → red "no network"
    /// True when the switcher is showing a robot (real, sim, or connecting to one)
    /// rather than a transport or an error — those keep their own symbols.
    private var isRobotState: Bool {
        switch controller.connection {
        case .connected, .connecting:
            return true
        default:
            return false
        }
    }

    private var statusIcon: String {
        switch controller.connection {
        case .connected:
            let isReal = controller.target == .realRobot || controller.robotStore.active?.kind == .real
            return isReal ? RobotKind.real.icon : RobotKind.sim.icon
        case .connecting:   return "antenna.radiowaves.left.and.right"
        case .demo:         return RobotKind.sim.icon
        case .disconnected: return "wifi.slash"
        }
    }

    /// The compact liquid-glass dropdown: pick a robot, or disconnect.
    private var robotPicker: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("ROBOTS").font(.system(size: 9, weight: .heavy)).tracking(1.2)
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 4)
            ForEach(controller.robotStore.robots) { robot in pickerRow(robot) }
            if controller.connection.isLive {
                Divider().overlay(Theme.Palette.hairlineSoft).padding(.horizontal, 8)
                Button {
                    Haptics.tap(); controller.disconnect(); showPicker = false
                } label: {
                    Label("Disconnect", systemImage: "xmark.circle")
                        .font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.danger)
                        .padding(.horizontal, 12).padding(.vertical, 9)
                }.buttonStyle(.plain)
            }
        }
        .frame(width: 248)
        .padding(.bottom, 6)
        .presentationBackground(.ultraThinMaterial)
    }

    private func pickerRow(_ robot: Robot) -> some View {
        let active = controller.robotStore.activeID == robot.id
        let live = active && controller.connection == .connected
        let connecting = active && controller.connection == .connecting
        // Status colour (green connected / amber connecting / muted offline) — the
        // glyph alone says sim vs real, so colour stays a pure connection signal.
        let statusColor: Color = live ? Theme.Palette.live
            : (connecting ? Theme.Palette.warn : Theme.Palette.textTertiary)
        return Button {
            Haptics.tap(); controller.connect(to: robot); showPicker = false
        } label: {
            HStack(spacing: 10) {
                Group {
                    if robot.kind == .real {
                        RobotGlyph(size: 14)
                    } else {
                        Image(icon: robot.kind.icon).font(.system(size: 14, weight: .semibold))
                    }
                }
                .foregroundStyle(statusColor).frame(width: 22)
                VStack(alignment: .leading, spacing: 1) {
                    Text(robot.name).font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                    Text(robot.routeSummary).font(.system(size: 9, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
                Spacer()
                if live {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.Palette.live)
                } else if active && controller.connection == .connecting {
                    ProgressView().controlSize(.mini)
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(active ? Theme.Palette.accent.opacity(0.10) : .clear)
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    // MARK: - Transport (how we reach the robot)

    /// The route in use, as one glanceable chip between "which robot" and "how is
    /// it doing". Wi-Fi and Bluetooth are genuinely different machines to drive —
    /// BLE carries the 20 Hz joint stream but no cameras, no map, no recording —
    /// so which one is live is a first-class piece of status, not a detail buried
    /// in Settings.
    private var transportChip: some View {
        Button {
            Haptics.select(); showTransport.toggle()
        } label: {
            ViewThatFits(in: .horizontal) {
                transportLabel(showTag: true)
                transportLabel(showTag: false)
            }
            .padding(.horizontal, 9).padding(.vertical, 7)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay { Capsule().strokeBorder(transportTint.opacity(0.45), lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("status.transport")
        .accessibilityLabel("Connection: \(transportTag)")
        .popover(isPresented: $showTransport, arrowEdge: .top) {
            transportPicker.presentationCompactAdaptation(.popover)
        }
    }

    private func transportLabel(showTag: Bool) -> some View {
        HStack(spacing: 5) {
            // The Bluetooth rune is DRAWN (TransportGlyph): SF Symbols has no Bluetooth
            // glyph, and the stand-in is the same broadcast icon as Wi-Fi — at 12pt the
            // chip could not tell you which transport was actually live.
            TransportGlyph(kind: route, size: 12, weight: .bold,
                           pulsing: controller.connection == .connecting)
                .foregroundStyle(transportTint)
            if showTag {
                Text(transportTag)
                    .font(.system(size: 9, weight: .heavy)).tracking(0.5)
                    .foregroundStyle(transportTint)
                    .fixedSize()
            }
            Image(systemName: "chevron.down").font(.system(size: 8, weight: .heavy))
                .foregroundStyle(Theme.Palette.textTertiary)
        }
    }

    /// The live transport, or nil when nothing is carrying commands.
    private var route: RouteKind? {
        if controller.overBluetooth { return .bluetooth }
        if let k = controller.activeRoute?.kind { return k }
        if controller.connection == .demo { return .demo }
        return nil
    }
    private var transportTag: String { route?.tag ?? "NO LINK" }
    private var transportTint: Color {
        guard route != nil else { return Theme.Palette.textTertiary }
        switch controller.connection {
        case .connected:    return Theme.Palette.live
        case .connecting:   return Theme.Palette.warn
        case .demo:         return Theme.Palette.accent
        case .disconnected: return Theme.Palette.textTertiary
        }
    }

    /// Both ways in, in one list, so switching is one tap rather than a trip
    /// through Settings. Rows that cannot work say why instead of failing.
    private var transportPicker: some View {
        VStack(alignment: .leading, spacing: 0) {
            sectionHeader("NETWORK")
            networkRow
            Divider().overlay(Theme.Palette.hairlineSoft).padding(.horizontal, 8).padding(.vertical, 4)
            sectionHeader("BLUETOOTH")
            bluetoothRows
        }
        .frame(width: 272)
        .padding(.bottom, 8)
        .presentationBackground(.ultraThinMaterial)
    }

    private func sectionHeader(_ s: String) -> some View {
        Text(s).font(.system(size: 9, weight: .heavy)).tracking(1.2)
            .foregroundStyle(Theme.Palette.textTertiary)
            .padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 4)
    }

    @ViewBuilder
    private var networkRow: some View {
        let online = controller.networkPath.isOnline
        let live = route != nil && route != .bluetooth && controller.connection.isLive
        // The route the racer actually picked, when it has raced; otherwise the
        // robot's own summary of where it can be found.
        let detail = live ? (controller.activeRoute.map { "\($0.kind.rawValue) · \($0.host)" } ?? "")
            : (online ? controller.networkPath.label
                      : "No network — connect Wi-Fi or cellular")
        transportRow(icon: online ? controller.networkPath.icon : "wifi.slash",
                     title: live ? "Network" : "Switch to network",
                     detail: detail, live: live, enabled: online && !live) {
            guard let robot = controller.robotStore.active
                    ?? controller.robotStore.robots.first else { return }
            controller.connect(to: robot)
            showTransport = false
        }
    }

    @ViewBuilder
    private var bluetoothRows: some View {
        // A reason only when there is nothing to offer. "Bluetooth is off" above a
        // list of robots is a contradiction the operator has to resolve.
        if ble.nearby.isEmpty {
            if ble.unauthorized {
                transportNote("Bluetooth access is off for MABEL — enable it in Settings")
            } else if !ble.poweredOn {
                transportNote("Bluetooth is off")
            } else {
                transportNote(controller.overBluetooth
                              ? "Connected — no other robots nearby"
                              : "No robots nearby")
            }
        }
        ForEach(ble.nearby) { robot in
            let live = controller.overBluetooth
                && controller.activeRoute?.host == robot.name
                && controller.connection.isLive
            transportRow(icon: RouteKind.bluetooth.icon, title: robot.name,
                         detail: live ? "Bluetooth · direct" : "\(robot.rssi) dBm",
                         live: live, enabled: !live) {
                controller.connectBluetooth(robot) { _ in }
                showTransport = false
            }
        }
    }

    private func transportNote(_ s: String) -> some View {
        Text(s).font(.system(size: 11, weight: .medium))
            .foregroundStyle(Theme.Palette.textTertiary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 12).padding(.vertical, 6)
    }

    private func transportRow(icon: String, title: String, detail: String,
                              live: Bool, enabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button(action: { Haptics.tap(); action() }) {
            HStack(spacing: 10) {
                TransportGlyph(icon: icon, size: 14)
                    .foregroundStyle(live ? Theme.Palette.live : Theme.Palette.textSecondary)
                    .frame(width: 22)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                    if !detail.isEmpty {
                        Text(detail).font(.system(size: 9, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
                    }
                }
                Spacer(minLength: 4)
                if live { Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.Palette.live) }
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(live ? Theme.Palette.accent.opacity(0.10) : .clear)
            .contentShape(Rectangle())
            .opacity(enabled || live ? 1 : 0.45)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }

    // MARK: - Compact vitals summary (taps to expand)

    private func summaryButton(compact: Bool) -> some View {
        Button {
            Haptics.select()
            expanded.toggle()
        } label: {
            // A SIM has no battery/CPU to report — show latency ONLY (the one live
            // sim number), and always keep it visible (never collapse to a bare
            // chevron). On a real robot, degrade gracefully when the bar is tight:
            // battery + latency if they fit, else battery only, else just the chevron.
            Group {
                if let notice = controller.safetyNotice {
                    Text(notice).font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Theme.Palette.warn).lineLimit(2).frame(maxWidth: 180)
                } else if controller.target.isSimulation {
                    vitals(showBattery: false, showLatency: true, showIcon: !compact)
                } else {
                    ViewThatFits(in: .horizontal) {
                        vitals(showBattery: true, showLatency: true)
                        vitals(showBattery: true, showLatency: false)
                        vitals(showBattery: false, showLatency: false)
                    }
                }
            }
            .padding(.horizontal, compact ? 8 : 10).padding(.vertical, 6)
            .background(Theme.Palette.surface.opacity(0.75), in: Capsule())
            .overlay { Capsule().strokeBorder(expanded ? Theme.Palette.accent.opacity(0.5) : Theme.Palette.hairlineSoft, lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("status.vitals")
        .layoutPriority(1)            // yields to the action buttons, beats the spacer
        // Floating liquid-glass popover — never pushes the page content (mirrors
        // the robot-switcher dropdown).
        .popover(isPresented: $expanded, arrowEdge: .top) {
            StatusDetailPanel(expanded: $expanded)
                .presentationCompactAdaptation(.popover)
        }
    }

    /// One vitals layout. Per-value fixed widths keep the % / ms from jiggling as
    /// the digit count changes (9->10->100); the chevron is always shown.
    private func vitals(showBattery: Bool, showLatency: Bool, showIcon: Bool = true) -> some View {
        HStack(spacing: 7) {
            if showBattery {
                // Honest gauge: a robot without a battery monitor reads "—",
                // not a fake constant (server echoes battery.available).
                let noBat = !controller.state.batteryAvailable
                Image(systemName: noBat ? "bolt.slash" : "bolt.fill")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(noBat ? Theme.Palette.textTertiary : batteryTint)
                Text(noBat ? "—" : "\(Int(controller.state.batteryLevel * 100))%")
                    .font(.mabelMonoSmall)
                    .foregroundStyle(noBat ? Theme.Palette.textTertiary : Theme.Palette.textPrimary)
                    .frame(width: 34, alignment: .leading)
            }
            if showLatency {
                if showBattery {   // separator only BETWEEN the two — never a stray leading line
                    Rectangle().fill(Theme.Palette.hairlineSoft).frame(width: 1, height: 11)
                }
                if showIcon {
                    Image(systemName: "bolt.horizontal").font(.system(size: 10, weight: .bold)).foregroundStyle(Theme.Palette.warn)
                }
                Text("\(Int(controller.state.latencyMs)) ms").font(.mabelMonoSmall).foregroundStyle(Theme.Palette.textPrimary)
                    .frame(width: 46, alignment: .leading)
            }
            Image(systemName: expanded ? "chevron.up" : "chevron.down")
                .font(.system(size: 9, weight: .bold)).foregroundStyle(Theme.Palette.textTertiary)
        }
    }

    // MARK: - Actions

    /// Tap = re-home (unchanged muscle memory). LONG-PRESS opens the menu with
    /// the extra sim resets — Menu(primaryAction:) gives exactly that split.
    private var resetButton: some View {
        Menu {
            Button {
                controller.resetSim()
            } label: { Label("Reset to Home", systemImage: "house.fill") }
            Button {
                controller.resetSimRandom()
            } label: { Label("Random Position", systemImage: "dice.fill") }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "arrow.counterclockwise")
                    .font(.system(size: 12, weight: .bold))
                if isLandscape {
                    Text("RESET").font(.mabel(12, weight: .heavy)).tracking(0.5)
                        .lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(Theme.Palette.textPrimary)
            .padding(.horizontal, isLandscape ? 13 : 9).padding(.vertical, 7)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        } primaryAction: {
            // Default TAP = random respawn (long-press menu still offers Home).
            controller.resetSimRandom()
        }
        .buttonStyle(PressableStyle())
        .layoutPriority(2)               // critical action — keep it on-screen
    }

    /// The world switcher (sim only, and only when the SCC studio is running
    /// beside the sim — scene switching is a supervised stack relaunch).
    @ViewBuilder private var sceneButton: some View {
        // Shown for ANY sim (canReset), not only when the SCC probe has already
        // confirmed — the button was "disappearing" whenever the scene backend
        // wasn't detected yet (plain start_teleop.sh, a late SCC, a transient probe
        // miss). Tapping RE-PROBES; the sheet shows the worlds, or a clear hint when
        // no scene backend is running.
        Group {
            Button {
                Haptics.select()
                controller.simStudio.reprobe()     // recover a missed / late SCC
                showScenes = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "square.stack.3d.up")
                        .font(.system(size: 12, weight: .bold))
                    if isLandscape {
                        Text("WORLD").font(.mabel(12, weight: .heavy)).tracking(0.5)
                            .lineLimit(1).fixedSize()
                    }
                }
                .foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, isLandscape ? 13 : 9).padding(.vertical, 7)
                .background(Theme.Palette.surfaceHi, in: Capsule())
                .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            }
            .buttonStyle(PressableStyle())
            .layoutPriority(2)
        }
    }

    /// Compact sim actions: WORLD + both resets behind one chip, for a bar too
    /// narrow to carry them separately (portrait iPhone on the sim).
    private var simMenu: some View {
        Menu {
            Button {
                Haptics.select()
                controller.simStudio.reprobe()
                showScenes = true
            } label: { Label("Switch World…", systemImage: "square.stack.3d.up") }
            Divider()
            Button {
                controller.resetSimRandom()
            } label: { Label("Random Position", systemImage: "dice.fill") }
            Button {
                controller.resetSim()
            } label: { Label("Reset to Home", systemImage: "house.fill") }
        } label: {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, 9).padding(.vertical, 7)
                .background(Theme.Palette.surfaceHi, in: Capsule())
                .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("status.simMenu")
        .accessibilityLabel("Simulation: world and reset")
        .layoutPriority(2)
    }

    /// Glide the robot to the minimal-torque REST pose — arms hanging down, torso &
    /// neck neutral, wheels forward, hands relaxed. A POSE home (works on sim + the
    /// real robot), distinct from the sim RESET that respawns the whole world.
    private var homeButton: some View {
        Button {
            Haptics.rigid(); controller.goHome()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "house.fill").font(.system(size: 12, weight: .bold))
                    .symbolEffect(.pulse, isActive: controller.homing)
                if isLandscape {
                    Text("HOME").font(.mabel(12, weight: .heavy)).tracking(0.5)
                        .lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(controller.homing ? Theme.Palette.accent : Theme.Palette.textPrimary)
            .padding(.horizontal, isLandscape ? 13 : 9).padding(.vertical, 7)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay {
                Capsule().strokeBorder(controller.homing ? Theme.Palette.accent.opacity(0.6)
                                                          : Theme.Palette.hairlineSoft, lineWidth: 1)
            }
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
        .layoutPriority(2)               // critical action — keep it on-screen
    }

    /// The e-stop, named for what it DOES to the motors (2026-07-17).
    ///
    /// "STOP" / "RESUME" described the robot's *motion*, which is only half the
    /// truth and the harmless-sounding half: on the wire this button cuts and
    /// restores MOTOR TORQUE. "Resume" reads like un-pausing a video — it gives no
    /// hint that pressing it re-energises every joint on a robot someone may have
    /// their hands inside. "TORQUE ON" says the limbs are about to become live;
    /// "E-STOP" says this is the emergency control, not a pause.
    private func stopButton(iconOnly: Bool) -> some View {
        let canResume = controller.estopEngaged && !controller.torqueOnRequested
        return Button {
            if canResume { Haptics.tap(); controller.releaseEStop() }
            else { Haptics.error(); controller.engageEStop() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: canResume ? "bolt.fill" : "exclamationmark.octagon.fill")
                    .font(.system(size: 12, weight: .bold))
                if !iconOnly {
                    VStack(spacing: 1) {
                        Text(canResume ? "TURN TORQUE ON" : "E-STOP")
                            .font(.mabel(12, weight: .heavy)).tracking(0.5)
                        Text(controller.stopStatusLabel).font(.system(size: 8, weight: .semibold))
                    }.lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(.white)
            .padding(.horizontal, iconOnly ? 11 : 14).padding(.vertical, 7)
            .background((canResume ? Theme.Palette.warn : Theme.Palette.danger).gradient, in: Capsule())
            .shadow(color: (canResume ? Theme.Palette.warn : Theme.Palette.danger).opacity(0.45), radius: 8, y: 2)
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("status.estop")
        .accessibilityLabel((canResume ? "Turn torque on. " : "Emergency stop. ") + controller.stopStatusLabel)
        .contextMenu { Button("E-STOP", role: .destructive) { controller.engageEStop() } }
        .layoutPriority(2)               // critical action — always reachable, never clipped
    }

    private var batteryTint: Color {
        let l = controller.state.batteryLevel
        return l > 0.4 ? Theme.Palette.live : (l > 0.18 ? Theme.Palette.warn : Theme.Palette.danger)
    }
}

/// Docked (full-width material + hairline) vs floating (rounded liquid-glass
/// capsule with a hairline border + soft shadow, so the camera shows through).
/// Shared by the top status bar and the bottom nav bar.
struct BarChrome: ViewModifier {
    let floating: Bool
    func body(content: Content) -> some View {
        if floating {
            content
                .background(.ultraThinMaterial, in: Capsule())
                .overlay { Capsule().strokeBorder(.white.opacity(0.22), lineWidth: 1) }
                .clipShape(Capsule())
                .shadow(color: .black.opacity(0.28), radius: 12, y: 4)
        } else {
            content
                .background(.regularMaterial)
                .overlay(alignment: .bottom) { Rectangle().fill(Theme.Palette.hairlineSoft).frame(height: 1) }
        }
    }
}
