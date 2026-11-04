//
//  GlobalStatusBar.swift
//  MabelTeleop
//
//  The slim, always-visible top bar — identical in content to the iOS app:
//  connection pill, SIM/REAL chip, compact vitals summary (tap for the
//  Control-Center-style detail POPOVER), speed governor, sim Reset, and the
//  always-reachable emergency STOP.
//
//  Performance: the governor slider writes a plain (non-published) var on the
//  controller and keeps its own local state, so dragging it never re-renders
//  the window; the vitals text refreshes on the controller's ~8 Hz uiTick.
//

import SwiftUI

struct GlobalStatusBar: View {
    @EnvironmentObject private var controller: RobotController
    @EnvironmentObject private var discovery: ServerDiscovery
    @State private var showDetail = false
    @State private var showPicker = false
    @State private var showScenes = false
    @State private var showTransport = false
    @ObservedObject private var ble = BLECentral.shared
    @State private var governor: Double = 1.0

    var body: some View {
        // STREAMLINED (2026-07-11): sim/real is part of the connection pill,
        // the speed governor lives under the status dropdown — the bar is just
        // [robot · vitals] ... [RESET · WORLD · STOP].
        HStack(spacing: 10) {
            robotSwitcher                            // which robot (sim / real)

            transportChip                            // HOW we reach it: Wi-Fi vs Bluetooth

            summaryButton

            // The robot's own word on torque: a REFUSAL ("Torque-on refused: …") or what torque-on
            // left limp. Both used to be decoded and never shown, so a refused Torque On just looked
            // like a dead button (2026-10-25).
            if let notice = controller.safetyNotice ?? controller.torqueDegradedLine {
                Text(notice).font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.Palette.warn).lineLimit(2).frame(maxWidth: 280, alignment: .leading)
                    .accessibilityIdentifier("status.safetyNotice")
            }

            Spacer(minLength: 6)

            if controller.canReset {
                resetButton                          // random respawn (sim only)
                sceneButton                          // world switcher (sim; SCC-backed)
            }
            // Home GLIDES EVERY JOINT to the rest pose — a motion command, so it
            // only exists once the motors can take one. While the e-stop holds
            // (torque off) it would silently do nothing. (iOS parity, 2026-07-17.)
            if controller.connection.isLive, !controller.estopEngaged {
                homeButton                           // glide to the rest pose (sim + real)
            }
            // (RECORD moved OUT of the status bar, 2026-07-28: it was the SECOND
            // Rec control on screen — the Robot space already carries one in its
            // left column under [3D | Physics], next to the map source, which is
            // where the other data-source controls live. Two of the same button
            // a few centimetres apart is worse than either alone.)
            stopButton                               // always reachable
        }
        .lineLimit(1)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .glassBackgroundEffect(in: Capsule())
        .onAppear { governor = controller.speedScale; controller.syncSimStudio() }
        // The scene library / physics viewer probe must re-run once the video
        // base + target are actually known — syncing only on the hello (which
        // can precede videoBase) left `simStudio.available` stuck false: the
        // WORLD + Physics buttons never appeared (iOS parity fix).
        .onChange(of: controller.target) { _, _ in controller.syncSimStudio() }
        .onChange(of: controller.videoBase) { _, _ in controller.syncSimStudio() }
        // The 3D stage must yield while the Worlds sheet is up (it occludes
        // window modals AND its invisible tap targets eat their pinches).
        .onChange(of: showScenes) { _, v in controller.modalPopupVisible = v }
    }

    // MARK: - Robot switcher (connection status + liquid-glass dropdown)

    /// The connection pill is a BUTTON: tap to drop a glass panel and switch
    /// robots from ANY page — Simulation, the real rig, or a custom host —
    /// mirroring the iOS app's robot switcher.
    private var robotSwitcher: some View {
        let color = controller.connection.color
        return Button {
            Haptics.select()
            if !controller.connectedToRealServer { discovery.start() }  // refresh the LAN list
            controller.syncSimStudio()   // late-started SCC studio gets picked up
            showPicker.toggle()
        } label: {
            HStack(spacing: 7) {
                ZStack {
                    Circle().fill(color.opacity(0.5)).frame(width: 9, height: 9)
                        .scaleEffect(controller.connection == .connecting ? 1.6 : 1)
                    Circle().fill(color).frame(width: 7, height: 7)
                }
                // Sim/real is PART of the connection pill (streamlined bar —
                // the separate target chip is gone).
                if controller.connection.isLive && controller.target != .unknown {
                    // The robot mark, drawn (RobotGlyph.swift). `.realRobot` used to
                    // borrow `gearshape.2.fill`, which reads as "settings" everywhere
                    // else — wrong for the control that says which robot you drive.
                    // The simulation keeps its cube: that one is not a robot.
                    if controller.target == .realRobot {
                        // Solid + eyes cut out when live, outlined when not; see the
                        // iOS twin. The eyes must read at every tint, which is why they
                        // are even-odd holes rather than a destinationOut punch.
                        RobotGlyph(size: 11, filled: controller.connection.isLive)
                            .foregroundStyle(controller.target.color)
                            .shadow(color: controller.connection.isLive
                                    ? controller.target.color.opacity(0.55) : .clear,
                                    radius: 3)
                    } else {
                        Image(icon: controller.target.icon).font(.system(size: 11, weight: .bold))
                            .foregroundStyle(controller.target.color)
                    }
                }
                Text(switcherLabel).font(.mabelCaption).foregroundStyle(Theme.Palette.textSecondary).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 9, weight: .heavy))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(.horizontal, 11).padding(.vertical, 6)
            .background(color.opacity(0.12), in: Capsule())
            .overlay { Capsule().strokeBorder(color.opacity(showPicker ? 0.55 : 0.28), lineWidth: 1) }
            .contentShape(Capsule())
        }
        .accessibilityIdentifier("status.robot")
        .buttonStyle(PressableStyle())
        .popover(isPresented: $showPicker, arrowEdge: .bottom) { robotPicker }
    }

    /// When live, name the connected robot; otherwise the connection state.
    private var switcherLabel: String {
        if controller.connectedToRealServer, let r = controller.robotStore.active { return r.name }
        return controller.connection.label
    }

    /// The compact liquid-glass dropdown: pick a robot, run the demo, or disconnect.
    private var robotPicker: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("ROBOTS").font(.system(size: 9, weight: .heavy)).tracking(1.2)
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.horizontal, 14).padding(.top, 10).padding(.bottom, 4)
            ForEach(controller.robotStore.robots) { pickerRow($0) }
            Divider().overlay(Theme.Palette.hairlineSoft).padding(.horizontal, 10).padding(.vertical, 2)
            demoRow
            if controller.connectedToRealServer { disconnectRow }
        }
        .frame(width: 280)
        .padding(.bottom, 8)
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
                if live { Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.Palette.live) }
                else if active && controller.connection == .connecting { ProgressView().controlSize(.mini) }
                else { Circle().fill(robot.online ? Theme.Palette.live : Theme.Palette.textTertiary.opacity(0.4)).frame(width: 7, height: 7) }
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(active ? Theme.Palette.accent.opacity(0.10) : .clear)
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    private var demoRow: some View {
        let active = controller.connection == .demo
        return Button {
            Haptics.tap(); controller.connect(host: "demo"); showPicker = false
        } label: {
            HStack(spacing: 10) {
                Image(systemName: RouteKind.demo.icon).font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(active ? Theme.Palette.live : Theme.Palette.accent).frame(width: 22)
                Text("On-device demo").font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
                Spacer()
                if active { Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.Palette.live) }
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(active ? Theme.Palette.accent.opacity(0.10) : .clear)
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    private var disconnectRow: some View {
        Button {
            Haptics.tap(); controller.disconnect(); showPicker = false
        } label: {
            Label("Disconnect", systemImage: "xmark.circle")
                .font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.danger)
                .padding(.horizontal, 14).padding(.vertical, 9)
                .frame(maxWidth: .infinity, alignment: .leading)
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
        #if DEBUG
        .task {
            // Screenshot harness: `defaults write <bundle> debugPopup -string transport`.
            guard UserDefaults.standard.string(forKey: "debugPopup") == "transport" else { return }
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            showTransport = true
        }
        #endif
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

    // MARK: - Compact vitals summary (tap → detail popover)

    private var summaryButton: some View {
        Button {
            Haptics.select()
            showDetail.toggle()
        } label: {
            HStack(spacing: 7) {
                // A SIM has no battery/CPU to report — show latency ONLY there (the
                // one live sim number); a real robot keeps battery + latency.
                if !controller.target.isSimulation {
                    Image(systemName: "bolt.fill").font(.system(size: 11, weight: .bold)).foregroundStyle(batteryTint)
                    Text("\(Int(controller.state.batteryLevel * 100))%").font(.mabelMonoSmall).foregroundStyle(Theme.Palette.textPrimary)
                    Rectangle().fill(Theme.Palette.hairlineSoft).frame(width: 1, height: 12)
                }
                Image(systemName: "bolt.horizontal").font(.system(size: 11, weight: .bold)).foregroundStyle(Theme.Palette.warn)
                Text("\(Int(controller.state.latencyMs)) ms").font(.mabelMonoSmall).foregroundStyle(Theme.Palette.textPrimary)
                Image(systemName: showDetail ? "chevron.up" : "chevron.down")
                    .font(.system(size: 10, weight: .bold)).foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(.horizontal, 11).padding(.vertical, 7)
            .background(Theme.Palette.surface, in: Capsule())
            .overlay { Capsule().strokeBorder(showDetail ? Theme.Palette.accent.opacity(0.5) : Theme.Palette.hairlineSoft, lineWidth: 1) }
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("status.vitals")
        #if DEBUG
        .task {
            // Screenshot harness: `defaults write <bundle> debugPopup -string status`.
            guard UserDefaults.standard.string(forKey: "debugPopup") == "status" else { return }
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            showDetail = true
        }
        #endif
        .popover(isPresented: $showDetail, arrowEdge: .bottom) {
            VStack(spacing: 0) {
                StatusDetailPanel()
                Divider().overlay(Theme.Palette.hairlineSoft).padding(.horizontal, 12)
                // Speed governor moved OUT of the bar (streamlined) — it lives
                // here with the rest of the robot vitals.
                HStack(spacing: 8) {
                    Image(systemName: "gauge.with.dots.needle.50percent")
                        .font(.system(size: 12)).foregroundStyle(Theme.Palette.textTertiary)
                    Text("SPEED").font(.system(size: 9, weight: .heavy)).tracking(1.1)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Slider(value: $governor, in: 0.1...1)
                        .tint(Theme.Palette.accent)
                        .onChange(of: governor) { _, v in controller.speedScale = v }
                    Text("\(Int(governor * 100))%")
                        .font(.mabelMonoSmall).foregroundStyle(Theme.Palette.accent)
                        .frame(width: 38, alignment: .trailing)
                }
                .padding(.horizontal, 14).padding(.vertical, 10)
            }
        }
    }

    // MARK: - Actions

    /// Simplified sim reset (iOS parity): TAP = respawn at a server-validated
    /// RANDOM open pose; long-press menu still offers Reset to Home.
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
                // iOS parity: the RESET pill wears a reset glyph (not the dice —
                // that stays on the "Random Position" menu row only).
                Image(systemName: "arrow.counterclockwise").font(.system(size: 13, weight: .bold))
                Text("RESET").font(.mabel(13, weight: .heavy)).tracking(0.5).lineLimit(1).fixedSize()
            }
            .foregroundStyle(Theme.Palette.textPrimary)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            .contentShape(Capsule())
        } primaryAction: {
            controller.resetSimRandom()
        }
        .buttonStyle(PressableStyle())
    }

    /// The world switcher (sim only, SCC studio running beside the sim).
    @ViewBuilder private var sceneButton: some View {
        // Shown for ANY sim (the parent gates on canReset), not only when the SCC
        // probe already confirmed — the button was vanishing whenever the scene
        // backend wasn't detected yet. Tapping re-probes; the sheet shows the worlds
        // or a clear hint when no scene backend is running.
        Button {
            Haptics.select(); controller.simStudio.reprobe(); showScenes = true
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "square.stack.3d.up").font(.system(size: 13, weight: .bold))
                Text("WORLD").font(.mabel(13, weight: .heavy)).tracking(0.5).lineLimit(1).fixedSize()
            }
            .foregroundStyle(Theme.Palette.textPrimary)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
        .sheet(isPresented: $showScenes) { SceneLibrarySheet(studio: controller.simStudio) }
    }

    /// Glide the robot to the minimal-torque REST pose — arms hanging down, torso &
    /// neck neutral, wheels forward, hands relaxed. A POSE home (works on sim + the
    /// real robot), distinct from the sim RESET that respawns the whole world.
    private var homeButton: some View {
        Button {
            Haptics.rigid(); controller.goRestPose()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "house.fill").font(.system(size: 13, weight: .bold))
                    .symbolEffect(.pulse, isActive: controller.homing)
                Text("HOME").font(.mabel(13, weight: .heavy)).tracking(0.5).lineLimit(1).fixedSize()
            }
            .foregroundStyle(controller.homing ? Theme.Palette.accent : Theme.Palette.textPrimary)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay {
                Capsule().strokeBorder(controller.homing ? Theme.Palette.accent.opacity(0.6)
                                                          : Theme.Palette.hairlineSoft, lineWidth: 1)
            }
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
    }

    /// The e-stop, named for what it DOES to the motors (iOS parity, 2026-07-17).
    /// "STOP"/"RESUME" described the robot's MOTION — the harmless-sounding half.
    /// On the wire this cuts and restores MOTOR TORQUE: "Resume" reads like
    /// un-pausing a video and gives no hint it re-energises every joint on a robot
    /// someone may have their hands inside. TORQUE ON says the limbs go live;
    /// E-STOP says this is the emergency control, not a pause.
    private var stopButton: some View {
        Button {
            // While a torque-on is pending the button is already an E-STOP: the
            // server may energise at any moment, and stop must always be one tap.
            let canResume = controller.estopEngaged && !controller.torqueOnRequested
            if canResume { Haptics.success(); controller.releaseEStop() }
            else { Haptics.error(); controller.engageEStop() }
        } label: {
            let canResume = controller.estopEngaged && !controller.torqueOnRequested
            HStack(spacing: 6) {
                Image(systemName: canResume ? "bolt.fill" : "exclamationmark.octagon.fill")
                    .font(.system(size: 13, weight: .bold))
                Text(canResume ? "TORQUE ON" : "E-STOP")
                    .font(.mabel(13, weight: .heavy)).tracking(0.5).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 16).padding(.vertical, 8)
            .background(((controller.estopEngaged && !controller.torqueOnRequested) ? Theme.Palette.live : Theme.Palette.danger).gradient, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
    }

    /// System-wide REC toggle: idle → red dot; recording → filled red pill with the
    /// elapsed task. Starts into the current `recordTask` (the Robot space's folder
    /// picker chooses a different one); stop finalises the episode.
    private var recordButton: some View {
        Button {
            if controller.recording { Haptics.rigid(); controller.endEpisode() }
            else { Haptics.rigid(); controller.startEpisode() }
        } label: {
            HStack(spacing: 6) {
                Circle().fill(controller.recording ? Color.white : Theme.Palette.danger)
                    .frame(width: 9, height: 9)
                Text(controller.recording ? "REC · \(controller.recordTask)" : "Rec")
                    .font(.mabel(13, weight: .heavy)).tracking(0.4).lineLimit(1).fixedSize()
            }
            .foregroundStyle(controller.recording ? .white : Theme.Palette.textPrimary)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background {
                if controller.recording { Capsule().fill(Theme.Palette.danger) }
                else { Capsule().fill(Theme.Palette.surface) }
            }
            .overlay { Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
    }

    /// Local-state slider: writes through to the controller without publishing,
    /// so dragging is buttery even while telemetry streams.
    private var batteryTint: Color {
        let l = controller.state.batteryLevel
        return l > 0.4 ? Theme.Palette.live : (l > 0.18 ? Theme.Palette.warn : Theme.Palette.danger)
    }
}
