//
//  RobotSettingsView.swift
//  MabelTeleop
//
//  The "Robot" settings hub — the SAME cards as the iOS app (status,
//  connection with live endpoints, data collection, permissions,
//  performance, preferences, about), adapted to the headset: ARKit hand /
//  world-sensing permissions replace the gyro, and appearance/haptics are
//  omitted (visionOS is glass; there is no haptic engine).
//

import SwiftUI
import UIKit

struct RobotSettingsView: View {
    @EnvironmentObject private var controller: RobotController
    @EnvironmentObject private var discovery: ServerDiscovery
    @AppStorage("autoConnect") private var autoConnect = true
    @State private var speedCap: Double = 1.0
    // Add-robot sheet + host inspector (tap a robot for details + connect).
    @State private var showAdd = false
    @State private var addName = ""
    @State private var addHost = ""
    @State private var addSSH = ""
    @State private var inspecting: Robot?
    // Real-robot access code entry (stored in the Keychain, never the repo).
    @State private var codeField = ""
    @State private var editingCode = false
    // Anthropic assistant key entry (Keychain) + model choice (the chat LLM).
    @State private var llmKeyField = ""
    @State private var editingLLMKey = false
    @AppStorage(ClaudeClient.Model.storageKey) private var llmModelRaw = ClaudeClient.Model.opus48.rawValue
    // Server-synced speed caps (pulled from serverMeta into the sliders).
    @State private var capLin: Double = 0.6
    @State private var capAng: Double = 1.2
    @AppStorage("settings.lastCapLin") private var lastCapLin: Double = 0.6   // last robot limits seen, shown greyed offline
    @AppStorage("settings.lastCapAng") private var lastCapAng: Double = 1.2
    // Body-calibration: guided-flow sheet + saved-profile inspector.
    @State private var showCalibration = false
    @State private var calibrationPreset = ""
    @State private var inspectingProfile: CalibrationProfile?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                header
                HStack(alignment: .top, spacing: 12) {
                    VStack(spacing: 12) { statusCard; connectionCard; BluetoothCard(); securityCard; speedCapsCard }.frame(maxWidth: .infinity, alignment: .top)
                    VStack(spacing: 12) { assistantCard; dataCollectionCard; permissionsCard; performanceCard }.frame(maxWidth: .infinity, alignment: .top)
                    VStack(spacing: 12) { calibrationCard; aboutCard }.frame(maxWidth: .infinity, alignment: .top)
                }
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, 52)
            .padding(.bottom, 60)
        }
        .scrollIndicators(.hidden)
        .consoleSurface()
        .onAppear {
            discovery.start()
            controller.refreshCollectionStatus()
            speedCap = controller.speedScale
            syncCaps()
        }
        .onChange(of: controller.serverMeta) { _, _ in syncCaps() }
    }

    /// Pull the active robot's server-reported caps into the sliders.
    private func syncCaps() {
        if let c = controller.speedCaps { capLin = c.vmax; capAng = c.wmax; lastCapLin = c.vmax; lastCapAng = c.wmax }
        else { capLin = lastCapLin; capAng = lastCapAng }            // last values seen, shown greyed
    }

    private var header: some View {
        HStack {
            Text("Settings").font(.mabelViewTitle).foregroundStyle(Theme.Palette.textPrimary)
            Spacer()
            StatusPill(title: controller.connection.label, color: controller.connection.color, pulses: true)
        }
    }

    // MARK: - Live status (derived from real state)

    private var statusCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("STATUS")
                statusRow("Connection", controller.connection.label, controller.connection.color,
                          icon: "antenna.radiowaves.left.and.right")
                if controller.connection.isLive {
                    // HOW we're connected (Wi-Fi LAN / Tailscale / Bluetooth), server-
                    // authoritative when reported, so the live transport is visible
                    // without opening a robot's sheet.
                    if let method = controller.connectionMethod {
                        statusRow("Method", method.rawValue, Theme.Palette.live, icon: method.icon)
                    }
                    statusRow("Driving", controller.target == .realRobot ? "Real robot" : "Simulation",
                              controller.target.color, icon: controller.target.icon)
                    statusRow("Latency", String(format: "%.0f ms", controller.state.latencyMs),
                              latencyTint, icon: "bolt.horizontal")
                }
                statusRow("Control mode", controller.controlSummary,
                          Theme.Palette.textSecondary, icon: "slider.horizontal.3")
                statusRow("Head tracking", controller.headEnabled ? "On" : "Off",
                          controller.headEnabled ? Theme.Palette.live : Theme.Palette.textTertiary, icon: "arkit")
                statusRow("Body tracking", controller.bodyTracking ? "On" : "Off",
                          controller.bodyTracking ? Theme.Palette.live : Theme.Palette.textTertiary,
                          icon: "figure.arms.open")
                statusRow("E-stop", controller.estopEngaged ? "ENGAGED" : "Clear",
                          controller.estopEngaged ? Theme.Palette.danger : Theme.Palette.live,
                          icon: "exclamationmark.octagon.fill")
            }
        }
    }

    // MARK: - Connection / Robots (1:1 with the iOS app)

    private var connectionCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                HStack {
                    cardTitle("ROBOTS")
                    Spacer()
                    if controller.autoConnecting || discovery.searching {
                        HStack(spacing: 5) {
                            ProgressView().controlSize(.mini)
                            Text(controller.autoConnecting ? "connecting" : "scanning")
                                .font(.system(size: 9, weight: .medium)).foregroundStyle(Theme.Palette.textTertiary)
                        }
                    }
                }

                if let note = discovery.note { warnRow(note) }

                // Remembered + discovered robots — tap to view details / switch.
                VStack(spacing: 7) {
                    ForEach(controller.robotStore.robots) { robotRow($0) }
                }

                // Forgetting the sim / thor is otherwise a ONE-WAY DOOR: they are
                // never re-seeded, and re-adding the host by hand yields a
                // single-route entry with NO Tailscale — the robot then only
                // works on the LAN. This puts the full built-in entries (every
                // LAN IP + Tailscale) back in one tap.
                if controller.robotStore.hasForgottenBuiltIns {
                    VStack(alignment: .leading, spacing: 6) {
                        secButton(title: "Restore built-in robots",
                                  icon: "arrow.clockwise.circle.fill",
                                  tint: Theme.Palette.accent) {
                            controller.robotStore.restoreDefaults()
                        }
                        Text("Brings back the Simulation and MABEL Real (thor) with all their "
                             + "routes — Wi-Fi and Tailscale. A hand-added host only "
                             + "reaches the robot over the LAN.")
                            .font(.system(size: 9.5, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                HStack {
                    Button { addName = ""; addHost = ""; addSSH = ""; showAdd = true } label: {
                        Label("Add robot", systemImage: "plus.circle.fill")
                            .font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.Palette.accent)
                    }.buttonStyle(.plain)
                    Spacer()
                    // Always available — also CANCELS an in-progress connection
                    // attempt and stops the auto-connect retry loop (disconnect()
                    // clears autoWanted), not just a live link. Disabled only when
                    // there's truly nothing to stop.
                    let nothingToStop = controller.connection == .disconnected && !controller.autoConnecting
                    Button { Haptics.tap(); controller.disconnect() } label: {
                        Label("Disconnect", systemImage: "xmark.circle")
                            .font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.Palette.danger)
                    }
                    .buttonStyle(.plain)
                    .disabled(nothingToStop)
                    .opacity(nothingToStop ? 0.5 : 1)
                }
                Toggle(isOn: $autoConnect) {
                    Label("Auto-connect on launch", systemImage: "wifi")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                }.tint(Theme.Palette.accent)
            }
        }
        .sheet(isPresented: $showAdd) { addRobotSheet }
        .sheet(item: $inspecting) { RobotInfoSheet(robot: $0).environmentObject(controller) }
    }

    /// One robot — name, kind, host, online/active state. Tap → details+connect
    /// sheet; the SAME row as the iOS app's robot list.
    private func robotRow(_ robot: Robot) -> some View {
        let isActive = controller.robotStore.activeID == robot.id
        let live = isActive && controller.connection == .connected
        let connecting = isActive && controller.connection == .connecting
        // Colour by CONNECTION STATE, never by sim-vs-real: green when this host is
        // connected, amber while connecting, muted grey when it's just a saved /
        // offline robot. Sim vs real is conveyed by the glyph (cube vs gears) and
        // the SIM/REAL badge text — colour stays a pure status signal.
        let statusColor: Color = live ? Theme.Palette.live
            : (connecting ? Theme.Palette.warn : Theme.Palette.textTertiary)
        // The transport actually in use right now (LAN / Tailscale / Bluetooth), so the
        // row shows HOW we're connected, not just how the robot CAN be reached.
        // Server-authoritative when reported; else the app's route guess.
        let activeKind = live ? controller.connectionMethod : nil
        let shownHost = live ? (controller.activeRoute?.host ?? robot.displayHost) : robot.displayHost
        return Button {
            Haptics.tap(); inspecting = robot
        } label: {
            HStack(spacing: 11) {
                Image(icon: robot.kind.icon).font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(statusColor).frame(width: 24)
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(robot.name).font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                        Text(robot.kind.badge).font(.system(size: 8, weight: .heavy)).foregroundStyle(statusColor)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(statusColor.opacity(0.14), in: Capsule())
                    }
                    HStack(spacing: 5) {
                        Text(shownHost).font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
                        Text("·").foregroundStyle(Theme.Palette.textTertiary)
                        Text(robot.routeSummary).font(.system(size: 9, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                }
                Spacer()
                if live {
                    VStack(alignment: .trailing, spacing: 3) {
                        if let k = activeKind { transportChip(k) }
                        else { Text("CONNECTED").font(.system(size: 8, weight: .heavy))
                            .foregroundStyle(Theme.Palette.live).tracking(0.4) }
                        Text(String(format: "%.0f ms", controller.state.latencyMs))
                            .font(.system(size: 10, weight: .semibold)).monospacedDigit().foregroundStyle(latencyTint)
                    }
                } else if isActive && controller.connection == .connecting {
                    ProgressView().controlSize(.mini)
                } else {
                    Circle().fill(robot.online ? Theme.Palette.live : Theme.Palette.textTertiary.opacity(0.4))
                        .frame(width: 7, height: 7)
                }
            }
            .padding(.vertical, 7).padding(.horizontal, 9)
            .background(live ? Theme.Palette.live.opacity(0.08)
                             : (connecting ? Theme.Palette.warn.opacity(0.06) : Color.clear),
                        in: RoundedRectangle(cornerRadius: 11))
            .overlay(RoundedRectangle(cornerRadius: 11)
                .strokeBorder(live ? Theme.Palette.live.opacity(0.30)
                                   : (connecting ? Theme.Palette.warn.opacity(0.25) : Theme.Palette.hairlineSoft),
                              lineWidth: 1))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            if !NetworkConfig.seedRobots().contains(where: { $0.id == robot.id }) {
                Button(role: .destructive) { controller.robotStore.forget(id: robot.id) } label: {
                    Label("Forget", systemImage: "trash")
                }
            }
        }
    }

    private var addRobotSheet: some View {
        NavigationStack {
            Form {
                Section("Robot") {
                    TextField("Name", text: $addName)
                    TextField("Host / IP", text: $addHost)
                        .autocorrectionDisabled().textInputAutocapitalization(.never)
                    TextField("SSH (optional, e.g. user@10.0.0.5)", text: $addSSH)
                        .autocorrectionDisabled().textInputAutocapitalization(.never)
                }
                Section {
                    Text("Saved robots are remembered across launches, so you never have to re-find them.")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                }
            }
            .navigationTitle("Add robot")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { showAdd = false } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add & Connect") {
                        let r = controller.robotStore.add(host: addHost, name: addName,
                                                          ssh: addSSH.isEmpty ? nil : addSSH)
                        showAdd = false
                        controller.connect(to: r)
                    }
                    .disabled(addHost.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
        .frame(minWidth: 440, minHeight: 360)
    }

    /// A small live-transport chip — the route the link is actually using right
    /// now (Wi-Fi LAN / Tailscale / Bluetooth), shown on the connected robot's row.
    private func transportChip(_ kind: RouteKind) -> some View {
        HStack(spacing: 3) {
            Image(systemName: kind.icon).font(.system(size: 8, weight: .bold))
            Text(kind.tag).font(.system(size: 8, weight: .heavy)).tracking(0.3)
        }
        .foregroundStyle(Theme.Palette.live)
        .padding(.horizontal, 6).padding(.vertical, 2)
        .background(Theme.Palette.live.opacity(0.15), in: Capsule())
        .overlay(Capsule().strokeBorder(Theme.Palette.live.opacity(0.3), lineWidth: 0.5))
    }

    /// An inline warning row (discovery diagnosis: permission / client isolation).
    private func warnRow(_ text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 10, weight: .bold)).foregroundStyle(Theme.Palette.warn)
            Text(text).font(.system(size: 11)).foregroundStyle(Theme.Palette.warn)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Real-robot security (access code + driver lock + arm)

    /// Shown when a physical robot is known or has reported it enforces security.
    private var showSecurity: Bool {
        controller.robotStore.robots.contains { $0.kind == .real } || controller.securityEnforced
    }

    @ViewBuilder private var securityCard: some View {
        if showSecurity {
            GlassCard {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        cardTitle("REAL ROBOT SECURITY")
                        Spacer()
                        Image(systemName: controller.hasAccessCode ? "lock.fill" : "lock.open.fill")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(controller.hasAccessCode ? Theme.Palette.live : Theme.Palette.warn)
                    }

                    // 1) Access code — the secret that gates every transport.
                    if controller.hasAccessCode && !editingCode {
                        statusRow("Access code", "Stored on this device", Theme.Palette.live, icon: "key.fill")
                        HStack(spacing: 10) {
                            secButton(title: "Change", icon: "pencil", tint: Theme.Palette.accent) {
                                codeField = ""; editingCode = true
                            }
                            secButton(title: "Remove", icon: "trash", tint: Theme.Palette.danger) {
                                controller.setAccessCode(""); codeField = ""
                            }
                        }
                    } else {
                        Text("Enter the secret access code for the physical robot. It's stored only in this device's Keychain — never in the app or the cloud.")
                            .font(.system(size: 11)).foregroundStyle(Theme.Palette.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack(spacing: 10) {
                            SecureField("Access code", text: $codeField)
                                .textInputAutocapitalization(.never).autocorrectionDisabled()
                                .font(.system(size: 13, design: .monospaced))
                                .foregroundStyle(Theme.Palette.textPrimary)
                                .padding(.horizontal, 12).padding(.vertical, 9)
                                .background(Theme.Palette.surfaceHi, in: RoundedRectangle(cornerRadius: 11))
                                .overlay(RoundedRectangle(cornerRadius: 11).strokeBorder(Theme.Palette.hairline, lineWidth: 1))
                                .frame(maxWidth: .infinity)
                            secButton(title: "Save", icon: "checkmark", tint: Theme.Palette.accent) {
                                controller.setAccessCode(codeField); editingCode = false; codeField = ""
                                Haptics.success()
                            }
                            .disabled(codeField.trimmingCharacters(in: .whitespaces).isEmpty)
                        }
                    }

                    // 2) Live driver-lock + arm state (only when on the real robot).
                    if controller.connection.isLive && controller.target == .realRobot {
                        Divider().overlay(Theme.Palette.hairlineSoft)
                        statusRow("Driver", controller.isDriver ? "You are driving"
                                  : (controller.driverHolder.isEmpty ? "Observing (lock free)"
                                     : "Locked by \(controller.driverHolder)"),
                                  controller.isDriver ? Theme.Palette.live : Theme.Palette.textSecondary,
                                  icon: controller.isDriver ? "steeringwheel" : "eye")
                        statusRow("Armed", controller.armed ? "Yes — motion enabled"
                                  : "No — run mabel-arm on thor",
                                  controller.armed ? Theme.Palette.live : Theme.Palette.warn,
                                  icon: controller.armed ? "bolt.fill" : "bolt.slash.fill")
                        HStack {
                            if controller.isDriver {
                                secButton(title: "Give up control", icon: "hand.raised.slash",
                                          tint: Theme.Palette.warn) { controller.releaseControl() }
                            } else {
                                secButton(title: "Request control", icon: "hand.raised.fill",
                                          tint: Theme.Palette.accent) { controller.requestControl() }
                            }
                            Spacer()
                        }
                        if !controller.controlNotice.isEmpty {
                            Text(controller.controlNotice)
                                .font(.system(size: 10)).foregroundStyle(Theme.Palette.warn)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Assistant (Claude-backed chat in the Policies tab)

    private var assistantCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    cardTitle("ASSISTANT")
                    Spacer()
                    Image(systemName: SecureStore.hasUsableAnthropicKey ? "sparkles" : "sparkle")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(SecureStore.hasUsableAnthropicKey ? Theme.Palette.live : Theme.Palette.warn)
                }
                Text("Power the Policies chat with Claude using your own Anthropic API key — stored only in this device's Keychain, never in the app or the cloud.")
                    .font(.system(size: 11)).foregroundStyle(Theme.Palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)

                if SecureStore.hasAnthropicKey && !editingLLMKey {
                    statusRow("API key",
                              SecureStore.ownerModeActive ? "Owner mode active" : "Stored on this device",
                              SecureStore.hasUsableAnthropicKey ? Theme.Palette.live : Theme.Palette.warn,
                              icon: SecureStore.ownerModeActive ? "checkmark.seal.fill" : "key.fill")
                    HStack(spacing: 10) {
                        secButton(title: "Change", icon: "pencil", tint: Theme.Palette.accent) {
                            llmKeyField = ""; editingLLMKey = true
                        }
                        secButton(title: "Remove", icon: "trash", tint: Theme.Palette.danger) {
                            SecureStore.anthropicKey = ""; llmKeyField = ""
                        }
                    }
                } else {
                    HStack(spacing: 10) {
                        SecureField("sk-ant-…", text: $llmKeyField)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundStyle(Theme.Palette.textPrimary)
                            .padding(.horizontal, 12).padding(.vertical, 9)
                            .background(Theme.Palette.surfaceHi, in: RoundedRectangle(cornerRadius: 11))
                            .overlay(RoundedRectangle(cornerRadius: 11).strokeBorder(Theme.Palette.hairline, lineWidth: 1))
                            .frame(maxWidth: .infinity)
                        secButton(title: "Save", icon: "checkmark", tint: Theme.Palette.accent) {
                            SecureStore.anthropicKey = llmKeyField.trimmingCharacters(in: .whitespacesAndNewlines)
                            editingLLMKey = false; llmKeyField = ""; Haptics.success()
                        }
                        .disabled(llmKeyField.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }

                Divider().overlay(Theme.Palette.hairlineSoft)
                HStack {
                    Label("Model", systemImage: "cpu").font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    Picker("", selection: $llmModelRaw) {
                        ForEach(ClaudeClient.Model.allCases) { Text($0.label).tag($0.rawValue) }
                    }.tint(Theme.Palette.accent)
                }
            }
        }
    }

    /// Compact pill button for the security card (VP has no GlassButton).
    private func secButton(title: String, icon: String, tint: Color, action: @escaping () -> Void) -> some View {
        Button { Haptics.tap(); action() } label: {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.system(size: 12, weight: .semibold))
                Text(title).font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(tint)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(tint.opacity(0.12), in: Capsule())
            .overlay(Capsule().strokeBorder(tint.opacity(0.35), lineWidth: 1))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Speed limits (live, server-synced per robot)

    private var speedCapsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                cardTitle("SPEED")
                // ALWAYS VISIBLE (2026-10-25, "I do not see the speed control anymore") — see the iOS RobotView.
                HStack {
                    Label("Drive speed", systemImage: "gauge.with.dots.needle.50percent")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    Text("\(Int(speedCap * 100))%").font(.mabelMono).foregroundStyle(Theme.Palette.accent)
                }
                Slider(value: $speedCap, in: 0.1...1)
                    .tint(Theme.Palette.accent)
                    .onChange(of: speedCap) { _, v in controller.speedScale = v }
                Divider().overlay(Theme.Palette.hairlineSoft)
                let caps = controller.speedCaps
                capSlider("Robot max linear", value: $capLin, range: 0.1...(caps?.vmaxCeil ?? max(capLin, 1.04)),
                          unit: "m/s", icon: "arrow.up.and.down.circle").disabled(caps == nil).opacity(caps == nil ? 0.45 : 1)
                capSlider("Robot max angular", value: $capAng, range: 0.2...(caps?.wmaxCeil ?? max(capAng, 3.0)),
                          unit: "rad/s", icon: "arrow.clockwise.circle").disabled(caps == nil).opacity(caps == nil ? 0.45 : 1)
                Text(caps != nil
                     ? "Drive speed scales the sticks on every link. Robot max is the full-stick limit for \(controller.serverMeta?.name ?? "this robot"), applied live."
                     : (controller.overBluetooth
                        ? "Drive speed applies over Bluetooth too. The robot's max limits can be changed over Wi-Fi or Tailscale."
                        : "Drive speed applies as soon as you connect. Connect over Wi-Fi or Tailscale to change the robot's max limits."))
                    .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func capSlider(_ label: String, value: Binding<Double>, range: ClosedRange<Double>,
                           unit: String, icon: String) -> some View {
        VStack(spacing: 4) {
            HStack {
                Label(label, systemImage: icon).font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                Spacer()
                Text(String(format: "%.2f %@", value.wrappedValue, unit))
                    .font(.mabelMono).foregroundStyle(Theme.Palette.accent)
            }
            Slider(value: value, in: range) { editing in
                if !editing { controller.setSpeedCaps(vmax: capLin, wmax: capAng) }
            }.tint(Theme.Palette.accent)
        }
    }

    // MARK: - Data collection (episodes recorded for imitation learning)

    private var dataCollectionCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    cardTitle("DATA COLLECTION")
                    Spacer()
                    if controller.state.serverRecording {
                        HStack(spacing: 5) {
                            Circle().fill(Theme.Palette.danger).frame(width: 7, height: 7)
                            Text("RECORDING").font(.system(size: 9, weight: .heavy))
                                .foregroundStyle(Theme.Palette.danger).tracking(0.6)
                        }
                    }
                }

                if !controller.connection.isLive {
                    Text("Connect to a robot or sim server to see collected episodes.")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    HStack(spacing: 10) {
                        statTile("\(controller.episodesTotal)", "episodes", icon: "film.stack")
                        statTile("\(controller.serverTasks.count)", "tasks", icon: "folder")
                        statTile(controller.recordTask.isEmpty ? "—" : controller.recordTask,
                                 "active task", icon: "record.circle")
                    }

                    if controller.taskCounts.isEmpty {
                        Text("No episodes yet. Record one from the Teleop screen.")
                            .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                    } else {
                        Divider().overlay(Theme.Palette.hairlineSoft)
                        ForEach(sortedTasks, id: \.self) { task in
                            taskRow(task, controller.taskCounts[task] ?? 0)
                        }
                    }

                    if !controller.episodesDir.isEmpty {
                        Divider().overlay(Theme.Palette.hairlineSoft)
                        endpoint("Path", controller.episodesDir)
                    }
                }
            }
        }
    }

    private var sortedTasks: [String] {
        controller.taskCounts.keys.sorted {
            let a = controller.taskCounts[$0] ?? 0, b = controller.taskCounts[$1] ?? 0
            return a != b ? a > b : $0 < $1
        }
    }

    private func statTile(_ value: String, _ label: String, icon: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.accent)
            Text(value).font(.system(size: 16, weight: .heavy)).foregroundStyle(Theme.Palette.textPrimary)
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(label).font(.system(size: 9, weight: .medium)).foregroundStyle(Theme.Palette.textTertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Theme.Palette.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }

    private func taskRow(_ task: String, _ count: Int) -> some View {
        HStack(spacing: 10) {
            Image(systemName: task == controller.recordTask ? "folder.fill" : "folder")
                .font(.system(size: 13))
                .foregroundStyle(task == controller.recordTask ? Theme.Palette.live : Theme.Palette.textTertiary)
                .frame(width: 22)
            Text(task).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.textSecondary)
                .lineLimit(1)
            Spacer()
            Text("\(count)").font(.system(size: 13, weight: .heavy)).monospacedDigit()
                .foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, 8).padding(.vertical, 2)
                .background(Theme.Palette.hairlineSoft, in: Capsule())
        }
    }

    // MARK: - Permissions (the headset's equivalents of the iOS card)

    private var permissionsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("PERMISSIONS")
                statusRow("Local Network", "Required", Theme.Palette.accent, icon: "network")
                Text("Needed to reach the robot / sim server. If the app can't find a server, enable it in visionOS Settings.")
                    .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                statusRow("Hand tracking", HandTrackingService.isSupported ? "Available" : "Unavailable",
                          HandTrackingService.isSupported ? Theme.Palette.live : Theme.Palette.warn,
                          icon: "hand.raised.fingers.spread")
                Text(HandTrackingService.isSupported
                     ? "Used for body tracking — your hands and fingers drive the robot's arms and ORCA hands inside immersive teleop."
                     : "No hand tracking on this device (e.g. Simulator) — body tracking is unavailable; the spatial joysticks still drive the base.")
                    .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                statusRow("World sensing", HeadTrackingService.isSupported ? "Available" : "Unavailable",
                          HeadTrackingService.isSupported ? Theme.Palette.live : Theme.Palette.warn,
                          icon: "arkit")
                SecondaryButton(title: "Open visionOS Settings", systemImage: "gear") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
            }
        }
    }

    // MARK: - Performance

    private var performanceCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                cardTitle("PERFORMANCE")
                statusRow("Command rate", "30 Hz", Theme.Palette.textSecondary, icon: "arrow.up.arrow.down")
                statusRow("Hand tracking", "~90 Hz (ARKit)", Theme.Palette.textSecondary, icon: "hand.raised")
                statusRow("Video", controller.videoBase == nil ? "On-device (demo)" : "MJPEG stream",
                          Theme.Palette.textSecondary, icon: "video.fill")
                if let cam = controller.headCamera {
                    statusRow("Head camera", String(format: "fovy %.1f° · %.2f:1", cam.fovy, cam.aspect),
                              Theme.Palette.textSecondary, icon: "camera.aperture")
                }
            }
        }
    }

    // MARK: - Preferences

    // MARK: - Body calibration profiles

    private var calibrationCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                HStack {
                    cardTitle("CALIBRATION PROFILES")
                    Spacer()
                    Image(systemName: "figure.arms.open")
                        .font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.Palette.textTertiary)
                }
                if controller.calibrationProfiles.isEmpty {
                    VStack(spacing: 5) {
                        Text("No profiles yet").font(.mabelBody).foregroundStyle(Theme.Palette.textSecondary)
                        Text("One quick arms-forward pose scales MABEL's reach to match yours.")
                            .font(.mabelCaption).foregroundStyle(Theme.Palette.textTertiary)
                            .multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity).padding(.vertical, 6)
                } else {
                    ForEach(controller.calibrationProfiles) { profileRow($0) }
                }
                PrimaryButton(title: "New Calibration", systemImage: "plus") {
                    calibrationPreset = ""; showCalibration = true
                }
            }
        }
        .sheet(isPresented: $showCalibration) {
            CalibrationFlowView(presetName: calibrationPreset).environmentObject(controller)
        }
        .sheet(item: $inspectingProfile) { p in
            CalibrationProfileDetailView(profile: p) { name in
                calibrationPreset = name
                showCalibration = true
            }
            .environmentObject(controller)
        }
    }

    /// One saved profile — name, a one-line summary, active state. Tap → detail.
    private func profileRow(_ p: CalibrationProfile) -> some View {
        let active = controller.isActiveProfile(p)
        return Button { inspectingProfile = p } label: {
            HStack(spacing: 11) {
                Image(systemName: "figure.arms.open").font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(active ? Theme.Palette.live : Theme.Palette.accent).frame(width: 24)
                VStack(alignment: .leading, spacing: 1) {
                    Text(p.name).font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                    Text(p.summary).font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
                }
                Spacer()
                if active { StatusPill(title: "Active", color: Theme.Palette.live) }
                Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(.vertical, 7).padding(.horizontal, 9)
            .background(active ? Theme.Palette.live.opacity(0.08) : Theme.Palette.surface,
                        in: RoundedRectangle(cornerRadius: 11))
            .overlay {
                RoundedRectangle(cornerRadius: 11)
                    .strokeBorder(active ? Theme.Palette.live.opacity(0.25) : Theme.Palette.hairline, lineWidth: 1)
            }
        }
        .buttonStyle(.plain).hoverEffect()
    }

    // MARK: - About / device

    private var aboutCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("ABOUT")
                statusRow("App version", appVersion, Theme.Palette.textSecondary, icon: "app.badge")
                statusRow("Device", "Apple Vision Pro", Theme.Palette.textSecondary, icon: "visionpro")
                statusRow("visionOS", UIDevice.current.systemVersion, Theme.Palette.textSecondary, icon: "applelogo")
                statusRow("Protocol", "teleop_frame · ws :9090", Theme.Palette.textSecondary, icon: "point.3.connected.trianglepath.dotted")
            }
        }
    }

    // MARK: - Helpers

    private func cardTitle(_ t: String) -> some View { SectionLabel(t) }

    private func statusRow(_ label: String, _ value: String, _ tint: Color, icon: String) -> some View {
        StatusRow(label, value, tint: tint, icon: icon)
    }

    private func endpoint(_ label: String, _ url: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.system(size: 9, weight: .bold)).foregroundStyle(Theme.Palette.textTertiary).frame(width: 42, alignment: .leading)
            Text(url).font(.system(size: 11, design: .monospaced)).foregroundStyle(Theme.Palette.textSecondary)
                .lineLimit(1).truncationMode(.head)
            Spacer()
        }
    }


    private var latencyTint: Color {
        let l = controller.state.latencyMs
        return l < 60 ? Theme.Palette.live : (l < 150 ? Theme.Palette.warn : Theme.Palette.danger)
    }
    private var appVersion: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(v) (\(b))"
    }
}
