//
//  RobotView.swift
//  MABEL Mobile App
//
//  The "Settings" hub — real, useful controls: live status, connection (with
//  the actual endpoints it resolves), permissions, performance, preferences,
//  and device info. No placeholder telemetry (that lives in the status-bar
//  dropdown now).
//

import SwiftUI
import CoreMotion

struct RobotView: View {
    @EnvironmentObject private var controller: RobotController
    @EnvironmentObject private var discovery: ServerDiscovery
    @Environment(\.isLandscape) private var isLandscape
    @Environment(\.bottomBarInset) private var bottomBarInset
    @AppStorage("autoConnect") private var autoConnect = true
    @AppStorage("haptics.enabled") private var hapticsEnabled = true
    @AppStorage("appearance") private var appearanceRaw = Appearance.system.rawValue

    // Notifications
    @ObservedObject private var notifier = MabelNotifier.shared
    @AppStorage(MabelNotifier.masterKey) private var notifMaster = true
    @AppStorage("notif.safety") private var notifSafety = true
    @AppStorage("notif.connection") private var notifConnection = true
    @AppStorage("notif.battery") private var notifBattery = true
    @AppStorage("notif.tasks") private var notifTasks = true
    @AppStorage("notif.episodes") private var notifEpisodes = false

    // Robot management UI state.
    @State private var renaming: Robot? = nil
    @State private var renameText = ""
    @State private var showAdd = false
    @State private var addHost = ""
    @State private var addName = ""
    @State private var addSSH = ""
    // Local mirrors of the active robot's live speed caps (synced from the server).
    @State private var capLin: Double = 0.6
    @State private var capAng: Double = 1.2
    @State private var inspecting: Robot? = nil
    // Real-robot access code entry (stored in the Keychain, never the repo).
    @State private var codeField = ""
    @State private var editingCode = false
    // Anthropic assistant key entry (Keychain) + model choice (the chat LLM).
    @State private var llmKeyField = ""
    @State private var editingLLMKey = false
    @AppStorage(ClaudeClient.Model.storageKey) private var llmModelRaw = ClaudeClient.Model.opus48.rawValue

    var body: some View {
        // Pin the scroll content to EXACTLY the viewport width so no card can ever
        // make the page drift left/right (a vertical ScrollView otherwise lets
        // over-wide content pan horizontally).
        GeometryReader { geo in
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.md) {
                    header
                    if isLandscape {
                        // Two balanced columns (was three — too sparse / narrow).
                        HStack(alignment: .top, spacing: 12) {
                            VStack(spacing: 12) { statusCard; connectionCard; BluetoothCard(); securityCard; speedCapsCard; dataCollectionCard; notificationsCard }
                                .frame(maxWidth: .infinity, alignment: .top)
                            VStack(spacing: 12) { assistantCard; permissionsCard; performanceCard; preferencesCard; aboutCard }
                                .frame(maxWidth: .infinity, alignment: .top)
                        }
                    } else {
                        VStack(spacing: 12) {
                            statusCard; connectionCard; BluetoothCard(); securityCard; assistantCard; speedCapsCard; dataCollectionCard; notificationsCard; permissionsCard
                            performanceCard; preferencesCard; aboutCard
                        }
                    }
                }
                .padding(.horizontal, Theme.Space.lg).padding(.bottom, Theme.Space.lg)
                // Clear the floating nav bar so the last card never hides under the glass.
                .padding(.bottom, bottomBarInset)
                .frame(width: geo.size.width)
            }
            .scrollIndicators(.hidden)
        }
        .onAppear { discovery.start(); controller.refreshCollectionStatus(); syncCaps() }
        .onChange(of: controller.serverMeta) { _, _ in syncCaps() }
    }

    private var header: some View {
        ViewHeader(title: "Settings") {
            StatusPill(title: controller.connection.label, color: controller.connection.color, pulses: true)
        }
        .padding(.top, Theme.Space.sm)
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
                            GlassButton(title: "Change", systemImage: "pencil",
                                        tint: Theme.Palette.accent, fullWidth: true) {
                                codeField = ""; editingCode = true
                            }
                            GlassButton(title: "Remove", systemImage: "trash",
                                        tint: Theme.Palette.danger, fullWidth: true,
                                        role: .destructive) {
                                controller.setAccessCode(""); codeField = ""
                            }
                        }
                    } else {
                        Text("Enter the secret access code for the physical robot. It's stored only in this device's Keychain — never in the app or the cloud.")
                            .font(.system(size: 11)).foregroundStyle(Theme.Palette.textSecondary)
                        HStack(spacing: 10) {
                            GlassField {
                                SecureField("Access code", text: $codeField)
                                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                                    .font(.system(size: 13, design: .monospaced))
                                    .frame(maxWidth: .infinity)
                            }
                            GlassButton(title: "Save", systemImage: "checkmark",
                                        tint: Theme.Palette.accent) {
                                controller.setAccessCode(codeField); editingCode = false; codeField = ""
                                Haptics.success()
                            }
                            .disabled(codeField.trimmingCharacters(in: .whitespaces).isEmpty)
                        }
                    }

                    // 2) Live driver-lock + arm state (only when on the real robot).
                    if controller.connection.isLive && controller.target == .realRobot {
                        Divider().background(Theme.Palette.hairline)
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
                                GlassButton(title: "Give up control", systemImage: "hand.raised.slash",
                                            tint: Theme.Palette.warn) { controller.releaseControl() }
                            } else {
                                GlassButton(title: "Request control", systemImage: "hand.raised.fill",
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
                        .font(.system(size: 12, weight: .bold))
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
                        GlassButton(title: "Change", systemImage: "pencil",
                                    tint: Theme.Palette.accent, fullWidth: true) {
                            llmKeyField = ""; editingLLMKey = true
                        }
                        GlassButton(title: "Remove", systemImage: "trash",
                                    tint: Theme.Palette.danger, fullWidth: true, role: .destructive) {
                            SecureStore.anthropicKey = ""; llmKeyField = ""
                        }
                    }
                } else {
                    HStack(spacing: 10) {
                        GlassField {
                            SecureField("sk-ant-…", text: $llmKeyField)
                                .textInputAutocapitalization(.never).autocorrectionDisabled()
                                .font(.system(size: 13, design: .monospaced))
                                .frame(maxWidth: .infinity)
                        }
                        GlassButton(title: "Save", systemImage: "checkmark", tint: Theme.Palette.accent) {
                            SecureStore.anthropicKey = llmKeyField.trimmingCharacters(in: .whitespacesAndNewlines)
                            editingLLMKey = false; llmKeyField = ""; Haptics.success()
                        }
                        .disabled(llmKeyField.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }

                Divider().background(Theme.Palette.hairline)
                // The row shows the model NAME; the menu carries the descriptions.
                // A stock Picker renders the SELECTED label in the row, and
                // "Opus 4.8 · most capable" wrapped to two lines and collided with
                // the chevron. Same accent + chevron as before, one clean line —
                // iOS Settings' own pattern: pick in the sheet, summarise in the row.
                HStack {
                    Label("Model", systemImage: "cpu").font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                    Spacer(minLength: 8)
                    Menu {
                        Picker("", selection: $llmModelRaw) {
                            ForEach(ClaudeClient.Model.allCases) { m in
                                Text(m.label).tag(m.rawValue)     // room here: name · blurb
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text((ClaudeClient.Model(rawValue: llmModelRaw) ?? .opus48).name)
                                .font(.mabelBody)
                                .lineLimit(1)
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 11, weight: .semibold))
                        }
                        .foregroundStyle(Theme.Palette.accent)
                        .contentShape(Rectangle())
                    }
                }
            }
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
                    // Show HOW we're connected (Wi-Fi LAN / Tailscale / Bluetooth) inline,
                    // so the live transport is visible without opening a robot's sheet.
                    // Server-authoritative (it knows the real path), client guess as fallback.
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
                statusRow("E-stop", controller.estopEngaged ? "ENGAGED" : "Clear",
                          controller.estopEngaged ? Theme.Palette.danger : Theme.Palette.live,
                          icon: "exclamationmark.octagon.fill")
            }
        }
    }

    // MARK: - Connection / Robots

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

                // Remembered + discovered robots — tap to view/switch.
                VStack(spacing: 7) {
                    ForEach(controller.robotStore.robots) { robotRow($0) }
                }

                // Forgetting the sim / thor is otherwise a ONE-WAY DOOR: they are
                // never re-seeded, and re-adding the host by hand yields a
                // single-route entry with NO Tailscale — the robot then only works
                // on the LAN. This puts the full built-in entries (every LAN IP +
                // Tailscale) back in one tap.
                if controller.robotStore.hasForgottenBuiltIns {
                    VStack(alignment: .leading, spacing: 6) {
                        GlassButton(title: "Restore built-in robots",
                                    systemImage: "arrow.clockwise.circle.fill",
                                    tint: Theme.Palette.accent) {
                            controller.robotStore.restoreDefaults()
                        }
                        Text("Brings back the Simulation and MABEL Real (thor) with all their "
                             + "routes — Wi-Fi and Tailscale. A hand-added host only "
                             + "reaches the robot over the LAN.")
                            .font(.system(size: 9.5, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                }

                HStack(spacing: 10) {
                    GlassButton(title: "Add robot", systemImage: "plus.circle.fill",
                                tint: Theme.Palette.accent) {
                        addName = ""; addHost = ""; addSSH = ""; showAdd = true
                    }
                    Spacer()
                    // Always available — also CANCELS an in-progress connection
                    // attempt and stops the auto-connect retry loop (not just a
                    // live link). Disabled only when there's truly nothing to stop.
                    GlassButton(title: "Disconnect", systemImage: "xmark.circle",
                                tint: Theme.Palette.danger, role: .destructive) {
                        controller.disconnect()
                    }
                    .disabled(controller.connection == .disconnected && !controller.autoConnecting)
                    .opacity(controller.connection == .disconnected && !controller.autoConnecting ? 0.5 : 1)
                }
                Toggle(isOn: $autoConnect) {
                    Label("Auto-connect on launch", systemImage: "wifi")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                }.tint(Theme.Palette.accent)
            }
        }
        .alert("Rename robot", isPresented: renameAlert) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) { renaming = nil }
            Button("Save") {
                if let r = renaming { controller.renameRobot(id: r.id, to: renameText) }
                renaming = nil
            }
        } message: {
            Text("The name is saved on the robot and syncs to every client.")
        }
        .sheet(isPresented: $showAdd) { addRobotSheet }
        .sheet(item: $inspecting) { RobotInfoSheet(robot: $0).environmentObject(controller) }
    }

    /// One robot in the list — name, kind, host, online/active state. Tap to
    /// connect/switch; long-press to rename or forget.
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
        return HStack(spacing: 11) {
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
                // Host + reachable routes on a single line (truncates rather than
                // wrapping, so the row stays one line and never widens the page).
                Text("\(shownHost) · \(robot.routeSummary)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .lineLimit(1).truncationMode(.tail)
            }
            Spacer(minLength: 8)
            if live {
                // The live transport chip (Wi-Fi / Tailscale / Bluetooth). Latency moved
                // out to save horizontal space — it's in the Status card.
                if let k = activeKind { transportChip(k) }
                else { Text("CONNECTED").font(.system(size: 8, weight: .heavy))
                    .foregroundStyle(Theme.Palette.live).tracking(0.4) }
                // A quick way off this robot, right where you connected.
                Button { controller.disconnect() } label: {
                    Text("Disconnect").font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Theme.Palette.danger)
                        .padding(.horizontal, 11).padding(.vertical, 6)
                        .background(Theme.Palette.danger.opacity(0.12), in: Capsule())
                }
                .buttonStyle(PressableStyle())
            } else if isActive && controller.connection == .connecting {
                ProgressView().controlSize(.mini)
            } else {
                // A small, explicit Connect button beside each host.
                Button { Haptics.tap(); controller.connect(to: robot) } label: {
                    Text("Connect").font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 13).padding(.vertical, 6)
                        .background(Theme.accentGradient, in: Capsule())
                }
                .buttonStyle(PressableStyle())
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
        .onTapGesture { Haptics.tap(); inspecting = robot }   // row → details
        .contextMenu {
            Button { renameText = robot.name; renaming = robot } label: { Label("Rename", systemImage: "pencil") }
            Button(role: .destructive) { controller.robotStore.forget(id: robot.id) } label: {
                Label("Forget", systemImage: "trash")
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
                Section { Text("Saved robots are remembered across launches, so you never have to re-find them.")
                    .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary) }
            }
            .navigationTitle("Add robot")
            .navigationBarTitleDisplayMode(.inline)
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
        .presentationDetents([.medium])
    }

    private var renameAlert: Binding<Bool> {
        Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })
    }

    // MARK: - Speed limits (live, per real robot)

    private var speedCapsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                cardTitle("SPEED LIMITS")
                // (width pinned below so this card matches the others even when its
                //  content is just the short "connect to a robot" hint.)
                if let caps = controller.speedCaps {
                    capSlider("Linear", value: $capLin, range: 0.1...caps.vmaxCeil, unit: "m/s",
                              icon: "arrow.up.and.down.circle")
                    capSlider("Angular", value: $capAng, range: 0.2...caps.wmaxCeil, unit: "rad/s",
                              icon: "arrow.clockwise.circle")
                    Text("Full-stick speed for \(controller.serverMeta?.name ?? "this robot"). Applied live to the robot.")
                        .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Connect to a robot to adjust its maximum linear and angular speed.")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)   // match the other cards' width
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

    /// Pull the active robot's server-reported caps into the sliders.
    private func syncCaps() {
        if let c = controller.speedCaps { capLin = c.vmax; capAng = c.wmax }
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
                        .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    // Headline totals.
                    HStack(spacing: 10) {
                        statTile("\(controller.episodesTotal)", "episodes", icon: "film.stack")
                        statTile("\(controller.serverTasks.count)", "tasks", icon: "folder")
                        statTile(controller.recordTask.isEmpty ? "—" : controller.recordTask,
                                 "active task", icon: "record.circle")
                    }

                    if controller.taskCounts.isEmpty {
                        Text("No episodes yet. Record one from the Teleop screen.")
                            .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                    } else {
                        Divider().overlay(Theme.Palette.hairlineSoft)
                        // Per-task breakdown, biggest first.
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

    /// Tasks sorted by episode count (desc), then name — stable, useful ordering.
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

    // MARK: - Permissions

    private var permissionsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("PERMISSIONS")
                statusRow("Local Network", "Required", Theme.Palette.accent, icon: "network")
                Text("Needed to reach the robot / sim server. If the app can't find a server, enable it in iOS Settings.")
                    .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                statusRow("Motion (gyro)", controller.motionAvailable ? "Available" : "Unavailable",
                          controller.motionAvailable ? Theme.Palette.live : Theme.Palette.warn,
                          icon: "gyroscope")
                Text(controller.motionAvailable
                     ? "Used for head tracking — tilt the phone to aim the robot's head."
                     : "No motion sensor (e.g. Simulator) — head tracking falls back to drag-to-look.")
                    .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                SecondaryButton(title: "Open iOS Settings", systemImage: "gear") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
            }
        }
    }

    // MARK: - Notifications

    private var notificationsAuthorized: Bool {
        notifier.authorization == .authorized || notifier.authorization == .provisional
    }

    private var notificationsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("NOTIFICATIONS")
                statusRow("Alerts", notifAuthText,
                          notificationsAuthorized ? Theme.Palette.live : Theme.Palette.warn,
                          icon: notificationsAuthorized ? "bell.fill" : "bell.slash.fill")

                if !notificationsAuthorized {
                    Text("Let MABEL alert you to faults, a lost link, low battery and finished tasks — even when the app is closed.")
                        .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                    SecondaryButton(title: "Enable Notifications", systemImage: "bell.badge") {
                        Task {
                            await notifier.requestAuthorization()
                            if notifier.authorization == .denied,
                               let url = URL(string: UIApplication.openSettingsURLString) {
                                await UIApplication.shared.open(url)
                            }
                        }
                    }
                }

                Divider().overlay(Theme.Palette.hairlineSoft)
                Toggle(isOn: $notifMaster) {
                    Label("Robot Alerts", systemImage: "bell.fill")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                }.tint(Theme.Palette.accent)

                if notifMaster {
                    catToggle(.safety, $notifSafety)
                    catToggle(.connection, $notifConnection)
                    catToggle(.battery, $notifBattery)
                    catToggle(.tasks, $notifTasks)
                    catToggle(.episodes, $notifEpisodes)
                    Text("Safety & critical alerts are time-sensitive and break through Focus. You're never alerted while you're actively using the app. Delivery while the phone is asleep needs the robot server's push service.")
                        .font(.system(size: 9.5)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var notifAuthText: String {
        switch notifier.authorization {
        case .authorized:    return "On"
        case .provisional:   return "Quiet delivery"
        case .denied:        return "Off — enable in iOS Settings"
        case .notDetermined: return "Not set up"
        case .ephemeral:     return "Limited"
        @unknown default:    return "—"
        }
    }

    private func catToggle(_ c: NotifCategory, _ binding: Binding<Bool>) -> some View {
        Toggle(isOn: binding) {
            VStack(alignment: .leading, spacing: 1) {
                Label(c.title, systemImage: c.icon)
                    .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                Text(c.blurb).font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .tint(Theme.Palette.accent)
        .disabled(!notificationsAuthorized)
    }

    // MARK: - Performance

    private var performanceCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                cardTitle("PERFORMANCE")
                HStack {
                    Label("Head sensitivity", systemImage: "dial.medium")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    Text("\(Int(controller.headSensitivity * 100))%").font(.mabelMono).foregroundStyle(Theme.Palette.accent)
                }
                Slider(value: $controller.headSensitivity, in: 0.3...1.2).tint(Theme.Palette.accent)
                Divider().overlay(Theme.Palette.hairlineSoft)
                statusRow("Command rate", "20 Hz", Theme.Palette.textSecondary, icon: "arrow.up.arrow.down")
                statusRow("Video", controller.videoBase == nil ? "On-device (demo)" : "MJPEG stream",
                          Theme.Palette.textSecondary, icon: "video.fill")
            }
        }
    }

    // MARK: - Preferences

    private var preferencesCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                cardTitle("PREFERENCES")
                HStack {
                    Label("Appearance", systemImage: "circle.lefthalf.filled")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    Picker("", selection: $appearanceRaw) {
                        ForEach(Appearance.allCases) { Text($0.label).tag($0.rawValue) }
                    }.pickerStyle(.segmented).frame(width: 190)
                }
                Divider().overlay(Theme.Palette.hairlineSoft)
                Toggle(isOn: $hapticsEnabled) {
                    Label("Haptic Feedback", systemImage: "hand.tap.fill").font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                }.tint(Theme.Palette.accent)
                Divider().overlay(Theme.Palette.hairlineSoft)
                HStack {
                    Label("Default Speed Cap", systemImage: "gauge.with.dots.needle.50percent")
                        .font(.mabelBody).foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    Text("\(Int(controller.speedScale * 100))%").font(.mabelMono).foregroundStyle(Theme.Palette.accent)
                }
                Slider(value: $controller.speedScale, in: 0.1...1).tint(Theme.Palette.accent)
            }
        }
    }

    // MARK: - About / device

    private var aboutCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("ABOUT")
                statusRow("App version", appVersion, Theme.Palette.textSecondary, icon: "app.badge")
                statusRow("Device", UIDevice.current.model, Theme.Palette.textSecondary, icon: "iphone")
                statusRow("iOS", UIDevice.current.systemVersion, Theme.Palette.textSecondary, icon: "applelogo")
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
            Text(url).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.Palette.textSecondary).lineLimit(1)
            Spacer()
        }
    }

    private func field(_ label: String, text: Binding<String>, icon: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).foregroundStyle(Theme.Palette.accent).frame(width: 24)
            Text(label).font(.mabelBody).foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            TextField("", text: text)
                .multilineTextAlignment(.trailing)
                .font(.mabelMono).foregroundStyle(Theme.Palette.textPrimary)
                .autocorrectionDisabled().textInputAutocapitalization(.never)
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
