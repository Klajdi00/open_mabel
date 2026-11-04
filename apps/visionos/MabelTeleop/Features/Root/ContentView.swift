//
//  ContentView.swift
//  MabelTeleop
//
//  Root of the main window — V2 has three destinations (Robot · Autonomy ·
//  Settings), mirroring MABEL iOS v2, with the global chrome as glass
//  ornaments: the status bar above (connection, SIM/REAL, vitals popover,
//  governor, Reset, STOP) and the bottom navigation bar below. Ornaments
//  render on the window plane, and all in-window 3D content spawns BEHIND
//  it, so the bars are never obstructed and stay closest to the user.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var controller: RobotController
    @EnvironmentObject private var discovery: ServerDiscovery
    @Environment(\.dismissImmersiveSpace) private var dismissImmersive
    @Environment(\.openImmersiveSpace) private var openImmersive
    @Environment(\.openWindow) private var openWindow
    @AppStorage("autoConnect") private var autoConnect = true
    @State private var selection: AppSection = .robot
    @State private var didAutoConnect = false
    @State private var showLaunch = true

    var body: some View {
        ZStack {
            if controller.immersiveActive && controller.fullImmersion {
                // Full immersion owns the experience: the window collapses to
                // a whisper (no camera tiles "stitched" behind the wrap, no
                // nav bars) until the operator exits.
                immersivePlaceholder
            } else {
                content
            }
            if showLaunch {
                LaunchView()
                    .transition(.opacity)
                    .zIndex(10)
            }
        }
        .frame(minWidth: 1100, minHeight: 700)
        .animation(Theme.Motion.smooth, value: controller.fullImmersion)
        .ornament(attachmentAnchor: .scene(.top), contentAlignment: .bottom) {
            if !(controller.immersiveActive && controller.fullImmersion) {
                GlobalStatusBar()
                    .padding(.bottom, 12)
                    .opacity(showLaunch ? 0 : 1)
            }
        }
        .ornament(attachmentAnchor: .scene(.leading), contentAlignment: .trailing) {
            // Navigation hides in FULL immersion — the floating teleop
            // console is the persistent control surface there. On the Robot
            // tab, RobotSpaceView renders the combined two-column rail (nav
            // leftmost + viewer buttons) so the columns sit side by side.
            if !(controller.immersiveActive && controller.fullImmersion),
               selection != .robot {
                LeftNavRail(selection: $selection)
                    .padding(.trailing, 12)
                    .opacity(showLaunch ? 0 : 1)
            }
        }
        // Walk up to the robot and it offers itself — see ProximityConnectCard. At the
        // ROOT so the panel can float over any tab.
        .proximityConnect(controller, discovery: discovery)
        .onAppear {
            #if DEBUG
            if let i = CommandLine.arguments.firstIndex(of: "-startTab"),
               i + 1 < CommandLine.arguments.count {
                switch CommandLine.arguments[i + 1] {
                case "policies": selection = .policies
                case "settings": selection = .settings
                default:         selection = .robot
                }
                showLaunch = false
            } else if let t = UserDefaults.standard.string(forKey: "debugTab"), !t.isEmpty {
                // Defaults twin of -startTab (argv is dropped on prewarmed launches).
                switch t {
                case "policies": selection = .policies
                case "settings": selection = .settings
                default:         selection = .robot
                }
                showLaunch = false
            }
            #endif
            controller.connect()        // start in demo so the app is usable instantly
            #if DEBUG
            // Deterministic test harness (mirrors iOS v2): `-connectHost <host>`
            // argv or the `debugConnectHost` user default pins the link to one
            // host ("demo" = on-device sim) and skips the auto-connect race.
            // Launch argv is silently dropped on prewarmed launches — the
            // defaults channel is the reliable one.
            var pinnedHost: String? = nil
            if let i = CommandLine.arguments.firstIndex(of: "-connectHost"),
               i + 1 < CommandLine.arguments.count {
                pinnedHost = CommandLine.arguments[i + 1]
            } else if let h = UserDefaults.standard.string(forKey: "debugConnectHost"), !h.isEmpty {
                pinnedHost = h
            }
            if let pinnedHost {
                didAutoConnect = true
                controller.connect(host: pinnedHost)
            }
            // Screenshot harness: `debugCameraDetail=<stream key>` opens the
            // camera-detail window without a tile tap (synthetic taps can't be
            // driven on the visionOS Simulator).
            if let key = UserDefaults.standard.string(forKey: "debugCameraDetail"), !key.isEmpty {
                Task {
                    try? await Task.sleep(for: .seconds(3))
                    openWindow(id: WindowID.cameraDetail.rawValue, value: key)
                }
            }
            // Rig-switch harness: `debugSwitchRig=YES` drives the SIM → REAL →
            // SIM sequence an operator performs from the status-bar picker, and
            // logs the resolved tunnel at each step. Synthetic taps can't be
            // driven on the visionOS Simulator, so this is the only way to
            // exercise the switch end-to-end there. The round TRIP is the point:
            // the failure that actually strands an operator is a real robot that
            // is offline (thor asleep, away from its network) leaving the app unable to get
            // back to the sim without a relaunch.
            if UserDefaults.standard.bool(forKey: "debugSwitchRig") {
                didAutoConnect = true
                Task { await runRigSwitchHarness() }
            }
            #endif
            discovery.start()           // search the LAN for a real server
            // Race the network and link up automatically: fastest LAN/Tailscale
            // first — same as iOS / web.
            if autoConnect, !didAutoConnect {
                didAutoConnect = true
                controller.connectAuto(discovered: discovery.found)
            }
            Task {
                try? await Task.sleep(for: .seconds(2.2))
                withAnimation(.easeOut(duration: 0.6)) { showLaunch = false }
            }
        }
        .onChange(of: selection) { _, tab in
            // Leaving Teleop ends the session: the space (joysticks, hand
            // tracking, wrap view) closes and the robot HOLDS its last
            // targets — the bridge freezes the pose when the spec changes.
            if tab != .robot && controller.immersiveActive {
                controller.noteIntendedImmersiveExit()
                Task { await dismissImmersive() }
            }
        }
        .onChange(of: controller.streamAlways) { _, _ in openLabTrackingIfNeeded() }
        .onChange(of: controller.immersiveActive) { _, _ in openLabTrackingIfNeeded() }
        // The SYSTEM closed the space under a live whole-body session (Digital
        // Crown…). Teleop must not end on anything but Stop (operator,
        // 2026-10-16): the robot is holding; re-open the tracking space hidden
        // behind the window and the session re-engages (re-anchored at the
        // hands' current pose, so nothing jumps).
        .onChange(of: controller.resumeTrackingRequested) { _, want in
            guard want else { return }
            controller.resumeTrackingRequested = false
            controller.trackingOnlyMode = true
            controller.fullImmersion = false
            Task {
                try? await Task.sleep(for: .milliseconds(400))   // let the teardown settle
                if controller.needsImmersiveOpen() {
                    _ = await openImmersive(id: WindowID.immersive.rawValue)
                }
            }
        }
        .onChange(of: discovery.servers) { _, servers in
            // Feed fresh Bonjour hits into the planner — a newly-appeared LAN
            // host can win the race (and upgrade us off Tailscale).
            guard autoConnect else { return }
            controller.noteDiscovered(servers.map { .init(name: $0.name, host: $0.host) })
        }
    }

    /// Connected to the Mac Retarget Lab: open the invisible tracking space so head +
    /// hands stream for as long as the link lasts (and again if the space closes while
    /// the lab is still connected — e.g. the operator left and came back to Robot).
    private func openLabTrackingIfNeeded() {
        guard controller.streamAlways, selection == .robot else { return }
        if controller.prepareLabTracking() {
            Task { _ = await openImmersive(id: WindowID.immersive.rawValue) }
        }
    }

    /// Shown while the immersive teleop space is open — fades away after a
    /// few seconds so no floating text lingers in front of the operator.
    private var immersivePlaceholder: some View {
        ImmersivePlaceholder()
    }

    @ViewBuilder private var content: some View {
        // The Robot space NEVER unmounts — its RealityKit rig, SceneKit scene
        // and map/video streams stay warm, so returning to the tab (and every
        // POV flip) is instant with zero loading. Other tabs overlay it.
        ZStack {
            RobotSpaceView(navSelection: $selection)
                .opacity(selection == .robot ? 1 : 0)
                .allowsHitTesting(selection == .robot)
            if selection == .policies {
                PoliciesView().background(Theme.Palette.canvas)
            } else if selection == .settings {
                RobotSettingsView().background(Theme.Palette.canvas)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    #if DEBUG
    /// SIM → REAL → SIM, the way the status-bar picker drives it, with the
    /// resolved tunnel logged at each step. Read the output with:
    ///   xcrun simctl launch --console-pty <UD> <bundle> | grep MABEL-SWITCH
    private func runRigSwitchHarness() async {
        func report(_ step: String) {
            let r = controller.robotStore.active
            print("""
            [MABEL-SWITCH] \(step) \
            active=\(r?.id ?? "nil") kind=\(r.map { "\($0.kind)" } ?? "nil") \
            status=\(controller.connection) \
            videoBase=\(controller.videoBase?.absoluteString ?? "nil") \
            head=\(controller.videoURL("main")?.absoluteString ?? "nil")
            """)
        }
        let store = controller.robotStore
        guard let sim = store.robots.first(where: { $0.kind == .sim }),
              let real = store.robots.first(where: { $0.kind == .real }) else {
            print("[MABEL-SWITCH] FAIL seeds missing — picker cannot offer both rigs")
            return
        }
        try? await Task.sleep(for: .seconds(4))

        controller.connect(to: sim)
        try? await Task.sleep(for: .seconds(10))
        report("1/3 SIM  ")

        controller.connect(to: real)
        try? await Task.sleep(for: .seconds(12))
        report("2/3 REAL ")

        // The one that matters: can the operator get BACK when the real robot
        // never answered?
        controller.connect(to: sim)
        try? await Task.sleep(for: .seconds(12))
        report("3/3 SIM' ")
        print("[MABEL-SWITCH] done — 3/3 must show kind=sim and a live status")
    }
    #endif
}

/// Left-edge vertical navigation rail (Robot · Autonomy · Settings) — the
/// visionOS placement: floats OUTSIDE the window on the leading edge, stacked
/// above the Robot tab's viewer rail, never occluded by 3D content.
struct LeftNavRail: View {
    @Binding var selection: AppSection

    var body: some View {
        VStack(spacing: 4) {
            ForEach(AppSection.allCases) { section in
                let active = selection == section
                Button {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { selection = section }
                } label: {
                    VStack(spacing: 3) {
                        Group {
                            // MABEL's own silhouette for the Robot tab (RobotGlyph.swift);
                            // `move.3d` describes a viewport, not the machine. The other
                            // sections keep their SF Symbols at matched weight.
                            if section == .robot {
                                MabelBodyGlyph(size: 16)
                            } else {
                                Image(systemName: section.icon)
                                    .font(.system(size: 16, weight: .semibold))
                            }
                        }
                        .frame(height: 16)
                        Text(section.title).font(.system(size: 9, weight: .bold)).lineLimit(1)
                            .minimumScaleFactor(0.8)
                    }
                    .foregroundStyle(active ? .white : Theme.Palette.textPrimary)
                    .frame(width: 58, height: 52)
                    .background {
                        if active {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(Theme.accentGradient)
                        }
                    }
                    .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                .buttonStyle(.plain)
                .contentShape(.hoverEffect, RoundedRectangle(cornerRadius: 12, style: .continuous))
                .hoverEffect()
            }
        }
        .padding(6)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 18, style: .continuous)
            .strokeBorder(.white.opacity(0.12), lineWidth: 1) }
    }
}

/// Bottom tab bar — kept for reference/reuse (the nav now rides LeftNavRail).
struct BottomBar: View {
    @Binding var selection: AppSection
    @Namespace private var ns

    var body: some View {
        HStack(spacing: 4) {
            ForEach(AppSection.allCases) { section in
                let active = selection == section
                Button {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { selection = section }
                } label: {
                    VStack(spacing: 3) {
                        ZStack {
                            if active {
                                Capsule().fill(Theme.Palette.accent.opacity(0.22))
                                    .frame(width: 52, height: 30)
                                    .matchedGeometryEffect(id: "btab", in: ns)
                            }
                            Group {
                            // MABEL's own silhouette for the Robot tab (RobotGlyph.swift);
                            // `move.3d` describes a viewport, not the machine. The other
                            // sections keep their SF Symbols at matched weight.
                            if section == .robot {
                                MabelBodyGlyph(size: 17)
                            } else {
                                Image(systemName: section.icon)
                                    .font(.system(size: 17, weight: .semibold))
                            }
                        }
                        .frame(height: 17)
                        }
                        .frame(height: 30)
                        Text(section.title).font(.system(size: 10, weight: .semibold))
                    }
                    .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.textTertiary)
                    .frame(width: 76)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .hoverEffect()
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .glassBackgroundEffect()
    }
}


/// Brief "teleop running" note that dissolves after 3 seconds.
private struct ImmersivePlaceholder: View {
    @State private var visible = true

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "visionpro")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(Theme.Palette.accent)
            Text("Immersive teleop running")
                .font(.mabelTitle).foregroundStyle(Theme.Palette.textPrimary)
            Text("Use the floating console to record, toggle tracking, stop, or exit.")
                .font(.mabelBody).foregroundStyle(Theme.Palette.textTertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .opacity(visible ? 1 : 0)
        .task {
            try? await Task.sleep(for: .seconds(3))
            withAnimation(.easeOut(duration: 0.8)) { visible = false }
        }
    }
}
