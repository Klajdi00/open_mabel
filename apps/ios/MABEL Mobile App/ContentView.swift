//
//  ContentView.swift
//  MABEL Mobile App
//
//  v2 root. Three destinations: the unified Robot space (one full-bleed 3D
//  viewer that is both visualizer and controller), Autonomy, and Settings.
//  Landscape hosts a slim floating nav rail (DJI/pro style); portrait uses the
//  Liquid-Glass bottom bar. The app opens on the Robot space.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var controller: RobotController
    @StateObject private var discovery = ServerDiscovery()
    @AppStorage("appearance") private var appearanceRaw = Appearance.system.rawValue
    @AppStorage("autoConnect") private var autoConnect = true
    @State private var selection: AppSection = ContentView.initialSection
    /// Where "back" returns from the immersive landscape Robot space.
    @State private var preImmersive: AppSection = .policies
    @State private var didAutoConnect = false
    @State private var statusExpanded = false
    // Deferred loading: show a tiny spinner only if a tab takes > 1s to appear.
    @State private var loadingTab: AppSection? = nil
    @State private var showLoader = false

    private var appearance: Appearance { Appearance(rawValue: appearanceRaw) ?? .system }

    /// Initial tab. DEBUG builds honour a `-startTab <section>` launch argument so a
    /// screenshot/UI harness can boot straight onto a specific tab; release always
    /// opens on the Robot space.
    private static var initialSection: AppSection {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-startTab"), i + 1 < args.count,
           let s = AppSection(rawValue: args[i + 1]) { return s }
        // argv-proof twin (argv is dropped on prewarmed launches): `debugStartTab`.
        if let raw = UserDefaults.standard.string(forKey: "debugStartTab"),
           let s = AppSection(rawValue: raw) { return s }
        #endif
        return .robot
    }

    var body: some View {
        GeometryReader { geo in
            let landscape = geo.size.width > geo.size.height
            ZStack {
                Theme.Palette.canvas.ignoresSafeArea()
                if landscape && selection == .robot {
                    // IMMERSIVE Robot space: no nav rail at all — every point of
                    // the screen is viewport. The floating status bar reserves its
                    // own strip; the space's back button leaves the tab.
                    RobotSpaceView(onBack: {
                        withAnimation(Theme.Motion.snappy) { selection = preImmersive }
                    })
                    .safeAreaInset(edge: .top, spacing: 0) {
                        GlobalStatusBar(expanded: $statusExpanded, floating: true)
                            .padding(.horizontal, 12).padding(.top, 2)
                    }
                    .transition(.identity)
                } else if landscape {
                    // Content fills the WHOLE screen (no dead margins); a FLOATING glass
                    // rail hovers over the leading edge, just clear of any Dynamic Island
                    // on that side. `railSpace` reserves room so a screen's text clears
                    // the rail + island while its background flows edge-to-edge under both.
                    let lead = geo.safeAreaInsets.leading
                    let trail = geo.safeAreaInsets.trailing
                    let railSpace = lead + NavRail.width + 16
                    ZStack(alignment: .leading) {
                        Group {
                            if selection == .robot {
                                // The unified 3D viewer flows under the rail + a floating
                                // status bar — the whole screen is the viewport.
                                content
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    .overlay(alignment: .top) {
                                        GlobalStatusBar(expanded: $statusExpanded, floating: true)
                                            .padding(.leading, railSpace).padding(.trailing, max(trail, 12)).padding(.top, 4)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                    }
                            } else {
                                // Titled / scrolling sections: docked bar + content sit to
                                // the right of the rail, filling the rest edge-to-edge.
                                VStack(spacing: 0) {
                                    GlobalStatusBar(expanded: $statusExpanded)
                                    content
                                }
                                .padding(.leading, railSpace)
                                .padding(.trailing, trail)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                            }
                        }
                        .environment(\.railLeadingInset, railSpace)

                        NavRail(selection: $selection)
                            .padding(.leading, lead + 8)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .ignoresSafeArea()
                    .transition(.identity)
                } else {
                    // Portrait: content fills the screen EDGE TO EDGE — it bleeds all
                    // the way to the bottom (under the home indicator) so there's no
                    // dead band beneath the bar. The nav bar is an OVERLAY that floats
                    // above the home indicator; each screen reserves `bottomBarInset`
                    // (bar + home indicator) for its interactive bottom elements so
                    // buttons never hide under the glass.
                    content
                        .frame(width: geo.size.width)   // constant width — never driven by the floating bar
                        // The inset content reserves below it INCLUDES the bottom safe
                        // area, because the bar sits at (safe area + pad) above the edge
                        // while content bleeds to the edge — so both are measured from
                        // the same origin and floating content always clears the bar.
                        .environment(\.bottomBarInset, BottomBar.reservedSpace + geo.safeAreaInsets.bottom)
                        .ignoresSafeArea(.container, edges: .bottom)
                        .safeAreaInset(edge: .top, spacing: 0) { floatingStatusInset() }
                }
            }
            // The bottom nav bar lives at the ZStack level (portrait only) so it is
            // NEVER torn down when the content swaps tabs — content carries `.id`, and
            // an overlay on it would blink on every switch. As a stable sibling the
            // bar just stays put; only its highlight slides.
            .overlay(alignment: .bottom) {
                if !landscape {
                    BottomBar(selection: $selection)
                        .padding(.horizontal, 28)
                        // Sit low — dip into the home-indicator safe area on devices
                        // that have one (negative pad), else a small gap. Stays clear
                        // of the screen edge so it's never clipped.
                        .padding(.bottom, geo.safeAreaInsets.bottom >= 24 ? -12 : 6)
                        // Stay pinned to the bottom when the keyboard opens (don't
                        // fly up above it).
                        .ignoresSafeArea(.keyboard, edges: .bottom)
                }
            }
            .environment(\.isLandscape, landscape)
        }
        .preferredColorScheme(appearance.colorScheme)
        .tint(Theme.Palette.accent)
        .environmentObject(discovery)
        // Walk up to the robot and it offers itself — see ProximityConnectCard. At the
        // ROOT so the card can appear over any tab, and above the nav bar.
        .proximityConnect(controller, discovery: discovery)
        .onOpenURL { url in
            // Tapping the Dynamic Island / Live Activity reopens the app where the
            // operator left off — we DON'T force a tab. Only route when a link
            // explicitly names a destination. Legacy v1 hosts (cockpit / body /
            // maps) all land on the unified Robot space. v2 registers mabelv2://
            // (so links never cross-route to a side-by-side v1 install) but the
            // handler accepts both schemes.
            guard url.scheme == "mabelv2" || url.scheme == "mabel" else { return }
            let target: AppSection?
            switch url.host {
            case "robot", "cockpit", "body", "wholebody", "maps", "navigate":
                target = .robot
            case "policies":           target = .policies
            case "settings":           target = .settings
            default:                   target = nil   // "island" / unknown → stay put
            }
            if let target { withAnimation(Theme.Motion.snappy) { selection = target } }
        }
        .onChange(of: selection) { old, new in
            // Entering the Robot space: remember where "back" should return.
            if new == .robot && old != .robot { preImmersive = old }
            // Arm the deferred loader: if the new tab hasn't appeared within 1s,
            // fade in a tiny spinner (cleared by the tab's onAppear).
            loadingTab = new
            showLoader = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                if loadingTab == new { withAnimation(.easeIn(duration: 0.2)) { showLoader = true } }
            }
        }
        .onChange(of: controller.collectingData) { _, on in
            // Legacy entry point (Policies "collect" button) → jump into the Robot space.
            if on {
                selection = .robot
                controller.collectingData = false
            }
        }
        .onAppear {
            controller.connect()        // start in demo so the app is usable instantly
            discovery.start()           // search the LAN for a real server
            #if DEBUG
            // Screenshot harness: open the status dropdown after layout settles
            // (a popover presented at init is silently dropped) — same defaults
            // pattern as debugConsole; synthetic taps don't land on the Simulator.
            if UserDefaults.standard.bool(forKey: "debugStatusPanel") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { statusExpanded = true }
            }
            // debugLandscape=YES — rotate to landscape (the Simulator can't be
            // rotated from simctl).
            if UserDefaults.standard.bool(forKey: "debugLandscape") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                    (UIApplication.shared.connectedScenes.first as? UIWindowScene)?
                        .requestGeometryUpdate(.iOS(interfaceOrientations: .landscapeRight))
                }
            }
            #endif
            // DEBUG harness hook (like `-startTab`): `-connectHost <host>` connects
            // straight to a given server, bypassing the race — so a simulator
            // test/screenshot run can pin the app to a specific live bridge.
            #if DEBUG
            let args = ProcessInfo.processInfo.arguments
            if let i = args.firstIndex(of: "-connectHost"), i + 1 < args.count {
                didAutoConnect = true
                controller.connect(host: args[i + 1])
                return
            }
            // Seed the real-robot access code into the Keychain (screenshot
            // harness: the simulator has no way to type it into Settings).
            if let c = UserDefaults.standard.string(forKey: "debugRealCode"), !c.isEmpty {
                SecureStore.realRobotCode = c
            }
            // Defaults fallback — launch argv is silently dropped on prewarmed
            // launches, so `defaults write … debugConnectHost <host>` is the
            // reliable channel (same pattern as debugPov/debugTeleop).
            if let h = UserDefaults.standard.string(forKey: "debugConnectHost"), !h.isEmpty {
                didAutoConnect = true
                controller.connect(host: h)
                return
            }
            #endif
            // Race the network and link up automatically: fastest LAN/Tailscale
            // first. Works even before Bonjour resolves (it probes the known hosts).
            if autoConnect, !didAutoConnect {
                didAutoConnect = true
                controller.connectAuto(discovered: discovery.found)
            }
        }
        .onChange(of: discovery.servers) { _, servers in
            // Feed fresh Bonjour hits into the planner — a newly-appeared LAN
            // host can win the race (and upgrade us off Tailscale).
            guard autoConnect else { return }
            controller.noteDiscovered(servers.map { .init(name: $0.name, host: $0.host) })
        }
    }

    /// Floating glass status bar that reserves its own space at the top. The
    /// status detail floats as a popover off the bar — it never pushes content.
    private func floatingStatusInset() -> some View {
        GlobalStatusBar(expanded: $statusExpanded, floating: true)
            .padding(.horizontal, 12).padding(.top, 2)
    }

    @ViewBuilder private var content: some View {
        Group {
            switch selection {
            case .robot:     RobotSpaceView()
            case .policies:  PoliciesView()
            case .settings:  RobotView()
            }
        }
        .onAppear {
            // The new tab finished building — cancel any pending/visible loader.
            if loadingTab == selection {
                loadingTab = nil
                if showLoader { withAnimation(.easeOut(duration: 0.2)) { showLoader = false } }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()        // keep a section's full-bleed content out of the nav rail
        // Instant swap between tabs — the default cross-fade routes through the
        // dark canvas and flashes black. The tab indicator still glides (it
        // animates separately via matchedGeometryEffect).
        .id(selection)
        .transition(.identity)
        .overlay(alignment: .center) {
            if showLoader { LoadingPip().transition(.opacity) }
        }
    }
}

/// A tiny Apple-style loading indicator shown only when a screen takes a moment.
struct LoadingPip: View {
    var body: some View {
        ProgressView()
            .controlSize(.regular)
            .tint(Theme.Palette.accent)
            .padding(18)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
            .shadow(color: .black.opacity(0.22), radius: 12, y: 5)
    }
}

/// Portrait bottom tab bar — Apple-Music-style **Liquid Glass**. The BAR is a
/// single glass capsule (so the content scrolling beneath refracts through it);
/// the selection is a soft accent-tinted capsule that glides between tabs and the
/// selected icon/label tint accent. Only the lightweight highlight animates, so
/// it stays smooth (no per-tab glass blob to re-render every frame).
struct BottomBar: View {
    @Binding var selection: AppSection
    var floating: Bool = true
    @Namespace private var glassNS

    /// Space above the bottom **safe area** that the floating bar needs — screens
    /// add this to `safeAreaInsets.bottom` (via the environment) so floating content
    /// (the viewport control bar, the last scroll item) always clears the glass with a
    /// comfortable gap. Kept in lockstep with the bar's own bottom offset.
    static let reservedSpace: CGFloat = 70

    var body: some View {
        HStack(spacing: 2) {
            ForEach(AppSection.allCases) { section in
                tab(section)
            }
        }
        .padding(4)
        // One liquid-glass pass on the static bar — cheap, and the content beneath
        // shows through it.
        .glassEffect(.regular, in: Capsule(style: .continuous))
        .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.12), lineWidth: 1) }
        .shadow(color: .black.opacity(0.16), radius: 10, y: 4)
    }

    @ViewBuilder private func tab(_ section: AppSection) -> some View {
        let active = selection == section
        Button {
            Haptics.select()
            withAnimation(.spring(response: 0.36, dampingFraction: 0.82)) { selection = section }
        } label: {
            VStack(spacing: 2) {
                // The Robot tab is the machine itself, so it gets MABEL's silhouette
                // rather than `move.3d` — a 3-axis gizmo, right for a viewport, wrong
                // for the destination. Drawn in RobotGlyph.swift; the other two keep
                // their SF Symbols, and the fill weight is matched so the three read
                // as one set.
                if section == .robot {
                    MabelBodyGlyph(size: 15)
                        .frame(height: 15)
                } else {
                    Image(systemName: section.icon)
                        .font(.system(size: 15, weight: .semibold))
                        .frame(height: 15)
                }
                Text(section.title).font(.system(size: 8.5, weight: .semibold))
            }
            // Accent-tinted selection (like Music) — not a solid block.
            .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.textSecondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .background {
                if active {
                    // Soft tinted capsule that slides between tabs — GPU-cheap, smooth.
                    Capsule(style: .continuous)
                        .fill(Theme.Palette.accent.opacity(0.16))
                        .matchedGeometryEffect(id: "nav.highlight", in: glassNS)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// A FLOATING Liquid-Glass rail — the vertical sibling of the portrait bottom bar.
/// It hovers over the leading edge (content flows edge-to-edge UNDER it), wordmark-
/// free and content-sized so it reads as glass on both Light and Dark Mode.
struct NavRail: View {
    static let width: CGFloat = 60        // capsule width — used for the content leading inset
    @EnvironmentObject private var controller: RobotController
    @Binding var selection: AppSection
    @Namespace private var ns

    var body: some View {
        VStack(spacing: 5) {
            ForEach(AppSection.allCases) { section in
                railButton(section)
            }
            // Connection dot
            Circle()
                .fill(controller.connection.color)
                .frame(width: 8, height: 8)
                .shadow(color: controller.connection.color.opacity(0.7), radius: 4)
                .padding(.top, 5).padding(.bottom, 1)
        }
        .padding(.vertical, 8).padding(.horizontal, 4)
        // One liquid-glass capsule (like the bottom tab bar) so the content behind
        // refracts through it — a true floating glass rail, not a flat panel.
        .glassEffect(.regular, in: Capsule(style: .continuous))
        .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        .shadow(color: .black.opacity(0.18), radius: 12, x: 1, y: 4)
    }

    private func railButton(_ section: AppSection) -> some View {
        let active = selection == section
        return Button {
            Haptics.select()
            withAnimation(Theme.Motion.snappy) { selection = section }
        } label: {
            VStack(spacing: 3) {
                Image(systemName: section.icon)
                    .font(.system(size: 18, weight: .semibold))
                Text(section.title)
                    .font(.system(size: 8.5, weight: .semibold, design: .rounded))
            }
            .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.textSecondary)
            .frame(width: 52, height: 48)
            .background {
                if active {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(Theme.Palette.accent.opacity(0.16))
                        .matchedGeometryEffect(id: "railhi", in: ns)
                }
            }
        }
        .buttonStyle(.plain)
    }
}
