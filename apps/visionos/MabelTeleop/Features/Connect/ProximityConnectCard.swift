//
//  ProximityConnectCard.swift
//  MabelTeleop
//
//  Walk up to the robot wearing the headset and a glass panel floats into view with the
//  robot turning slowly inside it and one button on it: Connect. No IP address, no
//  settings, no Wi-Fi — the AirPods moment, for a robot, over Bluetooth LE.
//
//  Attach it once, at the root, with `.proximityConnect(controller)`. It watches
//  `BLECentral` and presents itself CENTRED over the window when a robot comes within
//  range. It used to be a bottom ornament: below the window, where the protruding 3D
//  stage covered it and ate its pinches ("too low and under the 3D viewer — I cannot even
//  click on it", 2026-10-17). Centred, it is where the eye already is, and while it is up
//  the stage yields (`connectCardVisible` → RobotSpaceView.modalUp), as every pop-up does.
//
//  A NOTE ON THE APPLE CARD: the system AirPods sheet is Continuity proximity pairing,
//  which only MFi accessories may emit and only the shell may draw. This is the same
//  experience rebuilt inside the app — the part that is actually ours to build — and it
//  works with any robot running NOVA's ble_bridge.
//

import SwiftUI

// MARK: - What is being offered

/// A robot the app could connect to right now, and how it would get there.
///
/// The card used to take a `NearbyRobot`, which only Bluetooth produces — so standing next
/// to a robot that was plainly on the same Wi-Fi, the card offered Bluetooth and nothing
/// else. The offer is now transport-agnostic and `best(of:)` applies the same priority the
/// rest of the app uses (`LinkPriority`): Wi-Fi first, Bluetooth when there is no network.
enum ConnectOffer: Identifiable, Equatable {
    case wifi(ServerDiscovery.Server)
    case bluetooth(NearbyRobot)

    var id: String {
        switch self {
        case let .wifi(s):      return "wifi:\(s.host)"
        case let .bluetooth(r): return "ble:\(r.id.uuidString)"
        }
    }

    /// The robot's own name — NOVA, MABEL, or whatever it has been renamed to. Bonjour
    /// advertises it as "<name> Robot (<host>)", so the suffix is trimmed back off here:
    /// the card is introducing a robot, not a service record.
    var name: String {
        switch self {
        case let .wifi(s):
            var n = s.name
            if let r = n.range(of: " (") { n = String(n[n.startIndex..<r.lowerBound]) }
            for tail in [" Robot", " Sim"] where n.hasSuffix(tail) { n.removeLast(tail.count) }
            return n.isEmpty ? s.name : n
        case let .bluetooth(r): return r.name
        }
    }

    var route: RouteKind {
        switch self {
        case .wifi:      return .lan
        case .bluetooth: return .bluetooth
        }
    }

    /// Bluetooth carries live status in its advertisement — a passive scan already knows
    /// whether the gateway is up and whether the e-stop is engaged. Wi-Fi discovery is just
    /// a service record, so those are unknown until we connect, and the card says so rather
    /// than inventing a state.
    var beacon: BLEProtocol.Beacon? {
        if case let .bluetooth(r) = self { return r.beacon }
        return nil
    }

    /// Best first, by what the link can carry. Ties are impossible: one per transport.
    static func best(of offers: [ConnectOffer]) -> ConnectOffer? {
        offers.min { LinkPriority.rank($0.route) < LinkPriority.rank($1.route) }
    }
}

// MARK: - Presentation

extension View {
    /// Float the proximity connect panel whenever a robot is near and we are not already
    /// driving it.
    func proximityConnect(_ controller: RobotController,
                          discovery: ServerDiscovery) -> some View {
        modifier(ProximityConnectModifier(controller: controller, discovery: discovery))
    }
}

struct ProximityConnectModifier: ViewModifier {
    let controller: RobotController
    /// Passed in, NOT taken from the environment: this modifier is applied above the
    /// `.environmentObject(discovery)` injection, so its overlay is outside that subtree
    /// and an @EnvironmentObject here would be missing at runtime.
    @ObservedObject var discovery: ServerDiscovery
    @ObservedObject private var central = BLECentral.shared
    @ObservedObject private var controllerState: RobotController
    @Environment(\.scenePhase) private var scenePhase
    /// Offers the operator waved away, by id — the Wi-Fi equivalent of BLECentral's own
    /// suppression, which only knows about peripherals.
    @State private var dismissed: Set<String> = []

    init(controller: RobotController, discovery: ServerDiscovery) {
        self.controller = controller
        self.discovery = discovery
        self.controllerState = controller
    }

    /// What to offer, if anything: the best transport that is actually reachable, minus
    /// anything already connected or waved away.
    private var offer: ConnectOffer? {
        #if DEBUG
        // Placement harness: `debugConnectOffer=<name>` forces a Wi-Fi offer on screen
        // (Bonjour + an unconnected link can't be staged on demand in the Simulator).
        if let n = UserDefaults.standard.string(forKey: "debugConnectOffer"), !n.isEmpty,
           !dismissed.contains("wifi:192.0.2.1") {
            return .wifi(.init(name: n, host: "192.0.2.1"))
        }
        #endif
        // Driving it already? Then there is nothing to offer. (Moving UP from Bluetooth to
        // Wi-Fi is the transport handover's job, not a card the operator has to tap.)
        guard !controllerState.connection.isLive else { return nil }
        var candidates: [ConnectOffer] = discovery.servers.map { .wifi($0) }
        if let ble = central.promoted { candidates.append(.bluetooth(ble)) }
        return ConnectOffer.best(of: candidates.filter { !dismissed.contains($0.id) })
    }

    func body(content: Content) -> some View {
        content
            .overlay {
                if let offer {
                    ZStack {
                        // A soft scrim so the card reads as THE thing to answer; it passes
                        // pinches through — the card's own Not Now is the way out.
                        Color.black.opacity(0.28)
                            .ignoresSafeArea()
                            .allowsHitTesting(false)
                            .transition(.opacity)
                        ProximityConnectCard(offer: offer, controller: controller,
                                             onDismiss: { dismissed.insert(offer.id) })
                            // Lifted off the window plane so nothing in the window sits
                            // in front of it while the stage fades out.
                            .offset(z: 40)
                            .transition(.scale(scale: 0.92).combined(with: .opacity))
                            .id(offer.id)
                    }
                    .zIndex(20)
                }
            }
            .animation(Theme.Motion.sheet, value: offer?.id)
            .onChange(of: offer?.id, initial: true) { _, id in
                controller.connectCardVisible = id != nil
            }
            .onAppear { discovery.start() }
            // WI-FI LOST, BLUETOOTH IN RANGE. Asked, never silent: dropping a tier removes
            // video and most of the protocol, and from the operator's side the phone is
            // still on Wi-Fi — it is the robot that went. A silent downgrade would read as
            // the app losing features by itself.
            .alert("Switch to Bluetooth?",
                   isPresented: Binding(get: { controllerState.bluetoothFallback != nil },
                                        set: { if !$0 { controller.declineBluetoothFallback() } })) {
                Button("Switch") { controller.acceptBluetoothFallback() }
                Button("Not Now", role: .cancel) { controller.declineBluetoothFallback() }
            } message: {
                Text("\(controllerState.bluetoothFallback?.name ?? "The robot") is in Bluetooth range. "
                     + "Bluetooth drives the whole robot but carries no camera video.")
            }
            // `onChange` does not fire for the value a view is BORN with, so a
            // cold launch never reached the `.active` branch below. On a real
            // device the central starts scanning itself when the radio powers
            // on, which hid it; in the Simulator, where it never does, nothing
            // scanned at all.
            .onAppear { central.startScanning() }
            .onChange(of: scenePhase) { _, phase in
                // Scanning with duplicates is the expensive mode; only run it while the
                // operator can actually see the panel.
                if phase == .active { central.startScanning() } else { central.stopScanning() }
            }
    }
}

// MARK: - The panel

struct ProximityConnectCard: View {
    let offer: ConnectOffer
    let controller: RobotController
    let onDismiss: () -> Void

    @ObservedObject private var central = BLECentral.shared
    @ObservedObject private var link: RobotController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: Phase = .offered
    @State private var ripple = false

    enum Phase: Equatable { case offered, connecting, connected, failed }

    init(offer: ConnectOffer, controller: RobotController, onDismiss: @escaping () -> Void) {
        self.offer = offer
        self.controller = controller
        self.link = controller
        self.onDismiss = onDismiss
    }

    var body: some View {
        HStack(spacing: 28) {
            stage
            VStack(alignment: .leading, spacing: 14) {
                identity
                actions
            }
            .frame(width: 260, alignment: .leading)
        }
        .padding(.horizontal, 34)
        .padding(.vertical, 26)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 40, style: .continuous))
        .onAppear { if !reduceMotion { ripple = true } }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(offer.name) is available over \(offer.route.rawValue)")
        .onChange(of: link.connection) { _, c in
            guard phase == .connecting else { return }
            if c.isLive { succeeded() }
        }
    }

    // MARK: Pieces

    /// The robot, turning, over a soft bloom that keeps the silhouette legible against
    /// whatever the passthrough happens to be showing behind the glass.
    private var stage: some View {
        ZStack {
            RadialGradient(colors: [Theme.Palette.accent.opacity(0.24), .clear],
                           center: .center, startRadius: 6, endRadius: 140)
                .blur(radius: 20)

            if phase == .connecting {
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .strokeBorder(Theme.Palette.accent.opacity(0.35), lineWidth: 1.5)
                        .frame(width: 110, height: 110)
                        .scaleEffect(ripple ? 2.0 : 0.7)
                        .opacity(ripple ? 0 : 0.9)
                        .animation(.easeOut(duration: 1.9)
                            .repeatForever(autoreverses: false)
                            .delay(Double(i) * 0.63), value: ripple)
                }
            }

            PairingRobotView(pose: livePose, animated: !reduceMotion)
                .frame(width: 210, height: 210)
                .allowsHitTesting(false)
        }
        .frame(width: 210, height: 210)
    }

    private var identity: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(offer.name)
                .font(.system(.largeTitle, design: .rounded).weight(.semibold))
                .foregroundStyle(Theme.Palette.textPrimary)

            HStack(spacing: 7) {
                TransportGlyph(icon: statusIcon, size: 13)
                Text(statusText).font(.system(.title3, design: .rounded))
            }
            .foregroundStyle(statusTint)
            .animation(Theme.Motion.gentle, value: phase)
        }
    }

    @ViewBuilder
    private var actions: some View {
        switch phase {
        case .connected:
            Label("Connected", systemImage: "checkmark.circle.fill")
                .font(.system(.title3, design: .rounded).weight(.semibold))
                .foregroundStyle(Theme.Palette.live)
                .transition(.scale(scale: 0.9).combined(with: .opacity))

        default:
            HStack(spacing: 14) {
                Button(action: connect) {
                    ZStack {
                        if phase == .connecting {
                            ProgressView().tint(.white)
                        } else {
                            Text(phase == .failed ? "Try Again" : "Connect")
                                .font(.system(.title3, design: .rounded).weight(.semibold))
                        }
                    }
                    .foregroundStyle(.white)
                    .frame(width: 150, height: 54)
                    .background(Theme.accentGradient, in: Capsule())
                }
                .buttonStyle(.plain)
                .contentShape(.hoverEffect, Capsule(style: .continuous))
                .hoverEffect()
                .disabled(phase == .connecting)

                Button("Not Now") { dismiss() }
                    .font(.system(.body, design: .rounded))
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .hoverEffect()
            }
        }
    }

    // MARK: Behaviour

    /// The live pose once we are driving it; the REST pose before that. Not nil:
    /// nil leaves the rig in its ASSEMBLED pose, which is arms-forward with both
    /// elbows folded 90 degrees — the pose the whole app stopped showing when the
    /// rest pose was derived (controller/docs/rest_pose.tex). A card that
    /// introduces the robot should introduce it standing the way it stands.
    private var livePose: [String: Double]? {
        phase == .connected ? controller.state.joints : RobotModel.homePose
    }

    private var statusIcon: String {
        switch phase {
        case .connecting: return offer.route.icon
        case .connected:  return "checkmark.circle.fill"
        case .failed:     return "exclamationmark.triangle.fill"
        case .offered:    return offer.beacon?.estop == true ? "hand.raised.fill" : "bolt.horizontal.circle"
        }
    }

    /// The transport is named on every line, because which one you get changes what the
    /// robot can do — Bluetooth carries no video.
    private var transport: String { offer.route == .bluetooth ? "Bluetooth" : "Wi-Fi" }

    private var statusText: String {
        switch phase {
        case .connecting: return "Connecting over Bluetooth…"
        case .connected:  return "Ready to drive"
        case .failed:     return "Couldn't connect — move closer and retry"
        case .offered:
            guard let b = offer.beacon else { return "\(transport) · Ready" }
            if !b.gatewayUp { return "\(transport) · robot starting up" }
            return b.estop ? "\(transport) · E-stop engaged" : "\(transport) · Ready"
        }
    }

    private var statusTint: Color {
        switch phase {
        case .connected:  return Theme.Palette.live
        case .failed:     return Theme.Palette.danger
        case .offered:    return offer.beacon?.estop == true ? Theme.Palette.warn : Theme.Palette.textSecondary
        case .connecting: return Theme.Palette.accent
        }
    }

    private func connect() {
        withAnimation(Theme.Motion.snappy) { phase = .connecting }
        switch offer {
        case let .bluetooth(robot):
            controller.connectBluetooth(robot) { ok in
                if ok { succeeded() } else { withAnimation(Theme.Motion.snappy) { phase = .failed } }
            }
        case let .wifi(server):
            // No completion on this path: `connect(host:)` hands off to the route race and
            // the result arrives as a connection state, which `onChange` above watches.
            controller.connect(host: server.host)
            DispatchQueue.main.asyncAfter(deadline: .now() + 12) {
                if phase == .connecting { withAnimation(Theme.Motion.snappy) { phase = .failed } }
            }
        }
    }

    private func succeeded() {
        withAnimation(Theme.Motion.snappy) { phase = .connected }
        if case let .bluetooth(robot) = offer { central.suppress(robot.id) }
        // Let "Connected" read for a beat before the panel retires itself — dismissing on
        // the instant of success feels like the tap failed.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.1) { dismiss() }
    }

    private func dismiss() {
        if case let .bluetooth(robot) = offer { central.dismiss(robot) }
        onDismiss()
    }
}
