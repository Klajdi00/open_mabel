//
//  ProximityConnectCard.swift
//  MABEL Mobile App
//
//  Walk up to the robot and a card rises with the robot turning slowly inside it and one
//  button on it: Connect. No IP address, no settings, no Wi-Fi — the AirPods moment, for a
//  robot, over Bluetooth LE.
//
//  Attach it once, at the root, with `.proximityConnect(controller)`. It watches
//  `BLECentral` and presents itself when a robot comes within arm's reach; everything else
//  in the app carries on underneath.
//
//  A NOTE ON THE APPLE CARD: the system AirPods sheet is Continuity proximity pairing,
//  which only MFi accessories may emit and only SpringBoard may draw. This is the same
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
    /// Present the proximity connect card whenever a robot is near and we are not already
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
        // Driving it already? Then there is nothing to offer. (Moving UP from Bluetooth to
        // Wi-Fi is the transport handover's job, not a card the operator has to tap.)
        guard !controllerState.connection.isLive else { return nil }
        var candidates: [ConnectOffer] = discovery.servers.map { .wifi($0) }
        if let ble = central.promoted { candidates.append(.bluetooth(ble)) }
        return ConnectOffer.best(of: candidates.filter { !dismissed.contains($0.id) })
    }

    func body(content: Content) -> some View {
        content
            .overlay(alignment: .bottom) {
                if let offer {
                    ProximityConnectCard(offer: offer, controller: controller,
                                         onDismiss: { dismissed.insert(offer.id) })
                        .padding(.horizontal, Theme.Space.md)
                        .padding(.bottom, Theme.Space.lg)
                        .transition(.asymmetric(
                            insertion: .move(edge: .bottom)
                                .combined(with: .opacity)
                                .combined(with: .scale(scale: 0.94, anchor: .bottom)),
                            removal: .move(edge: .bottom).combined(with: .opacity)))
                        .id(offer.id)
                }
            }
            .animation(Theme.Motion.sheet, value: offer?.id)
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
                // operator can actually see a card.
                if phase == .active { central.startScanning() } else { central.stopScanning() }
            }
    }
}

// MARK: - The card

struct ProximityConnectCard: View {
    let offer: ConnectOffer
    let controller: RobotController
    let onDismiss: () -> Void

    @ObservedObject private var central = BLECentral.shared
    @ObservedObject private var link: RobotController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: Phase = .offered
    @State private var appeared = false
    @State private var ripple = false

    enum Phase: Equatable { case offered, connecting, connected, failed }

    init(offer: ConnectOffer, controller: RobotController, onDismiss: @escaping () -> Void) {
        self.offer = offer
        self.controller = controller
        self.link = controller
        self.onDismiss = onDismiss
    }

    var body: some View {
        VStack(spacing: Theme.Space.md) {
            handle
            stage
            identity
            actions
        }
        .padding(.top, Theme.Space.sm)
        .padding(.bottom, Theme.Space.lg)
        .padding(.horizontal, Theme.Space.lg)
        .frame(maxWidth: 420)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 34, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 34, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        }
        .mabelShadow(Theme.Shadow.card)
        .scaleEffect(appeared ? 1 : 0.96)
        .onAppear {
            withAnimation(Theme.Motion.adaptive(Theme.Motion.sheet, reduceMotion: reduceMotion)) {
                appeared = true
            }
            if !reduceMotion { ripple = true }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(offer.name) is available over \(offer.route.rawValue)")
        // Wi-Fi has no completion handler — `connect(host:)` races routes and reports
        // through the controller — so the card watches the link instead of guessing.
        .onChange(of: link.connection) { _, c in
            guard phase == .connecting else { return }
            if c.isLive { succeeded() }
        }
    }

    // MARK: Pieces

    private var handle: some View {
        Capsule()
            .fill(Theme.Palette.hairline)
            .frame(width: 38, height: 5)
    }

    /// The robot, turning, over a soft radial bloom — the bloom keeps the silhouette
    /// legible whether the material behind resolves light or dark.
    private var stage: some View {
        ZStack {
            RadialGradient(
                colors: [Theme.Palette.accent.opacity(0.22), .clear],
                center: .center, startRadius: 4, endRadius: 150)
                .blur(radius: 18)

            if phase == .connecting {
                // Concentric ripples, the "reaching out" beat between tap and link-up.
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .strokeBorder(Theme.Palette.accent.opacity(0.35), lineWidth: 1.5)
                        .frame(width: 96, height: 96)
                        .scaleEffect(ripple ? 2.1 : 0.7)
                        .opacity(ripple ? 0 : 0.9)
                        .animation(
                            .easeOut(duration: 1.9)
                                .repeatForever(autoreverses: false)
                                .delay(Double(i) * 0.63),
                            value: ripple)
                }
            }

            PairingRobotView(pose: livePose, animated: !reduceMotion)
                .frame(height: 212)
                .allowsHitTesting(false)
        }
        .frame(height: 212)
    }

    private var identity: some View {
        VStack(spacing: 5) {
            Text(offer.name)
                .font(.system(.title2, design: .rounded).weight(.semibold))
                .foregroundStyle(Theme.Palette.textPrimary)
                .contentTransition(.opacity)

            HStack(spacing: 6) {
                TransportGlyph(icon: statusIcon, size: 11)
                Text(statusText)
                    .font(.system(.subheadline, design: .rounded))
            }
            .foregroundStyle(statusTint)
            .contentTransition(.opacity)
            .animation(Theme.Motion.gentle, value: phase)
        }
    }

    @ViewBuilder
    private var actions: some View {
        switch phase {
        case .connected:
            Label("Connected", systemImage: "checkmark.circle.fill")
                .font(.system(.headline, design: .rounded))
                .foregroundStyle(Theme.Palette.live)
                .frame(maxWidth: .infinity)
                .frame(height: 52)
                .transition(.scale(scale: 0.9).combined(with: .opacity))

        default:
            VStack(spacing: Theme.Space.sm) {
                Button(action: connect) {
                    ZStack {
                        if phase == .connecting {
                            ProgressView().tint(.white)
                        } else {
                            Text(phase == .failed ? "Try Again" : "Connect")
                                .font(.system(.headline, design: .rounded))
                        }
                    }
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .background(Theme.accentGradient, in: Capsule())
                }
                .buttonStyle(PressableStyle())
                .disabled(phase == .connecting)

                Button("Not Now") { dismiss() }
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .frame(height: 30)
            }
        }
    }

    // MARK: Behaviour

    /// The live pose once we are driving it; the REST pose before that. Not nil:
    /// nil leaves the rig in its ASSEMBLED pose, which is arms-forward with both
    /// elbows folded 90 degrees — the pose the whole app stopped showing when the
    /// rest pose was derived (controller/docs/rest_pose.pdf). A card that
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
        case .connecting: return "Connecting over \(transport)…"
        case .connected:  return "Ready to drive"
        case .failed:
            return offer.route == .bluetooth
                ? "Couldn't connect — move closer and retry"
                : "Couldn't connect — check you are on the same network"
        case .offered:
            // Only Bluetooth knows the robot's state before connecting: its advertisement
            // carries the e-stop and gateway flags. Wi-Fi discovery is a service record, so
            // it says what it is and nothing it cannot know.
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
        Haptics.tap()
        withAnimation(Theme.Motion.snappy) { phase = .connecting }
        switch offer {
        case let .bluetooth(robot):
            controller.connectBluetooth(robot) { ok in
                if ok { succeeded() } else {
                    withAnimation(Theme.Motion.snappy) { phase = .failed }
                    Haptics.warning()
                }
            }
        case let .wifi(server):
            // No completion on this path: `connect(host:)` hands off to the route race and
            // the result arrives as a connection state, which `onChange` above is watching.
            controller.connect(host: server.host)
            DispatchQueue.main.asyncAfter(deadline: .now() + 12) {
                if phase == .connecting {
                    withAnimation(Theme.Motion.snappy) { phase = .failed }
                    Haptics.warning()
                }
            }
        }
    }

    private func succeeded() {
        withAnimation(Theme.Motion.snappy) { phase = .connected }
        Haptics.success()
        if case let .bluetooth(robot) = offer { central.suppress(robot.id) }
        // Let the "Connected" state read for a beat before the card retires itself —
        // dismissing on the instant of success feels like the tap failed.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.1) { dismiss() }
    }

    private func dismiss() {
        if case let .bluetooth(robot) = offer { central.dismiss(robot) }
        onDismiss()
    }
}
