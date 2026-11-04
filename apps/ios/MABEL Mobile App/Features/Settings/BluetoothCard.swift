//
//  BluetoothCard.swift
//  MABEL Mobile App
//
//  The Bluetooth section of Settings: what is on the air right now, and a direct Connect
//  for each robot — the manual path alongside the automatic proximity card.
//
//  It is deliberately plain. The card that matters is the one that arrives by itself when
//  you walk up to the robot (ProximityConnectCard); this is for the times you want to pick
//  a specific robot, or you waved the card away, or two robots are in the same room.
//

import SwiftUI

struct BluetoothCard: View {
    @EnvironmentObject private var controller: RobotController
    @ObservedObject private var central = BLECentral.shared
    @ObservedObject private var known = KnownRobots.shared
    @State private var connectingID: UUID?
    @State private var renamingID: UUID?
    @State private var draftName = ""

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                SectionLabel("BLUETOOTH")
                explainer

                if !known.isEmpty {
                    savedSection
                    Divider().overlay(Theme.Palette.hairlineSoft)
                }

                if central.unauthorized {
                    hint("Bluetooth access is off for MABEL. Turn it on in iOS Settings › Privacy › Bluetooth.",
                         icon: "exclamationmark.triangle.fill", tint: Theme.Palette.warn)
                } else if !central.poweredOn {
                    hint("Bluetooth is turned off.", icon: "power", tint: Theme.Palette.textTertiary)
                } else if central.nearby.isEmpty {
                    hint(known.isEmpty
                         ? "Looking for robots nearby… make sure the robot is powered on and within a few metres."
                         : "Nothing new on the air. A saved robot above can still be connected to directly.",
                         icon: "dot.radiowaves.left.and.right", tint: Theme.Palette.textTertiary)
                } else {
                    VStack(spacing: 8) {
                        ForEach(central.nearby) { robot in
                            row(robot)
                            if robot.id != central.nearby.last?.id {
                                Divider().overlay(Theme.Palette.hairlineSoft)
                            }
                        }
                    }
                    .animation(Theme.Motion.gentle, value: central.nearby.map(\.id))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear { central.startScanning() }
    }

    // MARK: Pieces

    private var explainer: some View {
        Text(controller.overBluetooth
             ? "Driving over Bluetooth — no Wi-Fi needed. Camera video is not available on this link."
             : "Drive the robot with no network at all. Whole-body, base, lift, torso and neck all work; video does not fit over Bluetooth.")
            .font(.system(size: 11))
            .foregroundStyle(controller.overBluetooth ? Theme.Palette.accent : Theme.Palette.textTertiary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func row(_ robot: NearbyRobot) -> some View {
        HStack(spacing: Theme.Space.sm) {
            BluetoothGlyph(size: 14)
                .foregroundStyle(signalTint(robot))
                .frame(width: 22)

            VStack(alignment: .leading, spacing: 2) {
                Text(robot.name)
                    .font(.mabelBody)
                    .foregroundStyle(Theme.Palette.textPrimary)
                Text(subtitle(robot))
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            Spacer(minLength: 6)

            if connectingID == robot.id {
                ProgressView().controlSize(.small)
            } else {
                Button("Connect") { connect(robot) }
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.Palette.accent)
                    .buttonStyle(PressableStyle())
            }
        }
        .padding(.vertical, 2)
    }

    private func hint(_ text: String, icon: String, tint: Color) -> some View {
        Label(text, systemImage: icon)
            .font(.system(size: 11))
            .foregroundStyle(tint)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Behaviour

    /// Phrase the signal rather than print a dBm number: BLE RSSI swings 10-15 dB just
    /// from the phone turning in your hand, so a figure would look precise and be noise.
    private func subtitle(_ robot: NearbyRobot) -> String {
        let range: String
        switch robot.proximity {
        case .immediate: range = "Right here"
        case .near:      range = "Nearby"
        case .far:       range = "Far — move closer"
        }
        if !robot.beacon.gatewayUp { return "\(range) · starting up" }
        return robot.beacon.estop ? "\(range) · E-stop engaged" : "\(range) · Ready"
    }

    private func signalTint(_ robot: NearbyRobot) -> Color {
        switch robot.proximity {
        case .immediate: return Theme.Palette.live
        case .near:      return Theme.Palette.accent
        case .far:       return Theme.Palette.textTertiary
        }
    }


    // MARK: Saved robots

    /// Robots this device has connected to before. They are listed whether or not anything
    /// is on the air: CoreBluetooth reconnects from the saved identifier alone, so Connect
    /// here works the instant the app opens instead of waiting for a scan to catch a
    /// broadcast. A saved robot that IS currently advertising shows its live signal too.
    private var savedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SAVED")
                .font(.system(size: 10, weight: .semibold, design: .rounded))
                .foregroundStyle(Theme.Palette.textTertiary)
            ForEach(known.robots) { robot in
                savedRow(robot)
                if robot.id != known.robots.last?.id {
                    Divider().overlay(Theme.Palette.hairlineSoft)
                }
            }
        }
    }

    private func savedRow(_ robot: KnownRobot) -> some View {
        let live = central.nearby.first { $0.id == robot.id }
        return HStack(spacing: Theme.Space.sm) {
            Group {
                if live == nil {
                    Image(systemName: "clock.arrow.circlepath").font(.system(size: 13, weight: .semibold))
                } else {
                    BluetoothGlyph(size: 14)
                }
            }
            .foregroundStyle(live == nil ? Theme.Palette.textTertiary : Theme.Palette.accent)
            .frame(width: 22)

            VStack(alignment: .leading, spacing: 2) {
                if renamingID == robot.id {
                    TextField(robot.advertisedName, text: $draftName)
                        .textFieldStyle(.plain)
                        .font(.mabelBody)
                        .foregroundStyle(Theme.Palette.textPrimary)
                        .submitLabel(.done)
                        .onSubmit { commitRename(robot) }
                } else {
                    Text(robot.displayName)
                        .font(.mabelBody)
                        .foregroundStyle(Theme.Palette.textPrimary)
                }
                Text(savedSubtitle(robot, live: live))
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            Spacer(minLength: 6)

            if renamingID == robot.id {
                Button("Save") { commitRename(robot) }
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.Palette.accent)
                    .buttonStyle(PressableStyle())
            } else if connectingID == robot.id {
                ProgressView().controlSize(.small)
            } else {
                Button("Connect") { connectSaved(robot) }
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.Palette.accent)
                    .buttonStyle(PressableStyle())
                Menu {
                    Button("Rename\u{2026}") { beginRename(robot) }
                    Button("Forget", role: .destructive) { known.forget(id: robot.id) }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 14))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
        }
        .padding(.vertical, 2)
    }

    /// Says what the operator needs to decide with: is it here, and what does the robot
    /// itself call it. A renamed robot shows BOTH names — a local label must never hide
    /// which machine is about to be driven.
    private func savedSubtitle(_ robot: KnownRobot, live: NearbyRobot?) -> String {
        var parts: [String] = []
        if let live {
            switch live.proximity {
            case .immediate: parts.append("Right here")
            case .near:      parts.append("Nearby")
            case .far:       parts.append("Far \u{2014} move closer")
            }
        } else {
            parts.append("Not in range")
        }
        if robot.isRenamed { parts.append(robot.advertisedName) }
        parts.append("last \(Self.stamp.localizedString(for: robot.lastConnected, relativeTo: Date()))")
        return parts.joined(separator: " \u{00B7} ")
    }

    private static let stamp: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter(); f.unitsStyle = .short; return f
    }()

    private func beginRename(_ robot: KnownRobot) {
        draftName = robot.nickname ?? ""
        renamingID = robot.id
    }

    private func commitRename(_ robot: KnownRobot) {
        known.rename(id: robot.id, to: draftName)
        renamingID = nil
        draftName = ""
        Haptics.tap()
    }

    private func connectSaved(_ robot: KnownRobot) {
        Haptics.tap()
        connectingID = robot.id
        controller.connectBluetooth(saved: robot) { ok in
            connectingID = nil
            if ok { Haptics.success(); central.suppress(robot.id) } else { Haptics.warning() }
        }
    }

    private func connect(_ robot: NearbyRobot) {
        Haptics.tap()
        connectingID = robot.id
        controller.connectBluetooth(robot) { ok in
            connectingID = nil
            if ok { Haptics.success(); central.suppress(robot.id) } else { Haptics.warning() }
        }
    }
}
