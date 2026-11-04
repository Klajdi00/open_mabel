//
//  RobotInfoSheet.swift
//  MabelTeleop
//
//  The host inspector — tap a robot to see its networking detail (how YOU reach
//  it: Wi-Fi / Tailscale), rename it, and Connect. Mirrors the iOS app's
//  RobotInfoSheet (the connected-clients list, which needs server `serverMeta`,
//  is omitted on the headset for now).
//

import SwiftUI
import UIKit
import Darwin

struct RobotInfoSheet: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.dismiss) private var dismiss
    let robot: Robot

    // Local mirror of the name so a rename reflects immediately here (the passed-in
    // `robot` is an immutable snapshot).
    @State private var displayName = ""
    @State private var renaming = false
    @State private var renameText = ""
    @State private var confirmForget = false

    private var isActive: Bool { controller.robotStore.activeID == robot.id }
    private var live: Bool { isActive && controller.connection == .connected }
    /// The connected-clients list is only known for the robot we're linked to.
    private var clients: [ClientInfo] { live ? (controller.serverMeta?.clients ?? []) : [] }
    /// Any host can be forgotten now (the store records it so even the built-in
    /// sim / thor stay gone until re-added).
    private var forgettable: Bool { true }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    nameCard
                    robotCard
                    thisDeviceCard
                    clientsCard
                    actionButton
                }
                .padding(16)
            }
            .scrollIndicators(.hidden)
            .navigationTitle("Host")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .frame(minWidth: 460, minHeight: 520)
        .onAppear { displayName = robot.name }
        .alert("Rename host", isPresented: $renaming) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Save") { commitRename() }
        } message: {
            Text("The name is saved on the robot and syncs to every client.")
        }
        .alert("Forget this host?", isPresented: $confirmForget) {
            Button("Cancel", role: .cancel) {}
            Button("Forget", role: .destructive) { doForget() }
        } message: {
            Text("“\(displayName)” will be removed from this device. You can add it again anytime.")
        }
    }

    // MARK: - Name (Rename button right beside it)

    private var nameCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                cardTitle("NAME")
                HStack(spacing: 10) {
                    Image(icon: robot.kind.icon).font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(robot.kind.tint).frame(width: 26)
                    Text(displayName.isEmpty ? robot.name : displayName)
                        .font(.system(size: 16, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
                        .lineLimit(1)
                    Text(robot.kind.badge).font(.system(size: 9, weight: .heavy)).foregroundStyle(robot.kind.tint)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(robot.kind.tint.opacity(0.14), in: Capsule())
                    Spacer(minLength: 8)
                    renameButton
                }
                Text("The name is stored on the robot and shown the same on every client.")
                    .font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary)
            }
        }
    }

    /// Compact "Rename" pill sitting beside the name.
    private var renameButton: some View {
        Button {
            renameText = displayName.isEmpty ? robot.name : displayName
            renaming = true
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "pencil").font(.system(size: 12, weight: .bold))
                Text("Rename").font(.system(size: 13, weight: .semibold))
            }
            .foregroundStyle(Theme.Palette.accent)
            .padding(.horizontal, 12).padding(.vertical, 7)
            .background(Theme.Palette.accent.opacity(0.14), in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.accent.opacity(0.35), lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
    }

    private func commitRename() {
        let t = renameText.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        controller.renameRobot(id: robot.id, to: t)
        displayName = t
        Haptics.success()
    }

    private func doForget() {
        if live { controller.disconnect() }
        controller.robotStore.forget(id: robot.id)
        Haptics.rigid()
        dismiss()
    }

    // MARK: - Robot routes

    private var robotCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                cardTitle("ROBOT")
                infoRow("Type", robot.kind.label, robot.kind.tint, icon: robot.kind.icon)
                infoRow("Reachable via", robot.routeSummary, Theme.Palette.textSecondary,
                        icon: "point.3.connected.trianglepath.dotted")
                // The transport actually carrying this live session right now.
                if live, let route = controller.activeRoute {
                    infoRow("Connected via", route.kind.rawValue, Theme.Palette.live, icon: route.kind.icon)
                    monoRow("Endpoint", route.host)
                }
                ForEach(robot.routes, id: \.self) { host in
                    monoRow(NetworkConfig.classify(host) == .tailscale ? "Tailscale" : "Wi-Fi", host)
                }
                if live {
                    infoRow("Latency", String(format: "%.0f ms", controller.state.latencyMs),
                            Theme.Palette.live, icon: "bolt.horizontal")
                }
            }
        }
    }

    // MARK: - This device

    private var thisDeviceCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                cardTitle("THIS DEVICE")
                infoRow("Device", "Apple Vision Pro", Theme.Palette.textSecondary, icon: "visionpro")
                infoRow("OS", "visionOS \(UIDevice.current.systemVersion)", Theme.Palette.textSecondary, icon: "applelogo")
                monoRow("IP", DeviceNet.localIPv4() ?? "—")
            }
        }
    }

    // MARK: - Connected clients (server GUI parity)

    private var clientsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack { cardTitle("CONNECTED CLIENTS"); Spacer()
                    if live { Text("\(clients.count)").font(.system(size: 11, weight: .heavy)).foregroundStyle(Theme.Palette.accent) } }
                if !live {
                    Text("Connect to this robot to see who else is connected.")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                } else if clients.isEmpty {
                    Text("No clients reported.").font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                } else {
                    ForEach(clients) { clientRow($0) }
                }
            }
        }
    }

    private func clientRow(_ c: ClientInfo) -> some View {
        HStack(spacing: 11) {
            Image(systemName: deviceIcon(c)).font(.system(size: 15, weight: .semibold))
                .foregroundStyle(c.driver ? Theme.Palette.live : Theme.Palette.accent).frame(width: 24)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(c.device + (c.os.isEmpty ? "" : " · \(c.os)"))
                        .font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                    if c.driver {
                        Text("DRIVING").font(.system(size: 8, weight: .heavy)).foregroundStyle(Theme.Palette.live)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Theme.Palette.live.opacity(0.15), in: Capsule())
                    }
                }
                Text("\(c.ip)  ·  \(c.location)").font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
            }
            Spacer()
            transportBadge(c.transport)
        }
        .padding(.vertical, 4)
    }

    private func transportBadge(_ t: String) -> some View {
        Text(t.isEmpty ? "—" : t).font(.system(size: 9, weight: .heavy))
            .foregroundStyle(Theme.Palette.textSecondary)
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background(Theme.Palette.surfaceHi, in: Capsule())
    }
    private func deviceIcon(_ c: ClientInfo) -> String {
        let d = c.device.lowercased()
        if d.contains("vision") { return "visionpro" }
        if d.contains("iphone") { return "iphone" }
        if d.contains("web") || d.contains("site") { return "globe" }
        if d.contains("monitor") { return "display" }
        return "desktopcomputer"
    }

    // MARK: - Action

    @ViewBuilder private var actionButton: some View {
        VStack(spacing: 10) {
            if live {
                tintedFullButton("Disconnect", "xmark.circle", tint: Theme.Palette.warn) {
                    Haptics.tap(); controller.disconnect(); dismiss()
                }
            } else {
                PrimaryButton(title: "Connect", systemImage: "antenna.radiowaves.left.and.right") {
                    Haptics.tap(); controller.connect(to: robot); dismiss()
                }
            }
            // Forget — red, destructive, right under Disconnect. Only custom hosts
            // you added show it; confirmed before removal.
            if forgettable {
                tintedFullButton("Forget this host", "trash", tint: Theme.Palette.danger) {
                    confirmForget = true
                }
            }
        }
    }

    /// Full-width tinted capsule (colored Disconnect / Forget).
    private func tintedFullButton(_ title: String, _ icon: String, tint: Color,
                                  action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) { Image(systemName: icon); Text(title) }
                .font(.mabelHeadline).foregroundStyle(.white)
                .frame(maxWidth: .infinity).padding(.vertical, 14)
                .background(tint.gradient, in: Capsule())
        }
        .buttonStyle(PressableStyle())
    }

    // MARK: - Bits

    private func cardTitle(_ t: String) -> some View { SectionLabel(t) }
    private func infoRow(_ label: String, _ value: String, _ tint: Color, icon: String) -> some View {
        HStack(spacing: 10) {
            TransportGlyph(icon: icon, size: 12).foregroundStyle(tint).frame(width: 20)
            Text(label).font(.system(size: 13)).foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            Text(value).font(.system(size: 13, weight: .semibold)).foregroundStyle(tint).lineLimit(1)
        }
    }
    private func monoRow(_ label: String, _ value: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.system(size: 9, weight: .bold)).foregroundStyle(Theme.Palette.textTertiary).frame(width: 64, alignment: .leading)
            Text(value).font(.system(size: 11, design: .monospaced)).foregroundStyle(Theme.Palette.textSecondary).lineLimit(1)
            Spacer()
        }
    }
}

/// This device's non-loopback IPv4 on the active interface (Wi-Fi first).
enum DeviceNet {
    static func localIPv4() -> String? {
        var address: String?
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return nil }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let flags = Int32(ptr.pointee.ifa_flags)
            let addr = ptr.pointee.ifa_addr.pointee
            guard (flags & (IFF_UP | IFF_RUNNING)) == (IFF_UP | IFF_RUNNING),
                  addr.sa_family == UInt8(AF_INET) else { continue }
            let name = String(cString: ptr.pointee.ifa_name)
            guard name == "en0" || name == "pdp_ip0" || address == nil else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            getnameinfo(ptr.pointee.ifa_addr, socklen_t(addr.sa_len), &host, socklen_t(host.count),
                        nil, 0, NI_NUMERICHOST)
            let ip = String(cString: host)
            if name == "en0" { return ip }     // Wi-Fi wins
            address = ip
        }
        return address
    }
}
