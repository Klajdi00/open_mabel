//
//  ServerDiscovery.swift
//  MabelTeleop
//
//  Finds MABEL teleop servers on the local network via Bonjour, using the
//  modern Network framework (NWBrowser) — the legacy NetServiceBrowser is
//  unreliable on visionOS about triggering the Local Network permission
//  prompt and browsing at all. Each discovered service is RESOLVED by
//  actually opening a TCP connection to it, which both yields the numeric
//  IP (network-independent, re-resolved on every Wi-Fi) and PROVES the
//  network allows device-to-device traffic — campus/hotel APs with client
//  isolation fail here, and that distinction is surfaced to Settings.
//

import Foundation
import Combine
import Network

@MainActor
final class ServerDiscovery: ObservableObject {
    struct Server: Identifiable, Equatable {
        let id = UUID()
        let name: String
        let host: String     // numeric LAN IP, verified reachable
    }

    @Published private(set) var servers: [Server] = []
    /// The discovered robots in the store's terms (advertised name + verified host).
    var found: [RobotStore.DiscoveredRobot] { servers.map { .init(name: $0.name, host: $0.host) } }
    @Published private(set) var searching = false
    /// Human-readable diagnosis for the Settings card.
    @Published private(set) var note: String? = nil

    private var browser: NWBrowser?
    private var probes: [NWConnection] = []

    func start() {
        guard !searching else { return }
        servers.removeAll()
        note = nil
        searching = true

        let params = NWParameters.tcp
        params.includePeerToPeer = true
        let b = NWBrowser(for: .bonjour(type: "_mabel-teleop._tcp", domain: nil), using: params)
        browser = b

        b.stateUpdateHandler = { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self else { return }
                switch state {
                case .failed(let error):
                    self.note = "Discovery failed: \(error.localizedDescription)"
                    self.searching = false
                case .waiting(let error):
                    // Typically: Local Network permission denied.
                    self.note = "Waiting for network access (\(error.localizedDescription)) — check Settings › Privacy › Local Network."
                default:
                    break
                }
            }
        }

        b.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor [weak self] in
                self?.handle(results: results)
            }
        }
        b.start(queue: .main)
    }

    func stop() {
        browser?.cancel(); browser = nil
        for p in probes { p.cancel() }
        probes.removeAll()
        searching = false
    }

    /// Re-scan from scratch — call after a Wi-Fi change so stale entries
    /// from the previous network are dropped.
    func restart() {
        stop()
        start()
    }

    // MARK: - Resolution (connect → learn the IP → proven reachable)

    private func handle(results: Set<NWBrowser.Result>) {
        if results.isEmpty {
            servers.removeAll()
            return
        }
        for result in results {
            guard case let .service(name, _, _, _) = result.endpoint else { continue }
            guard !servers.contains(where: { $0.name == name }) else { continue }
            resolve(endpoint: result.endpoint, name: name)
        }
        // Drop servers that vanished from the network.
        let liveNames = results.compactMap { result -> String? in
            if case let .service(name, _, _, _) = result.endpoint { return name }
            return nil
        }
        servers.removeAll { !liveNames.contains($0.name) }
    }

    private func resolve(endpoint: NWEndpoint, name: String) {
        // Force IPv4: a Bonjour host advertises both an A and a link-local
        // AAAA record, and if we resolve to the latter we get back
        // `fe80::…%en2` — a scoped link-local address that can't form a valid
        // `ws://` URL and won't route to the robot. The robot's LAN/Tailscale
        // address we want is always IPv4, so ask only for that.
        let params = NWParameters.tcp
        if let ip = params.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options {
            ip.version = .v4
        }
        let conn = NWConnection(to: endpoint, using: params)
        probes.append(conn)
        conn.stateUpdateHandler = { [weak self, weak conn] state in
            Task { @MainActor [weak self] in
                guard let self else { return }
                switch state {
                case .ready:
                    // The handshake succeeded → peer traffic is allowed and
                    // we can read the robot's CURRENT numeric address. Sanitize
                    // it (drop any %zone, reject link-local) before we store it.
                    if let remote = conn?.currentPath?.remoteEndpoint,
                       case let .hostPort(host, _) = remote,
                       let ip = Self.sanitized(Self.numericHost(host)) {
                        if !self.servers.contains(where: { $0.host == ip }) {
                            self.servers.append(Server(name: name, host: ip))
                        }
                        self.note = nil
                    }
                    conn?.cancel()
                case .failed(let error):
                    self.note = "Found '\(name)' but couldn't reach it (\(error.localizedDescription)) — this network may isolate devices; use a hotspot."
                    conn?.cancel()
                default:
                    break
                }
            }
        }
        conn.start(queue: .main)
        // Don't probe forever on an isolating network.
        Task { [weak conn] in
            try? await Task.sleep(for: .seconds(6))
            if let conn, conn.state != .ready, conn.state != .cancelled {
                conn.cancel()
            }
        }
    }

    private static func numericHost(_ host: NWEndpoint.Host) -> String {
        switch host {
        case .ipv4(let addr): return "\(addr)"
        case .ipv6(let addr): return "\(addr)"
        case .name(let name, _): return name
        @unknown default: return "\(host)"
        }
    }

    /// Make a host string safe to drop into a `ws://host:port` URL: strip any
    /// interface scope (`192.168.1.5%en2`, `fe80::1%en2`) and reject the
    /// non-routable link-local ranges (`fe80::/10`, `169.254/16`) that the OS
    /// sometimes resolves first. Returns nil when nothing usable remains — the
    /// caller then simply doesn't list / connect to it. Shared by the
    /// RobotController so a stale persisted host self-heals too.
    nonisolated static func sanitized(_ host: String) -> String? {
        let h = host.split(separator: "%").first.map(String.init)?
            .trimmingCharacters(in: .whitespaces) ?? ""
        if h.isEmpty { return nil }
        if h.lowercased().hasPrefix("fe80:") || h.hasPrefix("169.254.") { return nil }
        return h
    }
}
