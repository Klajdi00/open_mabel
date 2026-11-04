//
//  ConnectionPlanner.swift
//  MABEL Mobile App
//
//  Picks the BEST way to reach the bridge, fastest first. It races every direct
//  candidate (Bonjour-discovered LAN hosts + the known LAN / Tailscale IPs) with
//  a parallel TCP probe — the analog of Bonjour "open a connection; the one that
//  answers is the server" — and orders the reachable ones by measured latency,
//  so the lowest-latency network wins (LAN beats Tailscale when both are up).
//
//  This mirrors the ServerDiscovery TCP-handshake reachability test.
//

import Foundation
import Combine
import Network

@MainActor
final class ConnectionPlanner: ObservableObject {

    /// One candidate's live probe result, surfaced to the Settings network card.
    struct Probe: Identifiable, Equatable {
        let id = UUID()
        let host: String
        let kind: RouteKind
        var rttMs: Double? = nil        // round-trip of the TCP connect
        var reachable: Bool? = nil       // nil = still in flight
    }

    @Published private(set) var probes: [Probe] = []
    @Published private(set) var planning = false
    /// When the last plan finished — lets the UI show "scanned 3s ago".
    @Published private(set) var lastPlan: Date? = nil

    /// Per-probe cap. Only bounds the worst case (NOTHING reachable): a
    /// reachable host short-circuits the race the instant it answers, so a live
    /// LAN bridge connects in its RTT (~tens of ms), not this timeout.
    private let probeTimeout: TimeInterval = 1.5
    /// Guards against a superseded plan's in-flight probes writing into a newer
    /// probe list (each connectAuto bumps this).
    private var generation = 0

    /// Race the direct candidates and return the chosen route(s), fastest first.
    /// Resolves the moment the
    /// FIRST reachable host answers (instant connect) while the remaining probes
    /// keep filling the network readout in the background.
    /// Race a SPECIFIC robot's direct routes (LAN/Tailscale). Keeping the
    /// host set per-robot is what stops the sim and the real rig from being
    /// conflated (their routes are disjoint).
    func plan(hosts candidates: [String]) async -> [ConnectionRoute] {
        generation &+= 1
        let gen = generation
        planning = true

        var hosts: [String] = []
        func add(_ raw: String) {
            let h = raw.trimmingCharacters(in: .whitespaces)
            guard !h.isEmpty, !hosts.contains(h) else { return }
            hosts.append(h)
        }
        candidates.forEach(add)

        probes = hosts.map { Probe(host: $0, kind: NetworkConfig.classify($0)) }
        let timeout = probeTimeout

        // First reachable host wins; nil once every probe has failed.
        let winner: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            guard !hosts.isEmpty else { cont.resume(returning: nil); return }
            let race = ProbeRace(total: hosts.count) { cont.resume(returning: $0) }
            for h in hosts {
                Task { [weak self] in
                    let (_, rtt) = await ConnectionPlanner.tcpProbe(host: h, timeout: timeout)
                    await MainActor.run { [weak self] in
                        guard let self, self.generation == gen,
                              let i = self.probes.firstIndex(where: { $0.host == h }) else { return }
                        self.probes[i].rttMs = rtt
                        self.probes[i].reachable = (rtt != nil)
                    }
                    race.settle(host: h, reachable: rtt != nil)
                }
            }
        }

        if generation == gen { planning = false; lastPlan = Date() }

        // The winning direct route (fastest).
        var routes: [ConnectionRoute] = []
        if let w = winner {
            routes.append(ConnectionRoute(kind: NetworkConfig.classify(w), host: w,
                                          rttMs: probes.first { $0.host == w }?.rttMs))
        }
        return routes
    }

    /// Reset probe state (e.g. when leaving Settings).
    func clear() { probes.removeAll() }

    /// Thread-safe first-past-the-post for the probe race: fires `finish` once,
    /// on the first reachable host, or with nil after every probe has failed.
    private final class ProbeRace: @unchecked Sendable {
        private let lock = NSLock()
        private var done = false
        private var settled = 0
        private let total: Int
        private let finish: (String?) -> Void
        init(total: Int, finish: @escaping (String?) -> Void) { self.total = total; self.finish = finish }
        func settle(host: String, reachable: Bool) {
            lock.lock()
            settled += 1
            guard !done else { lock.unlock(); return }
            if reachable { done = true; lock.unlock(); finish(host); return }
            if settled >= total { done = true; lock.unlock(); finish(nil); return }
            lock.unlock()
        }
    }

    // MARK: - TCP probe

    /// Open a TCP connection to host:9090 and measure time-to-ready. A ready
    /// handshake proves the bridge is reachable on that path AND yields its
    /// connect latency (≈ RTT) for ranking. Returns nil if it never connects.
    nonisolated static func tcpProbe(host: String, timeout: TimeInterval) async -> (String, Double?) {
        await withCheckedContinuation { (cont: CheckedContinuation<(String, Double?), Never>) in
            guard let port = NWEndpoint.Port(rawValue: UInt16(NetworkConfig.wsPort)) else {
                cont.resume(returning: (host, nil)); return
            }
            let params = NWParameters.tcp
            // Force IPv4 so a link-local AAAA can't waste the slot.
            if let ip = params.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options {
                ip.version = .v4
            }
            let conn = NWConnection(host: NWEndpoint.Host(host), port: port, using: params)
            let start = DispatchTime.now()
            let lock = NSLock()
            var done = false
            func finish(_ rtt: Double?) {
                lock.lock(); if done { lock.unlock(); return }; done = true; lock.unlock()
                conn.cancel()
                cont.resume(returning: (host, rtt))
            }
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
                    finish(ms)
                case .failed, .cancelled:
                    finish(nil)
                default:
                    break
                }
            }
            conn.start(queue: .global(qos: .userInitiated))
            DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { finish(nil) }
        }
    }
}
