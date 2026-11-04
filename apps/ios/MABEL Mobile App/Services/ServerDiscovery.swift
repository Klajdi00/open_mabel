//
//  ServerDiscovery.swift
//  MABEL Mobile App
//
//  Finds MABEL teleop servers on the local network via Bonjour/mDNS, so the
//  operator never has to type an IP. The server (MABEL/server) advertises
//  `_mabel-teleop._tcp`; we browse for it, resolve each to its numeric LAN
//  IP (NOT the `.local` hostname — `ws://name.local:9090` resolves
//  unreliably on iOS Wi-Fi), and hand that to the RobotController.
//

import Foundation
import Combine
import Darwin

@MainActor
final class ServerDiscovery: NSObject, ObservableObject {
    struct Server: Identifiable, Equatable {
        let id = UUID()
        let name: String
        let host: String     // numeric LAN IP, e.g. "192.168.1.42"
    }

    @Published private(set) var servers: [Server] = []
    /// The hits as the robot store takes them: advertised name + numeric host.
    var found: [RobotStore.DiscoveredRobot] { servers.map { .init(name: $0.name, host: $0.host) } }
    @Published private(set) var searching = false

    private let browser = NetServiceBrowser()
    private var resolving: Set<NetService> = []

    override init() {
        super.init()
        browser.delegate = self
    }

    func start() {
        guard !searching else { return }
        servers.removeAll()
        searching = true
        browser.searchForServices(ofType: "_mabel-teleop._tcp", inDomain: "local.")
    }

    func stop() {
        browser.stop()
        resolving.removeAll()
        searching = false
    }
}

extension ServerDiscovery: NetServiceBrowserDelegate, NetServiceDelegate {
    nonisolated func netServiceBrowser(_ b: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
        Task { @MainActor in
            service.delegate = self
            resolving.insert(service)
            service.resolve(withTimeout: 5)
        }
    }

    nonisolated func netServiceBrowser(_ b: NetServiceBrowser, didRemove service: NetService, moreComing: Bool) {
        Task { @MainActor in
            servers.removeAll { $0.name == service.name }
        }
    }

    nonisolated func netServiceDidResolveAddress(_ service: NetService) {
        // Pick the REACHABLE numeric IPv4 from the resolved addresses; fall back
        // to the `.local` hostname only if no usable IP is available.
        let ip = Self.bestIPv4(from: service)
        let fallback = service.hostName.map { $0.hasSuffix(".") ? String($0.dropLast()) : $0 }
        let name = service.name
        Task { @MainActor in
            resolving.remove(service)
            guard let host = ip ?? fallback else { return }
            let s = Server(name: name, host: host)
            if !servers.contains(where: { $0.host == s.host }) { servers.append(s) }
        }
    }

    /// Choose the IPv4 a phone on the SAME Wi-Fi can actually reach.
    ///
    /// A resolved `.local` host carries one A record per active interface on the
    /// Mac. With Tailscale (`utun`/100.64/10 CGNAT), USB/Ethernet adapters
    /// (self-assigned `169.254.*` link-local), and loopback (`127.*`) all live at
    /// once, the addresses arrive in a NON-deterministic order — so blindly
    /// taking the first one frequently hands back an IP the iPhone can't route to,
    /// and the connection silently never comes up. Rank every IPv4 and keep the
    /// best, preferring a private LAN address (the Wi-Fi subnet) over everything.
    nonisolated static func bestIPv4(from service: NetService) -> String? {
        guard let addresses = service.addresses else { return nil }
        let ips = addresses.compactMap { data -> String? in
            data.withUnsafeBytes { raw -> String? in
                guard let sa = raw.baseAddress?.assumingMemoryBound(to: sockaddr.self),
                      sa.pointee.sa_family == sa_family_t(AF_INET) else { return nil }
                var addr = raw.baseAddress!.assumingMemoryBound(to: sockaddr_in.self).pointee.sin_addr
                var buf = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
                inet_ntop(AF_INET, &addr, &buf, socklen_t(INET_ADDRSTRLEN))
                return String(cString: buf)
            }
        }
        return bestReachable(ips)
    }

    /// Pick the most-reachable IPv4 from a list (extracted for testability).
    /// Highest `reachability` score wins; anything scoring < 0 is unreachable
    /// and skipped. Returns nil if none are usable.
    nonisolated static func bestReachable(_ ips: [String]) -> String? {
        ips.map { ($0, reachability($0)) }
           .filter { $0.1 >= 0 }
           .max { $0.1 < $1.1 }?.0
    }

    /// Rank an IPv4 by how likely a same-Wi-Fi phone can reach it.
    /// Higher = better; negative = never usable (loopback / link-local / 0.0.0.0).
    nonisolated static func reachability(_ ip: String) -> Int {
        let o = ip.split(separator: ".").compactMap { Int($0) }
        guard o.count == 4 else { return -1 }
        if o[0] == 0 || o[0] == 127 { return -1 }            // 0.0.0.0 / loopback
        if o[0] == 169 && o[1] == 254 { return -1 }          // link-local (APIPA)
        if o[0] == 192 && o[1] == 168 { return 100 }         // home Wi-Fi subnet
        if o[0] == 10 { return 90 }                          // private (often Tailscale too)
        if o[0] == 172 && (16...31).contains(o[1]) { return 90 } // private
        if o[0] == 100 && (64...127).contains(o[1]) { return 50 } // CGNAT (Tailscale) — last resort
        return 30                                            // routable public, if nothing better
    }

    nonisolated func netService(_ service: NetService, didNotResolve errorDict: [String: NSNumber]) {
        Task { @MainActor in resolving.remove(service) }
    }
}
