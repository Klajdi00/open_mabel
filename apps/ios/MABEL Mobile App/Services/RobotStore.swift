//
//  RobotStore.swift
//  MABEL Mobile App
//
//  The remembered set of robots, keyed by stable IDENTITY (not host). The SAME
//  robot reached over Wi-Fi, Tailscale, or Bluetooth is ONE entry with multiple
//  routes — so the simulation (LAN + Tailscale) shows once. Persists to UserDefaults, tracks the active robot, folds Bonjour
//  discoveries into the right robot, and adopts the server-owned display name.
//

import Foundation
import Combine

@MainActor
final class RobotStore: ObservableObject {
    @Published private(set) var robots: [Robot] = []
    /// Identity of the robot we're connected to / connecting to.
    @Published var activeID: String? = nil

    private let key = "mabel.robots.v6"   // bump when the persisted Robot schema changes
    private let forgottenKey = "mabel.robots.forgotten.v1"
    /// IDs the operator explicitly forgot. Suppresses re-seeding so even the
    /// built-in sim / thor stay gone once forgotten (until re-added).
    private var forgotten: Set<String> = []

    init() {
        load()
        seed()
        sort()
    }

    // MARK: - Derived

    var active: Robot? { robots.first { $0.id == activeID } }
    func robot(id: String) -> Robot? { robots.first { $0.id == id } }

    /// Which seeded robot a discovered/typed host belongs to: if a robot already
    /// lists this host as one of its routes it's that robot; otherwise it's the
    /// simulation (the sim is the Bonjour-advertised default on the Mac/LAN).
    private func ownerID(forHost host: String) -> String {
        if let r = robots.first(where: { $0.routes.contains(host) }) { return r.id }
        return "mabel-sim"
    }

    // MARK: - Seeding

    private func seed() {
        for r in NetworkConfig.seedRobots() where robot(id: r.id) == nil && !forgotten.contains(r.id) {
            robots.append(r)
        }
    }

    // MARK: - Discovery merge (Bonjour) — folds hosts into the owning robot

    func mergeDiscovered(_ hosts: [String]) {
        mergeDiscovered(named: hosts.map { DiscoveredRobot(name: "", host: $0) })
    }

    /// A Bonjour hit: the advertised instance name ("NOVA Robot (mabel-orin)",
    /// "MABEL Sim (dev-mac)") and its verified numeric host.
    struct DiscoveredRobot: Equatable {
        let name: String
        let host: String
    }

    /// The store id a discovered robot gets: keyed by its ADVERTISED NAME, which is
    /// stable across networks, while its LAN IP is not.
    static func discoveredID(_ name: String) -> String { "lan:" + name }

    /// Fold Bonjour discoveries into the robot list.
    ///
    /// A host one robot already lists stays that robot's. Otherwise the advertised
    /// NAME decides: a "MABEL Sim (…)" advert is the built-in simulation on some Mac
    /// (folded into it, as before); ANY OTHER name is its own robot. This used to
    /// fold every unknown host into the simulation, so NOVA — a different, physical
    /// robot advertising the same `_mabel-teleop._tcp` — never appeared in the robot
    /// list at all: it was a hidden extra route of "Simulation", raced against the
    /// Mac's sim (the first TCP answer won), and auto-connect only ever tried thor
    /// and the sim. "It won't connect to NOVA even on the same Wi-Fi."
    func mergeDiscovered(named found: [DiscoveredRobot]) {
        for f in found {
            if let i = robots.firstIndex(where: { $0.routes.contains(f.host) }) {
                robots[i].online = true
                continue
            }
            let lname = f.name.lowercased()
            if f.name.isEmpty || (lname.hasPrefix("mabel") && lname.contains("sim")) {
                if let i = index("mabel-sim") {
                    robots[i].routes.append(f.host); robots[i].online = true
                }
                continue
            }
            let id = Self.discoveredID(f.name)
            if let i = index(id) {
                // Same robot, new network: its newest address goes first.
                robots[i].routes.removeAll { $0 == f.host }
                robots[i].routes.insert(f.host, at: 0)
                robots[i].online = true
            } else {
                if forgotten.contains(id) { continue }
                var r = Robot(id: id, name: Self.displayName(f.name),
                              kind: lname.contains("sim") ? .sim : .real,
                              routes: [f.host], saved: true)
                r.online = true
                robots.append(r)
            }
        }
        persist(); sort()
    }

    /// "NOVA Robot (mabel-orin)" → "NOVA Robot". The machine in brackets is useful
    /// in a Bonjour list and noise in the status bar; the id keeps the full name.
    static func displayName(_ advertised: String) -> String {
        if let r = advertised.range(of: " (", options: .backwards), advertised.hasSuffix(")") {
            let head = String(advertised[..<r.lowerBound]).trimmingCharacters(in: .whitespaces)
            if !head.isEmpty { return head }
        }
        return advertised
    }

    /// Robots found on THIS network right now (Bonjour), best first: real rigs
    /// before simulations. What auto-connect falls to when the default rig is away.
    func discoveredOnline(_ found: [DiscoveredRobot]) -> [Robot] {
        let hosts = Set(found.map(\.host))
        return robots.filter { r in r.routes.contains(where: hosts.contains) }
            .sorted { kindRank($0.kind) < kindRank($1.kind) }
    }

    func setOnline(id: String, _ online: Bool) {
        if let i = index(id) { robots[i].online = online }
    }

    // MARK: - Active

    func setActive(id: String?) { activeID = id }

    // MARK: - Server-owned name sync

    /// The connected robot reported its identity in `hello`/`robot_state`. Adopt
    /// the server-owned name + kind so all clients converge, and remember it.
    func adoptServerMeta(id activeRobotID: String?, serverID: String?, name: String?, kind: RobotKind) {
        guard let rid = activeRobotID, let i = index(rid) else { return }
        if kind != .unknown { robots[i].kind = kind }
        // The seeded robots (sim, thor) adopt the server-owned name so every client
        // converges; user-added robots KEEP the name the operator typed (otherwise a
        // generic "Simulation" from the server would clobber it and make two custom
        // sims indistinguishable).
        let seeded = Set(NetworkConfig.seedRobots().map(\.id))
        if seeded.contains(rid), let name, !name.isEmpty { robots[i].name = name }
        robots[i].saved = true
        robots[i].online = true
        persist(); sort()
    }

    // MARK: - User edits

    @discardableResult
    func rename(id: String, to name: String) -> Bool {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, let i = index(id) else { return false }
        robots[i].name = trimmed
        robots[i].saved = true
        persist(); sort()
        return robots[i].id == activeID
    }

    /// Add a manually-typed robot (one direct route).
    ///
    /// Each distinct host is its OWN remembered robot, keyed by that host — so two
    /// simulations on different machines (e.g. the Mac and the Alienware) coexist
    /// instead of collapsing into one. We only fold the host into an existing entry
    /// when some robot ALREADY reaches it (a true duplicate), never by assuming an
    /// unknown host belongs to the seeded sim.
    @discardableResult
    func add(host: String, name: String, ssh: String?) -> Robot {
        let h = host.trimmingCharacters(in: .whitespaces)
        let nm = name.trimmingCharacters(in: .whitespaces)
        let sshTrim = ssh?.trimmingCharacters(in: .whitespaces)
        // Same host already known → update that robot in place, don't duplicate it.
        if let i = robots.firstIndex(where: { $0.id == h || $0.routes.contains(h) }) {
            if !nm.isEmpty { robots[i].name = nm }
            if let s = sshTrim, !s.isEmpty { robots[i].ssh = s }
            robots[i].saved = true; persist(); sort(); return robots[i]
        }
        // Brand-new, independent robot — keep its host + name, persist it.
        forgotten.remove(h)                       // re-adding un-forgets it
        let r = Robot(id: h, name: nm.isEmpty ? "Robot (\(h))" : nm, kind: .unknown,
                      routes: [h],
                      ssh: (sshTrim?.isEmpty == false) ? sshTrim : nil, saved: true)
        robots.append(r); persist(); sort()
        return r
    }

    /// Forget ANY host — custom or built-in. We record the id so a seeded robot
    /// (sim / thor) doesn't silently re-appear on the next launch; re-adding it
    /// (Add robot) clears the suppression.
    func forget(id: String) {
        robots.removeAll { $0.id == id }
        forgotten.insert(id)
        if activeID == id { activeID = nil }
        persist()
    }

    /// True when a built-in robot (the sim, thor) is missing — i.e. it was
    /// forgotten. Drives the "Restore Built-in Robots" affordance.
    var hasForgottenBuiltIns: Bool {
        NetworkConfig.seedRobots().contains { robot(id: $0.id) == nil }
    }

    /// Bring the BUILT-IN robots back, whole.
    ///
    /// Forgetting the sim / thor is a one-way door otherwise: `seed()` skips any
    /// forgotten id forever, and re-adding the host by hand (Add Robot) builds a
    /// DEGRADED entry — one route — so the robot works on the LAN and is
    /// unreachable over Tailscale. Restoring re-seeds the full multi-route
    /// entries (LAN + Tailscale) instead of the hand-typed stub.
    ///
    /// Hand-typed entries that a seed already covers (their host IS one of its
    /// routes) are dropped as redundant — otherwise the same robot shows up twice,
    /// once crippled. Genuinely custom robots (a second sim on another machine) are
    /// untouched.
    func restoreDefaults() {
        forgotten.removeAll()
        let seeds = NetworkConfig.seedRobots()
        let seedRoutes = Set(seeds.flatMap(\.routes))
        robots.removeAll { r in
            !seeds.contains(where: { $0.id == r.id }) && seedRoutes.contains(r.id)
        }
        for s in seeds where robot(id: s.id) == nil { robots.append(s) }
        persist(); sort()
    }

    // MARK: - Persistence

    private func index(_ id: String) -> Int? { robots.firstIndex { $0.id == id } }

    private func sort() {
        robots.sort {
            if $0.online != $1.online { return $0.online && !$1.online }
            if $0.kind != $1.kind { return kindRank($0.kind) < kindRank($1.kind) }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }
    private func kindRank(_ k: RobotKind) -> Int {
        switch k { case .real: return 0; case .sim: return 1; case .unknown: return 2 }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(robots.filter { $0.saved }) {
            UserDefaults.standard.set(data, forKey: key)
        }
        UserDefaults.standard.set(Array(forgotten), forKey: forgottenKey)
    }
    private func load() {
        if let f = UserDefaults.standard.array(forKey: forgottenKey) as? [String] {
            forgotten = Set(f)
        }
        guard let data = UserDefaults.standard.data(forKey: key),
              let saved = try? JSONDecoder().decode([Robot].self, from: data) else { return }
        robots = saved
    }
}
