//
//  KnownRobots.swift
//  MABEL Mobile App
//
//  The robots this device has connected to over Bluetooth before, remembered across
//  launches — so Settings can offer "Connect" to a robot that is not currently being
//  scanned for, and so a robot has a name the operator chose rather than whatever it
//  advertises.
//
//  WHY THIS EXISTS: BLE discovery only ever showed what was on the air RIGHT NOW. Walk
//  into the lab with the app already open and the list was empty until a scan happened to
//  catch an advertisement, with nothing to tap. CoreBluetooth can reconnect to a peripheral
//  it has seen before purely from its identifier — `retrievePeripherals(withIdentifiers:)`
//  needs no scan at all — so the only thing missing was remembering the identifier.
//
//  THE IDENTIFIER IS PER-INSTALL. CoreBluetooth hands out a UUID that is stable for this
//  app on this device and is NOT the robot's MAC address; reinstalling the app, or using a
//  different phone, produces a different one. That is why a saved entry stores the name
//  alongside it, and why "forget" has to exist: a stale identifier can never be matched
//  back to a robot by any other means.
//
//  THE NICKNAME IS LOCAL. Renaming here labels the robot on THIS device; it does not write
//  anything to the robot, whose advertised name comes from NOVA_BLE_NAME on the robot
//  itself. Both are shown when they differ, so a rename can never hide which machine is
//  actually being connected to.
//

import Combine
import Foundation

struct KnownRobot: Codable, Identifiable, Equatable {
    /// CoreBluetooth's per-install peripheral identifier.
    let id: UUID
    /// What the robot advertised the last time we connected to it.
    var advertisedName: String
    /// What the operator called it, if anything.
    var nickname: String?
    var lastConnected: Date

    /// What the UI shows. The nickname wins; the advertised name is the fallback.
    var displayName: String {
        let n = nickname?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return n.isEmpty ? advertisedName : n
    }
    /// True when the operator's label differs from what the robot calls itself, so the UI
    /// can show both instead of quietly renaming a machine out from under them.
    var isRenamed: Bool { displayName != advertisedName }
}

@MainActor
final class KnownRobots: ObservableObject {

    static let shared = KnownRobots()

    /// Most recently connected first — the one you want is almost always the last one.
    @Published private(set) var robots: [KnownRobot] = []

    private let key = "bluetooth.knownRobots"
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        load()
    }

    // MARK: Reading

    func robot(id: UUID) -> KnownRobot? { robots.first { $0.id == id } }
    var isEmpty: Bool { robots.isEmpty }

    // MARK: Writing

    /// Record a successful connection. Keeps any nickname the operator already gave it and
    /// refreshes the advertised name, which is the robot's to change.
    func remember(id: UUID, advertisedName: String) {
        var list = robots
        if let i = list.firstIndex(where: { $0.id == id }) {
            list[i].advertisedName = advertisedName
            list[i].lastConnected = Date()
        } else {
            list.append(KnownRobot(id: id, advertisedName: advertisedName,
                                   nickname: nil, lastConnected: Date()))
        }
        save(list)
    }

    /// Empty or whitespace clears the nickname back to the advertised name rather than
    /// storing a blank label.
    func rename(id: UUID, to name: String) {
        guard var list = Optional(robots), let i = list.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        list[i].nickname = trimmed.isEmpty ? nil : trimmed
        save(list)
    }

    func forget(id: UUID) {
        save(robots.filter { $0.id != id })
    }

    // MARK: Storage

    private func load() {
        guard let data = defaults.data(forKey: key),
              let decoded = try? JSONDecoder().decode([KnownRobot].self, from: data) else { return }
        robots = decoded.sorted { $0.lastConnected > $1.lastConnected }
    }

    private func save(_ list: [KnownRobot]) {
        robots = list.sorted { $0.lastConnected > $1.lastConnected }
        // A failure here costs the memory of a pairing, never a connection: everything
        // above works from the in-memory list for the rest of this launch.
        if let data = try? JSONEncoder().encode(robots) {
            defaults.set(data, forKey: key)
        }
    }
}
