//
//  BLECentral.swift
//  MABEL Mobile App
//
//  Finds the robot over Bluetooth LE and decides when it is CLOSE ENOUGH to offer a
//  one-tap connection — the AirPods "bring it near the phone and a card slides up" feel,
//  for a robot.
//
//  WHAT THIS CAN AND CANNOT DO. The system-level AirPods card is Apple's proprietary
//  Proximity Pairing: a Continuity advertisement that only MFi-licensed accessories may
//  emit, and only Apple's own SpringBoard may render. No third-party hardware can trigger
//  it. What we CAN do — and what this file does — is reproduce the whole experience inside
//  the app: a passive BLE scan gives us the robot's live RSSI, and when it crosses into
//  arm's reach a card slides up with the robot spinning in 3D and a single Connect button.
//  With the `bluetooth-central` background mode iOS also wakes the app on first sighting,
//  so a notification can invite the operator in before they have opened anything.
//
//  The scan is passive and reads the robot's ADVERTISEMENT, so the card already knows the
//  robot's name, whether its e-stop is engaged and whether its gateway is up — all before
//  anyone connects to anything.
//

import Foundation
import Combine
import CoreBluetooth
#if os(iOS)
import UIKit
#endif

/// A robot seen on the air, with the smoothed proximity we judge "near" by.
struct NearbyRobot: Identifiable, Equatable {
    let id: UUID                       // CoreBluetooth's per-device identifier (stable per app install)
    var name: String
    var rssi: Int                      // smoothed, dBm
    var beacon: BLEProtocol.Beacon
    var lastSeen: Date

    /// Very rough free-space distance, for a "closer / further" hint only. BLE RSSI is far
    /// too noisy to show as a number, so the UI uses it for phrasing, never as a readout.
    var approxMetres: Double {
        // Measured power at 1 m for a typical USB adapter at 0 dBm TX.
        let txAt1m = -59.0
        return pow(10.0, (txAt1m - Double(rssi)) / 20.0)
    }

    var proximity: Proximity {
        if rssi >= -55 { return .immediate }
        if rssi >= -70 { return .near }
        return .far
    }

    enum Proximity { case immediate, near, far }

    static func == (a: NearbyRobot, b: NearbyRobot) -> Bool {
        a.id == b.id && a.rssi == b.rssi && a.name == b.name
            && a.beacon.estop == b.beacon.estop && a.beacon.gatewayUp == b.beacon.gatewayUp
    }
}

@MainActor
final class BLECentral: NSObject, ObservableObject {

    static let shared = BLECentral()

    /// Everything currently on the air, strongest first.
    @Published private(set) var nearby: [NearbyRobot] = []
    /// The robot the proximity card should offer, or nil for "show nothing".
    @Published private(set) var promoted: NearbyRobot?
    @Published private(set) var poweredOn = false
    @Published private(set) var unauthorized = false
    @Published private(set) var scanning = false

    /// Thresholds. Promote at arm's reach, drop only well outside it — the gap is
    /// deliberate hysteresis, so a card cannot flicker while someone stands still at the
    /// boundary holding the phone.
    private let promoteRSSI = -58
    private let dropRSSI = -78
    private let staleAfter: TimeInterval = 6

    private var manager: CBCentralManager!
    private var smoothed: [UUID: Double] = [:]
    /// Robots the operator has waved away — never offered again until they leave and
    /// come back, so "not now" is respected without being permanent.
    private var dismissed: Set<UUID> = []
    /// Suppressed because we are already connected to it.
    private var suppressed: Set<UUID> = []
    private var sweeper: Timer?

    private override init() {
        super.init()
        // No CBCentralManagerOptionShowPowerAlertKey: we surface Bluetooth-off in our own
        // UI rather than throwing a system alert at someone who only opened the app.
        manager = CBCentralManager(delegate: self, queue: .main,
                                   options: [CBCentralManagerOptionRestoreIdentifierKey:
                                                "com.mabel.ble.central"])
    }

    /// The manager, for a `BLELink` to connect with — a peripheral can only be connected
    /// by the same central that discovered it.
    var centralManager: CBCentralManager { manager }

    // MARK: - Scanning

    /// Screenshot harness only: the Simulator has no Bluetooth radio, so the
    /// radio's "powered off" callback would wipe any stand-in we planted.
    static var debugStandIn: Bool {
        #if DEBUG
        let n = UserDefaults.standard.string(forKey: "debugNearbyRobot")
        return !(n ?? "").isEmpty
        #else
        return false
        #endif
    }

    func startScanning() {
        #if DEBUG
        //   defaults write <bundle> debugNearbyRobot -string "MABEL-01"
        if let name = UserDefaults.standard.string(forKey: "debugNearbyRobot"), !name.isEmpty {
            let fake = NearbyRobot(id: UUID(uuidString: "00000000-0000-0000-0000-00000000BEEF")!,
                                   name: name, rssi: -48,
                                   beacon: BLEProtocol.Beacon(version: 1, isNova: true,
                                                              estop: false, torque: true,
                                                              gatewayUp: true),
                                   lastSeen: Date())
            scanning = true
            nearby = [fake]
            promoted = fake
            return
        }
        #endif
        guard poweredOn, !scanning else { return }
        scanning = true
        // Duplicates ON: without them iOS reports each device once and the RSSI never
        // updates, so "is it near me yet?" can never be answered. Foreground only — iOS
        // ignores this flag in the background, which is fine, there we just want the wake.
        manager.scanForPeripherals(withServices: [CBUUID(string: BLEProtocol.UUIDs.service)],
                                   options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
        sweeper = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sweep() }
        }
    }

    func stopScanning() {
        guard scanning else { return }
        scanning = false
        manager.stopScan()
        sweeper?.invalidate(); sweeper = nil
    }

    // MARK: - Card lifecycle

    /// "Not now" — hide this robot's card until it goes away and comes back.
    func dismiss(_ robot: NearbyRobot) {
        dismissed.insert(robot.id)
        if promoted?.id == robot.id { promoted = nil }
    }

    /// Stop offering a robot we are now driving.
    func suppress(_ id: UUID) {
        suppressed.insert(id)
        if promoted?.id == id { promoted = nil }
    }

    func unsuppress(_ id: UUID) {
        suppressed.remove(id)
    }

    func peripheral(for id: UUID) -> CBPeripheral? {
        manager.retrievePeripherals(withIdentifiers: [id]).first
    }

    // MARK: - Internals

    /// Drop anything we have stopped hearing from, and retire its card.
    private func sweep() {
        let now = Date()
        let gone = nearby.filter { now.timeIntervalSince($0.lastSeen) > staleAfter }
        guard !gone.isEmpty else { return }
        for g in gone {
            smoothed[g.id] = nil
            dismissed.remove(g.id)          // left the room → a future approach may offer again
        }
        let goneIDs = Set(gone.map(\.id))
        nearby.removeAll { goneIDs.contains($0.id) }
        if let p = promoted, goneIDs.contains(p.id) { promoted = nil }
    }

    private func ingest(id: UUID, name: String, rssi: Int, beacon: BLEProtocol.Beacon) {
        // Exponential smoothing: raw BLE RSSI swings 10-15 dB sample to sample just from
        // the phone turning in someone's hand, which is more than the entire promote/drop
        // gap. Unsmoothed, the card would strobe.
        let prior = smoothed[id] ?? Double(rssi)
        let value = prior + 0.25 * (Double(rssi) - prior)
        smoothed[id] = value

        let robot = NearbyRobot(id: id, name: name, rssi: Int(value.rounded()),
                                beacon: beacon, lastSeen: Date())
        if let i = nearby.firstIndex(where: { $0.id == id }) { nearby[i] = robot }
        else { nearby.append(robot) }
        nearby.sort { $0.rssi > $1.rssi }

        // Promote / retire.
        if robot.rssi >= promoteRSSI, promoted == nil,
           !dismissed.contains(id), !suppressed.contains(id) {
            promoted = robot
            #if os(iOS)
            UIImpactFeedbackGenerator(style: .soft).impactOccurred()
            #endif
        } else if promoted?.id == id {
            if robot.rssi <= dropRSSI { promoted = nil } else { promoted = robot }
        }
    }
}

extension BLECentral: CBCentralManagerDelegate {

    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in
            poweredOn = central.state == .poweredOn
            unauthorized = central.state == .unauthorized
            if poweredOn { startScanning() }
            else if !Self.debugStandIn { scanning = false; nearby.removeAll(); promoted = nil }
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager,
                                    willRestoreState dict: [String: Any]) {
        // Woken by iOS after a background sighting. Nothing to rebuild: the scan restarts
        // from `centralManagerDidUpdateState`, and any live link re-announces itself.
    }

    nonisolated func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                                    advertisementData: [String: Any], rssi RSSI: NSNumber) {
        // iOS reports 127 for "RSSI unavailable"; it is not a real reading.
        let raw = RSSI.intValue
        guard raw < 0, raw > -120 else { return }

        let adName = (advertisementData[CBAdvertisementDataLocalNameKey] as? String)
            ?? peripheral.name ?? "Robot"
        var beacon = BLEProtocol.Beacon()
        if let mfg = advertisementData[CBAdvertisementDataManufacturerDataKey] as? Data,
           mfg.count >= 5 {
            // [company lo, company hi, payload…] — little-endian company id, per the spec.
            let company = UInt16(mfg[0]) | UInt16(mfg[1]) << 8
            if company == BLEProtocol.companyID,
               let b = BLEProtocol.decodeBeacon(mfg.subdata(in: 2..<mfg.count)) {
                beacon = b
            }
        }
        let id = peripheral.identifier
        // assumeIsolated, not a Task hop: with duplicates on, this fires for EVERY
        // advertisement packet (tens per second per robot), and the manager was created
        // with `queue: .main`, so we are already on the main actor — allocating a Task
        // per packet would be pure overhead.
        MainActor.assumeIsolated {
            self.ingest(id: id, name: adName, rssi: raw, beacon: beacon)
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        NotificationCenter.default.post(name: .bleDidConnect, object: peripheral)
    }

    nonisolated func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral,
                                    error: Error?) {
        NotificationCenter.default.post(name: .bleDidDisconnect, object: peripheral)
    }

    nonisolated func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral,
                                    error: Error?) {
        NotificationCenter.default.post(name: .bleDidDisconnect, object: peripheral)
    }
}

extension Notification.Name {
    static let bleDidConnect = Notification.Name("nova.ble.didConnect")
    static let bleDidDisconnect = Notification.Name("nova.ble.didDisconnect")
}
