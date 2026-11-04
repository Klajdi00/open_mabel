//
//  CalibrationProfile.swift
//  MabelTeleop
//
//  Per-operator calibration is ONE pose: extend both arms straight forward; the
//  robot measures your head->wrist reach against its own and sets the motion scale
//  (REACH_SCALE) so its hands cover your workspace. This file holds the live capture
//  progress (CalibrationProgress), the measured result the bridge streams back
//  (OperatorMetrics), and the named, on-device-saved profile (CalibrationProfile +
//  CalibrationStore). Server side: scripts/teleop/engine.py (TeleopEngine).
//

import SwiftUI
import Combine
import simd

// MARK: - Live capture coaching (on-device)

/// What the operator's body is doing RIGHT NOW during a capture.
struct CalibrationLive: Equatable {
    var hands: Int = 0          // wrists contributing (0 / 1 / 2)
    var reach: Double = 0       // m, head→wrist mean (the measured quantity)
    var steady: Bool = false    // held inside the server's stillness window
    var hold: Double = 0        // 0…1 through the server's hold time
    var hasHead: Bool = false
}

/// The live coaching signal behind the calibration card, kept PURE so it can be
/// tested without a headset (the simulator has no hand/head tracking).
///
/// It reproduces the server's measurement (`human_reach` in engine.py: the mean
/// distance from the head to each tracked WRIST — and ARKit's hand anchor IS the
/// wrist), so the number on screen is the quantity actually being captured, and
/// the ring fills in step with the hold the bridge is timing.
///
/// The SERVER stays authoritative — this never sets REACH_SCALE. It exists because
/// the capture was otherwise a black box: "extend your arms and hold" with no
/// evidence anything was happening, and silence when the hands weren't tracked.
struct CalibrationCoach {
    // The server's thresholds (engine.py `_calibrate_step` / hyperparams CAL_*),
    // mirrored so on-screen progress matches what the bridge is about to do. If
    // those change, change these — this is a UI echo of the server's rule, not a
    // second rule.
    // REDESIGNED 2026-07-19 ("the arm-length number stabilised" == success):
    // the old ≥5-samples-in-0.6 s + max−min ≤ 5 cm rule was rate-DEPENDENT — it
    // could never complete below ~8 Hz effective frame rate, and under real
    // tremor it got STRICTER the faster the stream (max−min grows with sample
    // count; server sweep: 60 Hz failed at 2 cm tremor where 8 Hz passed). The
    // rule is now time span + standard deviation (both rate-independent) and the
    // captured value is the window MEDIAN (single ARKit outliers don't matter).
    static let window: TimeInterval = 1.0    // s, rolling sample window (CAL_WINDOW)
    static let minSpan: TimeInterval = 0.5   // s the window must cover (CAL_MIN_SPAN)
    static let stdMax = 0.02                 // m, reach std that counts as still (CAL_STD)
    static let hold: TimeInterval = 0.25     // s of stillness before capture (CAL_HOLD)
    static let minReach = 0.1                // m, below this is not a real pose

    private var buf: [(t: TimeInterval, reach: Double)] = []
    private var steadySince: TimeInterval? = nil

    init() {}

    mutating func reset() { buf.removeAll(); steadySince = nil }

    /// Feed one sample of world-space positions; returns what to show.
    mutating func observe(head: SIMD3<Float>?,
                          leftWrist: SIMD3<Float>?,
                          rightWrist: SIMD3<Float>?,
                          now: TimeInterval) -> CalibrationLive {
        var live = CalibrationLive()

        // Head is required: human_reach() returns None without a head_mat, so no
        // head means the capture cannot progress at all — say so rather than
        // showing a hopeful spinner.
        guard let head else { reset(); return live }
        live.hasHead = true

        let ds = [leftWrist, rightWrist].compactMap { w -> Double? in
            guard let w else { return nil }
            return Double(simd_distance(head, w))
        }
        live.hands = ds.count
        guard !ds.isEmpty else { reset(); return live }

        let reach = ds.reduce(0, +) / Double(ds.count)
        live.reach = reach
        // Below the floor the server discards the frame outright.
        guard reach >= Self.minReach else { reset(); return live }

        buf.append((now, reach))
        buf.removeAll { $0.t < now - Self.window }
        let vals = buf.map(\.reach)
        // Same rule as the server, sample-for-sample: n ≥ 3, window SPAN ≥
        // minSpan (time-based, so any stream ≥ ~4 Hz can complete), reach STD ≤
        // stdMax. Show the window MEDIAN — the value the server will capture —
        // so the number on screen stabilises exactly when the rule is met.
        let sorted = vals.sorted()
        let median = sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        live.reach = median
        let mean = vals.reduce(0, +) / Double(vals.count)
        let std = (vals.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(vals.count))
            .squareRoot()
        let spanned = (buf.last?.t ?? now) - (buf.first?.t ?? now)
        let still = vals.count >= 3
            && spanned >= Self.minSpan
            && std <= Self.stdMax
        guard still else { steadySince = nil; return live }

        if steadySince == nil { steadySince = now }
        live.steady = true
        live.hold = min(1, (now - (steadySince ?? now)) / Self.hold)
        return live
    }
}

// MARK: - Live capture progress (telemetry)

/// Where the 1-pose capture is right now, from `robot_state.calibrationStatus`.
struct CalibrationProgress: Equatable {
    var active: Bool
    var pose: String?        // "arms_forward" | nil
    var index: Int           // 0 until captured, then 1
    var total: Int           // 1
    var done: Bool
    var reach: Double?       // measured REACH_SCALE once done
    // Live measurement (additive 2026-07-19, only while active): the server's
    // OWN median arm length + spread + hold progress — authoritative where the
    // local coach is an echo, so the card can show the number the bridge sees.
    var reachLive: Double?   // m, median head→wrist in the server's window
    var spread: Double?      // m, reach std in the window (≤ 0.02 = still)
    var hold: Double?        // 0…1 through the server's hold time

    init(active: Bool, pose: String?, index: Int, total: Int, done: Bool, reach: Double?,
         reachLive: Double? = nil, spread: Double? = nil, hold: Double? = nil) {
        self.active = active; self.pose = pose; self.index = index
        self.total = total; self.done = done; self.reach = reach
        self.reachLive = reachLive; self.spread = spread; self.hold = hold
    }

    init?(wire d: [String: Any]) {
        self.init(active: d["active"] as? Bool ?? false,
                  pose: d["pose"] as? String,
                  index: d["index"] as? Int ?? 0,
                  total: d["total"] as? Int ?? 1,
                  done: d["done"] as? Bool ?? false,
                  reach: d["reach"] as? Double,
                  reachLive: d["reachLive"] as? Double,
                  spread: d["spread"] as? Double,
                  hold: d["hold"] as? Double)
    }
}

// MARK: - Measured result (telemetry)

/// The measured profile the bridge streams in `robot_state.operatorProfile`.
struct OperatorMetrics: Equatable {
    var user: String = "operator"
    var reach: Double = 0           // REACH_SCALE = robot reach / human reach
    var humanReach: Double = 0      // m, head->wrist (arms forward)
    var robotReach: Double = 0      // m, robot head->wrist

    init() {}

    init?(wire d: [String: Any]) {
        func num(_ k: String) -> Double { (d[k] as? Double) ?? Double(d[k] as? Int ?? 0) }
        user = d["user"] as? String ?? "operator"
        reach = num("reach")
        humanReach = num("human_reach")
        robotReach = num("robot_reach")
        if reach <= 0.05 { return nil }    // not a real measurement yet
    }
}

// MARK: - Saved profile (persisted, named by the operator)

/// A named, on-device-saved calibration. `name` is also the server `user` key, so
/// re-applying it tells the bridge which saved reach scale to load.
struct CalibrationProfile: Codable, Identifiable, Equatable {
    var id: UUID = UUID()
    var name: String
    var reach: Double
    var humanReach: Double
    var robotReach: Double
    var createdAt: Date = Date()

    init(name: String, metrics m: OperatorMetrics, id: UUID = UUID(), createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.reach = m.reach
        self.humanReach = m.humanReach
        self.robotReach = m.robotReach
        self.createdAt = createdAt
    }

    /// One-line subtitle for a list row.
    var summary: String {
        String(format: "reach ×%.2f · arm %.0f cm", reach, humanReach * 100)
    }

    /// "Operator · ×1.12 · Oct 3" — what the "Use Saved Calibration" button shows,
    /// so the operator sees exactly which numbers they are about to drive with.
    var savedLabel: String {
        String(format: "%@ · ×%.2f · arm %.0f cm · ", name, reach, humanReach * 100)
            + createdAt.formatted(date: .abbreviated, time: .omitted)
    }

    /// Labelled rows for the detail / completion summary.
    var detailRows: [(String, String)] {
        [
            ("Motion scale", String(format: "×%.2f", reach)),
            ("Your reach", String(format: "%.0f cm", humanReach * 100)),
            ("Robot reach", String(format: "%.0f cm", robotReach * 100)),
        ]
    }
}

// MARK: - Persistence (UserDefaults, like RobotStore)

@MainActor
final class CalibrationStore: ObservableObject {
    @Published private(set) var profiles: [CalibrationProfile] = []
    /// The profile name currently applied on the robot (informational), if any.
    @Published var activeName: String? = nil

    private let key = "mabel.calibrationProfiles.v2"      // v2: reach-based (1-pose)
    private let activeKey = "mabel.calibrationProfiles.active"
    /// Injectable so tests persist into a throwaway suite, never the app's.
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        load()
        activeName = defaults.string(forKey: activeKey)
    }

    /// The saved calibration "Use Saved Calibration" applies — see
    /// `CalibrationStore.preferred(in:name:activeName:)`.
    func preferred(forName name: String) -> CalibrationProfile? {
        Self.preferred(in: profiles, name: name, activeName: activeName)
    }

    /// PURE choice of which saved numbers to use, in priority order:
    ///   1. the profile whose name matches the one typed on the card
    ///      (case/whitespace-insensitive) — "use MY saved numbers";
    ///   2. the profile last applied on the robot (`activeName`);
    ///   3. the most recently calibrated profile.
    /// nil only when nothing has ever been saved. Profiles with a nonsensical
    /// reach (≤ 0.05, the same floor `OperatorMetrics` rejects) are ignored.
    static func preferred(in profiles: [CalibrationProfile],
                                      name: String,
                                      activeName: String?) -> CalibrationProfile? {
        let usable = profiles.filter { $0.reach > 0.05 && $0.reach.isFinite }
        func match(_ n: String?) -> CalibrationProfile? {
            guard let n = n?.trimmingCharacters(in: .whitespacesAndNewlines), !n.isEmpty
            else { return nil }
            return usable.first { $0.name.caseInsensitiveCompare(n) == .orderedSame }
        }
        return match(name) ?? match(activeName)
            ?? usable.max { $0.createdAt < $1.createdAt }
    }

    /// Insert or replace by NAME (re-calibrating a name updates it in place).
    func upsert(_ profile: CalibrationProfile) {
        var p = profile
        if let i = profiles.firstIndex(where: { $0.name.caseInsensitiveCompare(profile.name) == .orderedSame }) {
            p.id = profiles[i].id
            p.createdAt = Date()
            profiles[i] = p
        } else {
            profiles.append(p)
        }
        sortAndPersist()
    }

    func remove(_ profile: CalibrationProfile) {
        profiles.removeAll { $0.id == profile.id }
        if activeName?.caseInsensitiveCompare(profile.name) == .orderedSame { setActive(nil) }
        sortAndPersist()
    }

    func setActive(_ name: String?) {
        activeName = name
        if let name { defaults.set(name, forKey: activeKey) }
        else { defaults.removeObject(forKey: activeKey) }
    }

    func isActive(_ profile: CalibrationProfile) -> Bool {
        activeName?.caseInsensitiveCompare(profile.name) == .orderedSame
    }

    private func sortAndPersist() {
        profiles.sort { $0.createdAt > $1.createdAt }
        if let data = try? JSONEncoder().encode(profiles) {
            defaults.set(data, forKey: key)
        }
    }
    private func load() {
        guard let data = defaults.data(forKey: key),
              let saved = try? JSONDecoder().decode([CalibrationProfile].self, from: data) else { return }
        profiles = saved.sorted { $0.createdAt > $1.createdAt }
    }
}

// MARK: - Stall watchdog (the capture that never finishes)

/// Has a capture waited long enough that the card should OFFER a way out?
///
/// Why (operator 2026-10-12: "calibration can never be completed again"): the
/// card only advances on the SERVER's `calibrationStatus.done`, and a gateway
/// that does not implement `body_calibration` at all (NOVA's) never sends a
/// status — the card sat on "Extend Both Arms Forward" forever. PURE so it is
/// testable on the Simulator (no hand tracking there).
enum CalibrationStall: Equatable {
    /// Keep waiting — nothing to offer yet.
    case waiting
    /// No `calibrationStatus` has arrived from the server at all: it almost
    /// certainly does not support calibration. Offer saved numbers / skip.
    case serverSilent
    /// The server is measuring but has not converged (hands out of view,
    /// tremor, ...). Offer saved numbers / skip alongside the live coaching.
    case notConverging

    /// No status at all for this long → `serverSilent`.
    static let silentAfter: TimeInterval = 10
    /// Status, but no capture for this long → `notConverging`. The measured
    /// capture completes in 0.8–2.6 s (exp_calibration_robustness.py), so 25 s
    /// is an order of magnitude past a capture that is going to happen.
    static let slowAfter: TimeInterval = 25

    static func verdict(elapsed: TimeInterval, serverHeard: Bool, done: Bool) -> CalibrationStall {
        if done { return .waiting }
        if !serverHeard { return elapsed >= silentAfter ? .serverSilent : .waiting }
        return elapsed >= slowAfter ? .notConverging : .waiting
    }
}
