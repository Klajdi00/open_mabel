//
//  ActionCatalog.swift
//  MABEL
//
//  What the robot can be ASKED to do on a timeline. The table itself is
//  generated from the web control studio's motions.js registry
//  (ActionCatalog+Generated.swift) so the app and the studio always offer the
//  same vocabulary; this file is the type layer and the live refresh.
//

import Combine
import Foundation

/// Which track an action occupies. The robot can drive and manipulate at the
/// same time, so these are genuinely parallel; `both` drops a linked pair.
enum ActionLane: String, Codable, CaseIterable, Identifiable {
    case navigation, manipulation, language, both
    var id: String { rawValue }

    /// `both` is an authoring convenience, never a track you can see.
    static var tracks: [ActionLane] { [.navigation, .manipulation, .language] }

    var title: String {
        switch self {
        case .navigation:   return "Navigation"
        case .manipulation: return "Manipulation"
        case .language:     return "Language"
        case .both:         return "Both"
        }
    }
    /// Fits the track gutter without truncating — "MANIPULATION" does not.
    var shortTitle: String {
        switch self {
        case .navigation:   return "Nav"
        case .manipulation: return "Manip"
        case .language:     return "Voice"
        case .both:         return "Both"
        }
    }

    var icon: String {
        switch self {
        case .navigation:   return "location.north.line.fill"
        case .manipulation: return "hand.raised.fill"
        case .language:     return "waveform"
        case .both:         return "square.split.2x1"
        }
    }
}

enum ActionCategory: String, Codable, CaseIterable {
    case animation, policy, nav, pose, transition, language

    var title: String {
        switch self {
        case .animation:  return "Animations"
        case .policy:     return "Policies"
        case .nav:        return "Navigation"
        case .pose:       return "Poses"
        case .transition: return "Transitions"
        case .language:   return "Speech"
        }
    }
}

struct ActionParam: Codable, Hashable, Identifiable {
    let id: String
    let label: String
    let def: Double
    let min: Double
    let max: Double
}

struct ActionLocation: Codable, Hashable {
    let x: Double
    let y: Double
    let label: String
}

/// One entry in the registry — the TEMPLATE. A block on the timeline is an
/// instance of one of these with its own start, length and parameter values.
struct ActionSpec: Codable, Hashable, Identifiable {
    let id: String
    let title: String
    let emoji: String
    let duration: TimeInterval
    let lane: ActionLane
    let category: ActionCategory
    /// `false` = declared but not deployable (learned policies without a
    /// checkpoint). Shown, greyed, never addable — the studio is honest about
    /// this and so is the app.
    let available: Bool
    /// May be stretched past its natural duration (poses and holds).
    let holdable: Bool
    let isGoal: Bool
    /// Motion authored as a JS closure in the studio; see the generator.
    let clientMath: Bool
    let location: ActionLocation?
    let params: [ActionParam]

    var defaults: [String: Double] {
        Dictionary(uniqueKeysWithValues: params.map { ($0.id, $0.def) })
    }
}

/// The registry. Bundled so the editor works with no robot attached (a timeline
/// is a document — you should be able to write one on a train), refreshed from
/// `/api/actions` when a robot is reachable because availability is a property
/// of the ROBOT: `move to goal` needs the ROS stack up, and a policy becomes
/// real the day its checkpoint deploys.
@MainActor
final class ActionCatalog: ObservableObject {
    @Published private(set) var actions: [ActionSpec] = ActionCatalog.all

    /// The generated registry plus the actions that exist only here. `Say` has no
    /// clip in motions.js — speech is not a MOTION — but a choreography without a
    /// voice track is half a performance, so the Language lane gets a first-class
    /// block whose "parameter" is its text.
    ///
    /// PER ROBOT, like `MotionClips`' table: NOVA's studio runs its own copy of
    /// the registry, so the library offered is the connected robot's
    /// (`RobotRig.isNova` → `bundledNova`, generated from NOVA's motions.js).
    static var all: [ActionSpec] {
        (RobotRig.isNova ? bundledNova : bundled) + [
            ActionSpec(id: "say", title: "Say", emoji: "💬", duration: 3,
                       lane: .language, category: .language, available: true,
                       holdable: true, isGoal: false, clientMath: false,
                       location: nil, params: []),
            ActionSpec(id: "say_wait", title: "Beat (silence)", emoji: "…", duration: 1,
                       lane: .language, category: .language, available: true,
                       holdable: true, isGoal: false, clientMath: false,
                       location: nil, params: []),
        ]
    }
    @Published private(set) var refreshedAt: Date?
    private var rigObserver: NSObjectProtocol?

    init() {
        // A different robot (hello.robot) means a different registry copy: reload,
        // exactly as `MotionClips` reloads its table on the same notification.
        rigObserver = NotificationCenter.default.addObserver(forName: .robotRigChanged, object: nil,
                                                             queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.actions = ActionCatalog.all
                self?.refreshedAt = nil
            }
        }
    }
    deinit { rigObserver.map(NotificationCenter.default.removeObserver) }

    func spec(_ id: String) -> ActionSpec? { actions.first { $0.id == id } }

    func grouped() -> [(ActionCategory, [ActionSpec])] {
        ActionCategory.allCases.compactMap { cat in
            let items = actions.filter { $0.category == cat }
            return items.isEmpty ? nil : (cat, items)
        }
    }

    /// Merge server availability onto the bundled table. The server's registry
    /// is thinner (it has no durations or parameter ranges), so it is treated as
    /// an AVAILABILITY OVERLAY rather than a replacement — otherwise connecting
    /// to a robot would strip the editor of everything the studio can author.
    func refresh(base: URL) async {
        struct Row: Decodable { let id: String; let status: String? }
        var req = URLRequest(url: base.appendingPathComponent("api/actions"))
        req.timeoutInterval = 6
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let rows = try? JSONDecoder().decode([Row].self, from: data)
        else { return }
        let live = Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0.status ?? "available") })
        actions = ActionCatalog.all.map { spec in
            guard let status = live[spec.id] else { return spec }
            return ActionSpec(
                id: spec.id, title: spec.title, emoji: spec.emoji,
                duration: spec.duration, lane: spec.lane, category: spec.category,
                available: status == "available", holdable: spec.holdable,
                isGoal: spec.isGoal, clientMath: spec.clientMath,
                location: spec.location, params: spec.params)
        }
        refreshedAt = Date()
    }
}
