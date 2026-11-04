//
//  LinkPriority.swift
//  MabelTeleop
//
//  WHICH TRANSPORT WINS — in one place, as a pure function, so every client agrees and it
//  can be tested without a robot, a radio or a network.
//
//  The order is by what the link can actually carry, best first:
//
//    1. Wi-Fi LAN      full protocol + video, lowest latency — the robot is on your network
//    2. Tailscale      same protocol, over the tunnel — same LAN quality, more hops
//    3. Bluetooth      NO VIDEO and a 102-byte budget per tick, but it needs no network at
//                      all. The right answer when nothing else is reachable and the wrong
//                      one whenever something else is.
//
//  `ConnectionPlanner` already races the NETWORK routes and returns them fastest
//  first (LAN beats Tailscale when both answer). This type adds the
//  Bluetooth tier below them and decides the transitions between tiers, which is the part
//  that used to be manual: the app would sit on Bluetooth with the robot plainly on the
//  same Wi-Fi, and a Wi-Fi drop left you with nothing while the robot was an arm's length
//  away.
//
//  TWO RULES ABOUT MOVING BETWEEN TIERS, and they are deliberately asymmetric:
//    · Moving UP (Bluetooth → network) is automatic. It strictly gains capability, and the
//      operator asked for exactly this: if Wi-Fi is there, use it.
//    · Moving DOWN (network → Bluetooth) ASKS FIRST. It silently removes video and most of
//      the protocol, and the phone is often still nominally "on Wi-Fi" when the robot has
//      gone — dropping to Bluetooth without a word would look like the app losing features
//      by itself. The one exception is a first connection, where there is nothing to lose.
//

import Foundation

enum LinkPriority {

    /// Lower is better. Anything not a real robot link ranks last.
    static func rank(_ kind: RouteKind?) -> Int {
        switch kind {
        case .lan:       return 0
        case .tailscale: return 1
        case .bluetooth: return 2
        case .demo, nil: return 99
        }
    }

    /// Is `candidate` strictly better than what we are on now?
    static func isUpgrade(from current: RouteKind?, to candidate: RouteKind) -> Bool {
        rank(candidate) < rank(current)
    }

    enum Decision: Equatable {
        /// Nothing to do — already on the best thing available.
        case stay
        /// Connect (or move) to this network route now.
        case connect(ConnectionRoute)
        /// Nothing on the network; take Bluetooth without asking (first connection).
        case connectBluetooth
        /// We had a network link and lost it, and Bluetooth is in range: ask the operator.
        case offerBluetooth
        /// Nothing at all is reachable.
        case nothingReachable
    }

    /// The whole policy.
    ///
    /// - Parameters:
    ///   - current: the route we are on, or nil when not connected.
    ///   - live: is that route actually carrying state right now?
    ///   - network: reachable network routes, best first (`ConnectionPlanner.plan`).
    ///   - bluetoothInRange: a robot advertising over BLE, if any.
    ///   - hadNetwork: were we on a network route before this moment? Decides whether
    ///     dropping to Bluetooth is a fallback (ask) or a first connection (just do it).
    static func decide(current: RouteKind?,
                       live: Bool,
                       network: [ConnectionRoute],
                       bluetoothInRange: Bool,
                       hadNetwork: Bool) -> Decision {
        // A live link that nothing outranks: leave it alone. Re-connecting a working link
        // to shave a few milliseconds is how you drop a frame mid-drive.
        if live, let best = network.first, !isUpgrade(from: current, to: best.kind) { return .stay }
        if live, network.isEmpty, current != nil { return .stay }

        // Something better than what we are on (or we are on nothing): take it.
        if let best = network.first, !live || isUpgrade(from: current, to: best.kind) {
            return .connect(best)
        }
        // No network at all.
        guard bluetoothInRange else { return live ? .stay : .nothingReachable }
        if live, current == .bluetooth { return .stay }
        return hadNetwork ? .offerBluetooth : .connectBluetooth
    }
}
