//
//  ConnectionStatus.swift
//  MABEL Mobile App
//

import SwiftUI

enum ConnectionStatus: Equatable {
    case disconnected
    case connecting
    case connected
    case demo            // running against the on-device simulator

    var label: String {
        switch self {
        case .disconnected: return "Offline"
        case .connecting:   return "Connecting…"
        case .connected:    return "Connected"
        case .demo:         return "Demo Mode"
        }
    }

    var color: Color {
        switch self {
        case .disconnected: return Theme.Palette.danger
        case .connecting:   return Theme.Palette.warn
        case .connected:    return Theme.Palette.live
        case .demo:         return Theme.Palette.accent
        }
    }

    var isLive: Bool { self == .connected || self == .demo }
}

/// What the app is actually driving — a simulation (on-device demo OR the Mac
/// MuJoCo bridge) or physical hardware. Learned from the server's `hello`.
enum ConnectionTarget: Equatable {
    case simulation
    case realRobot
    case unknown

    var label: String {
        switch self {
        case .simulation: return "SIM"
        case .realRobot:  return "REAL"
        case .unknown:    return "—"
        }
    }

    var icon: String {
        switch self {
        case .simulation: return "cube.transparent"       // virtual / simulated twin
        case .realRobot:  return "gearshape.2.fill"       // physical machine (system symbol)
        case .unknown:    return "questionmark"
        }
    }

    var color: Color {
        switch self {
        case .simulation: return Theme.Palette.accent
        case .realRobot:  return Theme.Palette.live
        case .unknown:    return Theme.Palette.textTertiary
        }
    }

    var isSimulation: Bool { self == .simulation }
}
