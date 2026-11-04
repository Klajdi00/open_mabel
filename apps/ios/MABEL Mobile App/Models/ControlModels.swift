//
//  ControlModels.swift
//  MABEL Mobile App
//
//  The three orthogonal teleop axes, mirroring the bridge's wire_protocol:
//
//      INPUT  ─▶  TELEOP METHOD  ─▶  CONTROLLER (type × region)  ─▶  ROBOT
//
//  • TeleopMethod  — how the operator's intent becomes a robot reference.
//  • ControllerType — how the robot realises it (rigid Hold vs soft Soft).
//  • ControlRegion — which part of the body the controller owns.
//
//  Keeping them separate (not one tangled "mode" enum) is what makes the Body
//  tab simple: one small picker per axis. The rawValues are the wire strings.
//

import SwiftUI

/// How the human's intent is mapped to a robot reference.
enum TeleopMethod: String, CaseIterable, Identifiable {
    case joint                       // per-joint targets (sliders / toggles)
    case navigation                  // base twist + lift (drive joysticks)
    case wrist                       // palm target → arm IK (ball drag / joystick)
    case retarget                    // whole-body IK from headset tracking

    var id: String { rawValue }
}

/// How the robot realises a reference — the control law on the joints.
enum ControllerType: String, CaseIterable, Identifiable {
    case impedance                   // rigid: track the target with a spring
    case compliance                  // soft: yield to a push (admittance)

    var id: String { rawValue }

    /// Friendly, Apple-plain UI label. NOTE: both modes hold position — the real
    /// distinction is stiffness, so we name them by feel (Stiff vs Soft), not by
    /// "hold" (which would be true of both). rawValue stays impedance/compliance.
    var title: String {
        switch self {
        case .impedance: return "Stiff"
        case .compliance: return "Soft"
        }
    }

    var icon: String {
        switch self {
        case .impedance: return "lock.fill"
        case .compliance: return "hand.draw.fill"
        }
    }

    var blurb: String {
        switch self {
        case .impedance: return "Stiff — the arm tracks your target and resists pushes (holds firm)."
        case .compliance: return "Soft — the arm goes weightless; guide it by hand and it stays."
        }
    }
}

/// Which part of the body the controller owns.
enum ControlRegion: String, CaseIterable, Identifiable {
    case arm        = "arm"          // arms only
    case upperBody  = "upper_body"   // + torso (+ lift held)
    case wholeBody  = "whole_body"   // + mobile base

    var id: String { rawValue }

    var title: String {
        switch self {
        case .arm: return "Arm"
        case .upperBody: return "Upper"
        case .wholeBody: return "Whole"
        }
    }

    var icon: String {
        switch self {
        case .arm: return "figure.arms.open"
        case .upperBody: return "figure"
        case .wholeBody: return "figure.walk"
        }
    }
}
