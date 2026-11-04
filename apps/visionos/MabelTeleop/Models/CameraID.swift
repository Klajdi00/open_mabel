//
//  CameraID.swift
//  MabelTeleop
//

import Foundation

public enum CameraID: String, CaseIterable, Codable, Sendable, Identifiable {
    case main
    case headLeft   = "head_left"
    case headRight  = "head_right"
    case leftWrist  = "left_wrist"
    case rightWrist = "right_wrist"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .main:       return "Main"
        case .headLeft:   return "Head Left"
        case .headRight:  return "Head Right"
        case .leftWrist:  return "Left Wrist"
        case .rightWrist: return "Right Wrist"
        }
    }

    /// The left/right eye source for the stereo head view.
    public static let stereoHead: (left: CameraID, right: CameraID) = (.headLeft, .headRight)
}
