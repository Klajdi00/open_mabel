//
//  Haptics.swift
//  MabelTeleop
//
//  visionOS has no haptic engine — this keeps the iOS app's call sites
//  compiling unchanged so ported views stay line-for-line comparable.
//  (Apple's guidance: spatial audio is the visionOS analogue; the system
//  already plays interaction sounds for buttons, so these are no-ops.)
//

import Foundation

enum Haptics {
    static func tap() {}
    static func select() {}
    static func soft() {}
    static func rigid() {}
    static func success() {}
    static func warning() {}
    static func error() {}
}
