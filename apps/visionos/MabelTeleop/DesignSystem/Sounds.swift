//
//  Sounds.swift
//  MabelTeleop
//
//  Tiny audible sibling of Haptics: short system chimes for moments the
//  operator may not be LOOKING at (arms extended forward during calibration,
//  eyes on the robot). System sound IDs need no bundled assets and play on
//  the headset speakers (Mac speakers in the simulator).
//

import AudioToolbox

enum Sounds {
    /// Clear success "ding" — calibration captured / a capture-grade success.
    /// 1407 is the payment-success chime: short, bright, unmistakable.
    static func success() {
        AudioServicesPlaySystemSound(1407)
    }

    /// Short countdown tick (3·2·1 before whole-body tracking engages).
    /// 1104 is the keyboard tock: quiet, percussive, reads as a metronome.
    static func tick() {
        AudioServicesPlaySystemSound(1104)
    }
}
