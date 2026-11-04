//
//  BodyControls.swift
//  MabelTeleop
//
//  WHICH PARTS OF THE BODY THE WHOLE-BODY RETARGETER DRIVES — one row, used
//  verbatim by the window's top bar AND the full-immersion control deck, so the
//  two can never disagree (operator, 2026-10-16: ablation studies — "freeze the
//  active neck … freeze the active torso … upper body only vs whole body …
//  re-anchor and re-calibration are the same icon … text to clearly show what
//  they do … no ambiguity").
//
//      [ Upper body | Whole body ]  [ Neck · Live ]  [ Torso · Live ]  │  [ Re-anchor forward ]  [ Recalibrate reach ]
//
//  It is ONE retargeter in every combination: a frozen part is PARKED at home
//  after it on the server (TeleopEngine.park — neck level & forward, torso 0,
//  lift at the home height; Upper body also takes the base out of the loop).
//  Every state is in words, and "frozen" always shows a lock, so a screenshot
//  of a trial says which condition it was. The same state is written into every
//  recorded frame (server curation_recorder.py, obs/control/*).
//

import SwiftUI

/// One body-part switch. LIVE = the part follows you; FROZEN = held at home.
struct BodyPartChip: View {
    let part: String
    let icon: String
    let live: Bool
    var action: () -> Void

    var body: some View {
        Button(action: { Haptics.select(); action() }) {
            // Glass applied DIRECTLY to the label (as every other bar control
            // does): wrapped in `.background {}` the glass layer drew OVER the
            // label and the chip rendered blank (seen in the simulator).
            let label = HStack(spacing: 6) {
                Image(systemName: live ? icon : "lock.fill")
                    .font(.system(size: 12, weight: .bold))
                Text("\(part) · \(live ? "Live" : "Frozen")")
                    .font(.system(size: 13, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(live ? Color.white : Color.black)
            .padding(.horizontal, 12)
            .frame(height: 36)
            .contentShape(Capsule(style: .continuous))
            if live {
                label.glassBackgroundEffect(in: Capsule(style: .continuous))
            } else {
                label.background(Theme.Palette.warn, in: Capsule(style: .continuous))
            }
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
        .accessibilityLabel("\(part) \(live ? "live" : "frozen")")
        .accessibilityIdentifier("body.\(part.lowercased())")
        .help(live ? "\(part) follows you — tap to freeze it at home"
                   : "\(part) frozen at home — tap to let it follow you again")
    }
}

/// A labelled verb in the same row (always icon + text).
struct BodyVerbButton: View {
    let title: String
    let icon: String
    let help: String
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.system(size: 13, weight: .bold))
                Text(title).font(.system(size: 13, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 13)
            .frame(height: 36)
            .contentShape(Capsule(style: .continuous))
            .glassBackgroundEffect(in: Capsule(style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
        .help(help)
    }
}

struct BodyControls: View {
    @EnvironmentObject private var controller: RobotController
    /// nil hides Recalibrate (e.g. no tracking session to measure with).
    var onRecalibrate: (() -> Void)?

    var body: some View {
        HStack(spacing: 8) {
            GlassSegmented(options: RetargetScope.allCases,
                           selection: Binding(get: { controller.retargetScope },
                                              set: { sc in
                                                  guard sc != controller.retargetScope else { return }
                                                  Haptics.select()
                                                  withAnimation(Theme.Motion.snappy) { controller.retargetScope = sc }
                                              }),
                           title: { $0.title },
                           icon: { $0.icon },
                           prominent: false)
                .fixedSize()
                .help(controller.retargetScope.help)
                .accessibilityIdentifier("body.scope")
            BodyPartChip(part: "Neck", icon: "eye", live: controller.trackNeck) {
                controller.trackNeck.toggle()
            }
            BodyPartChip(part: "Torso", icon: "figure.stand", live: controller.trackTorso) {
                controller.trackTorso.toggle()
            }
            Rectangle().fill(.white.opacity(0.18)).frame(width: 1, height: 22)
            BodyVerbButton(title: "Re-anchor forward", icon: "scope",
                           help: "Face the way you want to be straight ahead, then press — the robot's forward is re-zeroed there") {
                controller.setForward()
            }
            if let onRecalibrate {
                BodyVerbButton(title: "Recalibrate reach", icon: "ruler",
                               help: "Re-measure your arm length (the 1-pose capture) — reach scale",
                               action: { Haptics.select(); onRecalibrate() })
            }
        }
        .fixedSize()
    }
}
