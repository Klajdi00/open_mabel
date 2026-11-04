//
//  DriveControls.swift
//  MABEL Mobile App · MabelTeleop (visionOS)
//
//  The Drive cockpit as three SEPARATE controls (operator, 2026-10-16):
//
//      [ TRANSLATE 2-D stick ]   [ LIFT vertical slider ]   [ ROTATE 1-D stick ]
//
//  The old right stick mixed yaw (x) and lift (y) on one knob, so every turn
//  leaked a little lift and every lift nudge leaked a little turn. Splitting
//  them makes each axis impossible to bump by accident.
//
//  The LIFT slider is SPRING-LOADED (a rate, like the stick it replaces): push
//  up = raise, down = lower, let go = the column holds where it is. It still
//  rides `liftRate` → nav.ry, so the server's tip-safe drone path
//  (`_apply_stick_drone`) stays the one thing that moves the column while
//  driving. The live height is drawn as a fill behind the thumb.
//
//  BYTE-IDENTICAL in the iOS v2 and VisionOS V2 apps — edit one, copy to the
//  other (same rule as RecordingControls.swift).
//

import SwiftUI

/// A one-axis analog stick: a capsule track whose thumb springs back to the
/// centre on release. Reports −1…+1 — horizontal: right +, vertical: UP +.
struct AxisStick: View {
    enum Axis { case horizontal, vertical }

    let label: String
    let axis: Axis
    var tint: Color = Theme.Palette.accent
    /// Track length along the axis.
    var length: CGFloat = 150
    var thickness: CGFloat = 54
    /// Glyphs at the negative / positive ends (left/down, right/up).
    var negativeIcon: String
    var positiveIcon: String
    /// Optional 0…1 fill drawn along the track from the negative end (the
    /// lift's live height). nil = no gauge.
    var gauge: Double? = nil
    var readout: String? = nil
    var onChange: (Double) -> Void

    @State private var deflection: CGFloat = 0     // −1…+1, positive = right / up
    @State private var atEnd = false

    private var knob: CGFloat { thickness - 10 }
    private var travel: CGFloat { (length - knob) / 2 - 5 }
    private var horizontal: Bool { axis == .horizontal }

    var body: some View {
        VStack(spacing: 8) {
            if let readout {
                Text(readout)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.black.opacity(0.42), in: Capsule())
            }
            ZStack {
                // Dark glass track, like the cockpit's grip sliders: these sit
                // over a live 3-D scene, and a surface-coloured track vanished
                // into it (only the thumb read).
                Capsule().fill(.black.opacity(0.42))
                    .overlay { Capsule().strokeBorder(.white.opacity(0.18), lineWidth: 1) }
                if let gauge { gaugeFill(min(1, max(0, gauge))) }
                // Centre detent line + end glyphs.
                Capsule().fill(.white.opacity(0.22))
                    .frame(width: horizontal ? 2 : thickness - 22,
                           height: horizontal ? thickness - 22 : 2)
                endGlyphs
                Capsule()
                    .fill(tint.gradient)
                    .frame(width: knob, height: knob)
                    .overlay {
                        Image(systemName: horizontal ? "arrow.left.and.right" : "arrow.up.and.down")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(.black.opacity(0.35))
                    }
                    .shadow(color: tint.opacity(0.5), radius: 8, y: 2)
                    .offset(x: horizontal ? deflection * travel : 0,
                            y: horizontal ? 0 : -deflection * travel)
            }
            .frame(width: horizontal ? length : thickness,
                   height: horizontal ? thickness : length)
            .contentShape(Capsule())
            .gesture(drag)
            .accessibilityElement()
            .accessibilityLabel(label)
            .accessibilityValue(String(format: "%.0f%%", deflection * 100))

            Text(label)
                .font(.mabelCaption)
                .foregroundStyle(Theme.Palette.textTertiary)
                .tracking(1.2)
        }
    }

    private func gaugeFill(_ g: Double) -> some View {
        let inner = length - 8
        return ZStack(alignment: horizontal ? .leading : .bottom) {
            Color.clear
            Capsule().fill(tint.opacity(0.35))
                .frame(width: horizontal ? max(thickness - 8, inner * g) : thickness - 8,
                       height: horizontal ? thickness - 8 : max(thickness - 8, inner * g))
        }
        .frame(width: horizontal ? inner : thickness - 8,
               height: horizontal ? thickness - 8 : inner)
    }

    private var endGlyphs: some View {
        let glyph = { (name: String) in
            Image(systemName: name)
                .font(.system(size: 11, weight: .heavy))
                .foregroundStyle(.white.opacity(0.55))
        }
        return Group {
            if horizontal {
                HStack { glyph(negativeIcon); Spacer(minLength: 0); glyph(positiveIcon) }
                    .padding(.horizontal, 12)
            } else {
                VStack { glyph(positiveIcon); Spacer(minLength: 0); glyph(negativeIcon) }
                    .padding(.vertical, 12)
            }
        }
    }

    /// Drag translation (screen points, y DOWN+) → deflection in −1…+1 along
    /// `axis`, clamped at the ends. Horizontal: right +. Vertical: UP + (so a
    /// lift slider pushed up raises the column). The cross axis is ignored.
    /// Pure, so the sign conventions are unit-tested (`DriveControlsTests`).
    static func deflection(translation: CGSize, travel: CGFloat, axis: Axis) -> CGFloat {
        guard travel > 0 else { return 0 }
        let raw = axis == .horizontal ? translation.width : -translation.height
        return min(1, max(-1, raw / travel))
    }

    private var drag: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let d = Self.deflection(translation: value.translation, travel: travel, axis: axis)
                if abs(d) >= 1 {
                    if !atEnd { Haptics.soft(); atEnd = true }
                } else {
                    atEnd = false
                }
                deflection = d
                onChange(Double(d))
            }
            .onEnded { _ in
                Haptics.tap()
                atEnd = false
                withAnimation(.spring(response: 0.3, dampingFraction: 0.55)) { deflection = 0 }
                onChange(0)
            }
    }
}

/// The full Drive cluster: TRANSLATE (2-D) · LIFT (vertical) · ROTATE (1-D).
/// Pure view — every value goes out through the callbacks, so each app wires
/// it to its own controller (iOS: 30 Hz ramped tick; visionOS: direct send).
struct DrivePad: View {
    var stickSize: CGFloat = 132
    /// Live lift height in metres and the travel it is drawn against.
    var liftHeight: Double
    var liftMax: Double = 0.3175
    /// Space between the three controls (the slider sits in the middle).
    var spacing: CGFloat = 28
    var onTranslate: (CGVector) -> Void      // dx right+, dy DOWN+ (Joystick convention)
    var onRotate: (Double) -> Void           // −1…+1, right +
    var onLift: (Double) -> Void             // −1…+1, UP +

    var body: some View {
        HStack(alignment: .bottom, spacing: spacing) {
            Joystick(label: "TRANSLATE", tint: Theme.Palette.accent, size: stickSize,
                     onChange: onTranslate)
            AxisStick(label: "LIFT", axis: .vertical, tint: Color(hex: 0x8E6FD8),
                      length: stickSize + 8, thickness: 50,
                      negativeIcon: "arrow.down", positiveIcon: "arrow.up",
                      gauge: liftMax > 0 ? liftHeight / liftMax : nil,
                      readout: String(format: "%.2f m", liftHeight),
                      onChange: onLift)
            AxisStick(label: "ROTATE", axis: .horizontal, tint: Theme.Palette.clay,
                      length: stickSize + 8, thickness: 54,
                      negativeIcon: "arrow.turn.up.left", positiveIcon: "arrow.turn.up.right",
                      onChange: onRotate)
                // Bottom-align its label with the other two controls' labels.
                .frame(height: stickSize, alignment: .bottom)
        }
    }
}

#if DEBUG
/// `debugDrivePad` harness: the Simulator cannot synthesise a drag, so a boot
/// default scripts the pad instead — `"ly=-0.8:3;rot=0.8:3;lift=1:3"` holds
/// each axis at that deflection for that many seconds, releasing everything
/// for 1 s between steps. Axes are the pad's own callbacks: `lx`/`ly`
/// (TRANSLATE, Joystick convention: ly DOWN+), `rot` (right +), `lift` (up +).
/// `server/tools/exp_drive_pad.py` reads the robot's response off the wire.
struct DebugDriveScript {
    struct Step: Equatable { let axis: String; let value: Double; let seconds: Double }
    let steps: [Step]

    init(_ spec: String) {
        steps = spec.split(separator: ";").compactMap { part in
            let kv = part.split(separator: "=")
            guard kv.count == 2 else { return nil }
            let vs = kv[1].split(separator: ":")
            guard vs.count == 2, let v = Double(vs[0]), let s = Double(vs[1]),
                  ["lx", "ly", "rot", "lift"].contains(String(kv[0])) else { return nil }
            return Step(axis: String(kv[0]), value: v, seconds: s)
        }
    }

    /// Play the script on the main queue: `apply(step)` at each step's start,
    /// `release()` at its end. Starts after `delay` s (let the link come up).
    func play(delay: Double = 4, apply: @escaping (Step) -> Void, release: @escaping () -> Void) {
        var t = delay
        for step in steps {
            DispatchQueue.main.asyncAfter(deadline: .now() + t) {
                NSLog("[MABEL-DRIVEPAD] step \(step.axis)=\(step.value) for \(step.seconds)s")
                apply(step)
            }
            t += step.seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + t) { release() }
            t += 1
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + t) { NSLog("[MABEL-DRIVEPAD] done") }
    }
}
#endif
