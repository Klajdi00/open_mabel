//
//  AngleDial.swift
//  MABEL Mobile App
//
//  A circular (rotary) setpoint control for a single revolute joint. Unlike a
//  joystick — which springs back to centre and only reports a *rate* of change —
//  the dial's knob sits at the joint's ABSOLUTE angle and the whole travel is
//  drawn as an arc, so the operator sees the continuous range and the hard
//  limits at a glance. Dragging writes the angle DIRECTLY (one setpoint per
//  position), so there is no 30 Hz rate integration and none of the jitter that
//  streamed rate commands produce. Used for the neck yaw.
//
//  Geometry (dial-local, y-down as SwiftUI), hub at the frame centre:
//    a = screen angle, clockwise from straight-up (12 o'clock), radians
//    knob offset = (r·sin a, −r·cos a)          → a=0 top, +90° right, −90° left
//    value v maps linearly across `range` onto a ∈ [−90°, +90°] (a top
//    semicircle); `invert` flips which end of the range sits on the right.
//

import SwiftUI

struct AngleDial: View {
    let label: String
    @Binding var value: Double            // radians — the live joint setpoint
    let range: ClosedRange<Double>        // radians — the joint's hard limits
    /// End labels drawn under the left / right tips of the arc.
    var leftTag: String = ""
    var rightTag: String = ""
    var tint: Color = Theme.Palette.clay
    var size: CGFloat = 128
    /// When true, the LOW end of `range` sits on the RIGHT of the arc. Neck yaw
    /// is +left / −right, so inverting puts "look right" physically on the right.
    var invert: Bool = true

    private var span: Double { max(1e-6, range.upperBound - range.lowerBound) }
    private var knob: CGFloat { size * 0.19 }
    private var r: CGFloat { size / 2 - knob / 2 - 2 }

    /// value → screen angle a (radians, clockwise from top).
    private func angle(for v: Double) -> Double {
        let t = (v - range.lowerBound) / span
        let tt = invert ? (1 - t) : t
        return (tt * .pi) - (.pi / 2)
    }

    /// screen angle a → clamped value.
    private func value(for a: Double) -> Double {
        let ac = min(.pi / 2, max(-.pi / 2, a))
        let tt = (ac + .pi / 2) / .pi
        let t = invert ? (1 - tt) : tt
        return range.lowerBound + t * span
    }

    private func offset(_ a: Double) -> CGSize {
        CGSize(width: r * sin(a), height: -r * cos(a))
    }

    var body: some View {
        let a = angle(for: value)
        VStack(spacing: 6) {
            ZStack {
                // Range track (top semicircle) + active sweep top → knob.
                DialArc(fromDeg: -90, toDeg: 90, radius: r)
                    .stroke(Theme.Palette.hairline, style: .init(lineWidth: 3, lineCap: .round))
                DialArc(fromDeg: min(0, a * 180 / .pi), toDeg: max(0, a * 180 / .pi), radius: r)
                    .stroke(tint.opacity(0.9), style: .init(lineWidth: 3, lineCap: .round))

                // Limit + centre ticks so the travel reads even without the knob.
                ForEach([-90.0, 0.0, 90.0], id: \.self) { deg in
                    Capsule().fill(Theme.Palette.hairline)
                        .frame(width: 1.5, height: deg == 0 ? 9 : 6)
                        .offset(x: 0, y: -(r + 4))
                        .rotationEffect(.degrees(deg))
                }

                // Live readout at the hub.
                VStack(spacing: 1) {
                    Text(String(format: "%.0f°", value * 180 / .pi))
                        .font(.system(size: 16, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                    Text(label)
                        .font(.system(size: 7, weight: .bold)).tracking(1.4)
                        .foregroundStyle(.white.opacity(0.55))
                }
                .offset(y: r * 0.28)     // nudge below the hub so the knob has room up top

                // Knob.
                Circle()
                    .fill(tint.gradient)
                    .frame(width: knob, height: knob)
                    .overlay { Circle().strokeBorder(.white.opacity(0.85), lineWidth: 1.5) }
                    .shadow(color: tint.opacity(0.5), radius: 6, y: 2)
                    .offset(offset(a))
            }
            .frame(width: size, height: size)
            .contentShape(Rectangle())
            .gesture(drag)

            if !leftTag.isEmpty || !rightTag.isEmpty {
                HStack {
                    Text(leftTag); Spacer(); Text(rightTag)
                }
                .font(.system(size: 8, weight: .bold)).tracking(1)
                .foregroundStyle(.white.opacity(0.5))
                .frame(width: size - 10)
            }
        }
    }

    private var drag: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { g in
                let dx = g.location.x - size / 2       // hub is the frame centre
                let dy = g.location.y - size / 2
                let v = value(for: atan2(dx, -dy))      // atan2 clockwise from top
                guard abs(v - value) > 1e-4 else { return }
                let atEnd  = v <= range.lowerBound + 1e-4 || v >= range.upperBound - 1e-4
                let wasEnd = value <= range.lowerBound + 1e-4 || value >= range.upperBound - 1e-4
                if atEnd && !wasEnd { Haptics.soft() }
                value = v                                // binding setter writes the setpoint
            }
            .onEnded { _ in Haptics.tap() }
    }
}

/// Traces a top-oriented arc between two screen angles (degrees, clockwise from
/// 12 o'clock) by sampling the parametric curve — sidesteps SwiftUI's
/// flipped-`clockwise` arc convention entirely. Centre = the rect centre.
private struct DialArc: Shape {
    let fromDeg: Double
    let toDeg: Double
    let radius: CGFloat

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let steps = 40
        for i in 0...steps {
            let a = (fromDeg + (toDeg - fromDeg) * Double(i) / Double(steps)) * .pi / 180
            let pt = CGPoint(x: c.x + radius * sin(a), y: c.y - radius * cos(a))
            if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
        }
        return p
    }
}
