//
//  Joystick.swift
//  MABEL Mobile App
//
//  A virtual analog stick. Reports a normalized vector in [-1, 1] on both
//  axes, springs back to center on release, and gives subtle haptics at
//  the rim. Used for base driving and Cartesian end-effector control.
//

import SwiftUI

struct Joystick: View {
    let label: String
    /// Constrain motion to a single axis when the control is 1-DOF.
    var lockHorizontal: Bool = false
    var lockVertical: Bool = false
    var tint: Color = Theme.Palette.accent
    var size: CGFloat = 150
    /// Continuous callback with the live vector (dx right+, dy down+).
    var onChange: (CGVector) -> Void

    private var knob: CGFloat { size * 0.39 }
    @State private var offset: CGSize = .zero
    @State private var atRim = false

    private var radius: CGFloat { (size - knob) / 2 }

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                // Base well
                Circle()
                    .fill(Theme.Palette.surfaceHi)
                    .overlay { Circle().strokeBorder(Theme.Palette.hairline, lineWidth: 1) }

                // Axis guides
                if !lockVertical {
                    Capsule().fill(Theme.Palette.hairline)
                        .frame(width: 2, height: size - 36)
                }
                if !lockHorizontal {
                    Capsule().fill(Theme.Palette.hairline)
                        .frame(width: size - 36, height: 2)
                }

                // Reach ring
                Circle()
                    .strokeBorder(tint.opacity(0.25), lineWidth: 1)
                    .frame(width: size - knob + 8, height: size - knob + 8)

                // Knob
                Circle()
                    .fill(tint.gradient)
                    .frame(width: knob, height: knob)
                    .overlay {
                        Image(systemName: "circle.grid.cross.fill")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundStyle(.black.opacity(0.35))
                    }
                    .shadow(color: tint.opacity(0.5), radius: 10, y: 3)
                    .offset(offset)
            }
            .frame(width: size, height: size)
            .contentShape(Circle())
            .gesture(drag)

            Text(label)
                .font(.mabelCaption)
                .foregroundStyle(Theme.Palette.textTertiary)
                .tracking(1.2)
        }
    }

    private var drag: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                var dx = lockHorizontal ? 0 : value.translation.width
                var dy = lockVertical ? 0 : value.translation.height
                let dist = sqrt(dx * dx + dy * dy)
                if dist > radius {                  // clamp to the well
                    let scale = radius / dist
                    dx *= scale; dy *= scale
                    if !atRim { Haptics.soft(); atRim = true }
                } else {
                    atRim = false
                }
                offset = CGSize(width: dx, height: dy)
                onChange(CGVector(dx: dx / radius, dy: dy / radius))
            }
            .onEnded { _ in
                Haptics.tap()
                atRim = false
                withAnimation(.spring(response: 0.3, dampingFraction: 0.55)) {
                    offset = .zero
                }
                onChange(.zero)
            }
    }
}
