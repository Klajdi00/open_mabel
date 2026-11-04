//
//  ConsoleButton.swift
//  MabelTeleop
//
//  The immersive console's control chip: icon on the first line inside a
//  state circle, a short single-line label on the second. ON/OFF reads from
//  the FILLED vs hollow circle (Control-Center style), not from text colour
//  alone. Fixed width so labels never wrap.
//

import SwiftUI

struct ConsoleButton: View {
    let icon: String
    let label: String
    /// nil = momentary action; true/false = toggle state.
    var isOn: Bool? = nil
    /// Accent for the ON state / action emphasis.
    var tint: Color = Theme.Palette.accent
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 5) {
                ZStack {
                    Circle()
                        .fill(isOn == true ? AnyShapeStyle(tint) : AnyShapeStyle(Theme.Palette.surfaceHi))
                    Circle()
                        .strokeBorder(isOn == true ? tint : Theme.Palette.hairline, lineWidth: 1)
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(isOn == true ? .white : Theme.Palette.textPrimary)
                }
                .frame(width: 38, height: 38)

                Text(label)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .foregroundStyle(isOn == true ? tint : Theme.Palette.textSecondary)
            }
            .frame(width: 76)
            .padding(.vertical, 6)
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .help(label)
    }
}
