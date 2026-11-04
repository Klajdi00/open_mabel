//
//  JointSlider.swift
//  MABEL Mobile App
//
//  A labelled slider bound to one robot DOF, showing its live value and
//  units. Used in arm and whole-body control.
//

import SwiftUI

struct JointSlider: View {
    let spec: JointSpec
    @Binding var value: Double

    var body: some View {
        HStack(spacing: 10) {
            Text(spec.label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.Palette.textSecondary)
                .frame(width: 104, alignment: .leading)
                .lineLimit(1)
            Slider(value: $value, in: spec.range) { editing in
                if editing { Haptics.select() }
            }
            .tint(Theme.Palette.accent)
            .controlSize(.mini)
            Text(String(format: spec.unit.isEmpty ? "%.2f" : "%.2f", value))
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.Palette.accent)
                .frame(width: 38, alignment: .trailing)
                .contentTransition(.numericText())
        }
        .padding(.vertical, 3)
    }
}
