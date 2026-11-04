//
//  JointSlider.swift
//  MabelTeleop
//
//  One labelled joint slider row with a monospaced read-out.
//
//  visionOS uses the PLATFORM slider at its natural size, not `.controlSize(.mini)`
//  (2026-07-16). The mini slider is an iOS-density control: a hairline track and a
//  thumb far under the ~44 pt gaze target visionOS wants, which read as flat and
//  dated next to the glass panels around it — and was genuinely fiddly to grab by
//  eye. The stock control at full size IS the system's liquid glass (glass track,
//  lifting thumb, gaze highlight for free), so it matches the rest of the app by
//  construction instead of by imitation.
//

import SwiftUI

struct JointSlider: View {
    let spec: JointSpec
    @Binding var value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(spec.label)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .lineLimit(1)
                Spacer(minLength: 6)
                Text(readout)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.Palette.accent)
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .animation(Theme.Motion.snappy, value: value)
            }
            Slider(value: $value, in: spec.range)
                .tint(Theme.Palette.accent)
        }
        .padding(.vertical, 5)
    }

    private var readout: String {
        spec.unit.isEmpty
            ? String(format: "%.2f", value)
            : String(format: "%.2f %@", value, spec.unit)
    }
}
