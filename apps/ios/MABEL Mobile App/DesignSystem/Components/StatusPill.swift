//
//  StatusPill.swift
//  MABEL Mobile App
//
//  A small live-state indicator with a pulsing dot.
//

import SwiftUI

struct StatusPill: View {
    let title: String
    let color: Color
    var pulses: Bool = false

    @State private var pulse = false

    var body: some View {
        HStack(spacing: 7) {
            ZStack {
                if pulses {
                    Circle()
                        .fill(color.opacity(0.5))
                        .frame(width: 9, height: 9)
                        .scaleEffect(pulse ? 2.4 : 1)
                        .opacity(pulse ? 0 : 0.8)
                }
                Circle()
                    .fill(color)
                    .frame(width: 7, height: 7)
            }
            Text(title)
                .font(.mabelCaption)
                .foregroundStyle(Theme.Palette.textSecondary)
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 6)
        .background(color.opacity(0.12), in: Capsule())
        .overlay { Capsule().strokeBorder(color.opacity(0.28), lineWidth: 1) }
        .onAppear {
            guard pulses else { return }
            withAnimation(.easeOut(duration: 1.4).repeatForever(autoreverses: false)) {
                pulse = true
            }
        }
    }
}
