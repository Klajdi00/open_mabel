//
//  MiniWeekCalendar.swift
//  MabelTeleop
//
//  A compact at-a-glance week calendar for the Policies tab: time labels down the
//  left, the 7 days across the top, and each scheduled routine shown as a small
//  block at its day/time. Only hours that actually have routines are listed, so
//  it stays compact. Tapping opens the full, editable weekly timeline.
//

import SwiftUI

struct MiniWeekCalendar: View {
    let tasks: [ScheduledTask]
    var onTap: () -> Void

    private let dayLetters = ["S", "M", "T", "W", "T", "F", "S"]
    private let gutter: CGFloat = 42
    private let rowH: CGFloat = 30

    /// Distinct hours that have at least one routine, soonest first.
    private var hours: [Int] { Array(Set(tasks.map { $0.hour })).sorted() }

    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 0) {
                HStack(spacing: 0) {
                    Color.clear.frame(width: gutter)
                    ForEach(0..<7, id: \.self) { i in
                        Text(dayLetters[i]).font(.system(size: 9, weight: .bold))
                            .foregroundStyle(Theme.Palette.textTertiary).frame(maxWidth: .infinity)
                    }
                }
                .padding(.bottom, 5)

                if hours.isEmpty {
                    Text("No routines scheduled")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                        .frame(maxWidth: .infinity).padding(.vertical, 14)
                } else {
                    ForEach(hours, id: \.self) { row($0) }
                }
            }
            .padding(10)
            .background(Theme.Palette.ink, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 14).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func row(_ h: Int) -> some View {
        HStack(spacing: 0) {
            Text(timeLabel(h)).font(.system(size: 9, weight: .medium)).monospacedDigit()
                .foregroundStyle(Theme.Palette.textTertiary)
                .frame(width: gutter, alignment: .trailing).padding(.trailing, 5)
            ForEach(1...7, id: \.self) { wd in cell(day: wd, hour: h) }
        }
        .frame(height: rowH)
        .overlay(alignment: .top) { Rectangle().fill(Theme.Palette.hairlineSoft).frame(height: 0.5) }
    }

    private func cell(day: Int, hour: Int) -> some View {
        ZStack {
            if let t = tasks.first(where: { $0.weekdays.contains(day) && $0.hour == hour }) {
                let tint = t.enabled ? Theme.Palette.accent : Theme.Palette.textTertiary
                Image(systemName: t.icon).font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(tint.gradient, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .opacity(t.enabled ? 1 : 0.5)
                    .padding(2)
            }
        }
        .frame(maxWidth: .infinity, minHeight: rowH, maxHeight: rowH)
    }

    private func timeLabel(_ h: Int) -> String {
        let ampm = h < 12 ? "a" : "p"
        let h12 = h % 12 == 0 ? 12 : h % 12
        return "\(h12)\(ampm)"
    }
}
