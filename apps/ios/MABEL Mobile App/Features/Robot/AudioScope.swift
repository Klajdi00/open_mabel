//
//  AudioScope.swift
//  MABEL Mobile App
//
//  A compact two-lane audio scope for the VOICE console section — the operator's
//  own mic (UP → robot) over the robot's room (DOWN → me). It exists to make the
//  two SILENT failure modes visible without reading a log:
//
//    · DOWN flat while UP moves → the robot bar's physical mute (every sample is
//      zero, the peak=0 the plan warns about).
//    · UP flat while Talk is latched on → a dead capture or a denied mic permission.
//
//  Drawn with a single `Canvas`, NOT a stack of per-sample bar views. The first
//  cut used one Capsule per sample in an HStack; at 48 samples the HStack's
//  minimum width overflowed the (then 60 pt) lane and BLED OUT to the right, under
//  the Talk button, because nothing clipped it. A Canvas maps N samples onto
//  the EXACT available width — it cannot overflow — and is far cheaper than ~96
//  views re-laid-out at 14 Hz.
//
//  Observes the AudioStreamLink DIRECTLY (not through RobotSpaceView) so its
//  14 Hz refresh invalidates only these points, never the whole console.
//

import SwiftUI

struct AudioScope: View {
    @ObservedObject var link: AudioStreamLink

    private let up = Theme.Palette.live       // mic → robot
    private let down = Theme.Palette.accent   // robot → me

    var body: some View {
        VStack(spacing: 3) {
            lane(link.meter.up, now: link.meter.upNow, tint: up, glyph: "arrow.up")
            lane(link.meter.down, now: link.meter.downNow, tint: down, glyph: "arrow.down")
        }
        .padding(5)
        .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Theme.Palette.surface))
        .overlay { RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))   // belt + braces: nothing leaves the box
        .accessibilityLabel("Audio levels")
        .accessibilityValue("sending \(Int(link.meter.upNow * 100)) percent, receiving \(Int(link.meter.downNow * 100)) percent")
    }

    /// One lane: a direction glyph + a Canvas waveform that fits the width exactly,
    /// newest sample at the right, + a live 0–99 readout. Silence draws a faint
    /// centre baseline so a dead lane still reads as "connected but quiet", distinct
    /// from "no scope at all".
    ///
    /// The extra width the console row gave up (2026-07-28) is spent on INFORMATION,
    /// not on fatter bars: 128 samples ≈ 9 s of history, one-second gridlines so you
    /// can actually time what you heard, and a peak-hold rail at the loudest sample
    /// in the window — which is what tells you a lane that is quiet *now* was alive a
    /// moment ago, the difference between "nobody is talking" and "this path is dead".
    private func lane(_ samples: [Float], now: Float, tint: Color, glyph: String) -> some View {
        let live = now > 0.03
        let peak = CGFloat(min(1, max(0, samples.max() ?? 0)))
        return HStack(spacing: 4) {
            Image(systemName: glyph)
                .font(.system(size: 7, weight: .heavy))
                .foregroundStyle(live ? tint : Theme.Palette.textTertiary)
                .frame(width: 8)
            Canvas { ctx, size in
                let n = samples.count
                guard n > 1, size.width > 1 else { return }
                let mid = size.height / 2
                let half = size.height * 0.5 - 0.5
                let step = size.width / CGFloat(n - 1)

                // One-second gridlines (14 samples at the link's 14 Hz publish rate),
                // anchored at the RIGHT so "now" is always a gridline and the ticks
                // don't crawl as the window scrolls.
                var grid = Path()
                var i = n - 1
                while i > 0 {
                    let x = CGFloat(i) * step
                    grid.move(to: CGPoint(x: x, y: 1))
                    grid.addLine(to: CGPoint(x: x, y: size.height - 1))
                    i -= AudioScope.samplesPerSecond
                }
                ctx.stroke(grid, with: .color(Theme.Palette.hairline.opacity(0.55)), lineWidth: 0.5)

                // Faint baseline through the centre.
                var base = Path()
                base.move(to: CGPoint(x: 0, y: mid))
                base.addLine(to: CGPoint(x: size.width, y: mid))
                ctx.stroke(base, with: .color(Theme.Palette.hairline), lineWidth: 0.5)

                // Peak-hold rails, mirrored — the window's loudest sample.
                if peak > 0.02 {
                    let h = peak * half
                    var hold = Path()
                    hold.move(to: CGPoint(x: 0, y: mid - h)); hold.addLine(to: CGPoint(x: size.width, y: mid - h))
                    hold.move(to: CGPoint(x: 0, y: mid + h)); hold.addLine(to: CGPoint(x: size.width, y: mid + h))
                    ctx.stroke(hold, with: .color(tint.opacity(0.28)),
                               style: StrokeStyle(lineWidth: 0.5, dash: [2, 3]))
                }

                // Mirror bars around the centre — a classic VU look — one column
                // per sample, spread to fill the width exactly.
                for i in 0..<n {
                    let v = CGFloat(min(1, max(0, samples[i])))
                    guard v > 0.01 else { continue }
                    let x = CGFloat(i) * step
                    let h = v * half
                    var bar = Path()
                    bar.move(to: CGPoint(x: x, y: mid - h))
                    bar.addLine(to: CGPoint(x: x, y: mid + h))
                    // Newest (right) brightest, so motion reads left→right.
                    let a = 0.45 + 0.55 * (CGFloat(i) / CGFloat(n - 1))
                    ctx.stroke(bar, with: .color(tint.opacity(a)),
                               style: StrokeStyle(lineWidth: max(0.8, step * 0.7), lineCap: .round))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // Live level. Monospaced digits so the number doesn't shimmy at 14 Hz.
            Text("\(min(99, Int(now * 100)))")
                .font(.system(size: 8, weight: .bold).monospacedDigit())
                .foregroundStyle(live ? tint : Theme.Palette.textTertiary)
                .frame(width: 14, alignment: .trailing)
        }
    }

    /// Samples per second in the meter — `AudioStreamLink` publishes the scope at
    /// 14 Hz (`startMeter`). Only used to place the gridlines.
    private static let samplesPerSecond = 14
}
