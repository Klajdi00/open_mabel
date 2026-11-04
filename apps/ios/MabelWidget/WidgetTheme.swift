//
//  WidgetTheme.swift
//  MabelWidgetExtension
//
//  The widget's visual language, kept in lock-step with the app: warm rust
//  accent, warm-ink canvas, SF system type, Apple semantic status colours.
//  Colours come from the shared `ColorHex` table so the app and widget never
//  drift. Small reusable pieces (status dot, battery ring, metric cell) live
//  here so the per-family layouts stay readable.
//

import SwiftUI
import WidgetKit

enum WidgetTheme {
    /// Warm-ink canvas behind every widget (matches the app's dark canvas).
    static var canvas: some ShapeStyle {
        LinearGradient(colors: [Color(hex: 0x17130F), Color(hex: 0x0B0907)],
                       startPoint: .top, endPoint: .bottom)
    }
    /// The viewport gradient that sits under the rendered robot pose.
    static var viewport: LinearGradient {
        LinearGradient(colors: [Color(hex: 0x221C16), Color(hex: 0x0C0A08)],
                       startPoint: .top, endPoint: .bottom)
    }

    static let hairline = Color.white.opacity(0.10)
    static let surface  = Color.white.opacity(0.06)
}

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255)
    }
}

// MARK: - Staleness

extension MabelWidgetSnapshot {
    /// "live", "2m ago", "1h ago" — how fresh this snapshot is.
    func ageText(asOf now: Date) -> String {
        let s = max(0, now.timeIntervalSince(date))
        if s < 75 { return "live" }
        if s < 3600 { return "\(Int(s / 60))m ago" }
        if s < 86_400 { return "\(Int(s / 3600))h ago" }
        return "\(Int(s / 86_400))d ago"
    }
    /// Snapshot is old enough that the pose / numbers should read as stale.
    func isStale(asOf now: Date) -> Bool { now.timeIntervalSince(date) > 180 }
}

// MARK: - Status dot

struct WidgetStatusDot: View {
    let color: Color
    var size: CGFloat = 7
    var body: some View {
        Circle().fill(color)
            .frame(width: size, height: size)
            .shadow(color: color.opacity(0.8), radius: size * 0.6)
    }
}

// MARK: - Battery ring (Lock-Screen circular + accents)

struct WidgetBatteryRing: View {
    let level: Double          // 0…1
    let color: Color
    var charging: Bool = false
    var lineWidth: CGFloat = 6

    var body: some View {
        ZStack {
            Circle().stroke(Color.white.opacity(0.14), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: max(0.02, min(1, level)))
                .stroke(color, style: .init(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
            VStack(spacing: -1) {
                if charging {
                    Image(systemName: "bolt.fill").font(.system(size: 9, weight: .bold))
                        .foregroundStyle(color)
                }
                Text("\(Int((level * 100).rounded()))")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .minimumScaleFactor(0.6)
            }
        }
    }
}

// MARK: - Compact metric cell

struct WidgetMetric: View {
    let icon: String
    let value: String
    let unit: String
    let label: String
    var tint: Color = ColorHex.accent.color

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Image(systemName: icon).font(.system(size: 9, weight: .semibold)).foregroundStyle(tint)
                Text(label.uppercased())
                    .font(.system(size: 8, weight: .bold)).tracking(0.4)
                    .foregroundStyle(ColorHex.textTertiary.color)
                    .lineLimit(1)
            }
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white).lineLimit(1).minimumScaleFactor(0.7)
                if !unit.isEmpty {
                    Text(unit).font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(ColorHex.textTertiary.color)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 6).padding(.horizontal, 8)
        .background(WidgetTheme.surface, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(WidgetTheme.hairline, lineWidth: 0.5) }
    }
}

// MARK: - Connection pill

struct WidgetConnectionPill: View {
    let snapshot: MabelWidgetSnapshot
    let now: Date
    var body: some View {
        HStack(spacing: 5) {
            WidgetStatusDot(color: snapshot.connectionColor.color, size: 6)
            Text(snapshot.connectionLabel.uppercased())
                .font(.system(size: 9, weight: .bold)).tracking(0.4)
                .foregroundStyle(snapshot.connectionColor.color)
            if snapshot.isLive {
                Text("· \(snapshot.ageText(asOf: now))")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(ColorHex.textTertiary.color)
            }
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(snapshot.connectionColor.color.opacity(0.14), in: Capsule())
        .overlay { Capsule().strokeBorder(snapshot.connectionColor.color.opacity(0.28), lineWidth: 0.5) }
    }
}

// MARK: - SIM / REAL tag

struct WidgetTargetTag: View {
    let snapshot: MabelWidgetSnapshot
    var body: some View {
        Text(snapshot.target)
            .font(.system(size: 9, weight: .heavy)).tracking(0.6)
            .foregroundStyle(snapshot.targetColor.color)
            .padding(.horizontal, 6).padding(.vertical, 3)
            .background(snapshot.targetColor.color.opacity(0.16), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
}

// MARK: - Rendered pose, or a tasteful fallback

struct WidgetPose: View {
    let image: UIImage?
    /// Desaturate + dim (snapshot is stale, or the robot is offline).
    var dim: Bool = false
    /// Small caption pinned to the bottom ("LAST KNOWN" / "OFFLINE"), if any.
    var badge: String? = nil

    var body: some View {
        ZStack {
            WidgetTheme.viewport
            if let image {
                Image(uiImage: image)
                    .resizable().scaledToFit()
                    .saturation(dim ? 0.35 : 1)
                    .opacity(dim ? 0.6 : 1)
            } else {
                WidgetPoseFallback()
            }
            if let badge {
                VStack { Spacer()
                    Text(badge)
                        .font(.system(size: 7.5, weight: .bold)).tracking(1)
                        .foregroundStyle(ColorHex.textTertiary.color)
                        .padding(.vertical, 2).padding(.horizontal, 6)
                        .background(.black.opacity(0.35), in: Capsule())
                        .padding(.bottom, 5)
                }
            }
        }
    }
}

/// Branded placeholder shown before the app has ever rendered a pose. Reads as a
/// "robot powering on" mark rather than a bare SF Symbol.
private struct WidgetPoseFallback: View {
    var body: some View {
        ZStack {
            Circle()
                .stroke(ColorHex.accent.color.opacity(0.35), lineWidth: 1.5)
                .frame(width: 78, height: 78)
            Circle()
                .stroke(ColorHex.accent.color.opacity(0.12), lineWidth: 1)
                .frame(width: 104, height: 104)
            VStack(spacing: 5) {
                Image(systemName: "figure.stand")
                    .font(.system(size: 34, weight: .regular))
                    .foregroundStyle(ColorHex.accent.color)
                Text("MABEL")
                    .font(.system(size: 11, weight: .heavy)).tracking(2)
                    .foregroundStyle(ColorHex.textSecondary.color)
            }
        }
    }
}
