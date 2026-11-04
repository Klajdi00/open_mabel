//
//  MabelWidgetViews.swift
//  MabelWidgetExtension
//
//  Per-family layouts. Home Screen (small / medium / large) lead with the
//  rendered 3D pose and surround it with live telemetry; Lock Screen accessories
//  (rectangular / circular / inline) distil it to the essentials. All of it is
//  driven by the shared snapshot, in the app's warm-rust design language.
//

import WidgetKit
import SwiftUI

struct MabelWidgetEntryView: View {
    @Environment(\.widgetFamily) private var family
    let entry: MabelEntry

    var body: some View {
        switch family {
        case .systemSmall:        SmallWidget(entry: entry)
        case .systemMedium:       MediumWidget(entry: entry)
        case .systemLarge:        LargeWidget(entry: entry)
        case .accessoryRectangular: RectAccessory(entry: entry)
        case .accessoryCircular:  CircAccessory(entry: entry)
        case .accessoryInline:    InlineAccessory(entry: entry)
        default:                  MediumWidget(entry: entry)
        }
    }
}

// MARK: - Pose presentation

private extension MabelEntry {
    /// Desaturate the pose when it's stale or the robot is offline.
    var poseDim: Bool { snapshot.isStale(asOf: date) || !snapshot.isLive }
    /// Bottom caption — only when there's an actual render to qualify.
    var poseBadge: String? {
        guard poseImage != nil else { return nil }
        if !snapshot.isLive { return "OFFLINE" }
        return snapshot.isStale(asOf: date) ? "LAST KNOWN" : nil
    }
}

// MARK: - Small

private struct SmallWidget: View {
    let entry: MabelEntry
    var s: MabelWidgetSnapshot { entry.snapshot }

    var body: some View {
        ZStack {
            WidgetPose(image: entry.poseImage, dim: entry.poseDim, badge: entry.poseBadge)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

            VStack {
                HStack(alignment: .top) {
                    HStack(spacing: 4) {
                        WidgetStatusDot(color: s.connectionColor.color, size: 6)
                        Text(s.robotName).font(.system(size: 12, weight: .bold))
                            .foregroundStyle(.white).lineLimit(1)
                    }
                    Spacer()
                    WidgetTargetTag(snapshot: s)
                }
                Spacer()
                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 1) {
                        HStack(spacing: 3) {
                            Image(systemName: s.charging ? "battery.100.bolt" : "bolt.fill")
                                .font(.system(size: 10, weight: .bold)).foregroundStyle(s.batteryColor.color)
                            Text("\(s.batteryPercent)%").font(.system(size: 17, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                        }
                        Text(s.isLive ? s.ageText(asOf: entry.date) : s.connectionLabel)
                            .font(.system(size: 9, weight: .medium)).foregroundStyle(ColorHex.textTertiary.color)
                    }
                    Spacer()
                    Image(systemName: s.transportIcon)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(s.isLive ? s.connectionColor.color : ColorHex.textTertiary.color)
                }
            }
            .padding(11)
            .background(alignment: .bottom) {
                LinearGradient(colors: [.clear, .black.opacity(0.55)], startPoint: .center, endPoint: .bottom)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
        }
    }
}

// MARK: - Medium

private struct MediumWidget: View {
    let entry: MabelEntry
    var s: MabelWidgetSnapshot { entry.snapshot }

    var body: some View {
        HStack(spacing: 11) {
            WidgetPose(image: entry.poseImage, dim: entry.poseDim, badge: entry.poseBadge)
                .frame(width: 116)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(alignment: .topLeading) { WidgetTargetTag(snapshot: s).padding(6) }

            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Text(s.robotName).font(.system(size: 15, weight: .bold)).foregroundStyle(.white).lineLimit(1)
                    Spacer(minLength: 4)
                    WidgetConnectionPill(snapshot: s, now: entry.date)
                }

                LazyVGrid(columns: [GridItem(.flexible(), spacing: 6), GridItem(.flexible(), spacing: 6)], spacing: 6) {
                    WidgetMetric(icon: s.charging ? "battery.100.bolt" : "bolt.fill",
                                 value: "\(s.batteryPercent)", unit: "%", label: "Battery", tint: s.batteryColor.color)
                    WidgetMetric(icon: "bolt.horizontal", value: numel(s.latencyMs, 0), unit: "ms", label: "Latency",
                                 tint: ColorHex.warn.color)
                    WidgetMetric(icon: s.transportIcon, value: s.transport, unit: "", label: "Link",
                                 tint: s.isLive ? ColorHex.live.color : ColorHex.textTertiary.color)
                    WidgetMetric(icon: "speedometer", value: numel(s.linearSpeed, 2), unit: "m/s", label: "Speed",
                                 tint: ColorHex.accent.color)
                }

                ControlRow(snapshot: s)
            }
        }
    }
}

// MARK: - Large

private struct LargeWidget: View {
    let entry: MabelEntry
    var s: MabelWidgetSnapshot { entry.snapshot }

    var body: some View {
        VStack(spacing: 10) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(s.robotName).font(.system(size: 19, weight: .bold)).foregroundStyle(.white)
                    Text(s.controlSummary).font(.system(size: 11, weight: .medium))
                        .foregroundStyle(ColorHex.textSecondary.color).lineLimit(1)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 5) {
                    WidgetConnectionPill(snapshot: s, now: entry.date)
                    WidgetTargetTag(snapshot: s)
                }
            }

            HStack(spacing: 11) {
                WidgetPose(image: entry.poseImage, dim: entry.poseDim, badge: entry.poseBadge)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(alignment: .bottomLeading) {
                        if s.estop {
                            Label("E-STOP", systemImage: "hand.raised.fill")
                                .font(.system(size: 9, weight: .heavy))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 7).padding(.vertical, 4)
                                .background(ColorHex.danger.color, in: Capsule())
                                .padding(7)
                        }
                    }

                VStack(spacing: 6) {
                    WidgetMetric(icon: s.charging ? "battery.100.bolt" : "bolt.fill",
                                 value: "\(s.batteryPercent)", unit: "%", label: "Battery", tint: s.batteryColor.color)
                    WidgetMetric(icon: "battery.100", value: numel(s.batteryVoltage, 1), unit: "V", label: "Voltage",
                                 tint: ColorHex.textSecondary.color)
                    WidgetMetric(icon: "bolt.horizontal", value: numel(s.latencyMs, 0), unit: "ms", label: "Latency",
                                 tint: ColorHex.warn.color)
                    WidgetMetric(icon: "arrow.up.and.down", value: "\(Int((s.liftFraction * 100).rounded()))", unit: "%",
                                 label: "Lift", tint: ColorHex.accent.color)
                }
                .frame(width: 118)
            }
            .frame(maxHeight: .infinity)

            // Driver + transport strip
            HStack(spacing: 6) {
                pill(icon: s.isDriver ? "steeringwheel" : "eye", text: s.driver,
                     tint: s.isDriver ? ColorHex.live.color : ColorHex.textSecondary.color)
                pill(icon: s.transportIcon, text: s.transport,
                     tint: s.isLive ? ColorHex.live.color : ColorHex.textTertiary.color)
                Spacer()
                pill(icon: "thermometer.medium", text: "\(numel(s.cpuTemp, 0))°", tint: ColorHex.textSecondary.color)
                pill(icon: "wifi", text: "\(s.wifiRSSI)", tint: ColorHex.textSecondary.color)
            }

            // Diagnostics
            if !s.diagnostics.isEmpty {
                VStack(spacing: 4) {
                    ForEach(s.diagnostics.prefix(3)) { d in
                        HStack(spacing: 6) {
                            WidgetStatusDot(color: levelColor(d.level), size: 5)
                            Text(d.subsystem).font(.system(size: 10, weight: .semibold)).foregroundStyle(.white)
                            Spacer()
                            Text(d.message).font(.system(size: 9, weight: .medium))
                                .foregroundStyle(ColorHex.textTertiary.color).lineLimit(1)
                        }
                    }
                }
                .padding(8)
                .background(WidgetTheme.surface, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
        }
    }

    private func pill(icon: String, text: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon).font(.system(size: 9, weight: .semibold)).foregroundStyle(tint)
            Text(text).font(.system(size: 9.5, weight: .semibold)).foregroundStyle(.white).lineLimit(1)
        }
        .padding(.horizontal, 7).padding(.vertical, 4)
        .background(WidgetTheme.surface, in: Capsule())
    }
}

// MARK: - Shared bits

private struct ControlRow: View {
    let snapshot: MabelWidgetSnapshot
    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: snapshot.estop ? "hand.raised.fill" : "slider.horizontal.3")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(snapshot.estop ? ColorHex.danger.color : ColorHex.accent.color)
            Text(snapshot.estop ? "EMERGENCY STOP" : snapshot.controlSummary)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(snapshot.estop ? ColorHex.danger.color : .white)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 8).padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((snapshot.estop ? ColorHex.danger.color.opacity(0.16) : WidgetTheme.surface),
                    in: RoundedRectangle(cornerRadius: 9, style: .continuous))
    }
}

// MARK: - Lock Screen accessories

private struct RectAccessory: View {
    let entry: MabelEntry
    var s: MabelWidgetSnapshot { entry.snapshot }
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Image(systemName: "shippingbox.fill").font(.system(size: 11, weight: .bold))
                Text(s.robotName).font(.system(size: 13, weight: .bold)).lineLimit(1)
                Spacer()
                Text(s.target).font(.system(size: 10, weight: .heavy))
            }
            HStack(spacing: 4) {
                Image(systemName: "bolt.fill").font(.system(size: 9))
                Text("\(s.batteryPercent)%")
                Text("·")
                Image(systemName: s.transportIcon).font(.system(size: 9))
                Text(s.connectionLabel)
            }
            .font(.system(size: 11, weight: .medium))
            Text(s.estop ? "E-STOP" : s.controlSummary)
                .font(.system(size: 10, weight: .regular)).lineLimit(1)
                .widgetAccentable()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}

private struct CircAccessory: View {
    let entry: MabelEntry
    var s: MabelWidgetSnapshot { entry.snapshot }
    var body: some View {
        Gauge(value: min(1, max(0, s.batteryLevel))) {
            Image(systemName: "bolt.fill")
        } currentValueLabel: {
            Text("\(s.batteryPercent)")
        }
        .gaugeStyle(.accessoryCircularCapacity)
    }
}

private struct InlineAccessory: View {
    let entry: MabelEntry
    var s: MabelWidgetSnapshot { entry.snapshot }
    var body: some View {
        Label("\(s.robotName) · \(s.connectionLabel) · \(s.batteryPercent)%",
              systemImage: s.isLive ? "dot.radiowaves.left.and.right" : "wifi.slash")
    }
}

// MARK: - Helpers

private func numel(_ v: Double, _ decimals: Int) -> String {
    String(format: "%.\(decimals)f", v)
}

private func levelColor(_ l: MabelWidgetSnapshot.Diag.Level) -> Color {
    switch l {
    case .ok:    return ColorHex.live.color
    case .warn:  return ColorHex.warn.color
    case .fault: return ColorHex.danger.color
    }
}

// MARK: - Previews

#Preview(as: .systemMedium) { MabelStatusWidget() } timeline: {
    MabelEntry(date: .now, snapshot: .placeholder, poseImage: nil)
}
#Preview(as: .systemLarge) { MabelStatusWidget() } timeline: {
    MabelEntry(date: .now, snapshot: .placeholder, poseImage: nil)
}
#Preview(as: .systemSmall) { MabelStatusWidget() } timeline: {
    MabelEntry(date: .now, snapshot: .placeholder, poseImage: nil)
}
