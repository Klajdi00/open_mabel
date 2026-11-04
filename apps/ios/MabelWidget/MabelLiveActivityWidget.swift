//
//  MabelLiveActivityWidget.swift
//  MabelWidgetExtension
//
//  The Dynamic Island + Lock-Screen Live Activity for MABEL. Shown only while
//  the app is connected to a host (started/ended by MabelLiveActivity).
//
//  Design language borrows from Uber's ride activity: the COMPACT island leads
//  with a single mode emoji and trails with the ETA / progress; LONG-PRESS
//  EXPANDS into a rich card with the mode spelled out, a prominent progress bar
//  and the time remaining (for a policy run or navigation), the live 3D pose, and
//  a status strip. Tapping any presentation deep-links into the app.
//

import ActivityKit
import WidgetKit
import SwiftUI

private let deepLink = URL(string: "mabelv2://island")!

struct MabelLiveActivityWidget: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: MabelLiveActivityAttributes.self) { context in
            LiveBanner(state: context.state, name: context.attributes.robotName)
                .widgetURL(deepLink)
                .activityBackgroundTint(Color(hex: 0x0B0907))
                .activitySystemActionForegroundColor(ColorHex.accent.color)
        } dynamicIsland: { context in
            let s = context.state
            return DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    ExpandedPose(state: s)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    ExpandedStatus(state: s)
                }
                DynamicIslandExpandedRegion(.center) {
                    ExpandedIdentity(state: s, name: context.attributes.robotName)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    ExpandedBottom(state: s).widgetURL(deepLink)
                }
            } compactLeading: {
                // Connection identity on the LEFT: live dot + SIM/REAL.
                HStack(spacing: 3) {
                    WidgetStatusDot(color: s.connectionColor.color, size: 7)
                    Text(s.target).font(.system(size: 12, weight: .heavy))
                        .foregroundStyle(s.targetColor.color)
                }
            } compactTrailing: {
                // The mode on the RIGHT: emoji (+ ETA / progress when a task runs).
                CompactMode(state: s)
            } minimal: {
                if s.hasProgress {
                    ProgressRing(progress: s.progress, size: 18, line: 3)
                } else {
                    Text(s.activityEmoji).font(.system(size: 14))
                }
            }
            .widgetURL(deepLink)
            .keylineTint(s.connectionColor.color)
        }
    }
}

// MARK: - Compact

private struct CompactMode: View {
    let state: MabelLiveActivityAttributes.ContentState
    var body: some View {
        HStack(spacing: 3) {
            Text(state.activityEmoji).font(.system(size: 15))
            if state.hasETA {
                Text(state.etaText)
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .foregroundStyle(ColorHex.accent.color)
            } else if state.hasProgress {
                Text("\(Int((state.progress * 100).rounded()))%")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .foregroundStyle(ColorHex.accent.color)
            }
        }
    }
}

private struct ProgressRing: View {
    let progress: Double
    var size: CGFloat = 18
    var line: CGFloat = 3
    var body: some View {
        ZStack {
            Circle().stroke(Color.white.opacity(0.18), lineWidth: line)
            Circle().trim(from: 0, to: max(0.02, min(1, progress)))
                .stroke(ColorHex.accent.color, style: .init(lineWidth: line, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: size, height: size)
    }
}

// MARK: - Expanded regions

/// Leading: just the live pose (icon-only) so nothing crowds the narrow slot.
private struct ExpandedPose: View {
    let state: MabelLiveActivityAttributes.ContentState
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10, style: .continuous).fill(WidgetTheme.viewport)
            if let img = poseUIImage(state.poseImage) {
                Image(uiImage: img).resizable().scaledToFit().padding(2)
                    .opacity(state.isLive ? 1 : 0.6)
            } else {
                Text(state.activityEmoji).font(.system(size: 22))
            }
        }
        .frame(width: 44, height: 44)
    }
}

/// Center: the robot name + connection — the wide slot, so the name never clips.
private struct ExpandedIdentity: View {
    let state: MabelLiveActivityAttributes.ContentState
    let name: String
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name).font(.system(size: 15, weight: .bold))
                .foregroundStyle(.white).lineLimit(1).minimumScaleFactor(0.8)
            HStack(spacing: 4) {
                WidgetStatusDot(color: state.connectionColor.color, size: 5)
                Text(state.connectionLabel).font(.system(size: 10, weight: .medium))
                    .foregroundStyle(ColorHex.textSecondary.color).lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.leading, 2)
    }
}

private struct ExpandedStatus: View {
    let state: MabelLiveActivityAttributes.ContentState
    var body: some View {
        VStack(alignment: .trailing, spacing: 3) {
            HStack(spacing: 3) {
                Text("\(state.batteryPercent)%").font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Image(systemName: state.charging ? "battery.100.bolt" : "bolt.fill")
                    .font(.system(size: 11, weight: .bold)).foregroundStyle(state.batteryColor.color)
            }
            HStack(spacing: 3) {
                Image(systemName: state.transportIcon).font(.system(size: 9, weight: .semibold))
                Text("\(Int(state.latencyMs)) ms").font(.system(size: 9, weight: .medium))
            }
            .foregroundStyle(ColorHex.textSecondary.color)
        }
    }
}

private struct ExpandedBottom: View {
    let state: MabelLiveActivityAttributes.ContentState
    var body: some View {
        VStack(spacing: 8) {
            // Mode header — full width, so the words never clip. Emoji + mode on
            // the left, SIM/REAL on the right.
            HStack(spacing: 7) {
                Text(state.activityEmoji).font(.system(size: 17))
                VStack(alignment: .leading, spacing: 1) {
                    Text(state.activityTitle).font(.system(size: 14, weight: .bold))
                        .foregroundStyle(.white).lineLimit(1).minimumScaleFactor(0.8)
                    Text(state.activityDetail).font(.system(size: 10, weight: .medium))
                        .foregroundStyle(ColorHex.textTertiary.color).lineLimit(1)
                }
                Spacer(minLength: 6)
                Text(state.target).font(.system(size: 10, weight: .heavy)).tracking(0.5)
                    .foregroundStyle(state.targetColor.color)
                    .padding(.horizontal, 7).padding(.vertical, 3)
                    .background(state.targetColor.color.opacity(0.18),
                                in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            }

            if state.estop {
                Label("EMERGENCY STOP ENGAGED", systemImage: "hand.raised.fill")
                    .font(.system(size: 11, weight: .heavy)).foregroundStyle(.white)
                    .frame(maxWidth: .infinity).padding(.vertical, 6)
                    .background(ColorHex.danger.color.opacity(0.92), in: Capsule())
            } else if state.hasProgress {
                // Uber-style: a prominent bar with time remaining.
                VStack(spacing: 4) {
                    HStack {
                        Text(state.hasETA ? "Time remaining" : "Progress")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(ColorHex.textTertiary.color)
                        Spacer()
                        Text(state.hasETA ? "\(state.etaText) left"
                                          : "\(Int((state.progress * 100).rounded()))%")
                            .font(.system(size: 12, weight: .bold, design: .rounded))
                            .foregroundStyle(ColorHex.accent.color)
                    }
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.white.opacity(0.16)).frame(height: 8)
                            Capsule().fill(ColorHex.accent.color)
                                .frame(width: max(8, geo.size.width * min(1, state.progress)), height: 8)
                        }
                    }
                    .frame(height: 8)
                }
            } else {
                // Idle / teleop: a glanceable status strip.
                HStack(spacing: 6) {
                    chip(state.isDriver ? "steeringwheel" : "gauge.with.dots.needle.50percent",
                         String(format: "%.1f m/s", state.linearSpeed))
                    chip(state.transportIcon, state.transport)
                    Spacer(minLength: 0)
                    chip(state.charging ? "battery.100.bolt" : "bolt.fill", "\(state.batteryPercent)%")
                }
            }
        }
    }

    private func chip(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon).font(.system(size: 9, weight: .semibold)).foregroundStyle(ColorHex.accent.color)
            Text(text).font(.system(size: 10, weight: .semibold)).foregroundStyle(.white).lineLimit(1)
        }
        .padding(.horizontal, 7).padding(.vertical, 3)
        .background(Color.white.opacity(0.08), in: Capsule())
    }
}

// MARK: - Lock Screen banner

private struct LiveBanner: View {
    let state: MabelLiveActivityAttributes.ContentState
    let name: String

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous).fill(WidgetTheme.viewport)
                if let img = poseUIImage(state.poseImage) {
                    Image(uiImage: img).resizable().scaledToFit().padding(3)
                        .opacity(state.isLive ? 1 : 0.6)
                } else {
                    Text(state.activityEmoji).font(.system(size: 30))
                }
            }
            .frame(width: 64, height: 64)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(state.activityEmoji).font(.system(size: 14))
                    Text(state.activityTitle).font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.white).lineLimit(1)
                    Spacer(minLength: 0)
                    Text(state.target).font(.system(size: 9, weight: .heavy)).tracking(0.5)
                        .foregroundStyle(state.targetColor.color)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(state.targetColor.color.opacity(0.18), in: RoundedRectangle(cornerRadius: 5))
                }

                if state.estop {
                    Text("🛑 EMERGENCY STOP").font(.system(size: 11, weight: .heavy))
                        .foregroundStyle(ColorHex.danger.color)
                } else if state.hasProgress {
                    VStack(spacing: 3) {
                        ProgressView(value: state.progress).tint(ColorHex.accent.color)
                        HStack {
                            Text(state.activityDetail).font(.system(size: 10))
                                .foregroundStyle(ColorHex.textTertiary.color)
                            Spacer()
                            Text(state.hasETA ? "\(state.etaText) left" : "\(Int((state.progress * 100).rounded()))%")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundStyle(ColorHex.accent.color)
                        }
                    }
                } else {
                    HStack(spacing: 6) {
                        WidgetStatusDot(color: state.connectionColor.color, size: 5)
                        Text(state.connectionLabel).font(.system(size: 10, weight: .medium))
                            .foregroundStyle(ColorHex.textSecondary.color)
                        Spacer(minLength: 0)
                        Image(systemName: state.charging ? "battery.100.bolt" : "bolt.fill")
                            .font(.system(size: 10, weight: .bold)).foregroundStyle(state.batteryColor.color)
                        Text("\(state.batteryPercent)%").font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                    }
                }
            }
        }
        .padding(14)
    }
}

// MARK: - Pose image loader

private func poseUIImage(_ name: String?) -> UIImage? {
    guard let name, let url = MabelWidgetShared.fileURL(name) else { return nil }
    return UIImage(contentsOfFile: url.path)
}

// MARK: - Preview

#Preview("Island", as: .dynamicIsland(.expanded), using: MabelLiveActivityAttributes(robotName: "MABEL")) {
    MabelLiveActivityWidget()
} contentStates: {
    MabelLiveActivityAttributes.ContentState.preview
}
