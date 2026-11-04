//
//  PolicyDetailView.swift
//  MABEL Mobile App
//
//  Review sheet for a policy: a looping clip of the task, the full
//  description, offline-eval metrics, and a dispatch control.
//

import SwiftUI

struct PolicyDetailView: View {
    let policy: Policy
    let isRunning: Bool
    var onToggleRun: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    // Compact preview clip.
                    LoopingVideo(resource: policy.clip)
                        .aspectRatio(16.0/9.0, contentMode: .fill)
                        .frame(maxWidth: .infinity).frame(height: 150)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
                        .overlay(alignment: .topLeading) {
                            Label("PREVIEW", systemImage: "play.circle.fill")
                                .font(.system(size: 8.5, weight: .bold)).tracking(0.6).foregroundStyle(.white)
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .background(.black.opacity(0.4), in: Capsule()).padding(8)
                        }
                        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }

                    // Title + description card.
                    GlassCard {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(spacing: 11) {
                                Image(systemName: policy.icon)
                                    .font(.system(size: 16, weight: .semibold)).foregroundStyle(policy.kindTint)
                                    .frame(width: 38, height: 38)
                                    .background(policy.kindTint.opacity(0.15), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(policy.name).font(.mabelHeadline).foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                                    Text(policy.kind.rawValue.uppercased())
                                        .font(.system(size: 8.5, weight: .bold)).tracking(1).foregroundStyle(policy.kindTint)
                                }
                                Spacer()
                            }
                            Text(policy.detail)
                                .font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    // Eval metrics card.
                    GlassCard {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("OFFLINE EVAL").font(.mabelCaption).foregroundStyle(Theme.Palette.textTertiary).tracking(1.4)
                            metricRow("Success rate", String(format: "%.0f%%", policy.successRate * 100), "checkmark.seal.fill", Theme.Palette.live)
                            if policy.horizonSec > 0 { metricRow("Horizon", "\(policy.horizonSec)s", "timer", Theme.Palette.accent) }
                            metricRow("Control rate", "\(policy.freqHz) Hz", "waveform.path.ecg", Theme.Palette.warn)
                            if policy.episodes > 0 { metricRow("Demos", "\(policy.episodes)", "film.stack", Color(hex: 0xB18CFF)) }
                            metricRow("Last run", policy.lastRun, "clock.arrow.circlepath", Theme.Palette.textSecondary)
                        }
                    }

                    Button {
                        Haptics.rigid(); onToggleRun()
                        if !isRunning { dismiss() }
                    } label: {
                        Label(isRunning ? "Stop Policy" : "Dispatch to Robot",
                              systemImage: isRunning ? "stop.fill" : "play.fill")
                            .font(.mabelHeadline)
                            .foregroundStyle(isRunning ? Theme.Palette.danger : .white)
                            .frame(maxWidth: .infinity).padding(.vertical, 13)
                            .background(isRunning ? AnyShapeStyle(Theme.Palette.danger.opacity(0.16)) : AnyShapeStyle(Theme.accentGradient), in: Capsule())
                    }
                    .buttonStyle(PressableStyle())
                }
                .padding(Theme.Space.lg)
            }
            .scrollIndicators(.hidden)
            .background(Theme.canvasGradient.ignoresSafeArea())
            .navigationTitle(policy.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }.tint(Theme.Palette.accent)
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func metricRow(_ label: String, _ value: String, _ icon: String, _ tint: Color) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon).font(.system(size: 13, weight: .semibold)).foregroundStyle(tint).frame(width: 22)
            Text(label).font(.mabelBody).foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            Text(value).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
        }
    }
}
