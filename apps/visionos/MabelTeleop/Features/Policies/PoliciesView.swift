//
//  PoliciesView.swift
//  MabelTeleop
//
//  The robot's command center — parity with the iOS app: an Assistant chat
//  (Claude-backed), dedicated Robot Control buttons, Apple-Home-style Automations
//  with a weekly timeline, and the learned/scripted Skills library plus data
//  collection. Styled in the visionOS console language (GlassCard surfaces).
//

import SwiftUI

struct PoliciesView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.openImmersiveSpace) private var openImmersive
    @StateObject private var hub = RobotHub()
    @StateObject private var schedule = TaskScheduleStore()

    @State private var detail: Policy?
    @State private var editing: ScheduledTask?
    @State private var addingTask = false
    @State private var showTimeline = false
    @State private var showChat = false

    /// COMPOSE (the choreography workstation) or HUB (assistant, schedules,
    /// skills). Two different jobs; the timeline needs the whole window.
    enum Mode: String, CaseIterable, Identifiable {
        case compose, poses, hub
        var id: String { rawValue }
        var title: String {
            switch self { case .compose: return "Compose"; case .poses: return "Poses"; case .hub: return "Hub" }
        }
        var icon: String {
            switch self {
            case .compose: return "slider.horizontal.below.rectangle"
            case .poses:   return "figure.stand"
            case .hub:     return "square.grid.2x2"
            }
        }
    }
    @AppStorage("autonomyMode") private var mode: Mode = .compose

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $mode) {
                ForEach(Mode.allCases) { m in
                    Label(m.title, systemImage: m.icon).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 420)
            .padding(.top, 10)

            switch mode {
            case .compose: AutonomyTimelineView()
            // Saved poses — stored on the robot, synced across devices (Features/Poses).
            case .poses:   PosesPage()
            case .hub:     hubBody
            }
        }
    }

    private var hubBody: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                // Three columns — same structure as the Settings view.
                HStack(alignment: .top, spacing: 16) {
                    VStack(spacing: 16) { assistantCard; controlCard }.frame(maxWidth: .infinity, alignment: .top)
                    VStack(spacing: 16) { automationsCard }.frame(maxWidth: .infinity, alignment: .top)
                    VStack(spacing: 16) { skillsCard }.frame(maxWidth: .infinity, alignment: .top)
                }
            }
            .padding(.horizontal, 24)
            .padding(.top, 56)
            .padding(.bottom, 80)
        }
        .scrollIndicators(.hidden)
        .consoleSurface()
        .onAppear { hub.controller = controller; controller.refreshCollectionStatus() }
        .sheet(item: $detail) { PolicyDetailSheet(policy: $0) }
        .sheet(isPresented: $addingTask) {
            AutomationEditorView(store: schedule,
                                 task: ScheduledTask(title: "", icon: "sparkles", hour: 8, minute: 0,
                                                     weekdays: ScheduledTask.everyDay),
                                 isNew: true)
        }
        .sheet(item: $editing) { AutomationEditorView(store: schedule, task: $0, isNew: false) }
        .sheet(isPresented: $showTimeline) { WeeklyScheduleView(store: schedule) }
        .sheet(isPresented: $showChat) { AssistantChatSheet(hub: hub) }
    }

    private var header: some View {
        HStack {
            Text("Autonomy").font(.mabelViewTitle).foregroundStyle(Theme.Palette.textPrimary)
            Spacer()
            if let busy = hub.busyAction {
                StatusPill(title: busy, color: Theme.Palette.accent)
            }
        }
    }

    private func cardTitle(_ t: String) -> some View { SectionLabel(t) }

    // MARK: - Assistant

    private var assistantCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    cardTitle("ASSISTANT")
                    Spacer()
                    GlassIconButton(systemImage: "arrow.up.left.and.arrow.down.right",
                                    tint: Theme.Palette.accent) { showChat = true }
                    GlassIconButton(systemImage: hub.voiceReply ? "speaker.wave.2.fill" : "speaker.slash.fill",
                                    tint: hub.voiceReply ? Theme.Palette.accent : Theme.Palette.textTertiary,
                                    filled: hub.voiceReply) { hub.voiceReply.toggle() }
                }

                // Compact preview of the latest turns — tap to open the full chat.
                Button { showChat = true } label: {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(previewMessages) { ChatBubble(message: $0, lineLimit: 2) }
                        if hub.thinking { TypingBubble() }
                    }
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .topLeading)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                SuggestionBar(hub: hub)
                ChatInputBar(hub: hub)
            }
        }
    }

    /// The last couple of turns, shown as a tappable preview on the card.
    private var previewMessages: [RobotHub.Message] { Array(hub.messages.suffix(2)) }

    // MARK: - Robot control

    private var controlCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("ROBOT CONTROL")
                HStack(spacing: 10) {
                    controlTile("Dock & Charge", "bolt.batteryblock.fill", active: false, tint: Theme.Palette.live) { hub.dock() }
                    controlTile("Go Home", "house.fill", active: false, tint: Theme.Palette.accent) { hub.goHome() }
                }
                HStack(spacing: 10) {
                    controlTile(hub.sleeping ? "Asleep" : "Sleep", hub.sleeping ? "moon.zzz.fill" : "moon.fill",
                                active: hub.sleeping, tint: Color(hex: 0xB18CFF)) { hub.toggleSleep() }
                    controlTile("Security", "shield.lefthalf.filled", active: hub.security, tint: Theme.Palette.warn) { hub.toggleSecurity() }
                }
                controlTile(hub.speaker ? "Speaker live — talk in chat" : "Speak through robot",
                            hub.speaker ? "megaphone.fill" : "megaphone",
                            active: hub.speaker, tint: Theme.Palette.accent) { hub.toggleSpeaker() }
            }
        }
    }

    private func controlTile(_ title: String, _ icon: String, active: Bool, tint: Color,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon).font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(active ? .white : tint)
                    .frame(width: 24, height: 24)
                    .background(active ? Color.white.opacity(0.18) : tint.opacity(0.16),
                                in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                Text(title).font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(active ? .white : Theme.Palette.textPrimary)
                    .lineLimit(1).minimumScaleFactor(0.8)
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(minHeight: 44)
            .padding(.horizontal, 10)
            .background(active ? AnyShapeStyle(tint.gradient) : AnyShapeStyle(Theme.Palette.surface),
                        in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous)
                .strokeBorder(active ? .clear : Theme.Palette.hairlineSoft, lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
    }

    // MARK: - Automations

    private var automationsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    cardTitle("AUTOMATIONS")
                    Spacer()
                    Text("Tap to edit").font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Image(systemName: "chevron.right").font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
                // Tap the calendar to open the weekly timeline, where routines are
                // added (the + lives there) and dragged around.
                MiniWeekCalendar(tasks: schedule.tasks) { showTimeline = true }
                if schedule.tasks.isEmpty {
                    Text("No routines yet. Tap the calendar to schedule a task — e.g. make coffee at 7:00 AM daily.")
                        .font(.system(size: 12)).foregroundStyle(Theme.Palette.textTertiary)
                } else {
                    ForEach(schedule.tasks) { automationRow($0) }
                }
            }
        }
    }

    private func automationRow(_ task: ScheduledTask) -> some View {
        let tint = task.enabled ? Theme.Palette.accent : Theme.Palette.textTertiary
        return HStack(spacing: 11) {
            Image(systemName: task.icon).font(.system(size: 15, weight: .semibold))
                .foregroundStyle(tint).frame(width: 32, height: 32)
                .background(tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 1) {
                Text(task.title.isEmpty ? "Untitled" : task.title)
                    .font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                Text("\(task.timeString) · \(task.repeatString)")
                    .font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.Palette.textTertiary)
            }
            Spacer()
            Toggle("", isOn: Binding(get: { task.enabled }, set: { _ in schedule.toggle(task); Haptics.tap() }))
                .labelsHidden().tint(Theme.Palette.accent)
        }
        .padding(.vertical, 6).padding(.horizontal, 9)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 11).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        .contentShape(Rectangle())
        .onTapGesture { editing = task }
        .contextMenu {
            Button(role: .destructive) { schedule.remove(task) } label: { Label("Delete", systemImage: "trash") }
        }
    }

    // MARK: - Skills (behaviour library + data collection — compact half-column card)

    private var skillsCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("SKILLS")
                collectRow
                Divider().overlay(Theme.Palette.hairlineSoft)
                ForEach(Policy.library) { policyRow($0) }
            }
        }
    }

    private var collectRow: some View {
        Button {
            Haptics.rigid()
            Task { await openImmersive(id: WindowID.immersive.rawValue) }
        } label: {
            HStack(spacing: 11) {
                Image(systemName: "record.circle").font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white).frame(width: 32, height: 32)
                    .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                VStack(alignment: .leading, spacing: 1) {
                    Text("Collect Data").font(.system(size: 14, weight: .semibold)).foregroundStyle(.white)
                    Text(collectSummary).font(.system(size: 10)).foregroundStyle(.white.opacity(0.85)).lineLimit(2)
                }
                Spacer(minLength: 0)
                Image(systemName: "visionpro").font(.system(size: 14, weight: .semibold)).foregroundStyle(.white)
            }
            .padding(.vertical, 8).padding(.horizontal, 11)
            .background(Theme.accentGradient, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
        }.buttonStyle(PressableStyle())
    }

    private var collectSummary: String {
        if controller.episodesTotal > 0 {
            let tasks = controller.taskCounts.count
            return "\(controller.episodesTotal) episodes · \(tasks) task\(tasks == 1 ? "" : "s") on the server."
        }
        return "Record demonstration episodes in immersive teleop."
    }

    private func policyRow(_ policy: Policy) -> some View {
        Button { detail = policy } label: {
            HStack(spacing: 11) {
                Image(systemName: policy.icon).font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(policy.kindTint).frame(width: 32, height: 32)
                    .background(policy.kindTint.opacity(0.15), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(policy.name).font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                        Text(policy.kind.rawValue.uppercased()).font(.system(size: 8, weight: .bold)).tracking(0.6)
                            .foregroundStyle(policy.kindTint)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(policy.kindTint.opacity(0.16), in: Capsule())
                    }
                    Text(policy.summary).font(.system(size: 10)).foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
                }
                Spacer(minLength: 6)
                Text("\(Int(policy.successRate * 100))%").font(.system(size: 12, weight: .heavy)).monospacedDigit()
                    .foregroundStyle(Theme.Palette.textSecondary)
                Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(.vertical, 6).padding(.horizontal, 9)
            .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 11).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
    }
}

// MARK: - Detail sheet

private struct PolicyDetailSheet: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.dismiss) private var dismiss
    let policy: Policy

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                Image(systemName: policy.icon).font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(policy.kindTint).frame(width: 52, height: 52)
                    .background(policy.kindTint.opacity(0.14), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(policy.name).font(.mabelTitle).foregroundStyle(Theme.Palette.textPrimary)
                    Text(policy.kind.rawValue.uppercased()).font(.system(size: 10, weight: .heavy)).tracking(0.8)
                        .foregroundStyle(policy.kindTint)
                }
                Spacer()
                Button { dismiss() } label: {
                    Image(systemName: "xmark").font(.system(size: 13, weight: .bold))
                        .foregroundStyle(Theme.Palette.textTertiary).padding(10).background(Theme.Palette.surface, in: Circle())
                }.buttonStyle(.plain)
            }
            Text(policy.detail).font(.mabelBody).foregroundStyle(Theme.Palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 12) {
                metric("Success", "\(Int(policy.successRate * 100))%")
                metric("Rate", "\(policy.freqHz) Hz")
                metric("Horizon", policy.horizonSec > 0 ? "\(policy.horizonSec)s" : "—")
                metric("Demos", policy.episodes > 0 ? "\(policy.episodes)" : "—")
            }
            PrimaryButton(title: "Dispatch Policy", systemImage: "play.fill") {
                controller.dispatch(policy); dismiss()
            }
        }
        .padding(28)
        .frame(width: 520)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(spacing: 3) {
            Text(value).font(.mabel(18, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
            Text(label.uppercased()).font(.system(size: 9, weight: .bold)).tracking(0.6).foregroundStyle(Theme.Palette.textTertiary)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 10)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
    }
}
