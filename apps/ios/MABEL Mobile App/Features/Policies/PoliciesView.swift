//
//  PoliciesView.swift
//  MABEL Mobile App
//
//  The robot's command center. Four stacked glass cards — matching the Settings
//  view's language exactly (GlassCard surfaces, caps section titles, Theme
//  colours):
//
//    • Assistant     — talk or type to the robot; it replies in chat and aloud.
//    • Robot Control — dedicated buttons: dock & charge, go home, sleep,
//                      security patrol, remote speaker.
//    • Automations   — Apple-Home-style schedules ("Make coffee · 7:00 AM · Daily").
//    • Skills        — the learned/scripted behaviour library + data collection.
//

import SwiftUI

struct PoliciesView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.bottomBarInset) private var bottomBarInset
    @Environment(\.isLandscape) private var isLandscape
    @StateObject private var hub = RobotHub()
    @StateObject private var schedule = TaskScheduleStore()

    @State private var selected: Policy? = nil
    @State private var editing: ScheduledTask? = nil
    @State private var addingTask = false
    @State private var showTimeline = false
    @State private var showChat = false

    private var running: Policy.ID? { controller.runningPolicy?.id }

    /// The tab has two faces now: COMPOSE (the choreography timeline — build a
    /// sequence and run it) and HUB (the assistant, quick commands, schedules and
    /// skills that were already here). They are different jobs, and the timeline
    /// needs the whole screen, so they are a switch rather than one more card.
    enum Mode: String, CaseIterable, Identifiable {
        case compose, hub
        var id: String { rawValue }
        var title: String { self == .compose ? "Compose" : "Hub" }
        var icon: String { self == .compose ? "slider.horizontal.below.rectangle" : "square.grid.2x2" }
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
            .padding(.horizontal, 14).padding(.top, 8).padding(.bottom, 6)

            if mode == .compose {
                AutonomyTimelineView()
            } else {
                hubBody
            }
        }
        .background(Theme.Palette.canvas)
    }

    private var hubBody: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                if isLandscape {
                    // Two balanced columns so the wide landscape canvas isn't one
                    // giant single column.
                    HStack(alignment: .top, spacing: 12) {
                        VStack(spacing: 12) { assistantCard; controlCard }
                            .frame(maxWidth: .infinity, alignment: .top)
                        VStack(spacing: 12) { automationsCard; skillsCard }
                            .frame(maxWidth: .infinity, alignment: .top)
                    }
                } else {
                    assistantCard
                    controlCard
                    automationsCard
                    skillsCard
                }
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.sm)
            .padding(.bottom, Theme.Space.lg)
            .padding(.bottom, bottomBarInset)     // clear the floating nav bar
        }
        .scrollIndicators(.hidden)
        .scrollDismissesKeyboard(.interactively)
        .onAppear { hub.controller = controller }
        .sheet(item: $selected) { policy in
            PolicyDetailView(policy: policy, isRunning: running == policy.id) {
                if running == policy.id { controller.stopPolicy() }
                else { controller.dispatch(policy) }
            }
        }
        .sheet(isPresented: $addingTask) {
            AutomationEditorView(store: schedule,
                                 task: ScheduledTask(title: "", icon: "sparkles",
                                                     hour: 8, minute: 0,
                                                     weekdays: ScheduledTask.everyDay),
                                 isNew: true)
        }
        .sheet(item: $editing) { task in
            AutomationEditorView(store: schedule, task: task, isNew: false)
        }
        .sheet(isPresented: $showTimeline) {
            WeeklyScheduleView(store: schedule)
        }
        .sheet(isPresented: $showChat) {
            AssistantChatSheet(hub: hub)
        }
    }

    private var header: some View {
        ViewHeader(title: "Autonomy") {
            // Server-authoritative echoes first (2026-07): what the robot
            // REPORTS it is running — reflects every client, not just us.
            if let p = controller.state.policy, p.running, let name = p.selected {
                StatusPill(title: "Policy · \(name)", color: Theme.Palette.live, pulses: true)
            } else if let a = controller.state.animation, a.playing, let name = a.selected {
                StatusPill(title: "Anim · \(name)", color: Theme.Palette.accent, pulses: true)
            } else if let dc = controller.state.dataCollection, dc.active {
                StatusPill(title: "Recording · \(dc.label ?? "bag") · \(Int(dc.elapsedSec))s",
                           color: Theme.Palette.danger, pulses: true)
            } else if let p = controller.runningPolicy {
                StatusPill(title: "Running · \(p.name)", color: Theme.Palette.live, pulses: true)
            } else if let busy = hub.busyAction {
                StatusPill(title: busy, color: Theme.Palette.accent, pulses: true)
            }
        }
    }

    // MARK: - Assistant (chat + voice)

    private var assistantCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    cardTitle("ASSISTANT")
                    Spacer()
                    GlassIconButton(systemImage: "arrow.up.left.and.arrow.down.right",
                                    tint: Theme.Palette.accent) { showChat = true }
                    GlassIconButton(systemImage: hub.voiceReply ? "speaker.wave.2.fill" : "speaker.slash.fill",
                                    tint: hub.voiceReply ? Theme.Palette.accent : Theme.Palette.textSecondary,
                                    prominent: hub.voiceReply) { hub.voiceReply.toggle() }
                }

                // Compact preview of the latest turns — tap anywhere to open the
                // full conversation in a larger pop-up.
                Button { showChat = true; Haptics.tap() } label: {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(previewMessages) { ChatBubble(message: $0, lineLimit: 2) }
                        if hub.thinking { TypingBubble() }
                    }
                    .frame(maxWidth: .infinity, minHeight: 72, alignment: .topLeading)
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

    // MARK: - Robot control (dedicated buttons)

    private var controlCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                cardTitle("ROBOT CONTROL")
                HStack(spacing: 10) {
                    controlTile("Dock & Charge", "bolt.batteryblock.fill", active: false,
                                tint: Theme.Palette.live) { hub.dock() }
                    controlTile("Go Home", "house.fill", active: false,
                                tint: Theme.Palette.accent) { hub.goHome() }
                }
                HStack(spacing: 10) {
                    controlTile(hub.sleeping ? "Asleep" : "Sleep",
                                hub.sleeping ? "moon.zzz.fill" : "moon.fill",
                                active: hub.sleeping, tint: Color(hex: 0xB18CFF)) { hub.toggleSleep() }
                    controlTile("Security", "shield.lefthalf.filled",
                                active: hub.security, tint: Theme.Palette.warn) { hub.toggleSecurity() }
                }
                controlTile(hub.speaker ? "Speaker live — talk in chat" : "Speak through robot",
                            hub.speaker ? "megaphone.fill" : "megaphone",
                            active: hub.speaker, tint: Theme.Palette.accent, wide: true) { hub.toggleSpeaker() }
            }
        }
    }

    private func controlTile(_ title: String, _ icon: String, active: Bool, tint: Color,
                             wide: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(active ? .white : tint)
                    .frame(width: 24, height: 24)
                    .background((active ? Color.white.opacity(0.18) : tint.opacity(0.16)),
                                in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(active ? .white : Theme.Palette.textPrimary)
                    .lineLimit(1).minimumScaleFactor(0.8)
                Spacer(minLength: 0)
            }
            // Icon always hugs the leading edge — consistent across every tile.
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(minHeight: 42)
            .padding(.horizontal, 10)
            .background(active ? AnyShapeStyle(tint.gradient) : AnyShapeStyle(Theme.Palette.surface),
                        in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous)
                    .strokeBorder(active ? tint.opacity(0.0) : Theme.Palette.hairlineSoft, lineWidth: 1)
            }
        }
        .buttonStyle(PressableStyle())
    }

    // MARK: - Automations (scheduled routines)

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
                // Tap the calendar to open the full weekly timeline, where routines
                // are added (the + lives there) and dragged around.
                MiniWeekCalendar(tasks: schedule.tasks) { showTimeline = true; Haptics.tap() }
                if schedule.tasks.isEmpty {
                    Text("No routines yet. Tap the calendar to schedule a task — e.g. make coffee at 7:00 AM daily.")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
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
                .foregroundStyle(tint).frame(width: 30, height: 30)
                .background(tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 1) {
                Text(task.title.isEmpty ? "Untitled" : task.title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                Text("\(task.timeString) · \(task.repeatString)")
                    .font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.Palette.textTertiary)
            }
            Spacer()
            Toggle("", isOn: Binding(
                get: { task.enabled },
                set: { _ in schedule.toggle(task); Haptics.tap() }
            )).labelsHidden().tint(Theme.Palette.accent)
        }
        .padding(.vertical, 6).padding(.horizontal, 9)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 11).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        .contentShape(Rectangle())
        .onTapGesture { Haptics.tap(); editing = task }
        .contextMenu {
            Button(role: .destructive) { schedule.remove(task) } label: { Label("Delete", systemImage: "trash") }
        }
    }

    // MARK: - Skills (behaviour library + data collection)

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
            Haptics.rigid(); controller.collectingData = true
        } label: {
            HStack(spacing: 11) {
                Image(systemName: "record.circle").font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white).frame(width: 30, height: 30)
                    .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                VStack(alignment: .leading, spacing: 1) {
                    Text("Collect Data").font(.system(size: 14, weight: .semibold)).foregroundStyle(.white)
                    Text("Teleop & record episodes for training")
                        .font(.system(size: 10)).foregroundStyle(.white.opacity(0.85))
                }
                Spacer()
                Image(systemName: "arrow.right").font(.system(size: 13, weight: .semibold)).foregroundStyle(.white)
            }
            .padding(.vertical, 8).padding(.horizontal, 11)
            .background(Theme.accentGradient, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
        }
        .buttonStyle(PressableStyle())
    }

    private func policyRow(_ policy: Policy) -> some View {
        let isRunning = running == policy.id
        return HStack(spacing: 11) {
            Image(systemName: policy.icon).font(.system(size: 15, weight: .semibold))
                .foregroundStyle(policy.kindTint).frame(width: 30, height: 30)
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
            if isRunning {
                Image(systemName: "waveform").font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Theme.Palette.live).symbolEffect(.variableColor)
            } else {
                Text(String(format: "%.0f%%", policy.successRate * 100))
                    .font(.system(size: 12, weight: .heavy)).monospacedDigit()
                    .foregroundStyle(Theme.Palette.textSecondary)
                Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .padding(.vertical, 6).padding(.horizontal, 9)
        .background(isRunning ? Theme.Palette.live.opacity(0.08) : Theme.Palette.surface,
                    in: RoundedRectangle(cornerRadius: 11, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 11)
            .strokeBorder(isRunning ? Theme.Palette.live.opacity(0.3) : Theme.Palette.hairlineSoft, lineWidth: 1) }
        .contentShape(Rectangle())
        .onTapGesture { Haptics.tap(); selected = policy }
    }

    // MARK: - Helpers

    private func cardTitle(_ t: String) -> some View { SectionLabel(t) }
}
