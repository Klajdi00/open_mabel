//
//  RecordingControls.swift
//  MABEL — shared by the iPhone and Vision Pro apps (BYTE-IDENTICAL: diff
//  apps/ios/MABEL Mobile App/Features/Robot/RecordingControls.swift
//  against apps/visionos/MabelTeleop/Features/Robot/ — it must
//  print nothing. Platform differences are `#if os(visionOS)` inside this file.)
//
//  ONE recording flow (operator, 2026-10-14). There used to be two recorders
//  behind a drop-down — "Demo clip" (operator commands, replayable) and
//  "Policy data" (realized state, for training). Now there is one take:
//
//    Record ─▶ small "New Recording" card: folder name + "Episode N" preview
//           ─▶ Start ─▶ the card closes, the top-bar slot becomes a big
//              STOP RECORDING button (elapsed · folder · episode)
//           ─▶ Stop ─▶ episode saved; the next Record pre-fills the same
//              folder at Episode N+1.
//
//  The server saves each take TWICE from one `record` message: the curation
//  H5 (`<folder>/episode_XXXX.h5`) AND a replayable command clip named
//  `<folder>_episode_XXXX` (sim_teleop_bridge.py, the `_record_pending`
//  "start" branch). The Replay card lists those clips grouped by folder.
//

import SwiftUI

// MARK: - Naming (mirrors the server; pure, so it is testable)

enum EpisodeNaming {
    /// The server's folder sanitiser (`CurationRecorder._safe`): keep
    /// alphanumerics, '-', '_' and spaces, trim, spaces → '_'. Applied here so
    /// the "Episode N" preview counts the folder the server will ACTUALLY use.
    static func safeFolder(_ raw: String) -> String {
        let kept = raw.unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0) || "-_ ".unicodeScalars.contains($0)
        }
        return String(String.UnicodeScalarView(kept))
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " ", with: "_")
    }

    /// `pick_cube_episode_0003` → ("pick_cube", 3). nil for any other name
    /// (clips recorded before the merge, e.g. `demo_1423`).
    static func parse(_ clip: String) -> (folder: String, index: Int)? {
        guard let r = clip.range(of: "_episode_", options: .backwards) else { return nil }
        let folder = String(clip[..<r.lowerBound])
        guard !folder.isEmpty, let idx = Int(clip[r.upperBound...]) else { return nil }
        return (folder, idx)
    }

    /// Files are 0-based (`episode_0000.h5`, the IL writer's own index); people
    /// count from 1. Every on-screen label goes through here so the two never mix.
    static func label(_ index: Int) -> String { "Episode \(index + 1)" }

    /// Clips grouped by folder, folders A→Z, episodes newest first. Clips that
    /// do not follow the episode naming land in a trailing "Other clips" group.
    static func groups(_ clips: [RobotState.DemoClipInfo])
        -> [(folder: String, clips: [RobotState.DemoClipInfo])] {
        var by: [String: [RobotState.DemoClipInfo]] = [:]
        var other: [RobotState.DemoClipInfo] = []
        for c in clips {
            if let p = parse(c.name) { by[p.folder, default: []].append(c) } else { other.append(c) }
        }
        var out = by.keys.sorted().map { f in
            (folder: f, clips: by[f]!.sorted {
                (parse($0.name)?.index ?? 0) > (parse($1.name)?.index ?? 0) })
        }
        if !other.isEmpty { out.append((folder: otherGroup, clips: other)) }
        return out
    }
    static let otherGroup = "Other clips"

    static func duration(_ s: Double) -> String {
        String(format: "%d:%02d", Int(s) / 60, Int(s) % 60)
    }
}

// MARK: - Shared chrome

/// The one "small window" every recording pop-up uses: header row (glyph,
/// title, ✕), content, fixed width, glass card over a dim scrim. The Record
/// card and the Replay card are both this, so they cannot drift apart.
struct RecordingCard<Content: View>: View {
    let icon: String
    let iconTint: Color
    let title: String
    let onClose: () -> Void
    @ViewBuilder var content: Content

    #if os(visionOS)
    static var width: CGFloat { 400 }
    #else
    static var width: CGFloat { 340 }
    #endif

    var body: some View {
        ZStack {
            // Tap-outside dismisses, like every other card in the apps.
            Color.black.opacity(0.32).ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture { onClose() }
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 10) {
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(iconTint)
                    Text(title)
                        .font(.system(size: 17, weight: .bold))
                        .foregroundStyle(Theme.Palette.textPrimary)
                    Spacer(minLength: 8)
                    Button(action: onClose) {
                        Image(systemName: "xmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(Theme.Palette.textSecondary)
                            .frame(width: 30, height: 30)
                            .background(Theme.Palette.surfaceHi, in: Circle())
                            .contentShape(Circle())
                    }
                    .buttonStyle(PressableStyle())
                    .accessibilityLabel("Close")
                    #if os(visionOS)
                    .hoverEffect()
                    #endif
                }
                content
            }
            .padding(18)
            .frame(width: Self.width)
            .modifier(RecordingCardBackground())
            .transition(.scale(scale: 0.94).combined(with: .opacity))
        }
    }
}

private struct RecordingCardBackground: ViewModifier {
    func body(content: Content) -> some View {
        #if os(visionOS)
        content.glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        #else
        content
            .background(.regularMaterial,
                        in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
            .shadow(color: .black.opacity(0.35), radius: 24, y: 10)
        #endif
    }
}

/// Full-width card button. `prominent` = filled (the card's one primary verb).
private struct CardButton: View {
    let title: String
    var icon: String? = nil
    var tint: Color = Theme.Palette.textPrimary
    var prominent = false
    var enabled = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                if let icon { Image(systemName: icon).font(.system(size: 13, weight: .heavy)) }
                Text(title).font(.system(size: 15, weight: .bold))
            }
            .foregroundStyle(prominent ? Color.white : tint)
            .frame(maxWidth: .infinity).frame(height: 44)
            .background {
                RoundedRectangle(cornerRadius: 13, style: .continuous)
                    .fill(prominent ? AnyShapeStyle(tint) : AnyShapeStyle(Theme.Palette.surfaceHi))
            }
            .contentShape(RoundedRectangle(cornerRadius: 13, style: .continuous))
            .opacity(enabled ? 1 : 0.4)
        }
        .buttonStyle(PressableStyle())
        .disabled(!enabled)
        #if os(visionOS)
        .hoverEffect()
        #endif
    }
}

private struct RecordSectionLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10.5, weight: .heavy)).tracking(1)
            .foregroundStyle(Theme.Palette.textTertiary)
    }
}

// MARK: - Record card

/// "New Recording": type (or tap) a folder, see which episode this take will
/// be, press Start. Nothing else — every take is replayable AND training data.
struct RecordStartCard: View {
    @Binding var folder: String
    /// Existing folders on the robot, most-relevant first.
    let folders: [String]
    /// Episodes already saved per folder (`robot_state.recording.taskCounts`).
    let counts: [String: Int]
    let onCancel: () -> Void
    let onStart: (_ folder: String, _ episodeIndex: Int) -> Void

    @FocusState private var focused: Bool

    private var safe: String { EpisodeNaming.safeFolder(folder) }
    private var nextIndex: Int { counts[safe] ?? 0 }
    private var canStart: Bool { !safe.isEmpty }

    var body: some View {
        RecordingCard(icon: "record.circle.fill", iconTint: Theme.Palette.danger,
                      title: "New Recording", onClose: onCancel) {
            VStack(alignment: .leading, spacing: 8) {
                RecordSectionLabel(text: "Folder")
                HStack(spacing: 8) {
                    Image(systemName: "folder.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.Palette.accent)
                    TextField("e.g. pick_cube", text: $folder)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textPrimary)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .submitLabel(.go)
                        .focused($focused)
                        .onSubmit { if canStart { onStart(safe, nextIndex) } }
                    if !folder.isEmpty {
                        Button { folder = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(Theme.Palette.textTertiary)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Clear folder name")
                    }
                }
                .padding(.horizontal, 12).frame(height: 46)
                .background(Theme.Palette.surface,
                            in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }

                if !folders.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(folders, id: \.self) { f in
                                Button { folder = f; Haptics.select() } label: {
                                    HStack(spacing: 5) {
                                        Text(f).font(.system(size: 12.5, weight: .semibold))
                                        Text("\(counts[f] ?? 0)")
                                            .font(.system(size: 11, weight: .bold).monospacedDigit())
                                            .foregroundStyle(Theme.Palette.textTertiary)
                                    }
                                    .foregroundStyle(f == safe ? Color.white : Theme.Palette.textPrimary)
                                    .padding(.horizontal, 10).frame(height: 30)
                                    .background {
                                        Capsule(style: .continuous)
                                            .fill(f == safe ? AnyShapeStyle(Theme.Palette.accent)
                                                            : AnyShapeStyle(Theme.Palette.surfaceHi))
                                    }
                                    .contentShape(Capsule(style: .continuous))
                                }
                                .buttonStyle(PressableStyle())
                                #if os(visionOS)
                                .hoverEffect()
                                #endif
                            }
                        }
                    }
                }
            }

            // What this take will be called — the answer to "where does it go?"
            HStack(spacing: 10) {
                Image(systemName: "film.stack")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textSecondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(canStart ? EpisodeNaming.label(nextIndex) : "Name a folder to start")
                        .font(.system(size: 15, weight: .bold).monospacedDigit())
                        .foregroundStyle(canStart ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
                        .contentTransition(.numericText())
                    Text(canStart
                         ? (nextIndex == 0 ? "New folder · \(safe)" : "\(nextIndex) saved in \(safe)")
                         : "Episodes number themselves 1, 2, 3 …")
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(1).truncationMode(.middle)
                }
                Spacer(minLength: 0)
            }
            .animation(.snappy, value: nextIndex)

            HStack(spacing: 10) {
                CardButton(title: "Cancel", action: onCancel)
                CardButton(title: "Start", icon: "record.circle", tint: Theme.Palette.danger,
                           prominent: true, enabled: canStart) {
                    onStart(safe, nextIndex)
                }
            }
        }
        .onAppear {
            // An empty field gets the keyboard straight away; a pre-filled one
            // (same folder as last time) is one tap on Start.
            if folder.isEmpty { focused = true }
        }
    }
}

// MARK: - Replay card

/// Every recorded episode, grouped by folder. Tap ▶ to replay it through the
/// same controller (eases in, plays `repeats`×, eases home).
struct ReplayCard: View {
    let clips: [RobotState.DemoClipInfo]
    let live: Bool
    @Binding var repeats: Int
    let onPlay: (_ clip: String) -> Void
    let onDelete: (_ clip: String) -> Void
    let onClose: () -> Void

    @State private var confirmDelete: String?

    #if os(visionOS)
    private let listHeight: CGFloat = 340
    #else
    private let listHeight: CGFloat = 300
    #endif

    var body: some View {
        RecordingCard(icon: "play.circle.fill", iconTint: Theme.Palette.accent,
                      title: "Replay", onClose: onClose) {
            if clips.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "film.stack")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Text(live ? "No recordings yet — tap Record to make one."
                              : "Connect to MABEL to see its recordings.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity).padding(.vertical, 22)
            } else {
                ScrollView(showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(EpisodeNaming.groups(clips), id: \.folder) { g in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(spacing: 6) {
                                    Image(systemName: g.folder == EpisodeNaming.otherGroup
                                          ? "tray.full.fill" : "folder.fill")
                                        .font(.system(size: 11, weight: .bold))
                                        .foregroundStyle(Theme.Palette.accent)
                                    // Folder names are case-sensitive on disk —
                                    // show them as typed, not upper-cased.
                                    Text("\(g.folder) · \(g.clips.count)")
                                        .font(.system(size: 12, weight: .bold))
                                        .foregroundStyle(Theme.Palette.textSecondary)
                                }
                                .padding(.leading, 2)
                                VStack(spacing: 0) {
                                    ForEach(Array(g.clips.enumerated()), id: \.element.id) { i, c in
                                        if i > 0 { Divider().padding(.leading, 44) }
                                        row(c)
                                    }
                                }
                                .background(Theme.Palette.surface,
                                            in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            }
                        }
                    }
                }
                .frame(maxHeight: listHeight)
                .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                Image(systemName: "repeat")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Theme.Palette.textSecondary)
                Text("Play")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary)
                Spacer()
                stepButton("minus", enabled: repeats > 1) { repeats -= 1 }
                Text("\(repeats)×")
                    .font(.system(size: 15, weight: .bold).monospacedDigit())
                    .foregroundStyle(Theme.Palette.accent)
                    .frame(minWidth: 34)
                stepButton("plus", enabled: repeats < 20) { repeats += 1 }
            }
            Text("Eases to the start pose over 2 s, plays, then eases back home.")
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(Theme.Palette.textTertiary)
        }
    }

    private func title(_ c: RobotState.DemoClipInfo) -> String {
        if let p = EpisodeNaming.parse(c.name) { return EpisodeNaming.label(p.index) }
        return c.name
    }

    @ViewBuilder private func row(_ c: RobotState.DemoClipInfo) -> some View {
        HStack(spacing: 10) {
            Button { onPlay(c.name) } label: {
                HStack(spacing: 10) {
                    Image(systemName: "play.fill")
                        .font(.system(size: 12, weight: .black))
                        .foregroundStyle(.white)
                        .frame(width: 30, height: 30)
                        .background(Theme.Palette.accent, in: Circle())
                    VStack(alignment: .leading, spacing: 1) {
                        Text(title(c))
                            .font(.system(size: 14.5, weight: .semibold))
                            .foregroundStyle(Theme.Palette.textPrimary)
                        Text(EpisodeNaming.duration(c.duration)
                             + (c.recordedBy.isEmpty ? "" : " · \(c.recordedBy)"))
                            .font(.system(size: 11.5, weight: .medium).monospacedDigit())
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                    Spacer(minLength: 4)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(PressableStyle())
            .accessibilityLabel("Replay \(title(c))")

            // Two-step delete: the trash turns into a red "Delete" on first tap.
            if confirmDelete == c.name {
                Button { onDelete(c.name); confirmDelete = nil } label: {
                    Text("Delete").font(.system(size: 12.5, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10).frame(height: 28)
                        .background(Theme.Palette.danger, in: Capsule(style: .continuous))
                }
                .buttonStyle(PressableStyle())
            } else {
                Button { withAnimation(.snappy) { confirmDelete = c.name } } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .frame(width: 30, height: 30)
                        .contentShape(Rectangle())
                }
                .buttonStyle(PressableStyle())
                .accessibilityLabel("Delete \(title(c))")
            }
        }
        .padding(.horizontal, 8).padding(.vertical, 6)
        #if os(visionOS)
        .hoverEffect()
        #endif
    }

    private func stepButton(_ icon: String, enabled: Bool, _ act: @escaping () -> Void) -> some View {
        Button { act(); Haptics.select() } label: {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(Theme.Palette.textPrimary)
                .frame(width: 30, height: 30)
                .background(Theme.Palette.surfaceHi, in: Circle())
                .contentShape(Circle())
                .opacity(enabled ? 1 : 0.35)
        }
        .buttonStyle(PressableStyle())
        .disabled(!enabled)
    }
}

// MARK: - Top-bar control

/// What sits in the Record slot of the top bar.
///   idle      → [● Record] [▶ Replay]
///   recording → one big red STOP RECORDING (elapsed · folder · episode)
///   replaying → one big accent STOP REPLAY (rep / repeats)
/// The stop states are deliberately LARGE: during a take the operator is
/// watching the robot and their own hands, not hunting for a 34 pt pill.
struct RecordTopControl: View {
    enum Mode: Equatable {
        case idle
        case recording(folder: String, episode: Int, since: Date)
        case replaying(name: String, rep: Int, repeats: Int)
    }
    let mode: Mode
    var replayEnabled = true
    let onRecord: () -> Void
    let onReplay: () -> Void
    let onStop: () -> Void

    #if os(visionOS)
    private let pillH: CGFloat = 36       // top-bar control standard
    private let stopH: CGFloat = 52
    #else
    private let pillH: CGFloat = 34
    private let stopH: CGFloat = 48
    #endif

    var body: some View {
        Group {
            switch mode {
            case .idle: idle
            case let .recording(folder, episode, since):
                stopButton(tint: Theme.Palette.danger, title: "STOP RECORDING") {
                    TimelineView(.periodic(from: since, by: 1)) { ctx in
                        Text(EpisodeNaming.duration(max(0, ctx.date.timeIntervalSince(since))))
                            .font(.system(size: 15, weight: .bold).monospacedDigit())
                    }
                } detail: {
                    "\(folder) · \(EpisodeNaming.label(episode))"
                }
            case let .replaying(name, rep, repeats):
                stopButton(tint: Theme.Palette.accent, title: "STOP REPLAY") {
                    Text("\(max(rep, 1))/\(max(repeats, 1))")
                        .font(.system(size: 15, weight: .bold).monospacedDigit())
                } detail: {
                    if let p = EpisodeNaming.parse(name) {
                        "\(p.folder) · \(EpisodeNaming.label(p.index))"
                    } else { name }
                }
            }
        }
        .animation(.snappy, value: mode)
    }

    private var idle: some View {
        HStack(spacing: 6) {
            Button(action: onRecord) {
                HStack(spacing: 7) {
                    Circle().fill(Theme.Palette.danger).frame(width: 10, height: 10)
                    Text("Record").font(.system(size: 13, weight: .bold))
                }
                .foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, 14).frame(height: pillH)
                .modifier(PillGlass())
            }
            .buttonStyle(PressableStyle())
            .accessibilityIdentifier("recordButton")
            #if os(visionOS)
            .contentShape(.hoverEffect, Capsule(style: .continuous))
            .hoverEffect()
            #endif

            Button(action: onReplay) {
                Image(systemName: "play.fill")
                    .font(.system(size: 12, weight: .black))
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .frame(width: pillH, height: pillH)
                    .modifier(PillGlass())
                    .opacity(replayEnabled ? 1 : 0.4)
            }
            .buttonStyle(PressableStyle())
            .disabled(!replayEnabled)
            .accessibilityLabel("Replay")
            .accessibilityIdentifier("replayButton")
            #if os(visionOS)
            .contentShape(.hoverEffect, Capsule(style: .continuous))
            .hoverEffect()
            #endif
        }
        .transition(.opacity)
    }

    private func stopButton<Trailing: View>(tint: Color, title: String,
                                            @ViewBuilder trailing: () -> Trailing,
                                            detail: () -> String) -> some View {
        let d = detail()
        return Button(action: onStop) {
            HStack(spacing: 10) {
                RoundedRectangle(cornerRadius: 3, style: .continuous).fill(.white)
                    .frame(width: 14, height: 14)
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 8) {
                        Text(title).font(.system(size: 15, weight: .heavy)).tracking(0.4)
                        trailing().foregroundStyle(.white.opacity(0.8))
                    }
                    Text(d)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.75))
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            .foregroundStyle(.white)
            .padding(.leading, 18).padding(.trailing, 22).frame(height: stopH)
            .background { Capsule(style: .continuous).fill(tint) }
            .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.35), lineWidth: 1) }
            .shadow(color: tint.opacity(0.45), radius: 14, y: 5)
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier("stopRecordingButton")
        #if os(visionOS)
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
        #endif
        .transition(.scale(scale: 0.9).combined(with: .opacity))
    }
}

private struct PillGlass: ViewModifier {
    func body(content: Content) -> some View {
        #if os(visionOS)
        content.glassBackgroundEffect(in: Capsule(style: .continuous))
        #else
        content
            .background(.ultraThinMaterial, in: Capsule(style: .continuous))
            .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        #endif
    }
}
