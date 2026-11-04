//
//  ActionLibrarySheet.swift
//  MABEL Mobile App
//
//  The clip bin, as a POPUP rather than a panel. The library used to hold a
//  third of the screen permanently to show six of forty-four actions; it is now
//  one small button that opens this, you grab a clip, and it goes away again —
//  the shape every mobile editor's media browser has.
//
//  Everything is here in one sheet: a lane filter, search, and dense one-line
//  rows grouped by category. A `both` clip drops a linked pair, so it appears
//  under whichever lane you are filtering.
//

import SwiftUI

// MARK: - platform

/// How big an editor wants to be inside its MabelModal. This used to fork on
/// `presentationDetents`, which visionOS lacks; nothing presents a system sheet any
/// more, so there is nothing to fork and the #if is gone. These files stay
/// byte-identical between the iPhone and Vision Pro apps on purpose: a fix belongs
/// in both, and `diff` should say nothing.
enum TimelineSheetSize { case half, full }

extension View {
    /// Sizes the body inside a MabelModal. It used to apply `presentationDetents`
    /// and a drag indicator — system-sheet chrome that no longer exists, since
    /// editing floats in a MabelModal now. What is left is the honest part: how
    /// tall this particular editor wants to be, and making the list/form background
    /// transparent so the panel's material shows through instead of the
    /// inset-grouped grey fighting it.
    @ViewBuilder
    func timelineSheetSizing(_ size: TimelineSheetSize) -> some View {
        self
            .scrollContentBackground(.hidden)
            .frame(maxHeight: size == .half ? 420 : 620)
            // The bar carries Save/Cancel, so it stays — but wearing the app's
            // material and accent rather than the system's chrome.
            .toolbarBackground(.ultraThinMaterial, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .tint(Theme.Palette.accent)
    }
}

struct ActionLibrarySheet: View {
    @ObservedObject var catalog: ActionCatalog
    /// Where the clip lands.
    var playhead: TimeInterval
    var add: (ActionSpec) -> Void

    // Hosted in a MabelModal, not a system sheet, so the modal supplies the
    // way out; every dismiss() call below is unchanged.
    @Environment(\.mabelDismiss) private var dismiss
    @State private var lane: ActionLane?
    @State private var query = ""

    private var pool: [ActionSpec] {
        catalog.actions.filter { spec in
            // `blend` is real, executable motion, but it only means something
            // anchored to a specific pose jump: the safety guard inserts one
            // with the two poses it bridges baked into its params. A bare blend
            // dropped on the timeline would be a transition between nothing and
            // nothing, which is why the studio's palette does not offer it
            // either (motions.js, THE BLEND).
            spec.category != .transition
            && (lane == nil || spec.lane == lane || spec.lane == .both)
            && (query.isEmpty || spec.title.localizedCaseInsensitiveContains(query)
                || spec.id.localizedCaseInsensitiveContains(query))
        }
    }

    private var groups: [(ActionCategory, [ActionSpec])] {
        ActionCategory.allCases.compactMap { cat in
            let items = pool.filter { $0.category == cat }
            return items.isEmpty ? nil : (cat, items)
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                laneFilter
                List {
                    ForEach(groups, id: \.0) { cat, items in
                        Section {
                            ForEach(items) { spec in
                                Button {
                                    guard spec.available else { return }
                                    Haptics.select()
                                    add(spec)
                                    dismiss()
                                } label: { ActionRow(spec: spec) }
                                .buttonStyle(.plain)
                                .disabled(!spec.available)
                                .listRowInsets(EdgeInsets())
                                .accessibilityIdentifier("action.\(spec.id)")
                            }
                        } header: {
                            Text(cat.title).font(.system(size: 10, weight: .heavy)).tracking(0.5)
                        }
                    }
                }
                .listStyle(.insetGrouped)
                .environment(\.defaultMinListRowHeight, 34)
                .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always),
                            prompt: "Search \(pool.count) actions")
            }
            .navigationTitle("Add at \(TimelineCanvas.label(playhead))")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .timelineSheetSizing(.full)
    }

    private var laneFilter: some View {
        HStack(spacing: 6) {
            chip(nil, "All", "square.grid.2x2")
            ForEach(ActionLane.tracks) { l in
                chip(l, l.shortTitle, l.icon)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    private func chip(_ l: ActionLane?, _ title: String, _ icon: String) -> some View {
        let on = lane == l
        return Button {
            Haptics.select()
            withAnimation(Theme.Motion.gentle) { lane = l }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: icon).font(.system(size: 10, weight: .semibold))
                Text(title).font(.system(size: 11, weight: .bold))
            }
            .foregroundStyle(on ? Color.white : Theme.Palette.textSecondary)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(on ? Theme.Palette.accent : Theme.Palette.surface,
                        in: Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("lane.\(l?.rawValue ?? "all")")
    }
}

/// One dense line. 34 pt tall, so the sheet shows a dozen at a glance.
struct ActionRow: View {
    let spec: ActionSpec

    var body: some View {
        HStack(spacing: 8) {
            Text(spec.emoji.isEmpty ? "•" : spec.emoji).font(.system(size: 13))
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 0) {
                Text(spec.title).font(.system(size: 13, weight: .medium)).lineLimit(1)
                if let loc = spec.location {
                    Text("@ \(loc.label)").font(.system(size: 9))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
            Spacer(minLength: 6)
            if !spec.available {
                Text("SOON").font(.system(size: 8, weight: .heavy))
                    .padding(.horizontal, 4).padding(.vertical, 1)
                    .background(Theme.Palette.warn.opacity(0.22), in: Capsule())
            }
            Text(TimelineCanvas.label(spec.duration))
                .font(.system(size: 11, weight: .medium)).monospacedDigit()
                .foregroundStyle(Theme.Palette.textTertiary)
        }
        .padding(.horizontal, 12).padding(.vertical, 7)
        .contentShape(Rectangle())
        .opacity(spec.available ? 1 : 0.45)
    }
}

// MARK: - the clip's own settings, as a popover rather than a panel

/// What a selected clip does and how long it does it for. This is everything the
/// old inspector held, in a popover you open from the clip — so it costs nothing
/// when you are not using it, which a docked panel never manages.
struct ClipSettingsSheet: View {
    @ObservedObject var doc: TimelineDocument
    @ObservedObject var catalog: ActionCatalog
    let blockID: UUID
    var playhead: TimeInterval
    // Hosted in a MabelModal, not a system sheet, so the modal supplies the
    // way out; every dismiss() call below is unchanged.
    @Environment(\.mabelDismiss) private var dismiss

    private var block: TimelineBlock? { doc.timeline.blocks.first { $0.id == blockID } }

    var body: some View {
        NavigationStack {
            Group {
                if let b = block, let spec = catalog.spec(b.actionID) {
                    Form {
                        if let info = b.blend { blendSection(b, info) }
                        if spec.category == .language {
                            Section("Says") {
                                TextField("What the robot says",
                                          text: Binding(get: { b.say ?? "" },
                                                        set: { doc.setSay(b.id, $0) }),
                                          axis: .vertical)
                                    .lineLimit(1...4)
                                    .accessibilityIdentifier("clip.say")
                            }
                        }
                        if !spec.params.isEmpty {
                            Section("Parameters") {
                                ForEach(spec.params) { p in
                                    paramRow(b, p)
                                }
                            }
                        }
                        Section("Timing") {
                            row("Start", TimelineCanvas.timecode(b.start))
                            row("Length", String(format: "%.2f s", b.duration))
                            if b.inPoint > 0.001 {
                                row("Clip range",
                                    String(format: "%.1f – %.1f s", b.inPoint, b.outPoint))
                            }
                            if let clip = MotionClips.shared.clips[b.actionID], clip.peakVelocity > 0 {
                                row("Peak joint speed",
                                    String(format: "%.2f rad/s", clip.peakVelocity))
                            }
                            if !spec.holdable {
                                Text("Fixed-length motion — it can be trimmed, never stretched.")
                                    .font(.system(size: 11))
                                    .foregroundStyle(Theme.Palette.textTertiary)
                            }
                        }
                        Section {
                            Button {
                                doc.split(b.id, at: playhead); dismiss()
                            } label: { Label("Split at playhead", systemImage: "scissors") }
                            .disabled(playhead <= b.start + kMinBlockDuration
                                      || playhead >= b.end - kMinBlockDuration)
                            Button {
                                doc.duplicate(b.id); dismiss()
                            } label: { Label("Duplicate", systemImage: "plus.square.on.square") }
                            Button(role: .destructive) {
                                doc.remove([b.id]); dismiss()
                            } label: { Label("Delete", systemImage: "trash") }
                        }
                    }
                } else {
                    ContentUnavailableView("Clip removed", systemImage: "scissors")
                }
            }
            .navigationTitle(block.flatMap { catalog.spec($0.actionID)?.title } ?? "Clip")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        .timelineSheetSizing(.half)
    }

    private func paramRow(_ b: TimelineBlock, _ p: ActionParam) -> some View {
        let value = b.params[p.id] ?? p.def
        return VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(p.label).font(.system(size: 12, weight: .medium))
                Spacer()
                Text(String(format: "%.2f", value))
                    .font(.system(size: 12, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(Theme.Palette.accent)
            }
            Slider(value: Binding(get: { value }, set: { doc.setParam(b.id, p.id, $0) }),
                   in: p.min...p.max)
                .tint(Theme.Palette.accent)
                .accessibilityIdentifier("param.\(p.id)")
        }
    }

    // MARK: - a transition

    /// A bridge's inspector. The length is the only thing about it the operator
    /// decides — everything else is derived from the two poses it connects — so
    /// that is what this offers, with the consequence of the choice on screen
    /// next to it rather than discovered on the robot.
    @ViewBuilder
    private func blendSection(_ b: TimelineBlock, _ info: BlendInfo) -> some View {
        let floor = max(MotionMath.blendMinDur, info.minDuration)
        let peak = info.delta * MotionMath.easedPeakRate(b.duration)
        Section("Transition") {
            row("Bridges", info.edge == .lead ? "into the next clip"
                                              : "out of the last clip")
            if let j = info.joint {
                row("Widest gap", String(format: "%.2f rad · %@", info.delta, j))
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Ramp").font(.system(size: 13))
                    Spacer()
                    Text(String(format: "%.2f s", b.duration))
                        .font(.system(size: 13, weight: .semibold).monospacedDigit())
                        .foregroundStyle(Theme.Palette.textSecondary)
                }
                Slider(value: Binding(get: { b.duration },
                                      set: { doc.setBlendDuration(b.id, $0) }),
                       in: floor...max(floor + 0.1, MotionMath.blendMaxDur))
                    .accessibilityIdentifier("blend.ramp")
                // A bridge grows leftwards — its right edge is pinned to the
                // junction it closes — so a longer ramp reaches further back
                // into whatever precedes it. Say so; it is not obvious.
                Text(peak <= MotionMath.safeBlendVel + 1e-6
                     ? String(format: "%.2f rad/s — a gentle bridge.", peak)
                     : String(format: "%.2f rad/s — brisk. Lengthen it to soften, at the cost of reaching further back into the clip before it.", peak))
                    .font(.system(size: 11))
                    .foregroundStyle(peak <= MotionMath.maxJointVel
                                     ? Theme.Palette.textTertiary : Theme.Palette.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if info.userDuration != nil {
                Button("Back to automatic") { doc.setBlendDuration(b.id, nil) }
                    .font(.system(size: 13))
            }
            Text("Inserted automatically to keep the commanded pose continuous. "
                 + "It cannot be split, and its trailing edge is anchored to the "
                 + "junction it closes.")
                .font(.system(size: 11))
                .foregroundStyle(Theme.Palette.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            Text(v).font(.system(size: 12, weight: .semibold)).monospacedDigit()
        }
    }
}

// MARK: - annotations

/// Write the note that sits on the timeline at this moment.
struct NoteEditorSheet: View {
    @ObservedObject var doc: TimelineDocument
    /// `nil` = a NEW note, written only when you tap Done. Creating it on the
    /// way in left an orphan "note" chip behind every cancelled edit — and
    /// mutating the document in the same gesture that asks for this sheet stops
    /// the sheet presenting at all.
    let noteID: UUID?
    var at: TimeInterval = 0
    // Hosted in a MabelModal, not a system sheet, so the modal supplies the
    // way out; every dismiss() call below is unchanged.
    @Environment(\.mabelDismiss) private var dismiss
    @State private var text = ""
    @State private var seconds: Double = 2

    private var note: TimelineNote? {
        guard let noteID else { return nil }
        return doc.timeline.notes.first { $0.id == noteID }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Annotation") {
                    TextField("What happens here", text: $text, axis: .vertical)
                        .lineLimit(1...4)
                        .accessibilityIdentifier("note.text")
                }
                Section("Length") {
                    HStack {
                        Slider(value: $seconds, in: 0.5...20)
                        Text(String(format: "%.1f s", seconds))
                            .font(.system(size: 12, weight: .semibold)).monospacedDigit()
                    }
                }
                if let noteID {
                    Section {
                        Button(role: .destructive) {
                            doc.removeNote(noteID); dismiss()
                        } label: { Label("Delete note", systemImage: "trash") }
                    }
                }
            }
            .navigationTitle("Note")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        if let noteID {
                            doc.setNoteText(noteID, text)
                            doc.retimeNote(noteID, duration: seconds)
                        } else {
                            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                            let id = doc.addNote(trimmed.isEmpty ? "note" : trimmed,
                                                 at: at, duration: seconds)
                            doc.selectedNote = id
                        }
                        dismiss()
                    }
                }
            }
            .onAppear {
                text = note?.text ?? ""
                seconds = note?.duration ?? 2
            }
        }
        .timelineSheetSizing(.half)
    }
}
