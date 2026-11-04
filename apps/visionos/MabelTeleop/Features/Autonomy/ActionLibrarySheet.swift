//
//  ActionLibrarySheet.swift
//  MabelTeleop (visionOS)
//
//  The timeline's pop-ups, in the app's own overlay language (MabelOverlays):
//
//    • the LIBRARY — the one pop-up that should be BIG: a browser, not a list.
//      Category tabs (All · Animations · Poses · Navigation · Manipulation ·
//      Whole body · Voice · Coming soon), search, a recently-used row, a grid of
//      cards. Poses include the ROBOT-SAVED custom poses (added as a keyframe
//      that holds the pose) and a prominent "＋ New pose" that starts capture.
//      Tap = add at the playhead; long-press = details.
//    • the clip's settings, a text label, a keyframe — COMPACT editors on the
//      shared MabelModal panel (custom header, tiles, footer), no
//      NavigationStack/Form inside a modal (that drew an empty system bar on top
//      of every one of them and made them page-sized — 2026-10-25).
//
//  Byte-identical between the iPhone and Vision Pro apps bar this header; the
//  platform differences are the constants just below.
//

import SwiftUI

// MARK: - platform

#if os(visionOS)
private let kCardMinWidth: CGFloat = 180
private let kLibraryMinSize = CGSize(width: 980, height: 700)
#else
private let kCardMinWidth: CGFloat = 148
private let kLibraryMinSize = CGSize(width: 0, height: 0)
#endif

extension View {
    @ViewBuilder
    fileprivate func libraryKeyboardDismiss() -> some View {
        #if os(visionOS)
        self
        #else
        self.scrollDismissesKeyboard(.interactively)
        #endif
    }
}

/// Kept for callers that still size a body inside a MabelModal (transparent list
/// background, accent tint). Nothing in the timeline uses Form/List any more.
enum TimelineSheetSize { case half, full }

extension View {
    @ViewBuilder
    func timelineSheetSizing(_ size: TimelineSheetSize) -> some View {
        self
            .scrollContentBackground(.hidden)
            .frame(maxHeight: size == .half ? 420 : 620)
            .tint(Theme.Palette.accent)
    }
}

// MARK: - library model (pure — LibraryFilterTests pin it)

/// The library's tabs. Every catalog item has exactly ONE primary category
/// (`LibraryFilter.primary`); "All" shows everything.
enum LibraryCategory: String, CaseIterable, Identifiable {
    case all, animations, poses, navigation, manipulation, wholeBody, voice, comingSoon
    var id: String { rawValue }

    var title: String {
        switch self {
        case .all:          return "All"
        case .animations:   return "Animations"
        case .poses:        return "Poses"
        case .navigation:   return "Navigation"
        case .manipulation: return "Manipulation"
        case .wholeBody:    return "Whole body"
        case .voice:        return "Voice"
        case .comingSoon:   return "Coming soon"
        }
    }
    var icon: String {
        switch self {
        case .all:          return "square.grid.2x2"
        case .animations:   return "sparkles"
        case .poses:        return "figure.stand"
        case .navigation:   return "location.north.line.fill"
        case .manipulation: return "hand.raised.fill"
        case .wholeBody:    return "figure.walk"
        case .voice:        return "waveform"
        case .comingSoon:   return "hourglass"
        }
    }
    /// Accessibility ids. The lane tabs keep the ids the old lane chips had, so
    /// `AutonomyTimelineUITests` (lane.navigation, …) still finds them.
    var identifier: String {
        switch self {
        case .all:          return "lane.all"
        case .navigation:   return "lane.navigation"
        case .manipulation: return "lane.manipulation"
        case .voice:        return "lane.language"
        default:            return "lib.cat.\(rawValue)"
        }
    }
}

enum LibraryFilter {
    /// The ONE tab an action lives under (nil = never offered: the safety blend).
    ///   not deployable → Coming soon · pose → Poses · speech → Voice · nav → Navigation ·
    ///   otherwise by lane: nav → Navigation, both → Whole body,
    ///   manip → Animations (animations) / Manipulation (policies, skills).
    static func primary(_ s: ActionSpec) -> LibraryCategory? {
        if s.category == .transition { return nil }
        if !s.available { return .comingSoon }
        switch s.category {
        case .pose:      return .poses
        case .language:  return .voice
        case .nav:       return .navigation
        default: break
        }
        switch s.lane {
        case .navigation:   return .navigation
        case .both:         return .wholeBody
        case .language:     return .voice
        case .manipulation: return s.category == .animation ? .animations : .manipulation
        }
    }

    static func matches(_ s: ActionSpec, _ query: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return true }
        return s.title.localizedCaseInsensitiveContains(q)
            || s.id.replacingOccurrences(of: "_", with: " ").localizedCaseInsensitiveContains(q)
            || s.id.localizedCaseInsensitiveContains(q)
            || s.emoji == q
            || s.category.title.localizedCaseInsensitiveContains(q)
    }

    /// What a tab shows, registry order; "All" puts the not-yet-deployable last.
    static func items(_ specs: [ActionSpec], in cat: LibraryCategory, query: String = "") -> [ActionSpec] {
        let pool = specs.filter { primary($0) != nil && matches($0, query) }
        if cat == .all { return pool.filter(\.available) + pool.filter { !$0.available } }
        return pool.filter { primary($0) == cat }
    }

    static func counts(_ specs: [ActionSpec]) -> [LibraryCategory: Int] {
        var out: [LibraryCategory: Int] = [:]
        for s in specs { if let c = primary(s) { out[c, default: 0] += 1; out[.all, default: 0] += 1 } }
        return out
    }

    static func unavailableReason(_ s: ActionSpec) -> String? {
        s.available ? nil : "Coming soon — not deployable on this robot yet"
    }

    static func matches(_ p: SavedPose, _ query: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespaces)
        return q.isEmpty || p.name.localizedCaseInsensitiveContains(q) || "custom pose".contains(q.lowercased())
    }
}

/// Recently added actions, newest first, at most `limit`. Per device.
enum LibraryRecents {
    static let key = "timeline.library.recents"
    static let limit = 8

    static func record(_ id: String, into list: [String]) -> [String] {
        Array(([id] + list.filter { $0 != id }).prefix(limit))
    }
    static func load() -> [String] { UserDefaults.standard.stringArray(forKey: key) ?? [] }
    static func note(_ id: String) { UserDefaults.standard.set(record(id, into: load()), forKey: key) }
}

// MARK: - the library browser

struct ActionLibrarySheet: View {
    @ObservedObject var catalog: ActionCatalog
    /// Where the clip lands.
    var playhead: TimeInterval
    var add: (ActionSpec) -> Void
    /// Add a nav primitive already set to an amount (the Quick moves table).
    var addWith: (ActionSpec, [String: Double]) -> Void = { spec, _ in }
    /// Robot-saved poses (nil = this host does not offer them).
    var skills: RobotSkills? = nil
    /// Add a robot-saved pose at the playhead (as a holding keyframe).
    var addPose: (SavedPose) -> Void = { _ in }
    /// "＋ New pose" — the caller closes the library and opens the capture flow.
    var newPose: () -> Void = {}

    @Environment(\.mabelDismiss) private var mabelDismiss
    @Environment(\.dismiss) private var sysDismiss
    @State private var category: LibraryCategory = .all
    @State private var query = ""
    @State private var recents: [String] = LibraryRecents.load()
    @State private var detail: ActionSpec?
    @FocusState private var searching: Bool

    private func close() { mabelDismiss(); sysDismiss() }

    private var counts: [LibraryCategory: Int] { LibraryFilter.counts(catalog.actions) }
    private var items: [ActionSpec] { LibraryFilter.items(catalog.actions, in: category, query: query) }
    private var recentSpecs: [ActionSpec] {
        recents.compactMap { id in catalog.spec(id) }.filter { $0.available && LibraryFilter.primary($0) != nil }
    }
    private var showsSavedPoses: Bool { category == .poses || (category == .all && !query.isEmpty) }

    var body: some View {
        VStack(spacing: 0) {
            header
            searchField
                .padding(.horizontal, 14).padding(.bottom, 8)
            categoryBar
            Divider().overlay(Theme.Palette.hairlineSoft)
            ZStack(alignment: .bottom) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if category == .all, query.isEmpty, !recentSpecs.isEmpty { recentRow }
                        if category == .poses { posesHeader }
                        if category == .navigation, query.isEmpty {
                            NavQuickMoves(catalog: catalog) { spec, params in
                                Haptics.select(); LibraryRecents.note(spec.id)
                                addWith(spec, params); close()
                            }
                        }
                        if showsSavedPoses, let skills { SavedPoseCards(skills: skills, query: query, add: pick) }
                        grid
                    }
                    .padding(14)
                    .padding(.bottom, detail == nil ? 0 : 150)
                }
                .libraryKeyboardDismiss()
                if let d = detail { detailPanel(d).transition(.move(edge: .bottom).combined(with: .opacity)) }
            }
        }
        .frame(minWidth: kLibraryMinSize.width, minHeight: kLibraryMinSize.height)
        .background(Theme.Palette.canvas)
        .animation(Theme.Motion.snappy, value: detail?.id)
        .tint(Theme.Palette.accent)
    }

    // MARK: header / search / tabs

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Library").font(.system(size: 20, weight: .bold))
                    .foregroundStyle(Theme.Palette.textPrimary)
                Text("Tap to add at \(TimelineCanvas.timecode(playhead)) · long-press for details")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            Spacer(minLength: 0)
            Button(action: close) {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .frame(width: 30, height: 30)
                    .background(Theme.Palette.surface, in: Circle())
                    .overlay { Circle().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
            }
            .buttonStyle(PressableStyle())
            .accessibilityLabel("Close")
            .accessibilityIdentifier("modal.close")
        }
        .padding(.horizontal, 16).padding(.top, 16).padding(.bottom, 10)
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.Palette.textTertiary)
            TextField("Search \(counts[.all] ?? 0) actions — wave, bird, turn…", text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .focused($searching)
                .submitLabel(.search)
                .accessibilityIdentifier("library.search")
            if !query.isEmpty {
                Button { query = "" } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.Palette.textTertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 11).padding(.vertical, 9)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                .strokeBorder(searching ? Theme.Palette.accent.opacity(0.45) : Theme.Palette.hairlineSoft, lineWidth: 1)
        }
    }

    private var categoryBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(LibraryCategory.allCases) { c in
                    let n = c == .poses ? (counts[c] ?? 0) + (skills?.poses.count ?? 0) : (counts[c] ?? 0)
                    if n > 0 || c == .poses { tab(c, n) }
                }
            }
            .padding(.horizontal, 14).padding(.bottom, 10)
        }
    }

    private func tab(_ c: LibraryCategory, _ n: Int) -> some View {
        let on = category == c
        return Button {
            Haptics.select()
            withAnimation(Theme.Motion.snappy) { category = c; detail = nil }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: c.icon).font(.system(size: 11, weight: .semibold))
                Text(c.title).font(.system(size: 12.5, weight: .bold))
                Text("\(n)").font(.system(size: 10.5, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(on ? Color.white.opacity(0.8) : Theme.Palette.textTertiary)
            }
            .foregroundStyle(on ? Color.white : Theme.Palette.textSecondary)
            .padding(.horizontal, 11).padding(.vertical, 7)
            .background(on ? Theme.Palette.accent : Theme.Palette.surface, in: Capsule(style: .continuous))
            .overlay { Capsule().strokeBorder(on ? .clear : Theme.Palette.hairlineSoft, lineWidth: 1) }
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
        .accessibilityIdentifier(c.identifier)
    }

    // MARK: content

    private var recentRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Recently used")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(recentSpecs) { spec in
                        Button { pick(spec) } label: { RecentChip(spec: spec) }
                            .buttonStyle(PressableStyle())
                            .accessibilityIdentifier("recent.\(spec.id)")
                    }
                }
            }
        }
    }

    private var posesHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                Haptics.select(); newPose()
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "plus").font(.system(size: 16, weight: .heavy))
                        .foregroundStyle(.white)
                        .frame(width: 36, height: 36)
                        .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("New pose").font(.system(size: 15, weight: .bold)).foregroundStyle(.white)
                        Text(newPoseDetail).font(.system(size: 11)).foregroundStyle(.white.opacity(0.85))
                            .lineLimit(2)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "hand.draw.fill").font(.system(size: 18)).foregroundStyle(.white.opacity(0.85))
                }
                .padding(12)
                .background(Theme.accentGradient, in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
                .opacity(skills?.posesAvailability == .available ? 1 : 0.55)
            }
            .buttonStyle(PressableStyle())
            .disabled(skills?.posesAvailability != .available)
            .accessibilityIdentifier("library.newPose")
            sectionTitle("Saved on the robot")
        }
    }

    private var newPoseDetail: String {
        guard let skills else { return "Saved poses need a robot that stores them." }
        switch skills.posesAvailability {
        case .available:
            return "Robot goes limp (Torque OFF) — pose it by hand, then save it to the robot."
        default: return skills.posesAvailability.note
        }
    }

    @ViewBuilder
    private var grid: some View {
        let list = items
        if category == .poses { sectionTitle("Built-in poses") }
        if list.isEmpty {
            VStack(spacing: 6) {
                Image(systemName: "magnifyingglass").font(.system(size: 22)).foregroundStyle(Theme.Palette.textTertiary)
                Text(query.isEmpty ? "Nothing here yet." : "No action matches “\(query)”.")
                    .font(.system(size: 13, weight: .medium)).foregroundStyle(Theme.Palette.textSecondary)
            }
            .frame(maxWidth: .infinity).padding(.vertical, 30)
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: kCardMinWidth), spacing: 10)], spacing: 10) {
                ForEach(list) { spec in
                    ActionCard(spec: spec, selected: detail?.id == spec.id)
                        .onTapGesture { pick(spec) }
                        .onLongPressGesture(minimumDuration: 0.35) {
                            Haptics.rigid(); detail = spec
                        }
                        .accessibilityElement(children: .combine)
                        .accessibilityAddTraits(.isButton)
                        .accessibilityHint(spec.available ? "Adds at the playhead" : (LibraryFilter.unavailableReason(spec) ?? ""))
                        .accessibilityIdentifier("action.\(spec.id)")
                        .accessibilityAction(named: "Details") { detail = spec }
                }
            }
        }
    }

    private func sectionTitle(_ t: String) -> some View {
        Text(t.uppercased()).font(.system(size: 10, weight: .heavy)).tracking(0.8)
            .foregroundStyle(Theme.Palette.textTertiary)
    }

    private func detailPanel(_ spec: ActionSpec) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Text(spec.emoji.isEmpty ? "•" : spec.emoji).font(.system(size: 28))
                VStack(alignment: .leading, spacing: 3) {
                    Text(spec.title).font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.Palette.textPrimary)
                    Text(detailLine(spec)).font(.system(size: 11.5)).foregroundStyle(Theme.Palette.textSecondary)
                    if let r = LibraryFilter.unavailableReason(spec) {
                        Text(r).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Theme.Palette.warn)
                    }
                }
                Spacer(minLength: 0)
                Button { detail = nil } label: {
                    Image(systemName: "xmark").font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .frame(width: 24, height: 24).background(Theme.Palette.surface, in: Circle())
                }
                .buttonStyle(.plain)
            }
            if !spec.params.isEmpty {
                Text(spec.params.map { "\($0.label): \(String(format: "%g", $0.def)) (\(String(format: "%g", $0.min))–\(String(format: "%g", $0.max)))" }
                        .joined(separator: " · "))
                    .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
            }
            MabelModalFooter(cancelTitle: "Close", confirmTitle: "Add at \(TimelineCanvas.timecode(playhead))",
                             confirmEnabled: spec.available,
                             onCancel: { detail = nil }, onConfirm: { pick(spec) })
        }
        .padding(14)
        .mabelPanel()
        .mabelShadow(Theme.Shadow.card)
        .padding(12)
    }

    private func detailLine(_ s: ActionSpec) -> String {
        var parts = [LibraryFilter.primary(s)?.title ?? s.category.title, s.lane.title,
                     TimelineCanvas.label(MotionClips.shared.clips[s.id] != nil
                                          ? MotionClips.shared.duration(s.id, params: s.defaults) : s.duration)]
        if s.holdable { parts.append("can be held") }
        if let loc = s.location { parts.append("@ \(loc.label)") }
        if let v = MotionClips.shared.clips[s.id]?.peakVelocity, v > 0 { parts.append(String(format: "peak %.1f rad/s", v)) }
        return parts.joined(separator: " · ")
    }

    // MARK: actions

    private func pick(_ spec: ActionSpec) {
        guard spec.available else { Haptics.warning(); detail = spec; return }
        Haptics.select()
        LibraryRecents.note(spec.id)
        add(spec)
        close()
    }

    private func pick(_ pose: SavedPose) {
        Haptics.select()
        addPose(pose)
        close()
    }
}

/// One action, as a card: emoji, title, length, lane.
struct ActionCard: View {
    let spec: ActionSpec
    var selected = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                Text(spec.emoji.isEmpty ? "•" : spec.emoji).font(.system(size: 24))
                Spacer(minLength: 0)
                if !spec.available {
                    Text("SOON").font(.system(size: 8.5, weight: .heavy)).tracking(0.5)
                        .foregroundStyle(Theme.Palette.warn)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(Theme.Palette.warn.opacity(0.16), in: Capsule())
                }
            }
            Text(spec.title).font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.Palette.textPrimary)
                .lineLimit(2, reservesSpace: true)
                .multilineTextAlignment(.leading)
            HStack(spacing: 5) {
                LaneBadge(lane: spec.lane)
                Spacer(minLength: 0)
                Text(TimelineCanvas.label(spec.duration))
                    .font(.system(size: 10.5, weight: .medium)).monospacedDigit()
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(selected ? Theme.Palette.accent.opacity(0.10) : Theme.Palette.surface,
                    in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                .strokeBorder(selected ? Theme.Palette.accent.opacity(0.5) : Theme.Palette.hairlineSoft, lineWidth: 1)
        }
        .opacity(spec.available ? 1 : 0.5)
        .contentShape(Rectangle())
        .hoverEffect()
    }
}

struct LaneBadge: View {
    let lane: ActionLane
    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: lane.icon).font(.system(size: 8, weight: .bold))
            Text(lane == .both ? "Whole body" : lane.shortTitle).font(.system(size: 9, weight: .bold))
        }
        .foregroundStyle(Theme.Palette.textSecondary)
        .padding(.horizontal, 5).padding(.vertical, 2)
        .background(Theme.Palette.hairlineSoft, in: Capsule())
    }
}

private struct RecentChip: View {
    let spec: ActionSpec
    var body: some View {
        HStack(spacing: 6) {
            Text(spec.emoji.isEmpty ? "•" : spec.emoji).font(.system(size: 15))
            Text(spec.title).font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
        }
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(Theme.Palette.surface, in: Capsule())
        .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }
}

/// The robot's saved poses as cards ("CUSTOM", saved time). Observes the skills
/// object so a pose saved on another device appears while the library is open.
private struct SavedPoseCards: View {
    @ObservedObject var skills: RobotSkills
    var query: String
    var add: (SavedPose) -> Void

    var body: some View {
        let list = skills.poses.filter { LibraryFilter.matches($0, query) }
        if skills.posesAvailability != .available {
            Text(skills.posesAvailability.note)
                .font(.system(size: 12)).foregroundStyle(Theme.Palette.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        } else if list.isEmpty {
            Text(query.isEmpty ? "No poses saved on the robot yet — make one with New pose." : "No saved pose matches.")
                .font(.system(size: 12)).foregroundStyle(Theme.Palette.textTertiary)
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: kCardMinWidth), spacing: 10)], spacing: 10) {
                ForEach(list) { pose in
                    Button { add(pose) } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Image(systemName: "figure.stand").font(.system(size: 20, weight: .semibold))
                                    .foregroundStyle(Theme.Palette.accent)
                                Spacer(minLength: 0)
                                Text("CUSTOM").font(.system(size: 8.5, weight: .heavy)).tracking(0.5)
                                    .foregroundStyle(Theme.Palette.accent)
                                    .padding(.horizontal, 5).padding(.vertical, 2)
                                    .background(Theme.Palette.accent.opacity(0.14), in: Capsule())
                            }
                            Text(pose.name).font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(Theme.Palette.textPrimary).lineLimit(2, reservesSpace: true)
                                .multilineTextAlignment(.leading)
                            Text(SavedPose.savedLabel(pose.updated))
                                .font(.system(size: 9.5)).foregroundStyle(Theme.Palette.textTertiary).lineLimit(1)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Theme.Palette.surface,
                                    in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                                .strokeBorder(Theme.Palette.accent.opacity(0.3), lineWidth: 1)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(PressableStyle())
                    .accessibilityIdentifier("pose.card.\(pose.id)")
                }
            }
        }
    }
}

// MARK: - the clip's own settings

/// What a selected clip does and how long it does it for — a compact panel opened
/// from the clip (custom header, tiles; no Form).
struct ClipSettingsSheet: View {
    @ObservedObject var doc: TimelineDocument
    @ObservedObject var catalog: ActionCatalog
    let blockID: UUID
    var playhead: TimeInterval
    @Environment(\.mabelDismiss) private var dismiss

    private var block: TimelineBlock? { doc.timeline.blocks.first { $0.id == blockID } }

    var body: some View {
        let spec = block.flatMap { catalog.spec($0.actionID) }
        MabelModal(title: spec.map { "\($0.emoji.isEmpty ? "" : $0.emoji + "  ")\($0.title)" } ?? "Clip",
                   subtitle: block.map { "Starts \(TimelineCanvas.timecode($0.start)) · \(String(format: "%.2f s", $0.duration))" },
                   shrinkToFit: true, onClose: dismiss) {
            if let b = block, let spec {
                VStack(alignment: .leading, spacing: 14) {
                    if let info = b.blend { blendSection(b, info) }
                    if spec.category == .language { saySection(b) }
                    if !spec.params.isEmpty {
                        MabelModalSection(title: "Parameters") {
                            ForEach(spec.params) { p in MabelModalRow { paramRow(b, p) } }
                            if NavMoveMapping.isPrimitive(b.actionID) { navTimingNote(b) }
                        }
                    }
                    MabelModalSection(title: "Timing") {
                        MabelModalRow {
                            VStack(spacing: 6) {
                                row("Start", TimelineCanvas.timecode(b.start))
                                row("Length", String(format: "%.2f s", b.duration))
                                if b.inPoint > 0.001 {
                                    row("Clip range", String(format: "%.1f – %.1f s", b.inPoint, b.outPoint))
                                }
                                if let clip = MotionClips.shared.clips[b.actionID], clip.peakVelocity > 0 {
                                    row("Peak joint speed", String(format: "%.2f rad/s", clip.peakVelocity))
                                }
                            }
                        }
                        if !spec.holdable {
                            Text("Fixed-length motion — it can be trimmed, never stretched.")
                                .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                        }
                    }
                    HStack(spacing: 8) {
                        actionButton("Split", "scissors",
                                     enabled: !b.isBlend && playhead > b.start + kMinBlockDuration
                                        && playhead < b.end - kMinBlockDuration) {
                            doc.split(b.id, at: playhead); dismiss()
                        }
                        actionButton("Duplicate", "plus.square.on.square", enabled: !b.isBlend) {
                            doc.duplicate(b.id); dismiss()
                        }
                        actionButton("Delete", "trash", destructive: true) {
                            doc.remove([b.id]); dismiss()
                        }
                    }
                }
            } else {
                Text("This clip was removed.").font(.system(size: 13))
                    .foregroundStyle(Theme.Palette.textSecondary).padding(.vertical, 20)
            }
        }
    }

    private func saySection(_ b: TimelineBlock) -> some View {
        MabelModalSection(title: "Says") {
            TextField("What the robot says", text: Binding(get: { b.say ?? "" }, set: { doc.setSay(b.id, $0) }),
                      axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .lineLimit(1...4)
                .padding(10)
                .background(Theme.Palette.surface,
                            in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
                .accessibilityIdentifier("clip.say")
        }
    }

    /// How long the move takes: from the robot's calibration when the gateway
    /// publishes one, else the studio's estimate.
    private func navTimingNote(_ b: TimelineBlock) -> some View {
        let amount = b.params["dist"] ?? b.params["angle"] ?? 0
        let text: String
        if let cal = NavCalibration.current, let t = cal.seconds(for: b.actionID, amount: amount),
           let v = cal.speedText(for: b.actionID, amount: amount) {
            text = String(format: "≈ %.1f s at %@ — calibrated on the robot", t, v)
        } else {
            let t = MotionClips.shared.nominalNavDuration(b.actionID, params: b.params)
            let v = b.actionID.contains("turn") ? String(format: "%.0f°/s", MotionMath.navYawRate * 0.5 * 180 / .pi)
                                                : String(format: "%.2f m/s", MotionMath.navSpeed * 0.5)
            text = String(format: "≈ %.1f s at %@ — estimated (no robot calibration)", t, v)
        }
        return Label(text, systemImage: "gauge.with.dots.needle.33percent")
            .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
            .accessibilityIdentifier("clip.navTiming")
    }

    private func paramRow(_ b: TimelineBlock, _ p: ActionParam) -> some View {
        let value = b.params[p.id] ?? p.def
        return VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(p.label).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.Palette.textPrimary)
                Spacer()
                Text(p.id == "angle" ? String(format: "%.0f°", value) : p.id == "dist" ? String(format: "%.2f m", value) : String(format: "%.2f", value))
                    .font(.system(size: 12, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(Theme.Palette.accent)
            }
            if NavMoveMapping.isPrimitive(b.actionID) {
                NavAmountPicker(param: p, value: value) { doc.setParam(b.id, p.id, $0) }
            }
            Slider(value: Binding(get: { value }, set: { doc.setParam(b.id, p.id, $0) }), in: p.min...p.max)
                .tint(Theme.Palette.accent)
                .accessibilityIdentifier("param.\(p.id)")
        }
    }

    @ViewBuilder
    private func blendSection(_ b: TimelineBlock, _ info: BlendInfo) -> some View {
        let floor = max(MotionMath.blendMinDur, info.minDuration)
        let peak = info.delta * MotionMath.easedPeakRate(b.duration)
        MabelModalSection(title: "Transition") {
            MabelModalRow {
                VStack(alignment: .leading, spacing: 6) {
                    row("Bridges", info.edge == .lead ? "into the next clip" : "out of the last clip")
                    if let j = info.joint { row("Widest gap", String(format: "%.2f rad · %@", info.delta, j)) }
                    HStack {
                        Text("Ramp").font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
                        Spacer()
                        Text(String(format: "%.2f s", b.duration)).font(.system(size: 12, weight: .semibold).monospacedDigit())
                    }
                    Slider(value: Binding(get: { b.duration }, set: { doc.setBlendDuration(b.id, $0) }),
                           in: floor...max(floor + 0.1, MotionMath.blendMaxDur))
                        .tint(Theme.Palette.violet)
                        .accessibilityIdentifier("blend.ramp")
                    Text(peak <= MotionMath.safeBlendVel + 1e-6
                         ? String(format: "%.2f rad/s — a gentle bridge.", peak)
                         : String(format: "%.2f rad/s — brisk. Lengthen it to soften, at the cost of reaching further back into the clip before it.", peak))
                        .font(.system(size: 11))
                        .foregroundStyle(peak <= MotionMath.maxJointVel ? Theme.Palette.textTertiary : Theme.Palette.danger)
                        .fixedSize(horizontal: false, vertical: true)
                    if info.userDuration != nil {
                        Button("Back to automatic") { doc.setBlendDuration(b.id, nil) }
                            .font(.system(size: 12, weight: .semibold)).buttonStyle(.plain)
                            .foregroundStyle(Theme.Palette.accent)
                    }
                }
            }
            Text("Inserted automatically to keep the commanded pose continuous. It cannot be split, and its trailing edge is anchored to the junction it closes.")
                .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func actionButton(_ title: String, _ icon: String, enabled: Bool = true, destructive: Bool = false,
                              action: @escaping () -> Void) -> some View {
        Button { Haptics.select(); action() } label: {
            Label(title, systemImage: icon)
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(destructive ? Theme.Palette.danger : Theme.Palette.textPrimary)
                .frame(maxWidth: .infinity, minHeight: 38)
                .background(destructive ? Theme.Palette.danger.opacity(0.10) : Theme.Palette.surface,
                            in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                        .strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1)
                }
                .opacity(enabled ? 1 : 0.4)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
        .disabled(!enabled)
        .accessibilityIdentifier("clip.\(title.lowercased())")
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            Text(v).font(.system(size: 12, weight: .semibold)).monospacedDigit().foregroundStyle(Theme.Palette.textPrimary)
        }
    }
}

// MARK: - annotations

/// Write the text label that sits on the timeline at this moment. Compact, and
/// keyboard-safe: the modal host respects the keyboard safe area.
struct NoteEditorSheet: View {
    @ObservedObject var doc: TimelineDocument
    /// `nil` = a NEW note, written only when you tap Add (creating it on the way
    /// in left an orphan chip behind every cancelled edit).
    let noteID: UUID?
    var at: TimeInterval = 0
    @Environment(\.mabelDismiss) private var dismiss
    @State private var text = ""
    @State private var seconds: Double = 2
    @FocusState private var focused: Bool

    private var note: TimelineNote? {
        guard let noteID else { return nil }
        return doc.timeline.notes.first { $0.id == noteID }
    }

    var body: some View {
        MabelModal(title: noteID == nil ? "New text label" : "Text label",
                   subtitle: "At \(TimelineCanvas.timecode(note?.start ?? at))",
                   scrolls: false, size: .compact, onClose: dismiss) {
            VStack(alignment: .leading, spacing: 12) {
                MabelTextField(placeholder: "What happens here", text: $text, focused: $focused, axis: .vertical,
                               onSubmit: commit)
                    .accessibilityIdentifier("note.text")
                MabelModalRow {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Length").font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
                            Spacer()
                            Text(String(format: "%.1f s", seconds))
                                .font(.system(size: 12, weight: .semibold)).monospacedDigit()
                        }
                        Slider(value: $seconds, in: 0.5...20, step: 0.5).tint(Theme.Palette.accent)
                    }
                }
                if let noteID {
                    Button(role: .destructive) {
                        doc.removeNote(noteID); dismiss()
                    } label: {
                        Label("Delete label", systemImage: "trash")
                            .font(.system(size: 12.5, weight: .semibold))
                            .foregroundStyle(Theme.Palette.danger)
                    }
                    .buttonStyle(.plain)
                }
                MabelModalFooter(confirmTitle: noteID == nil ? "Add label" : "Done",
                                 onCancel: dismiss, onConfirm: commit)
            }
            .padding(MabelOverlay.pad)
        }
        .onAppear {
            text = note?.text ?? ""
            seconds = note?.duration ?? 2
            if noteID == nil { DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { focused = true } }
        }
    }

    private func commit() {
        if let noteID {
            doc.setNoteText(noteID, text)
            doc.retimeNote(noteID, duration: seconds)
        } else {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let id = doc.addNote(trimmed.isEmpty ? "note" : trimmed, at: at, duration: seconds)
            doc.selectedNote = id
        }
        dismiss()
    }
}

// MARK: - keyframes

/// Add (or edit) a keyframe — the ◆ that holds a whole-body pose at a moment.
/// Was a silent one-tap capture plus a system context menu; now a compact panel
/// that says WHAT is being captured, with a label, and a way to delete.
struct KeyframeEditorSheet: View {
    @ObservedObject var doc: TimelineDocument
    /// nil = a new keyframe at `at`.
    let keyID: UUID?
    var at: TimeInterval = 0
    /// The robot's current commanded pose, when a robot is connected.
    var livePose: [String: Double]?
    @Environment(\.mabelDismiss) private var dismiss
    @State private var label = ""
    @State private var source = 0       // 0 = robot now, 1 = rest
    @FocusState private var focused: Bool

    private var key: PoseKey? { keyID.flatMap { id in doc.timeline.keys.first { $0.id == id } } }

    var body: some View {
        MabelModal(title: keyID == nil ? "New keyframe" : "Keyframe",
                   subtitle: "At \(TimelineCanvas.timecode(key?.time ?? at))",
                   scrolls: false, size: .compact, onClose: dismiss) {
            VStack(alignment: .leading, spacing: 12) {
                MabelTextField(placeholder: "Label (optional)", text: $label, focused: $focused, onSubmit: commit)
                    .accessibilityIdentifier("key.label")
                if let key {
                    MabelModalRow {
                        HStack {
                            Text("Holds").font(.system(size: 12)).foregroundStyle(Theme.Palette.textSecondary)
                            Spacer()
                            Text("\(key.pose.count) joints").font(.system(size: 12, weight: .semibold))
                        }
                    }
                    Button(role: .destructive) {
                        doc.removeKey(key.id); dismiss()
                    } label: {
                        Label("Delete keyframe", systemImage: "trash")
                            .font(.system(size: 12.5, weight: .semibold)).foregroundStyle(Theme.Palette.danger)
                    }
                    .buttonStyle(.plain)
                } else {
                    MabelModalSection(title: "Capture") {
                        sourceRow(0, "Robot's current pose",
                                  livePose.map { "\($0.count) joints, as commanded now" } ?? "Connect to a robot to capture its pose",
                                  icon: "dot.radiowaves.left.and.right", enabled: livePose != nil)
                        sourceRow(1, "Rest pose", "Arms hanging, lift mid-travel", icon: "figure.stand", enabled: true)
                    }
                }
                MabelModalFooter(confirmTitle: keyID == nil ? "Add keyframe" : "Done",
                                 onCancel: dismiss, onConfirm: commit)
            }
            .padding(MabelOverlay.pad)
        }
        .onAppear {
            label = key?.label ?? ""
            source = livePose == nil ? 1 : 0
        }
    }

    private func sourceRow(_ tag: Int, _ title: String, _ detail: String, icon: String, enabled: Bool) -> some View {
        let on = source == tag
        return Button { if enabled { Haptics.select(); source = tag } } label: {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(on ? Theme.Palette.accent : Theme.Palette.textSecondary).frame(width: 20)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
                    Text(detail).font(.system(size: 10.5)).foregroundStyle(Theme.Palette.textTertiary)
                }
                Spacer(minLength: 0)
                Image(systemName: on ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(on ? Theme.Palette.accent : Theme.Palette.textTertiary)
            }
            .padding(9)
            .background(on ? Theme.Palette.accent.opacity(0.10) : Theme.Palette.surface,
                        in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                    .strokeBorder(on ? Theme.Palette.accent.opacity(0.45) : Theme.Palette.hairlineSoft, lineWidth: 1)
            }
            .opacity(enabled ? 1 : 0.45)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }

    private func commit() {
        let l = label.trimmingCharacters(in: .whitespacesAndNewlines)
        if let key {
            doc.setKeyLabel(key.id, l)
        } else {
            let pose = (source == 0 ? livePose : nil) ?? RobotModel.homePose
            doc.selectedKey = doc.addKey(at: at, pose: pose, label: l.isEmpty ? (source == 0 ? "captured" : "rest") : l)
        }
        dismiss()
    }
}
