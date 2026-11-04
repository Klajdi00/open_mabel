//
//  TimelineCanvas.swift
//  MabelTeleop (visionOS)
//
//  The tracks — ruler, markers, lanes, rows, blocks, playhead. This is the part
//  that has to feel like a video editor, so it owns the pixels-per-second
//  mapping and every gesture, and nothing else has an opinion about either.
//
//  Two things here are phone-specific rather than ports of the studio:
//
//  · The LANE GUTTER is pinned outside the horizontal scroll and SHRINKS to its
//    icons once you scroll in. On a desktop the labels can afford to scroll away
//    with the content; on a 390 pt screen, losing track of which row is Nav and
//    which is Manip is the fastest way to drop the wrong clip on the base.
//  · ANNOTATIONS are a marker strip above the lanes rather than a fourth lane.
//    A lane costs 40 pt of a screen that has three of them already, and a note
//    is a label on a moment, not a resource the robot allocates.
//

import SwiftUI

/// Seconds → points. One knob, so the ruler, the blocks and the playhead can
/// never disagree about where 3.5 s is.
struct TimeScale: Equatable {
    var pxPerSecond: CGFloat = 46
    /// The studio's zoom range.
    static let minZoom: CGFloat = 8
    static let maxZoom: CGFloat = 400

    func x(_ t: TimeInterval) -> CGFloat { CGFloat(t) * pxPerSecond }
    func time(_ x: CGFloat) -> TimeInterval { TimeInterval(max(0, x) / pxPerSecond) }

    mutating func zoom(_ factor: CGFloat) {
        pxPerSecond = min(Self.maxZoom, max(Self.minZoom, pxPerSecond * factor))
    }

    /// Tick spacing that stays readable at any zoom: 0.5 s when zoomed in,
    /// a minute when zoomed way out.
    var tickInterval: TimeInterval {
        let candidates: [TimeInterval] = [0.5, 1, 2, 5, 10, 15, 30, 60, 120]
        return candidates.first { x($0) >= 46 } ?? 300
    }
}

/// What the tracks are drawn ON. An iPhone has an opaque canvas behind them; a
/// visionOS window is GLASS over whatever room the operator is standing in, and
/// 8 pt tick labels over a sunlit wall are simply not there. The strip gets its
/// own material so the timeline is legible in any room — the rest of the window
/// stays glass, as it should be.
@ViewBuilder
func timelineBacking() -> some View {
    #if os(visionOS)
    Rectangle().fill(.regularMaterial)
    #else
    Theme.Palette.canvas
    #endif
}

struct TimelineCanvas: View {
    @ObservedObject var doc: TimelineDocument
    @ObservedObject var catalog: ActionCatalog
    @Binding var scale: TimeScale
    @Binding var playhead: TimeInterval
    var isPlaying: Bool
    var isExecuting: Bool
    /// Tapped a block twice / long-pressed "Adjust" — the editor opens the
    /// parameter popover. There is no inspector panel to put it in by design.
    var onEditBlock: (UUID) -> Void = { _ in }
    var onEditNote: (UUID) -> Void = { _ in }
    /// Second tap on a selected keyframe — the editor opens its compact panel.
    var onEditKey: (UUID) -> Void = { _ in }
    /// The robot's report for blocks it is driving itself (calibrated `nav_move`).
    var navBadges: [UUID: NavMoveStatus] = [:]

    @State private var scrollX: CGFloat = 0
    // NOT `ScrollPosition(edge: .leading)`: an edge-pinned position is RE-APPLIED
    // whenever the scroll view re-renders, so a document change (an auto-blend
    // appearing, a clip moving) yanked the tracks back to 0 under the operator's
    // finger. Start wherever the scroll view naturally starts, and move it only
    // when we mean to, in `follow`.
    @State private var position = ScrollPosition()
    /// Live drag state, kept out of the document so a drag is one undo entry.
    @State private var dragging: (id: UUID, start: TimeInterval, row: Int)?
    @State private var trimming: (id: UUID, edge: TimelineDocument.Edge, anchor: TimeInterval)?
    @State private var noteDrag: (id: UUID, start: TimeInterval)?
    @State private var zoomAnchor: CGFloat?
    @State private var lastSnapAt: TimeInterval = -1

    private let rowHeight: CGFloat = 38
    private let laneGap: CGFloat = 5
    private let rulerHeight: CGFloat = 22
    private let markerHeight: CGFloat = 19
    /// Full gutter, and what it collapses to once the tracks scroll.
    private let gutterWide: CGFloat = 76
    private let gutterNarrow: CGFloat = 28

    private var gutterWidth: CGFloat { scrollX > 6 ? gutterNarrow : gutterWide }
    private var collapsed: Bool { scrollX > 6 }
    /// Room to drag a clip past the end of the material.
    private var contentDuration: TimeInterval { max(doc.timeline.duration + 6, 14) }
    private var contentWidth: CGFloat { scale.x(contentDuration) + 40 }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Theme.Palette.hairlineSoft)
            tracks
        }
        .background(timelineBacking())
        .onChange(of: playhead) { _, t in follow(t) }
    }

    // MARK: - ruler + markers (pinned; they scroll by offset, not by a scroll view)

    private var header: some View {
        HStack(spacing: 0) {
            // The gutter's slice of the ruler carries the timecode, so the time
            // is readable even when the playhead is off-screen.
            Text(Self.label(playhead))
                .font(.system(size: 9, weight: .heavy)).monospacedDigit()
                .foregroundStyle(isExecuting ? Theme.Palette.danger : Theme.Palette.accent)
                .frame(width: gutterWidth, height: rulerHeight + markerHeight)
                .background(timelineBacking())
            // A viewport, NOT a wide view that happens to be clipped.
            // `.clipped()` clips DRAWING and leaves the layout size alone, so a
            // ruler `contentWidth` wide inside this HStack made the whole screen
            // that wide and shoved every other control off the left edge. An
            // OVERLAY does not participate in its host's sizing, which is
            // exactly the property a scrolling viewport needs.
            Color.clear
                .frame(maxWidth: .infinity)
                .frame(height: rulerHeight + markerHeight)
                .contentShape(Rectangle())
                .gesture(scrubGesture)
                .overlay(alignment: .topLeading) {
                    // `alignment: .leading` is load-bearing: a VStack CENTRES
                    // its children in the width it is handed, so with the
                    // default alignment every tick offset was measured from the
                    // middle of `contentWidth` and the ruler drew off-screen
                    // while the tracks below drew from zero.
                    VStack(alignment: .leading, spacing: 0) {
                        // Ticks never take a touch — the whole ruler row is the
                        // scrub target, and a drag that landed on a tick label
                        // would otherwise die there.
                        ruler.allowsHitTesting(false)
                        markers
                    }
                    .frame(width: contentWidth, alignment: .topLeading)
                    .offset(x: -scrollX)
                }
                .clipped()
        }
        .animation(Theme.Motion.gentle, value: gutterWidth)
    }

    private var ruler: some View {
        let step = scale.tickInterval
        return ZStack(alignment: .topLeading) {
            ForEach(Array(stride(from: 0, through: contentDuration, by: step)), id: \.self) { t in
                HStack(spacing: 3) {
                    Rectangle().fill(Theme.Palette.hairline).frame(width: 1, height: 6)
                    Text(Self.label(t))
                        .font(.system(size: 8.5, weight: .medium)).monospacedDigit()
                        .foregroundStyle(Theme.Palette.textSecondary)
                }
                .offset(x: scale.x(t))
            }
            // The playhead's grab handle rides the ruler.
            Triangle()
                .fill(isExecuting ? Theme.Palette.danger : Theme.Palette.accent)
                .frame(width: 11, height: 8)
                .offset(x: scale.x(playhead) - 5.5, y: rulerHeight - 8)
        }
        .frame(width: contentWidth, height: rulerHeight, alignment: .topLeading)
    }

    /// Annotation markers — the studio's LANGUAGE lane, compressed into a strip.
    private var markers: some View {
        ZStack(alignment: .topLeading) {
            Rectangle().fill(Theme.Palette.surface.opacity(0.5))
                .frame(height: markerHeight)
                .allowsHitTesting(false)
            ForEach(doc.timeline.notes) { note in
                let selected = doc.selectedNote == note.id
                HStack(spacing: 3) {
                    Image(systemName: "text.bubble.fill").font(.system(size: 7))
                    Text(note.text).font(.system(size: 9, weight: .semibold)).lineLimit(1)
                }
                .foregroundStyle(selected ? Color.white : Theme.Palette.textSecondary)
                .padding(.horizontal, 5)
                .frame(width: max(34, scale.x(note.duration)), height: markerHeight - 3,
                       alignment: .leading)
                .background(selected ? Theme.Palette.accent : Theme.Palette.surfaceHi,
                            in: RoundedRectangle(cornerRadius: 4, style: .continuous))
                .offset(x: scale.x(note.start), y: 1.5)
                .contentShape(Rectangle())
                .onTapGesture {
                    doc.selectedNote = note.id
                    doc.selection.removeAll()
                    onEditNote(note.id)
                }
                .gesture(
                    DragGesture(minimumDistance: 4)
                        .onChanged { g in
                            if noteDrag?.id != note.id {
                                noteDrag = (note.id, note.start)
                                doc.beginInteraction()
                                doc.selectedNote = note.id
                            }
                            guard let d = noteDrag else { return }
                            let raw = d.start + scale.time(abs(g.translation.width))
                                * (g.translation.width < 0 ? -1 : 1)
                            doc.moveNote(note.id, to: doc.snap(raw, playhead: playhead).time)
                        }
                        .onEnded { _ in
                            if noteDrag != nil { doc.endInteraction(); noteDrag = nil }
                        }
                )
            }
        }
        .frame(width: contentWidth, height: markerHeight, alignment: .topLeading)
    }

    // MARK: - the tracks

    private var tracks: some View {
        ScrollView(.vertical, showsIndicators: false) {
            HStack(alignment: .top, spacing: 0) {
                gutter
                ScrollView(.horizontal, showsIndicators: false) {
                    ZStack(alignment: .topLeading) {
                        lanes
                        playheadLine
                    }
                    .frame(width: contentWidth, alignment: .topLeading)
                    .contentShape(Rectangle())
                    .gesture(zoomGesture)
                }
                .scrollPosition($position)
                .onScrollGeometryChange(for: CGFloat.self) { $0.contentOffset.x } action: { _, x in
                    scrollX = max(0, x)
                }
            }
        }
        .animation(Theme.Motion.gentle, value: gutterWidth)
    }

    private var gutter: some View {
        VStack(spacing: laneGap) {
            ForEach(ActionLane.tracks) { lane in
                laneHeader(lane)
            }
        }
        .padding(.top, 3)
        .frame(width: gutterWidth)
        .background(timelineBacking())
        // The gutter sits over the tracks, so a clip dragged under it is hidden
        // rather than drawn across the labels.
        .zIndex(2)
    }

    private func laneHeader(_ lane: ActionLane) -> some View {
        let rows = doc.timeline.rowCount(lane)
        return VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Image(systemName: lane.icon).font(.system(size: 10, weight: .bold))
                if !collapsed {
                    Text(lane.shortTitle.uppercased())
                        .font(.system(size: 8.5, weight: .heavy)).tracking(0.4)
                        .lineLimit(1).minimumScaleFactor(0.75)
                    Spacer(minLength: 0)
                }
            }
            .foregroundStyle(Theme.Palette.textSecondary)
            if !collapsed {
                // Stack / unstack rows: "multiple timelines over each other".
                HStack(spacing: 5) {
                    Button { Haptics.select(); doc.addRow(lane) } label: {
                        Image(systemName: "plus.rectangle.on.rectangle")
                            .font(.system(size: 9, weight: .semibold))
                    }
                    .accessibilityIdentifier("lane.addRow.\(lane.rawValue)")
                    if rows > 1 {
                        Button { Haptics.select(); doc.removeEmptyRow(lane) } label: {
                            Image(systemName: "minus.rectangle").font(.system(size: 9, weight: .semibold))
                        }
                        .accessibilityIdentifier("lane.removeRow.\(lane.rawValue)")
                        Text("\(rows)").font(.system(size: 8, weight: .bold)).monospacedDigit()
                    }
                    Spacer(minLength: 0)
                }
                .foregroundStyle(Theme.Palette.textTertiary)
                .buttonStyle(.plain)
            }
            Spacer(minLength: 0)
        }
        .padding(.leading, 5)
        .frame(width: gutterWidth,
               height: CGFloat(rows) * rowHeight + CGFloat(rows - 1) * 2,
               alignment: .topLeading)
        // Identified whether or not it is showing its label, so a test can
        // assert the gutter SURVIVES a scroll (and measure that it shrank).
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("lane.header.\(lane.rawValue)")
    }

    private var lanes: some View {
        VStack(alignment: .leading, spacing: laneGap) {
            ForEach(ActionLane.tracks) { lane in
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(0..<doc.timeline.rowCount(lane), id: \.self) { row in
                        rowView(lane, row)
                    }
                }
            }
        }
        .padding(.top, 3)
    }

    private func rowView(_ lane: ActionLane, _ row: Int) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Theme.Palette.surface.opacity(row == 0 ? 0.9 : 0.55))
                .frame(width: contentWidth - 8, height: rowHeight)
                .contentShape(Rectangle())
                .onTapGesture { location in
                    playhead = max(0, scale.time(location.x))
                    doc.selection.removeAll()
                    doc.selectedNote = nil
                }
            ForEach(doc.timeline.blocks(in: lane, row: row)) { block in
                blockView(block)
                    .offset(x: scale.x(block.start))
            }
            if lane == .manipulation && row == 0 {
                keyframes
                jumpMarkers
            }
            if lane == .navigation && row == 0 {
                gapMarkers
            }
        }
        .frame(width: contentWidth, height: rowHeight, alignment: .topLeading)
    }

    // MARK: - a block

    @ViewBuilder
    private func blockView(_ block: TimelineBlock) -> some View {
        let spec = catalog.spec(block.actionID)
        let selected = doc.selection.contains(block.id)
        let w = max(scale.x(block.duration), 20)
        let blend = block.isBlend
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(tint(block, spec).opacity(selected ? 0.98 : 0.76))
            if blend {
                // A guard the studio inserted must not look like a clip you
                // authored — you are not allowed to blade it.
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.5), style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
            }
            HStack(spacing: 3) {
                if w > 40, let e = spec?.emoji, !e.isEmpty { Text(e).font(.system(size: 11)) }
                if w > 26 {
                    Text(label(block, spec))
                        .font(.system(size: 10, weight: .semibold))
                        .lineLimit(1).truncationMode(.tail)
                        .foregroundStyle(.white)
                }
                Spacer(minLength: 0)
                if w > 92 {
                    Text(String(format: "%.1fs", block.duration))
                        .font(.system(size: 8.5, weight: .medium)).monospacedDigit()
                        .foregroundStyle(.white.opacity(0.75))
                }
            }
            .padding(.horizontal, 6)

            if let st = navBadges[block.id] { navBadge(st, width: w) }

            // Trim handles — the two edges of a clip, same as any editor. A
            // blend only offers its leading edge (the trailing one is anchored
            // to the jump it guards).
            HStack {
                trimHandle(block, .leading)
                Spacer(minLength: 0)
                if !blend { trimHandle(block, .trailing) }
            }
        }
        .frame(width: w, height: rowHeight - 8)
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(selected ? Color.white : .clear, lineWidth: 1.5)
        }
        .padding(.top, 4)
        .contentShape(Rectangle())
        // A clip is drawn, not built from controls, so it has no accessibility
        // element of its own until we make one — and without that neither
        // VoiceOver nor a UI test can find a block at all.
        .accessibilityElement(children: .ignore)
        .accessibilityAddTraits(.isButton)
        .accessibilityLabel(label(block, spec))
        .accessibilityValue(String(format: "%.1f seconds at %.1f", block.duration, block.start))
        .accessibilityIdentifier("block.\(block.actionID)")
        .onTapGesture {
            Haptics.select()
            doc.selectedNote = nil
            if doc.selection == [block.id] { onEditBlock(block.id) }   // second tap = adjust
            else { doc.selection = [block.id] }
        }
        .gesture(blockDrag(block))
        .contextMenu { blockMenu(block) }
    }

    /// A calibrated move's live state, pinned to the block's top-right corner.
    @ViewBuilder
    private func navBadge(_ st: NavMoveStatus, width: CGFloat) -> some View {
        let color: Color = st.succeeded ? Theme.Palette.live
            : st.isTerminal ? Theme.Palette.warn : Color.white
        HStack(spacing: 2) {
            Image(systemName: st.succeeded ? "checkmark" : st.isTerminal ? "exclamationmark.triangle.fill"
                                                                        : "dot.radiowaves.right")
                .font(.system(size: 7, weight: .heavy))
            if width > 56 {
                Text(st.shortBadge).font(.system(size: 7.5, weight: .bold)).lineLimit(1)
            }
        }
        .foregroundStyle(color)
        .padding(.horizontal, 4).padding(.vertical, 1.5)
        .background(.black.opacity(0.45), in: Capsule())
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        .padding(.trailing, 3)
        .offset(y: -8)              // rides the block's top edge, clear of its label
        .allowsHitTesting(false)
        .accessibilityIdentifier("navbadge.\(st.state.rawValue)")
    }

    private func label(_ block: TimelineBlock, _ spec: ActionSpec?) -> String {
        if let say = block.say, !say.isEmpty { return say }
        if block.isBlend { return "blend" }
        guard let spec else { return block.actionID }
        // The studio's derived language label: a nav primitive says how far.
        if let d = block.params["dist"] { return String(format: "%@ %.2g m", spec.title, d) }
        if let a = block.params["angle"] { return String(format: "%@ %.0f°", spec.title, a) }
        if spec.isGoal {
            return String(format: "go to (%.1f, %.1f)", block.params["x"] ?? 0, block.params["y"] ?? 0)
        }
        return spec.title
    }

    private func blockDrag(_ block: TimelineBlock) -> some Gesture {
        DragGesture(minimumDistance: 5)
            .onChanged { g in
                guard trimming == nil else { return }
                if dragging?.id != block.id {
                    dragging = (block.id, block.start, block.row)
                    doc.beginInteraction()
                    doc.selection = [block.id]
                }
                guard let d = dragging else { return }
                let raw = d.start + scale.time(abs(g.translation.width))
                    * (g.translation.width < 0 ? -1 : 1)
                let snapped = doc.snap(raw, excluding: block.id, playhead: playhead)
                if snapped.magnetic, abs(snapped.time - lastSnapAt) > 1e-6 {
                    lastSnapAt = snapped.time
                    Haptics.select()
                }
                // A vertical drag of one row height moves the clip up a row —
                // how you layer one motion over another.
                let rowShift = Int((g.translation.height / rowHeight).rounded())
                doc.move(block.id, to: snapped.time, row: max(0, d.row + rowShift))
            }
            .onEnded { _ in
                guard let d = dragging else { return }
                dragging = nil
                lastSnapAt = -1
                doc.endInteraction()
                doc.settle(d.id)
            }
    }

    private func trimHandle(_ block: TimelineBlock, _ edge: TimelineDocument.Edge) -> some View {
        Rectangle()
            .fill(Color.white.opacity(0.001))          // invisible, still hittable
            .frame(width: 15)
            .overlay {
                Capsule().fill(.white.opacity(0.6)).frame(width: 2.5, height: 13)
            }
            .gesture(
                DragGesture(minimumDistance: 2)
                    .onChanged { g in
                        if trimming?.id != block.id {
                            trimming = (block.id, edge,
                                        edge == .leading ? block.start : block.end)
                            doc.beginInteraction()
                            doc.selection = [block.id]
                        }
                        guard let t = trimming else { return }
                        let raw = t.anchor + scale.time(abs(g.translation.width))
                            * (g.translation.width < 0 ? -1 : 1)
                        doc.trim(block.id, edge: edge,
                                 to: doc.snap(raw, excluding: block.id, playhead: playhead).time)
                    }
                    .onEnded { _ in
                        guard trimming != nil else { return }
                        trimming = nil
                        doc.endInteraction()
                    }
            )
    }

    @ViewBuilder
    private func blockMenu(_ block: TimelineBlock) -> some View {
        if block.isBlend {
            Text("Safety blend — bridges a \(String(format: "%.2f", block.blend?.delta ?? 0)) rad jump")
            // A bridge's RAMP is the one thing about it the operator decides, so
            // the menu has to offer it. It used to offer only "turn the guard
            // off", which is a strange only-choice: the answer to "this
            // transition is too abrupt" is a longer ramp, not no guard.
            Button { onEditBlock(block.id) } label: {
                Label("Adjust ramp…", systemImage: "slider.horizontal.3")
            }
            Button { doc.autoBlend = false } label: {
                Label("Turn auto-blend off", systemImage: "bolt.slash")
            }
        } else {
            Button { onEditBlock(block.id) } label: {
                Label("Adjust…", systemImage: "slider.horizontal.3")
            }
            Button { Haptics.select(); doc.split(block.id, at: playhead) } label: {
                Label("Split at Playhead", systemImage: "scissors")
            }
            Button { Haptics.select(); doc.duplicate(block.id) } label: {
                Label("Duplicate", systemImage: "plus.square.on.square")
            }
            Button { playhead = block.start } label: {
                Label("Playhead to Start", systemImage: "arrow.left.to.line")
            }
            Divider()
            Button(role: .destructive) { Haptics.select(); doc.remove([block.id]) } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private func tint(_ block: TimelineBlock, _ spec: ActionSpec?) -> Color {
        if block.isBlend { return Theme.Palette.violet }
        switch spec?.category {
        case .nav:        return Theme.Palette.clay
        case .animation:  return Theme.Palette.accent
        case .policy:     return Theme.Palette.dataHues[4]
        case .pose:       return Theme.Palette.accentDeep
        case .language:   return Theme.Palette.live
        case .transition: return Theme.Palette.violet
        default:          return Theme.Palette.textTertiary
        }
    }

    // MARK: - keyframes, jump + gap markers

    private var keyframes: some View {
        ForEach(doc.timeline.keys) { key in
            Image(systemName: "diamond.fill")
                .font(.system(size: 9))
                .foregroundStyle(doc.selectedKey == key.id ? Color.white : Theme.Palette.warn)
                .frame(width: 18, height: rowHeight)
                .contentShape(Rectangle())
                .offset(x: scale.x(key.time) - 9)
                .onTapGesture {
                    Haptics.select()
                    if doc.selectedKey == key.id { onEditKey(key.id) }    // second tap = edit
                    else { doc.selectedKey = key.id; doc.selection.removeAll() }
                }
                .accessibilityIdentifier("keyframe.\(key.label)")
                .contextMenu {
                    Text(key.label.isEmpty ? "\(key.pose.count) joints" : key.label)
                    Button(role: .destructive) { doc.removeKey(key.id) } label: {
                        Label("Delete keyframe", systemImage: "trash")
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 4)
                        .onChanged { g in
                            let raw = key.time + scale.time(abs(g.translation.width))
                                * (g.translation.width < 0 ? -1 : 1)
                            doc.moveKey(key.id, to: doc.snap(raw, playhead: playhead).time)
                        }
                )
        }
    }

    /// Unbridged pose jumps (auto-blend off) — tap to insert the blend.
    private var jumpMarkers: some View {
        ForEach(doc.poseJumps) { jump in
            Button {
                Haptics.select()
                doc.insertBlend(for: jump)
            } label: {
                Image(systemName: "bolt.trianglebadge.exclamationmark.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.Palette.danger)
                    .frame(width: 22, height: rowHeight)
            }
            .buttonStyle(.plain)
            .offset(x: scale.x(jump.time) - 11)
            .accessibilityIdentifier("jump.marker")
        }
    }

    /// Location gaps — tap to insert the missing move.
    private var gapMarkers: some View {
        ForEach(doc.gaps) { gap in
            Button {
                Haptics.select()
                doc.fixGap(gap)
            } label: {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.Palette.warn)
                    .frame(width: 22, height: rowHeight)
            }
            .buttonStyle(.plain)
            .offset(x: scale.x(gap.time) - 11)
            .accessibilityIdentifier("gap.marker")
        }
    }

    // MARK: - playhead

    private var playheadLine: some View {
        Rectangle()
            .fill(isExecuting ? Theme.Palette.danger : Theme.Palette.accent)
            .frame(width: 1.5)
            .frame(maxHeight: .infinity)
            .offset(x: scale.x(playhead))
            .allowsHitTesting(false)
    }

    // MARK: - gestures

    /// Drag anywhere on the ruler to scrub — the one gesture every editor has.
    private var scrubGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { g in
                playhead = max(0, scale.time(g.location.x + scrollX))
            }
    }

    /// Pinch to zoom, about the playhead so the thing you are working on stays
    /// put. A timeline you cannot zoom is a list.
    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { g in
                let anchor = zoomAnchor ?? scale.pxPerSecond
                if zoomAnchor == nil { zoomAnchor = scale.pxPerSecond }
                scale.pxPerSecond = min(TimeScale.maxZoom,
                                        max(TimeScale.minZoom, anchor * g.magnification))
            }
            .onEnded { _ in zoomAnchor = nil }
    }

    /// Keep the playhead on screen during playback, the way a scrolling editor
    /// follows its transport.
    private func follow(_ t: TimeInterval) {
        guard isPlaying else { return }
        let x = scale.x(t)
        // Only chase once it is past two thirds of the viewport, so a short
        // preview does not jitter the whole track.
        if x - scrollX > 240 || x < scrollX {
            position.scrollTo(x: max(0, x - 120))
        }
    }

    // MARK: - formatting

    static func label(_ t: TimeInterval) -> String {
        if t >= 60 { return String(format: "%d:%02d", Int(t) / 60, Int(t) % 60) }
        return t.truncatingRemainder(dividingBy: 1) == 0
            ? "\(Int(t))s" : String(format: "%.1fs", t)
    }

    /// mm:ss.t — the transport's readout, where a tenth matters.
    static func timecode(_ t: TimeInterval) -> String {
        String(format: "%d:%02d.%d", Int(t) / 60, Int(t) % 60, Int((t * 10).truncatingRemainder(dividingBy: 10)))
    }
}

/// The playhead's grab handle.
private struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.midX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        p.closeSubpath()
        return p
    }
}
