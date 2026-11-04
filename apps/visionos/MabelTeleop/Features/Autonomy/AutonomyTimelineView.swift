//
//  AutonomyTimelineView.swift
//  MabelTeleop (visionOS)
//
//  The Autonomy tab: a choreography editor shaped like a mobile video editor.
//
//    ┌──────────────────────────────────────┐
//    │ name · undo/redo · auto-blend · EXEC │  32 pt
//    ├──────────────────────────────────────┤
//    │                                      │
//    │   3-D VIEWER — the robot, posed by   │  the rest
//    │   the timeline at the playhead       │
//    │   [◀◀ ▶ ▶▶]  0:03.4 / 0:12.0         │
//    ├──────────────────────────────────────┤
//    │ + · ✂ · ⧉ · T · ◆ · 🗑 · − ⛶ + · ⚡  │  34 pt
//    ├──────────────────────────────────────┤
//    │ NAV  │ ███  ░░  ████                 │
//    │ MANIP│   ███████                     │  draggable height
//    │ VOICE│ ▬▬                            │
//    └──────────────────────────────────────┘
//
//  The viewer is the point. Everything else is a thin strip around it, because
//  the question a choreography editor answers is "what will the robot DO", and
//  a list of coloured rectangles only answers it if you can watch the robot do
//  it. Preview plays into the viewer and touches nothing; Execute plays the same
//  walk onto the wire.
//
//  There is no inspector panel: a clip's settings are a popover you open from
//  the clip, and the library is a button that opens a sheet. Both used to be
//  docked, and between them they took half the screen to show six of the
//  forty-four actions and the parameters of a clip nobody had selected.
//

import SwiftUI

// MARK: - platform sizing

// The two apps run the SAME editor; only the glass it is drawn on differs in
// size. Keeping these as compile-time constants keeps this file byte-identical
// between the iPhone and the Vision Pro app, so a fix diffs to nothing.
#if os(visionOS)
private let kTimelineHeight: CGFloat = 252      // a window, not a 6" screen
private let kDefaultZoom: CGFloat = 60          // pt per second
private let kFitWidth: CGFloat = 760            // what "zoom to fit" fits into
private let kTimelineMaxHeight: CGFloat = 620
#else
private let kTimelineHeight: CGFloat = 196
private let kDefaultZoom: CGFloat = 46
private let kFitWidth: CGFloat = 320
private let kTimelineMaxHeight: CGFloat = 440
#endif

struct AutonomyTimelineView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.isLandscape) private var isLandscape
    @Environment(\.bottomBarInset) private var bottomBarInset

    @StateObject private var catalog = ActionCatalog()
    @StateObject private var doc: TimelineDocument
    @StateObject private var player: TimelinePlayer
    @StateObject private var store = TimelineStore()
    /// The robot's measured wheel odometry in the viewer's world frame (TimelineWorldFrame.swift).
    @State private var odomTrack = LiveOdomTrack()

    @State private var scale = TimeScale(pxPerSecond: kDefaultZoom)
    @State private var playhead: TimeInterval = 0
    @State private var timelineHeight: CGFloat = kTimelineHeight
    @State private var dragStartHeight: CGFloat?
    /// ONE sheet, routed. Four separate `.sheet` modifiers on one view is a
    /// trap: SwiftUI keeps only the last one alive, so the note editor and the
    /// clip settings silently refused to present while the library (first in the
    /// stack) worked — which reads as "the button does nothing".
    @State private var sheet: SheetRoute?
    @State private var showLibrary = false
    @State private var docMenu = false          // project-switcher dropdown
    @State private var viewerRecenter = 0

    init() {
        let cat = ActionCatalog()
        let document = TimelineDocument(catalog: cat)
        _catalog = StateObject(wrappedValue: cat)
        _doc = StateObject(wrappedValue: document)
        _player = StateObject(wrappedValue: TimelinePlayer(doc: document, catalog: cat))
    }

    private var selectedBlock: TimelineBlock? {
        guard let id = doc.selection.first else { return nil }
        return doc.timeline.blocks.first { $0.id == id }
    }

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider().overlay(Theme.Palette.hairlineSoft)
            viewer
            grabber
            toolbar
            Divider().overlay(Theme.Palette.hairlineSoft)
            TimelineCanvas(doc: doc, catalog: catalog, scale: $scale, playhead: $playhead,
                           isPlaying: player.isPlaying, isExecuting: player.isExecuting,
                           onEditBlock: { sheet = .clip($0) },
                           onEditNote: { sheet = .note($0) },
                           onEditKey: { sheet = .keyframe($0) },
                           navBadges: player.navBadges)
                .frame(height: timelineHeight)
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("timeline.canvas")
        }
        .background(Theme.Palette.canvas)
        .safeAreaInset(edge: .bottom) { Color.clear.frame(height: bottomBarInset) }
        .onReceive(player.$playhead) { if player.isPlaying { playhead = $0 } }
        .onChange(of: playhead) { _, t in player.scrub(to: t) }
        // a new link may be a restarted gateway whose odometry began again at zero
        .onChange(of: controller.connection.isLive) { _, _ in odomTrack.reset() }
        .onChange(of: doc.timeline) { _, tl in
            store.persistOpen(tl)
            player.refreshBasePath()
        }
        // Editing floats in a MabelModal, not a system pull-up sheet. The sheet
        // brought its own grabber, detents and navigation bar — none of which match
        // anything else here — and its medium detent clipped the controls it was
        // opened for on a short screen.
        .mabelModal(item: $sheet) { route in
            // Every pop-up is the shared MabelModal panel now (compact editors and
            // dialogs build their own; see ActionLibrarySheet.swift / MabelOverlays).
            switch route {
            case .library:
                Color.clear.onAppear { sheet = nil; showLibrary = true }
            case .clip(let id):
                ClipSettingsSheet(doc: doc, catalog: catalog, blockID: id, playhead: playhead)
            case .note(let id):
                NoteEditorSheet(doc: doc, noteID: id)
            case .newNote(let when):
                NoteEditorSheet(doc: doc, noteID: nil, at: when)
            case .keyframe(let id):
                KeyframeEditorSheet(doc: doc, keyID: id, at: playhead,
                                    livePose: controller.connection.isLive ? controller.jointTargets : nil)
            case .documents:
                TimelineLibrarySheet(store: store, doc: doc) { tl in sheet = .deleteTimeline(tl.id, tl.name) }
            case .deleteTimeline(let id, let name):
                MabelConfirmDialog(icon: "trash", title: "Delete “\(name)”?",
                                   message: "The saved timeline is removed from this device. The open document is not affected.",
                                   confirmTitle: "Delete", destructive: true) {
                    if let tl = store.saved.first(where: { $0.id == id }) { store.delete(tl) }
                }
            case .rename:
                MabelPromptDialog(title: "Rename timeline", placeholder: "Timeline name",
                                  initial: doc.timeline.name, confirmTitle: "Rename") { doc.rename($0) }
            case .execute:
                MabelConfirmDialog(icon: "bolt.fill",
                                   tint: controller.estopEngaged ? Theme.Palette.warn : Theme.Palette.accent,
                                   title: "Run this timeline on the robot?",
                                   message: controller.estopEngaged
                                       ? "The E-stop is engaged — release it first, or nothing will move."
                                       : "The robot will move — the base drives and the arms swing. Clear the space around it first.",
                                   confirmTitle: "Execute \(TimelineCanvas.label(doc.timeline.duration))",
                                   destructive: true) {
                    player.execute(on: controller, from: 0)
                }
            case .newPose:
                PoseCaptureSheet(skills: controller.skills)
            }
        }
        // The library is the one BIG pop-up: a browser of every clip the studio has.
        .sheet(isPresented: $showLibrary) {
            ActionLibrarySheet(catalog: catalog, playhead: playhead,
                               add: { spec in doc.selection = Set(doc.add(spec, at: playhead)) },
                               addWith: { spec, params in
                                   let ids = doc.add(spec, at: playhead)
                                   for id in ids { for (k, v) in params { doc.setParam(id, k, v) } }   // re-lengths the block
                                   doc.selection = Set(ids)
                               },
                               skills: controller.skills,
                               addPose: { pose in
                                   doc.selectedKey = doc.addKey(at: playhead, pose: pose.joints, label: pose.name)
                               },
                               newPose: {
                                   showLibrary = false
                                   DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { sheet = .newPose }
                               })
                .timelineLibraryPresentation()
        }
        .task {
            #if DEBUG
            // Screenshot / UI harness. Launch argv is dropped on prewarmed
            // launches, so the defaults channel is the reliable one — same
            // pattern as `debugLayers` in the Robot space.
            //   defaults write <bundle> debugTimeline -string demo|empty
            switch UserDefaults.standard.string(forKey: "debugTimeline") {
            case "demo":  doc.replace(with: Self.demoTimeline(catalog))
            case "empty": doc.replace(with: Timeline(name: "Untitled"))
            default:      doc.replace(with: store.restoreOpen())
            }
            #else
            doc.replace(with: store.restoreOpen())
            #endif
            #if DEBUG
            openDebugPopup()
            #endif
            player.refreshBasePath()
            player.scrub(to: 0)
            #if DEBUG
            // Screenshot harness: park the preview at a time (`debugTimelineScrub -float 4`).
            let parkAt = UserDefaults.standard.double(forKey: "debugTimelineScrub")
            if parkAt > 0 { playhead = parkAt }
            #endif
            // Availability is a property of the ROBOT (a policy becomes real
            // when its checkpoint deploys; `move to goal` needs the ROS stack
            // up), so refresh from whichever host is serving this session. No
            // robot ⇒ the bundled registry stands and the editor still works.
            if let base = controller.videoBase { await catalog.refresh(base: base) }
        }
        .onDisappear { player.stop() }
    }

    // MARK: - header

    private var headerBar: some View {
        HStack(spacing: 8) {
            // The project switcher. Was a system `Menu`, which draws UIKit's context
            // menu — its own radius, material and type — so it looked nothing like the
            // status dropdown sitting directly above it. Same MabelDropdown as
            // everything else now.
            Button { docMenu = true } label: {
                HStack(spacing: 4) {
                    Text(doc.timeline.name)
                        .font(.system(size: 13, weight: .bold)).lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 8, weight: .bold))
                        .rotationEffect(.degrees(docMenu ? 180 : 0))
                }
                .foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(Theme.Palette.surface.opacity(docMenu ? 0.95 : 0.75), in: Capsule())
                .overlay {
                    Capsule().strokeBorder(docMenu ? Theme.Palette.accent.opacity(0.5)
                                                   : Theme.Palette.hairlineSoft, lineWidth: 1)
                }
            }
            .buttonStyle(PressableStyle())
            .animation(Theme.Motion.press, value: docMenu)
            .accessibilityIdentifier("timeline.menu")
            .mabelDropdown(isPresented: $docMenu) {
                MabelDropdown(title: doc.timeline.name) {
                    MabelDropdownRow(icon: "folder", title: "Timelines…",
                                     subtitle: "Open or switch project") {
                        docMenu = false; sheet = .documents
                    }
                    MabelDropdownRow(icon: "pencil", title: "Rename") {
                        docMenu = false; sheet = .rename
                    }
                    MabelDropdownRow(icon: "square.and.arrow.down", title: "Save to library") {
                        docMenu = false; store.save(doc.timeline, as: doc.timeline.name)
                    }
                    MabelDropdownDivider()
                    MabelDropdownRow(icon: "arrow.left.to.line", title: "Close gaps") {
                        docMenu = false; doc.compact()
                    }
                    MabelDropdownRow(icon: "arrow.left.and.right", title: "Zoom to fit") {
                        docMenu = false
                        scale = TimeScale(pxPerSecond: kDefaultZoom); zoomToFit()
                    }
                    MabelDropdownDivider()
                    MabelDropdownRow(icon: "trash", title: "Clear timeline",
                                     isDestructive: true) {
                        docMenu = false; doc.clear(); playhead = 0
                    }
                }
            }

            Spacer(minLength: 0)

            iconButton("arrow.uturn.backward", enabled: doc.canUndo) { doc.undo() }
            iconButton("arrow.uturn.forward", enabled: doc.canRedo) { doc.redo() }

            // Auto-blend: the guard that keeps a cut from becoming a step command.
            Button {
                Haptics.select()
                doc.autoBlend.toggle()
            } label: {
                Image(systemName: doc.autoBlend ? "wand.and.sparkles" : "wand.and.sparkles.inverse")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(doc.autoBlend ? Color.white : Theme.Palette.textSecondary)
                    .frame(width: 28, height: 24)
                    .background(doc.autoBlend ? Theme.Palette.violet : Theme.Palette.surface,
                                in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("timeline.autoblend")

            Button {
                if player.isExecuting { player.stop() } else { sheet = .execute }
            } label: {
                Label(player.isExecuting ? "Stop" : "Execute",
                      systemImage: player.isExecuting ? "stop.fill" : "bolt.fill")
                    .font(.system(size: 12, weight: .heavy))
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .tint(player.isExecuting ? Theme.Palette.danger : Theme.Palette.accent)
            .disabled(doc.timeline.blocks.isEmpty && !player.isExecuting)
            .accessibilityIdentifier("timeline.execute")
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
    }

    private func iconButton(_ icon: String, enabled: Bool = true,
                            identifier: String? = nil,
                            action: @escaping () -> Void) -> some View {
        Button {
            Haptics.select()
            action()
        } label: {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .semibold))
                .frame(width: 28, height: 24)
                .foregroundStyle(enabled ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityIdentifier(identifier ?? "tool.\(icon)")
    }

    // MARK: - the viewer

    private var viewer: some View {
        ZStack {
            // A soft studio backdrop rather than flat black: the scene's floor
            // meets the background at a hard horizon, and a gradient reads as a
            // cyclorama instead of as a rendering artefact.
            LinearGradient(colors: [Color(hex: 0x17140F), Theme.Palette.viewportInk],
                           startPoint: .top, endPoint: .bottom)
            // Connected and not previewing → the viewer IS the robot: live joints at its wheel-odometry
            // pose (a hand push moves it too), leaving a trail; the plan is drawn from where it stands.
            RealRobotView(pose: showsLiveRobot ? controller.livePose : player.previewPose, gravityComp: false,
                          recenterToken: viewerRecenter, basePlanar: true,
                          plannedPath: player.previewPath,
                          odomTrack: odomTrack, poseIsOdom: showsLiveRobot,
                          anchorPlanToLive: player.mode == .idle)
                .rebuildsOnRobotRigChange()   // new robot / swapped hands → rebuild the scene
                .ignoresSafeArea(edges: .horizontal)

            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 5) {
                        statusChip
                        if showsLiveRobot { LiveOdomChip(track: odomTrack) { odomTrack.rezero() } }
                        baseChip
                        if let nav = player.navLine { navChip(nav) }
                    }
                    Spacer(minLength: 0)
                    VStack(alignment: .trailing, spacing: 5) {
                        Button {
                            Haptics.select(); viewerRecenter += 1
                        } label: {
                            Image(systemName: "scope").font(.system(size: 12, weight: .semibold))
                                .frame(width: 28, height: 28)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .buttonStyle(.plain)
                        if player.isPlaying, player.progress.m > 0 {
                            Text("\(player.progress.n)/\(player.progress.m)")
                                .font(.system(size: 9, weight: .heavy)).monospacedDigit()
                                .padding(.horizontal, 6).padding(.vertical, 3)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                    }
                }
                Spacer(minLength: 0)
                transport
            }
            .padding(10)
            .foregroundStyle(Theme.Palette.onViewport)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("timeline.viewer")
    }

    /// The live robot, not the plan, is what the viewer shows: connected, and not rehearsing.
    private var showsLiveRobot: Bool { controller.connection.isLive && player.mode != .preview }

    private var statusChip: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(player.isExecuting ? Theme.Palette.danger
                      : player.isPlaying ? Theme.Palette.accent : Theme.Palette.textTertiary)
                .frame(width: 6, height: 6)
            Text(player.isExecuting ? "LIVE — ROBOT MOVING"
                 : player.isPlaying ? "PREVIEW" : (player.status ?? "READY"))
                .font(.system(size: 9, weight: .heavy)).tracking(0.5)
                .lineLimit(1)
        }
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(.ultraThinMaterial, in: Capsule())
        .accessibilityIdentifier("timeline.status")
    }

    /// A calibrated nav move's live state (gateway `nav_move`): running → done with the
    /// odometry's measured error, or why it stopped.
    /// How far the base has moved and turned at the playhead, in the frame the timeline starts in —
    /// the number that goes with the grid and the path in the viewer.
    private var baseChip: some View {
        let p = player.basePose(at: playhead)
        return Text(TimelineWorldFrame.readout(p))
            .font(.system(size: 11, weight: .semibold, design: .monospaced))
            .foregroundStyle(.white.opacity(0.92))
            .padding(.horizontal, 9).padding(.vertical, 5)
            .background(.black.opacity(0.45), in: Capsule())
            .accessibilityIdentifier("timeline.baseReadout")
    }

    private func navChip(_ st: NavMoveStatus) -> some View {
        HStack(spacing: 5) {
            Image(systemName: st.succeeded ? "checkmark.circle.fill"
                  : st.isTerminal ? "exclamationmark.triangle.fill" : "location.north.line.fill")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(st.succeeded ? Theme.Palette.live : st.isTerminal ? Theme.Palette.warn : Theme.Palette.onViewport)
            Text(st.sentence).font(.system(size: 9, weight: .semibold)).lineLimit(1)
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(.ultraThinMaterial, in: Capsule())
        .accessibilityIdentifier("timeline.navStatus")
    }

    /// The floating transport, over the viewer where a video editor puts it.
    private var transport: some View {
        HStack(spacing: 14) {
            Button { playhead = 0; player.scrub(to: 0) } label: {
                Image(systemName: "backward.end.fill")
            }
            .accessibilityIdentifier("transport.start")

            Button {
                Haptics.rigid()
                if player.isPlaying { player.stop() } else { player.preview(from: playhead) }
            } label: {
                Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 17, weight: .bold))
                    .frame(width: 34, height: 34)
                    .background(Theme.Palette.accent, in: Circle())
                    .foregroundStyle(.white)
            }
            .accessibilityIdentifier("transport.preview")

            Button {
                playhead = doc.timeline.duration
                player.scrub(to: playhead)
            } label: {
                Image(systemName: "forward.end.fill")
            }
            .accessibilityIdentifier("transport.end")

            Text(TimelineCanvas.timecode(playhead) + " / "
                 + TimelineCanvas.timecode(doc.timeline.duration))
                .font(.system(size: 11, weight: .semibold)).monospacedDigit()
                .accessibilityIdentifier("transport.timecode")
        }
        .font(.system(size: 13, weight: .semibold))
        .buttonStyle(.plain)
        .padding(.horizontal, 14).padding(.vertical, 7)
        .background(.ultraThinMaterial, in: Capsule())
    }

    // MARK: - the grabber

    /// Drag to trade viewer height for track height — the split every editor has.
    private var grabber: some View {
        ZStack {
            Color.clear.frame(height: 14).contentShape(Rectangle())
            Capsule().fill(Theme.Palette.hairline).frame(width: 38, height: 4)
        }
        .gesture(
            DragGesture(minimumDistance: 2)
                .onChanged { g in
                    let base = dragStartHeight ?? timelineHeight
                    if dragStartHeight == nil { dragStartHeight = timelineHeight }
                    timelineHeight = min(kTimelineMaxHeight, max(108, base - g.translation.height))
                }
                .onEnded { _ in dragStartHeight = nil }
        )
        .accessibilityIdentifier("timeline.grabber")
    }

    // MARK: - toolbar

    private var toolbar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 5) {
                Button {
                    Haptics.select(); showLibrary = true
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "plus").font(.system(size: 11, weight: .heavy))
                        Text("Library").font(.system(size: 11, weight: .bold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 9).padding(.vertical, 6)
                    .background(Theme.Palette.accent, in: Capsule(style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("tool.library")

                divider
                tool("scissors", "Split", enabled: canSplit) {
                    if let b = selectedBlock { doc.split(b.id, at: playhead) }
                }
                tool("plus.square.on.square", "Duplicate", enabled: selectedBlock != nil) {
                    if let b = selectedBlock { doc.duplicate(b.id) }
                }
                tool("slider.horizontal.3", "Adjust", enabled: selectedBlock != nil) {
                    if let id = selectedBlock?.id { sheet = .clip(id) }
                }
                tool("trash", "Delete", enabled: selectedBlock != nil || doc.selectedNote != nil,
                     destructive: true) {
                    if let n = doc.selectedNote { doc.removeNote(n) }
                    else { doc.remove(doc.selection) }
                }
                divider
                tool("text.bubble", "Note") { sheet = .newNote(playhead) }
                tool("diamond", "Keyframe") { sheet = .keyframe(nil) }
                divider
                zoomStepper
            }
            .padding(.horizontal, 12).padding(.vertical, 5)
        }
        .frame(height: 34)
    }

    private var divider: some View {
        Rectangle().fill(Theme.Palette.hairlineSoft).frame(width: 1, height: 18)
    }

    /// Zoom as ONE control. Two separate 28 pt buttons pushed the strip past the
    /// width of a 6.1" screen, and a tool strip that has to scroll to reach zoom
    /// is a tool strip whose last item nobody finds.
    private var zoomStepper: some View {
        HStack(spacing: 0) {
            zoomHalf("minus.magnifyingglass", "Zoom out") { scale.zoom(1 / 1.4) }
            Rectangle().fill(Theme.Palette.hairlineSoft).frame(width: 1, height: 14)
            zoomHalf("plus.magnifyingglass", "Zoom in") { scale.zoom(1.4) }
        }
        .background(Theme.Palette.surface,
                    in: RoundedRectangle(cornerRadius: 7, style: .continuous))
    }

    private func zoomHalf(_ icon: String, _ title: String,
                          action: @escaping () -> Void) -> some View {
        Button {
            Haptics.select()
            action()
        } label: {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .semibold))
                .frame(width: 30, height: 24)
                .foregroundStyle(Theme.Palette.textPrimary)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
        .accessibilityIdentifier("tool.\(title.lowercased().replacingOccurrences(of: " ", with: ""))")
    }

    private var canSplit: Bool {
        guard let b = selectedBlock, !b.isBlend else { return false }
        return playhead > b.start + kMinBlockDuration && playhead < b.end - kMinBlockDuration
    }

    private func tool(_ icon: String, _ title: String, enabled: Bool = true,
                      destructive: Bool = false, action: @escaping () -> Void) -> some View {
        Button {
            Haptics.select()
            action()
        } label: {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .semibold))
                .frame(width: 28, height: 24)
                .foregroundStyle(!enabled ? Theme.Palette.textTertiary
                                 : destructive ? Theme.Palette.danger : Theme.Palette.textPrimary)
                .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                // Without this the hit area is the GLYPH, and an outline symbol
                // (text.bubble, diamond) is hollow in the middle — the tap fell
                // straight through the hole and the button did nothing.
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityLabel(title)
        .accessibilityIdentifier("tool.\(title.lowercased().replacingOccurrences(of: " ", with: ""))")
    }

    // MARK: - actions

    private func zoomToFit() {
        let dur = max(2, doc.timeline.duration)
        // The narrowest track area this platform draws; fitting to it is right
        // on the small end and merely conservative on a big window.
        scale.pxPerSecond = min(TimeScale.maxZoom,
                                max(TimeScale.minZoom, kFitWidth / CGFloat(dur)))
    }

}

#if DEBUG
extension AutonomyTimelineView {
    /// Screenshot harness for the editor's pop-ups (the Simulator cannot be tapped
    /// headlessly): `defaults write <bundle> debugPopup -string <route>` with
    /// library · clip · note · newNote · documents · rename · execute · keyframe.
    fileprivate func openDebugPopup() {
        guard let route = UserDefaults.standard.string(forKey: "debugPopup") else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            switch route {
            case "library":   sheet = .library
            case "clip":
                if let b = doc.timeline.blocks.first(where: { !$0.isBlend }) {
                    doc.selection = [b.id]; sheet = .clip(b.id)
                }
            case "note":      if let n = doc.timeline.notes.first { sheet = .note(n.id) }
            case "newNote":   sheet = .newNote(playhead)
            case "documents": sheet = .documents
            case "rename":    sheet = .rename
            case "execute":   sheet = .execute
            case "keyframe":  sheet = .keyframe(nil)
            case "editKey":
                let id = doc.addKey(at: 1, pose: RobotModel.homePose, label: "rest")
                sheet = .keyframe(id)
            case "newPose":   sheet = .newPose
            case "run":       player.execute(on: controller, from: 0)    // nav badges in demo (debugSkills)
            case "deleteTimeline": sheet = .deleteTimeline(UUID(), "Greeting")
            default: break
            }
        }
    }
}

extension AutonomyTimelineView {
    /// A populated timeline for the screenshot harness: one of everything the
    /// editor can draw — a nav block, two manipulation clips whose junction
    /// forces a safety blend, a stacked second row, a voice line and an
    /// annotation. Built through the document so it obeys every rule the
    /// editor does (including the blend the guard inserts by itself).
    static func demoTimeline(_ catalog: ActionCatalog) -> Timeline {
        let doc = TimelineDocument(catalog: catalog)
        doc.rename("Greeting")
        if let s = catalog.spec("nav_forward") { doc.add(s, at: 0) }
        if let s = catalog.spec("pickup") { doc.add(s, at: 0) }
        if let s = catalog.spec("wave") { doc.add(s, at: 11) }
        doc.addRow(.manipulation)
        if let s = catalog.spec("pose_tall") { doc.add(s, at: 2.5, row: 1) }
        if let s = catalog.spec("say"), let id = doc.add(s, at: 11.5).first {
            doc.setSay(id, "Hello there")
        }
        doc.addNote("guest arrives", at: 10.5)
        return doc.timeline
    }
}
#endif

/// Which sheet the editor is showing. One route, one `.sheet` modifier.
enum SheetRoute: Identifiable, Hashable {
    case library
    case clip(UUID)
    case note(UUID)
    case newNote(TimeInterval)
    case keyframe(UUID?)
    case documents
    case deleteTimeline(UUID, String)
    case rename
    case execute
    case newPose

    var id: String {
        switch self {
        case .library:                 return "library"
        case .clip(let u):             return "clip-\(u)"
        case .note(let u):             return "note-\(u)"
        case .newNote(let t):          return "newNote-\(t)"
        case .keyframe(let u):         return "keyframe-\(u?.uuidString ?? "new")"
        case .documents:               return "documents"
        case .deleteTimeline(let u, _): return "deleteTimeline-\(u)"
        case .rename:                  return "rename"
        case .execute:                 return "execute"
        case .newPose:                 return "newPose"
        }
    }
}

extension View {
    /// The library sheet: full height on iPhone (large detent, the app's canvas
    /// behind it, no system grabber chrome fighting the header), a roomy window on
    /// Vision Pro (its min size lives in ActionLibrarySheet).
    @ViewBuilder
    func timelineLibraryPresentation() -> some View {
        #if os(visionOS)
        self
        #else
        self
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
            .presentationBackground(Theme.Palette.canvas)
            .presentationCornerRadius(MabelOverlay.corner + 8)
        #endif
    }
}

// MARK: - the document library

struct TimelineLibrarySheet: View {
    @ObservedObject var store: TimelineStore
    @ObservedObject var doc: TimelineDocument
    /// Ask before deleting a saved timeline (the editor swaps to a confirm dialog).
    var confirmDelete: (Timeline) -> Void = { _ in }
    @Environment(\.mabelDismiss) private var dismiss

    var body: some View {
        MabelModal(title: "Timelines", subtitle: "Saved on this device",
                   shrinkToFit: true, onClose: dismiss) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    tile("Save “\(doc.timeline.name)”", "square.and.arrow.down") {
                        store.save(doc.timeline, as: doc.timeline.name); dismiss()
                    }
                    tile("New timeline", "plus") {
                        doc.replace(with: Timeline(name: "Untitled")); dismiss()
                    }
                }
                MabelModalSection(title: "Saved") {
                    if store.saved.isEmpty {
                        Text("Nothing saved yet — Save keeps a copy you can reopen later.")
                            .font(.system(size: 12)).foregroundStyle(Theme.Palette.textTertiary)
                    } else {
                        ForEach(store.saved) { tl in
                            HStack(spacing: 10) {
                                Button { doc.replace(with: tl); dismiss() } label: {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(tl.name).font(.system(size: 14, weight: .semibold))
                                            .foregroundStyle(Theme.Palette.textPrimary)
                                        Text("\(tl.blocks.count) clips · \(TimelineCanvas.label(tl.duration))")
                                            .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                                Button { confirmDelete(tl) } label: {
                                    Image(systemName: "trash").font(.system(size: 12, weight: .semibold))
                                        .foregroundStyle(Theme.Palette.danger)
                                        .frame(width: 30, height: 30)
                                        .background(Theme.Palette.danger.opacity(0.10), in: Circle())
                                }
                                .buttonStyle(PressableStyle())
                                .accessibilityLabel("Delete \(tl.name)")
                            }
                            .padding(9)
                            .background(Theme.Palette.surface,
                                        in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
                            .overlay {
                                RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous)
                                    .strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1)
                            }
                        }
                    }
                }
            }
        }
        .onAppear { store.reload() }
    }

    private func tile(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button { Haptics.select(); action() } label: {
            Label(title, systemImage: icon)
                .font(.system(size: 12.5, weight: .semibold)).lineLimit(1)
                .foregroundStyle(Theme.Palette.accent)
                .frame(maxWidth: .infinity, minHeight: 40)
                .background(Theme.Palette.accent.opacity(0.10),
                            in: RoundedRectangle(cornerRadius: MabelOverlay.rowCorner, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
    }
}


/// "ODOM ↑ 0.31 m · ← 0.02 m · ⟲ 4°" — the robot's MEASURED pose in the viewer's world frame (origin = where
/// it stood when the viewer first saw it), and a button that re-zeroes the world there. Polled at 5 Hz from
/// the track rather than observed, so the 50 Hz odometry never re-evaluates the editor.
private struct LiveOdomChip: View {
    let track: LiveOdomTrack
    let onRezero: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            SwiftUI.TimelineView(.periodic(from: .now, by: 0.2)) { _ in
                Text("ODOM " + (track.current.map(TimelineWorldFrame.readout) ?? "waiting for wheels…"))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Color(red: 0.30, green: 0.85, blue: 1.0))
            }
            Button {
                Haptics.select(); onRezero()
            } label: {
                Image(systemName: "location.north.circle")
                    .font(.system(size: 12, weight: .semibold))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Re-zero odometry here")
            .accessibilityIdentifier("timeline.odomRezero")
        }
        .padding(.horizontal, 9).padding(.vertical, 5)
        .background(.black.opacity(0.45), in: Capsule())
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("timeline.odomReadout")
    }
}
