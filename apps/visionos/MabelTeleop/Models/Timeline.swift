//
//  Timeline.swift
//  MabelTeleop (visionOS)
//
//  The choreography DOCUMENT and its edit operations — the model a video-editor
//  style UI drives. Deliberately free of SwiftUI: the same model backs the iPhone
//  editor and the wide Vision Pro one, and it is unit-testable without a screen.
//
//  Mirrors the web studio's ChoreoTrack semantics so a timeline means the same
//  thing in the app and in the studio: blocks carry a clip-local `in`/`out`
//  sub-range (so a cut half plays ITS half of the clip, not the clip again),
//  edits snap to a deterministic 0.1 s grid plus block edges and the playhead,
//  `both` actions drop a linked pair, and a lane can hold several stacked rows.
//
//  Sampling (`pose(at:)`, `nav(at:)`) and the safety passes live next door in
//  `TimelineSafety.swift`.
//

import Combine
import Foundation

// MARK: - the pieces

/// One scheduled action: a template instance with its own placement and its own
/// slice of the clip.
struct TimelineBlock: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var actionID: String
    var lane: ActionLane
    /// Which stacked row inside the lane. Rows layer like an NLE's video tracks:
    /// a higher row wins the joints it commands, so a "wave" on row 1 can play
    /// over a "breathe" on row 0 without either being deleted.
    var row: Int = 0
    /// Seconds from the start of the timeline.
    var start: TimeInterval
    /// The slice of the clip this block plays, in CLIP-local seconds. A cut
    /// partitions the parent's range between the two halves; a trim moves an
    /// edge inside it.
    var inPoint: TimeInterval = 0
    var outPoint: TimeInterval
    var params: [String: Double] = [:]
    /// Set on the two halves of a `both` action so moving one moves its twin.
    var linkID: UUID?
    /// Free text for a `language` block (what the robot says).
    var say: String?
    /// Present only on a safety blend — the eased bridge the pose-jump guard
    /// inserts. See `TimelineSafety`.
    var blend: BlendInfo?

    var duration: TimeInterval { outPoint - inPoint }
    var end: TimeInterval { start + duration }
    var isBlend: Bool { blend != nil }

    /// Clip-local time for a timeline time — what the sampler asks the registry.
    func local(_ t: TimeInterval) -> TimeInterval { inPoint + (t - start) }

    func overlaps(_ other: TimelineBlock) -> Bool {
        start < other.end - 1e-6 && other.start < end - 1e-6
    }
    /// Two blocks contend for the same resource only inside the same row.
    func contends(_ other: TimelineBlock) -> Bool {
        lane == other.lane && row == other.row && overlaps(other)
    }

    init(id: UUID = UUID(), actionID: String, lane: ActionLane, row: Int = 0,
         start: TimeInterval, duration: TimeInterval, params: [String: Double] = [:],
         linkID: UUID? = nil, say: String? = nil, blend: BlendInfo? = nil) {
        self.id = id
        self.actionID = actionID
        self.lane = lane
        self.row = row
        self.start = start
        self.inPoint = 0
        self.outPoint = duration
        self.params = params
        self.linkID = linkID
        self.say = say
        self.blend = blend
    }
}

/// A safety blend's payload. `from`/`to` are POSES, so they cannot ride in the
/// block's `[String: Double]` parameter bag.
/// The two edges of a block, both of which can be a discontinuity.
enum JumpEdge: String, Codable, Hashable {
    case lead, tail
}

struct BlendInfo: Codable, Hashable {
    var from: [String: Double]
    var to: [String: Double]
    /// The block whose edge this bridges. A blend outlives an edit only as long
    /// as its jump does.
    var forBlock: UUID
    /// WHICH edge: a block has two, and both can jump. `.lead` ends at the
    /// block's start so the block plays from its own first tick; `.tail` begins
    /// at the block's end, easing off the pose it stopped commanding into
    /// whatever took the joints back.
    var edge: JumpEdge = .lead
    /// Inserted by the guard (true) or dropped by the operator (false).
    var auto: Bool = true
    /// The shortest length that keeps this jump inside the velocity budget.
    var minDuration: TimeInterval
    /// A length the operator chose by trimming. Survives reconciliation; the
    /// floor does not move.
    var userDuration: TimeInterval?
    var delta: Double
    var joint: String?
}

/// A free-text annotation on the timeline — the studio's LANGUAGE lane. Action
/// blocks render a derived label automatically; these are what you write
/// yourself ("say hello to the guests here").
struct TimelineNote: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var start: TimeInterval
    var duration: TimeInterval = 2
    var text: String
    var end: TimeInterval { start + duration }
}

/// A captured pose on the timeline — the studio's ◆ keyframe. Between two keys
/// the pose is interpolated linearly; block joints win over keys, per joint,
/// exactly as the studio's executor merges them.
struct PoseKey: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var time: TimeInterval
    var pose: [String: Double]
    var label: String = ""
}

/// A named choreography. `Codable` so it round-trips to disk and to the robot.
struct Timeline: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var name: String = "Untitled"
    var blocks: [TimelineBlock] = []
    var notes: [TimelineNote] = []
    var keys: [PoseKey] = []
    /// How many stacked rows each lane shows. Keyed by lane raw value so the
    /// document stays Codable without a custom coder.
    var rows: [String: Int] = [:]
    var modified: Date = Date()

    /// Total length — the timeline ends when its last piece of content does.
    var duration: TimeInterval {
        max(blocks.map(\.end).max() ?? 0,
            max(notes.map(\.end).max() ?? 0, keys.map(\.time).max() ?? 0))
    }

    func blocks(in lane: ActionLane) -> [TimelineBlock] {
        blocks.filter { $0.lane == lane }.sorted { $0.start < $1.start }
    }
    func blocks(in lane: ActionLane, row: Int) -> [TimelineBlock] {
        blocks.filter { $0.lane == lane && $0.row == row }.sorted { $0.start < $1.start }
    }
    /// Rows a lane draws: at least one, at least enough to hold its blocks.
    func rowCount(_ lane: ActionLane) -> Int {
        let declared = rows[lane.rawValue] ?? 1
        let used = (blocks.filter { $0.lane == lane }.map(\.row).max() ?? 0) + 1
        return max(1, max(declared, used))
    }
    var totalRows: Int { ActionLane.tracks.reduce(0) { $0 + rowCount($1) } }
}

/// Minimum a block can be trimmed to. Shorter than this is not a motion, it is
/// a glitch: the studio's blend floor is 0.4 s and a block below it would ask
/// the robot for a step change.
let kMinBlockDuration: TimeInterval = MotionMath.blendMinDur
/// Deterministic edit grid — the studio's `SNAP_GRID`.
let kSnapGrid: TimeInterval = 0.1

// MARK: - the document

/// Edit operations. Every mutation goes through here so undo, the linked-pair
/// rule, snapping and the safety passes cannot be forgotten at a call site.
@MainActor
final class TimelineDocument: ObservableObject {
    @Published private(set) var timeline: Timeline
    @Published var selection: Set<UUID> = []
    @Published var selectedNote: UUID?
    @Published var selectedKey: UUID?
    @Published private(set) var canUndo = false
    @Published private(set) var canRedo = false

    /// Bridge pose discontinuities automatically (the studio's Auto-fix, default
    /// ON). Off, the same jumps surface as ⚡ markers you can tap.
    @Published var autoBlend = true {
        didSet { guard autoBlend != oldValue else { return }; validate(); objectWillChange.send() }
    }
    /// Unbridged pose jumps, when `autoBlend` is off. Written only by
    /// `validate()` (TimelineSafety.swift) — never set one by hand.
    @Published var poseJumps: [PoseJump] = []
    /// Places the choreography assumes the robot is standing that it never
    /// drives to (the studio's location-gap validator). Same rule.
    @Published var gaps: [LocationGap] = []
    /// Where the robot is assumed to start, for gap detection.
    var startLocation: ActionLocation?

    private var undoStack: [Timeline] = []
    private var redoStack: [Timeline] = []
    let catalog: ActionCatalog

    /// `nil` = a fresh document. Taking an Optional rather than defaulting to
    /// `Timeline()` keeps the construction INSIDE this main-actor initialiser;
    /// a default argument is evaluated in a nonisolated context.
    init(timeline: Timeline? = nil, catalog: ActionCatalog) {
        self.timeline = timeline ?? Timeline()
        self.catalog = catalog
        validate()
    }

    // MARK: - history

    /// One edit = one undo entry = one revalidation. Nothing mutates `timeline`
    /// except through here.
    private func mutate(_ change: (inout Timeline) -> Void) {
        undoStack.append(timeline)
        if undoStack.count > 60 { undoStack.removeFirst() }
        redoStack.removeAll()
        var next = timeline
        change(&next)
        next.modified = Date()
        timeline = next
        canUndo = !undoStack.isEmpty
        canRedo = false
        validate()
    }

    /// A live drag is ONE edit: `begin` snapshots, `step` mutates without
    /// stacking, `end` validates. Without this a 60-frame drag left 60 undo
    /// entries and ran the safety reconcile 60 times.
    private var dragDepth = 0
    func beginInteraction() {
        if dragDepth == 0 {
            undoStack.append(timeline)
            if undoStack.count > 60 { undoStack.removeFirst() }
            redoStack.removeAll()
            canUndo = true
            canRedo = false
        }
        dragDepth += 1
    }
    func endInteraction() {
        dragDepth = max(0, dragDepth - 1)
        if dragDepth == 0 { validate() }
    }
    private func step(_ change: (inout Timeline) -> Void) {
        var next = timeline
        change(&next)
        next.modified = Date()
        timeline = next
    }
    /// Inside a drag, edits stack once; outside, each is its own undo entry.
    private func edit(_ change: (inout Timeline) -> Void) {
        dragDepth > 0 ? step(change) : mutate(change)
    }

    /// One undoable edit, for the safety passes next door (TimelineSafety.swift).
    func applyEdit(_ change: (inout Timeline) -> Void) { mutate(change) }

    /// Replace the document WITHOUT an undo entry and WITHOUT revalidating.
    /// Only the blend reconciler uses this: its output is a consequence of the
    /// operator's edit, not an edit of its own, and re-entering `validate()`
    /// from inside it would not terminate.
    func setTimelineDirect(_ next: Timeline) {
        var t = next
        t.modified = Date()
        timeline = t
    }

    func undo() {
        guard let prev = undoStack.popLast() else { return }
        redoStack.append(timeline)
        timeline = prev
        canUndo = !undoStack.isEmpty
        canRedo = true
        pruneSelection()
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(timeline)
        timeline = next
        canUndo = true
        canRedo = !redoStack.isEmpty
        pruneSelection()
    }

    private func pruneSelection() {
        let live = Set(timeline.blocks.map(\.id))
        selection.formIntersection(live)
    }

    // MARK: - snapping

    /// The studio's deterministic snap: the 0.1 s grid always, overridden by a
    /// block edge, the playhead or t=0 when one is within `tolerance` seconds.
    /// Returns the snapped time and what it snapped TO, so the UI can thump.
    func snap(_ t: TimeInterval, excluding: UUID? = nil, playhead: TimeInterval? = nil,
              tolerance: TimeInterval = 0.12) -> (time: TimeInterval, magnetic: Bool) {
        var best = (t / kSnapGrid).rounded() * kSnapGrid
        var bestDistance = tolerance
        var magnetic = false
        var candidates: [TimeInterval] = [0]
        if let playhead { candidates.append(playhead) }
        for b in timeline.blocks where b.id != excluding {
            candidates.append(b.start)
            candidates.append(b.end)
        }
        for c in candidates {
            let d = abs(c - t)
            if d < bestDistance { bestDistance = d; best = c; magnetic = true }
        }
        return (max(0, (best * 10_000).rounded() / 10_000), magnetic)
    }

    // MARK: - add

    /// Append an action at `at` (or at the end of its lane). A `both` action
    /// becomes a LINKED PAIR — one block per track — because driving and
    /// manipulating are separate resources that happen to start together.
    @discardableResult
    func add(_ spec: ActionSpec, at when: TimeInterval? = nil, row: Int = 0) -> [UUID] {
        guard spec.available else { return [] }
        let lanes: [ActionLane] = spec.lane == .both ? [.navigation, .manipulation] : [spec.lane]
        let link = spec.lane == .both ? UUID() : nil
        var ids: [UUID] = []
        let length = MotionClips.shared.clips[spec.id] != nil
            ? MotionClips.shared.duration(spec.id, params: spec.defaults)
            : spec.duration
        mutate { tl in
            let t = when ?? lanes.map { lane in
                tl.blocks.filter { $0.lane == lane }.map(\.end).max() ?? 0
            }.max() ?? 0
            for lane in lanes {
                var b = TimelineBlock(actionID: spec.id, lane: lane, row: min(row, tl.rowCount(lane) - 1),
                                      start: max(0, t), duration: max(kMinBlockDuration, length),
                                      params: spec.defaults, linkID: link)
                if spec.category == .language { b.say = "" }
                Self.place(&b, in: &tl)
                ids.append(b.id)
            }
        }
        return ids
    }

    /// Drop a block into a row, sliding it later until it stops colliding. Two
    /// motions on one resource at one instant is not a thing the robot can do,
    /// so one ROW refuses to represent it — stack a row if you want layers.
    private static func place(_ b: inout TimelineBlock, in tl: inout Timeline) {
        var candidate = b
        var guard_ = 0
        while let hit = tl.blocks.first(where: { $0.id != candidate.id && $0.contends(candidate) }),
              guard_ < 200 {
            candidate.start = hit.end
            guard_ += 1
        }
        b = candidate
        tl.blocks.append(b)
    }

    /// Stack another row on a lane — the "multiple timelines over each other"
    /// an NLE calls a track.
    func addRow(_ lane: ActionLane) {
        mutate { tl in tl.rows[lane.rawValue] = tl.rowCount(lane) + 1 }
    }

    /// Drop the lane's last row if nothing is on it (never below one).
    func removeEmptyRow(_ lane: ActionLane) {
        mutate { tl in
            let n = tl.rowCount(lane)
            guard n > 1, !tl.blocks.contains(where: { $0.lane == lane && $0.row == n - 1 }) else { return }
            tl.rows[lane.rawValue] = n - 1
        }
    }

    // MARK: - move / trim / split

    /// Move a block (and its linked twin) to `when`, optionally to another row.
    func move(_ id: UUID, to when: TimeInterval, row: Int? = nil) {
        edit { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }) else { return }
            let delta = max(0, when) - tl.blocks[i].start
            let link = tl.blocks[i].linkID
            if let row { tl.blocks[i].row = max(0, min(row, tl.rowCount(tl.blocks[i].lane) - 1)) }
            for j in tl.blocks.indices where tl.blocks[j].id == id
                || (link != nil && tl.blocks[j].linkID == link) {
                tl.blocks[j].start = max(0, tl.blocks[j].start + delta)
            }
        }
    }

    /// Settle a dragged block: if it landed on top of a neighbour in its own
    /// row, push it clear (the magnetic-timeline behaviour). Rows exist for
    /// deliberate layering; an accidental overlap inside one is just a mistake.
    func settle(_ id: UUID) {
        mutate { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }) else { return }
            var b = tl.blocks[i]
            var guard_ = 0
            while let hit = tl.blocks.first(where: { $0.id != b.id && $0.contends(b) }), guard_ < 200 {
                // Push to whichever side is nearer, so a nudge does not fling it.
                b.start = (b.start - hit.start) < (hit.end - b.end)
                    ? max(0, hit.start - b.duration) : hit.end
                guard_ += 1
            }
            tl.blocks[i] = b
        }
    }

    enum Edge { case leading, trailing }

    /// Trim from either edge. Dragging the left edge moves the start AND shortens
    /// (the block's tail stays put) — the behaviour every video editor has. The
    /// edge is clamped to the clip's own length: a fixed-length motion cannot be
    /// stretched into one the studio would refuse to play.
    func trim(_ id: UUID, edge: Edge, to when: TimeInterval) {
        edit { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }) else { return }
            let b = tl.blocks[i]

            // A blend trims by its LEFT edge only and never below its floor: the
            // right edge is anchored to the jump it guards, so moving it would
            // reopen the discontinuity the blend exists to close.
            if let info = b.blend {
                guard edge == .leading else { return }
                let floor = max(MotionMath.blendMinDur, info.minDuration)
                let dur = max(floor, b.end - when)
                tl.blocks[i].start = max(0, b.end - dur)
                tl.blocks[i].inPoint = 0
                tl.blocks[i].outPoint = dur
                tl.blocks[i].blend?.userDuration = dur
                return
            }

            let spec = self.catalog.spec(b.actionID)
            let natural = MotionClips.shared.clips[b.actionID] != nil
                ? MotionClips.shared.duration(b.actionID, params: b.params)
                : (spec?.duration ?? b.duration)
            let holdable = spec?.holdable ?? false
            switch edge {
            case .leading:
                // `in` walks into the clip; the tail (both t0+len and `out`) holds.
                let newIn = b.inPoint + (when - b.start)
                let v = max(0, min(b.outPoint - kMinBlockDuration, newIn))
                tl.blocks[i].start = b.start + (v - b.inPoint)
                tl.blocks[i].inPoint = v
            case .trailing:
                let newOut = b.inPoint + max(kMinBlockDuration, when - b.start)
                tl.blocks[i].outPoint = holdable ? newOut : min(natural, newOut)
            }
        }
    }

    /// Set a bridge's ramp duration outright — the numeric twin of dragging its
    /// leading edge, for when "a bit gentler" is easier to say than to drag.
    /// `nil` hands it back to the guard, which re-derives it on the next edit.
    ///
    /// A bridge grows LEFTWARDS: its right edge is anchored to the boundary it
    /// closes, so lengthening it reaches further back into whatever precedes it.
    /// The floor is the guard's own sizing (shortening past that re-opens the
    /// jump); the ceiling is the room before the previous block's start.
    func setBlendDuration(_ id: UUID, _ seconds: TimeInterval?) {
        edit { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }),
                  let info = tl.blocks[i].blend else { return }
            guard let seconds else {
                tl.blocks[i].blend?.userDuration = nil
                return                      // the reconciler re-sizes it
            }
            let end = tl.blocks[i].end
            let dur = max(max(MotionMath.blendMinDur, info.minDuration),
                          min(seconds, MotionMath.blendMaxDur))
            tl.blocks[i].start = max(0, end - dur)
            tl.blocks[i].inPoint = 0
            tl.blocks[i].outPoint = dur
            tl.blocks[i].params["dur"] = dur
            tl.blocks[i].blend?.userDuration = dur
        }
    }

    /// Cut a block in two at `when`. Each half plays ITS slice of the clip, so
    /// trimming one half is how you shorten a phrase — and playing the pair back
    /// to back is indistinguishable from the uncut block.
    @discardableResult
    func split(_ id: UUID, at when: TimeInterval) -> UUID? {
        var made: UUID?
        mutate { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }) else { return }
            let b = tl.blocks[i]
            // Half a blend is a NEW pose jump: blading one would leave two
            // fragments that each cross part of the gap, turning the guard into
            // the hazard it exists to prevent.
            guard b.blend == nil else { return }
            let local = when - b.start
            guard local >= kMinBlockDuration, b.duration - local >= kMinBlockDuration else { return }
            var right = b
            right.id = UUID()
            right.start = when
            right.inPoint = b.inPoint + local
            right.outPoint = b.outPoint
            right.linkID = nil                         // halves are independent after a cut
            tl.blocks[i].outPoint = b.inPoint + local
            tl.blocks.append(right)
            made = right.id
        }
        return made
    }

    @discardableResult
    func duplicate(_ id: UUID) -> UUID? {
        var made: UUID?
        mutate { tl in
            guard let b = tl.blocks.first(where: { $0.id == id }) else { return }
            var copy = b
            copy.id = UUID()
            copy.linkID = nil
            copy.blend = nil                            // a copy of a guard is not a guard
            copy.start = b.end
            Self.place(&copy, in: &tl)
            made = copy.id
        }
        return made
    }

    func remove(_ ids: Set<UUID>) {
        mutate { tl in
            let links = Set(tl.blocks.filter { ids.contains($0.id) }.compactMap(\.linkID))
            tl.blocks.removeAll { ids.contains($0.id)
                || ($0.linkID.map { links.contains($0) } ?? false) }
        }
        selection.subtract(ids)
    }

    func setParam(_ id: UUID, _ key: String, _ value: Double) {
        mutate { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }) else { return }
            tl.blocks[i].params[key] = value
            // A nav primitive's LENGTH is a function of its distance/angle, so
            // changing the parameter must re-length the block — otherwise the
            // block says 1.9 s and the robot drives for 4.
            let b = tl.blocks[i]
            if MotionMath.navPrimitives[b.actionID] != nil || b.actionID == "nav_hold" {
                let d = MotionClips.shared.duration(b.actionID, params: b.params)
                tl.blocks[i].inPoint = 0
                tl.blocks[i].outPoint = max(kMinBlockDuration, d)
            }
        }
    }

    func setSay(_ id: UUID, _ text: String) {
        mutate { tl in
            guard let i = tl.blocks.firstIndex(where: { $0.id == id }) else { return }
            tl.blocks[i].say = text
        }
    }

    func rename(_ name: String) { mutate { $0.name = name } }

    /// Close the gaps in every row — the "tidy up" a long edit always needs.
    func compact() {
        mutate { tl in
            for lane in ActionLane.tracks {
                for row in 0..<tl.rowCount(lane) {
                    var t: TimeInterval = 0
                    for b in tl.blocks(in: lane, row: row) {
                        guard let i = tl.blocks.firstIndex(where: { $0.id == b.id }) else { continue }
                        tl.blocks[i].start = t
                        t = tl.blocks[i].end
                    }
                }
            }
        }
    }

    // MARK: - annotations

    @discardableResult
    func addNote(_ text: String, at when: TimeInterval, duration: TimeInterval = 2) -> UUID {
        let note = TimelineNote(start: max(0, when), duration: duration, text: text)
        mutate { $0.notes.append(note) }
        return note.id
    }
    func moveNote(_ id: UUID, to when: TimeInterval) {
        edit { tl in
            guard let i = tl.notes.firstIndex(where: { $0.id == id }) else { return }
            tl.notes[i].start = max(0, when)
        }
    }
    func retimeNote(_ id: UUID, duration: TimeInterval) {
        edit { tl in
            guard let i = tl.notes.firstIndex(where: { $0.id == id }) else { return }
            tl.notes[i].duration = max(0.3, duration)
        }
    }
    func setNoteText(_ id: UUID, _ text: String) {
        mutate { tl in
            guard let i = tl.notes.firstIndex(where: { $0.id == id }) else { return }
            tl.notes[i].text = text
        }
    }
    func removeNote(_ id: UUID) {
        mutate { tl in tl.notes.removeAll { $0.id == id } }
        if selectedNote == id { selectedNote = nil }
    }

    // MARK: - keyframes

    /// Capture a pose at the playhead. This is how a timeline records something
    /// you posed by hand rather than something the registry knows how to do.
    @discardableResult
    func addKey(at when: TimeInterval, pose: [String: Double], label: String = "") -> UUID {
        let key = PoseKey(time: max(0, when), pose: pose, label: label)
        mutate { tl in
            tl.keys.append(key)
            tl.keys.sort { $0.time < $1.time }
        }
        return key.id
    }
    func moveKey(_ id: UUID, to when: TimeInterval) {
        edit { tl in
            guard let i = tl.keys.firstIndex(where: { $0.id == id }) else { return }
            tl.keys[i].time = max(0, when)
            tl.keys.sort { $0.time < $1.time }
        }
    }
    func setKeyLabel(_ id: UUID, _ label: String) {
        mutate { tl in
            guard let i = tl.keys.firstIndex(where: { $0.id == id }) else { return }
            tl.keys[i].label = label
        }
    }
    func removeKey(_ id: UUID) {
        mutate { tl in tl.keys.removeAll { $0.id == id } }
        if selectedKey == id { selectedKey = nil }
    }

    /// The keyframe lane's pose at `t`: hold before the first and after the
    /// last, linear interpolation between neighbours (`sequence.js`).
    func keyPose(at t: TimeInterval) -> [String: Double]? {
        let keys = timeline.keys
        guard let first = keys.first, let last = keys.last else { return nil }
        if t <= first.time { return first.pose }
        if t >= last.time { return last.pose }
        var i = 0
        while i < keys.count - 1 && keys[i + 1].time < t { i += 1 }
        let a = keys[i], b = keys[i + 1]
        let u = (t - a.time) / max(1e-6, b.time - a.time)
        return MotionMath.lerpPose(a.pose, b.pose, u)
    }

    // MARK: - whole-document

    func replace(with timeline: Timeline) {
        mutate { $0 = timeline }
        selection.removeAll()
    }

    func clear() {
        mutate { tl in
            let name = tl.name
            tl = Timeline(name: name)
        }
        selection.removeAll()
    }
}
