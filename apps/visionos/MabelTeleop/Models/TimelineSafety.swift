//
//  TimelineSafety.swift
//  MabelTeleop (visionOS)
//
//  Sampling a timeline, and the two passes that make one safe to run.
//
//  SAMPLING — `pose(at:)` / `nav(at:)` answer "what is commanded at t", merging
//  stacked rows (a higher row wins the joints it drives) over the keyframe lane,
//  exactly as the studio's executor merges blocks over its setpoint sequence.
//
//  POSE JUMPS — the executor HOLDS the last commanded pose across a gap, so at
//  the next block's first tick it would command the whole delta in one 50 ms
//  step. The guard detects every such discontinuity and bridges it with an eased
//  blend long enough to keep the peak inside `MotionMath.safeBlendVel`.
//
//  A block has TWO edges and both can jump. Looking only at starts missed the
//  handover that happens when a block ENDS while something is still running
//  underneath it — on this app's own demo document (`pose_tall` layered over a
//  running `pickup`) that handover commanded 31.75 rad/s, 13x the budget, and
//  was reported as no jump at all. Each blend samples its endpoints at its OWN
//  edges rather than at the boundary, which makes it continuous wherever it is
//  placed and lets it pre-empt the tail of the clip it is leaving instead of
//  needing a gap to live in. The one rule left is that a blend may not span
//  another block's start on its own lane and row: that block begins later and
//  would out-rank it halfway through.
//
//  LOCATION GAPS — a clip can declare the place it must run at ("make coffee" is
//  a kitchen action). If the choreography assumes the robot is somewhere it
//  never drove to, that is a gap, and auto-fix inserts the missing move.
//
//  Both passes are RECONCILERS, not one-shot inserters: they run after every
//  edit and must converge. The studio's did not, once — a blend hid the jump
//  that justified it, so the next pass deleted the blend, and the timeline
//  oscillated. `poseBefore` looks THROUGH blends for that reason.
//

import Foundation

/// A discontinuity in the commanded pose, at one edge of one block.
struct PoseJump: Identifiable, Hashable {
    var id: String { "\(blockID):\(edge.rawValue)" }
    let blockID: UUID
    /// Which edge of that block: `.lead` bridges INTO its start, `.tail` OUT of
    /// its end. A block can need both.
    let edge: JumpEdge
    let lane: ActionLane
    let row: Int
    /// The boundary itself — where the discontinuity is.
    let time: TimeInterval
    /// Worst-joint distance between the poses the bridge actually connects, in
    /// radian-equivalents.
    let delta: Double
    let joint: String?
    /// The shortest bridge that stays inside the velocity budget.
    let minDuration: TimeInterval
    /// What it can actually be, given what else is on this lane and row.
    let duration: TimeInterval
    /// How much clear span the placement has before it would swallow another
    /// block's start. A stretch the operator asks for is capped by this.
    let room: TimeInterval
    /// The peak this bridge will actually command, at the executor's own rate.
    let peak: Double
    /// True when the two differ: the guard is hemmed in and says so rather than
    /// pretending the junction is safe.
    let tight: Bool
    let from: [String: Double]
    let to: [String: Double]
    /// Where the blend goes.
    var start: TimeInterval { edge == .lead ? max(0, time - duration) : time }
}

/// A place the choreography assumes the robot is standing, with no move to it.
struct LocationGap: Identifiable, Hashable {
    var id: UUID { beforeBlockID }
    let beforeBlockID: UUID
    let time: TimeInterval
    let from: ActionLocation?
    let to: ActionLocation
}

extension TimelineDocument {

    // MARK: - what is active

    /// The block commanding `lane`/`row` at `t`. Later-starting wins an overlap,
    /// the way the top-most clip does in an NLE.
    func activeBlock(at t: TimeInterval, lane: ActionLane, row: Int,
                     skipBlends: Bool = false) -> TimelineBlock? {
        var winner: TimelineBlock?
        for b in timeline.blocks where b.lane == lane && b.row == row {
            if skipBlends && b.isBlend { continue }
            let holdEnd = (catalog.spec(b.actionID)?.holdable ?? false) ? Double.infinity : b.end
            guard t >= b.start, t < min(b.end, holdEnd) else { continue }
            if winner == nil || b.start >= winner!.start { winner = b }
        }
        return winner
    }

    /// Every block active at `t`, lowest row first — the order they merge in.
    func activeBlocks(at t: TimeInterval, skipBlends: Bool = false) -> [TimelineBlock] {
        var out: [TimelineBlock] = []
        for lane in ActionLane.tracks {
            for row in 0..<timeline.rowCount(lane) {
                if let b = activeBlock(at: t, lane: lane, row: row, skipBlends: skipBlends) {
                    out.append(b)
                }
            }
        }
        return out
    }

    /// The joints one block commands at timeline time `t`, or nil.
    func blockPose(_ b: TimelineBlock, at t: TimeInterval) -> [String: Double]? {
        // A `both` action's joints ride its manipulation twin; the nav half only
        // drives the base, or the pose would be applied twice.
        if catalog.spec(b.actionID)?.lane == .both && b.lane == .navigation { return nil }
        return MotionClips.shared.joints(b.actionID, at: b.local(t), params: b.params,
                                         blendFrom: b.blend?.from, blendTo: b.blend?.to)
    }

    /// The full pose commanded at `t`: keyframes underneath, blocks on top,
    /// higher rows over lower ones, and then a BRIDGE, if one is running, over
    /// all of it. `nil` when nothing is commanding anything.
    ///
    /// A bridge owns the pose while it runs. It has to: it is the one object on
    /// the timeline whose whole job is to be continuous with what is either side
    /// of it, and a lane or row that overrides part of it breaks exactly that.
    /// Letting the NAV lane keep the lift through a bridge (`lift_up` is lane
    /// nav but commands joints) meant the bridge never moved the lift — and the
    /// tick the lift block ended, the pose underneath was revealed: measured
    /// 19.9 rad/s, eight times the hard cap.
    func pose(at t: TimeInterval, skipBlends: Bool = false) -> [String: Double]? {
        var out = keyPose(at: t)
        var touched = out != nil
        for b in activeBlocks(at: t, skipBlends: true) {
            guard let j = blockPose(b, at: t) else { continue }
            out = out.map { $0.merging(j) { _, new in new } } ?? j
            touched = true
        }
        if !skipBlends, let x = activeBridge(at: t), let j = blockPose(x, at: t) {
            out = out.map { $0.merging(j) { _, new in new } } ?? j
            touched = true
        }
        return touched ? out : nil
    }

    /// The bridge running at `t`, latest-starting first — across lanes and rows,
    /// because a bridge is not a lane's business while it runs.
    func activeBridge(at t: TimeInterval) -> TimelineBlock? {
        var winner: TimelineBlock?
        for b in timeline.blocks where b.isBlend {
            guard t >= b.start, t < b.end else { continue }
            if winner == nil || b.start >= winner!.start { winner = b }
        }
        return winner
    }

    /// The nav stream commanded at `t` — the top-most active navigation block.
    func nav(at t: TimeInterval) -> NavStick? {
        for row in stride(from: timeline.rowCount(.navigation) - 1, through: 0, by: -1) {
            guard let b = activeBlock(at: t, lane: .navigation, row: row) else { continue }
            if let stick = MotionClips.shared.nav(b.actionID, at: b.local(t), params: b.params) {
                return stick
            }
        }
        return nil
    }

    /// The `move to goal` block active at `t`, if any. Goals are dispatched once
    /// on their leading edge and cancelled at their end.
    func goal(at t: TimeInterval) -> TimelineBlock? {
        for row in stride(from: timeline.rowCount(.navigation) - 1, through: 0, by: -1) {
            if let b = activeBlock(at: t, lane: .navigation, row: row),
               catalog.spec(b.actionID)?.isGoal == true { return b }
        }
        return nil
    }

    /// Speech active at `t` — what the Voice track is saying.
    func speech(at t: TimeInterval) -> TimelineBlock? {
        for row in stride(from: timeline.rowCount(.language) - 1, through: 0, by: -1) {
            if let b = activeBlock(at: t, lane: .language, row: row) { return b }
        }
        return nil
    }

    // MARK: - pose jumps

    /// The pose the executor commands at `t`, blends excluded. In a dead gap the
    /// last activity's final pose stands, because that is what the robot is
    /// physically doing.
    func poseAt(_ t: TimeInterval) -> [String: Double]? {
        if let direct = pose(at: t, skipBlends: true) { return direct }
        var lastEnd: TimeInterval?
        for b in timeline.blocks where !b.isBlend {
            guard blockCommandsJoints(b) else { continue }
            let e = b.end
            if e <= t + 1e-6, lastEnd == nil || e > lastEnd! { lastEnd = e }
        }
        guard let e = lastEnd else { return nil }
        return pose(at: e - 1e-3, skipBlends: true)
    }

    /// The pose the executor is holding just BEFORE `t`, blends excluded.
    func poseBefore(_ t: TimeInterval) -> [String: Double]? { poseAt(t - 1e-3) }

    /// The earliest block START strictly inside `(a, b)` on this lane and row —
    /// a blend may not span one, because that block begins later and would
    /// out-rank it halfway through.
    private func startInside(_ a: TimeInterval, _ b: TimeInterval,
                             lane: ActionLane, row: Int) -> TimeInterval? {
        var hit: TimeInterval?
        for blk in timeline.blocks
        where !blk.isBlend && blk.lane == lane && blk.row == row && blockCommandsJoints(blk) {
            if blk.start > a + 1e-6, blk.start < b - 1e-6, hit == nil || blk.start < hit! {
                hit = blk.start
            }
        }
        return hit
    }

    /// Does this block CONTRIBUTE JOINTS to the commanded pose? Lane is the wrong
    /// test and it cost a real defect: `lift_up` / `lift_down` live on the NAV
    /// lane but they command joints, so a lift block was invisible to the guard
    /// — it did not count as activity when measuring the dead gap, and its edges
    /// were not boundaries.
    private func blockCommandsJoints(_ b: TimelineBlock) -> Bool {
        if b.isBlend { return false }
        if catalog.spec(b.actionID)?.lane == .both && b.lane == .navigation { return false }
        return MotionClips.shared.clips[b.actionID]?.hasJoints ?? false
    }

    /// End of the last thing that commanded joints at or before `t`, and the
    /// start of the next one after it. Bridges excluded — a bridge is not an
    /// activity, it is the transition into one.
    private func prevActivityEnd(_ t: TimeInterval) -> TimeInterval {
        var last = 0.0
        for b in timeline.blocks where blockCommandsJoints(b) {
            if b.end <= t + 1e-6, b.end > last { last = b.end }
        }
        return last
    }
    private func nextActivityStart(_ t: TimeInterval) -> TimeInterval? {
        var next: TimeInterval?
        for b in timeline.blocks where blockCommandsJoints(b) {
            if b.start > t + 1e-6, next == nil || b.start < next! { next = b.start }
        }
        return next
    }

    /// Every discontinuity in the commanded pose, at both edges of every block.
    func detectPoseJumps() -> [PoseJump] {
        var jumps: [PoseJump] = []
        let blocks = timeline.blocks
            .filter { !$0.isBlend && blockCommandsJoints($0) }
            .sorted { $0.start < $1.start }
        // A start and an end at the same instant is ONE boundary, and the
        // arriving block wins it — hence starts are collected first.
        var bounds: [(t: TimeInterval, edge: JumpEdge, block: TimelineBlock)] =
            blocks.map { ($0.start, .lead, $0) }
        for b in blocks where !blocks.contains(where: { abs($0.start - b.end) < 1e-6 }) {
            bounds.append((b.end, .tail, b))
        }
        bounds.sort { $0.t < $1.t }

        var seen = Set<String>()
        for (t, edge, block) in bounds {
            let key = String(format: "%.4f", t)
            if seen.contains(key) { continue }
            seen.insert(key)
            guard let before = poseAt(t - 1e-3), let after = poseAt(t + 1e-3) else { continue }
            let probe = MotionMath.poseDelta(from: before, to: after)
            guard probe.0 >= MotionMath.blendThresh else { continue }

            // ── SIZING ──────────────────────────────────────────────────
            // The bridge is continuous whatever its length (its endpoints are
            // read off the signal at its own edges), so length is not a
            // correctness question — it is a SPEED question, and a question of
            // what the bridge is allowed to eat. Two rules, in order:
            //
            //   1. NEVER exceed maxJointVel. Hard.
            //   2. Prefer safeBlendVel, and prefer to live in the DEAD GAP.
            //
            // Rule 2 is what the first version got backwards: it solved for the
            // gentle speed unconditionally and so reached back into the outgoing
            // clip for as long as it took. On `wave` cut at 2.5 s with `happy` at
            // 4 s that was a 3.15 s bridge starting at 0.85 s — five sixths of
            // the wave erased to save 1.4 rad/s. It also iterated the length as a
            // fixed point, which has a degenerate solution: reach back far enough
            // and the outgoing clip is at REST there, the measured distance
            // collapses, and the guard concludes no bridge is needed. Cut a clip,
            // butt the next one on, and you got no bridge and 40 rad/s.
            let gap = edge == .lead ? t - prevActivityEnd(t)
                                    : (nextActivityStart(t) ?? .infinity) - t
            let want = MotionMath.blendDuration(for: probe.0)
            let ceiling: TimeInterval = edge == .lead
                ? t - (startInside(0, t, lane: block.lane, row: block.row) ?? -.infinity)
                : (startInside(t, .infinity, lane: block.lane, row: block.row) ?? .infinity) - t

            func spanAt(_ T: TimeInterval)
                -> (t0: TimeInterval, from: [String: Double], to: [String: Double],
                    delta: Double, joint: String?, peak: Double)? {
                let t0 = edge == .lead ? max(0, t - T) : t
                guard let a = edge == .lead ? poseAt(t0 - 1e-3) : before,
                      let b = edge == .lead ? after : poseAt(t + T + 1e-3) else { return nil }
                let (d, j) = MotionMath.poseDelta(from: a, to: b)
                return (t0, a, b, d, j, d * MotionMath.easedPeakRate(T))
            }

            // Start inside the gap — all of it, or only as much as the gentle
            // speed needs — then grow only while the HARD cap is still broken.
            var dur = min(max(MotionMath.blendMinDur, min(want, max(gap, 0))), ceiling)
            var span = spanAt(dur)
            if let s = span, s.peak > MotionMath.maxJointVel {
                var T = dur + MotionMath.snapGrid
                while T <= min(ceiling, MotionMath.blendMaxDur) + 1e-9 {
                    guard let next = spanAt(T) else { break }
                    dur = T; span = next
                    if next.peak <= MotionMath.maxJointVel { break }
                    T += MotionMath.snapGrid
                }
            }
            guard let s = span, s.delta > 1e-6 else { continue }
            jumps.append(PoseJump(blockID: block.id, edge: edge, lane: block.lane,
                                  row: block.row, time: t, delta: s.delta, joint: s.joint,
                                  minDuration: dur, duration: dur, room: ceiling,
                                  peak: s.peak,
                                  // TIGHT = could not be made gentle, because the
                                  // gap is short and growing would eat the clip.
                                  // Still inside the hard cap; the operator can
                                  // lengthen it, which is theirs to decide.
                                  tight: s.peak > MotionMath.safeBlendVel + 1e-6,
                                  from: s.from, to: s.to))
        }
        return jumps
    }

    /// Build the blend block that bridges a jump.
    func makeBlend(_ j: PoseJump, duration: TimeInterval? = nil) -> TimelineBlock {
        // Never shorter than the duration the detector sized for this placement:
        // `j.duration` is `minDuration` clamped so the blend does not swallow
        // another block's start.
        let d = max(MotionMath.blendMinDur, max(j.duration, duration ?? 0))
        let start = j.edge == .tail ? j.time : max(0, j.time - d)
        var b = TimelineBlock(actionID: "blend", lane: j.lane, row: j.row,
                              start: start, duration: d, params: ["dur": d])
        b.blend = BlendInfo(from: j.from, to: j.to, forBlock: j.blockID, edge: j.edge,
                            auto: true, minDuration: j.minDuration, userDuration: duration,
                            delta: j.delta, joint: j.joint)
        return b
    }

    /// Insert one blend by hand (the ⚡ marker, when auto-fix is off). It is
    /// marked NOT auto, so it belongs to the operator: the reconciler will
    /// neither re-anchor nor delete it, and turning auto-fix off again does not
    /// sweep it away with the machine's own.
    func insertBlend(for jump: PoseJump) {
        applyEdit { tl in
            var b = self.makeBlend(jump)
            b.blend?.auto = false
            tl.blocks.append(b)
        }
    }

    // MARK: - location gaps

    /// Walk the blocks in time order tracking where the robot is implied to be.
    /// A block that declares a location different from the tracked one, with no
    /// goal block bridging, is a gap. Location-less actions are agnostic.
    func detectGaps() -> [LocationGap] {
        var location = startLocation
        var out: [LocationGap] = []
        for b in timeline.blocks.sorted(by: { $0.start < $1.start }) {
            guard let spec = catalog.spec(b.actionID) else { continue }
            if spec.isGoal {
                location = ActionLocation(x: b.params["x"] ?? 0, y: b.params["y"] ?? 0,
                                          label: "goal")
                continue
            }
            guard let want = spec.location else { continue }
            if let at = location, abs(at.x - want.x) < 0.25, abs(at.y - want.y) < 0.25 {
                continue
            }
            out.append(LocationGap(beforeBlockID: b.id, time: b.start, from: location, to: want))
            location = want                       // report a place once, not per block
        }
        return out
    }

    /// Insert the missing `move to goal` for a gap, sized by the distance at the
    /// base's cruising speed and placed so it lands just before the block.
    func fixGap(_ gap: LocationGap) {
        guard let spec = catalog.spec("nav_goto") else { return }
        let from = gap.from
        let dist = from.map { hypot(gap.to.x - $0.x, gap.to.y - $0.y) } ?? 3.0
        let travel = max(3.0, min(30.0, dist / (MotionMath.navSpeed * 0.6) + 1.5))
        applyEdit { tl in
            var b = TimelineBlock(actionID: spec.id, lane: .navigation,
                                  row: 0, start: max(0, gap.time - travel),
                                  duration: travel,
                                  params: ["x": gap.to.x, "y": gap.to.y])
            b.say = nil
            while let hit = tl.blocks.first(where: { $0.id != b.id && $0.contends(b) }) {
                // Never push the move PAST the block it feeds — shorten instead.
                let room = gap.time - hit.end
                guard room >= kMinBlockDuration else { b.start = max(0, gap.time - b.duration); break }
                b.start = hit.end
                b.outPoint = b.inPoint + room
            }
            tl.blocks.append(b)
        }
    }

    // MARK: - the reconcile pass

    /// Re-run both validators after an edit. Called by `TimelineDocument.mutate`,
    /// so nothing has to remember to.
    func validate() {
        if autoBlend {
            reconcileBlends()
            poseJumps = []
        } else {
            // Off means the machine stops managing them: its own guards go, and
            // the jumps they were hiding become ⚡ markers you can act on. A
            // blend the OPERATOR inserted stays — they put it there — and the
            // jump it covers is not reported as unguarded.
            var next = timeline
            let before = next.blocks.count
            next.blocks.removeAll { $0.blend?.auto == true }
            if next.blocks.count != before { setTimelineDirect(next) }
            let manual = Set(timeline.blocks.compactMap { b -> String? in
                guard let i = b.blend, !i.auto else { return nil }
                return "\(i.forBlock):\(i.edge.rawValue)"
            })
            poseJumps = detectPoseJumps().filter { !manual.contains($0.id) }
        }
        gaps = detectGaps()
    }

    /// Insert / re-anchor / retarget / remove the auto blends so that the set of
    /// blends matches the set of jumps exactly. Deterministic and idempotent:
    /// running it twice must be the same as running it once, or an edit would
    /// oscillate between guarded and unguarded.
    private func reconcileBlends() {
        let jumps = detectPoseJumps()
        // Keyed by block AND edge: one block can need a bridge into its start
        // and another out of its end, and they are different objects.
        let jumpFor = Dictionary(uniqueKeysWithValues: jumps.map { ($0.id, $0) })
        let keyOf: (BlendInfo) -> String = { "\($0.forBlock):\($0.edge.rawValue)" }

        var next = timeline
        var seen = Set<String>()
        // Drop auto blends whose jump vanished, and duplicates for one jump.
        next.blocks.removeAll { b in
            guard let info = b.blend, info.auto else { return false }
            let k = keyOf(info)
            if jumpFor[k] == nil || seen.contains(k) { return true }
            seen.insert(k)
            return false
        }
        for j in jumps {
            let existing = next.blocks.firstIndex {
                $0.blend?.auto == true && $0.blend.map(keyOf) == j.id
            }
            // A duration the operator set by trimming is theirs to keep, as long
            // as it still fits between the blocks around it.
            let userDur = next.blocks[safe: existing]?.blend?.userDuration ?? 0
            let dur = max(j.duration, min(userDur, j.room))
            let t0 = j.edge == .tail ? j.time : max(0, j.time - dur)
            if let i = existing {
                let b = next.blocks[i]
                let same = abs(b.start - t0) < 1e-3 && abs(b.duration - dur) < 1e-3
                    && b.blend?.to == j.to && b.blend?.from == j.from
                if !same {
                    next.blocks[i].start = t0
                    next.blocks[i].inPoint = 0
                    next.blocks[i].outPoint = dur
                    next.blocks[i].params["dur"] = dur
                    next.blocks[i].row = j.row
                    next.blocks[i].lane = j.lane
                    next.blocks[i].blend = BlendInfo(
                        from: j.from, to: j.to, forBlock: j.blockID, edge: j.edge,
                        auto: true, minDuration: j.minDuration,
                        userDuration: b.blend?.userDuration,
                        delta: j.delta, joint: j.joint)
                }
            } else {
                next.blocks.append(makeBlend(j))
            }
        }
        if next.blocks != timeline.blocks { setTimelineDirect(next) }
    }
}

// MARK: - where the base ends up

/// One sample of the integrated base path, relative to wherever the robot
/// starts: metres in the odom frame plus heading.
/// Where the base is, in the frame the timeline STARTS in: `dx` metres FORWARD, `dy` metres
/// LEFT, `yaw` radians counter-clockwise. A plain body frame on purpose — the viewer maps it onto
/// the rig once (`TimelineWorldFrame.scene`), so no sign algebra lives here.
struct BasePose: Equatable {
    var dx: Double = 0
    var dy: Double = 0
    var yaw: Double = 0
}

extension TimelineDocument {
    /// Turn the navigation lane into a base trajectory in the START frame, so the 3-D preview can
    /// show the robot actually driving rather than miming it in place. Stick → body twist with the
    /// studio's calibration (forward = navSpeed·ly, left = navSpeedStrafe·lx, ω = navYawRate·rx CCW),
    /// integrated in the frame the robot holds at that instant.
    ///
    /// It used to integrate the WEB STUDIO's world-frame signs ((−cos θ, −sin θ) forward) and the
    /// viewer then mapped that through a 90°-rotated placement, so "drive forward" slid the robot
    /// sideways off its own nose — and on a featureless floor that read as not moving at all
    /// (2026-10-25). `TimelineModelTests` now pins DIRECTION, not just distance.
    func integrateBasePath(through end: TimeInterval, dt: TimeInterval = 0.05) -> [BasePose] {
        let n = max(1, Int((end / dt).rounded(.up)))
        var out = [BasePose](repeating: BasePose(), count: n + 1)
        var x = 0.0, y = 0.0, th = 0.0
        for i in 1...n {
            let s = nav(at: Double(i - 1) * dt) ?? .zero
            let vf = MotionMath.navSpeed * s.ly            // +forward
            let vl = MotionMath.navSpeedStrafe * s.lx      // +left (studio wire: lx = +LEFT strafe)
            x += (cos(th) * vf - sin(th) * vl) * dt
            y += (sin(th) * vf + cos(th) * vl) * dt
            th += MotionMath.navYawRate * s.rx * dt        // +CCW (studio wire: rx = +CCW yaw)
            out[i] = BasePose(dx: x, dy: y, yaw: th)
        }
        return out
    }
}

private extension Array {
    subscript(safe index: Int?) -> Element? {
        guard let index, indices.contains(index) else { return nil }
        return self[index]
    }
}
