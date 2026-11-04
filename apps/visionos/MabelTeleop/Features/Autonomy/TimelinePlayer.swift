//
//  TimelinePlayer.swift
//  MabelTeleop (visionOS)
//
//  Walks a timeline and turns it into motion. TWO SINKS, and the difference is
//  the whole point of having a Preview button next to an Execute one:
//
//    • PREVIEW  — the pose goes into the editor's own 3-D viewer. Nothing
//                 reaches the wire, the robot does not move, and you can scrub
//                 and re-run as often as you like.
//    • EXECUTE  — the SAME walk at the studio's 20 Hz command rate, but every
//                 tick commits: joints to `joint_command`, the nav lane to the
//                 base twist, `move to goal` to Nav2.
//
//  One walk, two sinks, so what you rehearse is what runs. The old player had
//  a hand-written `switch` over clip ids that approximated a handful of them;
//  this one samples the registry (`MotionClips`), which is the studio's own
//  math, so a clip plays here exactly as it plays there.
//

import AVFoundation
import Combine
import Foundation
import SwiftUI

@MainActor
final class TimelinePlayer: ObservableObject {
    enum Mode: Equatable { case idle, preview, executing }

    @Published private(set) var mode: Mode = .idle
    @Published private(set) var playhead: TimeInterval = 0
    @Published private(set) var status: String?
    /// Which block of how many has started — the progress chip.
    @Published private(set) var progress: (n: Int, m: Int) = (0, 0)
    /// CALIBRATED nav moves (gateway cap `nav_move`): the robot's report per block,
    /// drawn as a badge on the block. Outlives the run — a move may still be driving,
    /// and its result still worth reading, after the timeline itself has finished.
    @Published private(set) var navBadges: [UUID: NavMoveStatus] = [:]
    /// The latest calibrated move, in words — shown under the viewer's status chip.
    @Published private(set) var navLine: NavMoveStatus?

    var isPlaying: Bool { mode != .idle }
    var isExecuting: Bool { mode == .executing }

    /// The pose the editor's 3-D viewer renders. A plain `LivePose` (not
    /// `@Published`) for the same reason the teleop viewers use one: it is
    /// written at 30–60 Hz and must not invalidate the SwiftUI tree.
    let previewPose = LivePose()
    /// The planned base path the viewer draws on its world grid (TimelineWorldFrame).
    let previewPath = PreviewPath()

    private let doc: TimelineDocument
    private let catalog: ActionCatalog
    private let speech = AVSpeechSynthesizer()
    private var task: Task<Void, Never>?
    /// Blocks whose leading edge we have already acted on this pass.
    private var fired: Set<UUID> = []
    /// The goal block currently being serviced, so we cancel on the way out.
    private var activeGoal: UUID?
    private var lastSentPose: [String: Double] = [:]
    private var basePath: [BasePose] = []
    /// Sends ONE `nav_move` per nav primitive when the gateway advertises it, instead
    /// of streaming the open-loop stick (see RobotSkills.swift).
    private let calibratedNav = CalibratedNavDriver()
    private weak var runController: RobotController?
    private var moveBlocks: [String: UUID] = [:]
    private var navSub: AnyCancellable?

    /// The studio's executor rate. The bridge coalesces latest-wins, so a faster
    /// loop would only burn battery; a slower one would let a block's ramp
    /// become a staircase.
    /// MIRRORS the Control Center's EXEC_RATE (control_center/web/js/motions.js). It was 20 in
    /// both, feeding a robot whose gateway, CAN links and base link all run at 50: each pose was
    /// held for two or three ticks while the joint slewed toward it and stopped, which playback
    /// showed as step-like jitter. The web moved to 50 on 2026-09-24 and this follows it, because
    /// an animation must look the same from the app as it does from the studio.
    private static let executeHz = 50.0
    /// Preview runs at the display rate instead — it is drawing, not commanding.
    private static let previewHz = 60.0

    init(doc: TimelineDocument, catalog: ActionCatalog) {
        self.doc = doc
        self.catalog = catalog
    }

    // MARK: - transport

    /// Rehearse in the viewer. Never touches the wire.
    func preview(from t: TimeInterval) {
        start(from: t, controller: nil)
    }

    /// Run it on the robot.
    func execute(on controller: RobotController, from t: TimeInterval) {
        guard !controller.estopEngaged else {
            status = "E-stop engaged"
            return
        }
        runController = controller
        navBadges = [:]
        navLine = nil
        moveBlocks = [:]
        navSub = controller.skills.$navMove
            .compactMap { $0 }
            .sink { [weak self] st in self?.noteNavMove(st) }
        start(from: t, controller: controller)
    }

    func stop() {
        // Stop / pause cancels what the robot is still driving for us. A finished move
        // ignores the cancel; one that is mid-way stops where it is.
        calibratedNav.cancelAll(sink: runController?.skills)
        task?.cancel()
        task = nil
        fired.removeAll()
        lastSentPose = [:]
        mode = .idle
        speech.stopSpeaking(at: .immediate)
    }

    /// Park the viewer on the pose at `t` without running — what scrubbing does.
    func scrub(to t: TimeInterval) {
        guard mode == .idle else { return }
        playhead = t
        renderPreview(at: t)
    }

    private func start(from t: TimeInterval, controller: RobotController?) {
        task?.cancel()
        calibratedNav.reset()
        fired.removeAll()
        lastSentPose = [:]
        activeGoal = nil
        playhead = t
        mode = controller == nil ? .preview : .executing
        status = controller == nil ? "Previewing" : "Executing on the robot"
        let end = doc.timeline.duration
        // The base path is integrated ONCE per run: it is a function of the nav
        // lane, which cannot change while the run is in flight.
        basePath = doc.integrateBasePath(through: max(end, t) + 1)
        previewPath.set(basePath)
        let hz = controller == nil ? Self.previewHz : Self.executeHz

        task = Task { [weak self] in
            guard let self else { return }
            var last = Date()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(1e9 / hz))
                if Task.isCancelled { break }
                let now = Date()
                let dt = now.timeIntervalSince(last)
                last = now
                let finished: Bool = await MainActor.run {
                    self.playhead += dt
                    let t = self.playhead
                    self.renderPreview(at: t)
                    if let c = controller {
                        guard !c.estopEngaged else {
                            self.status = "Stopped — e-stop engaged"
                            return true
                        }
                        self.commit(at: t, on: c)
                    } else {
                        self.fireCues(at: t)
                    }
                    self.progress = self.blockProgress(at: t)
                    return t >= end
                }
                if finished { break }
            }
            await MainActor.run {
                let cancelled = Task.isCancelled
                if let c = controller { self.settle(c) }
                self.mode = .idle
                if self.status?.hasPrefix("Stopped —") != true {
                    self.status = cancelled ? "Stopped" : "Finished"
                }
                if !cancelled { self.playhead = end }
            }
        }
    }

    // MARK: - the preview sink

    /// Pose the editor's robot from the timeline. The BASE rides along too, so a
    /// "drive forward 0.5 m" block looks like driving forward rather than like
    /// nothing happening.
    private func renderPreview(at t: TimeInterval) {
        let joints = doc.pose(at: t) ?? RobotModel.homePose
        let base = basePose(at: t)
        previewPose.set(joints, heading: base.yaw, quat: [1, 0, 0, 0],
                        x: base.dx, y: base.dy)
    }

    /// Where the base is at `t`, interpolated from the integrated path.
    func basePose(at t: TimeInterval) -> BasePose {
        guard !basePath.isEmpty else { return BasePose() }
        let x = min(max(0, t / 0.05), Double(basePath.count - 1))
        let i = Int(x), j = min(basePath.count - 1, i + 1)
        let u = x - Double(i)
        let a = basePath[i], b = basePath[j]
        return BasePose(dx: a.dx + (b.dx - a.dx) * u,
                        dy: a.dy + (b.dy - a.dy) * u,
                        yaw: a.yaw + (b.yaw - a.yaw) * u)
    }

    /// Recompute the base path outside a run (so scrubbing shows the robot where
    /// the plan puts it). Cheap: 20 samples per second of timeline.
    func refreshBasePath() {
        basePath = doc.integrateBasePath(through: doc.timeline.duration + 1)
        previewPath.set(basePath)
        if mode == .idle { renderPreview(at: playhead) }
    }

    /// Non-motion cues that a preview should still show: what the robot would be
    /// saying, and which policy would be running.
    private func fireCues(at t: TimeInterval) {
        for b in doc.activeBlocks(at: t) where !fired.contains(b.id) {
            fired.insert(b.id)
            guard let spec = catalog.spec(b.actionID) else { continue }
            switch spec.category {
            case .language:
                if let text = b.say, !text.isEmpty { status = "“\(text)”" }
            case .policy:
                status = spec.available ? spec.title : "\(spec.title) — not deployable yet"
            default:
                status = spec.title
            }
        }
    }

    // MARK: - the wire sink

    /// Everything active at `t` gets driven; everything that just STARTED gets
    /// its one-shot fired.
    ///
    /// The nav lane must be re-sent every tick — the bridge treats a twist as a
    /// command with a watchdog, not a latched setpoint, so a single send stalls
    /// after ~200 ms. Joints are the opposite: an unchanged `joint_command`
    /// stream steals base-drive authority from a concurrent nav block (measured
    /// in the studio), so identical poses are NOT re-sent.
    private func commit(at t: TimeInterval, on c: RobotController) {
        if let pose = doc.pose(at: t) {
            let changed = pose.filter { abs(($0.value) - (lastSentPose[$0.key] ?? .infinity)) > 1e-4 }
            if !changed.isEmpty {
                c.setJoints(changed)
                lastSentPose.merge(pose) { _, new in new }
            }
        }

        // A nav primitive on a gateway with `nav_move`: the robot plays its CALIBRATED
        // profile from one request sent on the block's leading edge, so the app streams
        // zeros for that block (a non-zero stick would pre-empt the move robot-side).
        // No cap (MABEL, an older sim, Bluetooth) → the open-loop stream, unchanged.
        let driving = calibratedNav.tick(at: t, doc: doc, sink: c.skills)
        for (block, id) in calibratedNav.sent where moveBlocks[id] == nil {
            moveBlocks[id] = block
            navBadges[block] = NavMoveStatus(id: id, state: .pending)
            if let b = doc.timeline.blocks.first(where: { $0.id == block }) {
                status = "Calibrated move · \(catalog.spec(b.actionID)?.title ?? b.actionID)"
            }
        }
        let stick = driving != nil ? .zero : (doc.nav(at: t) ?? .zero)
        // Studio wire signs: ly = +forward, lx = +LEFT strafe, rx = +CCW yaw.
        // `driveTwist` wants +forward, +RIGHT strafe, +CLOCKWISE yaw.
        // EXACT: the m/s the preview integrated is what the wheels get (see driveTwistExact)
        c.driveTwistExact(forward: MotionMath.navSpeed * stick.ly,
                     strafe: -MotionMath.navSpeedStrafe * stick.lx,
                     yaw: -MotionMath.navYawRate * stick.rx)

        serviceGoal(at: t, on: c)

        for b in doc.activeBlocks(at: t) where !fired.contains(b.id) {
            fired.insert(b.id)
            guard let spec = catalog.spec(b.actionID) else { continue }
            switch spec.category {
            case .language:
                if let text = b.say, !text.isEmpty {
                    status = "“\(text)”"
                    say(text)
                }
            case .policy:
                status = spec.available ? "Running \(spec.title)"
                                        : "\(spec.title) — not deployable yet"
            default:
                status = spec.title
            }
        }
    }

    /// `move to goal` rides the same `nav_goal` wire the map's tap-to-go uses:
    /// dispatched once on the block's leading edge, cancelled at its end.
    private func serviceGoal(at t: TimeInterval, on c: RobotController) {
        let g = doc.goal(at: t)
        if let g, activeGoal != g.id {
            activeGoal = g.id
            let x = g.params["x"] ?? 0, y = g.params["y"] ?? 0
            c.sendNavGoal(x: x, y: y, yaw: 0)
            status = String(format: "Navigating to (%.1f, %.1f)", x, y)
        } else if g == nil, activeGoal != nil {
            activeGoal = nil
        }
    }

    /// STOP means zeros, repeatedly. Never leave a non-zero last frame: the
    /// base would keep the last twist until its watchdog expired.
    private func settle(_ c: RobotController) {
        activeGoal = nil
        // Deliberately NOT the run task: stopping the run must not stop the
        // zeros that follow it.
        Task { @MainActor in
            for _ in 0..<6 {
                c.driveTwist(forward: 0, strafe: 0, yaw: 0)
                try? await Task.sleep(nanoseconds: 40_000_000)
            }
            c.stopBase()          // and the lift, which rides its own axis
        }
    }

    private func say(_ text: String) {
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = 0.5
        speech.speak(u)
    }

    /// The robot's report for one of our moves → its block's badge (and the status line).
    private func noteNavMove(_ st: NavMoveStatus) {
        guard let block = moveBlocks[st.id] else { return }
        if navBadges[block] != st { navBadges[block] = st }
        if navLine != st { navLine = st }
        if isExecuting || st.isTerminal { status = st.sentence }
    }

    // MARK: - progress

    func blockProgress(at t: TimeInterval) -> (n: Int, m: Int) {
        let blocks = doc.timeline.blocks.filter { !$0.isBlend }
        let started = blocks.filter { t >= $0.start }.count
        return (min(started, blocks.count), blocks.count)
    }
}
