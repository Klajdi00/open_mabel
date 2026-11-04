//
//  MotionClips.swift
//  MABEL Mobile App
//
//  What each clip actually DOES, so the app can show it and play it.
//
//  The studio authors motion as JS closures (`motions.js`). Re-typing that math
//  in Swift would be a second source of truth for how MABEL moves, so the app
//  does not: `simulation/tools/gen_motion_clips.py` RUNS the registry and
//  samples every clip at the executor's own 20 Hz command rate into
//  `Resources/motion_clips.json`, and this file interpolates those samples.
//
//  That is not a lossy stand-in. The studio's executor sends a joint command
//  every 50 ms and the servo ramps between them, so at the rate that reaches
//  hardware the table IS the motion — and it cannot drift from the JS, because
//  it is produced from the JS. What is left parametric (the nav primitives and
//  the safety blend) lives in `MotionMath.swift` and is checked against golden
//  samples the same generator records.
//

import Foundation

/// The baked registry. One shared instance — parsing 150 KB of JSON once at
/// first use, off the hot path of any edit.
final class MotionClips {
    static let shared = MotionClips()

    struct Clip {
        let duration: TimeInterval
        let hold: Bool
        let isGoal: Bool
        let isBlend: Bool
        /// Joints that vary over the clip: id → one sample per frame at `rate`.
        let tracks: [String: [Double]]
        /// Joints the clip commands but never MOVES: id → the one value it
        /// holds for the whole clip. A pose clip is mostly these, and holding
        /// a joint still is as much a command as moving it — the lift stays
        /// half-extended and the wrists stay level because these say so.
        let holds: [String: Double]
        let frames: Int
        /// Baked nav stream for the clips that sway the base while they play.
        let navTracks: [String: [Double]]
        let navFrames: Int
        /// Fastest sampled joint, rad/s — surfaced in the UI and asserted in tests.
        let peakVelocity: Double
        let peakJoint: String?
        var hasJoints: Bool { !tracks.isEmpty || !holds.isEmpty }
    }

    private(set) var clips: [String: Clip] = [:]
    /// The studio's own constants, as recorded by the generator. Tests compare
    /// `MotionMath`'s literals against these so a drifted budget fails loudly.
    private(set) var constants: [String: Double] = [:]
    private(set) var navGolden: [String: [[Double]]] = [:]
    private(set) var navProfileGolden: [Double] = []
    private(set) var ranges: [String: ClosedRange<Double>] = [:]
    /// The studio's rest pose, as the generator recorded it. Every clip starts
    /// here and every non-hold clip ends here, so the app checks
    /// `RobotModel.homePose` against this rather than keeping a second copy.
    private(set) var rest: [String: Double] = [:]
    private(set) var rate: Double = 20

    private init() {
        load()
        // The table is baked per robot (MABEL: v1 hands, NOVA: v2 hands — different joint limits and
        // ranges); when the gateway hello switches the rig, reload so the next clip plays the right hand.
        NotificationCenter.default.addObserver(forName: .robotRigChanged, object: nil, queue: .main) { [weak self] _ in
            guard let self else { return }
            self.clips = [:]; self.constants = [:]; self.navGolden = [:]; self.ranges = [:]; self.rest = [:]
            self.load()
        }
    }

    // MARK: - loading

    private struct File: Decodable {
        let clips: [String: Entry]
        let consts: Consts
    }
    private struct Entry: Decodable {
        let dur: Double
        let hold: Bool
        let kind: String?
        let frames: Int?
        let tracks: [String: [Double]]?
        let holds: [String: Double]?
        let navTracks: [String: [Double]]?
        let navFrames: Int?
        let navGolden: [[Double]]?
        let peakVel: Double?
        let peakJoint: String?
    }
    private struct Consts: Decodable {
        let rate: Double
        let navSpeed: Double
        let navSpeedStrafe: Double
        let navYawRate: Double
        let safeBlendVel: Double
        let easePeak: Double
        let blendThresh: Double
        let blendMinDur: Double
        let maxJointVel: Double
        let gripScale: Double
        let navProfileGolden: [Double]
        let ranges: [String: [Double]]
        let rest: [String: Double]?
    }

    private func load() {
        let table = RobotRig.isNova ? "nova_motion_clips" : "motion_clips"   // baked per robot by gen_motion_clips.py
        guard let url = Bundle.main.url(forResource: table, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let file = try? JSONDecoder().decode(File.self, from: data) else {
            assertionFailure("\(table).json missing — run simulation/tools/gen_motion_clips.py (--robot nova for NOVA)")
            return
        }
        for (id, e) in file.clips {
            clips[id] = Clip(duration: e.dur, hold: e.hold,
                             isGoal: e.kind == "goal", isBlend: e.kind == "xfade",
                             tracks: e.tracks ?? [:], holds: e.holds ?? [:],
                             frames: e.frames ?? 0,
                             navTracks: e.navTracks ?? [:], navFrames: e.navFrames ?? 0,
                             peakVelocity: e.peakVel ?? 0, peakJoint: e.peakJoint)
            if let g = e.navGolden { navGolden[id] = g }
        }
        let c = file.consts
        rate = c.rate
        navProfileGolden = c.navProfileGolden
        constants = ["rate": c.rate, "navSpeed": c.navSpeed,
                     "navSpeedStrafe": c.navSpeedStrafe, "navYawRate": c.navYawRate,
                     "safeBlendVel": c.safeBlendVel, "easePeak": c.easePeak,
                     "blendThresh": c.blendThresh, "blendMinDur": c.blendMinDur,
                     "maxJointVel": c.maxJointVel, "gripScale": c.gripScale]
        ranges = c.ranges.compactMapValues { $0.count == 2 ? $0[0]...$0[1] : nil }
        rest = c.rest ?? [:]
    }

    // MARK: - duration

    /// How long a clip runs with these parameters. Parametric nav primitives are
    /// sized by their distance/angle; `nav_hold` by its seconds; a blend by the
    /// jump it bridges; everything else is fixed by its table.
    func duration(_ id: String, params: [String: Double] = [:]) -> TimeInterval {
        if let p = MotionMath.navPrimitives[id] {
            return p.duration(params, default: clips[id]?.duration ?? 2)
        }
        if id == "nav_hold" { return max(0.5, params["secs"] ?? clips[id]?.duration ?? 2) }
        if clips[id]?.isBlend == true {
            return max(MotionMath.blendMinDur, params["dur"] ?? MotionMath.blendMinDur)
        }
        return clips[id]?.duration ?? 0
    }

    // MARK: - sampling

    /// The pose a clip commands at clip-local time `t`, clamped into its range
    /// (a held clip therefore holds its final pose past its natural length —
    /// which is what "hold" means).
    ///
    /// `blend` carries its endpoints in `blendFrom`/`blendTo` because they are
    /// poses, not scalars, and a block's `params` dictionary is `[String: Double]`.
    func joints(_ id: String, at t: TimeInterval, params: [String: Double] = [:],
                blendFrom: [String: Double]? = nil,
                blendTo: [String: Double]? = nil) -> [String: Double]? {
        guard var out = handJoints(id, at: t, params: params, blendFrom: blendFrom, blendTo: blendTo) else { return nil }
        // A side wearing the OpenArm gripper plays the clip's hand as the gripper's one DOF.
        RobotModel.mapHandToGripper(&out, isGripper: { RobotRig.handGen($0) == "gripper" })
        return out
    }

    /// `joints` as the table stores it (ORCA fingers + grip), before the gripper mapping.
    private func handJoints(_ id: String, at t: TimeInterval, params: [String: Double],
                            blendFrom: [String: Double]?, blendTo: [String: Double]?) -> [String: Double]? {
        guard let clip = clips[id] else { return nil }
        if clip.isBlend {
            guard let from = blendFrom, let to = blendTo else { return nil }
            return MotionMath.blendPose(from: from, to: to, at: t,
                                        duration: duration(id, params: params))
        }
        guard clip.hasJoints else { return nil }
        var out: [String: Double] = [:]
        out.reserveCapacity(clip.tracks.count + clip.holds.count)
        for (k, v) in clip.holds { out[k] = v }
        guard clip.frames > 1 else {
            for (k, row) in clip.tracks { out[k] = row.first ?? 0 }
            return out
        }
        let (i, j, u) = index(t, frames: clip.frames)
        for (k, row) in clip.tracks {
            let a = row[i], b = row[j]
            out[k] = a + (b - a) * u
        }
        return out
    }

    /// The nav stream a clip commands at clip-local time `t` — analytic for the
    /// parametric primitives, table-driven for the clips that sway while they
    /// dance, and zero for `nav_hold` (which is a timed pause, not a command).
    func nav(_ id: String, at t: TimeInterval, params: [String: Double] = [:]) -> NavStick? {
        if let p = MotionMath.navPrimitives[id] {
            let d = p.duration(params, default: clips[id]?.duration ?? 2)
            return p.stick(at: min(max(0, t), d), duration: d)
        }
        if id == "nav_hold" { return .zero }
        guard let clip = clips[id], !clip.navTracks.isEmpty, clip.navFrames > 1 else { return nil }
        let (i, j, u) = index(t, frames: clip.navFrames)
        func axis(_ key: String) -> Double {
            guard let row = clip.navTracks[key] else { return 0 }
            return row[i] + (row[j] - row[i]) * u
        }
        return NavStick(lx: axis("lx"), ly: axis("ly"), rx: axis("rx"), ry: axis("ry"))
    }

    /// Frame pair + fraction for clip-local time `t`, clamped to the table.
    private func index(_ t: TimeInterval, frames: Int) -> (Int, Int, Double) {
        let x = min(max(0, t * rate), Double(frames - 1))
        let i = min(frames - 1, Int(x))
        let j = min(frames - 1, i + 1)
        return (i, j, x - Double(i))
    }

    /// The pose a clip ENDS on — what the next block has to start from. Used by
    /// the pose-jump guard and by "hold the last pose" at a gap.
    func finalPose(_ id: String, params: [String: Double] = [:]) -> [String: Double]? {
        joints(id, at: duration(id, params: params), params: params)
    }
}
