//
//  HandTrackingService.swift
//  MabelTeleop
//
//  Key architecture notes:
//
//  Lag elimination:
//  SnapshotBroadcaster is a lock-protected, actor-free fan-out. The master
//  Task runs via Task.detached (fully off the main actor) and calls
//  broadcaster.send() directly from the ARKit thread.
//  Zero main-actor hops in the hot path.
//
//  Restart safety:
//  ARKit providers are one-shot — calling session.run() on a stopped provider
//  causes a fatal assertion. HandTrackingService creates fresh ARKitSession +
//  HandTrackingProvider instances on every start() call.
//

import Foundation
import ARKit
import QuartzCore
import simd

// MARK: - Thread-safe broadcast helper

/// Lock-protected fan-out that delivers every Snapshot to all registered
/// subscribers without touching any actor. Safe to call from any thread.
public final class SnapshotBroadcaster: @unchecked Sendable {

    private let lock = NSLock()
    private var map: [UUID: AsyncStream<HandTrackingService.Snapshot>.Continuation] = [:]
    private var _latest: HandTrackingService.Snapshot?

    public init() {}

    public func add(_ c: AsyncStream<HandTrackingService.Snapshot>.Continuation) -> UUID {
        let id = UUID()
        lock.withLock { map[id] = c }
        return id
    }

    public func remove(_ id: UUID) {
        lock.withLock { _ = map.removeValue(forKey: id) }
    }

    public func send(_ snapshot: HandTrackingService.Snapshot) {
        // Copy under lock, yield outside to avoid lock contention.
        lock.lock(); _latest = snapshot; let values = Array(map.values); lock.unlock()
        for c in values { c.yield(snapshot) }
    }

    /// Newest snapshot, synchronously — for render-loop consumers (the hand-dot
    /// overlay) that must never ride the async Task machinery: a MainActor
    /// stream consumer is the lowest-priority customer of a congested main
    /// actor, which is exactly when the overlay must stay glued to the hand.
    /// Callers gate on `timestamp` for both freshness and de-dup.
    public func latest() -> HandTrackingService.Snapshot? {
        lock.withLock { _latest }
    }

    public func clearLatest() {
        lock.withLock { _latest = nil }
    }

    public func finish() {
        lock.lock(); let values = Array(map.values); map.removeAll(); lock.unlock()
        for c in values { c.finish() }
    }
}

// MARK: - HandTrackingService

@MainActor
public final class HandTrackingService {

    public struct Snapshot: Sendable {
        public var left:  HandPose?
        public var right: HandPose?
        public var timestamp: TimeInterval
        /// When each side was last REFRESHED by ARKit (not when the snapshot was
        /// emitted — a snapshot carries both hands, but only the side whose
        /// anchor just updated is actually new). Consumers expire a side against
        /// these, which is what stops a hand that left the field of view from
        /// streaming its last pose forever. Default 0 = "never seen".
        public var leftStamp: TimeInterval = 0
        public var rightStamp: TimeInterval = 0

        /// Sides fresher than `cutoff`, others nil.
        public func fresh(after cutoff: TimeInterval) -> (HandPose?, HandPose?) {
            (leftStamp >= cutoff ? left : nil, rightStamp >= cutoff ? right : nil)
        }
    }

    /// The broadcast loop's carried state — the merge + expiry rule as a PURE
    /// value, so it can be driven by tests (the Simulator has no hand tracking,
    /// so this rule has no other harness).
    ///
    /// **AN UNTRACKED ANCHOR IS THE ABSENCE OF A POSE, NOT A POSE** (operator,
    /// 2026-08-01: "the hand tracking is super super laggy — only one hand can
    /// be tracked at a time"). ARKit reports a hand it has momentarily lost as
    /// an anchor with `isTracked == false` and an EMPTY skeleton, and the loop
    /// used to fold those in like any other sample. Two things then went wrong
    /// at once, and they compounded:
    ///
    ///   1. the empty pose OVERWROTE the last good skeleton, and the server
    ///      drops any hand with `isTracked == false` (`mabel/sources/wire.py`
    ///      `_hand_world`) — so that arm stopped being retargeted;
    ///   2. it also REFRESHED the side's freshness stamp, so neither expiry
    ///      (producer or wire tick) could ever fire, and the side sat "fresh"
    ///      while carrying nothing. It could not even fall back to the last
    ///      good pose, because that had already been overwritten.
    ///
    /// Whichever hand ARKit is less sure about — the one further out of the
    /// field of view, typically — therefore dropped out on every blip and came
    /// back on every re-acquisition: one hand at a time, and stuttering.
    ///
    /// The rule now: a tracked sample refreshes the side; an untracked one is
    /// IGNORED, leaving the last good pose in place; and a side that goes
    /// `sideStaleTimeout` without a tracked sample expires to nil. Brief loss
    /// holds, sustained loss drops.
    public struct Merger: Sendable {
        public var left: HandPose?
        public var right: HandPose?
        public var leftStamp: TimeInterval = 0
        public var rightStamp: TimeInterval = 0

        public init() {}

        /// Fold in one ARKit sample. `nil` (or an untracked pose) is a no-op —
        /// deliberately, see the type's note.
        public mutating func apply(_ sample: HandPose?, at now: TimeInterval) {
            guard let sample, sample.isTracked else { return }
            switch sample.chirality {
            case .left:  left = sample;  leftStamp = now
            case .right: right = sample; rightStamp = now
            }
        }

        /// Drop any side not refreshed within `timeout`. A never-seen side has
        /// stamp 0, so it can never pass.
        public mutating func expire(at now: TimeInterval,
                                    timeout: TimeInterval = HandTrackingService.sideStaleTimeout) {
            let cutoff = now - timeout
            if leftStamp < cutoff { left = nil }
            if rightStamp < cutoff { right = nil }
        }

        public func snapshot(at now: TimeInterval) -> Snapshot {
            Snapshot(left: left, right: right, timestamp: now,
                     leftStamp: leftStamp, rightStamp: rightStamp)
        }
    }

    // broadcaster is nonisolated let: safe to access from any context — immutable after init.
    public nonisolated let broadcaster: SnapshotBroadcaster

    /// The LIVE provider, reachable from any thread. `predictedPair` is called
    /// from the 60 Hz wire queue (never main), and the provider itself is
    /// replaced on every `start()` — so the reference has to live somewhere
    /// nonisolated and mutable rather than in MainActor state.
    /// `handAnchors(at:)` is ARKit's poll-from-your-render-loop API; it is the
    /// one HandTrackingProvider call meant to be made off the main thread.
    public final class ProviderBox: @unchecked Sendable {
        private let lock = NSLock()
        private var provider: HandTrackingProvider?
        func set(_ p: HandTrackingProvider?) { lock.withLock { provider = p } }
        func get() -> HandTrackingProvider? { lock.withLock { provider } }
    }
    public nonisolated let providerBox = ProviderBox()

    // session and provider are recreated on every start() because ARKit providers
    // are one-shot — they cannot be restarted after stop().
    private var session:  ARKitSession
    private var provider: HandTrackingProvider

    /// ONE ARKit access path, polled at the display rate.
    ///
    /// This used to be two: a `Task` consuming `provider.anchorUpdates` AND, as
    /// of 2026-07-31, a second read via `handAnchors(at:)` on the 60 Hz wire
    /// tick. Two consumers of one provider at two cadences is what the operator
    /// felt as "the hand tracking is super super laggy … the point overlay
    /// disappears here and there" (2026-08-01): the event stream delivered in
    /// bursts, so the overlay — which paints whatever the last broadcast said —
    /// stuttered and blinked, and it got worse exactly when the second reader
    /// started (whole-body engaged).
    ///
    /// `handAnchors(at:)` is ARKit's poll-from-your-render-loop API and is the
    /// low-latency path; `anchorUpdates` is the event-driven one. So there is
    /// now a single sampler on its own `userInteractive` queue, polling at the
    /// display rate, and EVERY consumer (the wire tick, the dot overlay, the
    /// tracked-flag UI) reads what it publishes. Nothing else touches ARKit.
    private let pollQueue = DispatchQueue(label: "mabel.hands.poll", qos: .userInteractive)
    private var pollTimer: DispatchSourceTimer?
    private var isRunning = false

    /// Vision Pro renders at 90 Hz and ARKit's hand anchors arrive at the same
    /// cadence, so this is the fastest USEFUL rate — polling harder just re-reads
    /// the same anchor. (The wire ships at 60 Hz and the renderer draws at 90, so
    /// sampling at 90 keeps both fed without either waiting on a slower source.)
    public static let pollHz: Double = 90

    /// Prediction lead. visionOS EXTRAPOLATES anchors, so asking slightly AHEAD
    /// cancels transport + render latency instead of adding to it. The head is
    /// sampled with the same lead, which is what keeps the two in register — a
    /// mismatch here is what used to make the neck lead the arms.
    public static let predictLead: TimeInterval = 0.030

    public init() {
        self.broadcaster = SnapshotBroadcaster()
        self.session     = ARKitSession()
        self.provider    = HandTrackingProvider()
    }

    // MARK: - Authorization

    public static var isSupported: Bool { HandTrackingProvider.isSupported }

    public func requestAuthorization() async -> Bool {
        let result = await session.requestAuthorization(for: [.handTracking])
        return result[.handTracking] == .allowed
    }

    // MARK: - Lifecycle

    public func start() async throws {
        guard !isRunning else { return }

        // Always create fresh instances — ARKit providers cannot be restarted.
        session  = ARKitSession()
        provider = HandTrackingProvider()

        try await session.run([provider])
        isRunning = true
        providerBox.set(provider)          // publish to the sampler
        startPolling()
    }

    public func stop() {
        pollTimer?.cancel(); pollTimer = nil
        // Do NOT call broadcaster.finish() here.
        // Subscribers (e.g. the immersive hand-viz task) stay alive across
        // stop/start cycles — they just receive no data while ARKit is stopped
        // and resume automatically when startBroadcast() runs again.
        // DO clear the latest-snapshot slot: a render-loop consumer polling
        // `latest()` must not keep painting a frozen skeleton after tracking
        // stops (its timestamp staleness guard is belt-and-braces on top).
        broadcaster.clearLatest()
        providerBox.set(nil)               // no prediction off a stopped provider
        session.stop()
        isRunning = false
    }

    // MARK: - Prediction

    /// Hand poses at `time` straight from the provider — visionOS PREDICTS
    /// anchors for near-future timestamps, so asking for now + ~30 ms offsets
    /// the transport + retarget latency instead of adding to it. Falls back to
    /// nil when tracking is down (caller keeps the last broadcast snapshot).
    /// NONISOLATED so the 60 Hz wire tick can call it directly — the whole point
    /// is to sample the hands at the same predicted instant as the head without
    /// a hop to the main actor (a hop would re-add the latency this removes).
    public nonisolated func predictedPair(at time: TimeInterval)
        -> (left: HandPose?, right: HandPose?)? {
        guard let provider = providerBox.get() else { return nil }
        let pair = provider.handAnchors(at: time)
        // TRACKED ANCHORS ONLY. `handAnchors(at:)` hands back whatever the
        // provider holds, including a hand it has momentarily lost — an anchor
        // with `isTracked == false` and an empty skeleton. Mapping those
        // straight through made this method essentially never return nil, and
        // the caller's `p.left ?? lLive` fallback then ALWAYS took the
        // prediction: an untracked blip replaced a perfectly good broadcast
        // pose with an empty hand, which the server drops outright. That is the
        // other half of "only one hand can be tracked at a time" (operator,
        // 2026-08-01) — the freshness guard at the call site could not catch it,
        // because the BOX was fresh; it was the prediction that was empty.
        //
        // Skipping the untracked ones also saves building 27 joint transforms
        // per discarded hand, 60 times a second, on the wire queue.
        let l = pair.leftHand.flatMap { $0.isTracked ? Self.makePose(from: $0) : nil }
        let r = pair.rightHand.flatMap { $0.isTracked ? Self.makePose(from: $0) : nil }
        if l == nil && r == nil { return nil }
        return (l, r)
    }

    // MARK: - Subscription (nonisolated — callable from any Task or actor)

    /// Returns an independent AsyncStream for this subscriber.
    /// No actor hop required — uses the lock-based broadcaster directly.
    /// Buffering is NEWEST-ONLY: ARKit produces ~90 Hz, and a consumer that
    /// momentarily falls behind must skip to the present, not replay a
    /// growing backlog (the unbounded default made overlays lag further and
    /// further behind in real time).
    public nonisolated func snapshots() -> AsyncStream<Snapshot> {
        AsyncStream(Snapshot.self, bufferingPolicy: .bufferingNewest(1)) { [broadcaster] continuation in
            let id = broadcaster.add(continuation)
            continuation.onTermination = { [broadcaster, id] _ in
                broadcaster.remove(id)
            }
        }
    }

    // MARK: - Private

    /// A per-side pose older than this is treated as GONE, not as held. ARKit
    /// refreshes a tracked hand at ~90 Hz (11 ms), so 0.25 s is ~22 missed
    /// updates — far past jitter, well inside human reaction time.
    public static let sideStaleTimeout: TimeInterval = 0.25

    /// Carries the `Merger` across timer firings. A class because the timer's
    /// event handler is a closure and the state has to outlive each call;
    /// `@unchecked Sendable` because the handler runs on ONE serial queue, so
    /// there is no concurrent access to protect against.
    private final class SamplerState: @unchecked Sendable {
        private var merger = Merger()
        func step(_ pair: (left: HandPose?, right: HandPose?),
                  at now: TimeInterval) -> Snapshot {
            // BOTH sides every tick: a poll sees both hands at once (unlike the
            // event stream, which delivered one chirality at a time), so the
            // merge is now trivially symmetric — no side can be starved by the
            // other's update rate.
            merger.apply(pair.left, at: now)
            merger.apply(pair.right, at: now)
            // …and expire on EVERY tick, not only when an anchor happens to
            // arrive. This is what the event-stream version could not do: when
            // both hands left the view no update arrived at all, so nothing
            // expired and the wire tick had to re-check at read time.
            merger.expire(at: now)
            return merger.snapshot(at: now)
        }
    }

    private func startPolling() {
        pollTimer?.cancel()
        let interval = 1.0 / Self.pollHz
        let timer = DispatchSource.makeTimerSource(queue: pollQueue)
        // Small leeway: the sampler should keep a steady cadence (the overlay
        // is judged on smoothness) but need not be woken to the microsecond.
        timer.schedule(deadline: .now() + interval, repeating: interval,
                       leeway: .milliseconds(1))
        let state = SamplerState()
        timer.setEventHandler { [weak self, broadcaster] in
            guard let self else { return }
            let now = CACurrentMediaTime()
            // ONE ARKit read, shared by every consumer. `predictedPair` returns
            // nil when NEITHER hand is tracked — that must still drive the
            // expiry, hence the `?? (nil, nil)` rather than an early return.
            let pair = self.predictedPair(at: now + Self.predictLead) ?? (nil, nil)
            broadcaster.send(state.step(pair, at: now))
        }
        timer.resume()
        pollTimer = timer
    }

    // MARK: - ARKit → HandPose conversion (nonisolated static — callable from Task.detached)

    private nonisolated static func makePose(from anchor: HandAnchor) -> HandPose {
        let chirality: Chirality = anchor.chirality == .left ? .left : .right
        var joints: [HandJointPose] = []
        if anchor.isTracked, let skeleton = anchor.handSkeleton {
            joints.reserveCapacity(HandJoint.allCases.count)
            for joint in HandJoint.allCases {
                guard let name = joint.arkitName else { continue }
                let j = skeleton.joint(name)
                joints.append(HandJointPose(
                    joint: joint,
                    localTransform: Transform4x4(j.anchorFromJointTransform),
                    isTracked: j.isTracked
                ))
            }
        }
        return HandPose(
            chirality: chirality,
            anchorTransform: Transform4x4(anchor.originFromAnchorTransform),
            joints: joints,
            isTracked: anchor.isTracked
        )
    }
}

// MARK: - Joint name bridge

extension HandJoint {
    var arkitName: HandSkeleton.JointName? {
        switch self {
        case .wrist:                        return .wrist
        case .thumbKnuckle:                 return .thumbKnuckle
        case .thumbIntermediateBase:        return .thumbIntermediateBase
        case .thumbIntermediateTip:         return .thumbIntermediateTip
        case .thumbTip:                     return .thumbTip
        case .indexFingerMetacarpal:        return .indexFingerMetacarpal
        case .indexFingerKnuckle:           return .indexFingerKnuckle
        case .indexFingerIntermediateBase:  return .indexFingerIntermediateBase
        case .indexFingerIntermediateTip:   return .indexFingerIntermediateTip
        case .indexFingerTip:               return .indexFingerTip
        case .middleFingerMetacarpal:       return .middleFingerMetacarpal
        case .middleFingerKnuckle:          return .middleFingerKnuckle
        case .middleFingerIntermediateBase: return .middleFingerIntermediateBase
        case .middleFingerIntermediateTip:  return .middleFingerIntermediateTip
        case .middleFingerTip:              return .middleFingerTip
        case .ringFingerMetacarpal:         return .ringFingerMetacarpal
        case .ringFingerKnuckle:            return .ringFingerKnuckle
        case .ringFingerIntermediateBase:   return .ringFingerIntermediateBase
        case .ringFingerIntermediateTip:    return .ringFingerIntermediateTip
        case .ringFingerTip:               return .ringFingerTip
        case .littleFingerMetacarpal:       return .littleFingerMetacarpal
        case .littleFingerKnuckle:          return .littleFingerKnuckle
        case .littleFingerIntermediateBase: return .littleFingerIntermediateBase
        case .littleFingerIntermediateTip:  return .littleFingerIntermediateTip
        case .littleFingerTip:              return .littleFingerTip
        case .forearmWrist:                 return .forearmWrist
        case .forearmArm:                   return .forearmArm
        }
    }
}
