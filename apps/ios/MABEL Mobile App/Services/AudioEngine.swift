
//
//  AudioEngine.swift
//  MABEL Mobile App
//
//  The two halves of voice, kept out of AudioStreamLink so the wire code stays
//  readable: AudioCapture (operator mic → 16 kHz mono PCM frames) and
//  AudioPlayback (robot mic → the phone's speaker).
//
//  Three non-obvious requirements, all learned the hard way and all load-bearing:
//
//  · ECHO CANCELLATION IS MANDATORY, not polish. The robot's bar is a speakerphone
//    in the same room as its own mic. Without the OS DSP the operator hears
//    themselves a beat late and the loop howls. `voiceProcessingEnabled` + the
//    `.voiceChat` mode is the same thing the web client gets from
//    `echoCancellation: true`.
//
//  · PLAYBACK SCHEDULES ON A PLAYHEAD, never "now". Starting every frame at the
//    current time leaves a scheduling-jitter gap between each one — at 50 frames a
//    second that is a click track, not speech. We keep a running playhead and only
//    resync when it falls behind.
//
//  · THE SESSION IS CONFIGURED ONCE, AND NEVER ON THE TALK PRESS (2026-07-29).
//    This is the fix for "press Talk and the robot's room goes silent — and stays
//    silent after you let go". `setCategory`/`setActive` changes the hardware I/O
//    format, and AVAudioEngine answers a hardware-format change by STOPPING
//    ITSELF and tearing down every node connection. PTT used to flip the category
//    on press AND on release, so each press killed the *playback* engine — a
//    different AVAudioEngine, on the same shared session — while `fmt` stayed
//    non-nil, so `ensure()` short-circuited forever and `play()` went on
//    scheduling buffers into a dead graph. Silence with no error, until relaunch.
//
//    Two changes, both necessary:
//      1. ONE configuration (`.playAndRecord` + `.voiceChat`) is applied when the
//         voice PLANE comes up and held for its whole life. Talk then only
//         installs a tap — the session never moves, so nothing gets torn down.
//      2. Both engines SELF-HEAL. Configuration changes are not only ours to
//         cause: plugging in headphones, a Bluetooth handset connecting, a phone
//         call, Siri, or Media Services restarting all post one. Every single one
//         of those used to kill the downlink permanently. Now each engine
//         rebuilds its graph and restarts.
//
//    Ground truth for the server half: `server/tools/exp_audio_duplex.py` holds
//    :9610 at 50 Hz down THROUGH a 50 Hz uplink and after it (measured against
//    the real robot, 2026-07-29) — the plane is genuinely duplex, so a client
//    that goes quiet has broken itself.
//

import Foundation
import AVFoundation

// MARK: - Session arbiter

/// ONE owner of the shared `AVAudioSession`, and one CONFIGURATION for the whole
/// life of the voice plane.
///
/// The invariant: **`setCategory` is called when the voice plane opens, and not
/// again**. `claimForPTT()` deliberately does no session work in the common case
/// — it only asserts that the one configuration is up. `reconfigures` counts
/// every time we actually touch the session so a test can pin that a Talk press
/// costs zero (see `AudioSessionTests`).
///
/// `Features/Policies/RobotHub.swift` also runs an `AVAudioEngine` for
/// `SFSpeechRecognizer` dictation, and its `finishAudio()` calls
/// `setActive(false)`. Two owners on one session is a real conflict, not a
/// formality — so PTT CLAIMS the session and dictation yields. Dictation is a
/// convenience; talking to a person standing in front of the robot is not.
@MainActor
enum AudioSessionArbiter {

    /// The duplex configuration, used once the mic is ours. `.voiceChat` buys the
    /// platform AEC/AGC/NS chain. Before permission the session runs `.playback`
    /// instead — equally immune to the ring/silent switch, but it does not ask for
    /// the microphone (see `beginVoiceSession`).
    static let voiceMode: AVAudioSession.Mode = .voiceChat
    static let voiceOptions: AVAudioSession.CategoryOptions = [.allowBluetoothHFP, .defaultToSpeaker]

    private(set) static var pttHolds = false
    /// True once the voice configuration is applied and the session is active.
    private(set) static var sessionUp = false
    /// False when we had to fall back to playback-only (mic denied / unavailable):
    /// listening still works, Talk honestly refuses.
    private(set) static var canRecord = false
    /// Every real `setCategory` call. A Talk press MUST NOT increment this.
    private(set) static var reconfigures = 0
    /// Set by RobotHub so dictation can be told to stand down.
    static var onPTTClaim: ((Bool) -> Void)?

    /// Bring the voice configuration up. **Idempotent** — when the session is
    /// already in this exact configuration it touches nothing and returns.
    ///
    /// Called when the `:9610` plane connects and before the playback engine
    /// starts, so the engine is built against the format it will actually run at.
    @discardableResult
    static func beginVoiceSession() -> Bool {
        let s = AVAudioSession.sharedInstance()
        // `.playAndRecord` ONLY once the mic is actually ours. Claiming the record
        // half while the permission is still undetermined puts a system prompt in
        // front of someone who opened the Robot space to LOOK at the map — and in
        // a unit-test host, in front of nobody at all, which hangs the runner.
        // Listening never needs it: `.playback` is equally immune to the ring
        // switch. The one-time upgrade happens at the first Talk press.
        let record = micPermission == .granted
        let category: AVAudioSession.Category = record ? .playAndRecord : .playback
        let mode: AVAudioSession.Mode = record ? voiceMode : .default
        let options: AVAudioSession.CategoryOptions = record ? voiceOptions : []

        if sessionUp, s.category == category, s.mode == mode,
           s.categoryOptions.isSuperset(of: options) {
            return canRecord                       // already ours: do NOT re-apply
        }
        do {
            try s.setCategory(category, mode: mode, options: options)
            canRecord = record
        } catch {
            // Recording refused outright: keep LISTENING alive rather than losing
            // both directions.
            try? s.setCategory(.playback, mode: .default, options: [])
            canRecord = false
        }
        reconfigures += 1
        try? s.setActive(true, options: [])
        sessionUp = true
        #if DEBUG
        print("[MABEL-AUDIO] session up cat=\(s.category.rawValue) mode=\(s.mode.rawValue) "
              + "canRecord=\(canRecord) reconfigures=\(reconfigures)")
        #endif
        return canRecord
    }

    /// Reassert the configuration after the SYSTEM took the session away (an
    /// interruption ending, or Media Services restarting). Distinct from
    /// `beginVoiceSession()` only in that it does not trust `sessionUp`.
    static func reactivate() {
        sessionUp = false
        beginVoiceSession()
    }

    /// What the OS will let us do with the mic, as three cases the caller must
    /// each handle differently — `denied` is a sentence to show, `undetermined` is
    /// a prompt to raise, `granted` is a tap to install.
    enum MicPermission { case granted, denied, undetermined }

    static var micPermission: MicPermission {
        #if DEBUG
        if let forced = _micPermissionOverride { return forced }
        #endif
        switch AVAudioApplication.shared.recordPermission {
        case .granted:    return .granted
        case .denied:     return .denied
        default:          return .undetermined
        }
    }

    /// Where the sound is actually GOING, and how loud. The distinction this
    /// exists to make: a playback graph that is `live` and scheduling buffers has
    /// done its job, so if the operator still hears nothing the fault is the ROUTE
    /// or the volume, not the engine — and those two failures look identical from
    /// inside the app unless you print them.
    ///
    /// Worth knowing: in `.voiceChat` the output rides the CALL volume, which is a
    /// different slider from media volume and can sit at zero while everything
    /// else about the app is healthy.
    /// The hardware output level, 0…1. In `.voiceChat` this is the CALL volume.
    static var outputVolume: Float { AVAudioSession.sharedInstance().outputVolume }

    static var routeSummary: String {
        let s = AVAudioSession.sharedInstance()
        let out = s.currentRoute.outputs.map(\.portType.rawValue).joined(separator: "+")
        return "out=\(out.isEmpty ? "none" : out) vol=\(String(format: "%.2f", s.outputVolume)) "
             + "cat=\(s.category.rawValue.replacingOccurrences(of: "AVAudioSessionCategory", with: "")) "
             + "mode=\(s.mode.rawValue.replacingOccurrences(of: "AVAudioSessionMode", with: ""))"
    }

    #if DEBUG
    /// Test seam. A fresh Simulator reports `undetermined`, so without this the
    /// GRANTED path — the steady state every operator is actually in after the
    /// first press, and the one where "Talk costs zero reconfigurations" has to
    /// hold — would never be exercised by a test.
    static var _micPermissionOverride: MicPermission?
    #endif

    /// Ask for the mic. Called at the POINT OF USE (the first Talk press), never
    /// on connect: a prompt for something the operator has not asked to do yet is
    /// both rude and, in a test host with no one to answer it, a hang.
    static func requestRecordPermission(_ done: @escaping (Bool) -> Void) {
        AVAudioApplication.requestRecordPermission { granted in
            Task { @MainActor in done(granted) }
        }
    }

    /// Talk pressed. NOT a reconfiguration: the session is already exactly where
    /// it needs to be, so in the steady state this only raises a flag.
    static func claimForPTT() throws {
        if !beginVoiceSession() {
            throw AudioCapture.AudioError.noInput
        }
        pttHolds = true
        onPTTClaim?(true)
    }

    /// Talk released. Deliberately touches NOTHING on the session — the operator
    /// is still listening to the robot's room, and re-applying the category here
    /// is precisely what used to kill the playback engine on release.
    static func releaseFromPTT() {
        pttHolds = false
        onPTTClaim?(false)
    }

    /// The voice plane went away for good. Only now do we let the session go.
    static func endVoiceSession() {
        guard sessionUp else { return }
        sessionUp = false
        pttHolds = false
        try? AVAudioSession.sharedInstance()
            .setActive(false, options: .notifyOthersOnDeactivation)
    }

    #if DEBUG
    /// Test seam only.
    static func _resetForTesting() {
        sessionUp = false; canRecord = false; pttHolds = false; reconfigures = 0
        _micPermissionOverride = nil
    }
    #endif
}

// MARK: - Engine self-healing

/// The three notifications that silently kill an `AVAudioEngine`, in one place so
/// capture and playback cannot drift in how they answer them.
///
/// None of these are hypothetical: a Bluetooth handset connecting mid-call, a
/// phone call, and Siri each post one, and before 2026-07-29 every one of them
/// ended the downlink for the rest of the app's life.
private enum EngineHealth {
    static func observe(rebuild: @escaping () -> Void,
                        hardReset: @escaping () -> Void) -> [NSObjectProtocol] {
        let c = NotificationCenter.default
        var tokens: [NSObjectProtocol] = []

        // The engine stopped itself because the hardware format changed. Note the
        // OBJECT is deliberately nil: the other engine on this session changing
        // configuration is the same hardware event, and is exactly the case that
        // matters here (Talk starting the capture engine).
        tokens.append(c.addObserver(forName: .AVAudioEngineConfigurationChange,
                                    object: nil, queue: .main) { _ in rebuild() })

        // A phone call / Siri took the session. `.ended` is our cue to take it
        // back; without this the app stays silent after the call.
        tokens.append(c.addObserver(forName: AVAudioSession.interruptionNotification,
                                    object: nil, queue: .main) { note in
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt ?? 0
            guard AVAudioSession.InterruptionType(rawValue: raw) == .ended else { return }
            MainActor.assumeIsolated { AudioSessionArbiter.reactivate() }
            rebuild()
        })

        // mediaserverd restarted: every AVAudio object is invalid, nothing can be
        // reused. Full teardown and rebuild from scratch.
        tokens.append(c.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification,
                                    object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { AudioSessionArbiter.reactivate() }
            hardReset()
        })

        // Headphones in/out, Bluetooth in/out. Usually accompanied by a
        // configuration change, but not always — rebuild() is cheap and a no-op
        // when the engine is already healthy.
        tokens.append(c.addObserver(forName: AVAudioSession.routeChangeNotification,
                                    object: nil, queue: .main) { _ in rebuild() })
        return tokens
    }
}

// MARK: - Capture (operator → robot speaker)

/// Taps the mic, converts to the robot's advertised format, and emits EXACTLY
/// frame-sized chunks. Partial frames are buffered, never padded — padding a frame
/// with zeros is an audible tick every 20 ms.
final class AudioCapture {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var outFormat: AVAudioFormat?
    private var carry = Data()
    private var frameBytes = 640
    private let onFrame: (Data, Float) -> Void
    private var running = false
    private var format = AudioFormat()
    /// Guards the notification path against re-entering itself: `engine.start()`
    /// can itself post a configuration change.
    private var healing = false
    private var observers: [NSObjectProtocol] = []
    /// How many times the tap had to be rebuilt under us. Surfaced for debugging
    /// — a climbing count with audio flowing is fine; a climbing count with
    /// silence points at the route, not the wire.
    private(set) var restarts = 0

    init(onFrame: @escaping (Data, Float) -> Void) {
        self.onFrame = onFrame
        observers = EngineHealth.observe(rebuild: { [weak self] in self?.heal(hard: false) },
                                         hardReset: { [weak self] in self?.heal(hard: true) })
    }

    deinit { observers.forEach { NotificationCenter.default.removeObserver($0) } }

    func reconfigure(to f: AudioFormat) {
        guard running else { format = f; return }
        stop()
        try? MainActor.assumeIsolated { try start(format: f) }
    }

    @MainActor
    func start(format f: AudioFormat) throws {
        guard !running else { return }
        // Asserts the ONE configuration is up; does not change it (see the
        // arbiter). Throws when the mic is unavailable so the button can say so.
        try AudioSessionArbiter.claimForPTT()
        format = f
        try build()
        running = true
    }

    /// Build the tap + converter against whatever the hardware is doing RIGHT NOW.
    /// Separate from `start` because the healing path re-runs exactly this.
    @MainActor
    private func build() throws {
        let input = engine.inputNode
        // The platform AEC. Must be set before the tap. Only toggled when it is
        // not already on — re-setting it reconfigures the IO unit, which posts a
        // configuration change, which is the churn this file exists to avoid.
        if !input.isVoiceProcessingEnabled {
            try? input.setVoiceProcessingEnabled(true)
        }

        let inFmt = input.outputFormat(forBus: 0)
        guard inFmt.sampleRate > 0 else { throw AudioError.noInput }
        guard let outFmt = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                         sampleRate: Double(format.rate),
                                         channels: AVAudioChannelCount(format.channels),
                                         interleaved: true) else { throw AudioError.badFormat }
        outFormat = outFmt
        converter = AVAudioConverter(from: inFmt, to: outFmt)
        frameBytes = format.frameSamples * format.channels * 2      // S16
        carry.removeAll(keepingCapacity: true)

        input.removeTap(onBus: 0)                  // idempotent; a stale tap throws
        input.installTap(onBus: 0, bufferSize: 1024, format: inFmt) { [weak self] buf, _ in
            self?.convert(buf)
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        guard running else { return }
        running = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        carry.removeAll(keepingCapacity: true)
        Task { @MainActor in AudioSessionArbiter.releaseFromPTT() }
    }

    /// A configuration change tore the tap down while the operator was mid-
    /// sentence. Put it back rather than going quiet for the rest of the session.
    private func heal(hard: Bool) {
        guard running, !healing else { return }
        healing = true
        defer { healing = false }
        if hard || !engine.isRunning {
            engine.stop()
            restarts += 1
            do {
                try MainActor.assumeIsolated { try build() }
            } catch {
                running = false                    // honest: the mic really is gone
                #if DEBUG
                print("[MABEL-AUDIO] capture heal FAILED: \(error)")
                #endif
            }
        }
    }

    private func convert(_ buf: AVAudioPCMBuffer) {
        guard let converter, let outFmt = outFormat else { return }
        let ratio = outFmt.sampleRate / buf.format.sampleRate
        let cap = AVAudioFrameCount(Double(buf.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: cap) else { return }

        var err: NSError?
        var fed = false
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return buf
        }
        guard err == nil, out.frameLength > 0,
              let ch = out.int16ChannelData else { return }

        let n = Int(out.frameLength) * Int(outFmt.channelCount)
        let level = rms(ch[0], count: Int(out.frameLength))
        carry.append(UnsafeBufferPointer(start: ch[0], count: n).withMemoryRebound(to: UInt8.self) {
            Data(buffer: $0)
        })

        // Emit whole frames only.
        while carry.count >= frameBytes {
            let frame = carry.prefix(frameBytes)
            carry.removeFirst(frameBytes)
            onFrame(Data(frame), level)
        }
    }

    private func rms(_ p: UnsafeMutablePointer<Int16>, count: Int) -> Float {
        guard count > 0 else { return 0 }
        var acc: Double = 0
        for i in 0..<count { let v = Double(p[i]) / 32768.0; acc += v * v }
        return Float(min(1, sqrt(acc / Double(count)) * 3))     // ×3: speech is quiet on a 0…1 bar
    }

    enum AudioError: LocalizedError {
        case noInput, badFormat
        var errorDescription: String? {
            switch self {
            case .noInput:   return "no audio input device"
            case .badFormat: return "unsupported capture format"
            }
        }
    }
}

// MARK: - PCM16 → Float32

/// The wire-to-engine sample conversion, as a PURE function so it can be pinned
/// against a known waveform in a test.
///
/// It lives outside `AudioPlayback` because this is the step that decides whether
/// the operator hears speech or static, and "I only hear white noise" is a report
/// no amount of staring at an audio graph resolves. A conversion you can feed a
/// 440 Hz sine and assert on is one you can stop suspecting.
///
/// `AVAudioPlayerNode` renders **Float32 non-interleaved** only. Handing it the
/// wire's interleaved S16 bytes "works" in the sense that nothing errors — and
/// plays the integer bit patterns reinterpreted as floats, which is loud white
/// static. That is the failure this converts away from, and the reason the
/// docstring is longer than the loop.
enum PCM16 {

    /// Below this RMS a frame is treated as the microphone's NOISE FLOOR and the
    /// make-up gain is not applied to it.
    ///
    /// −46 dBFS. A bar mic in a quiet room sits around −40 dBFS of hiss with
    /// nothing happening; multiplying that by a blanket make-up gain is how
    /// "quiet" became "audible white noise" (reported 2026-07-29). Speech clears
    /// this threshold by 20 dB or more, so gating on it costs nothing real.
    static let noiseFloor: Float = 0.005

    /// Fill `dst` (non-interleaved Float32 planes) from `pcm` (interleaved S16_LE).
    /// Returns false without writing anything if the input is unusable — the
    /// caller must then NOT schedule the buffer, because its memory is undefined.
    @discardableResult
    static func decode(_ pcm: Data, into dst: UnsafePointer<UnsafeMutablePointer<Float>>,
                       frames: Int, channels ch: Int, gain: Float) -> Bool {
        guard frames > 0, ch > 0, pcm.count >= frames * ch * 2 else { return false }
        return pcm.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else { return false }
            // loadUnaligned, NOT assumingMemoryBound: `Data` slices carry no
            // alignment guarantee, and binding a possibly-odd address to Int16 is
            // undefined behaviour even where the hardware tolerates the load.
            func sample(_ k: Int) -> Int16 {
                base.loadUnaligned(fromByteOffset: k * 2, as: Int16.self).littleEndian
            }
            // One pass to measure, so the gate decides on the whole frame rather
            // than per sample — a per-sample gate chops the quiet parts out of
            // speech and sounds like a broken squelch.
            var acc: Double = 0
            let n = frames * ch
            for k in 0..<n {
                let v = Double(sample(k)) / 32768.0
                acc += v * v
            }
            let rms = Float((acc / Double(n)).squareRoot())
            let g = rms < noiseFloor ? 1.0 : gain
            for c in 0..<ch {
                let out = dst[c]
                for i in 0..<frames {
                    // Clamp, don't wrap: Float32 outside ±1 is a full-scale click.
                    out[i] = max(-1, min(1, Float(sample(i * ch + c)) / 32768.0 * g))
                }
            }
            return true
        }
    }
}

// MARK: - Playback (robot mic → operator)

/// Schedules incoming frames on a running playhead.
///
/// `@MainActor` because every caller already is (`AudioStreamLink.decodeAudio`
/// and `teardown`), and because the session arbiter is — being explicit about it
/// removes the hops that would otherwise sit between a frame arriving and it
/// being scheduled.
@MainActor
final class AudioPlayback {
    private var engine = AVAudioEngine()
    private var node = AVAudioPlayerNode()
    private var fmt: AVAudioFormat?
    private var playhead: AVAudioFramePosition = 0
    private var healing = false
    private var observers: [NSObjectProtocol] = []

    /// Rebuilds forced on us by the OS (Talk starting the capture engine, a route
    /// change, an interruption). Read by the debug HUD and the duplex test: this
    /// climbing while frames keep playing is the fix WORKING; before 2026-07-29
    /// the first one of these was terminal.
    private(set) var restarts = 0
    /// Buffers actually handed to the player node. The one number that answers
    /// "is the operator hearing anything?" without a microphone.
    private(set) var scheduled = 0

    /// How far ahead of the render clock we allow the queue to run before
    /// resyncing. Below this we are just absorbing jitter; above it, the operator
    /// is hearing the past.
    private let maxLeadSeconds = 0.35

    /// Software make-up gain on the downlink, applied per sample with a hard clamp.
    ///
    /// Not a preference — a correction. The robot's bar mic runs quiet (measured
    /// against thor, 2026-07-29: true RMS 0.004…0.016), and it plays back through
    /// `.voiceChat`, where iOS routes output at the CALL volume rather than the
    /// media volume. Those two stack into "the app says it is playing and I can't
    /// hear anything".
    ///
    /// REDUCED from 2.5 and GATED (see `PCM16.decode`). A blanket +8 dB amplifies
    /// the microphone's NOISE FLOOR along with the speech, and on a bar mic in a
    /// quiet room the noise floor is most of what there is — which the operator
    /// hears, correctly, as hiss. Gain belongs on signal, not on silence.
    ///
    /// The clamp is what makes it safe: a loud transient saturates instead of
    /// wrapping (Float32 wrap is a full-scale click, not a clip).
    private let downlinkGain: Float = 1.8

    init() {
        observers = EngineHealth.observe(rebuild: { [weak self] in
                                             MainActor.assumeIsolated { self?.heal(hard: false) }
                                         },
                                         hardReset: { [weak self] in
                                             MainActor.assumeIsolated { self?.heal(hard: true) }
                                         })
    }

    deinit { observers.forEach { NotificationCenter.default.removeObserver($0) } }

    /// True when the graph is actually capable of making sound right now. The
    /// symptom this whole file is about was `fmt != nil` (so `ensure` returned
    /// early) with `engine.isRunning == false` — a state that used to be
    /// unobservable.
    var isLive: Bool { fmt != nil && engine.isRunning }

    /// Muted at the OPERATOR's end. Frames keep arriving and keep being counted
    /// — the meter still shows the robot's room is live — but nothing is
    /// scheduled, so the phone is silent within one 20 ms frame.
    ///
    /// This is belt to the source-mute's braces, and it is the half that must
    /// never fail: `audio_config` asks the ROBOT to stop sending, which saves
    /// bandwidth but depends on the robot honouring it. When it does not, a
    /// button labelled "Muted" over audible sound is the app lying about its own
    /// state (reported 2026-07-29).
    var muted = false {
        didSet {
            guard muted, muted != oldValue else { return }
            node.stop()                 // drop everything already queued
            playhead = node.lastRenderTime?.sampleTime ?? 0
        }
    }

    func play(pcm: Data, rate: Int, channels: Int) {
        guard !muted else { return }
        guard let f = ensure(rate: rate, channels: channels) else { return }
        let ch = Int(f.channelCount)
        let frames = pcm.count / (ch * 2)                 // wire is S16_LE interleaved
        guard frames > 0,
              let buf = AVAudioPCMBuffer(pcmFormat: f, frameCapacity: AVAudioFrameCount(frames))
        else { return }
        buf.frameLength = AVAudioFrameCount(frames)
        // Wire S16_LE interleaved → the node's Float32 DEINTERLEAVED. This
        // conversion is not optional: AVAudioPlayerNode renders Float32 only —
        // scheduling int16 buffers "works" and plays the bytes reinterpreted as
        // floats, i.e. loud white static. Latent until 2026-07-27 because the old
        // speaker bar's mic only ever sent zeros, and zeros are silence in ANY
        // sample format. (The web client does the same /32768 in audiolink.js.)
        guard let dst = buf.floatChannelData else { return }
        // SCHEDULE ONLY A BUFFER WE ACTUALLY FILLED. `AVAudioPCMBuffer` does not
        // promise zeroed memory, so a fill that bailed halfway and still got
        // scheduled would play whatever was in that allocation — which sounds
        // exactly like the white static this file already warns about, and would
        // be blamed on the sample format rather than on the early return.
        guard PCM16.decode(pcm, into: dst, frames: frames, channels: ch,
                           gain: downlinkGain) else { return }

        // Rule 3: never latest-wins. If we've fallen far behind, drop the WHOLE
        // backlog and resync — reordering or dribbling stale frames turns speech
        // into word salad. This is what the server and the web client do.
        if let t = node.lastRenderTime, t.isSampleTimeValid {
            let lead = Double(playhead - t.sampleTime) / f.sampleRate
            if lead > maxLeadSeconds || lead < 0 {
                node.stop(); node.play()
                playhead = t.sampleTime
            }
        }
        node.scheduleBuffer(buf, at: nil, options: [])
        scheduled &+= 1
        playhead += AVAudioFramePosition(frames)
        if !node.isPlaying { node.play() }
    }

    func stop() {
        node.stop()
        engine.stop()
        fmt = nil
        playhead = 0
    }

    private func ensure(rate: Int, channels: Int) -> AVAudioFormat? {
        // The format is right AND the graph is alive. `engine.isRunning` is the
        // half that used to be missing: a configuration change stops the engine
        // and breaks its connections while leaving `fmt` set, so this returned a
        // perfectly good format for a graph that could not make a sound.
        if let f = fmt, Int(f.sampleRate) == rate, Int(f.channelCount) == channels,
           engine.isRunning {
            return f
        }
        // The session must be in its final configuration BEFORE the engine
        // starts, or starting it now and configuring later is itself the
        // hardware-format change that tears this graph down.
        AudioSessionArbiter.beginVoiceSession()

        node.stop()
        engine.stop()
        playhead = 0
        // Float32 NON-interleaved — the only format AVAudioPlayerNode renders.
        // Int16 here is the white-static bug (see play()); the S16→Float32
        // conversion happens per-frame in play().
        guard let f = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: Double(rate),
                                    channels: AVAudioChannelCount(channels),
                                    interleaved: false) else { return nil }
        if node.engine == nil { engine.attach(node) }
        // Reconnect unconditionally: a configuration change severs connections
        // without detaching the node, so "already attached" says nothing about
        // whether it still reaches the mixer.
        engine.connect(node, to: engine.mainMixerNode, format: f)
        engine.prepare()
        do { try engine.start() } catch {
            #if DEBUG
            print("[MABEL-AUDIO] playback start FAILED: \(error)")
            #endif
            return nil
        }
        node.play()
        fmt = f
        playhead = node.lastRenderTime?.sampleTime ?? 0
        return f
    }

    /// Put the downlink back after the OS moved the ground under it.
    ///
    /// `hard` means mediaserverd restarted and every AVAudio object we hold is a
    /// dead handle — the engine and node have to be NEW, which is why neither is
    /// a `let`.
    private func heal(hard: Bool) {
        guard !healing else { return }
        guard let f = fmt else { return }        // never played yet: nothing to restore
        guard hard || !engine.isRunning else { return }   // already healthy
        healing = true
        defer { healing = false }
        restarts += 1
        if hard {
            engine.stop()
            engine = AVAudioEngine()
            node = AVAudioPlayerNode()
        }
        fmt = nil                                 // force ensure() to rebuild fully
        _ = ensure(rate: Int(f.sampleRate), channels: Int(f.channelCount))
        #if DEBUG
        print("[MABEL-AUDIO] playback healed (hard=\(hard)) restarts=\(restarts) live=\(isLive)")
        #endif
    }

    #if DEBUG
    /// Test seam: do to this engine EXACTLY what the OS does on a hardware-format
    /// change — stop it, sever the node's connection, and post the notification.
    ///
    /// This is the 2026-07-29 regression in one call. Before the fix, `fmt` stayed
    /// set so `ensure()` returned early forever and the operator heard nothing for
    /// the rest of the app's life; `AudioDuplexTests` pins that it now comes back.
    func _simulateConfigurationChange() {
        _simulateConfigurationChangeSilently()
        NotificationCenter.default.post(name: .AVAudioEngineConfigurationChange,
                                        object: engine)
    }

    /// The same damage with NO notification — the case where the observer cannot
    /// help and `ensure()`'s own `engine.isRunning` check is the entire defence.
    func _simulateConfigurationChangeSilently() {
        engine.stop()
        engine.disconnectNodeOutput(node)
    }
    #endif
}
