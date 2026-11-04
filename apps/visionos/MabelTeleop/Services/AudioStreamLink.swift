//
//  AudioStreamLink.swift
//  MabelTeleop
//
//  The `:9610` VOICE plane — the one DUPLEX exception in the 96xx family: the
//  operator's mic goes UP to the robot's speaker, the robot's mic comes DOWN, on
//  one socket. Everything else in 96xx is read-only telemetry.
//
//  Wire (server/map_stream.py `encode_audio`, AUDIO 0x07, little-endian):
//      u8 tag=0x07 · u8 codec(0=PCM16) · u16 channels · u32 seq · u32 rate ·
//      u32 nBytes · payload            16 B header + interleaved S16_LE
//  Default: 16 kHz mono, 20 ms frames = 320 samples = 640 B = 656 B/frame @ 50 Hz.
//
//  Three rules the server ENFORCES — a frame that breaks one is REFUSED, not
//  corrected. They are the reason this file looks the way it does:
//
//  1. Send exactly the format `hello` advertises; never hardcode. The server will
//     not resample: 48 kHz played at 16 kHz is an octave and a half down out of a
//     real speaker in a room with people.
//  2. `seq` must increment. It is the ONLY gap detector. Audio is a SEQUENCE, so a
//     dropped frame is an audible click, not a stale render.
//  3. NEVER latest-wins the audio queue. Every other MABEL stream coalesces to the
//     newest frame; doing that to voice REORDERS SPEECH. On overflow, drop the
//     whole backlog and resync (what the server and the web client both do).
//
//  Heavy payloads never travel through `@Published` (MapStreamLink's rule — PCM is
//  heavier than geometry). Only the cheap facts the UI binds to are published.
//

import Foundation
import AVFoundation
import Combine

/// Format the robot advertises for the voice plane. Adopted from `hello`, never
/// assumed — see rule 1.
struct AudioFormat: Equatable {
    var rate: Int = 16_000
    var channels: Int = 1
    var frameMs: Int = 20
    /// Samples per frame, per channel.
    var frameSamples: Int { rate * frameMs / 1000 }
}

/// Rolling level history for the on-screen scope — the operator's own mic going
/// UP, and the robot's room coming DOWN. Both are 0…1, newest sample last.
///
/// This is a DEBUG-first surface: it makes the two failure modes that are
/// otherwise invisible obvious at a glance. A flat DOWN lane while UP moves is the
/// robot bar's physical mute (peak=0). A flat UP lane while you're holding Talk is
/// a dead capture / denied permission. Numbers in a log can tell you that; a
/// moving graph tells you *without* asking.
struct AudioMeter: Equatable {
    /// ~3.4 s at the 14 Hz publish rate. SHORTER than the iPhone's 128: the VP
    /// scope sits in a narrow rail rather than across a full console row, and a
    /// 9 s window squeezed into it reads as noise.
    static let window = 48
    var up: [Float] = Array(repeating: 0, count: window)      // mic → robot
    var down: [Float] = Array(repeating: 0, count: window)    // robot → me
    var upNow: Float = 0
    var downNow: Float = 0

    mutating func push(up u: Float, down d: Float) {
        upNow = u; downNow = d
        self.up.removeFirst(); self.up.append(u)
        self.down.removeFirst(); self.down.append(d)
    }
}

/// The AUDIO 0x07 codec, kept PURE so it can be pinned byte-for-byte against the
/// Python encoder (`server/map_stream.py encode_audio`) in tests. A rate or layout
/// mismatch here is silent and awful — the server REFUSES the frame rather than
/// resampling it, so "no sound" is the only symptom.
enum AudioWire {
    static let tag: UInt8 = 0x07
    static let headerSize = 16
    static let codecPCM16: UInt8 = 0

    struct Frame: Equatable {
        var codec: UInt8
        var channels: Int
        var seq: UInt32
        var rate: Int
        var pcm: Data
    }

    static func encode(pcm: Data, seq: UInt32, rate: Int, channels: Int,
                       codec: UInt8 = codecPCM16) -> Data {
        precondition(channels >= 1, "channels must be >= 1")
        var out = Data(capacity: headerSize + pcm.count)
        out.append(tag)
        out.append(codec)
        withUnsafeBytes(of: UInt16(channels).littleEndian) { out.append(contentsOf: $0) }
        withUnsafeBytes(of: seq.littleEndian)              { out.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(rate).littleEndian)     { out.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(pcm.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(pcm)
        return out
    }

    /// nil for anything that isn't a well-formed AUDIO record — a short read or a
    /// foreign tag is dropped, never half-played.
    ///
    /// Reads the header IN PLACE. This runs 50×/s per direction, and the first cut
    /// copied the whole frame into a `[UInt8]` just to index 16 bytes of it.
    static func decode(_ d: Data) -> Frame? {
        guard d.count >= headerSize else { return nil }
        let s = d.startIndex
        guard d[s] == tag else { return nil }
        func u16(_ o: Int) -> UInt16 { UInt16(d[s + o]) | UInt16(d[s + o + 1]) << 8 }
        func u32(_ o: Int) -> UInt32 {
            UInt32(d[s + o]) | UInt32(d[s + o + 1]) << 8
                | UInt32(d[s + o + 2]) << 16 | UInt32(d[s + o + 3]) << 24
        }
        let n = Int(u32(12))
        guard n >= 0, d.count >= headerSize + n else { return nil }
        return Frame(codec: d[s + 1], channels: Int(u16(2)), seq: u32(4), rate: Int(u32(8)),
                     pcm: d.subdata(in: (s + headerSize)..<(s + headerSize + n)))
    }
}

@MainActor
final class AudioStreamLink: NSObject, ObservableObject, URLSessionWebSocketDelegate {

    // Cheap UI facts ONLY.
    @Published private(set) var connected = false
    @Published private(set) var talking = false
    /// 0…1, the operator's own mic. Published at ~8 Hz, NOT per frame: this is a
    /// meter for human eyes and the uplink runs at 50 Hz, so publishing every
    /// frame fired objectWillChange 50×/s — 50 chances/second to invalidate every
    /// view observing this link, for a bar that cannot move that fast anyway.
    @Published private(set) var micLevel: Float = 0
    /// Downlink seq discontinuities. Counted on every frame but PUBLISHED only
    /// when it actually changes (see decodeAudio) — a monotonically-restated
    /// counter would invalidate observers at the frame rate.
    @Published private(set) var gaps = 0
    @Published private(set) var notice: String?
    /// The rolling scope, PUBLISHED at 14 Hz by `meterTimer` — NOT per frame. The
    /// audio runs at 50 Hz each way, so publishing the graph on every frame would
    /// invalidate the view 100×/s for something a human eye samples far slower.
    @Published private(set) var meter = AudioMeter()

    // MARK: Duplex instrumentation
    //
    // Talking must never cost listening. Before 2026-07-29 pressing Talk killed
    // the playback engine and the ONLY evidence was the operator's ears — the
    // socket stayed open, frames kept arriving, `decodeAudio` kept running, and
    // every counter in this class kept climbing. These three make the failure
    // MEASURABLE, which is what lets `AudioDuplexTests` and the on-device harness
    // assert it rather than listen for it.

    /// Downlink AUDIO 0x07 frames decoded since the socket opened.
    private(set) var rxFrames = 0
    /// Uplink frames handed to the socket since the socket opened.
    private(set) var txFrames = 0
    /// True when the playback graph can actually make sound RIGHT NOW (format
    /// chosen AND engine running) — not merely "we have a format".
    var playbackLive: Bool { player.isLive }
    /// Buffers scheduled on the player node, and how many times the OS forced the
    /// graph to be rebuilt under us. A rebuild is normal; silence after one is not.
    var playbackScheduled: Int { player.scheduled }
    var playbackRestarts: Int { player.restarts }

    /// Silence the phone NOW, whatever the robot does.
    ///
    /// `audio_config` asks the robot to stop sending, which saves bandwidth but
    /// depends on the robot honouring it. This is the half that cannot fail: a
    /// button labelled "Muted" over audible sound is the app lying about its own
    /// state, and that is what the operator hit (2026-07-29).
    var muted: Bool {
        get { player.muted }
        set { player.muted = newValue }
    }

    /// Latest peaks, written on the audio path (every frame) and READ by the 14 Hz
    /// sampler — decouples the publish rate from the frame rate. Not @Published.
    private var lastTxPeak: Float = 0
    private var lastRxPeak: Float = 0
    private var meterTimer: Timer?
    private var meterTicks = 0

    private(set) var format = AudioFormat()

    private var endpoint: URL?
    private var task: URLSessionWebSocketTask?
    private var wantConnection = false
    private var reconnecting = false
    private var reconnectAttempts = 0
    private var pingTimer: Timer?

    private lazy var session: URLSession = {
        URLSession(configuration: .default, delegate: self, delegateQueue: .main)
    }()

    // MARK: - Endpoint lifecycle (MapStreamLink's idiom)

    /// Point the plane at a robot (nil → tear down). `hello.ports` can arrive
    /// AFTER the connection flips, so this is called again on serverMeta changes.
    func setEndpoint(_ url: URL?) {
        guard url != endpoint else { return }
        objectWillChange.send()
        endpoint = url
        rxSeq = nil                    // a new route re-earns the "reachable" verdict
        notice = nil
        #if DEBUG
        print("[MABEL-AUDIO] endpoint -> \(url?.absoluteString ?? "nil")")
        #endif
        reconcile()
    }

    private func reconcile() {
        if let _ = endpoint {
            wantConnection = true
            reconnectAttempts = 0
            // The ONE session configuration, brought up with the PLANE and held
            // for its whole life — never on the Talk press (AudioEngine.swift's
            // third rule). Doing it here also means the playback engine is built
            // against the format it will actually run at, instead of starting at
            // the app's default category and being torn down when PTT arrives.
            AudioSessionArbiter.beginVoiceSession()
            openSocket()
        } else {
            wantConnection = false
            teardown()
            connected = false
            // Last one out: only NOW does the session go, so a reconnect (which
            // runs teardown() too) never disturbs a live listen.
            AudioSessionArbiter.endVoiceSession()
        }
    }

    private func openSocket() {
        guard wantConnection, let base = endpoint else { connected = false; return }
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false)
        var items = comps?.queryItems ?? []
        if !items.contains(where: { $0.name == "client" }) {
            items.append(URLQueryItem(name: "client", value: "ios"))
        }
        comps?.queryItems = items
        teardown()
        reconnecting = false
        rxFrames = 0; txFrames = 0
        let t = session.webSocketTask(with: comps?.url ?? base)
        task = t
        t.resume()
        receive()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.task?.sendPing { _ in } }
        }
        startMeter()
    }

    /// Drive the scope at a fixed 14 Hz off the LATEST peaks, decoupled from the
    /// 50 Hz frames. Runs whenever the socket is up: the DOWN lane keeps ticking
    /// (with real levels, or flat when the mic is muted) even when we're not
    /// talking, which is exactly the state the operator wants to see. Peaks decay
    /// toward 0 so a stopped stream falls to the baseline instead of freezing.
    private func startMeter() {
        meterTimer?.invalidate()
        meterTicks = 0
        let t = Timer(timeInterval: 1.0 / 14.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.meter.push(up: self.lastTxPeak, down: self.lastRxPeak)
                self.micLevel = self.lastTxPeak
                self.lastTxPeak *= 0.55          // decay: the tx path only writes while talking
                self.lastRxPeak *= 0.55          // and the rx path only while frames arrive
                self.meterTicks &+= 1
                if self.meterTicks % 14 == 0 { self.checkAudibility() }
            }
        }
        RunLoop.main.add(t, forMode: .common)
        meterTimer = t
    }

    private func teardown() {
        pingTimer?.invalidate(); pingTimer = nil
        meterTimer?.invalidate(); meterTimer = nil
        task?.cancel(with: .goingAway, reason: nil); task = nil
        lastTxPeak = 0; lastRxPeak = 0
        meter = AudioMeter()
        stopTalking()
        player.stop()
    }

    private func dropped() {
        guard !reconnecting else { return }
        reconnecting = true
        teardown()
        connected = false
        guard wantConnection else { return }
        reconnectAttempts += 1
        // NEVER RECEIVED A FRAME after several tries = the plane isn't actually
        // reachable at this URL. Say so, rather than looping silently behind a
        // live-looking control.
        if rxSeq == nil, reconnectAttempts >= 4, notice == nil {
            notice = "Voice plane isn't answering — check the robot's audio bring-up."
        }
        let delay = min(3.0, 0.4 * Double(reconnectAttempts))
        Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.openSocket() }
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                webSocketTask: URLSessionWebSocketTask,
                                didOpenWithProtocol proto: String?) {
        Task { @MainActor in
            self.connected = true; self.reconnectAttempts = 0
            self.notice = nil                      // the route works after all
            #if DEBUG
            print("[MABEL-AUDIO] socket OPEN")
            #endif
        }
    }

    private func receive() {
        task?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .failure:
                    self.dropped()
                case .success(let msg):
                    switch msg {
                    case let .data(d):   self.decodeAudio(d)
                    case let .string(s): self.adoptHello(s)
                    @unknown default:    break
                    }
                    self.receive()
                }
            }
        }
    }

    /// The server's JSON `hello` carries the format to send in. ADOPT it (rule 1).
    private func adoptHello(_ s: String) {
        guard let d = s.data(using: .utf8),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
        else { return }
        var f = format
        if let r = j["rate"] as? Int { f.rate = r }
        if let c = j["channels"] as? Int { f.channels = c }
        if let m = j["frameMs"] as? Int { f.frameMs = m }
        #if DEBUG
        print("[MABEL-AUDIO] hello \(s.prefix(140))")
        #endif
        if f != format {
            format = f
            capture.reconfigure(to: f)
        }
    }

    // MARK: - AUDIO 0x07 (codec in `AudioWire`, kept pure + byte-pinned in tests)

    private var txSeq: UInt32 = 0
    private var rxSeq: UInt32?
    #if DEBUG
    private var rxCount = 0
    private var txCount = 0
    #endif

    private func encode(_ pcm: Data) -> Data {
        defer { txSeq &+= 1 }                        // rule 2: ALWAYS increments
        return AudioWire.encode(pcm: pcm, seq: txSeq, rate: format.rate,
                                channels: format.channels)
    }

    private func decodeAudio(_ d: Data) {
        guard let f = AudioWire.decode(d), !f.pcm.isEmpty else { return }
        // THE CODEC BYTE IS NOT DECORATION. `AudioWire.decode` reads it and, until
        // 2026-07-29, nobody checked it — so a frame in any other encoding (the
        // header reserves 1 for Opus) was rendered as if it were PCM16, which is
        // precisely white static. "The audio signal type is wrong" is the only
        // symptom, and it points at the sample conversion, which is innocent.
        // Refuse the frame and say so once, rather than play noise at someone.
        guard f.codec == AudioWire.codecPCM16 else {
            if notice == nil {
                notice = "MABEL is sending audio in a format this app can't play "
                    + "(codec \(f.codec)). Expected PCM16."
            }
            #if DEBUG
            print("[MABEL-AUDIO] REFUSED frame: codec=\(f.codec) (expected "
                  + "\(AudioWire.codecPCM16)) rate=\(f.rate) ch=\(f.channels)")
            #endif
            return
        }
        // Rule 2: seq is the only gap detector. Count discontinuities so "is it
        // dropping?" is answerable instead of guessable.
        if let last = rxSeq, f.seq != last &+ 1 { gaps &+= 1 }
        rxSeq = f.seq
        // Cheap RMS for the scope's DOWN lane (robot → me). Stored, not published:
        // the 14 Hz sampler reads it. This is what makes the bar's physical mute
        // VISIBLE — a flat down lane while the up lane moves.
        lastRxPeak = Self.rms(pcm: f.pcm)
        rxFrames &+= 1
        player.play(pcm: f.pcm, rate: f.rate, channels: f.channels)
        #if DEBUG
        rxCount &+= 1
        // Once a second. `live` and `restarts` are the two fields that make the
        // duplex bug legible in a log: `live=false` while rx climbs IS the
        // symptom — frames arriving into a graph that cannot play them.
        if rxCount % 50 == 1 {
            print("[MABEL-AUDIO] rx#\(rxCount) seq=\(f.seq) rate=\(f.rate) "
                  + "n=\(f.pcm.count) level=\(lastRxPeak) gaps=\(gaps) "
                  + "talking=\(talking) live=\(player.isLive) "
                  + "sched=\(player.scheduled) restarts=\(player.restarts) "
                  + AudioSessionArbiter.routeSummary)
        }
        #endif
    }

    /// Normalised RMS (0…1) of an S16_LE buffer, ×3 so quiet speech reads on a
    /// 0…1 meter. The capture side already scales the same way.
    private static func rms(pcm: Data) -> Float {
        let n = pcm.count / 2
        guard n > 0 else { return 0 }
        var acc: Double = 0
        pcm.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: Int16.self)
            for i in 0..<n { let v = Double(p[i]) / 32768.0; acc += v * v }
        }
        return Float(min(1, (acc / Double(n)).squareRoot() * 3))
    }

    // MARK: - Talk (uplink): capture → 16 kHz → 0x07 → :9610

    private lazy var capture = AudioCapture { [weak self] pcm, level in
        Task { @MainActor in
            guard let self, self.talking, let t = self.task else { return }
            self.lastTxPeak = level                 // → the scope's UP lane (14 Hz sampler)
            self.txFrames &+= 1
            #if DEBUG
            self.txCount &+= 1
            if self.txCount % 50 == 1 {
                print("[MABEL-AUDIO] tx#\(self.txCount) bytes=\(pcm.count + 16) "
                      + "level=\(level) rx=\(self.rxFrames) live=\(self.player.isLive)")
            }
            #endif
            t.send(.data(self.encode(pcm))) { err in
                if let err { print("[MABEL-AUDIO] tx FAILED: \(err)") }
            }
        }
    }

    private let player = AudioPlayback()

    /// Open the mic. **Must not disturb the downlink** — that is the whole point
    /// of the session arbiter: this installs a tap, it does not reconfigure
    /// anything, so the playback graph keeps running underneath.
    func startTalking() {
        guard task != nil, !talking else { return }
        switch AudioSessionArbiter.micPermission {
        case .denied:
            notice = micDeniedNotice
            talking = false
        case .undetermined:
            // The prompt belongs HERE, on the press that wants the mic — not on
            // connect, where it interrupts someone who only opened the map. The
            // latch closes when the answer arrives, so the press the operator
            // already made is the one that opens the mic; `talking` is
            // @Published precisely so the button can follow it asynchronously.
            AudioSessionArbiter.requestRecordPermission { [weak self] granted in
                guard let self else { return }
                if granted { self.beginCapture() } else { self.notice = self.micDeniedNotice }
            }
        case .granted:
            beginCapture()
        }
    }

    /// "The app is playing and I hear nothing" has one cause the app CAN see and
    /// should never make the operator guess at: the hardware volume is down.
    ///
    /// This is not hypothetical housekeeping. `.voiceChat` routes output at the
    /// CALL volume, which is a different slider from media volume and is not
    /// touched by anything else the operator does in this app — so it can sit near
    /// zero while every counter in this class is healthy. Only complain when we are
    /// genuinely playing audible frames, so the notice can only ever mean "turn it
    /// up", never "something is broken".
    private func checkAudibility() {
        guard connected, player.isLive, rxFrames > 50 else { return }
        let vol = AudioSessionArbiter.outputVolume
        let quiet = vol < 0.12
        if quiet, notice == nil {
            notice = "Volume is at \(Int(vol * 100))% — MABEL's room is streaming, "
                + "but the phone is turned down. Use the volume buttons while this screen is open."
        } else if !quiet, notice?.hasPrefix("Volume is at") == true {
            notice = nil                     // they turned it up; stop nagging
        }
    }

    private var micDeniedNotice: String {
        "Microphone access is off — you can hear MABEL, but not talk through it. "
            + "Enable it in Settings › Privacy › Microphone."
    }

    private func beginCapture() {
        // The ONE moment the session legitimately moves: permission has just been
        // granted, so the configuration goes from listen-only `.playback` up to
        // duplex `.playAndRecord`. It happens once in the app's life — every later
        // launch begins at `.playAndRecord` and every later press costs nothing —
        // and the playback engine rides even this one out through its own
        // configuration-change observer.
        //
        // `beginVoiceSession()`, NOT `reactivate()`. reactivate() forces
        // `sessionUp = false`, which defeats the idempotence check and re-applies
        // `setCategory` on EVERY press — i.e. the exact hardware-format churn this
        // file exists to prevent, tearing the playback graph down once per press
        // and leaving the downlink dependent on winning a heal race. Reserve
        // reactivate() for the case it was written for: the SYSTEM took the
        // session (interruption ended, media services reset).
        AudioSessionArbiter.beginVoiceSession()
        do {
            try capture.start(format: format)
            talking = true
            notice = nil
            #if DEBUG
            print("[MABEL-AUDIO] startTalking OK fmt=\(format) "
                  + "playbackLive=\(player.isLive) reconfigures=\(AudioSessionArbiter.reconfigures)")
            #endif
        } catch {
            notice = "Microphone unavailable: \(error.localizedDescription)"
            talking = false
            #if DEBUG
            print("[MABEL-AUDIO] startTalking FAILED: \(error)")
            #endif
        }
    }

    /// Close the mic. Listening continues — the session is untouched, and if the
    /// capture engine's teardown did perturb the shared hardware, the playback
    /// engine's own configuration-change observer has already put it back (and
    /// `ensure()` would rebuild on the very next 20 ms frame regardless).
    func stopTalking() {
        guard talking else { capture.stop(); return }
        talking = false
        micLevel = 0
        capture.stop()
        #if DEBUG
        print("[MABEL-AUDIO] stopTalking playbackLive=\(player.isLive) "
              + "restarts=\(player.restarts) reconfigures=\(AudioSessionArbiter.reconfigures)")
        #endif
    }
}
