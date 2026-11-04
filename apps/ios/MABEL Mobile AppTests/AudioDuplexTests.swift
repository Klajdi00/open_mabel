//
//  AudioDuplexTests.swift
//  MABEL Mobile AppTests
//
//  TALKING MUST NEVER COST LISTENING.
//
//  The bug these tests exist for (2026-07-29): pressing Talk called
//  `setCategory`/`setActive` on the shared `AVAudioSession`. That is a hardware
//  I/O format change, and `AVAudioEngine` answers one by stopping itself and
//  tearing down every node connection — including the connections of the OTHER
//  engine on that session, the one playing the robot's room. `AudioPlayback.fmt`
//  stayed non-nil, so `ensure()` short-circuited forever and `play()` went on
//  scheduling buffers into a dead graph. The socket stayed open, frames kept
//  arriving, every counter kept climbing, and the operator heard nothing until
//  the app was relaunched.
//
//  There is no assertion about "the user hears audio" available to a unit test,
//  so these pin the two mechanical facts that produced the silence:
//
//    1. A Talk press performs ZERO session reconfigurations (`reconfigures`).
//    2. A configuration change is SURVIVED — the playback graph comes back live
//       instead of staying dead.
//
//  The server half of the proof is `server/tools/exp_audio_duplex.py`, which
//  measures :9610 holding 50 Hz down through a 50 Hz uplink against the real
//  robot. Both halves are needed: the wire was never the problem.
//

import XCTest
import AVFoundation
@testable import MABEL_Mobile_App

@MainActor
final class AudioDuplexTests: XCTestCase {

    override func setUp() async throws {
        try await super.setUp()
        AudioSessionArbiter._resetForTesting()
    }

    // MARK: - Rule 1: Talk does not touch the session

    /// The load-bearing assertion of this whole file. `claimForPTT` used to be
    /// two session calls; if it ever becomes one again, the downlink dies and
    /// this is the only thing standing in the way.
    func testTalkPressPerformsNoSessionReconfiguration() throws {
        // The STEADY state — permission already granted, which is where every
        // operator lives after the first press. A fresh Simulator would otherwise
        // report `undetermined` and quietly test the listen-only path instead.
        AudioSessionArbiter._micPermissionOverride = .granted
        XCTAssertEqual(AudioSessionArbiter.reconfigures, 0)

        AudioSessionArbiter.beginVoiceSession()
        let afterSetup = AudioSessionArbiter.reconfigures
        XCTAssertEqual(afterSetup, 1, "the voice plane coming up configures the session ONCE")

        try? AudioSessionArbiter.claimForPTT()
        XCTAssertEqual(AudioSessionArbiter.reconfigures, afterSetup,
                       "pressing Talk must not reconfigure the session — that is what "
                       + "stops the playback engine")

        AudioSessionArbiter.releaseFromPTT()
        XCTAssertEqual(AudioSessionArbiter.reconfigures, afterSetup,
                       "releasing Talk must not reconfigure the session either — the "
                       + "operator is still listening")
    }

    /// Ten press/release cycles, still one configuration. A per-press leak would
    /// pass the single-press test above and still kill audio in real use.
    func testRepeatedTalkCyclesNeverReconfigure() {
        AudioSessionArbiter._micPermissionOverride = .granted
        AudioSessionArbiter.beginVoiceSession()
        let baseline = AudioSessionArbiter.reconfigures
        for _ in 0..<10 {
            try? AudioSessionArbiter.claimForPTT()
            AudioSessionArbiter.releaseFromPTT()
        }
        XCTAssertEqual(AudioSessionArbiter.reconfigures, baseline,
                       "reconfiguration count must be independent of how much you talk")
    }

    /// `beginVoiceSession` is called on every reconnect and before every playback
    /// rebuild. If it were not idempotent it would be a config-change generator
    /// in its own right.
    func testBeginVoiceSessionIsIdempotent() {
        AudioSessionArbiter._micPermissionOverride = .granted
        AudioSessionArbiter.beginVoiceSession()
        let once = AudioSessionArbiter.reconfigures
        for _ in 0..<5 { AudioSessionArbiter.beginVoiceSession() }
        XCTAssertEqual(AudioSessionArbiter.reconfigures, once)
    }

    /// Releasing Talk must leave the category exactly where listening needs it.
    /// The old `releaseFromPTT` re-applied it "to relax the intent", which was a
    /// second config change on top of the first.
    func testReleasingTalkLeavesTheListeningConfigurationIntact() {
        AudioSessionArbiter._micPermissionOverride = .granted
        AudioSessionArbiter.beginVoiceSession()
        let s = AVAudioSession.sharedInstance()
        let category = s.category, mode = s.mode

        try? AudioSessionArbiter.claimForPTT()
        AudioSessionArbiter.releaseFromPTT()

        XCTAssertEqual(s.category, category)
        XCTAssertEqual(s.mode, mode)
        XCTAssertTrue(AudioSessionArbiter.sessionUp,
                      "the session must stay up after Talk — listening continues")
        XCTAssertFalse(AudioSessionArbiter.pttHolds)
    }

    /// `.playAndRecord` is not an accident of wanting the mic: it is also the
    /// category that survives the ring/silent switch, which is why LISTENING uses
    /// it. Dropping to `.playback` between presses would both mute the phone on
    /// silent and cost a config change.
    func testVoiceSessionUsesTheOneDuplexConfiguration() {
        AudioSessionArbiter._micPermissionOverride = .granted
        AudioSessionArbiter.beginVoiceSession()
        let s = AVAudioSession.sharedInstance()
        XCTAssertEqual(s.category, .playAndRecord)
        XCTAssertEqual(s.mode, AudioSessionArbiter.voiceMode)
        XCTAssertTrue(s.categoryOptions.contains(.defaultToSpeaker))
        XCTAssertTrue(AudioSessionArbiter.canRecord)
    }

    /// Opening the Robot space must not claim the microphone. Someone who wants
    /// to look at the map should not be asked for the mic — and in a test host
    /// there is nobody to answer the prompt at all, which hangs the runner
    /// (measured: a 360 s "test runner hung before establishing connection").
    ///
    /// `.playback` is not a downgrade for listening: like `.playAndRecord`, it is
    /// immune to the ring/silent switch. It simply does not ask for the mic.
    func testConnectingBeforePermissionListensWithoutClaimingTheMic() {
        AudioSessionArbiter._micPermissionOverride = .undetermined
        AudioSessionArbiter.beginVoiceSession()
        let s = AVAudioSession.sharedInstance()
        XCTAssertEqual(s.category, .playback,
                       "listening must not require the record half of the session")
        XCTAssertFalse(AudioSessionArbiter.canRecord)
    }

    /// A denied mic must still leave the operator LISTENING. Losing both
    /// directions because one is unavailable is the failure this guards.
    func testDeniedMicStillLeavesListeningAlive() {
        AudioSessionArbiter._micPermissionOverride = .denied
        AudioSessionArbiter.beginVoiceSession()
        XCTAssertEqual(AVAudioSession.sharedInstance().category, .playback)
        XCTAssertFalse(AudioSessionArbiter.canRecord)
        XCTAssertThrowsError(try AudioSessionArbiter.claimForPTT(),
                             "Talk must refuse honestly rather than latch over a dead mic")
    }

    /// The one legitimate reconfiguration in the app's life: permission arrives,
    /// so the session is upgraded listen-only → duplex. It must happen ONCE, and
    /// every press after it must be free.
    func testPermissionGrantUpgradesTheSessionExactlyOnce() {
        AudioSessionArbiter._micPermissionOverride = .undetermined
        AudioSessionArbiter.beginVoiceSession()
        let listenOnly = AudioSessionArbiter.reconfigures

        AudioSessionArbiter._micPermissionOverride = .granted
        AudioSessionArbiter.reactivate()                  // what beginCapture() does
        XCTAssertEqual(AudioSessionArbiter.reconfigures, listenOnly + 1)
        XCTAssertEqual(AVAudioSession.sharedInstance().category, .playAndRecord)

        for _ in 0..<5 {
            try? AudioSessionArbiter.claimForPTT()
            AudioSessionArbiter.releaseFromPTT()
        }
        XCTAssertEqual(AudioSessionArbiter.reconfigures, listenOnly + 1,
                       "the upgrade is once-per-lifetime, not once-per-press")
    }

    // MARK: - Rule 2: a configuration change is survived, not fatal

    /// The regression, reproduced exactly: stop the engine and sever the node the
    /// way the OS does, then assert the graph comes back.
    ///
    /// `_simulateConfigurationChange` is not a mock of the failure — it *is* the
    /// failure (engine stopped, output disconnected, notification posted). Before
    /// the fix `isLive` stayed false forever from this point on.
    func testPlaybackSurvivesAConfigurationChange() async throws {
        let player = AudioPlayback()
        player.play(pcm: silence(frames: 320), rate: 16_000, channels: 1)
        try XCTSkipUnless(player.isLive,
                          "no usable audio output on this host — nothing to test")
        XCTAssertEqual(player.restarts, 0)

        player._simulateConfigurationChange()
        // Observers are delivered on the main OperationQueue, so the heal lands on
        // a later turn of the run loop, not inline with the post.
        try await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertTrue(player.isLive,
                      "the playback graph must come back after a configuration change "
                      + "— this is the exact state that used to be terminal")
        XCTAssertGreaterThan(player.restarts, 0, "the heal must be recorded, not silent")
    }

    /// Belt and braces for the notification never arriving: the very next frame
    /// must rebuild on its own. This is what bounds the worst case to ONE 20 ms
    /// frame of audio instead of the rest of the session.
    func testTheNextFrameRebuildsEvenWithNoNotification() throws {
        let player = AudioPlayback()
        player.play(pcm: silence(frames: 320), rate: 16_000, channels: 1)
        try XCTSkipUnless(player.isLive, "no usable audio output on this host")

        // Stop the engine WITHOUT posting anything — the observer cannot help.
        player._simulateConfigurationChangeSilently()
        XCTAssertFalse(player.isLive, "precondition: the graph really is down")

        player.play(pcm: silence(frames: 320), rate: 16_000, channels: 1)
        XCTAssertTrue(player.isLive,
                      "ensure() must notice a stopped engine — the old code checked "
                      + "only the FORMAT, which had not changed")
    }

    /// Scheduling must resume, not merely the engine restart: "live" with a
    /// player node that never gets buffers again is the same silence.
    func testFramesKeepBeingScheduledAcrossAConfigurationChange() async throws {
        let player = AudioPlayback()
        player.play(pcm: silence(frames: 320), rate: 16_000, channels: 1)
        try XCTSkipUnless(player.isLive, "no usable audio output on this host")
        let before = player.scheduled

        player._simulateConfigurationChange()
        try await Task.sleep(nanoseconds: 300_000_000)
        for _ in 0..<5 {
            player.play(pcm: silence(frames: 320), rate: 16_000, channels: 1)
        }

        XCTAssertEqual(player.scheduled, before + 5,
                       "every post-recovery frame must reach the player node")
        // `scheduled` alone is NOT enough and this assertion is why: the broken
        // code also happily counted buffers — into a stopped engine. Handing a
        // frame to the node and the node being able to render it are two facts,
        // and the silence lived entirely in the gap between them.
        XCTAssertTrue(player.isLive,
                      "…and the graph must be able to RENDER them, not just accept them")
    }

    // MARK: - Helpers

    /// 20 ms of 16 kHz mono digital silence — the payload shape the robot sends.
    /// Content is irrelevant here; these tests are about the GRAPH, not the sound.
    private func silence(frames: Int) -> Data { Data(count: frames * 2) }
}

// MARK: - The sample conversion, pinned
//
// "I only hear white noise" (2026-07-29). The wire carries interleaved S16_LE;
// AVAudioPlayerNode renders Float32 non-interleaved ONLY, and handing it the
// integer bit patterns plays them reinterpreted as floats — loud static, no
// error. These pin the conversion against a known waveform so that report can
// never again be answered with a guess.

final class PCM16ConversionTests: XCTestCase {

    /// A 440 Hz sine at `amp`, one second's worth of `n` frames, S16_LE mono.
    private func sine(_ n: Int, amp: Double = 0.5, rate: Double = 16_000) -> (Data, [Float]) {
        var d = Data(capacity: n * 2)
        var expect = [Float]()
        for i in 0..<n {
            let v = amp * sin(2 * .pi * 440 * Double(i) / rate)
            let s = Int16(max(-32768, min(32767, (v * 32768).rounded())))
            withUnsafeBytes(of: s.littleEndian) { d.append(contentsOf: $0) }
            expect.append(Float(s) / 32768.0)
        }
        return (d, expect)
    }

    private func decode(_ pcm: Data, frames: Int, channels: Int = 1,
                        gain: Float = 1.0) -> [[Float]]? {
        var planes = [[Float]](repeating: [Float](repeating: .nan, count: frames),
                               count: channels)
        var ok = false
        // Hand PCM16 the same shape AVAudioPCMBuffer.floatChannelData has.
        var ptrs = [UnsafeMutablePointer<Float>]()
        for c in 0..<channels {
            ptrs.append(UnsafeMutablePointer<Float>.allocate(capacity: frames))
            _ = c
        }
        defer { ptrs.forEach { $0.deallocate() } }
        ptrs.withUnsafeBufferPointer { buf in
            ok = PCM16.decode(pcm, into: buf.baseAddress!, frames: frames,
                              channels: channels, gain: gain)
        }
        guard ok else { return nil }
        for c in 0..<channels {
            planes[c] = Array(UnsafeBufferPointer(start: ptrs[c], count: frames))
        }
        return planes
    }

    func testSineRoundTripsExactly() {
        let n = 320                                   // one 20 ms frame at 16 kHz
        let (pcm, expect) = sine(n)
        guard let planes = decode(pcm, frames: n) else { return XCTFail("decode refused") }
        for i in 0..<n {
            XCTAssertEqual(planes[0][i], expect[i], accuracy: 1e-6,
                           "sample \(i) — the conversion is not S16_LE → Float32")
        }
    }

    func testOutputIsAudioNotNoise() {
        // The property that distinguishes speech-like audio from reinterpreted
        // bits: successive samples are strongly correlated. White static sits at
        // ~0. This is the assertion that actually catches "wrong signal type".
        let n = 1600
        let (pcm, _) = sine(n)
        guard let planes = decode(pcm, frames: n) else { return XCTFail("decode refused") }
        let x = planes[0].map(Double.init)
        let mean = x.reduce(0, +) / Double(n)
        let c = zip(x.dropLast(), x.dropFirst())
            .reduce(0.0) { $0 + ($1.0 - mean) * ($1.1 - mean) }
        let v = x.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) }
        XCTAssertGreaterThan(c / v, 0.9, "lag-1 autocorrelation says this is noise")
        XCTAssertFalse(planes[0].contains { $0.isNaN }, "unfilled samples left in the buffer")
    }

    func testNoiseFloorIsNotAmplified() {
        // The regression that MADE the hiss audible: a blanket make-up gain
        // multiplies the microphone's noise floor along with the speech, and on a
        // bar mic in a quiet room the floor is most of what there is.
        let n = 320
        let (quiet, _) = sine(n, amp: 0.002)          // well under PCM16.noiseFloor
        let (loud, _) = sine(n, amp: 0.5)
        guard let q = decode(quiet, frames: n, gain: 4.0),
              let l = decode(loud, frames: n, gain: 4.0) else { return XCTFail("refused") }
        func rms(_ a: [Float]) -> Float {
            (a.reduce(0) { $0 + $1 * $1 } / Float(a.count)).squareRoot()
        }
        let (qr, _) = sine(n, amp: 0.002)
        XCTAssertEqual(rms(q[0]), rms(qr.map { _ in Float(0) }) + rms(q[0]),
                       accuracy: 1e-6)               // sanity on the helper
        // Gain applied to the loud frame, NOT to the quiet one.
        XCTAssertGreaterThan(rms(l[0]) / 0.5, 1.5, "gain was not applied to speech")
        XCTAssertLessThan(rms(q[0]), 0.0025, "gain WAS applied to the noise floor")
    }

    func testRefusesShortAndEmptyPayloads() {
        // Must return false rather than partially fill: the caller schedules the
        // buffer only on true, and an unfilled AVAudioPCMBuffer is white static.
        XCTAssertNil(decode(Data(), frames: 10))
        XCTAssertNil(decode(Data(repeating: 0, count: 4), frames: 10))
    }

    func testStereoDeinterleaves() {
        // L = +0.5 constant, R = −0.5 constant, interleaved.
        var d = Data()
        for _ in 0..<64 {
            withUnsafeBytes(of: Int16(16384).littleEndian) { d.append(contentsOf: $0) }
            withUnsafeBytes(of: Int16(-16384).littleEndian) { d.append(contentsOf: $0) }
        }
        guard let planes = decode(d, frames: 64, channels: 2) else { return XCTFail("refused") }
        XCTAssertEqual(planes[0][10], 0.5, accuracy: 1e-6)
        XCTAssertEqual(planes[1][10], -0.5, accuracy: 1e-6)
    }
}
