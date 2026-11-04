//
//  AudioWireTests.swift
//  MABEL Mobile AppTests
//
//  The AUDIO 0x07 codec, pinned BYTE-FOR-BYTE against the server's own encoder.
//
//  This matters more than it looks. The server REFUSES a malformed or wrong-rate
//  frame rather than correcting it, and the only symptom of a codec bug is
//  silence — or, worse, 48 kHz PCM played at 16 kHz, which is an octave and a half
//  down out of a real speaker in a room with people. There is no partial credit and
//  no visible error; the frame just never plays.
//
//  The reference vectors below were produced by the SERVER's encoder:
//
//      cd server && python3 -c "
//      from map_stream import encode_audio; import struct
//      pcm = b''.join(struct.pack('<h', (i*37 % 2000) - 1000) for i in range(320))
//      print(encode_audio(pcm, seq=7, rate=16000, channels=1)[:16].hex())"
//      -> 0700010007000000803e000080020000        (656 bytes total)
//
//  If this test fails, the app and the robot disagree about the wire — fix the app.
//

import XCTest
@testable import MABEL_Mobile_App

final class AudioWireTests: XCTestCase {

    /// The exact PCM the Python reference above encoded: 320 samples of S16_LE.
    private func referencePCM() -> Data {
        var d = Data(capacity: 640)
        for i in 0..<320 {
            let v = Int16((i * 37) % 2000 - 1000)
            withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
        }
        return d
    }

    // MARK: - Byte-for-byte parity with server/map_stream.py

    func testHeaderMatchesTheServerEncoderExactly() {
        let f = AudioWire.encode(pcm: referencePCM(), seq: 7, rate: 16_000, channels: 1)
        let header = f.prefix(16).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(header, "0700010007000000803e000080020000",
                       "the app and server disagree about the AUDIO 0x07 header")
    }

    /// 16 kHz · mono · 20 ms = 320 samples = 640 B payload + 16 B header.
    func testFrameIsExactly656Bytes() {
        let f = AudioWire.encode(pcm: referencePCM(), seq: 7, rate: 16_000, channels: 1)
        XCTAssertEqual(f.count, 656)
    }

    func testPayloadIsAppendedVerbatim() {
        let pcm = referencePCM()
        let f = AudioWire.encode(pcm: pcm, seq: 7, rate: 16_000, channels: 1)
        XCTAssertEqual(f.suffix(from: f.startIndex + 16), pcm, "PCM must not be transformed")
    }

    /// Every header field is little-endian — a big-endian slip puts `rate` at
    /// 1.09 GHz and the frame is refused, silently.
    func testFieldsAreLittleEndian() {
        let f = AudioWire.encode(pcm: Data([1, 2]), seq: 0x01020304, rate: 48_000, channels: 2)
        let d = AudioWire.decode(f)
        XCTAssertEqual(d?.seq, 0x01020304)
        XCTAssertEqual(d?.rate, 48_000)
        XCTAssertEqual(d?.channels, 2)
        XCTAssertEqual(f[0], 0x07)          // tag
        XCTAssertEqual(f[1], 0)             // PCM16
    }

    // MARK: - Round trip

    func testRoundTripPreservesEverything() {
        let pcm = referencePCM()
        let f = AudioWire.encode(pcm: pcm, seq: 42, rate: 16_000, channels: 1)
        guard let d = AudioWire.decode(f) else { return XCTFail("decode failed") }
        XCTAssertEqual(d.seq, 42)
        XCTAssertEqual(d.rate, 16_000)
        XCTAssertEqual(d.channels, 1)
        XCTAssertEqual(d.codec, AudioWire.codecPCM16)
        XCTAssertEqual(d.pcm, pcm)
    }

    /// The server carries `rate` PER FRAME precisely because the mic and the
    /// client can legitimately differ. A decoder that assumes 16 kHz plays
    /// chipmunks; assert we read what was sent.
    func testRateIsReadFromTheFrameNotAssumed() {
        let f = AudioWire.encode(pcm: Data(count: 4), seq: 1, rate: 48_000, channels: 1)
        XCTAssertEqual(AudioWire.decode(f)?.rate, 48_000)
    }

    // MARK: - Malformed input is dropped, never half-played

    func testForeignTagIsRejected() {
        var f = AudioWire.encode(pcm: referencePCM(), seq: 1, rate: 16_000, channels: 1)
        f[0] = 0x05                                     // a CLOUD record
        XCTAssertNil(AudioWire.decode(f))
    }

    func testShortHeaderIsRejected() {
        XCTAssertNil(AudioWire.decode(Data([0x07, 0x00, 0x01])))
    }

    /// A truncated frame must be dropped whole — playing the surviving half is a
    /// click, and trusting nBytes past the end is a buffer overrun.
    func testTruncatedPayloadIsRejected() {
        let f = AudioWire.encode(pcm: referencePCM(), seq: 1, rate: 16_000, channels: 1)
        XCTAssertNil(AudioWire.decode(f.prefix(400)), "a short read must not decode")
    }

    func testEmptyDataIsRejected() {
        XCTAssertNil(AudioWire.decode(Data()))
    }

    // MARK: - seq is the ONLY gap detector (rule 2)

    /// A wrapping counter must keep incrementing across UInt32.max — a crash or a
    /// reset there is a gap the receiver reads as a lost frame.
    func testSeqWrapsCleanly() {
        var seq = UInt32.max
        let a = AudioWire.encode(pcm: Data(count: 2), seq: seq, rate: 16_000, channels: 1)
        seq &+= 1
        let b = AudioWire.encode(pcm: Data(count: 2), seq: seq, rate: 16_000, channels: 1)
        XCTAssertEqual(AudioWire.decode(a)?.seq, UInt32.max)
        XCTAssertEqual(AudioWire.decode(b)?.seq, 0)
    }

    // MARK: - The advertised format is adopted, not guessed (rule 1)

    func testFrameSamplesFollowTheAdvertisedFormat() {
        XCTAssertEqual(AudioFormat(rate: 16_000, channels: 1, frameMs: 20).frameSamples, 320)
        XCTAssertEqual(AudioFormat(rate: 48_000, channels: 1, frameMs: 20).frameSamples, 960)
        XCTAssertEqual(AudioFormat(rate: 16_000, channels: 1, frameMs: 10).frameSamples, 160)
    }

    /// 640 B is what a 16 kHz mono 20 ms frame must weigh — the number the server
    /// checks against.
    func testDefaultFormatIsTheDocumentedOne() {
        let f = AudioFormat()
        XCTAssertEqual(f.rate, 16_000)
        XCTAssertEqual(f.channels, 1)
        XCTAssertEqual(f.frameSamples * f.channels * 2, 640)
    }
}
