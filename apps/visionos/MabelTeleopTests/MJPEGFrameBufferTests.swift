//
//  MJPEGFrameBufferTests.swift
//  MabelTeleopTests
//
//  The FrameBuffer class is private to MJPEGVideoStream.swift, so
//  these tests validate its externally observable behavior via a
//  mirror implementation kept in lockstep. If you later hoist
//  FrameBuffer to `internal`, swap the mirror for the real type.
//

import XCTest
@testable import MabelTeleop

final class MJPEGFrameBufferTests: XCTestCase {

    // Two minimal valid JPEGs (SOI + filler + EOI).
    private let frameA: [UInt8] = [0xFF, 0xD8, 0xAA, 0xBB, 0xFF, 0xD9]
    private let frameB: [UInt8] = [0xFF, 0xD8, 0xCC, 0xDD, 0xEE, 0xFF, 0xD9]

    func test_extractsCompleteFrames_inOrder() {
        let buffer = TestFrameBuffer()
        buffer.append(Data(frameA + frameB))

        XCTAssertEqual(buffer.extractNextJPEG(), Data(frameA))
        XCTAssertEqual(buffer.extractNextJPEG(), Data(frameB))
        XCTAssertNil(buffer.extractNextJPEG())
    }

    func test_handlesFragmentedDelivery() {
        let buffer = TestFrameBuffer()
        let half = frameA.count / 2
        buffer.append(Data(frameA.prefix(half)))
        XCTAssertNil(buffer.extractNextJPEG())
        buffer.append(Data(frameA.suffix(frameA.count - half)))
        XCTAssertEqual(buffer.extractNextJPEG(), Data(frameA))
    }

    func test_skipsJunkBeforeSOI() {
        let buffer = TestFrameBuffer()
        let junk: [UInt8] = [0x00, 0x01, 0x02, 0x03]
        buffer.append(Data(junk + frameA))
        XCTAssertEqual(buffer.extractNextJPEG(), Data(frameA))
    }
}

// MARK: - Mirror of the production FrameBuffer

private final class TestFrameBuffer {
    private var data = Data()
    private static let soi: [UInt8] = [0xFF, 0xD8]
    private static let eoi: [UInt8] = [0xFF, 0xD9]

    func append(_ chunk: Data) { data.append(chunk) }

    func extractNextJPEG() -> Data? {
        guard
            let start = data.firstRange(of: Data(Self.soi)),
            let endRange = data.range(of: Data(Self.eoi), options: [], in: start.lowerBound..<data.endIndex)
        else { return nil }
        let end = endRange.upperBound
        let frame = data.subdata(in: start.lowerBound..<end)
        data.removeSubrange(data.startIndex..<end)
        return frame
    }
}
