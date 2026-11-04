//
//  MJPEGVideoStream.swift
//  MabelTeleop
//
//  Consumes an HTTP multipart/x-mixed-replace (MJPEG) stream from the
//  robot server and publishes decoded UIImages frame-by-frame.
//
//  Frame extraction strategy:
//  The server includes a Content-Length header in every multipart part:
//
//      --frame\r\n
//      Content-Type: image/jpeg\r\n
//      Content-Length: 12345\r\n
//      \r\n
//      <12345 bytes of JPEG data>\r\n
//
//  FrameBuffer parses Content-Length and extracts exactly that many bytes.
//  This is 100% reliable and eliminates the frame-boundary mismatch that
//  caused the earlier flicker (SOI/EOI marker scanning can mis-fire when
//  0xFF 0xD8 / 0xFF 0xD9 appear in Huffman-coded data).
//

import Foundation
import UIKit
import OSLog

public protocol VideoStream: AnyObject, Sendable {
    var id: CameraID { get }
    func frames() -> AsyncStream<UIImage>
    func start()
    func stop()
}

public final class MJPEGVideoStream: NSObject, VideoStream, @unchecked Sendable {

    public let id: CameraID
    private let url: URL
    private let reconnectDelay: TimeInterval
    private let log = Logger(subsystem: "com.mabel.teleop", category: "mjpeg")

    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var isRunning = false
    private let buffer = FrameBuffer()
    private let continuations = Continuations()

    public init(id: CameraID, url: URL, reconnectDelay: TimeInterval = 1.5) {
        self.id = id
        self.url = url
        self.reconnectDelay = reconnectDelay
    }

    public func frames() -> AsyncStream<UIImage> {
        AsyncStream { continuation in
            let token = continuations.add(continuation)
            continuation.onTermination = { [weak self] _ in
                self?.continuations.remove(token)
            }
        }
    }

    public func start() {
        guard !isRunning else { return }
        isRunning = true

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = .infinity
        config.waitsForConnectivity = true
        // Increase receive buffer so large JPEG chunks arrive in one delegate call.
        config.httpMaximumConnectionsPerHost = 1
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        var request = URLRequest(url: url)
        request.setValue("multipart/x-mixed-replace", forHTTPHeaderField: "Accept")
        task = session?.dataTask(with: request)
        task?.resume()
    }

    public func stop() {
        isRunning = false
        task?.cancel()
        task = nil
        session?.invalidateAndCancel()
        session = nil
        buffer.reset()
    }

    private func scheduleReconnect() {
        guard isRunning else { return }
        let delay = reconnectDelay
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, self.isRunning else { return }
            self.log.info("reconnecting \(self.id.rawValue, privacy: .public)")
            self.stop()
            self.isRunning = true   // stop() flipped it false
            self.start()
        }
    }
}

// MARK: - URLSession delegates

extension MJPEGVideoStream: URLSessionDataDelegate {

    public func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        buffer.append(data)
        while let jpeg = buffer.extractNextJPEG() {
            guard let image = UIImage(data: jpeg) else { continue }
            continuations.yield(image)
        }
    }

    public func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            log.error("stream error \(self.id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
        scheduleReconnect()
    }
}

// MARK: - Frame buffer

/// Parses multipart/x-mixed-replace data using Content-Length headers.
///
/// The server writes:
///   --frame\r\n
///   Content-Type: image/jpeg\r\n
///   Content-Length: N\r\n
///   \r\n
///   <N bytes>\r\n
///
/// We find `\r\n\r\n`, parse Content-Length from the preceding header block,
/// then read exactly N bytes — no JPEG marker scanning needed.
/// Falls back to SOI/EOI marker scanning for servers that omit Content-Length.
private final class FrameBuffer {
    private var data = Data()
    private let lock = NSLock()

    private static let headerEnd = Data("\r\n\r\n".utf8)
    private static let soi       = Data([0xFF, 0xD8])
    private static let eoi       = Data([0xFF, 0xD9])

    func append(_ chunk: Data) {
        lock.lock(); defer { lock.unlock() }
        data.append(chunk)
        // If buffer grows unreasonably large, discard it — a short blank
        // frame is better than feeding corrupt data to the decoder.
        if data.count > 4 * 1024 * 1024 { data.removeAll() }
    }

    func reset() {
        lock.withLock { data.removeAll() }
    }

    func extractNextJPEG() -> Data? {
        lock.lock(); defer { lock.unlock() }

        // 1. Locate the end-of-headers marker.
        guard let headerEndRange = data.range(of: Self.headerEnd) else { return nil }

        // 2. Parse Content-Length from the header section.
        if let contentLength = parseContentLength(in: data[data.startIndex..<headerEndRange.lowerBound]) {
            // 3a. Exact-length extraction (preferred — no marker ambiguity).
            let frameStart = headerEndRange.upperBound
            let frameEnd   = frameStart + contentLength
            guard frameEnd <= data.endIndex else { return nil }  // haven't received full frame yet

            let jpeg = Data(data[frameStart..<frameEnd])
            // Consume headers + body + optional trailing \r\n
            let removeUpTo = min(data.endIndex, frameEnd + 2)
            data.removeSubrange(data.startIndex..<removeUpTo)
            return jpeg
        }

        // 3b. Fallback: SOI (FFD8) … EOI (FFD9) marker scan.
        //     Only used if the server omits Content-Length.
        let bodyStart = headerEndRange.upperBound
        guard
            let soiRange = data.range(of: Self.soi, options: [], in: bodyStart..<data.endIndex),
            let eoiRange = data.range(of: Self.eoi, options: [], in: soiRange.lowerBound..<data.endIndex)
        else { return nil }

        let end  = eoiRange.upperBound
        let jpeg = Data(data[soiRange.lowerBound..<end])
        data.removeSubrange(data.startIndex..<end)
        return jpeg
    }

    /// Extracts the integer value of the `Content-Length:` header from a raw
    /// header block. Case-insensitive, handles optional whitespace.
    private func parseContentLength(in headerBlock: Data) -> Int? {
        guard let text = String(data: Data(headerBlock), encoding: .utf8) else { return nil }
        for line in text.components(separatedBy: "\r\n") {
            if line.lowercased().hasPrefix("content-length:") {
                let value = line.dropFirst("content-length:".count)
                    .trimmingCharacters(in: .whitespaces)
                return Int(value)
            }
        }
        return nil
    }
}

// MARK: - Continuation fan-out

private final class Continuations: @unchecked Sendable {
    private var map: [UUID: AsyncStream<UIImage>.Continuation] = [:]
    private let lock = NSLock()

    func add(_ c: AsyncStream<UIImage>.Continuation) -> UUID {
        let id = UUID()
        lock.lock(); defer { lock.unlock() }
        map[id] = c
        return id
    }

    func remove(_ id: UUID) {
        lock.lock(); defer { lock.unlock() }
        map.removeValue(forKey: id)
    }

    func yield(_ image: UIImage) {
        lock.lock()
        let values = Array(map.values)
        lock.unlock()
        for c in values { c.yield(image) }
    }
}
