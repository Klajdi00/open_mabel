//
//  UDPVideoReceiver.swift
//  MabelTeleop
//
//  Latest-frame-wins JPEG video over UDP — the Swift twin of the bridge's
//  server/udp_video.py (and of the reference receiver in
//  vision_pro/docs/tools/exp_udp_video.py, which validates this protocol).
//
//  Why not the MJPEG/HTTP stream: MJPEG rides on TCP, so ONE lost packet
//  stalls the whole stream until the retransmit lands (head-of-line
//  blocking) and stale frames queue behind it — the dominant video latency
//  over Wi-Fi and Tailscale. Here each JPEG arrives as ≤1196-byte datagrams
//  (clears Tailscale's 1280 MTU unfragmented); a lost chunk costs exactly
//  that frame and the next renders on time. Latency = encode + wire + decode.
//
//  Protocol (server port 9091):
//    →  {"sub": ["main", …]}                every 1 s (subscribe/keepalive)
//    ←  {"map": {"main": 0, …}}             stream-name → u8 id
//    ←  [ "MV" | ver u8 | id u8 | seq u16 | chunk u16 | chunks u16 | ts u32 ]
//       + JPEG slice  (all big-endian; 14-byte header)
//
//  VideoFeed (bottom of file) is what views use: it prefers UDP and falls
//  back to the MJPEG/HTTP stream automatically when no datagrams arrive
//  (firewalled network), so video always works — UDP is purely an upgrade.
//

import Foundation
import CoreGraphics
import ImageIO
import Network
import OSLog

/// Wire codec of a UDP video stream. JPEG (:9091) is intra-only — each 'MV' frame is a
/// whole JPEG. H.264 (:9092) is inter-frame — each 'MV' frame is one Annex-B access unit
/// fed to VideoToolbox. Same datagram protocol; the codec only changes how a completed
/// frame is turned into a CGImage.
enum VideoCodec { case jpeg, h264 }

final class UDPVideoReceiver: @unchecked Sendable {

    // MARK: - Shared instance (the app talks to one robot at a time)

    private static var current: UDPVideoReceiver?
    private static let registryLock = NSLock()

    static func shared(host: String, port: UInt16, codec: VideoCodec = .jpeg) -> UDPVideoReceiver {
        registryLock.lock(); defer { registryLock.unlock() }
        if let r = current, r.host == host, r.port == port, r.codec == codec { return r }
        current?.shutdown()
        let r = UDPVideoReceiver(host: host, port: port, codec: codec)
        current = r
        return r
    }

    // MARK: - State

    let host: String
    let port: UInt16
    let codec: VideoCodec

    private let queue = DispatchQueue(label: "com.mabel.udp-video")
    private let decodeQueue = DispatchQueue(label: "com.mabel.udp-video.decode")
    private var conn: NWConnection?
    private var subTimer: DispatchSourceTimer?
    private var alive = true

    private let lock = NSLock()
    private var handlers: [String: [UUID: @Sendable (CGImage) -> Void]] = [:]
    private var lastFrame: [String: Date] = [:]

    private var idToStream: [UInt8: String] = [:]          // queue-confined
    private var partials: [UInt8: Partial] = [:]           // queue-confined
    private var pendingJPEG: [String: Data] = [:]          // lock-guarded
    private var decodeBusy: Set<String> = []               // lock-guarded
    private var h264Decoders: [String: H264Decoder] = [:]  // decodeQueue-confined
    private var lastSeq: [UInt8: UInt16] = [:]             // queue-confined: newest COMPLETED frame per stream

    private struct Partial {
        var seq: UInt16
        var chunks: [Data?]
        var got: Int
    }

    private static let headerSize = 14
    private let log = Logger(subsystem: "com.mabel.teleop", category: "UDPVideo")

    private init(host: String, port: UInt16, codec: VideoCodec) {
        self.host = host
        self.port = port
        self.codec = codec
        open()
        startSubTimer()
    }

    // MARK: - Handlers (what VideoFeed registers)

    func addHandler(stream: String,
                    _ handler: @escaping @Sendable (CGImage) -> Void) -> UUID {
        let token = UUID()
        lock.lock()
        handlers[stream, default: [:]][token] = handler
        lock.unlock()
        queue.async { [weak self] in self?.sendSubscribe() }   // don't wait 1 s
        return token
    }

    func removeHandler(stream: String, token: UUID) {
        lock.lock()
        handlers[stream]?.removeValue(forKey: token)
        if handlers[stream]?.isEmpty == true { handlers.removeValue(forKey: stream) }
        lock.unlock()
    }

    /// When the last UDP frame for `stream` arrived — VideoFeed's fallback signal.
    func lastFrameDate(stream: String) -> Date? {
        lock.lock(); defer { lock.unlock() }
        return lastFrame[stream]
    }

    // MARK: - Connection

    private func open() {
        guard alive else { return }
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        let c = NWConnection(host: NWEndpoint.Host(host),
                             port: NWEndpoint.Port(rawValue: port)!,
                             using: params)
        conn = c
        c.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.log.info("UDP video ready → \(self.host, privacy: .public):\(self.port)")
                self.sendSubscribe()
                self.receiveLoop(on: c)
            case .failed(let err):
                self.log.warning("UDP video failed: \(err.localizedDescription, privacy: .public) — retrying")
                c.cancel()
                self.queue.asyncAfter(deadline: .now() + 1.5) { [weak self] in self?.open() }
            default:
                break
            }
        }
        c.start(queue: queue)
    }

    private func shutdown() {
        alive = false
        subTimer?.cancel(); subTimer = nil
        conn?.cancel(); conn = nil
    }

    private func startSubTimer() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 1.0, repeating: 1.0)
        t.setEventHandler { [weak self] in self?.sendSubscribe() }
        t.resume()
        subTimer = t
    }

    private func sendSubscribe() {
        lock.lock()
        let streams = Array(handlers.keys)
        lock.unlock()
        guard !streams.isEmpty, let conn, conn.state == .ready,
              let data = try? JSONSerialization.data(withJSONObject: ["sub": streams])
        else { return }
        conn.send(content: data, completion: .idempotent)
    }

    // MARK: - Receive → reassemble (latest frame wins)

    private func receiveLoop(on c: NWConnection) {
        c.receiveMessage { [weak self] data, _, _, error in
            guard let self, self.alive else { return }
            if let data, !data.isEmpty { self.ingest(data) }
            if error == nil {
                self.receiveLoop(on: c)
            }
        }
    }

    private func ingest(_ data: Data) {
        let bytes = [UInt8](data.prefix(Self.headerSize))
        if bytes.first == UInt8(ascii: "{") {               // JSON map ack
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let map = obj["map"] as? [String: Int] {
                idToStream = Dictionary(uniqueKeysWithValues:
                    map.map { (UInt8($0.value & 0xFF), $0.key) })
            }
            return
        }
        guard bytes.count == Self.headerSize,
              bytes[0] == UInt8(ascii: "M"), bytes[1] == UInt8(ascii: "V"),
              bytes[2] == 1 else { return }
        let sid   = bytes[3]
        let seq   = UInt16(bytes[4]) << 8 | UInt16(bytes[5])
        let idx   = Int(UInt16(bytes[6]) << 8 | UInt16(bytes[7]))
        let count = Int(UInt16(bytes[8]) << 8 | UInt16(bytes[9]))
        // Cap is a sanity bound on a corrupt header, NOT a frame-size limit.
        // At 1182 B/chunk the old cap of 256 silently dropped every JPEG over
        // ~295 KB — exactly the high-res / high-quality head frames we want on a
        // LAN. 4096 chunks ≈ 4.6 MB covers any plausible single intra frame
        // while still rejecting a garbage count (the server seq is u16 so a
        // frame never legitimately needs more).
        guard count > 0, idx < count, count <= 4096 else { return }

        var p = partials[sid]
        // Wrap-aware "newer": forward u16 distance < 2^15. A newer frame
        // EVICTS an older partial — never wait for a frame time left behind.
        let newer = p == nil || (seq != p!.seq && (seq &- p!.seq) < 0x8000)
        if newer {
            p = Partial(seq: seq, chunks: [Data?](repeating: nil, count: count), got: 0)
        }
        guard var part = p, part.seq == seq, part.chunks.count == count,
              part.chunks[idx] == nil else { return }
        part.chunks[idx] = data.subdata(in: (data.startIndex + Self.headerSize)..<data.endIndex)
        part.got += 1
        if part.got == count {
            partials[sid] = nil
            var jpeg = Data(capacity: count * 1182)
            for c in part.chunks { jpeg.append(c!) }
            deliver(jpeg, sid: sid, seq: seq)
        } else {
            partials[sid] = part
        }
    }

    private func deliver(_ frame: Data, sid: UInt8, seq: UInt16) {
        guard let stream = idToStream[sid] else { return }
        // A gap = at least one frame lost or evicted half-assembled (one dropped chunk
        // is enough). Harmless for JPEG; for H.264 the decoder must resync on an IDR.
        let lost = lastSeq[sid].map { seq != $0 &+ 1 } ?? false
        lastSeq[sid] = seq

        if codec == .h264 {
            // FRESHNESS = A DECODED PICTURE, stamped in h264Decoder's output — NOT the
            // reassembly here. An H.264 stream that arrives but never decodes (no IDR
            // reaching us, a format the session rejects) used to count as healthy, so
            // VideoFeed never fell back to MJPEG and the head view stayed black forever.
            // Inter-frame: feed EVERY access unit, in order, on the serial decode queue.
            // No drop-while-busy (dropping a P-frame corrupts the GOP until the next IDR);
            // VideoToolbox hardware decode keeps up, and a lost UDP chunk already drops a
            // whole AU upstream — the next in-band IDR (~1 s) resyncs.
            decodeQueue.async { [weak self] in
                guard let dec = self?.h264Decoder(for: stream) else { return }
                if lost { dec.noteLoss() }
                dec.decode(accessUnit: frame)
            }
            return
        }
        lock.lock(); lastFrame[stream] = Date(); lock.unlock()

        // JPEG (intra-only): drop-to-latest — only ever decode the freshest frame.
        lock.lock()
        pendingJPEG[stream] = frame                 // newest replaces pending
        let shouldDecode = !decodeBusy.contains(stream)
        if shouldDecode { decodeBusy.insert(stream) }
        lock.unlock()
        guard shouldDecode else { return }          // decoder drains the newest

        decodeQueue.async { [weak self] in
            guard let self else { return }
            while true {
                self.lock.lock()
                guard let data = self.pendingJPEG.removeValue(forKey: stream) else {
                    self.decodeBusy.remove(stream)
                    self.lock.unlock()
                    return
                }
                let sinks = Array((self.handlers[stream] ?? [:]).values)
                self.lock.unlock()
                // Force the bitmap HERE, on the decode queue. CGImage is lazy by
                // default, so without this the JPEG decode is deferred to DRAW
                // time — i.e. it runs on the main thread once per frame, for
                // every tile. At 20+ fps of 960×540 that saturates main and the
                // whole UI (not just video) goes stuttery, which is the classic
                // "video is bad" symptom with a healthy network underneath.
                let opts: [CFString: Any] = [
                    kCGImageSourceShouldCacheImmediately: true,
                    kCGImageSourceShouldCache: true,
                ]
                if let src = CGImageSourceCreateWithData(data as CFData, nil),
                   let img = CGImageSourceCreateImageAtIndex(src, 0, opts as CFDictionary) {
                    for sink in sinks { sink(img) }
                }
            }
        }
    }

    /// One H264Decoder per stream (created on the decode queue, so no lock). Its output
    /// fans to the stream's current sinks — looked up live so handler changes are seen.
    private func h264Decoder(for stream: String) -> H264Decoder {
        if let d = h264Decoders[stream] { return d }
        let d = H264Decoder { [weak self] img in
            guard let self else { return }
            self.lock.lock()
            self.lastFrame[stream] = Date()            // a PICTURE, not just bytes
            let sinks = Array((self.handlers[stream] ?? [:]).values)
            self.lock.unlock()
            for sink in sinks { sink(img) }
        }
        h264Decoders[stream] = d
        return d
    }
}

// MARK: - VideoFeed — UDP-first with automatic MJPEG fallback

/// One camera stream for a view: subscribes over UDP immediately and starts
/// the legacy MJPEG/HTTP stream only if no datagrams arrive within 2.5 s
/// (e.g. a firewall eats UDP). When UDP frames resume, MJPEG is stopped.
final class VideoFeed: @unchecked Sendable {

    private let stream: String
    private let mjpegURL: URL?
    private let label: String
    private let onImage: @Sendable (CGImage) -> Void

    private let udpHost: String?
    private let udpPort: UInt16
    private let codec: VideoCodec

    private var receiver: UDPVideoReceiver?
    private var token: UUID?
    private var mjpeg: MJPEGTextureStreamer?
    private var watchdog: DispatchSourceTimer?
    private let lock = NSLock()
    private var started = false

    /// Transport selection (one source of truth for every tile + the head wrap):
    ///   • `host` set      → LAN UDP on `udpPort` with `codec` (H.264 :9092 / JPEG :9091)
    ///   • else            → MJPEG/HTTP only (demo / firewalled)
    /// MJPEG (`mjpegURL`) is the universal fallback whenever the primary goes stale.
    init(host: String?, udpPort: UInt16 = 9091, codec: VideoCodec = .jpeg,
         stream: String, mjpegURL: URL?, label: String,
         onImage: @escaping @Sendable (CGImage) -> Void) {
        self.udpHost = host
        self.udpPort = udpPort
        self.codec = codec
        self.stream = stream
        self.mjpegURL = mjpegURL
        self.label = label
        self.onImage = onImage
    }

    func start() {
        lock.lock(); defer { lock.unlock() }
        guard !started else { return }
        started = true

        if let host = udpHost {                       // LAN: UDP (H.264 :9092 or JPEG :9091)
            let r = UDPVideoReceiver.shared(host: host, port: udpPort, codec: codec)
            receiver = r
            token = r.addHandler(stream: stream) { [onImage] img in onImage(img) }
            startWatchdogLocked()
        } else {                                       // no UDP info → MJPEG only
            startMJPEGLocked()
        }
    }

    func stop() {
        lock.lock(); defer { lock.unlock() }
        started = false
        watchdog?.cancel(); watchdog = nil
        if let r = receiver, let token { r.removeHandler(stream: stream, token: token) }
        receiver = nil; token = nil
        mjpeg?.stop(); mjpeg = nil
    }

    private func startWatchdogLocked() {
        let t = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        t.schedule(deadline: .now() + 2.5, repeating: 1.5)
        t.setEventHandler { [weak self] in self?.checkFallback() }
        t.resume()
        watchdog = t
    }

    private func checkFallback() {
        lock.lock(); defer { lock.unlock() }
        guard started else { return }
        let last: Date
        if let r = receiver { last = r.lastFrameDate(stream: stream) ?? .distantPast }
        else { return }                                // MJPEG-only mode: nothing to watch
        let primaryFresh = Date().timeIntervalSince(last) < 2.5
        if primaryFresh {
            if mjpeg != nil {
                Logger(subsystem: "com.mabel.teleop", category: "VideoFeed")
                    .info("\(self.label, privacy: .public): primary video recovered — stopping MJPEG fallback")
                mjpeg?.stop(); mjpeg = nil
            }
        } else if mjpeg == nil {
            Logger(subsystem: "com.mabel.teleop", category: "VideoFeed")
                .info("\(self.label, privacy: .public): no primary frames — MJPEG fallback")
            startMJPEGLocked()
        }
    }

    private func startMJPEGLocked() {
        guard mjpeg == nil, let url = mjpegURL else { return }
        let s = MJPEGTextureStreamer(url: url, label: label) { [onImage] img in
            onImage(img)
        }
        s.start()
        mjpeg = s
    }
}
