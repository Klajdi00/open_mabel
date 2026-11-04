//
//  UDPVideoReceiver.swift
//  MABEL Mobile App
//
//  Low-latency video over UDP — the iOS twin of the bridge's server/udp_video.py
//  and of vision_pro/.../UDPVideoReceiver.swift. UDP DROPS instead of queueing, so
//  there is no TCP head-of-line backlog (the dominant MJPEG latency over Wi-Fi):
//  a lost chunk costs exactly its frame and the next renders on time. Glass-to-glass
//  = encode + wire + decode.
//
//  This receiver is CODEC-AGNOSTIC: it reassembles each 'MV' frame and hands the
//  complete payload `Data` to per-stream handlers. On port 9092 the payload is one
//  H.264 access unit (Annex-B) → feed H264Decoder; on port 9091 it is a JPEG.
//
//  Protocol (see server/udp_video.py):
//    →  {"sub": ["main", …]}              every 1 s (subscribe / keepalive)
//    ←  {"map": {"main": 0, …}}           stream-name → u8 id
//    ←  [ "MV" | ver u8 | id u8 | seq u16 | chunk u16 | chunks u16 | ts u32 ] + payload
//       (all big-endian; 14-byte header; ≤1196-byte datagrams clear the 1280 MTU)
//

import Foundation
import Network
import OSLog

final class UDPVideoReceiver: @unchecked Sendable {

    // MARK: - Shared instances, keyed by host:port (9091 JPEG, 9092 H.264 are distinct)

    private static var instances: [String: UDPVideoReceiver] = [:]
    private static let registryLock = NSLock()

    static func shared(host: String, port: UInt16) -> UDPVideoReceiver {
        registryLock.lock(); defer { registryLock.unlock() }
        let key = "\(host):\(port)"
        if let r = instances[key] { return r }
        let r = UDPVideoReceiver(host: host, port: port)
        instances[key] = r
        return r
    }

    // MARK: - State

    let host: String
    let port: UInt16

    private let queue = DispatchQueue(label: "com.mabel.udp-video")
    private var conn: NWConnection?
    private var subTimer: DispatchSourceTimer?
    private var alive = true

    private let lock = NSLock()
    /// Sinks get each complete frame plus `lost`: true when at least one frame of this
    /// stream before it never completed (an H.264 decoder must then resync on an IDR).
    private var handlers: [String: [UUID: @Sendable (Data, Bool) -> Void]] = [:]   // lock-guarded
    private var lastFrame: [String: Date] = [:]                              // lock-guarded

    private var idToStream: [UInt8: String] = [:]          // queue-confined
    private var partials: [UInt8: Partial] = [:]           // queue-confined
    private var lastSeq: [UInt8: UInt16] = [:]             // queue-confined: newest COMPLETED frame per stream

    private struct Partial { var seq: UInt16; var chunks: [Data?]; var got: Int }

    private static let headerSize = 14
    private let log = Logger(subsystem: "com.mabel.app", category: "UDPVideo")

    private init(host: String, port: UInt16) {
        self.host = host
        self.port = port
        open()
        startSubTimer()
    }

    // MARK: - Handlers

    /// Register for complete frame payloads of `stream`. Returns a token to remove it.
    func addHandler(stream: String,
                    _ handler: @escaping @Sendable (Data, Bool) -> Void) -> UUID {
        let token = UUID()
        lock.lock()
        handlers[stream, default: [:]][token] = handler
        lock.unlock()
        queue.async { [weak self] in self?.sendSubscribe() }   // subscribe now, don't wait 1 s
        return token
    }

    func removeHandler(stream: String, token: UUID) {
        lock.lock()
        handlers[stream]?.removeValue(forKey: token)
        if handlers[stream]?.isEmpty == true { handlers.removeValue(forKey: stream) }
        lock.unlock()
    }

    /// When the last UDP frame for `stream` arrived — the MJPEG-fallback signal.
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
                self.sendSubscribe()
                self.receiveLoop(on: c)
            case .failed:
                c.cancel()
                self.queue.asyncAfter(deadline: .now() + 1.5) { [weak self] in self?.open() }
            default:
                break
            }
        }
        c.start(queue: queue)
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
            if error == nil { self.receiveLoop(on: c) }
        }
    }

    private func ingest(_ data: Data) {
        let bytes = [UInt8](data.prefix(Self.headerSize))
        if bytes.first == UInt8(ascii: "{") {                       // JSON map ack
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
        guard count > 0, idx < count, count <= 4096 else { return }   // reject garbage header

        var p = partials[sid]
        // Wrap-aware "newer": a newer frame EVICTS an older partial (never wait for a
        // frame time left behind). Forward u16 distance < 2^15 means "newer".
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
            var frame = Data(capacity: count * 1182)
            for c in part.chunks { frame.append(c!) }
            deliver(frame, sid: sid, seq: seq)
        } else {
            partials[sid] = part
        }
    }

    private func deliver(_ frame: Data, sid: UInt8, seq: UInt16) {
        guard let stream = idToStream[sid] else { return }
        // A gap = at least one frame lost or evicted half-assembled (one dropped 1.2 kB
        // chunk is enough). Harmless for JPEG; H.264 must skip to the next IDR.
        let lost = lastSeq[sid].map { seq != $0 &+ 1 } ?? false
        lastSeq[sid] = seq
        lock.lock()
        lastFrame[stream] = Date()
        let sinks = Array((handlers[stream] ?? [:]).values)
        lock.unlock()
        for sink in sinks { sink(frame, lost) }
    }
}
