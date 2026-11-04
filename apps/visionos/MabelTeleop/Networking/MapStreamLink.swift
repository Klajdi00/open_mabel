//
//  MapStreamLink.swift
//  MabelTeleop
//
//  Live SLAM-map transport for visionOS — the same `/map` WebSocket the iOS app
//  and web console use (ws://<host>:9092/map). Heavy + only wanted while the
//  Navigate diorama is on screen, so it rides its OWN socket, opened lazily.
//
//  Wire protocol (BINARY, little-endian; see server/docs/data_streams.md). Each
//  .data message is ONE record, tagged by its first byte:
//    MESH  0x01  u8 type, u8 flags(bit0=final, bit1=has-indices), u16 pad,
//                u32 count, count×{f32 x,y,z; u8 r,g,b,a}  [+ u32 idxCount, u32×idx]
//    ODOM  0x02  u8 type, u8 pad×3, f32 x,y,z, f32 qx,qy,qz,qw
//    PATH  0x03  u8 type, u8 pad×3, u32 count, count×{f32 x,y,z}
//    GRID  0x04  u8 type, u8 flags, u16 pad, u32 w, u32 h, f32 res,
//                f32 ox,oy,oz,oyaw, u8×(w*h) cells (0=free…100=occ, 255=unknown)
//    CLOUD 0x05  like MESH (no indices) — the robot's live sensor view
//
//  All positions arrive in the ROS `odom` frame (x-fwd, y-left, z-up); this client
//  rebases them to RealityKit (y-up) as (x, z, -y), exactly like the RobotRig.
//

import Foundation
import RealityKit
import simd

/// A decoded vertex set in RealityKit space (mesh rendered as points, or a cloud).
struct LiveCloud {
    var points: [SIMD3<Float>] = []
    var colors: [SIMD3<Float>] = []   // 0…1 RGB
}

/// The 2D occupancy map, rebased to RealityKit space.
struct LiveGrid {
    var width = 0
    var height = 0
    var resolution: Float = 0.05
    var origin = SIMD3<Float>(repeating: 0)   // RealityKit pos of cell (0,0)
    var yaw: Float = 0
    var cells: [UInt8] = []
}

struct MapOdomPose {
    var position = SIMD3<Float>(repeating: 0)
    var orientation = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
    var yaw: Float = 0
}

/// Owns the live-map WebSocket; surfaces decoded mesh / cloud / grid / pose / path
/// on the main actor (so the RealityKit scene can be mutated directly).
@MainActor
final class MapStreamLink: NSObject, URLSessionWebSocketDelegate {

    var onMesh:  ((LiveCloud) -> Void)?
    var onCloud: ((LiveCloud) -> Void)?
    var onGrid:  ((LiveGrid)  -> Void)?
    var onPose:  ((MapOdomPose)  -> Void)?
    var onPath:  (([SIMD3<Float>]) -> Void)?

    private(set) var connected = false

    private var task: URLSessionWebSocketTask?
    private var session: URLSession!
    private var url: URL?
    private var want = false
    private var attempts = 0

    // Snapshot accumulators (mesh + cloud are chunked).
    private var pendingMesh = LiveCloud()
    private var pendingMeshIdx = [UInt32]()
    private var pendingCloud = LiveCloud()

    // ROS z-up → RealityKit y-up: rotate −90° about X (same as RobotRigEntity).
    private static let zUpToYUp = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))

    override init() {
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
    }

    // MARK: lifecycle

    func connect(_ url: URL?) {
        guard url != self.url || !want else { return }
        self.url = url
        want = url != nil
        attempts = 0
        if want { open() } else { teardown() }
    }

    func disconnect() { want = false; url = nil; teardown(); connected = false }

    private func open() {
        guard want, let base = url else { return }
        teardown()
        pendingMesh = LiveCloud(); pendingMeshIdx = []; pendingCloud = LiveCloud()
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false)
        var items = comps?.queryItems ?? []
        if !items.contains(where: { $0.name == "client" }) {
            items.append(URLQueryItem(name: "client", value: "visionpro"))
        }
        comps?.queryItems = items
        let t = session.webSocketTask(with: comps?.url ?? base)
        t.maximumMessageSize = 16 * 1024 * 1024     // mesh chunks ≈ 1 MB
        task = t
        t.resume()
        receive()
    }

    private func teardown() {
        task?.cancel(with: .goingAway, reason: nil); task = nil
    }

    private func dropped() {
        teardown(); connected = false
        guard want else { return }
        attempts += 1
        let delay = min(3.0, 0.4 * Double(attempts))
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in self?.open() }
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self else { return }
            Task { @MainActor in
                switch result {
                case .failure: self.dropped()
                case .success(let msg):
                    if case let .data(d) = msg { self.decode(d) }
                    self.connected = true; self.attempts = 0
                    self.receive()
                }
            }
        }
    }

    // MARK: decode

    private func decode(_ data: Data) {
        guard let type = data.first else { return }
        switch type {
        case 0x01: decodeVertices(data, into: &pendingMesh, idx: &pendingMeshIdx, isMesh: true)
        case 0x02: decodePose(data)
        case 0x03: decodePath(data)
        case 0x04: decodeGrid(data)
        case 0x05:
            // CLOUD (0x05) has no index buffer — feed a throwaway so the shared
            // reader's inout index param is satisfied.
            var noIdx = [UInt32]()
            decodeVertices(data, into: &pendingCloud, idx: &noIdx, isMesh: false)
        default: break
        }
    }

    /// Shared reader for MESH (0x01) + CLOUD (0x05): same 16-byte vertex layout,
    /// chunked (bit0 = final). MESH may carry a trailing index buffer (bit1).
    private func decodeVertices(_ data: Data, into acc: inout LiveCloud,
                                idx accIdx: inout [UInt32], isMesh: Bool) {
        guard data.count >= 8 else { return }
        let flags = data[data.startIndex + 1]
        let isFinal = (flags & 0x01) != 0
        let hasIdx = isMesh && (flags & 0x02) != 0
        let count: Int = data.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) }
        guard count >= 0, data.count >= 8 + count * 16 else { return }

        let base = UInt32(truncatingIfNeeded: acc.points.count)
        var pts = [SIMD3<Float>](); pts.reserveCapacity(count)
        var cols = [SIMD3<Float>](); cols.reserveCapacity(count)
        var idx = [UInt32]()
        let ok: Bool = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            var off = 8
            for _ in 0..<count {
                let x = raw.loadUnaligned(fromByteOffset: off, as: Float.self)
                let y = raw.loadUnaligned(fromByteOffset: off + 4, as: Float.self)
                let z = raw.loadUnaligned(fromByteOffset: off + 8, as: Float.self)
                let r = raw[off + 12], g = raw[off + 13], b = raw[off + 14]
                off += 16
                pts.append(SIMD3(x, z, -y))     // ROS z-up → RealityKit y-up
                cols.append(SIMD3(Float(r) / 255, Float(g) / 255, Float(b) / 255))
            }
            guard hasIdx else { return true }
            guard data.count >= off + 4 else { return false }
            let ic = Int(raw.loadUnaligned(fromByteOffset: off, as: UInt32.self)); off += 4
            guard ic >= 0, data.count >= off + ic * 4 else { return false }
            idx.reserveCapacity(ic)
            for _ in 0..<ic {
                idx.append((raw.loadUnaligned(fromByteOffset: off, as: UInt32.self)) &+ base); off += 4
            }
            return true
        }
        guard ok else { return }
        acc.points.append(contentsOf: pts)
        acc.colors.append(contentsOf: cols)
        if hasIdx { accIdx.append(contentsOf: idx) }
        guard isFinal else { return }
        let snap = acc
        acc = LiveCloud()
        if isMesh { accIdx = []; onMesh?(snap) } else { onCloud?(snap) }
    }

    private func decodePose(_ data: Data) {
        guard data.count >= 32 else { return }
        let f: [Float] = data.withUnsafeBytes { raw in
            (0..<7).map { raw.loadUnaligned(fromByteOffset: 4 + $0 * 4, as: Float.self) }
        }
        guard f.allSatisfy({ $0.isFinite }) else { return }
        let qRos = simd_quatf(ix: f[3], iy: f[4], iz: f[5], r: f[6])
        let c = Self.zUpToYUp
        let yaw = atan2(2 * (qRos.real * qRos.imag.z + qRos.imag.x * qRos.imag.y),
                        1 - 2 * (qRos.imag.y * qRos.imag.y + qRos.imag.z * qRos.imag.z))
        onPose?(MapOdomPose(position: SIMD3(f[0], f[2], -f[1]),
                         orientation: c * qRos * c.inverse, yaw: yaw))
    }

    private func decodePath(_ data: Data) {
        guard data.count >= 8 else { return }
        let count: Int = data.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) }
        guard count >= 0, data.count >= 8 + count * 12 else { return }
        var pts = [SIMD3<Float>](); pts.reserveCapacity(count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var off = 8
            for _ in 0..<count {
                let x = raw.loadUnaligned(fromByteOffset: off, as: Float.self)
                let y = raw.loadUnaligned(fromByteOffset: off + 4, as: Float.self)
                let z = raw.loadUnaligned(fromByteOffset: off + 8, as: Float.self)
                off += 12
                pts.append(SIMD3(x, z, -y))
            }
        }
        onPath?(pts)
    }

    private func decodeGrid(_ data: Data) {
        guard data.count >= 32 else { return }
        let (w, h): (Int, Int) = data.withUnsafeBytes {
            (Int($0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)),
             Int($0.loadUnaligned(fromByteOffset: 8, as: UInt32.self)))
        }
        guard w > 0, h > 0, data.count >= 32 + w * h else { return }
        let f: [Float] = data.withUnsafeBytes { raw in
            (0..<5).map { raw.loadUnaligned(fromByteOffset: 12 + $0 * 4, as: Float.self) }
        }
        guard f.allSatisfy({ $0.isFinite }) else { return }
        var cells = [UInt8](repeating: 0, count: w * h)
        cells.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { raw in
                if let s = raw.baseAddress { memcpy(dst.baseAddress!, s.advanced(by: 32), w * h) }
            }
        }
        // origin ROS odom (x,y,z up) → RealityKit (x, z, -y); yaw about +Y negated.
        onGrid?(LiveGrid(width: w, height: h, resolution: f[0],
                         origin: SIMD3(f[1], f[3], -f[2]), yaw: -f[4], cells: cells))
    }
}
