//
//  MapStreamLink.swift
//  MABEL Mobile App
//
//  Live SLAM-map transport. Separate from the 20 Hz teleop link (WebSocketLink):
//  the map stream is heavy (nvblox mesh) and only wanted while the Maps tab is
//  on screen, so it rides its OWN WebSocket — ws://<host>:9092/map — and is
//  opened lazily.
//
//  Wire protocol (BINARY frames, little-endian). Each WebSocket .data message is
//  ONE record, tagged by its first byte:
//
//    MESH  (0x01)  — a mesh snapshot (replaces whatever we hold):
//        u8   type      = 0x01
//        u8   flags     (bit0 = 1 → FINAL chunk of the snapshot; chunks with
//                                   bit0=0 are appended and the snapshot is
//                                   swapped in when the final chunk arrives. A
//                                   single self-contained snapshot sets bit0=1.
//                        bit1 = 1 → an index buffer FOLLOWS the vertices, so the
//                                   record is a real TRIANGLE MESH (nvblox ships
//                                   `triangles`); bit1=0 → vertices only → the
//                                   client renders a point cloud.)
//        u16  reserved  (0)
//        u32  vertexCount
//        vertexCount × {                         (16 bytes / vertex)
//            f32 x, f32 y, f32 z   // ROS/nvblox `odom` frame: x-fwd, y-left, z-up
//            u8  r, u8 g, u8 b, u8 a
//        }
//        — present only when flags bit1 = 1 (a triangle mesh): —
//        u32  indexCount        (= 3 × triangleCount)
//        indexCount × u32       // triangle indices into THIS chunk's vertices;
//                               // the client rebases them by the running vertex
//                               // count so chunked meshes stitch correctly.
//
//    ODOM  (0x02)  — robot pose in the SAME map frame as the mesh (`odom`):
//        u8   type      = 0x02
//        u8   reserved[3]
//        f32  x, f32 y, f32 z              // position, metres
//        f32  qx, f32 qy, f32 qz, f32 qw   // orientation (ROS order x,y,z,w)
//
//    PATH  (0x03)  — the Nav2 planned route, in the SAME `odom` frame as the mesh:
//        u8   type      = 0x03
//        u8   reserved[3]
//        u32  count                        // waypoints (0 → clear the route)
//        count × { f32 x, f32 y, f32 z }   // ROS odom, z-up (rebased like the mesh)
//
//  String (JSON) frames are tolerated and ignored (room for a future `hello`),
//  so the server can greet without breaking older clients.
//
//  The server side that produces these frames (ROS /nvblox_node/mesh_pointcloud
//  + /odom → /map) is implemented in server/map_stream.py and is built
//  separately; this client is the consumer. Callbacks fire on the main actor
//  (the delegate queue is .main, matching WebSocketLink), so the SceneKit
//  coordinator can mutate nodes directly.
//

import Foundation
import Combine
import SceneKit
import simd

/// A decoded mesh snapshot, already converted into SceneKit space (y-up) so the
/// scene can drop it straight into a geometry. Points and colours are parallel.
struct MapMesh {
    var points: [SCNVector3] = []      // SceneKit frame (x, z, -y of ROS)
    var colors: [SCNVector3] = []      // 0…1 RGB
    /// Triangle indices (3 per face) into `points`. Empty → render as a point
    /// cloud (a server that only has vertices); non-empty → a shaded surface mesh
    /// (the real nvblox mesh, which ships `triangles`).
    var indices: [Int32] = []
    var count: Int { points.count }
    var isMesh: Bool { indices.count >= 3 }
}

/// Robot pose in the map frame, converted to SceneKit conventions.
struct MapPose {
    var position: SCNVector3           // SceneKit frame
    var yaw: Float                     // heading about SceneKit +Y (rad)
    var orientation: simd_quatf        // full orientation, SceneKit frame
}

/// The Nav2 planned route, already rebased into SceneKit space (y-up). Empty →
/// the goal was reached / canceled; drop the drawn route.
struct MapPath {
    var points: [SCNVector3] = []      // SceneKit frame (x, z, -y of ROS)
}

/// The 2D occupancy map (RViz "Map"), GRID frame 0x04. Laid flat at y=origin.y in
/// SceneKit space. `cells` is row-major (x fastest, row 0 at the origin corner):
/// 0=free … 100=occupied, 255=unknown.
struct MapGrid {
    var width: Int = 0
    var height: Int = 0
    var resolution: Float = 0.05       // metres / cell
    var origin: SCNVector3 = .init()   // SceneKit pos of the cell-(0,0) corner
    var yaw: Float = 0                 // grid +x heading about SceneKit +Y (rad)
    var cells: [UInt8] = []
    var count: Int { width * height }
}

/// Owns the live-map WebSocket and surfaces the decoded mesh + robot pose.
///
/// Geometry never travels through `@Published` (a 100k-point mesh would thrash
/// SwiftUI): the SceneKit coordinator consumes it via the `onMesh` / `onPose`
/// callbacks and mutates nodes directly. The `@Published` fields are only the
/// lightweight stats the overlay chrome needs.
final class LiveMapStore: NSObject, ObservableObject, URLSessionWebSocketDelegate {

    /// True while the Maps tab is on screen and wants the stream. Toggling this
    /// connects / disconnects so we never stream mesh in the background.
    @Published var active = false { didSet { reconcile() } }

    /// Live readouts for the overlay.
    @Published private(set) var connected = false
    @Published private(set) var vertexCount = 0
    @Published private(set) var hasPose = false
    /// Has the SLAM plane ever delivered a non-empty 3D mesh / 2D occupancy grid?
    ///
    /// The console's layer chips read these so an EMPTY layer can dim itself and
    /// say why. Without them "Mesh" and "2D Map" are indistinguishable from broken
    /// buttons on any robot brought up without nvblox — which is the normal case on
    /// thor (`cuvslam:=false`), and is exactly how they were reported (2026-07-29).
    /// Latched, not live: a stream that has produced a map once is a stream with a
    /// map, and flickering the chip on every idle gap would be worse than useless.
    @Published private(set) var hasMesh = false
    @Published private(set) var hasGrid = false
    /// Last time a REAL map record (odom pose) arrived — the "is SLAM actually
    /// streaming" signal for the map-source chip. Throttled to ~1 Hz updates so
    /// the 20 Hz pose stream never thrashes SwiftUI. Demo never sets it.
    @Published private(set) var lastDataAt: Date?
    /// True when we're feeding a SYNTHETIC map (no robot/ROS — e.g. the Mac dev
    /// box). Lets the overlay label it honestly while still rendering.
    @Published private(set) var demo = false
    /// Whether there is anywhere to connect (nil → on-device demo / no robot).
    var hasEndpoint: Bool { endpoint != nil }

    /// The coordinator currently consuming the callbacks below. Rotation
    /// re-creates the viewer, and SwiftUI may dismantle the OLD view AFTER the
    /// NEW one wired itself — an unguarded teardown then nils the fresh hooks
    /// and every stream (mesh/pose/clouds) dies silently. Teardown must check
    /// ownership before releasing.
    weak var consumer: AnyObject?

    /// Heavy payloads go straight to the renderer, never through @Published.
    var onMesh: ((MapMesh) -> Void)?
    var onPose: ((MapPose) -> Void)?
    /// The live Nav2 route (PATH frame) — drawn on the map as the planned trajectory.
    var onPath: ((MapPath) -> Void)?
    /// The 2D occupancy map (GRID 0x04) — drawn flat under the mesh, like RViz.
    var onMap: ((MapGrid) -> Void)?
    /// The robot's live sensor pointcloud (CLOUD 0x05) — its CURRENT view, drawn as
    /// a transient overlay that's replaced every frame.
    var onCloud: ((MapMesh) -> Void)?
    /// Per-CAMERA pointclouds (96xx data plane, :9601–9604): the same CLOUD 0x05
    /// records on one dedicated WS per camera. Emits (stream key, cloud); an
    /// EMPTY cloud means "stream closed — clear the overlay".
    var onCameraCloud: ((String, MapMesh) -> Void)?

    /// The map WebSocket URL for the currently-connected robot (already carrying
    /// any `?key=` for a gated robot). nil disconnects.
    private(set) var endpoint: URL?

    private var task: URLSessionWebSocketTask?
    private var session: URLSession!
    private var pingTimer: Timer?
    private var wantConnection = false
    private var reconnectAttempts = 0
    private var reconnecting = false

    // Snapshot accumulation across multi-chunk meshes.
    private var pending = MapMesh()
    // Separate accumulator for the live sensor cloud (0x05), also chunked.
    private var pendingCloud = MapMesh()

    // ROS (z-up) → SceneKit (y-up): rotate −90° about X, matching the rest of
    // the app's MuJoCo→SceneKit convention (RealRobotView).
    private static let zUpToYUp = simd_quatf(angle: -.pi / 2, axis: simd_float3(1, 0, 0))

    override init() {
        super.init()
        // Delegate on the main queue, like WebSocketLink: decode + publish + the
        // SceneKit hand-off all happen on the main actor. The mesh is only 2 Hz.
        session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
    }

    // MARK: - Endpoint / lifecycle

    /// Point the stream at a new robot (or nil to detach). Reconnects if active.
    func setEndpoint(_ url: URL?) {
        guard url != endpoint else { return }
        objectWillChange.send()        // `endpoint`/`hasEndpoint` aren't @Published
        endpoint = url
        clearRender()                  // the previous robot's map must not persist
        reconcile()
    }

    /// Bring the link in line with `active` + `endpoint`.
    private func reconcile() {
        if active, endpoint != nil {
            // Real robot/bridge: stream over the socket.
            stopDemo()
            connected = false           // reflect the real link, not a stale demo
            wantConnection = true
            reconnectAttempts = 0
            openSocket()
        } else if active, endpoint == nil {
            // No robot reachable. v2: DON'T auto-play a synthetic map — offline
            // stays an honest empty state. The operator can explicitly load the
            // LOCAL demo map (console ▸ Map Source), which streams through the
            // same callbacks (nvblox-style mesh + occupancy + goal-driven robot).
            wantConnection = false
            teardown()
            if demoRequested { startDemo() } else { stopDemo(); connected = false }
        } else {
            // Tab off screen: stop everything.
            wantConnection = false
            teardown()
            stopDemo()
            connected = false
            setCameraClouds([:])       // tab off screen → drop the 96xx links too
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
        let url = comps?.url ?? base

        teardown()
        reconnecting = false
        pending = MapMesh()
        let t = session.webSocketTask(with: url)
        // Mesh chunks can approach ~1 MB; lift the 1 MiB default so a large frame
        // is never silently rejected (the server still chunks to stay well under).
        t.maximumMessageSize = 16 * 1024 * 1024
        task = t
        t.resume()
        receive()
        pingTimer = scheduled(5.0) { [weak self] in self?.task?.sendPing { _ in } }
    }

    private func teardown() {
        pingTimer?.invalidate(); pingTimer = nil
        task?.cancel(with: .goingAway, reason: nil); task = nil
    }

    /// Unexpected drop — reconnect (capped backoff) while the tab still wants it.
    private func dropped() {
        guard !reconnecting else { return }
        reconnecting = true
        teardown()
        connected = false
        guard wantConnection else { return }
        reconnectAttempts += 1
        let delay = min(3.0, 0.4 * Double(reconnectAttempts))
        Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.openSocket()
        }
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure:
                self.dropped()
            case .success(let msg):
                if case let .data(d) = msg { self.decode(d) }
                // String frames (a future JSON `hello`) are intentionally ignored.
                self.receive()
            }
        }
    }

    // MARK: - Per-camera cloud links (96xx data plane)

    // stream key → live socket / candidate URLS / rotation index / chunk
    // accumulator. Each enabled camera cloud is its own WS emitting the SAME
    // CLOUD 0x05 records as :9092. Multiple candidates (LAN + Tailscale) are
    // CYCLED on failure until one answers.
    private var camCloudTasks: [String: URLSessionWebSocketTask] = [:]
    private var camCloudWanted: [String: [URL]] = [:]
    private var camCloudIdx: [String: Int] = [:]
    private var camCloudPending: [String: MapMesh] = [:]
    /// 5 s keepalive per cloud socket — without it a silently-reaped link never
    /// reports a failure and the overlay just stops (see openCameraCloud).
    private var camCloudPings: [String: Timer] = [:]
    /// Has THIS host ever delivered points? Decides whether a drop rotates hosts.
    private var camCloudGotData: [String: Bool] = [:]

    /// Reconcile the per-camera cloud links to `urls` (stream → candidate WS
    /// URLs, tried in order). Removed streams close + clear their overlay;
    /// added streams open. Idempotent — safe to call on every toggle / route /
    /// connection change.
    func setCameraClouds(_ urls: [String: [URL]]) {
        for (stream, task) in camCloudTasks {
            let candidates = urls[stream] ?? []
            if let cur = task.originalRequest?.url, candidates.contains(cur) { continue }
            task.cancel(with: .goingAway, reason: nil)
            camCloudTasks[stream] = nil
            camCloudPending[stream] = nil
            camCloudPings[stream]?.invalidate()        // or the timer outlives its socket
            camCloudPings[stream] = nil
            camCloudGotData[stream] = nil
            onCameraCloud?(stream, MapMesh())          // clear the scene overlay
        }
        camCloudWanted = urls
        for (stream, candidates) in urls where camCloudTasks[stream] == nil && !candidates.isEmpty {
            camCloudIdx[stream] = 0
            openCameraCloud(stream)
        }
    }

    /// Force-close and reopen EVERY per-camera cloud link. Needed after the
    /// sim stack relaunches (scene switch): the URLs don't change, so the
    /// idempotent reconcile keeps the old sockets — which may be stalled on a
    /// connection the dead server never closed cleanly.
    func bounceCameraClouds() {
        for (stream, task) in camCloudTasks {
            task.cancel(with: .goingAway, reason: nil)
            camCloudTasks[stream] = nil
            camCloudPending[stream] = nil
            camCloudPings[stream]?.invalidate()
            camCloudPings[stream] = nil
            camCloudGotData[stream] = nil
            onCameraCloud?(stream, MapMesh())
        }
        for stream in camCloudWanted.keys {
            camCloudIdx[stream] = 0
            openCameraCloud(stream)
        }
    }

    private func openCameraCloud(_ stream: String) {
        guard let candidates = camCloudWanted[stream], !candidates.isEmpty else { return }
        let url = candidates[(camCloudIdx[stream] ?? 0) % candidates.count]
        let t = session.webSocketTask(with: url)
        t.maximumMessageSize = 16 * 1024 * 1024
        camCloudTasks[stream] = t
        camCloudPending[stream] = MapMesh()
        camCloudGotData[stream] = false
        t.resume()
        camCloudReceive(stream, t)
        // KEEPALIVE. The main map socket has always pinged every 5 s; these did
        // not, and that is why the clouds "stream for a while, then vanish": an
        // un-pinged WebSocket idling through NAT / Tailscale / a Wi-Fi roam gets
        // reaped by the middlebox, and the client is never told. The socket goes
        // half-open — no data, no failure callback, no retry — so the overlay
        // just stops updating and eventually clears. A ping makes the death
        // OBSERVABLE, which is what turns it into a reconnect.
        camCloudPings[stream]?.invalidate()
        camCloudPings[stream] = scheduled(5.0) { [weak self] in
            guard let self, let task = self.camCloudTasks[stream] else { return }
            task.sendPing { [weak self] err in
                guard err != nil else { return }
                // A failed ping IS the drop — surface it now rather than waiting
                // for a read that will never come.
                DispatchQueue.main.async { self?.camCloudDropped(stream, task) }
            }
        }
    }

    /// One place that handles a dead cloud link, from a read error OR a dead ping.
    private func camCloudDropped(_ stream: String, _ task: URLSessionWebSocketTask) {
        guard camCloudTasks[stream] === task else { return }      // already replaced
        let hadData = camCloudGotData[stream] ?? false
        camCloudPings[stream]?.invalidate(); camCloudPings[stream] = nil
        camCloudTasks[stream] = nil
        camCloudPending[stream] = nil
        onCameraCloud?(stream, MapMesh())
        // ROTATE ONLY AWAY FROM A HOST THAT NEVER WORKED. Rotating on EVERY
        // failure walked a proven-good host onto a dead candidate on the first
        // hiccup — e.g. off Tailscale onto a LAN IP with 100% packet loss, where
        // the retry then blocks on a long TCP timeout and the cloud is gone for
        // a minute. If this host was delivering points, it is reachable; retry it.
        if !hadData { camCloudIdx[stream] = (camCloudIdx[stream] ?? 0) + 1 }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self, self.camCloudWanted[stream] != nil,
                  self.camCloudTasks[stream] == nil else { return }
            self.openCameraCloud(stream)
        }
    }

    private func camCloudReceive(_ stream: String, _ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self, self.camCloudTasks[stream] === task else { return }
            switch result {
            case .failure:
                self.camCloudDropped(stream, task)
            case .success(let msg):
                if case let .data(d) = msg, d.first == 0x05 {
                    self.camCloudGotData[stream] = true    // this host is proven good
                    var pending = self.camCloudPending[stream] ?? MapMesh()
                    if let snapshot = Self.parseCloudChunk(d, into: &pending) {
                        self.camCloudPending[stream] = MapMesh()
                        self.onCameraCloud?(stream, snapshot)
                    } else {
                        self.camCloudPending[stream] = pending
                    }
                }
                self.camCloudReceive(stream, task)
            }
        }
    }

    // MARK: - Decoding

    private func decode(_ data: Data) {
        guard let type = data.first else { return }
        markAlive()
        switch type {
        case 0x01: decodeMesh(data)
        case 0x02: decodePose(data)
        case 0x03: decodePath(data)
        case 0x04: decodeGrid(data)
        case 0x05: decodeCloud(data)
        default: break
        }
    }

    /// CLOUD 0x05 — the robot's live sensor cloud. Same 16-byte vertex layout as
    /// MESH (no index buffer), chunked, swapped in on the final chunk. Rendered as
    /// a transient point overlay the scene replaces each frame.
    private func decodeCloud(_ data: Data) {
        if let snapshot = Self.parseCloudChunk(data, into: &pendingCloud) {
            pendingCloud = MapMesh()
            onCloud?(snapshot)
        }
    }

    /// One CLOUD 0x05 chunk: append its vertices to `pending`; on the FINAL
    /// chunk return the completed snapshot (caller resets its accumulator).
    /// Shared by the :9092 map stream and the per-camera :960x links — the
    /// wire framing is identical (server/map_stream.py).
    static func parseCloudChunk(_ data: Data, into pending: inout MapMesh) -> MapMesh? {
        guard data.count >= 8 else { return nil }
        let isFinal = (data[data.startIndex + 1] & 0x01) != 0
        let count: Int = data.withUnsafeBytes { raw in
            Int(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        }
        guard count >= 0, data.count >= 8 + count * 16 else { return nil }
        var pts = [SCNVector3](); pts.reserveCapacity(count)
        var cols = [SCNVector3](); cols.reserveCapacity(count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var off = 8
            for _ in 0..<count {
                let x = raw.loadUnaligned(fromByteOffset: off, as: Float.self)
                let y = raw.loadUnaligned(fromByteOffset: off + 4, as: Float.self)
                let z = raw.loadUnaligned(fromByteOffset: off + 8, as: Float.self)
                let r = raw[off + 12], g = raw[off + 13], b = raw[off + 14]
                off += 16
                pts.append(SCNVector3(x, z, -y))                 // ROS z-up → SceneKit y-up
                cols.append(SCNVector3(Float(r) / 255, Float(g) / 255, Float(b) / 255))
            }
        }
        pending.points.append(contentsOf: pts)
        pending.colors.append(contentsOf: cols)
        return isFinal ? pending : nil
    }

    /// GRID 0x04 — the 2D occupancy map. Header is 32 bytes, then width*height u8
    /// cells. The origin is converted ROS odom (z-up) → SceneKit (y-up).
    private func decodeGrid(_ data: Data) {
        guard data.count >= 32 else { return }
        let (w, h): (Int, Int) = data.withUnsafeBytes { raw in
            (Int(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self)),
             Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)))
        }
        guard w > 0, h > 0, data.count >= 32 + w * h else { return }
        let f: [Float] = data.withUnsafeBytes { raw in
            (0..<5).map { raw.loadUnaligned(fromByteOffset: 12 + $0 * 4, as: Float.self) }
        }
        guard f.allSatisfy({ $0.isFinite }) else { return }
        let res = f[0], ox = f[1], oy = f[2], oz = f[3], yawRos = f[4]
        var cells = [UInt8](repeating: 0, count: w * h)
        cells.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { raw in
                if let src = raw.baseAddress {
                    memcpy(dst.baseAddress!, src.advanced(by: 32), w * h)
                }
            }
        }
        // ROS odom (x,y,z, z-up) → SceneKit (x, z, -y). The grid lies in odom XY
        // (z=oz), i.e. SceneKit's XZ plane at y=oz. Yaw maps to SceneKit +Y
        // UNNEGATED: rows already advance toward scene −z (that quad direction
        // encodes the z-up→y-up flip), so R_y(+θ) reproduces ROS R_z(+θ) for
        // both the column (+x′) and row (+y′) axes. The old −θ rotated the grid
        // by −2θ relative to the mesh whenever the SLAM grid pose had yaw.
        hasGrid = w * h > 0
        onMap?(MapGrid(width: w, height: h, resolution: res,
                       origin: SCNVector3(ox, oz, -oy), yaw: yawRos, cells: cells))
    }

    private func decodeMesh(_ data: Data) {
        // Header: type(1) flags(1) reserved(2) count(4) = 8 bytes.
        guard data.count >= 8 else { return }
        let flags = data[data.startIndex + 1]
        let isFinal = (flags & 0x01) != 0
        let hasIndices = (flags & 0x02) != 0
        let count: Int = data.withUnsafeBytes { raw in
            Int(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        }
        guard count >= 0, data.count >= 8 + count * 16 else { return }

        var pts = [SCNVector3](); pts.reserveCapacity(count)
        var cols = [SCNVector3](); cols.reserveCapacity(count)
        var idx = [Int32]()
        // Triangle indices are LOCAL to this chunk's vertices; rebase them onto the
        // snapshot we're accumulating so multi-chunk meshes stitch correctly.
        let base = Int32(truncatingIfNeeded: pending.points.count)

        let ok: Bool = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Bool in
            var off = 8
            for _ in 0..<count {
                let x = raw.loadUnaligned(fromByteOffset: off, as: Float.self)
                let y = raw.loadUnaligned(fromByteOffset: off + 4, as: Float.self)
                let z = raw.loadUnaligned(fromByteOffset: off + 8, as: Float.self)
                let r = raw[off + 12], g = raw[off + 13], b = raw[off + 14]
                off += 16
                // ROS z-up (x,y,z) → SceneKit y-up (x, z, -y).
                pts.append(SCNVector3(x, z, -y))
                cols.append(SCNVector3(Float(r) / 255, Float(g) / 255, Float(b) / 255))
            }
            guard hasIndices else { return true }
            guard data.count >= off + 4 else { return false }
            let icount = Int(raw.loadUnaligned(fromByteOffset: off, as: UInt32.self))
            off += 4
            guard icount >= 0, data.count >= off + icount * 4 else { return false }
            idx.reserveCapacity(icount)
            for _ in 0..<icount {
                let v = raw.loadUnaligned(fromByteOffset: off, as: UInt32.self)
                off += 4
                idx.append(Int32(truncatingIfNeeded: v) &+ base)
            }
            return true
        }
        guard ok else { return }   // malformed index section → drop the frame

        pending.points.append(contentsOf: pts)
        pending.colors.append(contentsOf: cols)
        pending.indices.append(contentsOf: idx)
        guard isFinal else { return }
        let snapshot = pending
        pending = MapMesh()
        vertexCount = snapshot.count
        if snapshot.count > 0 { hasMesh = true }
        onMesh?(snapshot)
    }

    private func decodePose(_ data: Data) {
        // type(1) reserved(3) then 7 × f32 = 4 + 28 = 32 bytes.
        guard data.count >= 32 else { return }
        let f: [Float] = data.withUnsafeBytes { raw in
            (0..<7).map { raw.loadUnaligned(fromByteOffset: 4 + $0 * 4, as: Float.self) }
        }
        guard f.allSatisfy({ $0.isFinite }) else { return }   // ignore a garbled pose
        let posRos = simd_float3(f[0], f[1], f[2])
        let qRos = simd_quatf(ix: f[3], iy: f[4], iz: f[5], r: f[6])

        // Convert pose into SceneKit space (same −90°-about-X rebasing the rig uses).
        let c = Self.zUpToYUp
        let qScn = c * qRos * c.inverse
        let posScn = SCNVector3(posRos.x, posRos.z, -posRos.y)
        // Yaw about SceneKit +Y, for callers that only steer a ground robot.
        let yaw = atan2(2 * (qRos.real * qRos.imag.z + qRos.imag.x * qRos.imag.y),
                        1 - 2 * (qRos.imag.y * qRos.imag.y + qRos.imag.z * qRos.imag.z))

        hasPose = true
        let now = Date()
        if lastDataAt == nil || now.timeIntervalSince(lastDataAt!) > 1.0 {
            lastDataAt = now
        }
        // Latest odom in ODOM coordinates (x, y, yaw) — used e.g. to "cancel"
        // navigation by re-targeting the robot's own position.
        lastOdom = (Double(posScn.x), Double(-posScn.z), Double(yaw))
        onPose?(MapPose(position: posScn, yaw: yaw, orientation: qScn))
    }

    private func decodePath(_ data: Data) {
        // Header: type(1) reserved(3) count(4) = 8 bytes, then count × (3 × f32).
        guard data.count >= 8 else { return }
        let count: Int = data.withUnsafeBytes { raw in
            Int(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        }
        guard count >= 0, data.count >= 8 + count * 12 else { return }
        var pts = [SCNVector3](); pts.reserveCapacity(count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var off = 8
            for _ in 0..<count {
                let x = raw.loadUnaligned(fromByteOffset: off, as: Float.self)
                let y = raw.loadUnaligned(fromByteOffset: off + 4, as: Float.self)
                let z = raw.loadUnaligned(fromByteOffset: off + 8, as: Float.self)
                off += 12
                // ROS z-up (x,y,z) → SceneKit y-up (x, z, -y), same as the mesh.
                pts.append(SCNVector3(x, z, -y))
            }
        }
        navRouteActive = pts.count >= 2
        onPath?(MapPath(points: pts))
    }

    private func markAlive() {
        if !connected { connected = true; reconnectAttempts = 0 }
    }

    // MARK: - Demo map (no ROS / no robot)
    //
    // A synthetic "server": a procedural room that is REVEALED as a patrolling
    // robot drives through it (so the map visibly builds up, like real SLAM), with
    // an animated odom pose. Feeds `onMesh`/`onPose` exactly like the real decoder,
    // so the renderer is exercised end-to-end on a Mac with no ROS bridge.

    /// One triangle of the synthetic room (already in SceneKit y-up space).
    private struct DemoTri {
        let v: (SCNVector3, SCNVector3, SCNVector3)
        let c: (SCNVector3, SCNVector3, SCNVector3)
        let cx: Float, cz: Float        // centroid (for the reveal test)
    }

    private var demoTimer: Timer?
    private var demoTick = 0
    private var demoTris: [DemoTri] = []
    private var demoSeen = [Bool]()
    private var demoMesh = MapMesh()
    /// Local demo is OPT-IN (console ▸ Map Source), never automatic.
    private(set) var demoRequested = false
    /// Whether a Nav2 (or demo) route is currently active — drives the GO/STOP
    /// goal button (goal reached / cancelled → the route empties → false).
    @Published private(set) var navRouteActive = false
    /// Latest odom (x, y, yaw in the ODOM frame) — nil until a pose arrives.
    private(set) var lastOdom: (x: Double, y: Double, yaw: Double)? = nil
    // Goal-driven demo robot (scene coords; the operator's taps steer it).
    private var demoX: Float = 0
    private var demoZ: Float = 0
    private var demoYaw: Float = 0
    private var demoGoal: (x: Float, z: Float, finalYaw: Float)?
    // Occupancy revealed as the robot explores (matches the mesh reveal).
    private var demoGrid = MapGrid()
    private var demoKnown: [Bool] = []
    // On-device navigation (Models/MPPINavigator.swift): A* over the room's
    // occupancy + an MPPI local controller. The demo used to slide to the goal in a
    // straight line THROUGH the walls; now it plans and drives like the real stack.
    private var demoCostmap: NavCostmap?
    private var demoPath: [CGPoint] = []
    private var demoMPPI = MPPINavigator()
    /// Set when the operator taps a goal that cannot be reached (inside a wall).
    @Published private(set) var navGoalUnreachable = false
    /// The demo robot's lift height (m), pushed by `RobotController` from the
    /// commanded lift joints. The navigator's speed cap is the robot's REAL tip-over
    /// model evaluated here, so driving with the column up is slower — same as the
    /// machine. Plain stored property: it changes at joint-command rate, and routing
    /// it through `@Published` would churn SwiftUI for a number only the planner reads.
    var demoLiftHeight: Double = 0

    private static let demoHz = 20.0          // pose update rate
    private static let demoMeshEvery = 6      // emit a mesh snapshot every N ticks (~3 Hz)
    private static let demoSensorRange: Float = 4.2   // metres the "sensor" reveals around the robot
    private static let demoSpeed: Float = 0.8         // m/s toward the goal

    /// Explicitly load the LOCAL demo map. Streams the same record kinds a real
    /// robot sends — nvblox-style growing mesh, occupancy grid, odom, Nav2-like
    /// route — so the whole UI behaves identically offline.
    func startLocalDemo() {
        demoRequested = true
        objectWillChange.send()
        reconcile()
    }

    /// Tear the local demo down (back to the clean "connect a robot" state).
    func stopLocalDemo() {
        demoRequested = false
        objectWillChange.send()
        reconcile()
    }

    /// A tapped goal for the DEMO robot (odom coords, like `nav_goal`): it
    /// drives there at a walking pace, revealing map as it goes, and clears the
    /// route on arrival — the same lifecycle as a real Nav2 goal.
    func setDemoGoal(x: Double, y: Double, yaw: Double) {
        guard demo else { return }
        let gx = Float(x), gz = Float(-y)

        // PLAN it (A* over the room's occupancy, inflated by the robot radius) —
        // the route the operator sees is the route the robot will actually take.
        let costmap = demoCostmap ?? NavCostmap(grid: navGrid())
        demoCostmap = costmap
        let path = NavPlanner.plan(from: CGPoint(x: CGFloat(demoX), y: CGFloat(demoZ)),
                                   to: CGPoint(x: CGFloat(gx), y: CGFloat(gz)),
                                   costmap: costmap)
        guard !path.isEmpty else {
            // Tapped inside a wall / a sealed room: refuse honestly instead of
            // gliding through the geometry like the old straight-line demo.
            navGoalUnreachable = true
            demoGoal = nil
            demoPath = []
            navRouteActive = false
            onPath?(MapPath(points: []))
            return
        }
        navGoalUnreachable = false
        demoGoal = (gx, gz, Float(yaw))
        demoPath = path
        demoMPPI.reset()                       // fresh goal → don't inherit old momentum
        navRouteActive = true
        onPath?(MapPath(points: path.map { SCNVector3(Float($0.x), 0, Float($0.y)) }))
    }

    /// The demo occupancy in the planner's convention (`MapGrid` and `NavGrid` share it).
    private func navGrid() -> NavGrid {
        NavGrid(cells: demoGrid.cells, width: demoGrid.width, height: demoGrid.height,
                resolution: demoGrid.resolution,
                originX: Float(demoGrid.origin.x), originZ: Float(demoGrid.origin.z))
    }

    /// Abort the demo drive (the GO/STOP button's STOP): clears the goal, the
    /// route and the marker — the robot simply stays where it is.
    func cancelDemoGoal() {
        demoGoal = nil
        navRouteActive = false
        onPath?(MapPath(points: []))
    }

    /// SLAM restarted (locally for the demo, or optimistically while the server
    /// rebuilds): wipe the rendered map — mesh, grid, route — and, in the demo,
    /// re-explore from the robot's CURRENT pose (localization re-anchors there).
    func restartMapping() {
        if demo {
            demoSeen = [Bool](repeating: false, count: demoTris.count)
            demoMesh = MapMesh()
            demoKnown = [Bool](repeating: false, count: demoKnown.count)
            demoGoal = nil
        }
        clearRender()
    }

    /// The link to the robot is gone → the map must go with it.
    ///
    /// `clearRender()` only ever ran when the endpoint CHANGED (`setEndpoint`) or on a
    /// SLAM restart. An unexpected drop changes neither: the socket just retries the
    /// same URL underneath, so the last mesh and the last pointcloud stayed on screen
    /// indefinitely — indistinguishable from live data. Anything that decides "we are
    /// not talking to a host any more" calls this instead.
    func goOffline() {
        setCameraClouds([:])        // close the 96xx per-camera cloud links + clear them
        clearRender()               // mesh / grid / route / live cloud
        hasPose = false
        lastDataAt = nil            // the "SLAM is streaming" chip must read dead too
    }

    /// Wipe EVERY rendered map layer (mesh / occupancy grid / route / live
    /// cloud / per-camera clouds). Called when the STREAM SOURCE changes —
    /// robot switch, disconnect, SLAM restart — so a new world never renders
    /// underneath the previous robot's map. Empty frames double as "clear"
    /// all the way down (scene containers + minimap raster).
    private func clearRender() {
        vertexCount = 0
        navRouteActive = false
        onMesh?(MapMesh())
        onMap?(MapGrid())
        onPath?(MapPath(points: []))
        onCloud?(MapMesh())
        for stream in camCloudTasks.keys { onCameraCloud?(stream, MapMesh()) }
    }

    private func startDemo() {
        guard demoTimer == nil else { return }
        demo = true
        connected = true        // the overlay reads this as "streaming"
        hasPose = true
        if demoTris.isEmpty { demoTris = Self.buildDemoMesh() }
        demoSeen = [Bool](repeating: false, count: demoTris.count)
        demoMesh = MapMesh()
        (demoGrid, demoKnown) = Self.buildDemoGrid(from: demoTris)
        // The planner sees the WHOLE room (walls + furniture), not just the part the
        // robot has revealed: the demo's occupancy is ground truth, and a navigator
        // that forgot the walls behind it would happily reverse through them.
        demoCostmap = NavCostmap(grid: navGrid())
        demoPath = []
        demoMPPI.reset()
        navGoalUnreachable = false
        demoTick = 0
        demoX = 0; demoZ = 0; demoYaw = 0; demoGoal = nil
        vertexCount = 0
        demoTimer = scheduled(1.0 / Self.demoHz) { [weak self] in self?.demoStep() }
    }

    private func stopDemo() {
        demoTimer?.invalidate(); demoTimer = nil
        demoGoal = nil
        if demo { demo = false; hasPose = false; vertexCount = 0 }
    }

    private func demoStep() {
        demoTick += 1
        // Drive toward the operator's goal (stand still without one) — A* corridor,
        // MPPI local controller, obstacles respected. See Models/MPPINavigator.swift.
        if let g = demoGoal, let costmap = demoCostmap {
            let dt = Float(1.0 / Self.demoHz)
            let d = ((g.x - demoX) * (g.x - demoX) + (g.z - demoZ) * (g.z - demoZ)).squareRoot()
            if d < 0.12 {                          // arrived
                demoX = g.x; demoZ = g.z
                // Settle on the marker's aimed heading (arrow dir → nose −X).
                demoYaw = g.finalYaw + .pi
                demoGoal = nil
                demoPath = []
                navRouteActive = false
                onPath?(MapPath(points: []))      // goal reached → route + marker clear
            } else {
                // Speed cap comes from the SAME tip-over motion model the real robot
                // obeys, evaluated at the demo's CURRENT lift height — raise the lift
                // and the demo robot slows down exactly like the machine does.
                let caps = MotionModel.maxSpeedDir(demoLiftHeight)
                let vMax = min(Self.demoSpeed, Float(min(caps.0, caps.2)))
                let u = demoMPPI.step(x: demoX, z: demoZ, path: demoPath,
                                      goal: CGPoint(x: CGFloat(g.x), y: CGFloat(g.z)),
                                      costmap: costmap, vMax: vMax)
                let speed = (u * u).sum().squareRoot()
                if speed > 1e-4 {
                    demoX += u.x * dt
                    demoZ += u.y * dt
                    demoYaw = atan2(u.y, -u.x)     // nose (local −X) faces travel
                }
            }
        }
        let q = simd_quatf(angle: demoYaw, axis: simd_float3(0, 1, 0))
        // y = the base_link/free-joint height above the floor (rig.json no longer
        // bakes the website's feet-to-ground drop, so demo odom must supply it —
        // same value the sim publishes in its POSE stream).
        let pose = MapPose(position: SCNVector3(demoX, 0.078, demoZ), yaw: demoYaw, orientation: q)
        lastOdom = (Double(demoX), Double(-demoZ), Double(demoYaw))
        onPose?(pose)
        revealDemoTris(around: pose.position)
        if demoTick % Self.demoMeshEvery == 0 {
            vertexCount = demoMesh.count
            onMesh?(demoMesh)
            onMap?(revealedDemoGrid())
        }
    }

    /// Rasterise the demo room to an occupancy grid ONCE (0.1 m cells): walls +
    /// furniture → occupied, floor → free, outside the room → unknown.
    private static func buildDemoGrid(from tris: [DemoTri]) -> (MapGrid, [Bool]) {
        var g = MapGrid()
        g.resolution = 0.1
        g.width = 112; g.height = 112
        g.origin = SCNVector3(-5.6, 0, 5.6)   // cell (0,0) corner; +cols → +x, +rows → −z
        g.cells = [UInt8](repeating: 255, count: g.width * g.height)
        func cellIndex(_ x: Float, _ z: Float) -> Int? {
            let c = Int((x - (-5.6)) / g.resolution)
            let r = Int((5.6 - z) / g.resolution)
            guard c >= 0, c < g.width, r >= 0, r < g.height else { return nil }
            return r * g.width + c
        }
        // Free floor inside the room bounds…
        for r in 0..<g.height {
            for c in 0..<g.width {
                let x = -5.6 + (Float(c) + 0.5) * g.resolution
                let z = 5.6 - (Float(r) + 0.5) * g.resolution
                if abs(x) <= 5.0, abs(z) <= 5.0 { g.cells[r * g.width + c] = 0 }
            }
        }
        // …then stamp anything tall as occupied.
        for t in tris {
            let ys = [Float(t.v.0.y), Float(t.v.1.y), Float(t.v.2.y)]
            guard ys.contains(where: { $0 > 0.08 }) else { continue }
            for v in [t.v.0, t.v.1, t.v.2] {
                if let i = cellIndex(Float(v.x), Float(v.z)) { g.cells[i] = 100 }
            }
        }
        return (g, [Bool](repeating: false, count: g.cells.count))
    }

    /// Mark cells near the robot as explored and return the grid with everything
    /// still-unexplored masked to unknown — the grid grows like a real SLAM map.
    private func revealedDemoGrid() -> MapGrid {
        let r = Self.demoSensorRange
        let res = demoGrid.resolution
        let cr = Int(r / res)
        let cc = Int((demoX - (-5.6)) / res)
        let cRow = Int((5.6 - demoZ) / res)
        for dr in -cr...cr {
            for dc in -cr...cr {
                let rr = cRow + dr, c = cc + dc
                guard rr >= 0, rr < demoGrid.height, c >= 0, c < demoGrid.width else { continue }
                if Float(dr * dr + dc * dc) <= Float(cr * cr) { demoKnown[rr * demoGrid.width + c] = true }
            }
        }
        var out = demoGrid
        for i in 0..<out.cells.count where !demoKnown[i] { out.cells[i] = 255 }
        return out
    }

    /// Reveal (map) any not-yet-seen triangles whose centroid is within sensor
    /// range, appending them as a triangle soup so the surface mesh grows as the
    /// robot explores — exactly the snapshot shape a real nvblox stream sends.
    private func revealDemoTris(around p: SCNVector3) {
        let r2 = Self.demoSensorRange * Self.demoSensorRange
        let px = Float(p.x), pz = Float(p.z)
        for i in 0..<demoTris.count where !demoSeen[i] {
            let tri = demoTris[i]
            let dx = tri.cx - px, dz = tri.cz - pz
            if dx * dx + dz * dz <= r2 {
                demoSeen[i] = true
                let base = Int32(truncatingIfNeeded: demoMesh.points.count)
                demoMesh.points.append(contentsOf: [tri.v.0, tri.v.1, tri.v.2])
                demoMesh.colors.append(contentsOf: [tri.c.0, tri.c.1, tri.c.2])
                demoMesh.indices.append(contentsOf: [base, base + 1, base + 2])
            }
        }
    }

    /// A smooth figure-eight patrol on the floor, facing the direction of travel.
    private static func demoPose(t: Double) -> MapPose {
        let w = 2 * Double.pi / 26.0          // ~26 s loop
        let ax = 3.4, az = 3.4
        let x = ax * sin(w * t)
        let z = az * sin(2 * w * t) * 0.6
        let dx = ax * w * cos(w * t)          // velocity → heading
        let dz = az * 2 * w * cos(2 * w * t) * 0.6
        // SceneKit: the rig now applies the odom orientation with NO front offset, so
        // the model's front (local −X) faces wherever q·(−X) points. A +Y rotation by
        // atan2(dz, −dx) makes q·(−X) line up with the velocity vector (dx, dz), so the
        // demo robot still drives nose-first along its figure-eight.
        let yaw = Float(atan2(dz, -dx))
        let pos = SCNVector3(Float(x), Float(0), Float(z))
        let q = simd_quatf(angle: yaw, axis: simd_float3(0, 1, 0))
        return MapPose(position: pos, yaw: yaw, orientation: q)
    }

    /// Build a believable room ONCE as a triangle mesh: floor + four walls + a few
    /// furniture blocks, coloured by height with the `turbo` ramp (nvblox-style).
    /// Vertices are already in SceneKit y-up space, so they bypass the ROS→SceneKit
    /// rebasing the wire decoder does.
    private static func buildDemoMesh() -> [DemoTri] {
        var tris: [DemoTri] = []
        let halfW: Float = 5.0, halfD: Float = 5.0, wallH: Float = 2.6, s: Float = 0.5
        func col(_ y: Float) -> SCNVector3 { turbo(min(1, max(0, (y + 0.05) / (wallH + 0.2)))) }
        func addTri(_ a: SCNVector3, _ b: SCNVector3, _ c: SCNVector3) {
            let cx = (Float(a.x) + Float(b.x) + Float(c.x)) / 3
            let cz = (Float(a.z) + Float(b.z) + Float(c.z)) / 3
            tris.append(DemoTri(v: (a, b, c),
                                c: (col(Float(a.y)), col(Float(b.y)), col(Float(c.y))),
                                cx: cx, cz: cz))
        }
        func addQuad(_ p0: SCNVector3, _ p1: SCNVector3, _ p2: SCNVector3, _ p3: SCNVector3) {
            addTri(p0, p1, p2); addTri(p0, p2, p3)
        }
        // Floor.
        var x = -halfW
        while x < halfW - 1e-3 {
            var z = -halfD
            while z < halfD - 1e-3 {
                addQuad(SCNVector3(x, 0, z), SCNVector3(x + s, 0, z),
                        SCNVector3(x + s, 0, z + s), SCNVector3(x, 0, z + s))
                z += s
            }
            x += s
        }
        // Walls parallel to z (at x = ±halfW) and parallel to x (at z = ±halfD).
        func wallX(_ fx: Float) {
            var z = -halfD
            while z < halfD - 1e-3 {
                var y: Float = 0
                while y < wallH - 1e-3 {
                    addQuad(SCNVector3(fx, y, z), SCNVector3(fx, y, z + s),
                            SCNVector3(fx, y + s, z + s), SCNVector3(fx, y + s, z))
                    y += s
                }
                z += s
            }
        }
        func wallZ(_ fz: Float) {
            var x = -halfW
            while x < halfW - 1e-3 {
                var y: Float = 0
                while y < wallH - 1e-3 {
                    addQuad(SCNVector3(x, y, fz), SCNVector3(x + s, y, fz),
                            SCNVector3(x + s, y + s, fz), SCNVector3(x, y + s, fz))
                    y += s
                }
                x += s
            }
        }
        wallX(-halfW); wallX(halfW); wallZ(-halfD); wallZ(halfD)
        // Furniture blocks: a top face + four sides each.
        func box(_ bcx: Float, _ bcz: Float, _ bw: Float, _ bd: Float, _ bh: Float) {
            let x0 = bcx - bw / 2, x1 = bcx + bw / 2, z0 = bcz - bd / 2, z1 = bcz + bd / 2
            addQuad(SCNVector3(x0, bh, z0), SCNVector3(x1, bh, z0), SCNVector3(x1, bh, z1), SCNVector3(x0, bh, z1))
            addQuad(SCNVector3(x0, 0, z0), SCNVector3(x1, 0, z0), SCNVector3(x1, bh, z0), SCNVector3(x0, bh, z0))
            addQuad(SCNVector3(x1, 0, z1), SCNVector3(x0, 0, z1), SCNVector3(x0, bh, z1), SCNVector3(x1, bh, z1))
            addQuad(SCNVector3(x0, 0, z1), SCNVector3(x0, 0, z0), SCNVector3(x0, bh, z0), SCNVector3(x0, bh, z1))
            addQuad(SCNVector3(x1, 0, z0), SCNVector3(x1, 0, z1), SCNVector3(x1, bh, z1), SCNVector3(x1, bh, z0))
        }
        box(-2.2, -1.6, 1.4, 0.8, 0.75)   // table
        box( 2.4,  1.9, 2.0, 0.9, 0.50)   // couch
        box( 2.9, -2.8, 0.8, 0.8, 1.80)   // shelf
        return tris
    }

    /// Google "turbo" colormap (polynomial approximation) — perceptually even, the
    /// usual nvblox/SLAM height ramp. t in [0,1].
    private static func turbo(_ t: Float) -> SCNVector3 {
        let x = min(1, max(0, t))
        let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
        let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
        let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
        return SCNVector3(min(1, max(0, r)), min(1, max(0, g)), min(1, max(0, b)))
    }

    private func scheduled(_ interval: TimeInterval, _ block: @escaping () -> Void) -> Timer {
        let t = Timer(timeInterval: interval, repeats: true) { _ in block() }
        RunLoop.main.add(t, forMode: .common)
        return t
    }

    // MARK: - URLSessionWebSocketDelegate

    // The session is SHARED by the /map task and the per-camera cloud tasks —
    // route every callback by task identity. An unfiltered dropped() here let a
    // camera-cloud close (server restart, bounceCameraClouds' own cancels) tear
    // down the healthy /map socket. Camera-cloud failures are handled by
    // camCloudReceive's identity-guarded retry; the delegate must ignore them.
    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol proto: String?) {
        guard webSocketTask === task else { return }
        markAlive()
    }
    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith code: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        guard webSocketTask === task else { return }
        dropped()
    }
    func urlSession(_ s: URLSession, task t: URLSessionTask, didCompleteWithError error: Error?) {
        guard error != nil, t === task else { return }
        dropped()
    }
}
