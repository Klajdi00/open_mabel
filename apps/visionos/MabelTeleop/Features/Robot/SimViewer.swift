//
//  SimViewer.swift
//  MabelTeleop (ported from MABEL Mobile App v2)
//
//  The MuJoCo viewer, in the app: consumes the Simulation Control Center's
//  /ws/simview stream (JPEG frames over a WebSocket, rendered server-side by
//  MuJoCo's offscreen renderer from the live physics state) so the operator
//  sees exactly what the simulator sees — while every drive/joint/task control
//  keeps working. Drag orbits (azimuth/elevation), pinch dollies; the camera
//  follows the chassis by default, so the robot never wanders off-frame.
//

import Combine
import SwiftUI
import UIKit

/// One /ws/simview connection: sends the render config, receives JPEG frames.
final class SimViewStream: NSObject, URLSessionWebSocketDelegate {
    private var task: URLSessionWebSocketTask?
    private var session: URLSession!
    private let url: URL
    private let onFrame: (UIImage) -> Void
    private var stopped = false

    // Orbit state (server default parity; drag/pinch mutate + resend).
    var az: Double = 135, el: Double = -20, dist: Double = 3.2
    /// Render size/rate (set before open for full-bleed use).
    var width = 640, height = 400, fps = 15, quality = 70
    /// "chassis" (console tile) or nil + explicit lookat (physics overlay —
    /// the SAME point the SceneKit camera orbits, so the views match exactly).
    var follow: String? = "chassis"
    var lookat: [Double]? = nil
    /// Wall-clock of the last decoded frame — readiness gate for the crossfade.
    private(set) var lastFrameAt: Date?

    init(url: URL, onFrame: @escaping (UIImage) -> Void) {
        self.url = url
        self.onFrame = onFrame
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    /// Call once after configuring size/follow — separated from init so the
    /// overlay can size the render to the actual view aspect first.
    func connect() { open() }

    private func open() {
        let t = session.webSocketTask(with: url)
        task = t
        t.resume()
        sendConfig()
        receive()
    }

    func stop() {
        stopped = true
        task?.cancel(with: .goingAway, reason: nil)
        session.invalidateAndCancel()
    }

    /// Camera-only update: {az, el, dist, lookat} — the hot path for drags.
    /// The server's update() applies just the present keys, so this never
    /// touches the render size. Rate-limited to ~30 Hz.
    private var lastCamSend: CFTimeInterval = 0
    func sendCamera(force: Bool = false) {
        let now = CACurrentMediaTime()
        guard force || now - lastCamSend > 0.033 else { return }
        lastCamSend = now
        var cfg: [String: Any] = ["az": az, "el": el, "dist": dist]
        if follow == nil, let lookat { cfg["lookat"] = lookat }
        if let d = try? JSONSerialization.data(withJSONObject: cfg),
           let s = String(data: d, encoding: .utf8) {
            task?.send(.string(s)) { _ in }
        }
    }

    func sendConfig() {
        var cfg: [String: Any] = ["w": width, "h": height, "fps": fps,
                                  "quality": quality,
                                  "az": az, "el": el, "dist": dist]
        if let follow { cfg["follow"] = follow }
        else {
            cfg["follow"] = NSNull()
            if let lookat { cfg["lookat"] = lookat }
        }
        if let d = try? JSONSerialization.data(withJSONObject: cfg),
           let s = String(data: d, encoding: .utf8) {
            task?.send(.string(s)) { _ in }
        }
    }

    // DROP-TO-LATEST: the receive loop never waits on JPEG decode. The newest
    // blob replaces any undecoded one, and a single in-flight decode drains
    // whatever is freshest — so a slow decode raises the frame INTERVAL, never
    // the LATENCY (frames don't queue in the socket).
    private var pendingBlob: Data?
    private var decoding = false

    private func receive() {
        task?.receive { [weak self] result in
            guard let self, !self.stopped else { return }
            switch result {
            case .success(.data(let d)):
                self.pendingBlob = d            // newest wins
                self.drainDecode()
                self.receive()                  // immediately — no decode wait
            case .success:                      // text frames = scene meta; ignore
                self.receive()
            case .failure:
                // stream dropped (scene switch bounces the backend) — retry
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                    guard let self, !self.stopped else { return }
                    self.open()
                }
            }
        }
    }

    private func drainDecode() {
        guard !decoding, let blob = pendingBlob else { return }
        pendingBlob = nil
        decoding = true
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            let img = UIImage(data: blob)?.preparingForDisplay()
            DispatchQueue.main.async {
                guard let self else { return }
                self.decoding = false
                if let img {
                    self.lastFrameAt = Date()
                    self.onFrame(img)
                }
                self.drainDecode()              // decode whatever arrived meanwhile
            }
        }
    }
}

/// SHARED camera state for the physics (MuJoCo) viewer takeover: one set of
/// MuJoCo spherical parameters (azimuth°, elevation°, distance, look-at) drives
/// BOTH the SceneKit camera (LiveMapView's cinematic arc + tracking) and the
/// server render — so the crossfade lands on mathematically identical views.
/// Plain class (not @Published): consumed per display frame.
final class PhysicsCam {
    var active = false
    var az = 150.0, el = -20.0, dist = 3.2
    /// Orbit centre in SCENEKIT space (the robot's chest); LiveMapView owns it.
    var lookatScn = SIMD3<Float>(0, 0.9, 0)
    /// The cinematic engage-arc has finished its minimum sweep.
    var arcDone = false
    /// The crossfade has completed — the coordinator stops its slow drift and
    /// only TRACKS the values (the overlay's gestures own them now).
    var faded = false

    /// MuJoCo-space lookat (SceneKit (x,y,z) → MuJoCo (x, −z, y)).
    var lookatMj: [Double] {
        [Double(lookatScn.x), Double(-lookatScn.z), Double(lookatScn.y)]
    }

    /// Re-frame the orbit on the robot (sim reset / relocate): classic
    /// three-quarter view at fit distance — lookat keeps tracking the robot
    /// every frame, so only the spherical pose needs resetting.
    func refit() {
        el = -20
        dist = 3.2
    }
}

/// UIKit image plane: full-bleed aspect-fill blits with ZERO SwiftUI diffing —
/// a @State-driven Image at 20 fps re-laid-out the whole overlay every frame.
private struct FrameBlitView: UIViewRepresentable {
    let register: (UIImageView) -> Void

    func makeUIView(context: Context) -> UIImageView {
        let v = UIImageView()
        v.contentMode = .scaleAspectFill
        v.clipsToBounds = true
        v.setContentHuggingPriority(.init(1), for: .horizontal)
        v.setContentHuggingPriority(.init(1), for: .vertical)
        v.setContentCompressionResistancePriority(.init(1), for: .horizontal)
        v.setContentCompressionResistancePriority(.init(1), for: .vertical)
        register(v)
        return v
    }

    func updateUIView(_ uiView: UIImageView, context: Context) {}

    func sizeThatFits(_ proposal: ProposedViewSize, uiView: UIImageView, context: Context) -> CGSize? {
        proposal.replacingUnspecifiedDimensions()
    }
}

/// Full-bleed MuJoCo viewer that REPLACES the 3rd-person visualizer ("physics
/// view"): fades in over the SceneKit render once the engage arc completes and
/// frames are flowing; drag orbits / pinch dollies (mirrored back into
/// PhysicsCam so the SceneKit camera silently tracks — toggling back is
/// seamless). Teleop chrome stays on top and keeps working.
struct PhysicsViewerOverlay: View {
    let url: URL
    let cam: PhysicsCam
    @Binding var faded: Bool

    /// Late-bound blit target (the UIImageView registers after the stream's
    /// closure is created).
    final class FrameSink { weak var view: UIImageView? }
    private let frameSink = FrameSink()

    @State private var stream: SimViewStream?
    @State private var dragStart: (az: Double, el: Double)?
    @State private var pinchStart: Double?
    /// TRUE while a drag/pinch is in flight: the MuJoCo frame blends DOWN and
    /// the (mathematically identical, display-rate) SceneKit camera underneath
    /// carries the motion — instant orbit feedback, the render catches up on
    /// release. The cached last frame never shows stale-angle content at full
    /// opacity while the camera is somewhere else.
    @State private var interacting = false
    /// No frame for several seconds after mount → say WHY instead of panning
    /// forever (SCC down / stream refused).
    @State private var stalled = false
    @State private var mountedAt = Date()
    /// 10 Hz camera sync + readiness polling.
    private let tick = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                FrameBlitView { frameSink.view = $0 }
                    .frame(width: geo.size.width, height: geo.size.height)
                    .opacity(interacting ? 0.35 : 1)
                    .animation(.easeOut(duration: 0.22), value: interacting)
                // Status chip until the fade lands (doubles as the "loading"
                // cue while the engage pan runs). After a few seconds with no
                // frame at all, say WHY instead of panning forever.
                if !faded {
                    VStack {
                        Spacer()
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.mini).tint(.white)
                            Text(stalled
                                 ? "SIM VIEWER UNREACHABLE — IS THE SIM CONTROL CENTER RUNNING?"
                                 : "ENTERING PHYSICS VIEW")
                                .font(.system(size: 9, weight: .heavy)).tracking(1.2)
                                .foregroundStyle(stalled ? Color.orange : .white.opacity(0.9))
                        }
                        .padding(.horizontal, 12).padding(.vertical, 7)
                        .background(.black.opacity(0.55), in: Capsule())
                        .padding(.bottom, 200)
                    }
                    .allowsHitTesting(false)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture()
                    .onChanged { v in
                        guard faded else { return }
                        if dragStart == nil { dragStart = (cam.az, cam.el) }
                        guard let s0 = dragStart else { return }
                        cam.az = s0.az - Double(v.translation.width) * 0.4
                        cam.el = max(-89, min(60, s0.el + Double(v.translation.height) * 0.3))
                        interacting = true
                        pushCamera()               // immediate — never waits the tick
                    }
                    .onEnded { _ in
                        dragStart = nil
                        pushCamera(force: true)
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                            if dragStart == nil, pinchStart == nil { interacting = false }
                        }
                    }
            )
            .simultaneousGesture(
                MagnificationGesture()
                    .onChanged { m in
                        guard faded else { return }
                        if pinchStart == nil { pinchStart = cam.dist }
                        if let d0 = pinchStart { cam.dist = max(1.2, min(10, d0 / Double(m))) }
                        interacting = true
                        pushCamera()
                    }
                    .onEnded { _ in
                        pinchStart = nil
                        pushCamera(force: true)
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                            if dragStart == nil, pinchStart == nil { interacting = false }
                        }
                    }
            )
            .onAppear { openStream(size: geo.size) }
            .onChange(of: geo.size) { _, sz in
                // Rotation: re-render at the new aspect (server clamps bounds).
                let (w, h) = renderSize(sz)
                stream?.width = w
                stream?.height = h
                stream?.sendConfig()
            }
        }
        .ignoresSafeArea()
        .onReceive(tick) { _ in
            guard let stream else { return }
            // Mirror the shared camera to the server (arc, drift or gestures).
            stream.az = cam.az; stream.el = cam.el; stream.dist = cam.dist
            stream.lookat = cam.lookatMj
            stream.sendConfig()
            // Crossfade once the arc has swept AND frames are fresh.
            if !faded, cam.arcDone, let t = stream.lastFrameAt,
               Date().timeIntervalSince(t) < 1.0 {
                withAnimation(.easeInOut(duration: 0.9)) { faded = true }
                cam.faded = true
            }
        }
        .onDisappear {
            stream?.stop()
            stream = nil
        }
    }

    /// Longest render edge ≤ 960 px (the server's own cap for height): the
    /// benchmark shows fps is work-bound, not pixel-bound, but smaller frames
    /// cut decode + wire cost — and q65 halves the bytes of q80 invisibly.
    private func renderSize(_ size: CGSize) -> (Int, Int) {
        let k = min(1.0, 960.0 / max(size.width, size.height))
        return (Int(max(160, size.width * k)), Int(max(120, size.height * k)))
    }

    /// Hot-path camera push (drag/pinch): mirror the holder into the stream
    /// and send ONLY the camera keys, rate-limited inside the stream.
    private func pushCamera(force: Bool = false) {
        guard let stream else { return }
        stream.az = cam.az; stream.el = cam.el; stream.dist = cam.dist
        stream.lookat = cam.lookatMj
        stream.sendCamera(force: force)
    }

    private func openStream(size: CGSize) {
        guard stream == nil else { return }
        let s = SimViewStream(url: url) { img in
            // the blit target registers via @State; resolve at frame time
            frameSink.view?.image = img
        }
        let (w, h) = renderSize(size)
        s.width = w; s.height = h
        s.fps = 24; s.quality = 65
        s.follow = nil                      // explicit lookat = the matched point
        s.lookat = cam.lookatMj
        s.az = cam.az; s.el = cam.el; s.dist = cam.dist
        s.connect()
        stream = s
    }
}

/// The console tile: live MuJoCo render + orbit gestures.
struct SimViewerTile: View {
    let url: URL
    @State private var frame: UIImage?
    @State private var stream: SimViewStream?
    @State private var dragStart: (az: Double, el: Double)?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.black.opacity(0.5))
            if let frame {
                Image(uiImage: frame)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            } else {
                VStack(spacing: 6) {
                    ProgressView().tint(Theme.Palette.textTertiary)
                    Text("Connecting to the simulator…")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
            // Badge (top-left): what this view IS.
            VStack {
                HStack {
                    Text("MUJOCO · LIVE PHYSICS")
                        .font(.system(size: 8, weight: .heavy)).tracking(1)
                        .foregroundStyle(.white.opacity(0.85))
                        .padding(.horizontal, 7).padding(.vertical, 4)
                        .background(.black.opacity(0.45), in: Capsule())
                    Spacer()
                }
                Spacer()
            }
            .padding(8)
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture()
                .onChanged { v in
                    guard let stream else { return }
                    if dragStart == nil { dragStart = (stream.az, stream.el) }
                    guard let s = dragStart else { return }
                    stream.az = s.az - Double(v.translation.width) * 0.5
                    stream.el = max(-89, min(89, s.el + Double(v.translation.height) * 0.4))
                    stream.sendConfig()
                }
                .onEnded { _ in dragStart = nil }
        )
        .gesture(
            MagnificationGesture()
                .onChanged { m in
                    guard let stream else { return }
                    stream.dist = max(0.8, min(12, stream.dist / m))
                    stream.sendConfig()
                }
        )
        .onAppear {
            guard stream == nil else { return }
            let s = SimViewStream(url: url) { frame = $0 }
            s.connect()
            stream = s
        }
        .onDisappear {
            stream?.stop()
            stream = nil
            frame = nil
        }
    }
}
