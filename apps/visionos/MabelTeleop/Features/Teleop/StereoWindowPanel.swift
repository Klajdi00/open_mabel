//
//  StereoWindowPanel.swift
//  MabelTeleop
//
//  Full immersion's head view: a FLAT, HEAD-LOCKED stereo window at true
//  scale (operator, 2026-10-16: "instead of the wrapped-around view keep it
//  windowed, but when the human moves the head the window should follow …
//  scale the view to the right distance to match the camera … left and right
//  eye should project the left and right footage from the stereo cam").
//
//  WHY FLAT. The head cameras are pinhole cameras: a pinhole image is exactly
//  the picture a flat plane perpendicular to the optical axis shows. A plane
//  of width W = 2·D·tan(hfov/2) and height H = 2·D·tan(vfov/2) at distance D
//  subtends precisely the camera's field of view, so every pixel lands at the
//  visual angle its ray was captured from — true scale, no reprojection (the
//  old cylinder needed the tanθ remap because it was NOT the image's shape).
//
//  WHY HEAD-LOCKED. The window is a child of `AnchorEntity(.head)`, so it
//  moves rigidly with the operator's head: the display's rays stay aligned
//  with the operator's eyes, which is what makes the true-scale claim hold
//  wherever they look. (During whole-body teleop the robot's neck follows the
//  head, so the picture also follows what they look at — late by the neck +
//  video latency; that lag is the robot's, not the display's.)
//
//  WHY A PER-EYE SHIFT. The stereo cameras are parallel, ~63 mm apart. One
//  plane centred BETWEEN the eyes puts each eye's image IPD/2 off that eye's
//  own axis — a fixed vergence error of 2·atan(IPD/2 / D) that pulls the whole
//  scene toward the plane (1.80° at 2 m: a point 1 m away reads 0.67 m).
//  The StereoCameraMaterial therefore samples the LEFT image at u + s and the
//  RIGHT image at u − s, s = (IPD/2)/W: each eye's image is then centred on
//  its own eye, and the display is ORTHOSTEREO at any D (D only sets where the
//  eyes focus and what draws in front of the window). Derivation + numbers:
//  docs/mabel_apps_guide.tex §"Full immersion: the head-locked stereo window".
//
//  EYE ROUTING. LEFT eye ← "main" (= zed_left in sim, the stitched module's
//  left half on the real rig), RIGHT eye ← "head_right". The ShaderGraph's
//  CameraIndexSwitch picks per rendered eye; the Simulator renders ONE (mono)
//  view, so the right-eye path only exists on device — `debugEyeTest=YES`
//  replaces the feeds with a red "L" / blue "R" card so the operator can
//  close one eye and check the routing.
//

import Foundation
import RealityKit
import RealityKitContent
import CoreGraphics
import UIKit
import OSLog

/// The true-scale geometry, pure so it can be unit-tested
/// (`StereoWindowGeometryTests`).
struct StereoWindowGeometry: Equatable {
    /// Head camera vertical field of view (deg) and image aspect (w/h).
    var fovyDeg: Float
    var aspect: Float
    /// Viewing distance of the window (m) — where the eyes focus.
    var distance: Float
    /// Interpupillary distance (m). visionOS does not expose the wearer's,
    /// so this is the camera BASELINE (63 mm in the MJCF), which is also the
    /// adult mean — orthostereo wants them equal anyway.
    var ipd: Float
    /// Window magnification: 1 = TRUE SCALE (subtends the camera FOV exactly);
    /// < 1 shrinks the picture uniformly (operator, 2026-10-16: "a little too
    /// big"). The per-eye shift still centres each image on its eye, so the
    /// stereo stays consistent — every angle (and disparity) scales by ≈ this.
    var scale: Float = 1
    /// When set, the window is THIS wide (m, before `scale`) and as tall as the camera aspect
    /// makes it, instead of true scale. Full immersion uses `fpvWidth` (operator, 2026-10-30:
    /// "keep the window size the same as the 1st person window view"): true scale is 83° × 100°,
    /// taller than the headset's view, so its bottom was always cut off.
    var matchedWidth: Float? = nil

    static let defaultDistance: Float = 2.0
    /// The 1st-person window's video: the 1280 × 820 pt main window less the FPV insets
    /// (72 pt each side) = 1136 pt, at visionOS's ~1360 pt per metre.
    static let fpvWidth: Float = Float(1280 - 2 * 72) / 1360
    /// Where the full-immersion window sits — roughly where visionOS opens a window, so the
    /// same metres read as the same size. (visionOS does not expose the distance; "Smaller" /
    /// "Larger" on the deck rescale it.)
    static let fpvDistance: Float = 1.5
    /// Full immersion's window-size range, 1 = the 1st-person window's size. The top stops where
    /// the status bar still fits ABOVE the window under `DeckLayout.statusTopMax` (pinned by
    /// DeckLayoutTests), so no size puts a bar over the picture.
    static let fpvScaleRange: ClosedRange<Float> = 0.6...1.2
    static let defaultIPD: Float = 0.063
    /// Operator-adjustable range (full-immersion control deck).
    static let scaleRange: ClosedRange<Float> = 0.6...1.0
    static let defaultScale: Float = 0.8

    var fovyRad: Float { fovyDeg * .pi / 180 }
    var hfovRad: Float { 2 * atan(tan(fovyRad / 2) * aspect) }
    var width: Float {
        if let w = matchedWidth { return w * scale }
        return 2 * distance * tan(hfovRad / 2) * scale
    }
    var height: Float {
        if matchedWidth != nil { return width / aspect }
        return 2 * distance * tan(fovyRad / 2) * scale
    }
    /// Half-angle the window subtends vertically (where its bottom edge sits).
    var halfHeightAngle: Float { atan((height / 2) / distance) }
    var halfWidthAngle: Float { atan((width / 2) / distance) }
    /// Texture-space shift that centres each eye's image on that eye:
    /// left samples at u + s, right at u − s.
    var eyeShiftU: Float { (ipd / 2) / width }
    /// Vergence error (rad) a SHARED, unshifted window would add:
    /// 2·atan((ipd/2)/distance). The shift cancels it.
    var unshiftedVergenceError: Float { 2 * atan((ipd / 2) / distance) }
}

@MainActor
final class StereoWindowPanel {

    /// Parent this to the HEAD anchor; the window sits at −Z (straight ahead).
    let entity = Entity()
    /// The video quad itself.
    private let video = ModelEntity()
    /// A thin dark bezel behind the video — reads as a window, not a hole.
    private let bezel = ModelEntity()

    // Head camera defaults (MJCF zed_left: fovy 83.3°, 4:3); the server's
    // `hello` overrides with the live intrinsics (real rig: from calibration).
    static let defaultFovyDeg: Float = 83.3
    static let defaultAspect: Float = 4.0 / 3.0

    private(set) var geometry: StereoWindowGeometry
    /// True once both eyes stream and the stereo material is bound.
    private(set) var isStereo = false

    private let materialName = "StereoCameraMaterial"
    private var stereoMaterial: ShaderGraphMaterial?
    private var monoMaterial = UnlitMaterial(color: .black)
    private var leftTexture: TextureResource?
    private var rightTexture: TextureResource?
    private var leftBusy = false
    private var rightBusy = false

    private var leftFeed: VideoFeed?
    private var rightFeed: VideoFeed?

    /// TRUE once at least one eye has painted a frame. Hidden until then — an
    /// empty window in full immersion is a black slab across the whole view.
    private(set) var hasVideo = false
    /// What the caller wants (full immersion on/off); actual visibility is
    /// wantVisible AND hasVideo.
    var wantVisible = false { didSet { entity.isEnabled = wantVisible && hasVideo } }
    var onFirstFrame: (() -> Void)?

    private let log = Logger(subsystem: "com.mabel.teleop", category: "StereoWindowPanel")

    init(fovyDeg: Float = StereoWindowPanel.defaultFovyDeg,
         aspect: Float = StereoWindowPanel.defaultAspect,
         distance: Float = StereoWindowGeometry.defaultDistance,
         ipd: Float = StereoWindowGeometry.defaultIPD,
         scale: Float = StereoWindowGeometry.defaultScale,
         matchedWidth: Float? = nil) {
        geometry = StereoWindowGeometry(fovyDeg: fovyDeg, aspect: aspect,
                                        distance: distance, ipd: ipd, scale: scale,
                                        matchedWidth: matchedWidth)
        entity.name = "stereoWindow"
        video.name = "stereoWindowVideo"
        bezel.name = "stereoWindowBezel"
        video.model = ModelComponent(mesh: Self.quad(geometry), materials: [UnlitMaterial(color: .black)])
        bezel.model = ModelComponent(mesh: Self.bezelMesh(geometry),
                                     materials: [UnlitMaterial(color: UIColor(white: 0.08, alpha: 1))])
        entity.addChild(bezel)
        entity.addChild(video)
        entity.isEnabled = false
    }

    /// Re-size when the server reports different head intrinsics.
    func updateIntrinsics(fovyDeg: Float, aspect: Float) {
        guard abs(fovyDeg - geometry.fovyDeg) > 0.05 || abs(aspect - geometry.aspect) > 0.01 else { return }
        geometry.fovyDeg = fovyDeg
        geometry.aspect = aspect
        rebuildMeshes()
    }

    /// Operator window size (60–100 %, 100 % = true scale).
    func setScale(_ s: Float) {
        let r = geometry.matchedWidth == nil ? StereoWindowGeometry.scaleRange : StereoWindowGeometry.fpvScaleRange
        let c = min(max(s, r.lowerBound), r.upperBound)
        guard abs(c - geometry.scale) > 0.001 else { return }
        geometry.scale = c
        rebuildMeshes()
    }

    private func rebuildMeshes() {
        let mats = video.model?.materials ?? [UnlitMaterial(color: .black)]
        video.model = ModelComponent(mesh: Self.quad(geometry), materials: mats)
        bezel.model?.mesh = Self.bezelMesh(geometry)
        if isStereo { try? assignMaterialIfReady() }   // re-apply the new shift
    }

    /// The video quad at z = −D, centred on the head axis. UV origin bottom-
    /// left (RealityKit convention — the same the old wrap used, so the image
    /// is upright and un-mirrored).
    private static func quad(_ g: StereoWindowGeometry) -> MeshResource {
        rect(w: g.width, h: g.height, z: -g.distance)
    }

    /// 1.5 % larger, 1 cm behind the video.
    private static func bezelMesh(_ g: StereoWindowGeometry) -> MeshResource {
        rect(w: g.width * 1.015 + 0.02, h: g.height * 1.015 + 0.02, z: -g.distance - 0.01)
    }

    private static func rect(w: Float, h: Float, z: Float) -> MeshResource {
        var d = MeshDescriptor()
        d.positions = MeshBuffer([SIMD3(-w / 2, -h / 2, z), SIMD3(w / 2, -h / 2, z),
                                  SIMD3(-w / 2,  h / 2, z), SIMD3(w / 2,  h / 2, z)])
        d.normals = MeshBuffer([SIMD3<Float>](repeating: SIMD3(0, 0, 1), count: 4))
        d.textureCoordinates = MeshBuffer([SIMD2<Float>(0, 0), SIMD2(1, 0), SIMD2(0, 1), SIMD2(1, 1)])
        // Counter-clockwise seen from the viewer (+Z) → front face toward them.
        d.primitives = .triangles([0, 1, 2, 2, 1, 3])
        return try! MeshResource.generate(from: [d])
    }

    // MARK: - Lifecycle

    /// Starts the head video. PRIMARY = ONE stream, "head_stereo": the server's
    /// left|right halves from the same capture (thor: one ZED grab; sim: one
    /// state snapshot; Mac --real: one stitched frame), side by side. Both eyes
    /// sample that ONE texture, so the pair can never be out of step — the
    /// left/right mismatch operators felt as discomfort came from TWO
    /// independently-paced eye streams. FALLBACK (no "head_stereo" frame within
    /// `stereoGrace` — older server, or a path that cannot carry it): the two
    /// eye streams, committed in LOCKSTEP (`StereoLockstep`). Safe to call again
    /// on reconnect: existing feeds are stopped first.
    func start(sources: HeadVideoSources) async {
        stop()
        await loadMaterial()
        #if DEBUG
        if UserDefaults.standard.bool(forKey: "debugEyeTest") {
            // Eye-routing check THROUGH THE PRIMARY (side-by-side) PATH: red "L"
            // on the left half, blue "R" on the right. On the headset, close one
            // eye at a time — the open eye must read its own letter, and the
            // crosses must fuse. (The Simulator renders one view = the left.)
            log.info("debugEyeTest: synthetic L|R side-by-side card replaces the head feeds")
            ingestSBS(Self.sbsEyeCard())
            return
        }
        #endif
        self.sources = sources
        sbsFeed = VideoFeed(host: sources.udpHost, udpPort: sources.udpPort, codec: sources.codec,
                            stream: "head_stereo", mjpegURL: sources.stereoURL,
                            label: "head_stereo") { [weak self] cg in
            Task { @MainActor [weak self] in self?.ingestSBS(cg) }
        }
        sbsFeed?.start()
        graceTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(Self.stereoGrace))
            guard let self, !Task.isCancelled, self.mode != .sideBySide else { return }
            self.log.warning("[MABEL-WINDOW] no head_stereo frame in \(Self.stereoGrace)s — falling back to two eye streams (lockstep)")
            self.startEyeFeeds()
        }
    }

    /// How long to wait for the synced stereo stream before falling back.
    static let stereoGrace: Double = 2.5

    private func startEyeFeeds() {
        guard let s = sources, leftFeed == nil else { return }
        leftFeed = VideoFeed(host: s.udpHost, udpPort: s.udpPort, codec: s.codec, stream: "main",
                             mjpegURL: s.leftURL,
                             label: "head_left") { [weak self] cg in
            Task { @MainActor [weak self] in self?.ingestEye(cg, eye: .left) }
        }
        rightFeed = VideoFeed(host: s.udpHost, udpPort: s.udpPort, codec: s.codec, stream: "head_right",
                              mjpegURL: s.rightURL,
                              label: "head_right") { [weak self] cg in
            Task { @MainActor [weak self] in self?.ingestEye(cg, eye: .right) }
        }
        leftFeed?.start()
        rightFeed?.start()
    }

    private func stopEyeFeeds() {
        leftFeed?.stop();  leftFeed = nil
        rightFeed?.stop(); rightFeed = nil
    }

    #if DEBUG
    /// Side-by-side card: red "L" on the left half, blue "R" on the right, each
    /// with a centre cross (the crosses must FUSE into one with both eyes open —
    /// a doubled cross means the routing or the per-eye shift is wrong).
    private static func sbsEyeCard() -> CGImage {
        let eye = CGSize(width: 960, height: 720)
        let size = CGSize(width: eye.width * 2, height: eye.height)
        let img = UIGraphicsImageRenderer(size: size).image { ctx in
            for (i, (letter, color)) in [("L", UIColor(red: 0.85, green: 0.15, blue: 0.15, alpha: 1)),
                                         ("R", UIColor(red: 0.15, green: 0.35, blue: 0.95, alpha: 1))].enumerated() {
                let x0 = CGFloat(i) * eye.width
                color.setFill(); ctx.fill(CGRect(x: x0, y: 0, width: eye.width, height: eye.height))
                let c = ctx.cgContext
                UIColor.white.setStroke(); c.setLineWidth(4)
                c.move(to: CGPoint(x: x0 + eye.width / 2, y: eye.height / 2 - 60))
                c.addLine(to: CGPoint(x: x0 + eye.width / 2, y: eye.height / 2 + 60))
                c.move(to: CGPoint(x: x0 + eye.width / 2 - 60, y: eye.height / 2))
                c.addLine(to: CGPoint(x: x0 + eye.width / 2 + 60, y: eye.height / 2))
                c.strokePath()
                let t = NSAttributedString(string: letter, attributes: [
                    .font: UIFont.systemFont(ofSize: 260, weight: .black),
                    .foregroundColor: UIColor.white])
                t.draw(at: CGPoint(x: x0 + (eye.width - t.size().width) / 2, y: 40))
            }
        }
        return img.cgImage!
    }
    #endif

    func stop() {
        graceTask?.cancel(); graceTask = nil
        sbsFeed?.stop(); sbsFeed = nil
        stopEyeFeeds()
        lockstep = StereoLockstep()
    }

    private func loadMaterial() async {
        for resource in ["Materials/StereoCameraMaterial", "StereoCameraMaterial",
                         "Materials/StereoCameraMaterial.usda"] {
            do {
                let mat = try await ShaderGraphMaterial(
                    named: "/Root/\(materialName)", from: resource, in: realityKitContentBundle)
                stereoMaterial = mat
                log.info("Loaded stereo material (\(resource, privacy: .public)) — true per-eye head view")
                return
            } catch {
                log.error("Stereo material load '\(resource, privacy: .public)' failed: \(String(describing: error), privacy: .public)")
            }
        }
        stereoMaterial = nil
        log.warning("Stereo material unavailable — mono fallback (left eye only)")
    }

    // MARK: - Frame ingest (drop-while-busy, atomic double-buffered textures)

    private enum Eye { case left, right }
    private enum Mode { case none, sideBySide, separate, mono }
    private var mode: Mode = .none
    private var sources: HeadVideoSources?
    private var sbsFeed: VideoFeed?
    private var graceTask: Task<Void, Never>?
    private var sbsBusy = false
    private var lockstep = StereoLockstep()
    private var pendingLeft: TextureResource?
    private var pendingRight: TextureResource?

    /// Upload one decoded frame as a BRAND-NEW texture and swap the material to
    /// it atomically — never `TextureResource.replace(withImage:)`.
    ///
    /// `replace(withImage:)` mutates the GPU texture in place while the
    /// material is still bound to it, so the rasterizer can sample a half-
    /// written upload (on the old curved wrap that showed as a moving zig-zag
    /// tear). A fresh TextureResource per frame is double-buffering: an
    /// in-flight render keeps the OLD texture while the GPU fills the NEW one.
    /// `.allocateAndGenerateAll` regenerates the FULL mip chain per frame, so
    /// trilinear minification never mixes live detail with stale coarse mips
    /// (the wiggle/moiré on walls and edges). Drop-while-busy keeps one upload
    /// in flight so the freshest frame wins.
    private func makeTexture(_ image: CGImage) async throws -> TextureResource {
        let opts = TextureResource.CreateOptions(semantic: .color,
                                                 mipmapsMode: .allocateAndGenerateAll)
        return try await TextureResource(image: image, options: opts)
    }

    /// PRIMARY: one side-by-side frame → one texture → both eyes.
    private func ingestSBS(_ image: CGImage) {
        guard !sbsBusy else { return }
        sbsBusy = true
        Task { @MainActor in
            defer { sbsBusy = false }
            do {
                let tex = try await makeTexture(image)
                if mode != .sideBySide {
                    // The synced stream is live — the eye-stream fallback (if it
                    // started) is now redundant and only costs bandwidth.
                    graceTask?.cancel(); graceTask = nil
                    stopEyeFeeds()
                }
                try bindSideBySide(tex)
            } catch {
                log.error("SBS texture upload failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    /// FALLBACK: two eye streams, committed in LOCKSTEP — both textures swap
    /// together, never one alone (see StereoLockstep).
    private func ingestEye(_ image: CGImage, eye: Eye) {
        guard mode != .sideBySide else { return }
        switch eye {
        case .left  where leftBusy:  return
        case .right where rightBusy: return
        default: break
        }
        if eye == .left { leftBusy = true } else { rightBusy = true }
        Task { @MainActor in
            defer { if eye == .left { leftBusy = false } else { rightBusy = false } }
            do {
                let tex = try await makeTexture(image)
                let now = CACurrentMediaTime()
                let decision: StereoLockstep.Decision
                if eye == .left { pendingLeft = tex; decision = lockstep.arrived(.left, at: now) }
                else { pendingRight = tex; decision = lockstep.arrived(.right, at: now) }
                switch decision {
                case .commitPair:
                    if let l = pendingLeft, let r = pendingRight { try bindSeparate(left: l, right: r) }
                case .commitMono:
                    if let l = pendingLeft { bindMono(l) }
                case .wait:
                    break
                }
            } catch {
                log.error("Texture upload failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    private func bindSideBySide(_ tex: TextureResource) throws {
        guard var mat = stereoMaterial else {
            // No per-eye material: show the LEFT half to both eyes.
            monoMaterial.color = .init(tint: .white, texture: .init(tex))
            monoMaterial.textureCoordinateTransform = .init(offset: .zero, scale: SIMD2(0.5, 1))
            video.model?.materials = [monoMaterial]
            noteFirstFrame(); mode = .sideBySide; isStereo = false
            return
        }
        try mat.setParameter(name: "LeftTexture",  value: .textureResource(tex))
        try mat.setParameter(name: "RightTexture", value: .textureResource(tex))
        try mat.setParameter(name: "ScaleU",      value: .float(0.5))
        try mat.setParameter(name: "OffsetLeft",  value: .float(0))
        try mat.setParameter(name: "OffsetRight", value: .float(0.5))
        try mat.setParameter(name: "EyeShift",    value: .float(geometry.eyeShiftU))
        stereoMaterial = mat
        video.model?.materials = [mat]
        if mode != .sideBySide || !isStereo {
            log.info("[MABEL-WINDOW] STEREO bound (side-by-side head_stereo, synced): left eye ← left half, right eye ← right half, eyeShiftU=\(self.geometry.eyeShiftU)")
        }
        mode = .sideBySide; isStereo = true
        noteFirstFrame()
    }

    private func bindSeparate(left l: TextureResource, right r: TextureResource) throws {
        guard var mat = stereoMaterial else { bindMono(l); return }
        try mat.setParameter(name: "LeftTexture",  value: .textureResource(l))
        try mat.setParameter(name: "RightTexture", value: .textureResource(r))
        try mat.setParameter(name: "ScaleU",      value: .float(1))
        try mat.setParameter(name: "OffsetLeft",  value: .float(0))
        try mat.setParameter(name: "OffsetRight", value: .float(0))
        // Per-eye shift (see StereoWindowGeometry.eyeShiftU). The MONO path
        // never shifts: one image for both eyes has no stereo geometry.
        try mat.setParameter(name: "EyeShift",    value: .float(geometry.eyeShiftU))
        stereoMaterial = mat
        video.model?.materials = [mat]
        if mode != .separate {
            log.info("[MABEL-WINDOW] STEREO bound (two eye streams, lockstep fallback): left eye ← main, right eye ← head_right, eyeShiftU=\(self.geometry.eyeShiftU)")
        }
        mode = .separate; isStereo = true
        noteFirstFrame()
    }

    private func bindMono(_ l: TextureResource) {
        monoMaterial.color = .init(tint: .white, texture: .init(l))
        monoMaterial.textureCoordinateTransform = .init(offset: .zero, scale: SIMD2(1, 1))
        video.model?.materials = [monoMaterial]
        if mode != .mono { log.warning("[MABEL-WINDOW] MONO: right eye stream silent — left image to both eyes") }
        mode = .mono; isStereo = false
        noteFirstFrame()
    }

    private func noteFirstFrame() {
        guard !hasVideo else { return }
        hasVideo = true
        entity.isEnabled = wantVisible
        onFirstFrame?()
    }

    /// Re-apply the current binding (new intrinsics → new eye shift).
    private func assignMaterialIfReady() throws {
        guard var mat = stereoMaterial, isStereo else { return }
        try mat.setParameter(name: "EyeShift", value: .float(geometry.eyeShiftU))
        stereoMaterial = mat
        video.model?.materials = [mat]
    }
}

/// Where the head video comes from (built by the immersive view from the
/// controller's endpoints).
struct HeadVideoSources {
    var stereoURL: URL?        // MJPEG "head_stereo" (synced L|R)
    var leftURL: URL           // fallback eye streams
    var rightURL: URL
    var udpHost: String?
    var udpPort: UInt16
    var codec: VideoCodec
}

/// LOCKSTEP for the two-stream fallback: both eyes' textures are swapped
/// TOGETHER, only once each eye has a frame newer than the last commit —
/// never one eye alone (which shows the two eyes different instants, the
/// discomfort the operator reported). Within a pair the newest frame of each
/// eye wins. If one eye goes silent for `staleAfter`, the other is shown MONO
/// rather than frozen against a stale partner. Pure → `StereoLockstepTests`.
struct StereoLockstep {
    enum Side { case left, right }
    enum Decision: Equatable { case wait, commitPair, commitMono }

    var staleAfter: Double = 0.5
    private var leftNew = false
    private var rightNew = false
    private var lastLeft: Double = -.infinity
    private var lastRight: Double = -.infinity

    mutating func arrived(_ side: Side, at t: Double) -> Decision {
        switch side {
        case .left:  leftNew = true;  lastLeft = t
        case .right: rightNew = true; lastRight = t
        }
        if leftNew && rightNew {
            leftNew = false; rightNew = false
            return .commitPair
        }
        // Only the LEFT may go mono (the right image alone is the wrong view
        // for a mono fallback — `main` is the left eye everywhere).
        if side == .left && t - lastRight > staleAfter {
            leftNew = false
            return .commitMono
        }
        return .wait
    }
}
