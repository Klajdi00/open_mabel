//
//  CameraProjections.swift
//  MABEL Mobile App
//
//  Control-center-style "Cam 3D" projections: each robot camera gets a frustum
//  wireframe + a LIVE video plane hung at its true lens geometry, parented to
//  the camera's mount link so it rides every joint and base motion. The image
//  is a uniform 0.3 m tall; its depth D is derived from the camera's TRUE
//  vertical FOV (D = (H/2)/tan(atan((h/2)/fy))) so wide lenses sit close and
//  narrow lenses sit farther — the same math as the web control center's
//  viewport3d.js `_buildCamViz`.
//
//  Intrinsics/extrinsics come from the bundled `mabel_camera_rig.json`
//  (mount_pos / mount_quat are the MuJoCo camera mounts in each body frame;
//  both MuJoCo and SceneKit cameras look along −Z with +Y up, so the mount
//  transform drops straight in — verified by the 1st-person POV camera).
//

import SceneKit
import UIKit
import simd

/// One camera from the bundled rig JSON (a subset of its fields).
struct CameraRigSpec: Decodable {
    let name: String
    let w: Int
    let h: Int
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let mount_body: String
    let mount_pos: [Double]
    let mount_quat: [Double]      // (w, x, y, z) — body ← camera
}

enum CameraRigFile {
    /// Load the bundled camera rig (generated alongside the MJCF pipeline).
    static func load() -> [CameraRigSpec] {
        guard let url = Bundle.main.url(forResource: "mabel_camera_rig", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let specs = try? JSONDecoder().decode([CameraRigSpec].self, from: data) else { return [] }
        return specs
    }
}

/// Streams an MJPEG URL and delivers the LATEST decoded frame — a headless
/// MJPEGView for texturing SceneKit materials. Drop-to-latest: only the newest
/// complete JPEG in each network batch is decoded, so a slow consumer never
/// backs up the stream.
final class MJPEGTextureSource: NSObject, URLSessionDataDelegate {
    private var buffer = Data()
    private var session: URLSession!
    private var task: URLSessionDataTask?
    private let url: URL
    private let onFrame: (UIImage) -> Void
    private var decoding = false
    private var stopped = false

    init(url: URL, onFrame: @escaping (UIImage) -> Void) {
        self.url = url
        self.onFrame = onFrame
        super.init()
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 8
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
        open()
    }

    private func open() {
        buffer.removeAll(keepingCapacity: true)
        task = session.dataTask(with: url)
        task?.resume()
    }

    func stop() {
        stopped = true
        task?.cancel()
        session.invalidateAndCancel()
    }

    /// A dead/failed stream RETRIES every 2 s until stopped — the projections
    /// attach during connection racing, when the camera server may not be
    /// reachable yet (a one-shot fetch left the planes dark forever).
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard !stopped else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self, !self.stopped else { return }
            self.open()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        buffer.append(data)
        // Keep only the newest complete JPEG (SOI FFD8 … EOI FFD9).
        var latest: Data? = nil
        while let eoi = buffer.range(of: Data([0xFF, 0xD9])) {
            if let soi = buffer.range(of: Data([0xFF, 0xD8]), options: [], in: buffer.startIndex..<eoi.lowerBound) {
                latest = buffer.subdata(in: soi.lowerBound..<eoi.upperBound)
            }
            buffer.removeSubrange(buffer.startIndex..<eoi.upperBound)
        }
        if buffer.count > 4_000_000 { buffer.removeAll(keepingCapacity: true) }   // runaway guard
        guard let jpeg = latest, !decoding else { return }
        decoding = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let img = UIImage(data: jpeg)?.preparingForDisplay()
            DispatchQueue.main.async {
                self?.decoding = false
                if let img { self?.onFrame(img) }
            }
        }
    }
}

/// Builds + owns the per-camera projection nodes and their video streams.
@MainActor
final class CameraProjector {
    /// wire stream key → camera-rig camera name (same table as the control center).
    static let streamToCam: [(stream: String, cam: String)] = [
        ("main", "zed_left"),
        ("front", "body_front_cam"),
        ("wrist_left", "l_wrist_cam"),
        ("wrist_right", "r_wrist_cam"),
    ]
    /// camera-rig mount body → rig node name (a group's articulation node is
    /// named by its driving JOINT when it has one, else by the group name).
    static let mountToNode: [String: String] = [
        "head": "neck_roll",
        "mabel_chassis": "base_link",
        "torso": "torso",
        "left_palm": "left_palm",
        "right_palm": "right_palm",
    ]

    /// Uniform projected-image height in metres (control center default 0.3;
    /// user-adjustable 0.1–1.0 from the pull-up console).
    private(set) var projSize = 0.3
    /// Flat = rectified pinhole plane; curved = ray-cap through the lens model
    /// (the control center's "undistort" toggle).
    private(set) var flat = true

    private var groups: [SCNNode] = []
    private var entries: [(spec: CameraRigSpec, grp: SCNNode, material: SCNMaterial, stream: String)] = []
    private var sources: [MJPEGTextureSource] = []
    /// Aspect each camera's plane geometry was BUILT at. Rebuilt live when the
    /// decoded frames disagree (>1%) — the rig JSON's w×h is the exporter
    /// canvas, not the wire stream (sim head = 4:3, spec said 16:9), and a
    /// mismatch renders the video squeezed.
    private var builtAspect: [String: Double] = [:]
    private(set) var attached = false

    /// Hang a projection group off each enabled camera's mount link. `streams`
    /// picks WHICH cameras get a projection (per-camera console toggles;
    /// default = head only). `urlProvider` maps a wire stream key to its MJPEG
    /// URL (nil → frustum only, no video).
    func attach(to robot: SCNNode, streams: Set<String>, urlProvider: (String) -> URL?) {
        detach()
        let specs = CameraRigFile.load()
        for (stream, camName) in Self.streamToCam where streams.contains(stream) {
            guard let c = specs.first(where: { $0.name == camName }),
                  c.mount_pos.count == 3, c.mount_quat.count == 4,
                  let nodeName = Self.mountToNode[c.mount_body],
                  let mount = robot.childNode(withName: nodeName, recursively: true) else { continue }

            let grp = SCNNode()
            grp.name = "camviz_\(camName)"
            grp.position = SCNVector3(c.mount_pos[0], c.mount_pos[1], c.mount_pos[2])
            let q = c.mount_quat
            grp.orientation = SCNQuaternion(q[1], q[2], q[3], q[0])   // (w,x,y,z) → SCN (x,y,z,w)

            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = UIColor(white: 0.05, alpha: 0.85)
            material.isDoubleSided = true
            material.transparency = 0.97

            buildGeometry(for: c, in: grp, material: material)
            mount.addChildNode(grp)
            groups.append(grp)
            entries.append((c, grp, material, stream))
            builtAspect[stream] = Double(c.w) / Double(c.h)

            if let url = urlProvider(stream) {
                sources.append(MJPEGTextureSource(url: url) { [weak self, weak material] img in
                    material?.diffuse.contents = img
                    // Live video landed: full opacity — the 0.97 transparency
                    // was for the dark placeholder pane, and it dimmed the
                    // video vs the 1st-person overlay (visible at handover).
                    material?.transparency = 1.0
                    self?.frameArrived(stream: stream, size: img.size)
                })
            }
        }
        attached = true
    }

    /// Re-generate the frustum + image geometry (size / flat-curved change) while
    /// KEEPING the live video streams and materials — sliders never restart MJPEG.
    func setProjection(size: Double, flat: Bool) {
        guard abs(size - projSize) > 1e-6 || flat != self.flat else { return }
        projSize = min(1.0, max(0.1, size))
        self.flat = flat
        for e in entries {
            e.grp.childNodes.forEach { $0.removeFromParentNode() }
            buildGeometry(for: effectiveSpec(e.spec, stream: e.stream), in: e.grp, material: e.material)
        }
    }

    /// A decoded frame for `stream`: record the wire size and, if its aspect
    /// disagrees with the built plane, rebuild that camera's geometry so the
    /// texture is never stretched. Called on the main queue (MJPEGTextureSource
    /// delivers frames there).
    private func frameArrived(stream: String, size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        VideoStreamAspect.set(stream, size)
        let aspect = Double(size.width / size.height)
        guard let built = builtAspect[stream], abs(aspect - built) > 0.01,
              let e = entries.first(where: { $0.stream == stream }) else { return }
        builtAspect[stream] = aspect
        e.grp.childNodes.forEach { $0.removeFromParentNode() }
        buildGeometry(for: effectiveSpec(e.spec, stream: e.stream), in: e.grp, material: e.material)
    }

    /// The rig spec with its aspect replaced by the measured stream aspect:
    /// the vertical geometry (h / fy / cy — the true vertical FOV) is kept,
    /// width is derived for square pixels.
    private func effectiveSpec(_ c: CameraRigSpec, stream: String) -> CameraRigSpec {
        guard let aspect = builtAspect[stream],
              abs(aspect - Double(c.w) / Double(c.h)) > 0.01 else { return c }
        let w = max(2.0, (Double(c.h) * aspect).rounded())
        return CameraRigSpec(name: c.name, w: Int(w), h: c.h, fx: c.fy, fy: c.fy,
                             cx: w / 2, cy: c.cy, mount_body: c.mount_body,
                             mount_pos: c.mount_pos, mount_quat: c.mount_quat)
    }

    /// Frustum wireframe + image surface at the current projection params.
    private func buildGeometry(for c: CameraRigSpec, in grp: SCNNode, material: SCNMaterial) {
        // Depth for a uniform image height, from the TRUE vertical FOV.
        let halfFovY = atan((Double(c.h) / 2) / c.fy)
        let d = (projSize / 2) / tan(halfFovY)
        // Back-project pixel (u,v): flat → rectified pinhole plane at depth d;
        // curved → along the unit ray, capped at range d (lens-shaped shell).
        func pt(_ u: Double, _ v: Double) -> SCNVector3 {
            let xn = (u - c.cx) / c.fx, yn = -(v - c.cy) / c.fy
            if flat { return SCNVector3(xn * d, yn * d, -d) }
            let n = 1.0 / (xn * xn + yn * yn + 1).squareRoot()
            return SCNVector3(xn * n * d, yn * n * d, -n * d)
        }
        let tl = pt(0, 0), tr = pt(Double(c.w), 0)
        let br = pt(Double(c.w), Double(c.h)), bl = pt(0, Double(c.h))
        grp.addChildNode(Self.frustumNode(tl: tl, tr: tr, br: br, bl: bl))
        grp.addChildNode(Self.imageSurface(for: c, sampler: pt, flat: flat, material: material))
    }

    func detach() {
        sources.forEach { $0.stop() }
        sources.removeAll()
        groups.forEach { $0.removeFromParentNode() }
        groups.removeAll()
        entries.removeAll()
        builtAspect.removeAll()
        attached = false
    }

    // MARK: - Geometry

    /// Apex→corner rays + the image-rectangle outline (control-center blue).
    private static func frustumNode(tl: SCNVector3, tr: SCNVector3, br: SCNVector3, bl: SCNVector3) -> SCNNode {
        let apex = SCNVector3(0, 0, 0)
        let verts = [apex, tl, tr, br, bl]
        let idx: [Int32] = [0, 1, 0, 2, 0, 3, 0, 4,   // apex → corners
                            1, 2, 2, 3, 3, 4, 4, 1]   // rectangle
        let src = SCNGeometrySource(vertices: verts)
        let elem = SCNGeometryElement(indices: idx, primitiveType: .line)
        let geo = SCNGeometry(sources: [src], elements: [elem])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = UIColor(rgb: 0x4D9DFF)
        m.emission.contents = UIColor(rgb: 0x4D9DFF)
        geo.materials = [m]
        let n = SCNNode(geometry: geo)
        n.name = "frustum"
        return n
    }

    /// The textured image surface at the frustum base — a single quad when flat,
    /// a subdivided lens-shaped shell when curved. Until the first video frame
    /// lands it reads as a translucent dark pane.
    private static func imageSurface(for c: CameraRigSpec,
                                     sampler pt: (Double, Double) -> SCNVector3,
                                     flat: Bool,
                                     material: SCNMaterial) -> SCNNode {
        let nx = flat ? 1 : 16, ny = flat ? 1 : 10
        var verts: [SCNVector3] = []
        var uvs: [CGPoint] = []
        for j in 0...ny {
            for i in 0...nx {
                let u = Double(i) / Double(nx), v = Double(j) / Double(ny)
                verts.append(pt(u * Double(c.w), v * Double(c.h)))
                uvs.append(CGPoint(x: u, y: v))
            }
        }
        var idx: [Int32] = []
        for j in 0..<ny {
            for i in 0..<nx {
                let a = Int32(j * (nx + 1) + i), b = a + 1
                let d2 = a + Int32(nx + 1), e = d2 + 1
                idx += [a, d2, b, b, d2, e]
            }
        }
        let vSrc = SCNGeometrySource(vertices: verts)
        let tSrc = SCNGeometrySource(textureCoordinates: uvs)
        let elem = SCNGeometryElement(indices: idx, primitiveType: .triangles)
        let geo = SCNGeometry(sources: [vSrc, tSrc], elements: [elem])
        geo.materials = [material]
        let n = SCNNode(geometry: geo)
        n.name = "camimage"
        return n
    }
}
