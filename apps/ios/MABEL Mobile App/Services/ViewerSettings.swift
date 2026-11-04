//
//  ViewerSettings.swift
//  MABEL Mobile App
//
//  What the 3D viewer DRAWS — layers, per-camera streams, density, projection
//  size and surface. One shared, persisted store instead of per-view @State.
//
//  WHY THIS EXISTS: these used to be `@State` inside RobotSpaceView. Rotating the
//  phone makes ContentView swap between its portrait and landscape branches, which
//  gives RobotSpaceView a NEW view identity — and fresh @State. Every layer toggle
//  the operator had set (clouds off, a wrist camera projected, 1/4 density) silently
//  reverted to the defaults on every rotation, which is exactly the wrong moment:
//  you rotate to landscape BECAUSE you are about to drive.
//
//  These are operator preferences, not view-local UI state, so they live at app
//  scope and persist across rotation AND across launches (UserDefaults). The 3D
//  viewer reads them; nothing else writes them.
//

import Foundation
import Combine

@MainActor
final class ViewerSettings: ObservableObject {
    static let shared = ViewerSettings()

    // ── Layers ───────────────────────────────────────────────────────────────
    /// The accumulated SLAM SURFACE (nvblox mesh). Console chip "Mesh".
    @Published var meshOn: Bool          { didSet { d.set(meshOn, forKey: K.mesh) } }
    /// MASTER pointcloud switch (see RobotSpaceView.hasCamCloudPlanes).
    @Published var cloudOn: Bool         { didSet { d.set(cloudOn, forKey: K.cloud) } }
    /// The SLAM 2D occupancy floorplan. Console chip "2D Map" — renamed from
    /// "2D Grid" (2026-07-29) because it is not the grid on the floor, and the two
    /// names next to each other made both chips unreadable. Vision Pro's rail has
    /// called this "2D Map" all along; iOS now matches.
    @Published var gridOn: Bool          { didSet { d.set(gridOn, forKey: K.grid) } }

    // ── The stage (GroundPlane.swift) ────────────────────────────────────────
    /// MABEL's own mesh. Chip "Robot". Sensor layers in the same subtree stay up.
    @Published var robotOn: Bool         { didSet { d.set(robotOn, forKey: K.robot) } }
    /// The ground SURFACE + contact shadow. Chip "Floor".
    @Published var floorOn: Bool         { didSet { d.set(floorOn, forKey: K.floor) } }
    /// The 1 m / 5 m reference lines ON that floor. Chip "Grid".
    @Published var floorGridOn: Bool     { didSet { d.set(floorGridOn, forKey: K.floorGrid) } }
    @Published var minimapOn: Bool       { didSet { d.set(minimapOn, forKey: K.minimap) } }
    @Published var camProjections: Bool  { didSet { d.set(camProjections, forKey: K.camProj) } }
    /// Planar LiDAR sweep drawn as a 3D ring around the base (Vision Pro parity,
    /// its `laser` rail layer). Its own layer, NOT part of `cloudOn`: the sweep
    /// arrives on `robot_state` from EVERY plant (sim mj_ray, real RPLIDAR, demo
    /// ray-cast) at ~5 Hz, so it is the one sensor that is always live even when
    /// the /map depth cloud is idle — burying it under the cloud master hid the
    /// only geometry the Mac sim path actually has.
    @Published var laserOn: Bool         { didSet { d.set(laserOn, forKey: K.laser) } }
    /// RViz-style TF tree: an axis triad on base_link and every articulated joint
    /// frame (Vision Pro parity, its `tf` rail layer). Default OFF — ~40 triads
    /// riding the FK is a debugging view, not a driving view.
    @Published var tfOn: Bool            { didSet { d.set(tfOn, forKey: K.tf) } }

    // ── Per-camera sets (Cam 3D projections / camera pointclouds) ────────────
    @Published var camStreams: Set<String> { didSet { d.set(Array(camStreams), forKey: K.camStreams) } }
    @Published var camClouds: Set<String>  { didSet { d.set(Array(camClouds), forKey: K.camClouds) } }

    // ── Fine viz controls ────────────────────────────────────────────────────
    /// Cam 3D image height, metres.
    @Published var camProjSize: Double   { didSet { d.set(camProjSize, forKey: K.projSize) } }
    /// Rectified plane (true) vs curved through-the-lens shell (false).
    @Published var camFlat: Bool         { didSet { d.set(camFlat, forKey: K.camFlat) } }
    /// Live-cloud decimation: render every Nth point (1 = full).
    @Published var cloudStride: Int      { didSet { d.set(cloudStride, forKey: K.stride) } }

    private let d = UserDefaults.standard

    private enum K {
        static let mesh = "viewer.meshOn", cloud = "viewer.cloudOn", grid = "viewer.gridOn"
        static let minimap = "viewer.minimapOn"
        static let robot = "viewer.robotOn", floor = "viewer.floorOn"
        static let floorGrid = "viewer.floorGridOn"
        /// Retired 2026-07-29. Read ONCE as the seed for `floorGrid` so an operator
        /// who had deliberately turned the old reference grid off does not have it
        /// come back on after an update.
        static let legacyGround = "viewer.groundOn"
        static let camProj = "viewer.camProjections", camStreams = "viewer.camStreams"
        static let camClouds = "viewer.camClouds", projSize = "viewer.camProjSize"
        static let camFlat = "viewer.camFlat", stride = "viewer.cloudStride"
        static let laser = "viewer.laserOn", tf = "viewer.tfOn"
    }

    private init() {
        // `object(forKey:) == nil` distinguishes "never set" from "set to false" —
        // `bool(forKey:)` alone would silently turn every default-on layer OFF on a
        // fresh install. Reads UserDefaults directly: `self.d` is not available
        // until every stored property is initialised.
        let store = UserDefaults.standard
        func flag(_ key: String, default def: Bool) -> Bool {
            store.object(forKey: key) == nil ? def : store.bool(forKey: key)
        }
        meshOn = flag(K.mesh, default: true)
        cloudOn = flag(K.cloud, default: true)
        gridOn = flag(K.grid, default: true)
        robotOn = flag(K.robot, default: true)
        floorOn = flag(K.floor, default: true)
        floorGridOn = flag(K.floorGrid, default: flag(K.legacyGround, default: true))
        minimapOn = flag(K.minimap, default: true)
        camProjections = flag(K.camProj, default: true)
        laserOn = flag(K.laser, default: true)
        tfOn = flag(K.tf, default: false)
        camFlat = flag(K.camFlat, default: true)
        camStreams = Set((store.array(forKey: K.camStreams) as? [String]) ?? ["main"])
        camClouds = Set((store.array(forKey: K.camClouds) as? [String]) ?? ["main", "front"])
        let size = store.double(forKey: K.projSize)
        camProjSize = size > 0 ? size : 0.3
        let stride = store.integer(forKey: K.stride)
        cloudStride = stride > 0 ? stride : 1
    }
}
