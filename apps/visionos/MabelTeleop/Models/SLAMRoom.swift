//
//  SLAMRoom.swift
//  MABEL Mobile App
//
//  A saved visual-SLAM reconstruction of a space MABEL has mapped. Each
//  room is a dense 3D point cloud the operator can open and inspect like a
//  3D model. (Clouds here are procedurally synthesized stand-ins for real
//  RTAB-Map / ORB-SLAM exports.)
//

import Foundation

struct SLAMRoom: Identifiable, Equatable {
    let id = UUID()
    var name: String
    var icon: String
    var capturedAgo: String
    var areaM2: Double
    var pointCount: Int
    /// Footprint used by the cloud generator (metres).
    var size: SIMD3<Float>
    /// Generator seed so each room looks distinct but stable.
    var seed: UInt64
    /// Layout preset.
    var kind: Kind
    /// The live SLAM stream (mesh + odom over MapStreamLink), not a saved capture.
    /// Exactly one room — `SLAMRoom.live` — sets this; it pins to the top of the
    /// list and renders a LiveMapInspector instead of a procedural cloud.
    var isLive: Bool = false
    /// A saved map loaded from the ROS map server: it streams over `/map` (mesh +
    /// occupancy) just like live SLAM, so it renders in the LiveMapInspector. Set
    /// for server-advertised maps that aren't one of the procedural demo rooms.
    var streamed: Bool = false

    enum Kind { case studio, livingRoom, workshop, corridor }

    /// Build a render-room for a saved map the server advertised (rendered live
    /// from the `/map` stream once `load_map` is sent).
    static func streamed(name: String) -> SLAMRoom {
        SLAMRoom(name: name, icon: "square.stack.3d.up.fill", capturedAgo: "saved map",
                 areaM2: 0, pointCount: 0, size: .init(10, 3, 10), seed: 5, kind: .studio,
                 streamed: true)
    }

    /// The live reconstruction streamed from the connected robot (sim or real).
    static let live = SLAMRoom(
        name: "Live SLAM", icon: "dot.radiowaves.left.and.right", capturedAgo: "streaming",
        areaM2: 0, pointCount: 0, size: .init(10, 3, 10), seed: 1, kind: .studio, isLive: true)

    static let library: [SLAMRoom] = [
        .init(name: "Studio Floor A", icon: "video.fill", capturedAgo: "2h ago",
              areaM2: 48, pointCount: 182_000, size: .init(8, 3.2, 6), seed: 7, kind: .studio),
        .init(name: "Set — Living Room", icon: "sofa.fill", capturedAgo: "Yesterday",
              areaM2: 26, pointCount: 141_000, size: .init(6, 2.8, 4.4), seed: 21, kind: .livingRoom),
        .init(name: "Prop Workshop", icon: "wrench.and.screwdriver.fill", capturedAgo: "3d ago",
              areaM2: 34, pointCount: 158_000, size: .init(7, 3.0, 5), seed: 42, kind: .workshop),
        .init(name: "Stage Corridor", icon: "arrow.left.and.right", capturedAgo: "Last week",
              areaM2: 18, pointCount: 96_000, size: .init(9, 2.6, 2), seed: 99, kind: .corridor),
    ]
}
